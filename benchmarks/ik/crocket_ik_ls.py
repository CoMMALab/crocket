"""IK with GRiM ``ls_ik``'s formulation, traced by crocket: isolates tracing from the solver.

The step direction is ``ik_ls_step`` (weighted residual, Jacobi column scaling, the chain's
n x n normal equations ``(Js^T Js + lam I) d = -Js^T W r``, unscaled) and the line-search merit
``ik_pose_cost``, both traced. The loop around them copies kernels/ik/ls_ik.cu line for line:
convergence test at the head, GRiM's trust-region radii, the five trial step sizes
(1, 0.5, 0.25, 0.1, 0.025) clamped to the joint limits, accept below ``curr * (1 - 1e-4)``
(lam / 2, else lam * 3), all-time best tracking, the best configuration returned.

Precisions: ``f32`` (everything in float), ``f64`` (everything in double) and ``mixed``, GRiM's
own split: ``ik_ls_system`` (FK, Jacobian, residual) and the line search in float,
``ik_ls_solve`` (scaling, normal equations, Cholesky) in double. Thread tier only (one seed
per thread). Timed like crocket_ik.py (CUDA events around K back-to-back launches).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

PRECISIONS = ("f32", "f64", "mixed")

LOOP = r"""
namespace r = crocket::robots::robot;
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr int N = r::n_q;

// mixed: the traced system in float, the traced solve in double, scattered onto the chain.
template <class T, bool MIXED>
__device__ void step(const T* cfg, const T* target, T* lam, T* delta, T* cost, T* pe2, T* oe2)
{
    if constexpr (MIXED) {
        float Jw[6 * NC], rr[6];
        r::ik_ls_system_thread<float>(cfg, target, Jw, rr, cost, pe2, oe2, 1);
        double Jd[6 * NC], rd[6], ld = *lam, dd[NC];
        for (int i = 0; i < 6 * NC; ++i) Jd[i] = Jw[i];
        for (int i = 0; i < 6; ++i) rd[i] = rr[i];
        r::ik_ls_solve_thread<double>(Jd, rd, &ld, dd, 1);
        for (int i = 0; i < N; ++i) delta[i] = 0;
        for (int i = 0; i < NC; ++i) delta[COLS[i]] = (float)dd[i];
    } else {
        r::ik_ls_step_thread<T>(cfg, target, lam, delta, cost, pe2, oe2, 1);
    }
}

template <class T, bool MIXED>
__global__ void k_ls(const float* __restrict__ seeds, const float* __restrict__ targets,
                     float* __restrict__ out, int n_seeds, int S)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_seeds) return;
    T target[7];
    for (int i = 0; i < 7; ++i) target[i] = targets[7 * (b / S) + i];
    T cfg[N], best_cfg[N], delta[N], trial[N], lam = T(LAM0), cost, pe2, oe2, best_err;
    for (int i = 0; i < N; ++i) best_cfg[i] = cfg[i] = seeds[(size_t)b * N + i];
    step<T, MIXED>(cfg, target, &lam, delta, &best_err, &pe2, &oe2);
    const T alphas[5] = {1, 0.5, 0.25, 0.1, 0.025};
    for (int it = 0; it < MAX_ITER; ++it) {
        step<T, MIXED>(cfg, target, &lam, delta, &cost, &pe2, &oe2);
        if (pe2 < T(EPS_POS * EPS_POS) && oe2 < T(EPS_ORI * EPS_ORI)) break;
        const T p = sqrt(pe2), o = sqrt(oe2);
        const T R = (p > T(1e-2) || o > T(0.6)) ? T(0.38) : (p > T(1e-3) || o > T(0.25)) ? T(0.22)
                  : (p > T(2e-4) || o > T(0.08)) ? T(0.12) : T(0.05);
        T dn = 0;
        for (int i = 0; i < N; ++i) dn += delta[i] * delta[i];
        dn = sqrt(dn);
        if (dn > R) for (int i = 0; i < N; ++i) delta[i] *= R / (dn + T(1e-18));
        T best_a_err = T(1e30);
        int best_a = 0;
        for (int a = 0; a < 5; ++a) {
            for (int i = 0; i < N; ++i) trial[i] = fmin(fmax(cfg[i] + alphas[a] * delta[i], T(LO[i])), T(HI[i]));
            T c;
            r::ik_pose_cost_thread<T>(trial, target, &c, 1);
            if (c < best_a_err) { best_a_err = c; best_a = a; }
        }
        for (int i = 0; i < N; ++i) trial[i] = fmin(fmax(cfg[i] + alphas[best_a] * delta[i], T(LO[i])), T(HI[i]));
        if (best_a_err < cost * T(1 - 1e-4)) {
            for (int i = 0; i < N; ++i) cfg[i] = trial[i];
            lam = fmax(lam * T(0.5), T(1e-10));
        } else {
            lam = fmin(lam * T(3), T(1e6));
        }
        if (best_a_err < best_err) {
            best_err = best_a_err;
            for (int i = 0; i < N; ++i) best_cfg[i] = trial[i];
        }
    }
    for (int i = 0; i < N; ++i) out[(size_t)b * N + i] = (float)best_cfg[i];
}

int main(int argc, char** argv)
{
    // <f32|f64|mixed> <n_seeds> <S> <seeds.f32> <targets.f32> <out.f32> <K> <REPS> <WARMUP>
    const bool f64 = !strcmp(argv[1], "f64"), mixed = !strcmp(argv[1], "mixed");
    const int n = atoi(argv[2]), S = atoi(argv[3]), K = atoi(argv[7]), REPS = atoi(argv[8]), W = atoi(argv[9]);
    std::vector<float> seeds((size_t)n * N), targets((size_t)(n / S) * 7), out((size_t)n * N);
    FILE* f = fopen(argv[4], "rb");
    if (!f || fread(seeds.data(), sizeof(float), seeds.size(), f) != seeds.size()) return 1;
    fclose(f);
    f = fopen(argv[5], "rb");
    if (!f || fread(targets.data(), sizeof(float), targets.size(), f) != targets.size()) return 1;
    fclose(f);
    float *ds, *dt, *dout;
    cudaMalloc(&ds, seeds.size() * sizeof(float));
    cudaMalloc(&dt, targets.size() * sizeof(float));
    cudaMalloc(&dout, out.size() * sizeof(float));
    cudaMemcpy(ds, seeds.data(), seeds.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dt, targets.data(), targets.size() * sizeof(float), cudaMemcpyHostToDevice);
    auto launch = [&] {
        if (f64) k_ls<double, false><<<(n + 31) / 32, 32>>>(ds, dt, dout, n, S);
        else if (mixed) k_ls<float, true><<<(n + 31) / 32, 32>>>(ds, dt, dout, n, S);
        else k_ls<float, false><<<(n + 31) / 32, 32>>>(ds, dt, dout, n, S);
    };
    for (int i = 0; i < W; ++i) launch();
    cudaDeviceSynchronize();
    cudaEvent_t a, e;
    cudaEventCreate(&a);
    cudaEventCreate(&e);
    std::vector<float> times;
    for (int rep = 0; rep < REPS; ++rep) {
        cudaEventRecord(a);
        for (int k = 0; k < K; ++k) launch();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, a, e);
        times.push_back(ms / K);
    }
    if (cudaGetLastError() != cudaSuccess) return 2;
    cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    f = fopen(argv[6], "wb");
    fwrite(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    std::nth_element(times.begin(), times.begin() + REPS / 2, times.end());
    printf("%.6f\n", times[REPS / 2]);
    return 0;
}
"""


def build(robot: str, frame: str, max_iter: int, pos_w: float, ori_w: float) -> Path:
    import crocket
    from crocket import cuda
    r = crocket.Robot(str(common.MODEL_URDF[robot]))
    opts = {"frame": frame, "pos_weight": pos_w, "ori_weight": ori_w}
    # The chain columns: where a traced step's delta can be nonzero (zero elsewhere by construction).
    probe = crocket.trace(r, "ik_ls_step", frame=frame).evaluate(
        q=np.random.default_rng(0).uniform(-1, 1, (4, r.n_q)), target=np.tile([1.0, 0, 0, 0, 0.3, 0.1, 0.5], (4, 1)),
        lam=np.full((4, 1), 0.05))["delta"]
    cols = np.flatnonzero(np.abs(probe).sum(0))
    src = crocket.generate(r, {"ik_ls_step": opts, "ik_pose_cost": opts, "ik_ls_system": opts,
                               "ik_ls_solve": {"n": len(cols)}},
                           variants=[cuda.Variant("thread")], name="robot")
    owner = {j.q_index: j for j in r.joints if not j.mimic}
    lim = [owner[c].limits for c in range(len(owner))]
    arr = lambda xs: "{" + ", ".join(f"{x:.9g}" for x in xs) + "}"  # noqa: E731
    src += (f"\n#define LAM0 5e-3\n#define MAX_ITER {max_iter}\n#define EPS_POS 1e-4\n#define EPS_ORI 1e-3\n"
            f"#define NC {len(cols)}\n__constant__ int COLS[] = {{{', '.join(map(str, cols))}}};\n"
            f"__constant__ float LO[] = {arr(l for l, _ in lim)};\n__constant__ float HI[] = {arr(h for _, h in lim)};\n"
            + LOOP)
    out = common.RESULTS / "build" / f"crocket_ik_ls_{robot}"
    cu = out.with_suffix(".cu")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or not cu.exists() or cu.read_text() != src:
        cu.write_text(src)
        subprocess.check_call(["nvcc", "-O3", "-std=c++17", "-arch=sm_86", str(cu), "-o", str(out)])
    return out


def run(robot, seeds, target, frame, max_iter, pos_w, ori_w, precision):
    """(ms, (P, S, n) solutions in Pinocchio's joint order, precision)."""
    import crocket
    names = common.load_inputs("ik", robot)["joint_names"].tolist()
    perm = [names.index(n) for n in crocket.Robot(str(common.MODEL_URDF[robot])).joint_names]
    inv = np.argsort(perm)
    exe = build(robot, frame, max_iter, pos_w, ori_w)
    P, S, n = seeds.shape
    with tempfile.TemporaryDirectory() as d:
        sp, tp, op = Path(d) / "seeds.f32", Path(d) / "targets.f32", Path(d) / "out.f32"
        np.ascontiguousarray(seeds[..., perm], np.float32).tofile(sp)
        np.ascontiguousarray(target, np.float32).tofile(tp)
        ms = float(subprocess.check_output(
            [str(exe), precision, str(P * S), str(S), str(sp), str(tp), str(op), str(common.K),
             str(common.REPS), str(common.WARMUP)], text=True))
        q = np.fromfile(op, np.float32).reshape(P, S, n)[..., inv]
    return ms, q, precision
