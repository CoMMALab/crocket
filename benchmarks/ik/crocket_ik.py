"""IK with crocket's traced LM step (``ik_lm_step``): the step -- FK, Jacobian, weighted pose
residual with a smooth SO(3) log, the inner Levenberg-Marquardt solve ``J^T (J J^T + lam I)^-1 r``
by an unrolled 6x6 Cholesky, and the projection onto the joint limits -- is one traced function;
the loop around it is ordinary CUDA, as in McVAMP (Iyer et al., 2026): accept if the cost drops
(lam / 3) else reject (lam * 4) and retry from the same q, stop at max_iter or once position and
orientation are within GRiM's eps_pos / eps_ori. Settings match the GRiM row.

Tiers: ``thread`` (one seed per thread) and ``warp`` (one seed per warp, the step's stages split
across its lanes); the fastest is reported. Timed like the pRRTC harness (CUDA events around K
back-to-back launches).
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

TIERS = ("thread", "warp")
WARPS = 4

LOOP = r"""
namespace r = crocket::robots::robot;
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr int N = r::n_q;
constexpr float EPS_POS2 = 1e-4f * 1e-4f, EPS_ORI2 = 1e-3f * 1e-3f;

// One seed's solve; STEP(q, out) evaluates the traced step at q with damping lam into
// out = [q_next (N), cost, pos_err2, ori_err2]. Identical control flow for both tiers.
#define SOLVE(STEP, SYNC)                                                                    \
    float lam = LAM0;                                                                        \
    STEP(q, cur);                                                                            \
    for (int it = 0; it < MAX_ITER; ++it) {                                                  \
        if (cur[N + 1] < EPS_POS2 && cur[N + 2] < EPS_ORI2) break;                           \
        STEP(cur, nxt);                                                                      \
        if (nxt[N] < cur[N]) {                                                               \
            for (int i = 0; i < N; ++i) q[i] = cur[i];                                       \
            SYNC;                                                                            \
            float* t = cur; cur = nxt; nxt = t;                                              \
            lam = fmaxf(lam / 3.f, 1e-7f);                                                   \
        } else {                                                                             \
            lam = fminf(lam * 4.f, 1e7f);                                                    \
            STEP(q, cur);                                                                    \
        }                                                                                    \
    }

__global__ void k_thread(const float* __restrict__ seeds, const float* __restrict__ targets,
                         float* __restrict__ out, int n_seeds, int S)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_seeds) return;
    const float* target = targets + 7 * (b / S);
    float q[N], buf_a[N + 3], buf_b[N + 3];
    float *cur = buf_a, *nxt = buf_b;
    for (int i = 0; i < N; ++i) q[i] = seeds[(size_t)b * N + i];
#define STEP_T(x, y) r::ik_lm_step_thread<float>(x, target, &lam, y, y + N, y + N + 1, y + N + 2, 1)
    SOLVE(STEP_T, (void)0)
    for (int i = 0; i < N; ++i) out[(size_t)b * N + i] = q[i];
}

__global__ void k_warp(const float* __restrict__ seeds, const float* __restrict__ targets,
                       float* __restrict__ out, int n_seeds, int S)
{
    extern __shared__ float smem[];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.x * WARPS + w;
    if (b >= n_seeds) return;
    float* scratch = smem + w * (r::ik_lm_step_warp_smem + 3 * (N + 3));
    float *q = scratch + r::ik_lm_step_warp_smem, *cur = q + N + 3, *nxt = cur + N + 3;
    const float* target = targets + 7 * (b / S);
    for (int i = lane; i < N; i += 32) q[i] = seeds[(size_t)b * N + i];
    __syncwarp();
#define STEP_W(x, y) do { r::ik_lm_step_warp<float>(x, target, &lam, y, y + N, y + N + 1, y + N + 2, scratch, 1); __syncwarp(); } while (0)
    SOLVE(STEP_W, __syncwarp())
    for (int i = lane; i < N; i += 32) out[(size_t)b * N + i] = q[i];
}

int main(int argc, char** argv)
{
    // <tier> <n_seeds> <S> <seeds.f32> <targets.f32> <out.f32> <K> <REPS> <WARMUP>
    const bool warp = !strcmp(argv[1], "warp");
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
    const size_t smem = WARPS * (r::ik_lm_step_warp_smem + 3 * (N + 3)) * sizeof(float);
    if (warp) cudaFuncSetAttribute(k_warp, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    auto launch = [&] {
        if (warp) k_warp<<<(n + WARPS - 1) / WARPS, WARPS * 32, smem>>>(ds, dt, dout, n, S);
        else k_thread<<<(n + 63) / 64, 64>>>(ds, dt, dout, n, S);
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
    src = crocket.generate(crocket.Robot(str(common.MODEL_URDF[robot])),
                           {"ik_lm_step": {"frame": frame, "pos_weight": pos_w, "ori_weight": ori_w}},
                           variants=[cuda.Variant("thread"), cuda.Variant("warp", 32)], name="robot")
    src += f"\n#define LAM0 5e-3f\n#define MAX_ITER {max_iter}\n#define WARPS {WARPS}\n" + LOOP
    out = common.RESULTS / "build" / f"crocket_ik_{robot}"
    cu = out.with_suffix(".cu")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or not cu.exists() or cu.read_text() != src:
        cu.write_text(src)
        subprocess.check_call(["nvcc", "-O3", "-std=c++17", "-arch=sm_86", str(cu), "-o", str(out)])
    return out


def run(robot, seeds, target, frame, max_iter, pos_w, ori_w):
    """Fastest tier: (ms, (P, S, n) solutions in Pinocchio's joint order, tier)."""
    import crocket
    names = common.load_inputs("ik", robot)["joint_names"].tolist()
    perm = [names.index(n) for n in crocket.Robot(str(common.MODEL_URDF[robot])).joint_names]
    inv = np.argsort(perm)
    exe = build(robot, frame, max_iter, pos_w, ori_w)
    P, S, n = seeds.shape
    best = None
    with tempfile.TemporaryDirectory() as d:
        sp, tp, op = Path(d) / "seeds.f32", Path(d) / "targets.f32", Path(d) / "out.f32"
        np.ascontiguousarray(seeds[..., perm], np.float32).tofile(sp)
        np.ascontiguousarray(target, np.float32).tofile(tp)
        for tier in TIERS:
            ms = float(subprocess.check_output(
                [str(exe), tier, str(P * S), str(S), str(sp), str(tp), str(op), str(common.K),
                 str(common.REPS), str(common.WARMUP)], text=True))
            if best is None or ms < best[0]:
                q = np.fromfile(op, np.float32).reshape(P, S, n)[..., inv]
                best = (ms, q, tier)
    return best
