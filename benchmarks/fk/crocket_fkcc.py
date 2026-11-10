"""crocket's fused FK + collision (``sphere_collision``) as a raw CUDA launch, timed exactly like
the pRRTC harness: same input files, same loop of K launches between CUDA events, same output
(one value per group of 16 configurations).

Tiers: ``thread`` (one configuration per thread; a half-warp ballot gives the group) and
``warp`` (one configuration per warp, 8 per block so the scratch fits in shared memory; the
group is an atomicOr into a buffer zeroed inside every timed call).
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
WARPS = 8
# One straight-line function, no __noinline__ stage split: 2-3x faster here. crocket splits by
# default because ptxas has miscompiled huge traces; tests/test_cuda.py::test_unsplit_sphere_collision
# checks this build (interpreter, -O0 vs -O3) and every run here is checked group by group.
BUDGET = int(__import__("os").environ.get("CROCKET_FKCC_BUDGET", 10 ** 9))

MAIN = r"""
namespace r = crocket::robots::robot;
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void k_thread(const float* __restrict__ q, const float* __restrict__ boxes,
                         unsigned* __restrict__ out, int B)
{
    const int b0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = b0 < B ? b0 : B - 1;  // lanes past the batch repeat the last configuration
    float m;
    r::sphere_collision_thread<float>(q + (size_t)b * r::n_q, boxes, &m, 1);
    const unsigned hit = __ballot_sync(0xffffffffu, m < 0.f);
    const int lane = threadIdx.x & 31;
    if ((lane & 15) == 0 && b0 < B) out[b0 / 16] = (hit >> lane) & 0xFFFFu ? 1u : 0u;
}

__global__ void k_warp(const float* __restrict__ q, const float* __restrict__ boxes,
                       unsigned* __restrict__ out, int B)
{
    extern __shared__ float scratch[];
    const int w = threadIdx.x >> 5;
    const int b = blockIdx.x * WARPS + w;
    if (b >= B) return;
    __shared__ float m[WARPS];
    r::sphere_collision_warp<float>(q + (size_t)b * r::n_q, boxes, m + w, scratch + w * r::sphere_collision_warp_smem, 1);
    __syncwarp();
    if ((threadIdx.x & 31) == 0 && m[w] < 0.f) atomicOr(out + b / 16, 1u);
}

int main(int argc, char** argv)
{
    // <tier> <B> <q.f32> <boxes.f32> <out.f32> <K> <REPS> <WARMUP>
    const bool warp = !strcmp(argv[1], "warp");
    const int B = atoi(argv[2]), K = atoi(argv[6]), REPS = atoi(argv[7]), W = atoi(argv[8]);
    std::vector<float> q((size_t)B * r::n_q), boxes(N_BOXES * 15);
    FILE* f = fopen(argv[3], "rb");
    if (!f || fread(q.data(), sizeof(float), q.size(), f) != q.size()) return 1;
    fclose(f);
    f = fopen(argv[4], "rb");
    if (!f || fread(boxes.data(), sizeof(float), boxes.size(), f) != boxes.size()) return 1;
    fclose(f);
    const int groups = (B + 15) / 16;
    float *dq, *db;
    unsigned* dout;
    cudaMalloc(&dq, q.size() * sizeof(float));
    cudaMalloc(&db, boxes.size() * sizeof(float));
    cudaMalloc(&dout, groups * sizeof(unsigned));
    cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, boxes.data(), boxes.size() * sizeof(float), cudaMemcpyHostToDevice);
    const size_t smem = WARPS * r::sphere_collision_warp_smem * sizeof(float);
    if (warp) cudaFuncSetAttribute(k_warp, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    auto launch = [&] {
        if (warp) {
            cudaMemsetAsync(dout, 0, groups * sizeof(unsigned));
            k_warp<<<(B + WARPS - 1) / WARPS, WARPS * 32, smem>>>(dq, db, dout, B);
        } else {
            k_thread<<<(B + 127) / 128, 128>>>(dq, db, dout, B);
        }
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
    std::vector<unsigned> out(groups);
    cudaMemcpy(out.data(), dout, groups * sizeof(unsigned), cudaMemcpyDeviceToHost);
    std::vector<float> outf(out.begin(), out.end());
    f = fopen(argv[5], "wb");
    fwrite(outf.data(), sizeof(float), outf.size(), f);
    fclose(f);
    std::nth_element(times.begin(), times.begin() + REPS / 2, times.end());
    printf("%.6f\n", times[REPS / 2]);
    return 0;
}
"""


def build(robot: str, n_boxes: int) -> Path:
    import crocket
    from crocket import cuda
    urdf = common.FK_URDF[robot]
    links, centres, radii = common.spheres(urdf)
    src = crocket.generate(crocket.Robot(str(urdf)),
                           {"sphere_collision": {"spheres": tuple(zip(links, centres, radii)),
                                                 "n_boxes": n_boxes}},
                           variants=[cuda.Variant("thread"), cuda.Variant("warp", 32)], name="robot",
                           budget=BUDGET)
    src += f"\n#define N_BOXES {n_boxes}\n#define WARPS {WARPS}\n" + MAIN
    out = common.RESULTS / "build" / f"crocket_fkcc_{robot}_{BUDGET}"
    cu = out.with_suffix(".cu")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or not cu.exists() or cu.read_text() != src:
        cu.write_text(src)
        subprocess.check_call(["nvcc", "-O3", "-std=c++17", "-arch=sm_86", str(cu), "-o", str(out)])
    return out


def run(robot: str, q: np.ndarray, env: dict) -> tuple[float, np.ndarray, str]:
    """Fastest tier: (ms, per-group hit, tier). Joint order: crocket's (q is Pinocchio's)."""
    import crocket
    names = common.load_inputs("fk", robot)["joint_names"].tolist()
    perm = [names.index(n) for n in crocket.Robot(str(common.FK_URDF[robot])).joint_names]
    exe = build(robot, len(env["boxes"]))
    best = None
    with tempfile.TemporaryDirectory() as d:
        qp, bp, op = Path(d) / "q.f32", Path(d) / "boxes.f32", Path(d) / "out.f32"
        np.ascontiguousarray(q[:, perm], np.float32).tofile(qp)
        env["boxes"].astype(np.float32).tofile(bp)
        for tier in TIERS:
            ms = float(subprocess.check_output(
                [str(exe), tier, str(len(q)), str(qp), str(bp), str(op), str(common.K),
                 str(common.REPS), str(common.WARMUP)], text=True))
            if best is None or ms < best[0]:
                best = (ms, np.fromfile(op, np.float32) > 0.5, tier)
    return best
