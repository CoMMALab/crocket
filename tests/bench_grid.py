"""Milestone-2 gate: crocket's RNEA (every tier, autotuned) against GRiD's generated
``inverse_dynamics_kernel`` on the same GPU, float, kernel time by CUDA events.

    python tests/bench_grid.py [robot ...]      # default: a 7-DoF arm and a humanoid
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crocket  # noqa: E402
from crocket import harness, tune  # noqa: E402
from crocket.robot import GRID_DIR  # noqa: E402
from robots import URDFS, robot  # noqa: E402

BATCHES = (16, 128, 1024, 16384)
REPS = 200

GRID_BENCH = r"""
#include "grid.cuh"
extern "C" float bench(const float* q_qd, const float* qdd, float* c, int batch, int reps)
{
    grid::robotModel<float>* model = grid::init_robotModel<float>();
    float *dqqd, *dqdd, *dc;
    const size_t n = grid::NUM_JOINTS;
    cudaMalloc(&dqqd, 2 * n * batch * sizeof(float));
    cudaMalloc(&dqdd, n * batch * sizeof(float));
    cudaMalloc(&dc, n * batch * sizeof(float));
    cudaMemcpy(dqqd, q_qd, 2 * n * batch * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dqdd, qdd, n * batch * sizeof(float), cudaMemcpyHostToDevice);
    const size_t smem = grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<float>();
    const dim3 blocks(batch < 65535 ? batch : 65535), threads(grid::MAX_PERF_LEVEL_THREADS);
    grid::inverse_dynamics_kernel<float><<<blocks, threads, smem>>>(dc, dqqd, 2 * n, dqdd, nullptr, model, -9.81f, batch);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int k = 0; k < reps; ++k)
        grid::inverse_dynamics_kernel<float><<<blocks, threads, smem>>>(dc, dqqd, 2 * n, dqdd, nullptr, model, -9.81f, batch);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(c, dc, n * batch * sizeof(float), cudaMemcpyDeviceToHost);
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? ms / reps : -1.f;
}
"""


def grid_library(name: str) -> ctypes.CDLL:
    sys.path.insert(0, str(GRID_DIR))
    sys.path.insert(0, str(GRID_DIR / "external"))
    from grid_codegen import GRiDCodeGenerator
    from URDFParser import URDFParser

    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        parsed = URDFParser().parse(str(URDFS[name]))
        cwd = os.getcwd()
        try:
            os.chdir(d)
            GRiDCodeGenerator(parsed, False, False).gen_all_code()
        finally:
            os.chdir(cwd)
        header = (Path(d) / "grid.cuh").read_text()
    key = hashlib.sha1((header + GRID_BENCH).encode()).hexdigest()
    so = harness.cache_dir() / "grid" / f"{key}.so"
    if not so.exists():
        so.parent.mkdir(parents=True, exist_ok=True)
        (so.parent / f"{key}.cuh").write_text(header)
        src = so.with_suffix(".cu")
        src.write_text(GRID_BENCH.replace('"grid.cuh"', f'"{key}.cuh"'))
        cmd = ["nvcc", "-O3", "-std=c++17", harness.ARCH, "--shared", "--compiler-options", "-fPIC",
               f"-I{GRID_DIR / 'external' / 'GLASS'}", f"-I{GRID_DIR / 'external' / 'GLASS' / 'src'}",
               "-o", str(so), str(src)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode:
            raise RuntimeError(res.stderr[-3000:])
    return ctypes.CDLL(str(so))


def bench(name: str) -> list[tuple]:
    r = robot(name)
    p = crocket.trace(r, "rnea")
    lib = grid_library(name)
    fptr = ctypes.POINTER(ctypes.c_float)
    lib.bench.restype = ctypes.c_float
    lib.bench.argtypes = [fptr, fptr, fptr, ctypes.c_int, ctypes.c_int]
    rows = []
    for batch in BATCHES:
        x = {k: v.astype(np.float32) for k, v in harness.sample_inputs(p, batch).items()}
        q_qd = np.ascontiguousarray(np.concatenate([x["q"], x["qd"]], axis=1))
        qdd, c = np.ascontiguousarray(x["qdd"]), np.zeros((batch, r.n_q), np.float32)
        grid_ms = lib.bench(q_qd.ctypes.data_as(fptr), qdd.ctypes.data_as(fptr), c.ctypes.data_as(fptr), batch, REPS)
        err = harness.scaled_error(c, p.evaluate(**x)["tau"])
        best = tune.tune(p, batch, reps=REPS)
        rows.append((name, r.n_q, batch, grid_ms, err, best.ms, f"{best.tier}/{best.units}/t{best.threshold}/b{best.block}"))
        print(f"{name:14s} n={r.n_q:3d} B={batch:6d}  GRiD {grid_ms * 1e3:9.2f} us (err {err:.1e})  "
              f"crocket {best.ms * 1e3:9.2f} us [{rows[-1][-1]}]  speedup {grid_ms / best.ms:5.2f}x", flush=True)
    return rows


if __name__ == "__main__":
    for name in sys.argv[1:] or ["grid_iiwa14", "grid_g1"]:
        bench(name)
