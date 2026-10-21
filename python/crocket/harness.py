"""Compile emitted headers into a shared library and run them on the GPU (tests, autotuning).

Each (algorithm, variant) gets a ``__global__`` wrapper and an ``extern "C"`` host entry
``run_<i>(inputs, outputs, batch, block, reps) -> ms``: inputs packed per problem
``[problem][inputs, concatenated]``, outputs structure-of-arrays (``ys = batch``), float or double.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cuda
from .trace import Program

ARCH = os.environ.get("CROCKET_ARCH", "-arch=native")
SMEM_LIMIT = 96 * 1024  # bytes of shared scratch per block before falling back to global memory


def cache_dir() -> Path:
    root = os.environ.get("CROCKET_CACHE") or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "crocket"
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def nvcc_version() -> str:
    return subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]


def _kernel(i: int, p: Program, v: cuda.Variant, ns: str, T: str) -> str:
    offsets = np.cumsum([0] + [n for _, n in p.inputs])
    ins = ", ".join(f"x + {int(o)}" for o in offsets[:-1])
    stride = int(offsets[-1])
    outs, off = [], 0
    for name, ids in p.outputs:
        outs.append(f"y + {off} * batch + b")
        off += len(ids)
    outs = ", ".join(outs)
    fn = f"r::{p.name}_{v.tier}<{T}>"
    if v.tier == "thread":
        body = (f"    const int b = blockIdx.x * blockDim.x + threadIdx.x;\n    if (b >= batch) return;\n"
                f"    const {T}* x = X + (size_t)b * {stride};\n    {fn}({ins}, {outs}, batch);\n")
        launch, smem, scratch = "(batch + block - 1) / block, block", "0", "0"
    elif v.tier == "split":
        # 32 problems per block; lanes past the batch redo the last problem (identical writes)
        # so every lane reaches every barrier
        body = (f"    const int b0 = blockIdx.x * 32 + (threadIdx.x & 31);\n"
                f"    const int b = b0 < batch ? b0 : batch - 1;\n"
                f"    extern __shared__ unsigned char raw[];\n"
                f"    {T}* s = G ? G + (size_t)blockIdx.x * r::{p.name}_split_smem : ({T}*)raw;\n"
                f"    const {T}* x = X + (size_t)b * {stride};\n    {fn}({ins}, {outs}, s, batch);\n")
        launch = "(batch + 31) / 32, 32 * r::" + f"{p.name}_split_warps"
        smem = f"r::{p.name}_split_smem * sizeof({T})"
        scratch = f"(size_t)((batch + 31) / 32) * r::{p.name}_split_smem"
    elif v.tier == "warp":
        body = (f"    const int warps = blockDim.x >> 5;\n"
                f"    const int b = blockIdx.x * warps + (threadIdx.x >> 5);\n    if (b >= batch) return;\n"
                f"    extern __shared__ unsigned char raw[];\n"
                f"    {T}* s = (G ? G + (size_t)blockIdx.x * warps * r::{p.name}_warp_smem : ({T}*)raw)\n"
                f"        + (threadIdx.x >> 5) * r::{p.name}_warp_smem;\n"
                f"    const {T}* x = X + (size_t)b * {stride};\n    {fn}({ins}, {outs}, s, batch);\n")
        launch = "(batch + block / 32 - 1) / (block / 32), block"
        smem = f"(block / 32) * r::{p.name}_warp_smem * sizeof({T})"
        scratch = f"(size_t)(batch + block / 32 - 1) / (block / 32) * (block / 32) * r::{p.name}_warp_smem"
    else:
        body = (f"    const int b = blockIdx.x;\n"
                f"    extern __shared__ unsigned char raw[];\n"
                f"    {T}* s = G ? G + (size_t)b * r::{p.name}_block_smem : ({T}*)raw;\n"
                f"    const {T}* x = X + (size_t)b * {stride};\n    {fn}({ins}, {outs}, s, batch);\n")
        launch = f"batch, r::{p.name}_block_threads"
        smem = f"r::{p.name}_block_smem * sizeof({T})"
        scratch = f"(size_t)batch * r::{p.name}_block_smem"
    return f"""
__global__ void k{i}(const {T}* __restrict__ X, {T}* __restrict__ y, int batch, {T}* G)
{{
{body}}}
extern "C" float run_{i}(const {T}* X, {T}* Y, int batch, int block, int reps)
{{
    const size_t nx = (size_t)batch * {stride}, ny = (size_t)batch * {off};
    {T} *dx, *dy;
    cudaMalloc(&dx, nx * sizeof({T}));
    cudaMalloc(&dy, ny * sizeof({T}));
    cudaMemcpy(dx, X, nx * sizeof({T}), cudaMemcpyHostToDevice);
    // scratch over SMEM_LIMIT bytes goes to global memory instead of shared
    {T}* G = nullptr;
    size_t smem = {smem};
    if (smem > SMEM_LIMIT)
    {{
        cudaMalloc(&G, {scratch} * sizeof({T}));
        smem = 0;
    }}
    else if (smem > 48 * 1024)
        cudaFuncSetAttribute(k{i}, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    k{i}<<<{launch}, smem>>>(dx, dy, batch, G);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int k = 0; k < reps; ++k) k{i}<<<{launch}, smem>>>(dx, dy, batch, G);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(Y, dy, ny * sizeof({T}), cudaMemcpyDeviceToHost);
    const cudaError_t err = cudaGetLastError();
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(G);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return err == cudaSuccess ? (reps ? ms / reps : 0.f) : -1.f - (float)err;
}}
"""


@dataclass
class Built:
    lib: ctypes.CDLL
    cases: list[tuple[Program, cuda.Variant]]
    dtype: type

    def run(self, i: int, inputs: dict[str, np.ndarray], block: int = 128, reps: int = 0):
        """Run case ``i`` on (B, len) inputs; returns (name -> (B, len) outputs, ms per launch)."""
        p, _ = self.cases[i]
        batch = next(iter(inputs.values())).shape[0]
        X = np.ascontiguousarray(np.concatenate([inputs[n] for n, _ in p.inputs], axis=1), self.dtype)
        n_out = sum(len(ids) for _, ids in p.outputs)
        Y = np.zeros((n_out, batch), self.dtype)
        ptr = ctypes.POINTER(ctypes.c_float if self.dtype == np.float32 else ctypes.c_double)
        fn = getattr(self.lib, f"run_{i}")
        fn.restype = ctypes.c_float
        fn.argtypes = [ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        ms = fn(X.ctypes.data_as(ptr), Y.ctypes.data_as(ptr), batch, block, reps)
        if ms < 0:
            raise RuntimeError(f"CUDA error {int(-ms - 1)} in {p.name} {self.cases[i][1]}")
        out, k = {}, 0
        for name, ids in p.outputs:
            out[name], k = Y[k:k + len(ids)].T.copy(), k + len(ids)
        return out, ms


def build(cases: list[tuple[Program, cuda.Variant]], dtype=np.float32, flags: tuple[str, ...] = ("-O3",),
          budget: int = cuda.BUDGET) -> Built:
    """Emit, compile (cached on disk by source and flags) and load ``cases``."""
    T = "float" if dtype == np.float32 else "double"
    robot = cases[0][0].robot
    by_program: dict[int, list[cuda.Variant]] = {}
    programs: dict[int, Program] = {}
    for p, v in cases:
        by_program.setdefault(id(p), []).append(v)
        programs[id(p)] = p
    sections = [cuda.emit(programs[k], vs, budget) for k, vs in by_program.items()]
    src = cuda.header(robot, sections, name="robot")
    src += f"namespace r = crocket::robots::robot;\nconstexpr size_t SMEM_LIMIT = {SMEM_LIMIT};\n" + "".join(
        _kernel(i, p, v, "robot", T) for i, (p, v) in enumerate(cases))
    flags = (*flags, "-std=c++17", ARCH, "--shared", "--compiler-options", "-fPIC")
    key = hashlib.sha1((src + " ".join(flags)).encode()).hexdigest()
    out = cache_dir() / "build" / f"{key}.so"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        cu = out.with_suffix(".cu")
        cu.write_text(src)
        tmp = out.with_suffix(f".{os.getpid()}.tmp.so")
        res = subprocess.run(["nvcc", *flags, "-o", str(tmp), str(cu)], capture_output=True, text=True)
        if res.returncode:
            raise RuntimeError(f"nvcc failed on {cu}:\n{res.stderr[-4000:]}")
        tmp.replace(out)
    return Built(ctypes.CDLL(str(out)), list(cases), dtype)


# Float relative bounds per algorithm: GRiM's accepted bars for traced float dynamics vs float64
# (scaled: atol = rtol * max(1, max|expected|)). Double is held to 1e-9.
TOLERANCE = {"rnea": 2e-5, "crba": 2e-5, "rnea_derivatives": 5e-5, "aba": 1e-3,
             "frame_poses": 2e-5, "frame_jacobian": 2e-5,
             "ik_lm_step": 1e-3}  # a 6x6 solve, as aba: conditioning scales with 1 / lam


def scaled_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """``max |a - e| / (max(1, max|e|) + |e|)``: below ``rtol`` passes GRiM's scaled check."""
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if not np.all(np.isfinite(actual)):
        return np.inf
    if actual.size == 0:
        return 0.0
    return float((np.abs(actual - expected) / (max(np.abs(expected).max(), 1.0) + np.abs(expected))).max())


def sample_inputs(p: Program, batch: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Random inputs within joint limits, the last problem at zero q and qd."""
    rng = np.random.default_rng(seed)
    r = p.robot
    owner = {j.q_index: j for j in r.joints if not j.mimic}
    lo = np.array([max(owner[c].limits[0], -np.pi) for c in range(r.n_q)])
    hi = np.array([min(owner[c].limits[1], np.pi) for c in range(r.n_q)])
    x = {"q": rng.uniform(lo, hi, (batch, r.n_q)), "qd": rng.uniform(-1, 1, (batch, r.n_q)),
         "qdd": rng.uniform(-1, 1, (batch, r.n_q)), "tau": rng.uniform(-5, 5, (batch, r.n_q))}
    x["q"][-1], x["qd"][-1] = np.clip(0.0, lo, hi), 0.0
    if any(n == "target" for n, _ in p.inputs):  # a reachable pose: unit quaternion, nearby point
        quat = rng.normal(size=(batch, 4))
        x["target"] = np.concatenate([quat / np.linalg.norm(quat, axis=1, keepdims=True),
                                      rng.uniform(-0.5, 0.5, (batch, 3))], 1)
        x["lam"] = rng.uniform(1e-3, 1.0, (batch, 1))
    for n, length in p.inputs:
        if n == "boxes":  # [centre, orthonormal axes, half-extents] per box, the same for the batch
            m = length // 15
            axes = np.linalg.qr(rng.normal(size=(m, 3, 3)))[0]
            boxes = np.concatenate([rng.uniform(-0.8, 0.8, (m, 3)), axes.reshape(m, 9),
                                    rng.uniform(0.02, 0.3, (m, 3))], 1).reshape(-1)
            x["boxes"] = np.tile(boxes, (batch, 1))
    return {n: x[n] for n, _ in p.inputs}


def validate(built: Built, batch: int = 64, seed: int = 0) -> list[float]:
    """Every case against the float64 interpreter; raises on the first one out of tolerance.
    Returns the scaled error per case."""
    errors = []
    for i, (p, v) in enumerate(built.cases):
        x = sample_inputs(p, batch, seed)
        want = p.evaluate(**x)
        got, _ = built.run(i, x, block=128)
        tol = TOLERANCE.get(p.name, 2e-5) if built.dtype == np.float32 else 1e-9
        err = max(scaled_error(got[n], want[n]) for n in want)
        if not err <= tol:
            raise AssertionError(f"{p.robot.name} {p.name} {v}: scaled error {err:.3e} > {tol:.0e}")
        errors.append(err)
    return errors
