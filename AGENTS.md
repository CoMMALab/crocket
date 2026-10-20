# AGENTS.md: crocket

crocket compiles rigid-body algorithms into robot-specific CUDA. You write an algorithm once in
Python. crocket traces it for one robot and emits a header with thread, warp, and block versions.
GRiM uses it for `traced=True` kinematics and for `grim.motion.dynamics`.

## How it works

1. **Python front end** (`algorithms.py`, `trace.py`). Algorithms such as RNEA, CRBA, ABA,
   `rnea_derivatives`, `frame_poses` and `frame_jacobian` are plain Python. They operate on
   traced spatial types: `Scalar`, `Vec3`, `Motion`, `Force`, `Inertia`, `SE3`. The robot comes
   from `robot.py`, which uses GRiD's `URDFParser`, so joint order and constants match GRiD.
   Running an algorithm records it. Loops over joints run during tracing, so they leave no loops
   in the output.

2. **Intermediate graph** (`ir.py`). The recording is an SSA graph with two levels.
   - Each node is a spatial operation named after its GLASS equivalent. It is tagged with its
     joint, tree depth, subtree, and pass (forward or backward).
   - Each node also carries its scalar expansion.
   - Both levels are optimized: constant folding (robot constants become literals), sparsity
     (multiply-by-zero is dropped), CSE, and dead-output elimination.
   - A float64 interpreter evaluates the graph. It is the reference that the CUDA output is
     tested against.
   - Python order does not fix GPU order. Only data dependencies do. The exception is a sum
     written as a running total (`f[p] = f[p] + ...`), which stays in written order. Keeping
     that order is what makes the tiers bit-identical.

3. **Scheduling and CUDA codegen** (`schedule.py`, `cuda.py`). The scheduler uses the tree tags.
   - It assigns independent branches to separate lanes, longest work first.
   - It inserts only the barriers that are needed.
   - It packs shared memory with interval colouring.
   - Functions over the statement budget become `__noinline__` stages. ptxas at -O2/-O3 has
     silently miscompiled huge straight-line traces, and this keeps each function small.

4. **Robot-specific header**. `crocket.generate(robot, {...})` emits one header per robot,
   `namespace crocket::robots::<name>`, with these functions:
   - `<alg>_thread`: one problem per thread.
   - `<alg>_split`: 32 problems per block, with warps splitting the stages.
   - `<alg>_warp`: one problem per warp.
   - `<alg>_block`: one problem per block.

   The functions are templated on `T` and write outputs as `y[i * ys]`. The cooperative tiers
   end without a barrier. `tune.py` times candidate layouts and caches the winner under
   `~/.cache/crocket`.

## When to use crocket vs GLASS vs GRiD/GRiM hand-written kernels

| Use | When |
|---|---|
| **crocket** | Robot-specific rigid-body math (FK, Jacobians, dynamics) at **large batch**, with one problem per thread. Specializing for the robot removes loops, branches, constant loads and zero products. Also use it when one algorithm source should produce several tiers. |
| **GLASS** | Linear algebra that does not depend on the robot: Cholesky, GEMM, reductions. It is already templated on compile-time `N`, so tracing adds nothing. Traced code calls into it, as `fd_crba` does with GLASS Cholesky. |
| **GRiD (via GRiM `generated_dynamics`)** | **Small batch on high-DOF robots** (latency), derivatives at small batch, runtime inertia, external forces, and second order (IDSVA). crocket folds inertias into constants and does not implement second order. |
| **GRiM hand-written motion kernels** | Solver logic (IK, trajopt, collision, LM). These call FK and Jacobians through `kernels/robot.cuh`, from baked tables or from crocket traces (`traced=True`). Choose by measurement. |

Measured speedups, traced vs GRiD, as JAX wall time (`test/benchmarks/bench_traced_vs_generated.py`
in GRiM, RTX A5000). Ratios above 1 favor crocket:

| | B ≤ 128 | B = 1024 | B = 16384 |
|---|---|---|---|
| iiwa14 id / crba / fd | 1.1–1.6× | 2.2–3.6× | 12–27× |
| iiwa14 ID derivatives | 0.76× | 1.5× | 5.8× |
| g1 (29 DOF) id / crba / fd | 0.86–1.06× | 2.5–3.9× | 2.5–25× |
| g1 ID derivatives | 0.45–0.64× | 2.5× | 2.7× |
| g1 fd via CRBA + Cholesky | 0.25× | 1.1× | 5.7× |

The rule of thumb is that traced code wins at batch ≥ 1024 and GRiD wins small-batch high-DOF
derivatives. Times below B=128 include about 0.12 ms of JAX dispatch, so use nsys there. When
the choice matters, benchmark it and do not assume.

## Rules

- **The trace is only as good as the Python.** Tracing removes overhead. It does not find a
  better algorithm or a better schedule. Port GRiD's formulation when crocket loses.
- **Validate every build.** `tests/test_oracle.py` checks the interpreter against RBDReference
  and Pinocchio on CPU. `tests/test_cuda.py` checks every tier against the interpreter,
  bit-identity under `-fmad=false`, and the -O0/-O3 guard.
- **gpu-proof.** `tests/test_cuda.py` is the signed receipt (`python -m pytest tests/test_cuda.py
  --gpu-proof-enable` on a clean tree). Any change to `python/crocket`, `tests`, `resources` or
  `pyproject.toml` invalidates it. After that, re-record it, bump GRiM's pointer, and re-record
  GRiM's receipt.
- **Known gaps.** There is no floating-base humanoid in the tests. Derivatives are checked only
  on panda and fr3, against finite differences. fr3 ABA is excluded with no stated reason. ABA
  and derivatives are unsupported on mimic joints, so use `fd_crba`.
