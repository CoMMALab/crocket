# AGENTS.md: crocket

crocket compiles rigid-body algorithms into robot-specific CUDA. You write an algorithm once in
Python. crocket traces it for one robot and emits a header with thread, warp, and block versions.
GRiM uses it for `traced=True` kinematics and for `grim.motion.dynamics`.

## How it works

1. **Python front end** (`algorithms.py`, `trace.py`). Algorithms such as RNEA, CRBA, ABA,
   `rnea_derivatives`, `frame_poses`, `frame_jacobian`, `sphere_collision` (fused FK +
   environment check), `ik_lm_step` (one projected Levenberg-Marquardt IK step) and
   `ik_geometry` (everything GRiM's IK kernels read about the robot; see the README) are plain
   Python. They operate on
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
| **GRiM hand-written motion kernels** | Solver control flow that branches on data: iteration loops, accept/reject, early exit, convergence. The fixed-size math inside each iteration (FK, Jacobian, residual, the damped solve, the limit projection, collision terms) should be a traced crocket algorithm, not hand-written code calling traced FK through `kernels/robot.cuh` (`traced=True`). That swap alone gained 1.2-3x on IK; tracing the whole step gained 13-74x (see `benchmarks/README.md` on the `benchmarks` branch). GRiM's IK is built this way (`kernels/ik_crocket`: CUDA solver loops over `ik_geometry`), a median 13x over the hand-written kernels it replaced (GRiM `test/benchmarks/IK_RESULTS.md`). |

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

## Adding a traced algorithm or op

Trace the whole computation, not just its FK. The benchmarks show the cost of doing less:
- **IK:** GRiM's hand-written LM solver with crocket's FK swapped in (`traced=True`) gained only
  1.2-3x. Tracing the whole step (`ik_lm_step`) gained 13-74x.
- **FK + collision:** GRiM's fused collision kernel with traced FK did not beat its own
  table-driven FK. Tracing FK, sphere placement and the box test together (`sphere_collision`)
  beat pRRTC at large batch.

The fixed-size math around FK, done by hand-written code, is where the time goes. When a new
workload needs robot math, write a new algorithm for it. Do not wrap an existing one.

**An algorithm** is a function in `algorithms.py` registered with
`@algorithm(inputs=..., outputs=...)`:
- Write it over the traced types. Use `robot.joints` loops for the tree; they unroll at trace
  time.
- An input is a name (`n_q` long) or `(name, lambda options: length)` for anything else (boxes,
  targets, damping).
- Anything known when the kernel is built is a trace option: spheres, frames, weights, limits,
  the size of the system. Options fold in as constants and their sparsity propagates. Pass
  per-call values (obstacle poses, targets, `lam`) as inputs.
- Outputs are flat lists of Scalars or traced values.
- Reuse the helpers: `_poses`, `_frame`, `_jacobian`, `_pose_out`, `_cholesky_solve`, `_so3_log`,
  `_min_tree`. Build reductions as balanced trees (`_min_tree`), so the cooperative tiers can
  split them; a running chain serialises them.

**Control flow:**
- No data-dependent control flow inside a trace. Express branch-free choices with `min`/`max`:
  clamps, hinges, projections.
- Trace one iteration and leave loops, accept/reject and early exit to the caller. This is the
  McVAMP pattern (Iyer et al. 2026; reference code in `prrax/mcvamp/cricket/src/tracing/`):
  constraint error and Jacobian traced, one LM step traced through an unrolled Cholesky, the
  loop in plain code.
- Small fixed-size solves (6x6 JJ^T, n x n for small n) belong in the trace, unrolled
  (`_cholesky_solve`). Solves whose size grows with the robot belong to GLASS.

**Adding a scalar op** touches four places. `abs`, `max`, `min` and `atan2` were added this way:
- `ir.py` `_FOLD`: a NumPy-compatible function, used for constant folding and the interpreter.
  Also `_NUMPY` if the `math` version does not take arrays.
- Algebraic simplifications in `Graph.s`, if any.
- A `Scalar` method in `trace.py`.
- The CUDA spelling in `cuda._CALL`, when it is not the op's name.

**Numerics:**
- Kernels run in float32, so prefer well-conditioned formulas. Example: the SO(3) log uses
  `atan2(|vee|/2, cos)`. McVAMP's `acos(cos)` with `1 - cos^2` cancels near pi and left float32
  three correct digits there.
- Smooth singularities by blending, not by branching, as in `_so3_log`.

**Tests**, before any benchmark:
- `tests/robots.py` `options()`: the algorithm's trace options for the test robots. Synthesise
  any geometry the URDFs lack, as `sphere_set` does.
- `harness.sample_inputs`: sample values for any new input name.
- `harness.TOLERANCE`: a float bar justified by conditioning, not widened until the test passes.
- `tests/test_oracle.py`: the interpreter against an independent float64 implementation
  (Pinocchio or NumPy). Check the step's math in isolation from crocket's kinematics, which
  `test_kinematics_vs_pinocchio` covers: a solve amplifies the ~1e-8 kinematics differences by
  its condition number.
- `tests/test_cuda.py` `ALGORITHMS` / `CASES`: every tier against the interpreter, bit-identity
  across tiers, and the -O0/-O3 guard.
- A build with a raised `budget` (no `__noinline__` stage split) needs its own -O0/-O3 and
  interpreter check (`test_unsplit_sphere_collision`). The split guards against ptxas
  miscompiles and can cost 2-3x on large traces.

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
- **Cooperative variants need `units`.** `Variant("warp")` defaults to `units=1`, which runs
  the whole trace on lane 0. Use `Variant("warp", units=32)` and `Variant("block", units=N)`.
- **ptxas also miscompiles the cooperative tiers.** CUDA 13.3 reused uniform register UR4 across
  the per-lane arms of a block stage's `switch (threadIdx.x)`: one arm materialised a constant in
  it while another stored through it as a shared-memory base, giving an illegal address
  (GRiM's g1-with-collision `ik_geometry` block build, float only; `-Xptxas -O1` did not help, and
  the `__noinline__` stage split does not prevent it). An illegal shared address of the form
  `0x4...` (a float's bits) inside a `_block_s<k>` stage points here. Run the block tier under
  `compute-sanitizer` when adding a large traced algorithm.
- **Known gaps.** There is no floating-base humanoid in the tests. Derivatives are checked only
  on panda and fr3, against finite differences. fr3 ABA is excluded with no stated reason. ABA
  and derivatives are unsupported on mimic joints, so use `fd_crba`.
