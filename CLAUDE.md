# CLAUDE.md: crocket

## What crocket does

crocket is a trace compiler for rigid-body robotics kernels. You write an algorithm once in Python
(RNEA, CRBA, ABA, FK, Jacobians, fused FK + collision, an IK step). crocket then runs it on one
robot and emits CUDA specialized to that robot. Robot constants become literals and loops over joints
are unrolled. Multiplications by zero and dead outputs are removed. The output is straight-line code
for each problem: a thread, warp or block runs one problem with no loops over joints, no branches and
no table lookups. GRiM builds every IK kernel and every trajectory optimizer on top of it.

## Two-stage compilation

1. **Stage 1, crocket (Python, at build time).**
   - **Trace:** `algorithms.py` / `trace.py` run the algorithm over traced spatial types, which
     records it.
   - **Optimize:** `ir.py` holds the recording as a two-level SSA graph. The upper level is spatial
     ops named after GLASS, tagged with joint, depth, subtree and pass. Each op carries its scalar
     expansion. Both levels get constant folding, sparsity elimination, CSE and dead-output
     elimination. A float64 interpreter of this graph is the reference the CUDA is tested against.
   - **Schedule:** `schedule.py` assigns lanes, longest work first, and inserts only the barriers
     that are needed. It assigns shared-memory slots by interval colouring.
   - **Emit:** `cuda.py` emits one header per robot: `<alg>_thread / _split / _warp / _block`,
     templated on `T`. Traces over the statement budget are split into `__noinline__` stages.
2. **Stage 2, nvcc (inside the consumer's kernel).** The header is `#include`d into a kernel
   written by hand, such as GRiM's `kernels/ik_crocket/*.cu`. That kernel owns everything that
   depends on data at runtime: iteration loops, accept/reject, early exit and convergence. nvcc
   compiles the traced step and the solver loop together, so the robot math is inlined into the
   loop. `tune.py` / `harness.py` compile candidate layouts, validate them and time them. The
   fastest layout for each robot is cached under `~/.cache/crocket`.

The split rule: fixed-size math goes in the trace (stage 1), and control flow that branches on data
stays in CUDA (stage 2). Trace one whole iteration (`ik_lm_step`, `ik_geometry`), not just its FK.
The benchmarks show tracing only the FK leaves most of the speedup unused.

## Analysis tools: Grama's parallel computing model

`python/crocket/parallel_analysis.py` (test: `tests/test_parallel_analysis.py`) measures a traced
program with the metrics of Grama, Gupta, Karypis and Kumar, *Introduction to Parallel Computing*.
The thread tier is the serial baseline and a `schedule.build` schedule is the parallel one. The unit
is folded scalar ops, the same cost model the scheduler balances. It only reads the IR and schedule
and changes nothing.

- **Work and span:** `T_s = W` is the number of live scalar ops. `D` is the longest dependency
  chain. `W / D` is the available parallelism, and `T_ideal = max(W/p, D)` is Brent's bound.
- **`T_p`:** for each stage, the busiest lane's statements plus its shared-memory traffic, plus a
  cost for each barrier.
- **Speedup, efficiency, cost:** `S = T_s / T_p` and `E = S / p`. The tier is cost-optimal when
  `p T_p = Θ(T_s)`, which is fitted as the slope of log cost against log W across robots.
- **Overhead:** `T_o = p T_p - T_s`, split into recomputation, shared memory, barriers and idle
  lanes.
- **Scaling:** strong scaling for each robot, weak scaling (p grows with W), and isoefficiency (the
  W needed to hold `E` as p grows, fitted as `W ~ p^k`).
- **Batch model:** `batch()` computes makespan as waves × time per problem for each tier. At batch
  sizes at or above the GPU's lane count, the thread tier already runs at `E = 1`. Cooperative tiers
  only pay off when the batch leaves cores idle.

```bash
python -m crocket.parallel_analysis rnea panda.urdf baxter.urdf --units 4 8 16 32 64 --out rnea.json
```

The barrier and shared-memory costs (`SYNC_COST`, `SMEM_COST`) and `CORES` are estimates. Use the
model to explain results, and use `tune.py` timings to choose a tier.

## Spatial algebra: Featherstone's representation

The traced types and algorithms follow Featherstone, *Rigid Body Dynamics Algorithms* (2008;
`CITATIONS.md`). Spatial vectors are `Motion` / `Force` laid out as `[angular; linear]`. Transforms
are `SE3`, stored as a rotation `E` and a translation `r`, never a dense 6×6 Plücker matrix.
Inertias use the compact forms: `RigidInertia` (`m, h, I_O`, 10 numbers) and `ArticulatedInertia`
(`I, H, M`, 21 numbers), never a dense 6×6. Each spatial op has a scalar expansion in the book's
efficient form for every type it accepts. That is where tracing's flop savings come from, so keep
it when adding ops (RBDA §2.13; test against the dense product in `tests/test_oracle.py`). RNEA,
CRBA and ABA are the book's recursive algorithms. Their loops over joints unroll at trace time, and
the tree tags let the scheduler run independent branches in parallel.

## GLASS parallelism: thread, warp, block

The tiers follow GLASS (GPU Linear Algebra Simple Subroutines; `external/GRiD/external/GLASS`). One
traced program yields every tier:

| tier | layout | when |
|---|---|---|
| `_thread` | one problem per thread, the full trace in registers | large batch: every lane busy at `E = 1` |
| `_split` | 32 problems per block, warps split the stages | medium batch, several robots per warp |
| `_warp` | one problem per warp (32 lanes) | small batch, moderate `W / D` |
| `_block` | one problem per block (`units` lanes) | latency at batch ≈ 1 on high-DOF robots |

Conventions shared with GLASS: one problem never spans blocks. Outputs are strided as `y[i * ys]`.
Cooperative tiers take `smem` and end without a barrier, so the caller syncs. Spatial ops carry their
GLASS names, and robot-independent linear algebra (Cholesky, GEMM, reductions at a size that grows
with the robot) is left to GLASS rather than traced. Every output element is computed by one lane
with one fixed expression, so every tier and group size gives bit-identical results under
`-fmad=false`. `tests/test_cuda.py` enforces this.

Cooperative variants need `units`: `Variant("warp", units=32)` and `Variant("block", units=N)`. The
default `units=1` runs the whole trace on lane 0.

@AGENTS.md
