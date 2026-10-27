# crocket: a algorithmic CUDA trace compiler for rigid-body robotics kernels

crocket traces rigid-body algorithms, written once at the level of spatial-algebra operations, and
compiles them per robot into CUDA for a thread, a warp or a block. Like cricket, it folds the
robot's constants into straight-line code. Like GRiD, it can also run *inside* one problem in
parallel, following the kinematic tree. Parallelism tiers are modeled after GLASS.



| stage | module | what it does |
|---|---|---|
| robot | `robot.py` | URDF → GRiD's `URDFParser` → per-joint constants (transform coefficients, subspace, inertia), the same model and joint order as GRiD |
| frontend | `trace.py`, `algorithms.py` | algorithms are plain Python over traced `Scalar` / `Vec3` / `Motion` / `Force` / `Inertia` / `SE3`; running one records the IR, and topology loops unroll |
| IR | `ir.py` | SSA graph at two levels: GLASS-named spatial ops tagged with joint / depth / subtree / pass, each carrying its folded scalar expansion (constant folding, sparsity, CSE at both levels, dead-output elimination) and a float64 interpreter |
| scheduler | `schedule.py` | coarsen (scalarise cheap single-consumer nodes), level by tree, longest-first lane assignment, minimal syncs, shared-memory slots by interval colouring |
| backend | `cuda.py` | one header per robot: `<alg>_thread / _split / _warp / _block`, templated on `T`, `y[i * ys]` outputs, `__noinline__` stage functions over the statement budget |
| tuning | `tune.py`, `harness.py` | compile, validate and time candidate layouts; the winner is cached per robot |

Algorithms: `frame_poses`, `frame_jacobian` (LOCAL_WORLD_ALIGNED), `rnea`, `crba`, `aba`,
`rnea_derivatives` (analytical), and `sphere_collision` (fused FK + sphere-vs-box environment
check: `spheres` folded in as constants, `n_boxes` boxes per call, output `min(d² - r²)`),
`ik_lm_step` (one Levenberg-Marquardt IK step for a frame's pose target, projected onto
the joint limits; the caller runs the loop), and GRiM's LS IK step as `ik_ls_step` (Jacobi-scaled
normal equations over the frame's chain), split into `ik_ls_system` and `ik_ls_solve` so the two
halves can run at different precisions, with `ik_pose_cost` for the line search.

GRiM's IK kernels (`kernels/ik_crocket`) read everything about the robot from `ik_geometry`: the
stacked pose residual and Jacobian of several frames, the world centres of collision spheres, and
every joint's world twist, from which a collision gradient is a dot product. `ik_geometry_value`
is the same without the Jacobian, for line-search trials. With `rot_joint`, a per-call `rot`
(wxyz) replaces that movable joint's parent-to-joint rotation at q = 0 (a floating base's chart
reference). Fixed joints are folded into the joints below them, so they cannot take one.
`ik_sqp_system` and `ik_masked_solve` are SQP's Jacobi-scaled QP and its branch-free active-set
re-solve. `analytic_ik_candidates` and `analytic_ik_chain` are the closed-form IK of the Panda
family (8 branches per redundancy value) and its sphere-placement chain.

Joints: revolute, continuous (a revolute joint without limits), prismatic, fixed, and mimic
(except ABA).

## Usage

```python
import crocket

robot = crocket.Robot("panda.urdf")
src = crocket.generate(robot, {"rnea": {}, "crba": {}, "frame_jacobian": {"frame": "panda_joint7"}})

# or pick each algorithm's layout by measurement (cached under ~/.cache/crocket)
from crocket import tune
best = tune.tune(crocket.trace(robot, "rnea"), batch=1024)
src = crocket.generate(robot, {"rnea": {}}, variants={"rnea": [crocket.Variant("thread"), best.variant]})
```

```cpp
namespace r = crocket::robots::panda;
r::rnea_thread<float>(q, qd, qdd, tau, ys);                // one problem per thread
r::rnea_split<float>(q, qd, qdd, tau, smem, ys);           // 32 per block, warps split the stages
r::rnea_warp<float>(q, qd, qdd, tau, smem, ys);            // one per warp; r::rnea_warp_smem per warp
r::rnea_block<float>(q, qd, qdd, tau, smem, ys);           // one per block of r::rnea_block_threads
```

The cooperative tiers end without a barrier. Sync before you read another lane's outputs or reuse
`smem`.

`crocket.generate(..., kinematics={"frames": [...], "end_effector": ...})` also emits cricket's
float kinematics interface (`frame_poses`, `ee_pose`, `ee_pose_jacobian`), which GRiM's
`traced=True` builds use.

## Test

```bash
pip install -e .                       # numpy; URDFParser / RBDReference come from the GRiD checkout
python -m pytest tests/test_oracle.py  # interpreter vs RBDReference and Pinocchio (CPU)
python -m pytest tests/test_cuda.py    # every tier vs interpreter, bit-identity, invariance, -O0/-O3 guard
```

Every tier computes each output element with one lane and one fixed expression, so tiers and
group sizes are bit-identical under `-fmad=false`.

## Benchmarks

The benchmarks are on the [`benchmarks`](https://github.com/CoMMALab/crocket/tree/benchmarks/benchmarks) branch, under
`benchmarks/`; `bash benchmarks/run_all.sh` reruns them. They compare crocket against pRRTC,
cuRobo, pyroki, GRiD/GRiM and frax on Panda (7 DOF), Baxter (14/15) and Unitree G1 with hands
(43). All runs are float32 on one RTX A5000, every output is checked against Pinocchio, and the
times are per call at large batch.

**FK + environment collision, B = 16384 (ms per call)**

| robot | crocket | pRRTC | cuRobo | pyroki | GRiM | GRiM + crocket FK |
|---|---|---|---|---|---|---|
| panda | **0.37** | 0.45 | 4.39 | 7.11 | 0.53 | 0.53 |
| baxter | **0.53** | 0.75 | 5.90 | 12.89 | 2.20 | 2.33 |
| g1 | **0.31** | 0.85 | 5.06 | 11.37 | 1.82 | 2.09 |

**IK, 1024 problems x 8 seeds (ms per call, success rate)**

crocket traces GRiM's own LS algorithm at GRiM's precision (float kinematics, double solve), so
the column measures tracing alone.

| robot | crocket | GRiM | pyroki |
|---|---|---|---|
| panda | **14.3 (95.4%)** | 70.0 (95.4%) | 842.1 (98.2%) |
| baxter | **14.3 (95.8%)** | 124.0 (95.8%) | 2807.4 (98.9%) |
| g1 | **23.8 (98.5%)** | 207.5 (98.5%) | 5573.2 (100.0%) |

**Inverse dynamics (RNEA), B = 16384 (ms per call)**

| robot | crocket | GRiD | frax |
|---|---|---|---|
| panda | **0.045** | 1.715 | 1.365 |
| baxter | **0.059** | 3.634 | 3.106 |
| g1 | **0.473** | 12.733 | 11.108 |

Fastest per robot in bold. At small batch crocket gives some of this back: pRRTC is 1.5x faster
on FK + collision at B <= 1024 for Panda and Baxter, GRiD is mostly faster at B <= 128, most of
all on g1 dynamics derivatives, and GRiM's block tier is faster on Panda IK at 1 x 8.

Trace the whole computation. Swapping crocket's FK into GRiM's hand-written kernels gained
1.2-3x on IK, nothing on collision, and nothing on FK alone, where pRRTC and cuRobo stay fastest;
tracing GRiM's whole IK step gained 4.9-8.7x at equal algorithm and precision.

GRiM now runs every IK method on crocket's `ik_geometry`. Over 4 robots, 3 target sets and 8
methods, at 16384 problems x 8 seeds, the float32 build wins 75 of 78 cells against the
hand-written kernels' fastest tier: 73 at the thread tier and the 2 analytic IK cells at the block
tier. The median speedup is 13x: 15-80x for LS, SQP and HJCD LM without collision, 4-10x for
them with collision, 3-17x for DLS, MPPI and canonical IK. The hand-written kernel keeps
multi-end-effector DLS (crocket is 1.3-1.8x slower there). The full
table is GRiM's `test/benchmarks/IK_RESULTS.md`.
