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
and `ik_lm_step` (one Levenberg-Marquardt IK step for a frame's pose target, projected onto
the joint limits; the caller runs the loop). Joints: revolute, prismatic, fixed, and mimic (except ABA).

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
times are per call at large batch:

| workload | crocket | best other | where crocket loses |
|---|---|---|---|
| FK + environment collision (`sphere_collision`), B = 16384 | 0.31-0.53 ms | pRRTC 0.45-0.85 ms (1.2-2.8x slower) | B <= 1024: pRRTC is 1.5x faster on Panda and Baxter |
| IK, traced LM step (`ik_lm_step`), 1024 problems x 8 seeds | 1.6-2.8 ms | GRiM LS IK 70-207 ms (13-74x slower) | success 0.1-1.6 points below GRiM's |
| Inverse dynamics, B = 16384 | 0.05-0.47 ms | GRiD 1.7-12.7 ms (27-62x slower) | B <= 128: GRiD is mostly faster, most of all on g1 derivatives (0.35 vs 1.04 ms) |

Trace the whole computation. Swapping crocket's FK into GRiM's hand-written kernels gained
1.2-3x on IK, nothing on collision, and nothing on FK alone, where pRRTC and cuRobo stay fastest.
