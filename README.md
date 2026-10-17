# cricket2: a trace compiler for GRiD/GLASS rigid-body kernels

cricket2 traces rigid-body algorithms, written once at the level of spatial-algebra operations, and
compiles them per robot into CUDA for a thread, a warp or a block. Like cricket, it folds the
robot's constants into straight-line code. Like GRiD, it can also run *inside* one problem in
parallel, following the kinematic tree.

It is a new compiler, not a fork. The pipeline has cricket's shape (trace, then generate), with a
new IR and back end:

| stage | module | what it does |
|---|---|---|
| robot | `robot.py` | URDF → GRiD's `URDFParser` → per-joint constants (transform coefficients, subspace, inertia), the same model and joint order as GRiD |
| frontend | `trace.py`, `algorithms.py` | algorithms are plain Python over traced `Scalar` / `Vec3` / `Motion` / `Force` / `Inertia` / `SE3`; running one records the IR, and topology loops unroll |
| IR | `ir.py` | SSA graph at two levels: GLASS-named spatial ops tagged with joint / depth / subtree / pass, each carrying its folded scalar expansion (constant folding, sparsity, CSE at both levels, dead-output elimination) and a float64 interpreter |
| scheduler | `schedule.py` | coarsen (scalarise cheap single-consumer nodes), level by tree, longest-first lane assignment, minimal syncs, shared-memory slots by interval colouring |
| backend | `cuda.py` | one header per robot: `<alg>_thread / _split / _warp / _block`, templated on `T`, `y[i * ys]` outputs, `__noinline__` stage functions over the statement budget |
| tuning | `tune.py`, `harness.py` | compile, validate and time candidate layouts; the winner is cached per robot |

Algorithms: `frame_poses`, `frame_jacobian` (LOCAL_WORLD_ALIGNED), `rnea`, `crba`, `aba`, and
`rnea_derivatives` (analytical). Joints: revolute, prismatic, fixed, and mimic (except ABA).

## Usage

```python
import cricket2

robot = cricket2.Robot("panda.urdf")
src = cricket2.generate(robot, {"rnea": {}, "crba": {}, "frame_jacobian": {"frame": "panda_joint7"}})

# or pick each algorithm's layout by measurement (cached under ~/.cache/cricket2)
from cricket2 import tune
best = tune.tune(cricket2.trace(robot, "rnea"), batch=1024)
src = cricket2.generate(robot, {"rnea": {}}, variants={"rnea": [cricket2.Variant("thread"), best.variant]})
```

```cpp
namespace r = cricket2::robots::panda;
r::rnea_thread<float>(q, qd, qdd, tau, ys);                // one problem per thread
r::rnea_split<float>(q, qd, qdd, tau, smem, ys);           // 32 per block, warps split the stages
r::rnea_warp<float>(q, qd, qdd, tau, smem, ys);            // one per warp; r::rnea_warp_smem per warp
r::rnea_block<float>(q, qd, qdd, tau, smem, ys);           // one per block of r::rnea_block_threads
```

The cooperative tiers end without a barrier. Sync before you read another lane's outputs or reuse
`smem`.

`cricket2.generate(..., kinematics={"frames": [...], "end_effector": ...})` also emits cricket's
float kinematics interface (`frame_poses`, `ee_pose`, `ee_pose_jacobian`), which GRiM's
`traced=True` builds use.

## Test

```bash
pip install -e .                       # numpy; URDFParser / RBDReference come from the GRiD checkout
python -m pytest tests/test_oracle.py  # interpreter vs RBDReference and Pinocchio (CPU)
python -m pytest tests/test_cuda.py    # every tier vs interpreter, bit-identity, invariance, -O0/-O3 guard
python tests/bench_grid.py             # RNEA vs GRiD's generated kernel (milestone-2 gate)
```

Every tier computes each output element with one lane and one fixed expression, so tiers and
group sizes are bit-identical under `-fmad=false`.
