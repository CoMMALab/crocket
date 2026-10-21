# Benchmarks

FK, IK and dynamics of crocket against the libraries it competes with, on three robots: Franka
Panda (7 DOF), Rethink Baxter (14/15 DOF, two arms) and Unitree G1 with hands (43 DOF tree).

```bash
bash benchmarks/run_all.sh        # picks a free GPU, runs everything, rewrites the tables below
```

## Protocol

Every method gets the same float32 inputs, cached once in `results/inputs/` (so methods in
other conda envs read identical arrays), and is timed the same way (`common.amortized_ms`):
50 back-to-back asynchronous calls, one sync, divided by 50; the median of 9 such runs after
3 warm-up calls. That is per-call throughput time with inputs already on the GPU, measured the
same way for JAX, PyTorch and raw CUDA (the pRRTC harness runs the identical loop with CUDA
events). All runs use one GPU (RTX A5000).

Every output is checked against Pinocchio in float64 on 65 rows: the first 32, 32 spread over
the largest batch, and the last row of every batch. JAX runs with
`JAX_DEFAULT_MATMUL_PRECISION=highest`: JAX's default TF32 matmuls on Ampere gave frax
relative errors of 1e-3 to 4e-2; true float32 costs it about 5%.

Where a library offers several kernels for the same job, the fastest per cell is reported and
named: GRiM IK's thread / warp / block tiers, crocket dynamics' thread / warp-split tiers.
A call whose memory does not fit on the GPU at some batch is reported as OOM, not chunked.

## Robots

| | FK | IK and dynamics |
|---|---|---|
| panda | `robots/panda.urdf` (pRRTC's spherized panda, 59 spheres) | same file |
| baxter | `robots/baxter_spherized.urdf` (pRRTC's, 75 spheres, 14 DOF) | `robots/baxter.urdf` (GRiD's, 15 DOF incl. head pan) |
| g1 | `robots/g1.urdf` (pyroffi-grim's spherized G1 with hands, 68 spheres, 43 DOF) | same file |

pRRTC's spherized baxter has no inertias, so IK and dynamics use GRiD's. The panda file drops
one `<mimic>` tag on a *fixed* finger joint (meaningless there; pyroki refuses it).

## Methods

**FK** computes the world centre of every collision sphere, `(B, S, 3)`: the one output all five
produce. pRRTC and cuRobo emit spheres from their kernels; pyroki, GRiM and crocket compute frame
poses and place the spheres in the same jit (identical placement code for all three).

- **pRRTC** `fk<Robot>` from its robot headers, launched as `pRRTC.cu` launches it (64 threads,
  16 configurations per block, shared-memory output), by `fk/prrtc/harness.cu`. Panda and Baxter
  are pRRTC's shipped headers. G1 is ported with pRRTC's documented flow (README "Adding a
  Robot"): cricket's `gpu-cc-early-exit` branch (CoMMALab/cricket@98582c3) on the spherized G1,
  configs in `fk/prrtc/cricket_config/`, output `fk/prrtc/g1.cuh` and the robot struct
  `g1_struct.hh`. Only the main (fine-sphere) kernel is generated; FK uses no approx kernel.
  Building that cricket branch against current conda-forge needed four fixes: C++17 (Pinocchio
  3.9), no `Boost::system` (header-only since Boost 1.89), `fmt/format.h` (fmt 11), and a signed
  `joint_id_to_dof` (the universe joint's -1 was emitted as 18446744073709551615). With them,
  regenerating Baxter reproduces pRRTC's shipped `fk<Baxter>` and its tables exactly. CUDA 13
  removed `cudaDeviceProp::memoryClockRate`, read only by a pRRTC print helper; the harness
  build aliases it.
- **cuRobo** (v0.8 dev) `Kinematics.compute_kinematics` with the URDF's own spheres (not
  cuRobo's yml spheres) and one tool frame. cuRobo drops joints that move no sphere.
- **pyroki** `Robot.forward_kinematics`.
- **GRiD/GRiM** `grim.motion.kinematics(traced=False)`: GRiM's FK kernel over GRiD-style baked
  tables, poses only (no end-effector Jacobian).
- **crocket** the same call with `traced=True`: crocket's straight-line FK.

**Fused FK + collision** answers, for each group of 16 consecutive configurations (an edge,
pRRTC's unit of work), whether any of them collides with the environment: oriented boxes
(half-extents 2-6 cm) with centres uniform in the bounds of the robot's reach and clear of its
static spheres, added until 40-60% of the groups collide. A sphere collides when its centre is
closer to a box than its radius. Self-collision is left out: the libraries' self pair sets
differ (SRDF ranges, adjacency rules), so it would not be one problem. The group is the output
because pRRTC's check votes across the warp and stops at the first hit; it has no
per-configuration answer. Every method reduces to the group inside its own call.

- **pRRTC** `fk<Robot>` then `env_collision_check<Robot>` in one kernel, every sphere checked
  (its templates index 20 joint gates per configuration, an assumption g1's 44 joints break in
  the planner; the harness sets every gate, so the check is unaffected).
- **cuRobo** `RobotCollisionChecker`: its kinematics kernel, then its scene-collision kernel
  over cuboids, with the URDF's spheres.
- **pyroki** `RobotCollision.from_sphere_decomposition` (the same spheres) and
  `compute_world_collision_distance` against `Box`es, vmapped over configurations.
- **GRiD/GRiM** `grim.motion.collision.world_distances`: one fused kernel (FK in the thread,
  then per-link, per-box distances), baked tables.
- **GRiM fused, crocket-traced FK** the same kernel built `traced=True`: only its FK is traced.
- **crocket `sphere_collision`** (`fk/crocket_fkcc.py`): crocket's fused algorithm. FK, sphere
  placement (centres and radii as constants) and the squared sphere-box test, `min(d² - r²)`
  over spheres and boxes, are traced into one straight-line kernel per robot; the boxes are a
  per-call input. Raw CUDA launch timed like the pRRTC harness; thread and warp tiers, fastest
  reported (thread wins every cell). Built as one function, without crocket's default
  `__noinline__` stage split (2-3x slower here); `tests/test_cuda.py::test_unsplit_sphere_collision`
  checks that build against the interpreter and ptxas -O0. Straight-line code has no early exit.

**IK**: one end-effector (`panda_grasptarget`, `right_gripper`, `right_hand_palm_link`), P pose
targets from FK of random in-limit configurations, 8 random seeds each, position weight 50,
orientation weight 10, at most 60 iterations. Success: some seed within 1 mm and 0.01 rad, inside
joint limits, by Pinocchio.

- **GRiM LS IK, GRiD tables**: `grim.motion.ik` `ls` kernel (Levenberg-Marquardt), `traced=False`.
- **crocket traced LM step** (`ik/crocket_ik.py`): crocket's `ik_lm_step` algorithm, after
  McVAMP (Iyer et al., *Vectorizing Projection in Manifold-Constrained Motion Planning*, 2026):
  one traced function computes FK, the Jacobian, the weighted pose residual with a smooth SO(3)
  log, the inner Levenberg-Marquardt step `J^T (J J^T + λI)^-1 r` through an unrolled 6x6
  Cholesky, and the projection onto the joint limits. The loop around it is plain CUDA: accept
  if the cost drops (λ / 3), else reject (λ × 4) and retry; λ starts at 5e-3 (GRiM's
  `lambda_init`); stop at 60 iterations or within GRiM's `eps_pos` / `eps_ori`. Thread and
  warp tiers, fastest reported. A raw CUDA launch timed like the pRRTC harness, so it pays no
  JAX dispatch (about 0.05 ms, small against these times). It replaces the earlier row that
  only swapped crocket's FK into GRiM's hand-written solver.
- **pyroki + jaxls**: pyroki's own IK formulation (`pose_cost_analytic_jac` + `limit_constraint`,
  dense-Cholesky LM, as in its `_solve_ik` example), `vmap`ped over problems × seeds.

**Dynamics**: ID (RNEA), mass matrix (CRBA), FD, and ∂ID/∂(q, q̇); random in-limit q, q̇ and q̈
in [-1, 1], FD fed the ID torques; gravity -9.81 z. Error is `max |x - ref| / max(1, max |ref|)`.

- **GRiD**: GRiD's generated kernels (`grid_codegen` header via `grim.motion.generated_dynamics`).
- **crocket**: `grim.motion.dynamics.TracedDynamics`; FD is ABA.
- **frax**: `jax.jit(jax.vmap(...))` of `rnea`, `crba`, `forward_dynamics`. frax has no
  analytical derivatives; its ∂ID is `jax.jacfwd` of its RNEA.

## Results

<!-- tables -->
### FK: world centres of every collision sphere

Spheres per robot: panda 59, baxter 75, g1 68. Time per call in ms (lower is better).

**B = 1**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 0.076 | 0.094 | 0.155 |
| cuRobo | 0.185 | 0.184 | 0.186 |
| pyroki | 0.269 | 0.824 | 0.974 |
| GRiD/GRiM (baked tables) | 0.096 | 0.104 | 0.166 |
| crocket (traced) | 0.051 | 0.066 | 0.129 |

**B = 128**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 0.075 | 0.093 | 0.152 |
| cuRobo | 0.184 | 0.184 | 0.186 |
| pyroki | 0.352 | 1.117 | 1.337 |
| GRiD/GRiM (baked tables) | 0.097 | 0.290 | 0.408 |
| crocket (traced) | 0.103 | 0.369 | 0.504 |

**B = 1024**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 0.075 | 0.093 | 0.153 |
| cuRobo | 0.185 | 0.189 | 0.186 |
| pyroki | 0.345 | 1.060 | 1.272 |
| GRiD/GRiM (baked tables) | 0.107 | 0.321 | 0.426 |
| crocket (traced) | 0.116 | 0.405 | 0.530 |

**B = 16384**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 0.346 | 0.501 | 0.653 |
| cuRobo | 0.186 | 0.504 | 0.788 |
| pyroki | 1.527 | 4.880 | 5.902 |
| GRiD/GRiM (baked tables) | 0.450 | 1.343 | 1.759 |
| crocket (traced) | 0.455 | 1.503 | 1.966 |

**Max sphere-centre error vs Pinocchio float64 (m), over all batches**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 8.4e-07 | 1.3e-06 | 1.1e-06 |
| cuRobo | 2.0e-07 | 3.1e-07 | 1.6e-07 |
| pyroki | 2.8e-07 | 4.9e-07 | 3.3e-07 |
| GRiD/GRiM (baked tables) | 2.9e-07 | 6.1e-07 | 3.2e-07 |
| crocket (traced) | 2.0e-07 | 2.4e-07 | 1.7e-07 |

### Fused FK + collision: is any of each 16-configuration group in collision with the boxes?

Boxes per robot: panda 11, baxter 11, g1 7; about 46% of groups collide. Time per call in ms. Every method agrees with the float64 reference on every group.

**B = 1**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk` + `env_collision_check` | 0.128 | 0.171 | 0.204 |
| cuRobo `RobotCollisionChecker` | 0.714 | 0.712 | 0.713 |
| pyroki `RobotCollision` | 0.294 | 0.847 | 1.005 |
| GRiD/GRiM fused (baked tables) | 0.229 | 0.377 | 0.350 |
| GRiM fused, crocket-traced FK | 0.214 | 0.365 | 0.337 |
| crocket `sphere_collision` (fully traced) | 0.194 | 0.256 | 0.184 |

**B = 128**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk` + `env_collision_check` | 0.127 | 0.169 | 0.200 |
| cuRobo `RobotCollisionChecker` | 0.701 | 0.709 | 0.706 |
| pyroki `RobotCollision` | 0.432 | 1.197 | 1.385 |
| GRiD/GRiM fused (baked tables) | 0.261 | 0.470 | 0.420 |
| GRiM fused, crocket-traced FK | 0.272 | 0.538 | 0.499 |
| crocket `sphere_collision` (fully traced) | 0.195 | 0.257 | 0.188 |

**B = 1024**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk` + `env_collision_check` | 0.126 | 0.169 | 0.200 |
| cuRobo `RobotCollisionChecker` | 0.718 | 0.710 | 0.707 |
| pyroki `RobotCollision` | 0.702 | 1.491 | 1.650 |
| GRiD/GRiM fused (baked tables) | 0.261 | 0.469 | 0.419 |
| GRiM fused, crocket-traced FK | 0.273 | 0.539 | 0.491 |
| crocket `sphere_collision` (fully traced) | 0.184 | 0.247 | 0.179 |

**B = 16384**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk` + `env_collision_check` | 0.449 | 0.753 | 0.852 |
| cuRobo `RobotCollisionChecker` | 4.390 | 5.898 | 5.055 |
| pyroki `RobotCollision` | 7.108 | 12.891 | 11.365 |
| GRiD/GRiM fused (baked tables) | 0.529 | 2.201 | 1.821 |
| GRiM fused, crocket-traced FK | 0.529 | 2.329 | 2.089 |
| crocket `sphere_collision` (fully traced) | 0.367 | 0.535 | 0.307 |

### IK: one end-effector pose, 8 seeds per problem, 60 iterations max

Time per call in ms, then success rate (position < 1 mm, orientation < 0.01 rad, within limits; Pinocchio float64). GRiM and crocket report their fastest tier.

**1 problem × 8 seeds**

| method | panda | baxter | g1 |
|---|---|---|---|
| GRiM LS IK, GRiD tables | 8.02 (100.0%, block) | 15.70 (100.0%, block) | 24.68 (100.0%, block) |
| crocket traced LM step | 1.55 (100.0%, thread) | 1.65 (100.0%, thread) | 2.18 (100.0%, thread) |
| pyroki + jaxls | 139.38 (100.0%, vmap) | 452.21 (100.0%, vmap) | 229.94 (100.0%, vmap) |

**128 problems × 8 seeds**

| method | panda | baxter | g1 |
|---|---|---|---|
| GRiM LS IK, GRiD tables | 16.42 (97.7%, warp) | 30.36 (97.7%, warp) | 64.98 (99.2%, warp) |
| crocket traced LM step | 1.55 (95.3%, thread) | 1.67 (95.3%, thread) | 2.30 (97.7%, thread) |
| pyroki + jaxls | 284.18 (99.2%, vmap) | 826.01 (98.4%, vmap) | 1114.59 (100.0%, vmap) |

**1024 problems × 8 seeds**

| method | panda | baxter | g1 |
|---|---|---|---|
| GRiM LS IK, GRiD tables | 69.91 (95.4%, thread) | 123.49 (95.8%, thread) | 206.91 (98.5%, thread) |
| crocket traced LM step | 1.60 (93.8%, thread) | 1.78 (95.5%, thread) | 2.80 (98.4%, thread) |
| pyroki + jaxls | 842.13 (98.2%, vmap) | 2807.43 (98.9%, vmap) | 5573.17 (100.0%, vmap) |

### Dynamics

Time per call in ms. crocket reports its fastest tier. OOM: the call's workspace does not fit in the GPU's 24 GB at that batch.

**B = 1**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · GRiD | 0.041 | 0.055 | 0.121 |
| ID (RNEA) · crocket | 0.049 | 0.055 | 0.139 |
| ID (RNEA) · frax | 0.104 | 0.206 | 0.458 |
| M (CRBA) · GRiD | 0.051 | 0.056 | 0.123 |
| M (CRBA) · crocket | 0.043 | 0.062 | 0.182 |
| M (CRBA) · frax | 0.098 | 0.188 | 0.442 |
| FD · GRiD | 0.060 | 0.091 | 0.233 |
| FD · crocket | 0.046 | 0.111 | 0.286 |
| FD · frax | 0.200 | 0.309 | 0.646 |
| ∂ID/∂(q, q̇) · GRiD | 0.055 | 0.091 | 0.352 |
| ∂ID/∂(q, q̇) · crocket | 0.115 | 0.196 | 1.042 |
| ∂ID/∂(q, q̇) · frax | 0.260 | 0.514 | 1.352 |

**B = 128**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · GRiD | 0.040 | 0.062 | 0.146 |
| ID (RNEA) · crocket | 0.047 | 0.068 | 0.168 |
| ID (RNEA) · frax | 0.163 | 0.277 | 0.670 |
| M (CRBA) · GRiD | 0.051 | 0.065 | 0.158 |
| M (CRBA) · crocket | 0.043 | 0.066 | 0.216 |
| M (CRBA) · frax | 0.122 | 0.241 | 0.564 |
| FD · GRiD | 0.064 | 0.100 | 0.430 |
| FD · crocket | 0.045 | 0.131 | 0.307 |
| FD · frax | 0.272 | 0.413 | 1.016 |
| ∂ID/∂(q, q̇) · GRiD | 0.062 | 0.114 | 0.722 |
| ∂ID/∂(q, q̇) · crocket | 0.118 | 0.209 | 1.145 |
| ∂ID/∂(q, q̇) · frax | 0.556 | 1.410 | 9.488 |

**B = 1024**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · GRiD | 0.136 | 0.309 | 0.936 |
| ID (RNEA) · crocket | 0.046 | 0.053 | 0.200 |
| ID (RNEA) · frax | 0.229 | 0.435 | 1.198 |
| M (CRBA) · GRiD | 0.170 | 0.371 | 1.079 |
| M (CRBA) · crocket | 0.044 | 0.104 | 0.418 |
| M (CRBA) · frax | 0.172 | 0.387 | 1.173 |
| FD · GRiD | 0.229 | 0.648 | 3.240 |
| FD · crocket | 0.045 | 0.159 | 0.367 |
| FD · frax | 0.365 | 0.647 | 2.264 |
| ∂ID/∂(q, q̇) · GRiD | 0.280 | 0.593 | 5.517 |
| ∂ID/∂(q, q̇) · crocket | 0.135 | 0.307 | 1.713 |
| ∂ID/∂(q, q̇) · frax | 2.209 | 7.654 | 65.492 |

**B = 16384**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · GRiD | 1.715 | 3.634 | 12.733 |
| ID (RNEA) · crocket | 0.045 | 0.059 | 0.473 |
| ID (RNEA) · frax | 1.365 | 3.106 | 11.108 |
| M (CRBA) · GRiD | 1.985 | 5.560 | OOM |
| M (CRBA) · crocket | 0.061 | 0.446 | 11.719 |
| M (CRBA) · frax | 1.199 | 3.061 | 12.229 |
| FD · GRiD | 3.270 | 9.511 | OOM |
| FD · crocket | 0.046 | 0.389 | 1.183 |
| FD · frax | 2.190 | 5.572 | 26.925 |
| ∂ID/∂(q, q̇) · GRiD | 3.853 | 8.556 | OOM |
| ∂ID/∂(q, q̇) · crocket | 0.535 | 1.714 | 30.708 |
| ∂ID/∂(q, q̇) · frax | 30.469 | 118.475 | OOM |

**Max scaled error vs Pinocchio float64, over all batches**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · GRiD | 3.1e-07 | 2.2e-07 | 2.8e-07 |
| ID (RNEA) · crocket | 3.1e-07 | 2.9e-07 | 2.8e-07 |
| ID (RNEA) · frax | 2.4e-07 | 3.6e-07 | 2.9e-07 |
| M (CRBA) · GRiD | 2.3e-07 | 2.1e-07 | 2.0e-07 |
| M (CRBA) · crocket | 2.3e-07 | 1.9e-07 | 3.3e-07 |
| M (CRBA) · frax | 3.8e-07 | 3.9e-07 | 3.5e-07 |
| FD · GRiD | 1.1e-05 | 4.8e-05 | 2.2e-04 |
| FD · crocket | 1.3e-05 | 3.7e-05 | 3.9e-04 |
| FD · frax | 3.2e-05 | 1.1e-03 | 4.0e-03 |
| ∂ID/∂(q, q̇) · GRiD | 2.5e-07 | 2.2e-07 | 2.5e-07 |
| ∂ID/∂(q, q̇) · crocket | 3.1e-07 | 2.6e-07 | 2.8e-07 |
| ∂ID/∂(q, q̇) · frax | 2.0e-07 | 2.9e-07 | 2.1e-07 |
<!-- /tables -->
