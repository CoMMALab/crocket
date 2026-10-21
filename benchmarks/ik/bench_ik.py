"""IK: P pose targets for one end-effector, S seeds each, best seed per problem.

Same targets (FK of random in-limit configurations), seeds, weights (position 50, orientation
10) and iteration cap (60) for every method:

* ``grim``    GRiM ``ls_ik`` (Levenberg-Marquardt kernel) on GRiD-style baked tables
* ``crocket`` crocket's traced LM step (``ik_lm_step``: FK, Jacobian, residual, damped 6x6
              Cholesky solve and joint-limit projection in one traced function) inside a CUDA
              accept/reject loop with the same settings (ik/crocket_ik.py)
* ``crocket_ls_{f32,f64,mixed}``  GRiM's ``ls_ik`` formulation and loop, its step and
              merit traced by crocket, all-float / all-double / GRiM's float-FK, double-solve split (ik/crocket_ik_ls.py)
* ``pyroki``  pyroki's own IK formulation (``pose_cost_analytic_jac`` + ``limit_constraint``)
              solved by jaxls (dense Cholesky LM), vmapped over problems x seeds

Success is judged in float64 by Pinocchio on the returned configuration: position < 1 mm,
orientation < 0.01 rad, inside joint limits.

    python ik/bench_ik.py <prepare|grim|crocket|crocket_ls_f32|crocket_ls_f64|crocket_ls_mixed|pyroki> [robot ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

BATCHES = (1, 128, 1024)
SEEDS = 8
MAX_ITER = 60
POS_W, ORI_W = 50.0, 10.0
EE = {"panda": "panda_grasptarget", "baxter": "right_gripper", "g1": "right_hand_palm_link"}
POS_TOL, ORI_TOL = 1e-3, 1e-2


def prepare(robot: str) -> None:
    urdf = common.MODEL_URDF[robot]
    P = max(BATCHES)
    goal_q = common.configs(urdf, P, seed=1)
    T = common.link_poses_reference(urdf, goal_q, EE[robot])
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_matrix(T[:, :3, :3]).as_quat()
    target = np.concatenate([xyzw[:, [3, 0, 1, 2]], T[:, :3, 3]], 1).astype(np.float32)
    lo, hi = common.limits(urdf)
    seeds = common.configs(urdf, P * SEEDS, seed=2).reshape(P, SEEDS, -1)
    path = common.inputs_path("ik", robot)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, joint_names=np.array(common.joint_names(urdf)), target=target, seeds=seeds,
             lower=lo, upper=hi)


def score(robot: str, q: np.ndarray, target: np.ndarray, lo, hi) -> dict:
    """Best seed per problem, by Pinocchio's float64 pose error."""
    from scipy.spatial.transform import Rotation
    P, S, n = q.shape
    T = common.link_poses_reference(common.MODEL_URDF[robot], q.reshape(P * S, n), EE[robot])
    T = T.reshape(P, S, 4, 4)
    pos = np.linalg.norm(T[..., :3, 3] - target[:, None, 4:], axis=-1)
    R_goal = Rotation.from_quat(target[:, [1, 2, 3, 0]]).as_matrix()
    R_err = np.einsum("pji,psjk->psik", R_goal, T[..., :3, :3])
    ori = np.arccos(np.clip((np.trace(R_err, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    inside = np.all((q >= lo - 1e-4) & (q <= hi + 1e-4), axis=-1)
    ok = (pos < POS_TOL) & (ori < ORI_TOL) & inside
    best = np.argmin(np.where(inside, pos + 0.1 * ori, np.inf), axis=1)
    i = np.arange(P)
    return {"success": float(ok.any(axis=1).mean()), "pos_mm": 1e3 * float(np.median(pos[i, best])),
            "ori_rad": float(np.median(ori[i, best]))}


def run_grim(robot, seeds, target, traced):
    import jax
    import jax.numpy as jnp
    from grim.motion import MotionRobot, ik
    m = MotionRobot.from_urdf(str(common.MODEL_URDF[robot]))
    names = list(common.load_inputs("ik", robot)["joint_names"])
    perm = [names.index(n) for n in m.actuated_names]
    inv = np.argsort(perm)
    t = ik.build("ls", m, (m.link_joint(EE[robot]),), traced=traced)
    s, g = jnp.asarray(seeds[..., perm]), jnp.asarray(target[:, None])
    best = None
    for tier in ik.TIERS:  # the fastest tier per cell, as GRiM recommends choosing by batch
        f = jax.jit(lambda s, g, tier=tier: ik.run(
            t, s, g, max_iter=MAX_ITER, pos_weight=POS_W, ori_weight=ORI_W, lambda_init=5e-3,
            eps_pos=1e-4, eps_ori=1e-3, collision_weight=1e4, collision_margin=0.02, tier=tier,
            block_threads=64)[0])
        out = f(s, g)
        ms = common.amortized_ms(lambda: f(s, g), jax.block_until_ready)
        if best is None or ms < best[0]:
            best = (ms, np.asarray(out)[..., inv], tier)
    return best


def run_pyroki(robot, seeds, target):
    import jax
    import jax.numpy as jnp
    import jaxlie
    import jaxls
    import pyroki as pk
    import yourdfpy
    urdf = common.MODEL_URDF[robot]
    pk_robot = pk.Robot.from_urdf(yourdfpy.URDF.load(str(urdf), load_meshes=False))
    names = list(common.load_inputs("ik", robot)["joint_names"])
    perm = [names.index(n) for n in pk_robot.joints.actuated_names]
    inv = np.argsort(perm)
    link = pk_robot.links.names.index(EE[robot])

    def solve(seed, goal):
        var = pk_robot.joint_var_cls(0)
        costs = [pk.costs.pose_cost_analytic_jac(pk_robot, var, jaxlie.SE3(goal), jnp.array(link),
                                                 pos_weight=POS_W, ori_weight=ORI_W),
                 pk.costs.limit_constraint(pk_robot, var)]
        sol = (jaxls.LeastSquaresProblem(costs=costs, variables=[var]).analyze().solve(
            initial_vals=jaxls.VarValues.make([var.with_value(seed)]), verbose=False,
            linear_solver="dense_cholesky", trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            termination=jaxls.TerminationConfig(max_iterations=MAX_ITER)))
        return sol[var]

    f = jax.jit(jax.vmap(jax.vmap(solve, in_axes=(0, None)), in_axes=(0, 0)))
    s, g = jnp.asarray(seeds[..., perm]), jnp.asarray(target)
    out = f(s, g)
    ms = common.amortized_ms(lambda: f(s, g), jax.block_until_ready)
    return ms, np.asarray(out)[..., inv], "vmap"


def run_crocket(robot, seeds, target):
    import crocket_ik
    return crocket_ik.run(robot, seeds, target, EE[robot], MAX_ITER, POS_W, ORI_W)


def run_crocket_ls(precision):
    def run(robot, seeds, target):
        import crocket_ik_ls
        return crocket_ik_ls.run(robot, seeds, target, EE[robot], MAX_ITER, POS_W, ORI_W, precision)
    return run


RUNNERS = {"grim": lambda r, s, t: run_grim(r, s, t, False), "crocket": run_crocket,
           "crocket_ls_f32": run_crocket_ls("f32"), "crocket_ls_f64": run_crocket_ls("f64"),
           "crocket_ls_mixed": run_crocket_ls("mixed"),
           "pyroki": run_pyroki}


def main():
    method, robots = sys.argv[1], sys.argv[2:] or list(common.ROBOTS)
    if method == "prepare":
        for robot in robots:
            prepare(robot)
        return
    rows = []
    for robot in robots:
        data = common.load_inputs("ik", robot)
        for P in BATCHES:
            seeds, target = data["seeds"][:P], data["target"][:P]
            ms, q, tier = RUNNERS[method](robot, seeds, target)
            row = {"method": method, "robot": robot, "batch": P, "seeds": SEEDS, "ms": ms, "tier": tier,
                   **score(robot, q, target, data["lower"], data["upper"])}
            rows.append(row)
            print(f"{method:8s} {robot:7s} P={P:5d}x{SEEDS} {ms:9.3f} ms ({tier})  success {row['success']:.3f}  "
                  f"pos {row['pos_mm']:.2e} mm  ori {row['ori_rad']:.2e} rad", flush=True)
    common.save(f"ik_{method}_{'_'.join(robots)}", rows)


if __name__ == "__main__":
    main()
