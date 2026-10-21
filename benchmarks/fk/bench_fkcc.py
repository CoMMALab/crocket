"""Fused FK + environment collision: is any configuration of each 16-configuration group (an
edge, pRRTC's unit of work) in collision with the boxes?

The robot is its spherized URDF's spheres (the FK benchmark's); the world is oriented boxes
placed in the bounds of the robot's reach (uniform centres clear of its static spheres, half-extents
2-6 cm), added until
about half of the groups collide. A sphere collides
when the distance from its centre to a box is below its radius. Self-collision is left out:
the libraries' self pair sets differ (SRDF ranges, adjacency rules).

* ``prrtc``   ``fk<Robot>`` + ``env_collision_check<Robot>`` in one kernel (fk/prrtc/harness.cu)
* ``curobo``  ``RobotCollisionChecker``: kinematics, then the scene-collision kernel
* ``pyroki``  ``RobotCollision.from_sphere_decomposition`` + ``compute_world_collision_distance``
* ``grim``    ``grim.motion.collision`` fused FK + world kernel, baked tables
* ``crocket`` the same kernel built ``traced=True``
* ``crocket_fused`` crocket's ``sphere_collision`` algorithm: FK, sphere placement and the
              squared sphere-box test traced into one straight-line kernel (fk/crocket_fkcc.py)

Agreement is the fraction of groups whose answer matches a float64 reference (Pinocchio
sphere centres, exact sphere-box test) over every configuration of every batch.

    python fk/bench_fkcc.py <prepare|prrtc|curobo|pyroki|grim|crocket|crocket_fused> [robot ...]
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common  # noqa: E402

BATCHES = (1, 128, 1024, 16384)
GROUP = 16
TARGET = 0.5  # fraction of groups in collision


def groups(hit: np.ndarray) -> np.ndarray:
    """(B,) per-configuration hits -> per-group any."""
    g = min(len(hit), GROUP)
    return hit.reshape(-1, g).any(axis=1)


def sphere_box_hit(centres, radii, boxes) -> np.ndarray:
    """(B, S, 3) centres, (S,) radii, (M, 15) boxes -> (B,) any sphere in any box."""
    hit = np.zeros(centres.shape[0], bool)
    for b in boxes:
        R = b[3:12].reshape(3, 3).T  # columns: the box axes
        local = (centres - b[:3]) @ R
        d = np.maximum(np.abs(local) - b[12:15], 0)
        hit |= ((d ** 2).sum(-1) < radii ** 2).any(-1)
    return hit


def prepare(robot: str) -> None:
    from scipy.spatial.transform import Rotation
    urdf = common.FK_URDF[robot]
    fk = common.load_inputs("fk", robot)
    q = fk[f"q{max(BATCHES)}"]
    _, _, radii = common.spheres(urdf)
    centres = common.reference_spheres(urdf, q)
    rng = np.random.default_rng(7)
    sample = common.reference_spheres(urdf, common.configs(urdf, 256, seed=7))  # (256, S, 3)
    # only spheres that move: a box on a base sphere would collide with every configuration
    moving = np.linalg.norm(sample.std(axis=0), axis=-1) > 0.1
    pool = sample[:, moving].reshape(-1, 3)
    static_c, static_r = sample[0, ~moving], radii[~moving]
    lo, hi = pool.min(axis=0), pool.max(axis=0)  # box centres: uniform in the reachable bounds
    boxes, hit = [], np.zeros(len(q), bool)
    while groups(hit).mean() < TARGET - 0.1:  # add boxes until 40-60% of the groups collide
        axes = Rotation.random(random_state=rng).as_matrix()
        half = rng.uniform(0.02, 0.06, 3)
        c = rng.uniform(lo, hi)
        if not (np.linalg.norm(c - static_c, axis=-1) > static_r + np.linalg.norm(half)).all():
            continue
        box = np.r_[c, axes.T.reshape(-1), half]
        new = hit | sphere_box_hit(centres, radii, box[None])
        if groups(new).mean() <= TARGET + 0.1:
            boxes.append(box)
            hit = new
    boxes = np.array(boxes)
    wxyz = Rotation.from_matrix(boxes[:, 3:12].reshape(-1, 3, 3).transpose(0, 2, 1)).as_quat()[:, [3, 0, 1, 2]]
    np.savez(common.inputs_path("fkcc", robot), boxes=boxes.astype(np.float32),
             wxyz=wxyz.astype(np.float32), hit=hit)


def run_prrtc(robot, q, env):
    from bench_fk import build_prrtc
    exe = build_prrtc()
    with tempfile.TemporaryDirectory() as d:
        qp, op, bp = Path(d) / "q.f32", Path(d) / "out.f32", Path(d) / "boxes.f32"
        q.astype(np.float32).tofile(qp)
        env["boxes"].astype(np.float32).tofile(bp)
        ms = float(subprocess.check_output(
            [str(exe), robot, str(len(q)), str(qp), str(op), str(common.K), str(common.REPS),
             str(common.WARMUP), str(bp)], text=True))
        n = -(-len(q) // GROUP)
        return ms, np.fromfile(op, np.float32)[:n] > 0.5


def jax_groups(fn, q):
    """Time ``fn`` (q -> per-configuration hit) plus the group reduction, in one jit."""
    import jax
    g = min(q.shape[0], GROUP)
    f = jax.jit(lambda x: fn(x).reshape(-1, g).any(axis=1))
    out = f(q)
    return common.amortized_ms(lambda: f(q), jax.block_until_ready), np.asarray(out)


def run_grim(robot, q, env, traced):
    import jax.numpy as jnp
    from grim.motion import MotionRobot, collision
    from grim.motion._build import SelfCollision
    from grim.motion.ik import World
    urdf = common.FK_URDF[robot]
    m = MotionRobot.from_urdf(str(urdf))
    links, centres, radii = common.spheres(urdf)
    root = m.link_names[int(np.flatnonzero(m.link_parent_joint < 0)[0])]
    order = list(dict.fromkeys(links))
    sph, start, joint = [], [0], []
    for l in order:
        sph += [np.r_[c, r] for ll, c, r in zip(links, centres, radii) if ll == l]
        start.append(len(sph))
        joint.append(-1 if l == root else m.link_joint(l))
    model = SelfCollision(np.array(sph, np.float32), np.array(start, np.int32),
                          np.array(joint, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32))
    world = World(boxes=env["boxes"])
    c = collision.build(m, model, world_counts=world.counts(), traced=traced)
    perm = [common.load_inputs("fk", robot)["joint_names"].tolist().index(n) for n in m.actuated_names]
    return jax_groups(lambda x: collision.world_distances(c, x, world).min(axis=(1, 2)) < 0,
                      jnp.asarray(q[:, perm]))


def run_pyroki(robot, q, env):
    import jax
    import jax.numpy as jnp
    import pyroki as pk
    import yourdfpy
    from pyroki.collision import Box, RobotCollision
    urdf = common.FK_URDF[robot]
    y = yourdfpy.URDF.load(str(urdf), load_meshes=False)
    pk_robot = pk.Robot.from_urdf(y)
    links, centres, radii = common.spheres(urdf)
    dec: dict[str, dict[str, list]] = {}
    for l, c, r in zip(links, centres, radii):
        dec.setdefault(l, {"centers": [], "radii": []})
        dec[l]["centers"].append(c.tolist())
        dec[l]["radii"].append(float(r))
    rc = RobotCollision.from_sphere_decomposition(dec, y)
    boxes = Box.from_extent(2 * env["boxes"][:, 12:15], env["boxes"][:, :3], env["wxyz"])
    perm = [common.load_inputs("fk", robot)["joint_names"].tolist().index(n)
            for n in pk_robot.joints.actuated_names]
    # one configuration per call, vmapped: the call does not broadcast a config batch
    one = lambda x: rc.compute_world_collision_distance(pk_robot, x, boxes).min() < 0
    return jax_groups(jax.vmap(one), jnp.asarray(q[:, perm]))


def run_curobo(robot, q, env):
    import torch
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from curobo_robot import kinematics_dict
    names = common.load_inputs("fk", robot)["joint_names"].tolist()
    cuboids = {f"box{i}": {"dims": (2 * b[12:15]).tolist(), "pose": [*b[:3].tolist(), *w.tolist()]}
               for i, (b, w) in enumerate(zip(env["boxes"], env["wxyz"]))}
    cfg = RobotCollisionCheckerCfg.load_from_config(
        robot_config={"kinematics": kinematics_dict(common.FK_URDF[robot])},
        scene_model={"cuboid": cuboids}, n_cuboids=len(cuboids),
        collision_activation_distance=0.0)
    checker = RobotCollisionChecker(cfg)
    perm = [names.index(n) for n in checker.kinematics.joint_names]
    qt = torch.as_tensor(np.ascontiguousarray(q[:, perm]), device="cuda")[:, None]
    g = min(len(q), GROUP)

    def call():
        d = checker.get_collision_distance(checker.get_kinematics(qt))
        return (d.reshape(len(q), -1) > 0).any(dim=1).reshape(-1, g).any(dim=1)
    out = call()
    return common.amortized_ms(call, lambda _: torch.cuda.synchronize()), out.cpu().numpy()


def run_crocket_fused(robot, q, env):
    import crocket_fkcc
    return crocket_fkcc.run(robot, q, env)


RUNNERS = {"crocket_fused": run_crocket_fused, "prrtc": run_prrtc, "curobo": run_curobo, "pyroki": run_pyroki,
           "grim": lambda r, q, e: run_grim(r, q, e, False),
           "crocket": lambda r, q, e: run_grim(r, q, e, True)}


def main():
    method, robots = sys.argv[1], sys.argv[2:] or list(common.ROBOTS)
    if method == "prepare":
        for robot in robots:
            prepare(robot)
        return
    rows = []
    for robot in robots:
        q_all = common.load_inputs("fk", robot)[f"q{max(BATCHES)}"]
        env = common.load_inputs("fkcc", robot)
        for B in BATCHES:
            ms, out, *variant = RUNNERS[method](robot, q_all[:B], env)
            want = groups(env["hit"][:B])
            agree = float((np.asarray(out, bool) == want).mean())
            rows.append({"method": method, "robot": robot, "batch": B, "ms": ms,
                         "agree": agree, "boxes": len(env["boxes"]), "variant": (variant or [""])[0],
                         "collide": float(want.mean())})
            print(f"{method:8s} {robot:7s} B={B:6d} {ms:9.4f} ms {''.join(variant)}  agree {agree:.4f}  "
                  f"({len(env['boxes'])} boxes, {want.mean():.2f} of groups collide)", flush=True)
    common.save(f"fkcc_{method}_{'_'.join(robots)}", rows)


if __name__ == "__main__":
    main()
