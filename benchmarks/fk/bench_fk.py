"""FK: world centres of every collision sphere of the robot's spherized URDF, (B, S, 3).

That is the one FK output every method can produce: pRRTC's and cuRobo's kernels emit spheres
natively; pyroki, GRiM and crocket compute frame poses and place the spheres in the same jit.
GRiM (``traced=False``, GRiD-style baked tables) and crocket (``traced=True``) share everything
but the FK backend. Accuracy is the max error against Pinocchio in float64.

    python fk/bench_fk.py <prrtc|pyroki|curobo|grim|crocket> [robot ...]
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

BATCHES = (1, 128, 1024, 16384)


def _sphere_table(urdf, frame_index):
    """Per sphere: index of the frame it rides on (-1: fixed in the world) and its centre."""
    links, centres, _ = common.spheres(urdf)
    return np.array([frame_index(l) for l in links]), centres.astype(np.float32)


def _place(poses_wxyz_xyz, idx, centres):
    """Sphere centres from (B, F, 7) wxyz+xyz frame poses; idx -1 means the identity frame."""
    import jax.numpy as jnp
    p = jnp.take(poses_wxyz_xyz, jnp.maximum(idx, 0), axis=1)
    w, v, t = p[..., :1], p[..., 1:4], p[..., 4:]
    c = jnp.asarray(centres)
    u = jnp.cross(v, c)
    world = c + 2 * (w * u + jnp.cross(v, u)) + t
    return jnp.where((idx >= 0)[None, :, None], world, c)


def jax_timer(fn, *args):
    import jax
    f = jax.jit(fn)
    out = f(*args)
    return common.amortized_ms(lambda: f(*args), jax.block_until_ready), np.asarray(out)


def run_pyroki(robot, q):
    import jax.numpy as jnp
    import pyroki as pk
    import yourdfpy
    urdf = common.FK_URDF[robot]
    pk_robot = pk.Robot.from_urdf(yourdfpy.URDF.load(str(urdf), load_meshes=False))
    names = list(pk_robot.links.names)
    root = names[0]
    idx, centres = _sphere_table(urdf, lambda l: -1 if l == root else names.index(l))
    perm = [common.joint_names(urdf).index(n) for n in pk_robot.joints.actuated_names]
    qj = jnp.asarray(q[:, perm])
    return jax_timer(lambda x: _place(pk_robot.forward_kinematics(x), idx, centres), qj)


def run_grim(robot, q, traced):
    import jax.numpy as jnp
    from grim.motion import MotionRobot
    from grim.motion.kinematics import kinematics
    urdf = common.FK_URDF[robot]
    m = MotionRobot.from_urdf(str(urdf))
    root = m.link_names[int(np.flatnonzero(m.link_parent_joint < 0)[0])]
    idx, centres = _sphere_table(urdf, lambda l: -1 if l == root else m.link_joint(l))
    perm = [common.joint_names(urdf).index(n) for n in m.actuated_names]
    qj = jnp.asarray(q[:, perm])
    # no end-effector: poses only, no residual or Jacobian
    return jax_timer(lambda x: _place(kinematics(m, x, traced=traced)[0], idx, centres), qj)


def run_curobo(robot, q):
    import torch
    from curobo_robot import curobo_model
    from curobo.types import JointState
    model, perm = curobo_model(robot, list(common.load_inputs("fk", robot)["joint_names"]))
    js = JointState.from_position(torch.as_tensor(np.ascontiguousarray(q[:, perm]), device="cuda"),
                                  joint_names=model.joint_names)
    call = lambda: model.compute_kinematics(js).robot_spheres
    out = call().reshape(len(q), -1, 4)
    ms = common.amortized_ms(call, lambda _: torch.cuda.synchronize())
    return ms, out[..., :3].cpu().numpy()


def run_prrtc(robot, q):
    exe = build_prrtc()
    with tempfile.TemporaryDirectory() as d:
        qp, op = Path(d) / "q.f32", Path(d) / "out.f32"
        q.astype(np.float32).tofile(qp)
        ms = float(subprocess.check_output(
            [str(exe), robot, str(len(q)), str(qp), str(op), str(common.K), str(common.REPS),
             str(common.WARMUP)], text=True))
        return ms, np.fromfile(op, np.float32).reshape(len(q), -1, 3)


def build_prrtc() -> Path:
    import os
    prrtc = Path(os.environ.get("PRRTC_DIR", Path.home() / "Work/prrax/pRRTC"))
    exe = common.RESULTS / "build" / "prrtc_fk"
    src = HERE / "prrtc"
    if not exe.exists() or exe.stat().st_mtime < max(p.stat().st_mtime for p in src.iterdir()):
        exe.parent.mkdir(parents=True, exist_ok=True)
        # CUDA 13 dropped cudaDeviceProp::memoryClockRate, read only by a pRRTC print helper.
        subprocess.check_call(["nvcc", "-O3", "-std=c++17", "-arch=sm_86",
                               "-DmemoryClockRate=major", "-I", str(prrtc), "-I", str(src),
                               str(src / "harness.cu"), "-o", str(exe)])
    return exe


def match(out, ref):
    """Max error after pairing spheres by position: pRRTC and cuRobo order spheres their own
    way. The pairing is fixed from the first configuration and reused for the rest."""
    from scipy.optimize import linear_sum_assignment
    cost = np.linalg.norm(out[0][:, None] - ref[0][None], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(len(cols), int)
    perm[cols] = rows
    return float(np.abs(out[:, perm] - ref).max())


RUNNERS = {"prrtc": run_prrtc, "pyroki": run_pyroki, "curobo": run_curobo,
           "grim": lambda r, q: run_grim(r, q, False), "crocket": lambda r, q: run_grim(r, q, True)}


def main():
    method, robots = sys.argv[1], sys.argv[2:] or list(common.ROBOTS)
    if method == "prepare":
        for robot in robots:
            common.prepare_fk(robot, BATCHES)
        return
    rows = []
    for robot in robots:
        data = common.load_inputs("fk", robot)
        for batch in BATCHES:
            q = data[f"q{batch}"]
            ms, out = RUNNERS[method](robot, q)
            rows_, pos = common.checked(data, batch)
            err = match(out[rows_].astype(np.float64), data["ref"][pos])
            rows.append({"method": method, "robot": robot, "batch": batch, "ms": ms, "max_err": err})
            print(f"{method:8s} {robot:7s} B={batch:6d} {ms:9.4f} ms  err {err:.2e}", flush=True)
    common.save(f"fk_{method}_{'_'.join(robots)}", rows)


if __name__ == "__main__":
    main()
