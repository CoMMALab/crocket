"""The robot set and inputs shared by the tests: cricket's six test robots (kinematic copies in
``resources/``) plus GRiD's ``config/robot_assets`` that stay inside v1's joint types (continuous
joints have their own test: Pinocchio gives them nq = 2), including humanoids for wide trees."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

import crocket
from crocket.robot import GRID_DIR

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
URDFS = {name: RESOURCES / f"{name}.urdf" for name in ("panda", "fr3", "ur5", "baxter", "fetch", "pr2")}
URDFS.update({f"grid_{name}": GRID_DIR / "config" / "robot_assets" / f"{name}.urdf"
              for name in ("fr3", "iiwa14", "rizon4", "baxter", "go2", "g1", "h1_2", "h2_plus")})
ROBOTS = sorted(URDFS)
SMALL = ["panda", "fr3", "ur5", "grid_iiwa14", "grid_go2"]  # for the expensive (GPU / derivative) checks


@lru_cache(maxsize=None)
def robot(name: str) -> crocket.Robot:
    return crocket.Robot(URDFS[name], name=name)


def inputs(r: crocket.Robot, batch: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
    """Random configurations within joint limits, then a zero (often singular) one with zero
    velocity, where conditioning bugs show."""
    rng = np.random.default_rng(seed)
    lo = np.array([max(r.joints[r.joint_names.index(n)].limits[0], -np.pi) for n in r.joint_names])
    hi = np.array([min(r.joints[r.joint_names.index(n)].limits[1], np.pi) for n in r.joint_names])
    q = rng.uniform(lo, hi, (batch, r.n_q))
    q[-1] = np.clip(0.0, lo, hi)
    qd = rng.uniform(-1, 1, (batch, r.n_q))
    qd[-1] = 0.0
    return {"q": q, "qd": qd, "qdd": rng.uniform(-1, 1, (batch, r.n_q)),
            "tau": rng.uniform(-5, 5, (batch, r.n_q))}


def options(r: crocket.Robot, name: str) -> dict:
    """Trace options: frames are every joint frame, the end-effector the deepest movable joint."""
    if name == "frame_poses":
        return {"frames": tuple(r.frames)}
    if name == "frame_jacobian":
        return {"frame": max(r.joints, key=lambda j: j.depth).name}
    if name == "ik_lm_step":
        return {"frame": max(r.joints, key=lambda j: j.depth).name}
    if name == "sphere_collision":
        return {"spheres": sphere_set(r), "n_boxes": 3}
    return {}


def sphere_set(r: crocket.Robot) -> tuple:
    """Two off-centre spheres on every joint frame and one on the base: every frame's pose and
    every centre component reach the margin."""
    base = next(iter(sorted(r.root_links)))
    spheres = [(base, (0.02, -0.01, 0.05), 0.08)]
    for k, j in enumerate(r.joints):
        spheres += [(j.name, (0.03, 0.02 * (k % 3), 0.06), 0.05), (j.name, (-0.04, 0.05, -0.02), 0.04)]
    return tuple(spheres)
