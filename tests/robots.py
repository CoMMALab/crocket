"""The robot set and inputs shared by the tests: cricket's six test robots (kinematic copies in
``resources/``) plus GRiD's ``config/robot_assets`` that stay inside v1's joint types (gen3's
continuous joints are a v1 non-goal), including humanoids for wide trees."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

import cricket2
from cricket2.robot import GRID_DIR

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
URDFS = {name: RESOURCES / f"{name}.urdf" for name in ("panda", "fr3", "ur5", "baxter", "fetch", "pr2")}
URDFS.update({f"grid_{name}": GRID_DIR / "config" / "robot_assets" / f"{name}.urdf"
              for name in ("fr3", "iiwa14", "rizon4", "baxter", "go2", "g1", "h1_2", "h2_plus")})
ROBOTS = sorted(URDFS)
SMALL = ["panda", "fr3", "ur5", "grid_iiwa14", "grid_go2"]  # for the expensive (GPU / derivative) checks


@lru_cache(maxsize=None)
def robot(name: str) -> cricket2.Robot:
    return cricket2.Robot(URDFS[name], name=name)


def inputs(r: cricket2.Robot, batch: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
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


def options(r: cricket2.Robot, name: str) -> dict:
    """Trace options: frames are every joint frame, the end-effector the deepest movable joint."""
    if name == "frame_poses":
        return {"frames": tuple(r.frames)}
    if name == "frame_jacobian":
        return {"frame": max(r.joints, key=lambda j: j.depth).name}
    return {}
