"""Shared by every benchmark: robots, inputs, the float64 Pinocchio reference, and the timer.

Every method gets the same float32 inputs (actuated joints in Pinocchio's order) and is timed
the same way, ``amortized_ms``: ``K`` back-to-back asynchronous calls, one sync at the end,
divided by ``K``; the median of ``REPS`` such runs after warm-up. That is per-call throughput
time, which is what a batched caller sees, and it is measured the same way for JAX, PyTorch and
raw CUDA (the pRRTC harness runs the same loop with CUDA events).
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROBOT_DIR = HERE / "robots"
RESULTS = HERE / "results"
GRIM = HERE.parents[2]  # .../GRiM (crocket is GRiM's external/crocket)

ROBOTS = ("panda", "baxter", "g1")
# FK runs on the spherized URDFs (pRRTC's panda and baxter, pyroffi-grim's g1). IK and dynamics
# need inertias, which pRRTC's baxter lacks, so they use GRiD's baxter; panda and g1 are the
# same file in every benchmark.
FK_URDF = {"panda": ROBOT_DIR / "panda.urdf", "baxter": ROBOT_DIR / "baxter_spherized.urdf",
           "g1": ROBOT_DIR / "g1.urdf"}
MODEL_URDF = {"panda": ROBOT_DIR / "panda.urdf", "baxter": ROBOT_DIR / "baxter.urdf",
              "g1": ROBOT_DIR / "g1.urdf"}

K = 50
REPS = 9
WARMUP = 3


def amortized_ms(call, sync, k: int = K, reps: int = REPS) -> float:
    for _ in range(WARMUP):
        sync(call())
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(k):
            out = call()
        sync(out)
        times.append((time.perf_counter() - t) / k)
    return 1e3 * float(np.median(times))


@lru_cache(maxsize=None)
def model(urdf: Path):
    import pinocchio as pin
    return pin.buildModelFromUrdf(str(urdf))


def joint_names(urdf: Path) -> list[str]:
    """Actuated joints in Pinocchio's order: the column order of every input."""
    m = model(urdf)
    return [m.names[i] for i in range(1, m.njoints)]


def limits(urdf: Path) -> tuple[np.ndarray, np.ndarray]:
    m = model(urdf)
    return (np.maximum(m.lowerPositionLimit, -np.pi), np.minimum(m.upperPositionLimit, np.pi))


def configs(urdf: Path, batch: int, seed: int = 0) -> np.ndarray:
    lo, hi = limits(urdf)
    return np.random.default_rng(seed).uniform(lo, hi, (batch, len(lo))).astype(np.float32)


def spheres(urdf: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Collision spheres in URDF order: (link per sphere, centre in the link frame, radius)."""
    links, centres, radii = [], [], []
    for link in ET.parse(urdf).getroot().iter("link"):
        for col in link.iter("collision"):
            s = col.find("geometry/sphere")
            if s is None:
                continue
            o = col.find("origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
            links.append(link.get("name"))
            centres.append(xyz)
            radii.append(float(s.get("radius")))
    return links, np.array(centres), np.array(radii)


def reference_spheres(urdf: Path, q: np.ndarray) -> np.ndarray:
    """(B, S, 3) world sphere centres in float64."""
    import pinocchio as pin
    m = model(urdf)
    d = m.createData()
    links, centres, _ = spheres(urdf)
    fids = [m.getFrameId(l, pin.BODY) for l in links]
    out = np.empty((len(q), len(links), 3))
    for b, qb in enumerate(q.astype(np.float64)):
        pin.framesForwardKinematics(m, d, qb)
        for s, (f, c) in enumerate(zip(fids, centres)):
            out[b, s] = d.oMf[f].act(c)
    return out


def link_poses_reference(urdf: Path, q: np.ndarray, link: str) -> np.ndarray:
    """(B, 4, 4) world pose of one link in float64."""
    import pinocchio as pin
    m = model(urdf)
    d = m.createData()
    f = m.getFrameId(link, pin.BODY)
    out = np.empty((len(q), 4, 4))
    for b, qb in enumerate(q.astype(np.float64)):
        pin.framesForwardKinematics(m, d, qb)
        out[b] = d.oMf[f].homogeneous
    return out


def check_rows(batches) -> np.ndarray:
    """Rows checked against the reference: the first 32, 32 spread over the largest batch, and
    the last row of every batch (every batch is a prefix of the largest: same seed)."""
    top = max(batches)
    rows = set(range(min(32, top))) | set(np.linspace(0, top - 1, 32).astype(int).tolist())
    return np.array(sorted(rows | {b - 1 for b in batches}))


def checked(data: dict, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """(row indices < batch, their positions in the reference arrays)."""
    pos = np.flatnonzero(data["check_rows"] < batch)
    return data["check_rows"][pos], pos


def inputs_path(bench: str, robot: str) -> Path:
    return RESULTS / "inputs" / f"{bench}_{robot}.npz"


def prepare_fk(robot: str, batches) -> None:
    """Joint order, configurations and float64 reference spheres, written once (Pinocchio is
    only in the JAX envs) and read by every method."""
    urdf = FK_URDF[robot]
    data = {"joint_names": np.array(joint_names(urdf))}
    for b in batches:
        data[f"q{b}"] = configs(urdf, b)
    data["check_rows"] = check_rows(batches)
    data["ref"] = reference_spheres(urdf, data[f"q{max(batches)}"][data["check_rows"]])
    path = inputs_path("fk", robot)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def load_inputs(bench: str, robot: str) -> dict:
    with np.load(inputs_path(bench, robot)) as f:
        return {k: f[k] for k in f.files}


def save(name: str, rows: list[dict]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(rows, indent=1))
    return path
