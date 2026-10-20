"""Autotuning: pick the tier, group size and scalarisation threshold per robot and algorithm by
measurement, as GRiD's ``config/launch_configs`` does by hand.

The cost model first rejects layouts whose shared scratch would cut residency below
``MIN_WARPS_PER_SM``; the remaining candidates are compiled (in parallel), validated against the
interpreter and timed once. The winner is cached on disk under
(robot hash, algorithm, options, batch, precision, crocket version, nvcc version).
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from . import __version__, cuda, harness, schedule
from .trace import Program

SMEM_PER_SM = 100 * 1024  # bytes (RTX A5000 / sm_86)
MIN_WARPS_PER_SM = 8


@dataclass
class LaunchConfig:
    tier: str
    units: int
    threshold: int
    block: int  # threads per block at launch (thread / warp tiers; fixed by units otherwise)
    ms: float

    @property
    def variant(self) -> cuda.Variant:
        return cuda.Variant(self.tier, self.units, self.threshold)


def candidates(p: Program, dtype=np.float32) -> list[tuple[cuda.Variant, int]]:
    """(variant, launch block size) pairs that pass the occupancy floor."""
    size = np.dtype(dtype).itemsize
    out = [(cuda.Variant("thread"), b) for b in (64, 128, 256)]
    for threshold in (1, 8, 64):
        layouts = ([(cuda.Variant("split", w, threshold), 32 * w) for w in (2, 4, 8)]
                   + [(cuda.Variant("warp", 32, threshold), b) for b in (64, 128, 256)]
                   + [(cuda.Variant("block", t, threshold), t) for t in (32, 64, 128, 256)])
        scheds = {}
        for v, block in layouts:
            key = (v.units if v.tier != "warp" else 32)
            sc = scheds.get((v.tier, key)) or schedule.build(p, key, threshold)
            scheds[v.tier, key] = sc
            per_block = cuda.smem(sc, v.tier) * size * (block // 32 if v.tier == "warp" else 1)
            blocks = SMEM_PER_SM // per_block if per_block else 32
            if per_block <= harness.SMEM_LIMIT and blocks * block // 32 >= MIN_WARPS_PER_SM:
                out.append((v, block))
    return out


def _key(p: Program, batch: int, dtype, options: dict) -> str:
    robot = p.robot
    model = repr([(j.name, j.kind, j.parent, j.q_index, j.multiplier, j.offset, j.E.tobytes(), j.r.tobytes(),
                   j.S.tobytes(), j.I.tobytes()) for j in robot.joints]
                 + sorted((f.name, f.parent, f.T.tobytes()) for f in robot.frames.values()))
    return hashlib.sha1("\x00".join([model, p.name, repr(sorted(options.items())), str(batch),
                                     np.dtype(dtype).name, __version__, harness.nvcc_version()]).encode()).hexdigest()


def tune(p: Program, batch: int, dtype=np.float32, reps: int = 50, options: dict | None = None,
         refresh: bool = False) -> LaunchConfig:
    """The fastest validated layout of ``p`` at ``batch`` problems per launch (cached)."""
    path = harness.cache_dir() / "launch_configs" / f"{_key(p, batch, dtype, options or {})}.json"
    if path.exists() and not refresh:
        return LaunchConfig(**json.loads(path.read_text()))
    cands = candidates(p, dtype)
    variants = list({(v.tier, v.units, v.threshold): v for v, _ in cands}.values())
    with ThreadPoolExecutor() as pool:
        built = dict(zip(((v.tier, v.units, v.threshold) for v in variants),
                         pool.map(lambda v: harness.build([(p, v)], dtype), variants)))
    x = harness.sample_inputs(p, batch)
    best = None
    for v, block in cands:
        b = built[v.tier, v.units, v.threshold]
        harness.validate(b)
        _, ms = b.run(0, x, block=block, reps=reps)
        if best is None or ms < best.ms:
            best = LaunchConfig(v.tier, v.units, v.threshold, block, ms)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(best)))
    return best
