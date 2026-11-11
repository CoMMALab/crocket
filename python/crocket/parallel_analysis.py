"""Grama-style parallel metrics for a traced program: the thread tier (one lane, every live scalar
in order) is the serial baseline, a :func:`crocket.schedule.build` schedule is the parallel one.

Read-only over the IR and the schedule; nothing in tracing or emission changes.

Units are folded scalar ops ("FLOPs"), the static cost model the scheduler itself balances:

- ``T_s = W``: live scalar ops, exactly the thread tier's statements.
- ``D``: the span, longest dependency chain in the scalar graph.
- ``T_p``: per stage, the busiest lane's statements (as ``cuda.cooperative`` emits them: the union
  of its items' in-lane walks, so recomputed scalarised nodes count) plus its shared-memory
  traffic, plus ``sync_cost`` per barrier.
- ``T_overhead = p T_p - T_s``, split into redundancy (recomputation), communication
  (smem + barriers) and idle lanes.

This models latency of ONE problem. At batch >= the GPU's thread count, the thread tier already
uses every core at E = 1, so a cooperative tier only pays off when the batch leaves cores idle.

CLI::

    python -m crocket.parallel_analysis rnea panda.urdf baxter.urdf --units 4 8 16 32 64 --out rnea.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import schedule as sched_mod
from .trace import Program

# ponytail: barrier / smem costs in FLOP-equivalents are a guess; calibrate against tune.py timings.
SYNC_COST = 8.0
SMEM_COST = 1.0
# ponytail: FP32 lanes on an RTX A5000 (64 SMs x 128); ignores occupancy, latency hiding and
# register spills, which hit the thread tier first on big traces. Use tune.py for the real pick.
CORES = 8192


def work_span(program: Program) -> tuple[int, int]:
    """``(W, D)`` over the live scalar graph; constants and inputs are free."""
    g = program.g
    live = g.live(program.flat_outputs)
    depth = np.zeros(len(g.op), np.int64)
    w = 0
    for s in np.flatnonzero(live):  # SSA ids are a topological order
        if g.op[s] in ("const", "in"):
            continue
        w += 1
        depth[s] = 1 + max((depth[a] for a in g.args[s]), default=0)
    return w, int(depth.max(initial=0))


def analyze(program: Program, units: int, threshold: int = 8,
            sync_cost: float = SYNC_COST, smem_cost: float = SMEM_COST) -> dict:
    """All metrics for ``program`` scheduled on ``units`` lanes."""
    g = program.g
    W, D = work_span(program)
    sc = sched_mod.build(program, units, threshold)
    t_p, lane_work, smem = 0.0, 0, 0.0
    for st in range(1, sc.stages + 1):
        lanes: dict[int, list] = {}
        for it in sc.stage_items(st):
            lanes.setdefault(it.rank, []).append(it)
        busiest = 0.0
        for items in lanes.values():
            seen, leaves = set(), set()
            for it in items:
                if it.task >= 0:
                    sched_mod.walk(g, it.scalar, it.task, sc.task_of, seen, leaves)
            mem = smem_cost * (len(leaves) + sum(it.scalar in sc.slots for it in items))
            lane_work += len(seen)
            smem += mem
            busiest = max(busiest, len(seen) + mem)
        t_p += busiest + (sync_cost if st in sc.sync_before else 0.0)
    syncs = len(sc.sync_before)
    cost = units * t_p
    S = W / t_p if t_p else 0.0
    t_ideal = max(W / units, D)
    parts = {"recomputation": float(lane_work - W), "smem": smem,
             "barriers": sync_cost * syncs * units}  # every lane waits at a barrier
    parts["idle"] = cost - W - sum(parts.values())
    total = cost - W
    return {
        "units": units, "threshold": threshold, "sync_cost": sync_cost, "smem_cost": smem_cost,
        "W": W, "D": D, "T_s": W, "T_p": t_p, "T_ideal": t_ideal, "schedule_gap": t_p / t_ideal,
        "cost": cost, "speedup": S, "efficiency": S / units, "cost_ratio": cost / W if W else 0.0,
        "parallelism": W / D if D else 0.0,
        "overhead": dict(total=total, **parts),
        "overhead_pct": {k: 100 * v / total if total else 0.0 for k, v in parts.items()},
        "stages": sc.stages, "syncs": syncs, "smem_slots": sc.n_slots,
    }


def batch(program: Program, B: int, units: list[int], cores: int = CORES, **kw) -> dict:
    """Makespan of ``B`` problems per tier: ``ceil(lanes / cores)`` waves of the per-problem time.

    - ``thread``: one problem per lane, the thread-tier code (``W`` statements, no smem/syncs).
    - ``warp``: one problem per 32 lanes, ``T_p`` at 32.
    - ``block<u>``: one problem per ``u`` lanes, ``T_p`` at ``u``.
    - ``split<k>``: 32 problems per ``k`` warps (lane = problem, rank = warp), so ``k`` lanes per
      problem at ``T_p`` with ``k`` ranks: several robots per warp.
    """
    W, _ = work_span(program)
    tp = {u: analyze(program, u, **kw)["T_p"] for u in {32, *units}}
    groups = math.ceil(B / 32)  # warps are the issue unit: partial warps still occupy 32 lanes
    tiers = {"thread": (groups * 32, W), "warp": (B * 32, tp[32])}
    tiers |= {f"block{u}": (B * max(u, 32), tp[u]) for u in units if u >= 32}
    tiers |= {f"split{k}": (groups * 32 * k, tp[k]) for k in units if k < 32}
    out = {name: {"lanes": lanes, "waves": math.ceil(lanes / cores), "per_problem": t,
                  "makespan": math.ceil(lanes / cores) * t} for name, (lanes, t) in tiers.items()}
    best = min(out, key=lambda k: out[k]["makespan"])
    return {"B": B, "cores": cores, "best": best, "tiers": out}


def _slope(xs, ys) -> float | None:
    """Least-squares slope of log y on log x; None with fewer than two distinct x."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len({x for x, _ in pts}) < 2:
        return None
    return float(np.polyfit(*zip(*pts), 1)[0])


def isoefficiency(points: list[dict], target: float) -> dict:
    """Per ``p``, the smallest ``W`` (over the traced robots) that reaches ``efficiency >= target``,
    and the fitted exponent ``k`` in ``W ~ p^k`` (smaller = more scalable)."""
    need = {}
    for pt in points:
        if pt["efficiency"] >= target:
            need[pt["units"]] = min(need.get(pt["units"], math.inf), pt["W"])
    ps = sorted(need)
    return {"target": target, "W_needed": {str(p): need[p] for p in ps},
            "exponent": _slope(ps, [need[p] for p in ps])}


def report(algorithm: str, programs: dict[str, Program], units: list[int], threshold: int = 8,
           target: float = 0.25, sync_cost: float = SYNC_COST, smem_cost: float = SMEM_COST,
           batches: tuple[int, ...] = (), cores: int = CORES) -> dict:
    """Strong scaling per robot, weak scaling (p grown with W), cost optimality and isoefficiency
    across robots, and the batched tier model at each of ``batches``. ``programs`` maps robot name to ``algorithm`` traced on it."""
    kw = dict(threshold=threshold, sync_cost=sync_cost, smem_cost=smem_cost)
    strong = {name: [analyze(p, u, **kw) for u in units] for name, p in programs.items()}
    flat = [pt for pts in strong.values() for pt in pts]

    # weak: the smallest robot gets units[0]; larger ones get p scaled with W (W/p held fixed)
    by_w = sorted(programs.items(), key=lambda kv: work_span(kv[1])[0])
    w0 = work_span(by_w[0][1])[0]
    weak = [dict(robot=name, **analyze(p, max(1, round(units[0] * work_span(p)[0] / w0)), **kw))
            for name, p in by_w]

    # cost optimal iff p T_p = Theta(T_s): slope of log cost vs log W across robots ~ 1 per p
    cost_opt = {str(u): _slope([pt["W"] for pt in flat if pt["units"] == u],
                               [pt["cost"] for pt in flat if pt["units"] == u]) for u in units}
    return {"algorithm": algorithm, "units": units, "strong_scaling": strong, "weak_scaling": weak,
            "cost_optimality_slope": cost_opt, "isoefficiency": isoefficiency(flat, target),
            "batch": {name: [batch(p, B, units, cores, **kw) for B in batches] for name, p in programs.items()}}


def save(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2))


def main(argv=None) -> None:
    from .robot import Robot
    from .trace import ALGORITHMS
    from . import algorithms  # noqa: F401  (registers the algorithms)

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("algorithm")
    ap.add_argument("urdfs", nargs="+")
    ap.add_argument("--units", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--threshold", type=int, default=8)
    ap.add_argument("--target", type=float, default=0.25)
    ap.add_argument("--sync-cost", type=float, default=SYNC_COST)
    ap.add_argument("--smem-cost", type=float, default=SMEM_COST)
    ap.add_argument("--batch", type=int, nargs="*", default=[1, 32, 256, 1024, 4096, 16384, 65536])
    ap.add_argument("--cores", type=int, default=CORES)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    programs = {Path(u).stem: ALGORITHMS[a.algorithm].trace(Robot(u, name=Path(u).stem)) for u in a.urdfs}
    result = report(a.algorithm, programs, a.units, a.threshold, a.target, a.sync_cost, a.smem_cost,
                    a.batch, a.cores)
    for name, pts in result["strong_scaling"].items():
        print(f"{name}: W={pts[0]['W']} D={pts[0]['D']} W/D={pts[0]['parallelism']:.1f}")
        for pt in pts:
            pct = " ".join(f"{k[:5]}={v:3.0f}%" for k, v in pt["overhead_pct"].items())
            print(f"  p={pt['units']:4d} T_p={pt['T_p']:7.1f} T_ideal={pt['T_ideal']:7.1f} "
                  f"gap={pt['schedule_gap']:5.2f} S={pt['speedup']:5.2f} E={pt['efficiency']:.3f} "
                  f"pT_p/T_s={pt['cost_ratio']:6.2f} | {pct}")
        for b in result["batch"][name]:
            span = {k: v["makespan"] for k, v in b["tiers"].items()}
            print(f"  B={b['B']:6d} best={b['best']:8s} " + " ".join(f"{k}={v:.0f}" for k, v in span.items()))
    print("cost-optimality slope (1 = cost optimal):", result["cost_optimality_slope"])
    print("isoefficiency:", result["isoefficiency"])
    if a.out:
        save(result, a.out)


if __name__ == "__main__":
    main()
