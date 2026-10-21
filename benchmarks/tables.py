"""results/*.json -> the tables between the markers in README.md (robots as columns)."""

from __future__ import annotations

import json
import re

import common

FK = {"prrtc": "pRRTC `fk<Robot>`", "curobo": "cuRobo", "pyroki": "pyroki",
      "grim": "GRiD/GRiM (baked tables)", "crocket": "crocket (traced)"}
FKCC = {"prrtc": "pRRTC `fk` + `env_collision_check`", "curobo": "cuRobo `RobotCollisionChecker`",
        "pyroki": "pyroki `RobotCollision`", "grim": "GRiD/GRiM fused (baked tables)",
        "crocket": "GRiM fused, crocket-traced FK", "crocket_fused": "crocket `sphere_collision` (fully traced)"}
IK = {"grim": "GRiM LS IK, GRiD tables", "crocket": "crocket traced LM step",
      "pyroki": "pyroki + jaxls"}
DYN = {"grid": "GRiD", "crocket": "crocket", "frax": "frax"}
OPS = {"id": "ID (RNEA)", "crba": "M (CRBA)", "fd": "FD", "id_du": "∂ID/∂(q, q̇)"}


def load(prefix: str) -> list[dict]:
    rows = []
    for p in sorted(common.RESULTS.glob(f"{prefix}_*.json")):
        rows += json.loads(p.read_text())
    return rows


def table(head: str, rows: list[tuple[str, list[str]]]) -> str:
    out = [f"| {head} | " + " | ".join(common.ROBOTS) + " |",
           "|---|" + "---|" * len(common.ROBOTS)]
    out += [f"| {name} | " + " | ".join(cells) + " |" for name, cells in rows]
    return "\n".join(out)


def cell(rows, **key):
    hit = [r for r in rows if all(r[k] == v for k, v in key.items())]
    return hit[0] if hit else None


def fk_section() -> str:
    rows = load("fk")
    parts = ["### FK: world centres of every collision sphere\n",
             f"Spheres per robot: panda 59, baxter 75, g1 68. Time per call in ms (lower is better)."]
    for B in sorted({r["batch"] for r in rows}):
        parts.append(f"\n**B = {B}**\n")
        parts.append(table("method", [(label, [f"{c['ms']:.3f}" if (c := cell(rows, method=m, robot=r, batch=B)) else "—"
                                              for r in common.ROBOTS]) for m, label in FK.items()]))
    parts.append("\n**Max sphere-centre error vs Pinocchio float64 (m), over all batches**\n")
    parts.append(table("method", [(label, [f"{max(x['max_err'] for x in rows if x['method'] == m and x['robot'] == r):.1e}"
                                          if any(x['method'] == m and x['robot'] == r for x in rows) else "—"
                                          for r in common.ROBOTS]) for m, label in FK.items()]))
    return "\n".join(parts)


def fkcc_section() -> str:
    rows = load("fkcc")
    env = {r["robot"]: r for r in rows if r["batch"] == max(x["batch"] for x in rows)}
    parts = ["### Fused FK + collision: is any of each 16-configuration group in collision with the boxes?\n",
             "Boxes per robot: " + ", ".join(f"{r} {env[r]['boxes']}" for r in common.ROBOTS if r in env)
             + f"; about {100 * min(e['collide'] for e in env.values()):.0f}% of groups collide. "
             "Time per call in ms. Every method agrees with the float64 reference on "
             + ("every group." if all(r["agree"] == 1.0 for r in rows) else "the share given below.")]
    for B in sorted({r["batch"] for r in rows}):
        parts.append(f"\n**B = {B}**\n")
        parts.append(table("method", [(label, [f"{c['ms']:.3f}" if (c := cell(rows, method=m, robot=r, batch=B)) else "—"
                                              for r in common.ROBOTS]) for m, label in FKCC.items()]))
    if not all(r["agree"] == 1.0 for r in rows):
        parts.append("\n**Agreement with the reference (min over batches)**\n")
        parts.append(table("method", [(label, [f"{min(x['agree'] for x in rows if x['method'] == m and x['robot'] == r):.4f}"
                                              for r in common.ROBOTS]) for m, label in FKCC.items()]))
    return "\n".join(parts)


def ik_section() -> str:
    rows = load("ik")
    parts = ["### IK: one end-effector pose, 8 seeds per problem, 60 iterations max\n",
             "Time per call in ms, then success rate (position < 1 mm, orientation < 0.01 rad, "
             "within limits; Pinocchio float64). GRiM and crocket report their fastest tier."]
    for P in sorted({r["batch"] for r in rows}):
        parts.append(f"\n**{P} problem{'s' if P > 1 else ''} × 8 seeds**\n")
        parts.append(table("method", [(label, [
            f"{c['ms']:.2f} ({100 * c['success']:.1f}%, {c['tier']})" if (c := cell(rows, method=m, robot=r, batch=P)) else "—"
            for r in common.ROBOTS]) for m, label in IK.items()]))
    return "\n".join(parts)


def dyn_section() -> str:
    rows = load("dynamics")
    parts = ["### Dynamics\n", "Time per call in ms. crocket reports its fastest tier. OOM: the call's workspace does not "
             "fit in the GPU's 24 GB at that batch."]
    for B in sorted({r["batch"] for r in rows}):
        parts.append(f"\n**B = {B}**\n")
        parts.append(table("op · method", [(f"{OPS[op]} · {label}", [
            ("OOM" if c["ms"] is None else f"{c['ms']:.3f}") if (c := cell(rows, method=m, robot=r, batch=B, op=op)) else "—"
            for r in common.ROBOTS]) for op in OPS for m, label in DYN.items()]))
    parts.append("\n**Max scaled error vs Pinocchio float64, over all batches**\n")
    parts.append(table("op · method", [(f"{OPS[op]} · {label}", [
        f"{max(x['err'] for x in rows if x['method'] == m and x['robot'] == r and x['op'] == op and x['err'] is not None):.1e}"
        if any(x['method'] == m and x['robot'] == r and x['op'] == op for x in rows) else "—"
        for r in common.ROBOTS]) for op in OPS for m, label in DYN.items()]))
    return "\n".join(parts)


def main():
    readme = common.HERE / "README.md"
    body = "\n\n".join([fk_section(), fkcc_section(), ik_section(), dyn_section()])
    text = readme.read_text()
    new = re.sub(r"(<!-- tables -->\n).*(<!-- /tables -->)", lambda m: m[1] + body + "\n" + m[2],
                 text, flags=re.S)
    readme.write_text(new)
    print(body)


if __name__ == "__main__":
    main()
