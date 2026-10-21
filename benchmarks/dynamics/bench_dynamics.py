"""Dynamics: inverse dynamics (RNEA), mass matrix (CRBA), forward dynamics, and the inverse
dynamics derivatives d tau / d(q, qd), batched, float32, gravity -9.81 z.

* ``grid``    GRiD's generated kernels (``grid_codegen`` header via ``grim.motion.generated_dynamics``)
* ``crocket`` crocket-traced kernels (``grim.motion.dynamics``), fastest of the thread and
              warp-split tiers per cell; fd is ABA
* ``frax``    frax under ``jax.jit(jax.vmap(...))``; it has no analytical derivatives, so its
              ``id_du`` is ``jax.jacfwd`` of its RNEA

Error is ``max |x - ref| / max(1, max |ref|)`` against Pinocchio in float64.

    python dynamics/bench_dynamics.py <prepare|grid|crocket|frax> [robot ...]
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

BATCHES = (1, 128, 1024, 16384)
OPS = ("id", "crba", "fd", "id_du")
GRAVITY = np.array([0, 0, 9.81, 0, 0, 0])  # frax: spatial gravity acceleration, linear first


def prepare(robot: str) -> None:
    import pinocchio as pin
    urdf = common.MODEL_URDF[robot]
    m = common.model(urdf)
    d = m.createData()
    B = max(BATCHES)
    rng = np.random.default_rng(3)
    q = common.configs(urdf, B, seed=3)
    qd, qdd = (rng.uniform(-1, 1, q.shape).astype(np.float32) for _ in range(2))
    tau = np.empty_like(q)
    ref = {op: [] for op in OPS}
    check = set(common.check_rows(BATCHES).tolist())
    for b in range(B):
        x = [v[b].astype(np.float64) for v in (q, qd, qdd)]
        tau[b] = pin.rnea(m, d, *x)
        if b in check:
            ref["id"].append(tau[b])
            M = pin.crba(m, d, x[0])
            ref["crba"].append(np.triu(M) + np.triu(M, 1).T)
            ref["fd"].append(pin.aba(m, d, x[0], x[1], tau[b].astype(np.float32).astype(np.float64)))
            dq, dv, _ = pin.computeRNEADerivatives(m, d, *x)
            ref["id_du"].append(np.hstack([dq, dv]))
    path = common.inputs_path("dynamics", robot)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, joint_names=np.array(common.joint_names(urdf)), q=q, qd=qd, qdd=qdd, tau=tau,
             check_rows=common.check_rows(BATCHES),
             **{f"ref_{op}": np.array(v) for op, v in ref.items()})


def _reorder(op, x, inv):
    """Outputs from a method's joint order back to Pinocchio's (``inv`` maps it)."""
    if op in ("id", "fd"):
        return x[:, inv]
    if op == "crba":
        return x[:, inv][:, :, inv]
    n = len(inv)
    return np.concatenate([x[:, inv][:, :, :n][:, :, inv], x[:, inv][:, :, n:][:, :, inv]], 2)


def grid_header(urdf: Path) -> tuple[str, list[str]]:
    from grid_codegen import GRiDCodeGenerator
    from URDFParser import URDFParser
    parsed = URDFParser().parse(str(urdf))
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        cwd = os.getcwd()
        try:
            os.chdir(d)
            GRiDCodeGenerator(parsed, False, False).gen_all_code()
        finally:
            os.chdir(cwd)
        header = (Path(d) / "grid.cuh").read_text()
    return header, [j.name for j in parsed.get_joints_ordered_by_id()]


def run_grid(robot, data):
    import jax
    import jax.numpy as jnp
    from grim.motion import generated_dynamics
    header, order = grid_header(common.MODEL_URDF[robot])
    targets, _ = generated_dynamics.build(header)
    perm, inv = _perms(data, order)
    n = len(order)
    shapes = {"id": (n,), "crba": (n, n), "fd": (n,), "id_du": (2 * n, n)}
    names = {"id": "id", "crba": "crba", "fd": "fd", "id_du": "id_grad"}
    # GRiD returns crba and the gradients column-major: transpose to rows.
    post = {"id": lambda x: x, "fd": lambda x: x, "crba": lambda x: jnp.swapaxes(x, 1, 2),
            "id_du": lambda x: jnp.swapaxes(x, 1, 2)}

    def make(op, B):
        call = jax.ffi.ffi_call(targets[names[op]], jax.ShapeDtypeStruct((B, *shapes[op]), jnp.float32))
        fn = lambda q, qd, x: post[op](call(*((q,) if op == "crba" else (q, qd, x)),
                                            gravity=np.float32(-9.81)))
        return {"single": fn}
    return _bench(data, perm, inv, make)


def run_crocket(robot, data):
    from grim.motion import MotionRobot
    from grim.motion.dynamics import TracedDynamics
    m = MotionRobot.from_urdf(str(common.MODEL_URDF[robot]))
    dyn = TracedDynamics(m)
    perm, inv = _perms(data, list(m.actuated_names))
    fns = {"id": dyn.inverse_dynamics, "crba": lambda q, qd, x, tier: dyn.mass_matrix(q, tier),
           "fd": dyn.forward_dynamics, "id_du": dyn.inverse_dynamics_gradient}

    def make(op, B):
        return {tier: (lambda q, qd, x, tier=tier: fns[op](q, qd, x, tier=tier))
                for tier in ("thread", "block")}
    return _bench(data, perm, inv, make)


def run_frax(robot, data):
    import jax
    import jax.numpy as jnp
    import frax
    fr = frax.Robot(str(common.MODEL_URDF[robot]))
    perm, inv = _perms(data, list(fr.joint_names))
    g = jnp.asarray(GRAVITY, jnp.float32)
    rnea = lambda q, qd, qdd: fr.rnea(q, qd, qdd, g, None)
    one = {"id": rnea, "crba": lambda q, qd, x: fr.crba(q),
           "fd": lambda q, qd, tau: fr.forward_dynamics(q, qd, tau, None),
           "id_du": lambda q, qd, qdd: jnp.concatenate(
               jax.jacfwd(rnea, argnums=(0, 1))(q, qd, qdd), axis=1)}

    def make(op, B):
        return {"vmap": jax.vmap(one[op])}
    return _bench(data, perm, inv, make)


def _perms(data, order):
    names = list(data["joint_names"])
    perm = [names.index(n) for n in order]
    return perm, np.argsort(perm)


def _bench(data, perm, inv, make):
    """Per op and batch: the fastest variant's time and its error on the reference rows."""
    import jax
    import jax.numpy as jnp
    rows = []
    for B in BATCHES:
        q, qd, qdd, tau = (jnp.asarray(data[k][:B][:, perm]) for k in ("q", "qd", "qdd", "tau"))
        for op in OPS:
            x = tau if op == "fd" else qdd
            best = None
            for variant, fn in make(op, B).items():
                f = jax.jit(fn)
                try:
                    out = f(q, qd, x)
                    ms = common.amortized_ms(lambda: f(q, qd, x), jax.block_until_ready)
                except jax.errors.JaxRuntimeError as e:
                    if "RESOURCE_EXHAUSTED" not in str(e):
                        raise
                    continue  # out of memory at this batch: reported as OOM
                if best is None or ms < best[0]:
                    best = (ms, variant, np.asarray(out))
            if best is None:
                rows.append({"batch": B, "op": op, "ms": None, "variant": "OOM", "err": None})
                continue
            ms, variant, out = best
            rows_, pos = common.checked(data, B)
            ref = data[f"ref_{op}"][pos]
            got = _reorder(op, out[rows_].astype(np.float64), inv)
            err = float(np.abs(got - ref).max() / max(1.0, np.abs(ref).max()))
            rows.append({"batch": B, "op": op, "ms": ms, "variant": variant, "err": err})
    return rows


RUNNERS = {"grid": run_grid, "crocket": run_crocket, "frax": run_frax}


def main():
    method, robots = sys.argv[1], sys.argv[2:] or list(common.ROBOTS)
    if method == "prepare":
        for robot in robots:
            prepare(robot)
        return
    rows = []
    for robot in robots:
        for row in RUNNERS[method](robot, common.load_inputs("dynamics", robot)):
            row = {"method": method, "robot": robot, **row}
            rows.append(row)
            print(f"{method:8s} {robot:7s} B={row['batch']:6d} {row['op']:6s} "
                  + ("OOM" if row["ms"] is None else f"{row['ms']:9.4f} ms ({row['variant']})  err {row['err']:.1e}"),
                  flush=True)
    common.save(f"dynamics_{method}_{'_'.join(robots)}", rows)


if __name__ == "__main__":
    main()
