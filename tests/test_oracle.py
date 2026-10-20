"""Surface one: every trace, evaluated by the float64 IR interpreter, against RBDReference and
Pinocchio. CPU only; catches frontend and folding bugs before any CUDA.

    python -m pytest tests/test_oracle.py
"""

from __future__ import annotations

import contextlib
import io
import sys
import warnings

import numpy as np
import pinocchio as pin
import pytest

import crocket
from robots import ROBOTS, SMALL, URDFS, inputs, options, robot

# n^2 tangents: derivatives on the small set plus one humanoid keeps the suite fast. Pinocchio's
# derivatives refuse mimic joints; fr3's are checked by finite differences below.
CASES = ([(r, n) for r in ROBOTS for n in ("rnea", "crba", "aba")]
         + [(r, "rnea_derivatives") for r in SMALL + ["grid_g1"] if r != "fr3"])


def _supported(r, name):
    return not (name == "aba" and any(j.mimic for j in r.joints))


def _pinocchio(r):
    model = pin.buildModelFromUrdf(str(URDFS[r.name]), mimic=True)
    cols = [model.joints[model.getJointId(n)].idx_q for n in r.joint_names]
    vcols = [model.joints[model.getJointId(n)].idx_v for n in r.joint_names]
    return model, np.array(cols), np.array(vcols)


def _rbd(r):
    sys.path.insert(0, str(crocket.robot.GRID_DIR / "external"))
    from RBDReference.RBDReference import RBDReference
    from URDFParser import URDFParser
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(), crocket.robot.exact_origins():
        warnings.simplefilter("ignore")
        return RBDReference(URDFParser().parse(str(URDFS[r.name])))


def _close(got, want, scale=None):
    scale = max(1.0, np.abs(want).max()) if scale is None else scale
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-8 * scale)


@pytest.mark.parametrize("robot_name,name", CASES)
def test_dynamics_vs_pinocchio(robot_name, name):
    r = robot(robot_name)
    if not _supported(r, name):
        with pytest.raises(ValueError, match="mimic"):
            crocket.trace(r, name)
        return
    prog = crocket.trace(r, name)
    x = inputs(r)
    out = prog.evaluate(**{k: x[k] for k, _ in prog.inputs})
    model, cols, vcols = _pinocchio(r)
    data = model.createData()
    n = r.n_q
    for b in range(len(x["q"])):
        q, v = np.zeros(model.nq), np.zeros(model.nv)
        q[cols], v[vcols] = x["q"][b], x["qd"][b]
        a, tau = np.zeros(model.nv), np.zeros(model.nv)
        a[vcols], tau[vcols] = x["qdd"][b], x["tau"][b]
        if name == "rnea":
            _close(out["tau"][b], pin.rnea(model, data, q, v, a)[vcols])
        elif name == "crba":
            M = pin.crba(model, data, q)
            M = np.triu(M) + np.triu(M, 1).T
            _close(out["M"][b].reshape(n, n).T, M[np.ix_(vcols, vcols)])
        elif name == "aba":
            # As a residual, relative to the size of M qdd: qdd itself carries cond(M) (2e7 on
            # h2_plus) times any last-digit difference between the two parsers' inertias.
            a[vcols] = out["qdd"][b]
            scale = 1.0 + np.abs(pin.crba(model, data, q)).max() * np.abs(a).max() + np.abs(x["tau"][b]).max()
            _close(pin.rnea(model, data, q, v, a)[vcols], x["tau"][b], scale)
        else:
            dq, dv, _ = pin.computeRNEADerivatives(model, data, q, v, a)
            _close(out["dtau_dq"][b].reshape(n, n).T, dq[np.ix_(vcols, vcols)])
            _close(out["dtau_dqd"][b].reshape(n, n).T, dv[np.ix_(vcols, vcols)])


@pytest.mark.parametrize("robot_name,name", [(r, n) for r in SMALL for n in ("rnea", "crba", "aba")
                                              if r != "fr3" or n != "aba"])
def test_dynamics_vs_rbdreference(robot_name, name):
    r = robot(robot_name)
    prog = crocket.trace(r, name)
    x = inputs(r)
    out = prog.evaluate(**{k: x[k] for k, _ in prog.inputs})
    ref = _rbd(r)
    for b in range(len(x["q"])):
        q, qd = x["q"][b], x["qd"][b]
        if name == "rnea":
            _close(out["tau"][b], ref.inverse_dynamics(q, qd, x["qdd"][b])[0])
        elif name == "crba":
            _close(out["M"][b].reshape(r.n_q, r.n_q).T, ref.crba(q))
        else:
            _close(out["qdd"][b], ref.forward_dynamics(q, qd, x["tau"][b]))


@pytest.mark.parametrize("robot_name", ROBOTS)
def test_kinematics_vs_pinocchio(robot_name):
    r = robot(robot_name)
    poses = crocket.trace(r, "frame_poses", **options(r, "frame_poses"))
    jac = crocket.trace(r, "frame_jacobian", **options(r, "frame_jacobian"))
    ee = options(r, "frame_jacobian")["frame"]
    x = inputs(r)
    P = poses.evaluate(q=x["q"])["poses"]
    J = jac.evaluate(q=x["q"])
    model, cols, vcols = _pinocchio(r)
    data = model.createData()
    for b in range(len(x["q"])):
        q = np.zeros(model.nq)
        q[cols] = x["q"][b]
        pin.framesForwardKinematics(model, data, q)
        for k, name in enumerate(r.frames):
            M = data.oMf[model.getFrameId(name, pin.JOINT if name in r.joint_names or
                                          any(j.name == name for j in r.joints) else pin.FIXED_JOINT)]
            got = P[b, 12 * k:12 * (k + 1)]
            _close(got[:3], M.translation)
            _close(got[3:].reshape(3, 3).T, M.rotation)
        Jp = pin.computeFrameJacobian(model, data, q, model.getFrameId(ee, pin.JOINT), pin.LOCAL_WORLD_ALIGNED)
        _close(J["jacobian"][b].reshape(6, r.n_q), Jp[:, vcols])
        _close(J["pose"][b][:3], data.oMf[model.getFrameId(ee, pin.JOINT)].translation)


@pytest.mark.parametrize("robot_name", ["panda", "fr3"])
def test_derivatives_match_finite_differences(robot_name):
    """Checks the analytical tangents against the traced RNEA itself (mimic joints included)."""
    r = robot(robot_name)
    x = inputs(r, batch=2)
    rnea = crocket.trace(r, "rnea")
    d = crocket.trace(r, "rnea_derivatives").evaluate(q=x["q"], qd=x["qd"], qdd=x["qdd"])
    h, n = 1e-6, r.n_q
    for wrt, key in (("q", "dtau_dq"), ("qd", "dtau_dqd")):
        fd = np.zeros((2, n, n))
        for c in range(n):
            lo, hi = dict(x), dict(x)
            lo[wrt], hi[wrt] = x[wrt].copy(), x[wrt].copy()
            lo[wrt][:, c] -= h
            hi[wrt][:, c] += h
            args = lambda z: {k: z[k] for k in ("q", "qd", "qdd")}  # noqa: E731
            fd[:, :, c] = (rnea.evaluate(**args(hi))["tau"] - rnea.evaluate(**args(lo))["tau"]) / (2 * h)
        np.testing.assert_allclose(d[key].reshape(2, n, n).transpose(0, 2, 1), fd, atol=1e-5)


def test_folding_removes_constant_work():
    """A revolute joint's transform is a few sin/cos products, not a dense 6x6."""
    r = robot("panda")
    prog = crocket.trace(r, "rnea")
    X = [nd for nd in prog.g.nodes if nd.op == "joint_transform"]
    assert len(X) == len(r.joints)  # recorded once per joint (CSE across both passes)
    assert all(nd.cost <= 14 for nd in X)


def test_dead_outputs_are_eliminated():
    r = robot("panda")
    prog = crocket.trace(r, "rnea")
    live = prog.g.live(prog.flat_outputs[-1:])  # the leaf joint's torque only
    assert live.sum() < prog.g.live(prog.flat_outputs).sum()
