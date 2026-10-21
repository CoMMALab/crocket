"""Robot model: a URDF parsed by GRiD's ``URDFParser``, reduced to the constants tracing folds.

Each movable joint's transform is recovered numerically from URDFParser's own
``get_Xmat_Func_by_id`` so both compilers see the same model, joint order and conventions:
the parent-to-child Plücker transform ``X(q) = [[E, 0], [-E rx, E]]`` is stored as its
rotation ``E`` and translation ``r``, each entry a fitted ``a + b cos q + c sin q``
(revolute) or ``a + b q`` (prismatic). Fixed joints are merged by the parser; their frames
are kept as constant poses relative to their movable parent joint.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRID_DIR = Path(os.environ.get("CROCKET_GRID_PATH", Path(__file__).resolve().parents[3] / "GRiD"))
GRAVITY = -9.81
_TOL = 1e-12


def _import_parser():
    """``URDFParser`` from an installed GRiD, else from the ``GRiD`` checkout next to crocket."""
    try:
        from URDFParser import URDFParser
    except ImportError:
        sys.path.insert(0, str(GRID_DIR / "external"))
        from URDFParser import URDFParser
    return URDFParser


@contextlib.contextmanager
def exact_origins():
    """Parse URDF origins as written. URDFParser snaps rpy within 1e-5 of a multiple of pi/2 onto
    it; Pinocchio, which is authoritative, does not, and the snap moves frames by up to 1e-5 rad."""
    _import_parser()
    joint_module = importlib.import_module("URDFParser.Joint")
    snap = joint_module._snap_to_pi_grid
    joint_module._snap_to_pi_grid = lambda value, tolerance=0.0: value
    try:
        yield
    finally:
        joint_module._snap_to_pi_grid = snap


@dataclass
class Joint:
    name: str
    kind: str  # "revolute" or "prismatic"
    parent: int  # movable parent joint, -1 for the base
    depth: int
    q_index: int  # column in q / qd (the mimicked joint's for a mimic joint)
    multiplier: float  # q_joint = multiplier * q[q_index] + offset
    offset: float
    mimic: bool
    limits: tuple[float, float]
    E: np.ndarray  # (3, 3, 3): [constant, cos, sin] or [constant, q, 0] coefficients
    r: np.ndarray  # (3, 3)
    S: np.ndarray  # (6,) motion subspace, [angular; linear]
    I: np.ndarray  # (6, 6) spatial inertia of the child body, joint frame


@dataclass
class Frame:
    name: str
    parent: int  # movable joint, -1 for the base
    T: np.ndarray  # (4, 4) pose in the parent joint's frame


def _clean(a: np.ndarray) -> np.ndarray:
    a = np.where(np.abs(a) < _TOL, 0.0, a)
    for v in (1.0, -1.0):
        a = np.where(np.abs(a - v) < _TOL, v, a)
    return a


def _split(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E = X[:3, :3]
    rx = -E.T @ X[3:, :3]
    return E, np.array([rx[2, 1], rx[0, 2], rx[1, 0]])


def _fit(kind: str, X) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients of ``E(q)`` and ``r(q)`` from samples of the parser's transform."""
    if kind == "revolute":
        (E0, r0), (E1, r1), (E2, r2) = (_split(X(t)) for t in (0.0, np.pi / 2, np.pi))
        Ea, ra = (E0 + E2) / 2, (r0 + r2) / 2
        E, r = np.stack([Ea, (E0 - E2) / 2, E1 - Ea]), np.stack([ra, (r0 - r2) / 2, r1 - ra])
        model = lambda C, t: C[0] + C[1] * np.cos(t) + C[2] * np.sin(t)  # noqa: E731
    else:
        (E0, r0), (E1, r1) = (_split(X(t)) for t in (0.0, 1.0))
        E = np.stack([E0, E1 - E0, np.zeros((3, 3))])
        r = np.stack([r0, r1 - r0, np.zeros(3)])
        model = lambda C, t: C[0] + C[1] * t  # noqa: E731
    Et, rt = _split(X(0.7))
    if not (np.allclose(model(E, 0.7), Et, atol=1e-9) and np.allclose(model(r, 0.7), rt, atol=1e-9)):
        raise ValueError(f"{kind} joint transform is not of the expected form")
    return _clean(E), _clean(r)


class Robot:
    """The kinematic tree of one URDF, movable joints in URDFParser's (Pinocchio) order."""

    def __init__(self, urdf: str | Path, name: str | None = None):
        urdf = Path(urdf)
        URDFParser = _import_parser()
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(), exact_origins():
            warnings.simplefilter("ignore")
            parsed = URDFParser().parse(str(urdf))
        self.name = name or parsed.get_name()
        self.n_q = parsed.get_num_pos()
        self.joints: list[Joint] = []
        for jid in range(parsed.get_num_bodies()):
            j = parsed.get_joint_by_id(jid)
            if j.jtype not in ("revolute", "prismatic"):
                raise ValueError(f"Unsupported joint type {j.jtype!r} (joint {j.name!r}): "
                                 "crocket supports revolute, prismatic and fixed joints")
            parent = parsed.get_parent_id(jid)
            E, r = _fit(j.jtype, parsed.get_Xmat_Func_by_id(jid))
            self.joints.append(Joint(
                name=j.name, kind=j.jtype, parent=parent,
                depth=0 if parent == -1 else self.joints[parent].depth + 1,
                q_index=int(parsed.get_joint_index_q(jid)),
                multiplier=float(j.mimic_multiplier) if j.is_mimic else 1.0,
                offset=float(j.mimic_offset) if j.is_mimic else 0.0, mimic=bool(j.is_mimic),
                limits=tuple(float(x) for x in parsed.get_joint_limits_by_id(jid)) or (-np.pi, np.pi),
                E=E, r=r, S=_clean(np.asarray(parsed.get_S_by_id(jid), float).reshape(6)),
                I=np.asarray(parsed.get_Imat_by_id(jid), float)))
        by_name = {j.name: i for i, j in enumerate(self.joints)}
        self.frames: dict[str, Frame] = {
            j.name: Frame(j.name, i, np.eye(4)) for i, j in enumerate(self.joints)}
        for f in parsed.fixed_joints:
            parent = -1 if f.parent_name in (-1, None) else by_name[f.parent_name]
            self.frames[f.name] = Frame(f.name, parent, _clean(np.array(f.Xmat_hom, dtype=float)))
        # link -> the joint whose child it is, so a frame can also be named by its link
        root = ET.parse(urdf).getroot()
        self.link_joint = {el.find("child").get("link"): el.get("name") for el in root.findall("joint")}
        self.root_links = {l.get("name") for l in root.findall("link")} - set(self.link_joint)
        # q's columns, by the joint that owns each (mimic joints own none)
        self.joint_names = [""] * self.n_q
        for j in self.joints:
            if not j.mimic:
                self.joint_names[j.q_index] = j.name

    def frame(self, name: str) -> Frame:
        """A joint or link frame by name."""
        if name in self.frames:
            return self.frames[name]
        if self.link_joint.get(name) in self.frames:
            return self.frames[self.link_joint[name]]
        if name in self.root_links:  # the base link is the world frame
            return Frame(name, -1, np.eye(4))
        raise ValueError(f"Invalid EE name {name!r}: no joint or link of that name")

    def X(self, j: int, q: float) -> np.ndarray:
        """Joint ``j``'s 6x6 parent-to-child transform at its own coordinate ``q`` (float64)."""
        jt = self.joints[j]
        basis = (1.0, np.cos(q), np.sin(q)) if jt.kind == "revolute" else (1.0, q, 0.0)
        E = sum(b * c for b, c in zip(basis, jt.E))
        r = sum(b * c for b, c in zip(basis, jt.r))
        rx = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        return np.block([[E, np.zeros((3, 3))], [-E @ rx, E]])
