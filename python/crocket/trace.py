"""Tracing by execution: algorithms are plain Python over these spatial types, and running one
against a robot records the IR (cricket's idea, moved from scalars to spatial values).

Topology loops run in Python and vanish. Iterating ``robot.joints`` (or ``reversed`` of it)
tags every op recorded inside the loop body with that joint, its depth and subtree and the
pass, so the scheduler sees the tree without the algorithm author writing anything extra.

Spatial vectors are ``[angular; linear]``. :class:`SE3` is a parent-to-child Plücker
transform ``X = [[E, 0], [-E rx, E]]``, held as ``E`` (3x3) and ``r`` (the child origin in
parent coordinates); ``X @ v`` is ``motion_transform``, ``X.T @ f`` is
``force_transform_inv``, matching GLASS names.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass

import numpy as np

from .ir import Graph
from .robot import GRAVITY, Robot

ALGORITHMS: dict[str, "Algorithm"] = {}


class Tracer:
    def __init__(self, robot: Robot):
        self.g = Graph()
        self.robot = robot
        self.tags: dict = {}

    def record(self, op, cls, args, expand, key=None):
        frame = inspect.currentframe().f_back.f_back
        while frame and frame.f_code.co_filename == __file__:
            frame = frame.f_back
        tags = dict(self.tags, line=frame.f_lineno if frame else 0)
        nid = self.g.node(op, cls.__name__, tuple(a.node for a in args), expand, tags, key)
        return cls(self, nid)

    def const(self, cls, values):
        values = np.asarray(values, np.float64)
        key = (cls.__name__, values.shape, values.tobytes())
        return self.record("const", cls, (), lambda: np.vectorize(self.g.const, otypes=[np.int64])(values)
                           if values.size else np.zeros(values.shape, np.int64), key)


class Value:
    def __init__(self, tracer: Tracer, node: int):
        self.t = tracer
        self.node = node

    @property
    def elems(self) -> np.ndarray:
        return self.t.g.nodes[self.node].elems

    def __getitem__(self, k: int) -> "Scalar":
        return self.t.record("element", Scalar, (self,), lambda e: e.reshape(-1)[k], key=k)

    def _s(self, op, *args):
        return self.t.g.s(op, *args)

    def _binary(self, other, op, name):
        if type(other) is not type(self):
            return NotImplemented
        return self.t.record(name, type(self), (self, other), lambda a, b: np.array(
            [self._s(op, int(x), int(y)) for x, y in zip(a.flat, b.flat)]).reshape(a.shape))

    def __add__(self, other):
        return self._binary(other, "add", "add")

    def __sub__(self, other):
        return self._binary(other, "sub", "sub")

    def __neg__(self):
        return self.t.record("neg", type(self), (self,),
                             lambda a: np.vectorize(lambda x: self._s("neg", int(x)), otypes=[np.int64])(a))

    def _scale(self, k):
        k = k if isinstance(k, Scalar) else Scalar.const(self.t, k)
        return self.t.record("scale", type(self), (self, k),
                             lambda a, s: np.vectorize(lambda x: self._s("mul", int(x), int(s)),
                                                       otypes=[np.int64])(a))

    def __mul__(self, k):
        if isinstance(k, (Scalar, int, float)):
            return self._scale(k)
        return NotImplemented

    __rmul__ = __mul__


def _cross(g, a, b):
    return [g.s("sub", g.s("mul", a[1], b[2]), g.s("mul", a[2], b[1])),
            g.s("sub", g.s("mul", a[2], b[0]), g.s("mul", a[0], b[2])),
            g.s("sub", g.s("mul", a[0], b[1]), g.s("mul", a[1], b[0]))]


def _matvec(g, M, v):
    out = []
    for row in M:
        acc = 0
        for m, x in zip(row, v):
            acc = g.s("add", acc, g.s("mul", int(m), int(x)))
        out.append(acc)
    return out


def _dot(g, a, b):
    acc = 0
    for x, y in zip(a, b):
        acc = g.s("add", acc, g.s("mul", int(x), int(y)))
    return acc


class Scalar(Value):
    @staticmethod
    def const(t, v):
        return t.const(Scalar, v)

    def _lift(self, other):
        return other if isinstance(other, Scalar) else Scalar.const(self.t, other)

    def _op(self, op, *args):
        return self.t.record(op, Scalar, args, lambda *e: self._s(op, *(int(x) for x in e)))

    def __add__(self, o):
        return self._op("add", self, self._lift(o))

    def __radd__(self, o):
        return self._op("add", self._lift(o), self)

    def __sub__(self, o):
        return self._op("sub", self, self._lift(o))

    def __rsub__(self, o):
        return self._op("sub", self._lift(o), self)

    def __mul__(self, o):
        if isinstance(o, (Scalar, int, float)):
            return self._op("mul", self, self._lift(o))
        return o.__rmul__(self)

    def __rmul__(self, o):
        return self._op("mul", self._lift(o), self)

    def __truediv__(self, o):
        return self._op("div", self, self._lift(o))

    def __rtruediv__(self, o):
        return self._op("div", self._lift(o), self)

    def __neg__(self):
        return self._op("neg", self)

    def sin(self):
        return self._op("sin", self)

    def cos(self):
        return self._op("cos", self)

    def sqrt(self):
        return self._op("sqrt", self)


class Vec3(Value):
    def cross(self, o):
        return self.t.record("cross", Vec3, (self, o), lambda a, b: _cross(self.t.g, a, b))


class Motion(Value):
    @staticmethod
    def const(t, v):
        return t.const(Motion, v)

    @staticmethod
    def zero(t):
        return t.const(Motion, np.zeros(6))

    def cross(self, o: "Motion") -> "Motion":
        """``self x o`` (``motion_cross``)."""
        g = self.t.g
        return self.t.record("motion_cross", Motion, (self, o), lambda a, b: np.array(
            _cross(g, a[:3], b[:3])
            + [g.s("add", x, y) for x, y in zip(_cross(g, a[:3], b[3:]), _cross(g, a[3:], b[:3]))]))

    def cross_dual(self, f: "Force") -> "Force":
        """``self x* f`` (``force_cross``)."""
        g = self.t.g
        return self.t.record("force_cross", Force, (self, f), lambda a, b: np.array(
            [g.s("add", x, y) for x, y in zip(_cross(g, a[:3], b[:3]), _cross(g, a[3:], b[3:]))]
            + _cross(g, a[:3], b[3:])))

    def dot(self, f: "Force") -> Scalar:
        return self.t.record("dot", Scalar, (self, f), lambda a, b: _dot(self.t.g, a, b))

    def at(self, p: Vec3) -> "Motion":
        """The same motion with its linear part taken at point ``p`` (``motion_shift``)."""
        g = self.t.g
        return self.t.record("motion_shift", Motion, (self, p), lambda a, b: np.array(
            list(a[:3]) + [g.s("add", int(x), y) for x, y in zip(a[3:], _cross(g, a[:3], b))]))


class Force(Value):
    def dot(self, m: Motion) -> Scalar:
        return m.dot(self)

    def outer(self, o: "Force") -> "Inertia":
        """``self o^T`` (a symmetric rank-1 update when ``o`` is ``self`` scaled)."""
        g = self.t.g
        return self.t.record("outer", Inertia, (self, o), lambda a, b: np.array(
            [[g.s("mul", int(x), int(y)) for y in b] for x in a]))


class Inertia(Value):
    @staticmethod
    def const(t, v):
        return t.const(Inertia, v)

    def __matmul__(self, m: Motion) -> Force:
        return self.t.record("inertia_mul", Force, (self, m), lambda I, v: np.array(_matvec(self.t.g, I, v)))


def _x6(g, e):
    """The dense 6x6 ``X`` of an SE3 expansion ``[E row-major (9), r (3)]``."""
    E, r = e[:9].reshape(3, 3), e[9:]
    rx = np.array([[0, g.s("neg", r[2]), r[1]], [r[2], 0, g.s("neg", r[0])],
                   [g.s("neg", r[1]), r[0], 0]], np.int64)
    X = np.zeros((6, 6), np.int64)
    X[:3, :3] = X[3:, 3:] = E
    X[3:, :3] = [[g.s("neg", _dot(g, E[i], rx[:, k])) for k in range(3)] for i in range(3)]
    return X


class SE3(Value):
    @staticmethod
    def const_pose(t, T):
        """The transform whose child frame sits at pose ``T`` (4x4) in its parent."""
        return t.const(SE3, np.concatenate([T[:3, :3].T.reshape(9), T[:3, 3]]))

    @property
    def translation(self) -> Vec3:
        return self.t.record("translation", Vec3, (self,), lambda e: e[9:].copy())

    @property
    def T(self) -> "_Transposed":
        return _Transposed(self)

    def inv(self) -> "_Inverse":
        return _Inverse(self)

    def __matmul__(self, o):
        g = self.t.g
        if isinstance(o, SE3):  # compose: o first, then self
            def compose(a, b):
                Ea, ra, Eb, rb = a[:9].reshape(3, 3), a[9:], b[:9].reshape(3, 3), b[9:]
                E = [[_dot(g, Ea[i], Eb[:, k]) for k in range(3)] for i in range(3)]
                r = [g.s("add", int(rb[i]), _dot(g, Eb[:, i], ra)) for i in range(3)]
                return np.array(sum(E, []) + r)
            return self.t.record("se3_compose", SE3, (self, o), compose)
        if isinstance(o, Motion):
            def mt(e, v):
                E, r = e[:9].reshape(3, 3), e[9:]
                u = [g.s("sub", int(x), y) for x, y in zip(v[3:], _cross(g, r, v[:3]))]
                return np.array(_matvec(g, E, v[:3]) + _matvec(g, E, u))
            return self.t.record("motion_transform", Motion, (self, o), mt)
        if isinstance(o, Force):
            def ft(e, f):
                E, r = e[:9].reshape(3, 3), e[9:]
                n = [g.s("sub", int(x), y) for x, y in zip(f[:3], _cross(g, r, f[3:]))]
                return np.array(_matvec(g, E, n) + _matvec(g, E, f[3:]))
            return self.t.record("force_transform", Force, (self, o), ft)
        return NotImplemented


@dataclass
class _Transposed:
    X: SE3

    def __matmul__(self, o):
        X, g = self.X, self.X.t.g
        if isinstance(o, Force):  # X^T f: a child force expressed at the parent
            def ft(e, f):
                Et, r = e[:9].reshape(3, 3).T, e[9:]
                lin = _matvec(g, Et, f[3:])
                return np.array([g.s("add", x, y) for x, y in zip(_matvec(g, Et, f[:3]), _cross(g, r, lin))]
                                + lin)
            return X.t.record("force_transform_inv", Force, (X, o), ft)
        if isinstance(o, _Congruence):
            raise TypeError("use X.T @ I @ X")
        if isinstance(o, Inertia):
            return _Congruence(X, o)
        return NotImplemented


@dataclass
class _Congruence:
    X: SE3
    I: Inertia

    def __matmul__(self, X2: SE3) -> Inertia:
        if X2 is not self.X and X2.node != self.X.node:
            raise ValueError("X.T @ I @ X needs the same X on both sides")
        X, g = self.X, self.X.t.g

        def congruence(e, I):
            X6 = _x6(g, e)
            IX = [[_dot(g, I[i], X6[:, k]) for k in range(6)] for i in range(6)]
            IX = np.array(IX, np.int64)
            out = np.zeros((6, 6), np.int64)
            for i in range(6):
                for k in range(i, 6):
                    out[i, k] = out[k, i] = _dot(g, X6[:, i], IX[:, k])
            return out
        return X.t.record("inertia_transform_inv", Inertia, (X, self.I), congruence)


@dataclass
class _Inverse:
    X: SE3

    def __matmul__(self, m: Motion) -> Motion:
        X, g = self.X, self.X.t.g

        def mt(e, v):
            Et, r = e[:9].reshape(3, 3).T, e[9:]
            w = _matvec(g, Et, v[:3])
            return np.array(w + [g.s("add", x, y) for x, y in zip(_matvec(g, Et, v[3:]), _cross(g, r, w))])
        return X.t.record("motion_transform_inv", Motion, (X, m), mt)


class _Joints:
    """``robot.joints``: iterating sets the tags recorded on every op in the loop body."""

    def __init__(self, view: "RobotView"):
        self.view = view

    def _walk(self, order, pass_):
        t = self.view.t
        saved = dict(t.tags)
        try:
            for j in order:
                t.tags = dict(saved, joint=j, depth=self.view.depth(j), subtree=self.view.subtree(j),
                              **{"pass": pass_})
                yield j
        finally:
            t.tags = saved

    def __iter__(self):
        return self._walk(range(len(self.view.robot.joints)), "forward")

    def __reversed__(self):
        return self._walk(reversed(range(len(self.view.robot.joints))), "backward")

    def __len__(self):
        return len(self.view.robot.joints)


class RobotView:
    """The robot as an algorithm sees it: constants as traced values, topology as Python."""

    def __init__(self, t: Tracer):
        self.t = t
        self.robot = t.robot
        self.n_q = t.robot.n_q
        self.joints = _Joints(self)
        self.gravity = Motion.const(t, [0, 0, 0, 0, 0, -GRAVITY])

    def parent(self, j: int) -> int:
        return self.robot.joints[j].parent

    def depth(self, j: int) -> int:
        return self.robot.joints[j].depth

    def subtree(self, j: int) -> int:
        while self.robot.joints[j].parent >= 0:
            j = self.robot.joints[j].parent
        return j

    def ancestors(self, j: int) -> list[int]:
        """``j`` and every joint above it, base first."""
        chain = []
        while j >= 0:
            chain.append(j)
            j = self.robot.joints[j].parent
        return chain[::-1]

    def col(self, j: int) -> int:
        """The column of q / qd that drives joint ``j``."""
        return self.robot.joints[j].q_index

    def scale(self, j: int) -> float:
        """Joint ``j``'s mimic multiplier (1 for an ordinary joint)."""
        return self.robot.joints[j].multiplier

    def q(self, j: int, q) -> Scalar:
        """Joint ``j``'s own coordinate from the generalized ``q``."""
        jt = self.robot.joints[j]
        qj = q[jt.q_index]
        return qj * jt.multiplier + jt.offset if jt.mimic else qj

    def S(self, j: int) -> Motion:
        return Motion.const(self.t, self.robot.joints[j].S)

    def I(self, j: int) -> Inertia:
        return Inertia.const(self.t, self.robot.joints[j].I)

    def Xup(self, j: int, qj: Scalar) -> SE3:
        """Joint ``j``'s parent-to-child transform, constants folded, ``qj`` symbolic."""
        jt, g = self.robot.joints[j], self.t.g

        def expand(qe):
            q = int(qe)
            basis = (1, g.s("cos", q), g.s("sin", q)) if jt.kind == "revolute" else (1, q, 0)

            def entry(c):
                acc = 0
                for b, k in zip(basis, c):
                    if k != 0.0:
                        acc = g.s("add", acc, g.s("mul", g.const(k), b))
                return acc
            return np.array([entry(jt.E[:, i, k]) for i in range(3) for k in range(3)]
                            + [entry(jt.r[:, i]) for i in range(3)])
        return self.t.record("joint_transform", SE3, (qj,), expand, key=j)

    def frame(self, name: str):
        return self.robot.frame(name)


@dataclass
class Program:
    """A traced algorithm: its graph, named inputs (name, length) and flat scalar outputs."""

    name: str
    robot: Robot
    g: Graph
    inputs: list[tuple[str, int]]
    outputs: list[tuple[str, np.ndarray]]  # name -> scalar ids, flattened

    @property
    def flat_outputs(self) -> np.ndarray:
        return np.concatenate([o for _, o in self.outputs]) if self.outputs else np.zeros(0, np.int64)

    def evaluate(self, **inputs) -> dict[str, np.ndarray]:
        """Float64 interpreter on (B, n) inputs; returns name -> (B, len) outputs."""
        flat = self.g.evaluate(inputs, self.flat_outputs)
        out, k = {}, 0
        for name, ids in self.outputs:
            out[name], k = flat[:, k:k + len(ids)], k + len(ids)
        return out


@dataclass
class Algorithm:
    name: str
    fn: object
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def trace(self, robot: Robot, **options) -> Program:
        t = Tracer(robot)
        view = RobotView(t)
        args = [[Scalar(t, t.g.node("input", "Scalar", (), lambda i=i, nm=nm: t.g.input(nm, i), {}, key=(nm, i)))
                 for i in range(robot.n_q)] for nm in self.inputs]
        result = self.fn(view, *args, **options)
        result = result if isinstance(result, tuple) else (result,)
        outputs = []
        for name, value in zip(self.outputs, result):
            outputs.append((name, _flatten(value)))
        return Program(self.name, robot, t.g, [(nm, robot.n_q) for nm in self.inputs], outputs)


def _flatten(value) -> np.ndarray:
    if isinstance(value, Value):
        return value.elems.reshape(-1).astype(np.int64)
    if isinstance(value, (list, tuple)):
        parts = [_flatten(v) for v in value]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)
    if isinstance(value, np.ndarray):
        return value.astype(np.int64).reshape(-1)
    raise TypeError(f"cannot output {type(value).__name__}")


def algorithm(inputs: tuple[str, ...], outputs: tuple[str, ...]):
    """Register ``fn(robot, *inputs, **options)`` as a traceable algorithm."""
    def wrap(fn):
        ALGORITHMS[fn.__name__] = Algorithm(fn.__name__, fn, inputs, outputs)
        return functools.wraps(fn)(fn)
    return wrap
