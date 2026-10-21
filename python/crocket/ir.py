"""The IR: one SSA graph at two levels.

Spatial level: :class:`Node`, one GLASS-named op on a typed value (``Motion``, ``Force``,
``Inertia``, ``SE3``, ``Vec3``, ``Scalar``) with its tree tags. Scheduling reads this level.

Scalar level: every node carries its *expansion*, the scalar ids of its elements, built in a
hash-consed scalar graph that folds as it goes. Constants are literals, ``x * 0`` and
``x + 0`` vanish and identical expressions are one id, so sparsity propagates through every
spatial op and the same ``Xup(j, q[j])`` recorded twice is one value (CSE at both levels).
Folding and thread-tier emission read this level.

Scalar ids 0 and 1 are the constants 0 and 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_FOLD = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "neg": lambda a: -a,
    "sin": math.sin,
    "cos": math.cos,
    "sqrt": math.sqrt,
    "atan2": np.arctan2,
    "abs": abs,
    "max": np.maximum,
    "min": np.minimum,
}
_NUMPY = {"sin": np.sin, "cos": np.cos, "sqrt": np.sqrt}


@dataclass
class Node:
    op: str
    type: str
    args: tuple[int, ...]
    elems: np.ndarray  # scalar ids, shaped like the value
    tags: dict = field(default_factory=dict)  # joint, depth, subtree, pass, line
    created: list[int] = field(default_factory=list)  # scalars this node introduced

    @property
    def cost(self) -> int:
        """FLOPs after folding: the scalar ops this node adds to the graph."""
        return len(self.created)


class Graph:
    def __init__(self):
        self.op: list[str] = []
        self.args: list[tuple[int, ...]] = []
        self.value: list = []  # float for "const", (input, index) for "in"
        self.owner: list[int] = []  # node that created each scalar, -1 for constants/inputs
        self.nodes: list[Node] = []
        self._memo: dict = {}
        self._node_memo: dict = {}
        self._current = -1
        self.const(0.0)
        self.const(1.0)

    # ---- scalar level -------------------------------------------------------------------

    def _add(self, key, op, args, value) -> int:
        sid = self._memo.get(key)
        if sid is None:
            sid = self._memo[key] = len(self.op)
            self.op.append(op)
            self.args.append(args)
            self.value.append(value)
            self.owner.append(self._current if op not in ("const", "in") else -1)
            if self._current >= 0 and op not in ("const", "in"):
                self.nodes[self._current].created.append(sid)
        return sid

    def const(self, v: float) -> int:
        v = float(v) + 0.0  # -0.0 -> 0.0
        return self._add(("const", v), "const", (), v)

    def input(self, name: str, index: int) -> int:
        return self._add(("in", name, index), "in", (), (name, index))

    def is_const(self, s: int) -> bool:
        return self.op[s] == "const"

    def s(self, op: str, *args: int) -> int:
        """The scalar ``op(*args)``, folded."""
        if all(self.is_const(a) for a in args):
            return self.const(_FOLD[op](*(self.value[a] for a in args)))
        if op == "add":
            a, b = args
            if a == 0:
                return b
            if b == 0:
                return a
            args = tuple(sorted(args))
        elif op == "sub":
            a, b = args
            if b == 0:
                return a
            if a == 0:
                return self.s("neg", b)
            if a == b:
                return 0
        elif op == "mul":
            a, b = args
            if a == 0 or b == 0:
                return 0
            if a == 1:
                return b
            if b == 1:
                return a
            for x, y in ((a, b), (b, a)):
                if self.is_const(x) and self.value[x] == -1.0:
                    return self.s("neg", y)
            args = tuple(sorted(args))
        elif op == "div":
            a, b = args
            if a == 0:
                return 0
            if b == 1:
                return a
        elif op == "neg":
            (a,) = args
            if self.op[a] == "neg":
                return self.args[a][0]
        elif op == "abs":
            (a,) = args
            if self.op[a] in ("neg", "abs"):
                return self.s("abs", self.args[a][0])
        elif op in ("max", "min"):
            a, b = args
            if a == b:
                return a
            args = tuple(sorted(args))
        return self._add((op, args), op, args, None)

    # ---- spatial level -------------------------------------------------------------------

    def node(self, op: str, type_: str, args: tuple[int, ...], expand, tags: dict, key=None) -> int:
        """Record ``op`` on spatial ``args``; ``expand(*arg_elems)`` builds its scalar expansion.

        Nodes are hash-consed on ``(op, args, key)``: recording the same op twice is one node.
        """
        memo = (op, args, key)
        if memo in self._node_memo:
            return self._node_memo[memo]
        nid = len(self.nodes)
        self.nodes.append(Node(op, type_, args, np.zeros(0, np.int64), dict(tags)))
        prev, self._current = self._current, nid
        try:
            elems = np.asarray(expand(*(self.nodes[a].elems for a in args)), np.int64)
        finally:
            self._current = prev
        self.nodes[nid].elems = elems
        self._node_memo[memo] = nid
        return nid

    # ---- passes ---------------------------------------------------------------------------

    def live(self, outputs) -> np.ndarray:
        """Dead-output elimination: mask of the scalars ``outputs`` depend on."""
        mask = np.zeros(len(self.op), bool)
        stack = [int(s) for s in outputs]
        while stack:
            s = stack.pop()
            if not mask[s]:
                mask[s] = True
                stack.extend(self.args[s])
        return mask

    def evaluate(self, inputs: dict[str, np.ndarray], outputs) -> np.ndarray:
        """Float64 interpreter: ``inputs`` maps name -> (B, n); returns (B, len(outputs))."""
        mask = self.live(outputs)
        batch = next(iter(inputs.values())).shape[0] if inputs else 1
        vals: dict[int, np.ndarray] = {}
        for s in np.flatnonzero(mask):
            op, a = self.op[s], self.args[s]
            if op == "const":
                vals[s] = np.full(batch, self.value[s])
            elif op == "in":
                name, i = self.value[s]
                vals[s] = np.asarray(inputs[name], np.float64)[:, i]
            elif op in _NUMPY:
                vals[s] = _NUMPY[op](vals[a[0]])
            else:
                vals[s] = _FOLD[op](*(vals[x] for x in a))
        return np.stack([vals[int(s)] for s in outputs], axis=1) if len(outputs) else np.zeros((batch, 0))
