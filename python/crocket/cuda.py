"""CUDA emission: one header per robot, plain CUDA with the robot's constants as literals.

Every function is templated on ``T`` (float or double) and takes caller-owned pointers: no
allocation, no globals. Outputs are written at ``y[i * ys]`` so structure-of-arrays layouts
coalesce (GRiM's ``ys`` convention). Per algorithm ``<alg>``:

``<alg>_thread``  one problem per thread, straight-line (cricket's tier).
``<alg>_split``   one problem per lane, 32 per block, the block's warps splitting the stages
                  (``cricket.tiered``'s warp mode); scratch ``<alg>_split_smem`` per block.
``<alg>_warp``    one problem per warp, its lanes splitting each stage; ``<alg>_warp_smem``
                  per warp.
``<alg>_block``   one problem per block of ``<alg>_block_threads``; ``<alg>_block_smem``.

A unit over ``budget`` statements is emitted as ``__noinline__`` stage functions: large
straight-line functions are where ptxas has silently miscompiled cricket traces at -O2/-O3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import schedule as sched_mod
from .robot import Robot
from .trace import ALGORITHMS, Program

TIERS = ("thread", "split", "warp", "block", "uniform")  # PROTOTYPE(uniform): last entry
BUDGET = 4000

_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_CALL = {"abs": "fabs", "max": "fmax", "min": "fmin"}


def literal(v: float) -> str:
    return f"T({v!r})"


def expr(g, s: int, ref) -> str:
    op, args = g.op[s], [ref(a) for a in g.args[s]]
    if op in _INFIX:
        return f"{args[0]} {_INFIX[op]} {args[1]}"
    if op == "neg":
        return f"-{args[0]}"
    if op == "lt":
        return f"({args[0]} < {args[1]} ? T(1) : T(0))"
    return f"{_CALL.get(op, op)}({', '.join(args)})"


@dataclass
class Variant:
    """How one algorithm is emitted at one tier."""

    tier: str
    units: int = 1  # lanes per problem (warp: 32, block: threads), warps per block for split
    threshold: int = 8  # scalarisation threshold, see :mod:`crocket.schedule`
    narrow: int = 0  # stages of at most this many items keep tasks whole on one lane


class _Emitter:
    def __init__(self, program: Program, budget: int):
        self.p = program
        self.g = program.g
        self.budget = budget
        names = [n for n, _ in program.outputs]
        self.where: list[tuple[str, int]] = []
        for name, ids in program.outputs:
            self.where += [(name, i) for i in range(len(ids))]
        self.params = ", ".join([f"const T* __restrict__ {n}" for n, _ in program.inputs]
                                + [f"T* __restrict__ {n}" for n in names])
        self.args = ", ".join([n for n, _ in program.inputs] + names)

    def ref_base(self, a: int):
        if self.g.op[a] == "const":
            return literal(self.g.value[a])
        if self.g.op[a] == "in":
            name, i = self.g.value[a]
            return f"{name}[{i}]"
        return None

    def store_outputs(self, s: int, value: str, out_index) -> list[str]:
        return [f"{self.where[k][0]}[{self.where[k][1]} * ys] = {value};" for k in out_index.get(s, [])]

    # ---- thread tier ----------------------------------------------------------------------

    def thread(self) -> str:
        g, outputs = self.g, self.p.flat_outputs
        live = g.live(outputs)
        body = [s for s in range(len(g.op)) if live[s] and g.op[s] not in ("const", "in")]
        if len(body) > self.budget:
            return self.thread_chunked(body)
        ref = lambda a: self.ref_base(a) or f"t{a}"  # noqa: E731
        lines = [f"        const T t{s} = {expr(g, s, ref)};" for s in body]
        for k, s in enumerate(outputs):
            name, i = self.where[k]
            lines.append(f"        {name}[{i} * ys] = {ref(int(s))};")
        return (f"    template <class T>\n    CROCKET_HD void {self.p.name}_thread({self.params}, int ys = 1)\n"
                "    {\n" + "\n".join(lines) + "\n    }\n")

    def thread_chunked(self, body: list[int]) -> str:
        """The thread tier over budget: the statements in trace order, cut every ``budget`` into
        ``__noinline__`` chunks. A value read by a later chunk (or by none, as an output) passes
        through the scratch ``s``; everything else stays a register of its own chunk, so each
        chunk's live set is bounded by the budget, not by the size of a tree level. This is the
        register guard's lever (``harness.fit``): a smaller budget gives smaller live sets."""
        g, name = self.g, self.p.name
        chunks = [body[i:i + self.budget] for i in range(0, len(body), self.budget)]
        chunk_of = {s: k for k, ch in enumerate(chunks) for s in ch}
        slots: dict[int, int] = {}
        for k, ch in enumerate(chunks):
            for s in ch:
                for a in g.args[s]:
                    if a in chunk_of and chunk_of[a] != k and a not in slots:
                        slots[a] = len(slots)
        out_index: dict[int, list[int]] = {}
        for k, s in enumerate(self.p.flat_outputs):
            out_index.setdefault(int(s), []).append(k)
        fn, sig = f"{name}_thread", f"{self.params}, T* __restrict__ s, int ys"
        funcs, calls = [], []
        for k, ch in enumerate(chunks):
            mine = set(ch)
            ref = lambda a, mine=mine: (self.ref_base(a)  # noqa: E731
                                        or (f"t{a}" if a in mine else f"s[{slots[a]}]"))
            lines = [f"const T t{s} = {expr(g, s, ref)};" for s in ch]
            for s in ch:
                if s in slots:
                    lines.append(f"s[{slots[s]}] = t{s};")
                lines += self.store_outputs(s, f"t{s}", out_index)
            text = "\n        ".join(lines)
            funcs.append(f"    template <class T>\n    __device__ __noinline__ void {fn}_c{k}({sig})\n"
                         f"    {{\n        {text}\n    }}\n")
            calls.append(f"        {fn}_c{k}<T>({self.args}, s, ys);\n")
        for k, s in enumerate(self.p.flat_outputs):  # outputs that are inputs or constants
            if int(s) not in chunk_of:
                nm, i = self.where[k]
                calls.append(f"        {nm}[{i} * ys] = {self.ref_base(int(s))};\n")
        head = (f"    template <class T>\n    CROCKET_HD void {fn}({self.params}, int ys = 1)\n    {{\n"
                f"        T s[{max(len(slots), 1)}];\n")
        return "".join(funcs) + head + "".join(calls) + "    }\n"

    # ---- cooperative tiers ----------------------------------------------------------------

    def cooperative(self, sc: sched_mod.Schedule, mode: str) -> str:
        # REDESIGN(ranks): "warp" and "block" switch on the LANE, so each lane runs its own case
        # and SIMT serialises them (RNEA panda B=1: thread 8.3 us, warp 87 us). Only "split",
        # which switches on the warp, is a sound rank model. Replace warp/block with warps as
        # ranks (lanes = problems, or later lanes = a task's elements); GRiM's IK kernels
        # (ik_crocket/ck.cuh, region_ik/brownian_ik.cu) call _warp/_block and move with it.
        g, name = self.g, self.p.name
        rank, slot = {
            "warp": ("threadIdx.x & 31", "s[{k}]"),
            "block": ("threadIdx.x", "s[{k}]"),
            "split": ("threadIdx.x >> 5", "s[{k} * 32 + (threadIdx.x & 31)]"),
            "thread": ("0", "s[{k}]"),
        }[mode]
        barrier = {"warp": "__syncwarp();", "block": "__syncthreads();", "split": "__syncthreads();",
                   "thread": None}[mode]

        stages, total = [], 0
        for st in range(1, sc.stages + 1):
            by_rank: dict[int, list] = {}
            for it in sc.stage_items(st):
                by_rank.setdefault(it.rank, []).append(it)
            cases = []
            for r in sorted(by_rank):
                seen: set[int] = set()
                for it in by_rank[r]:
                    if it.task >= 0:
                        sched_mod.walk(g, it.scalar, it.task, sc.task_of, seen, set())
                ref = lambda a: (self.ref_base(a)  # noqa: E731
                                 or (slot.format(k=sc.slots[a]) if a in sc.task_of and a not in seen
                                     else f"t{a}"))
                lines = [f"const T t{s} = {expr(g, s, ref)};" for s in sorted(seen)]
                for it in by_rank[r]:
                    v = ref(it.scalar) if it.task < 0 else f"t{it.scalar}"
                    if it.scalar in sc.slots:
                        lines.append(f"{slot.format(k=sc.slots[it.scalar])} = {v};")
                    lines += self.store_outputs(it.scalar, v, sc.out_index)
                total += len(lines)
                body = "\n            ".join(lines)
                cases.append(f"        case {r}:\n        {{\n            {body}\n        }}\n        break;")
            stages.append((st, cases))

        split = total > self.budget
        fn = f"{name}_{mode}"
        sig = f"{self.params}, T* __restrict__ s, int ys"
        out, calls = [], []
        for st, cases in stages:
            sync = f"        {barrier}\n" if barrier and st in sc.sync_before else ""
            sw = (f"        switch ({rank})\n        {{\n" + "\n".join(cases)
                  + "\n        default:\n            break;\n        }\n")
            if split:
                out.append(f"    template <class T>\n    __device__ __noinline__ void {fn}_s{st}({sig})\n"
                           f"    {{\n{sw}    }}\n")
                calls.append(sync + f"        {fn}_s{st}<T>({self.args}, s, ys);\n")
            else:
                calls.append(sync + sw)
        if mode == "thread":
            decl = f"        T s[{max(sc.n_slots, 1)}];\n"
            head = f"    template <class T>\n    CROCKET_HD void {fn}({self.params}, int ys = 1)\n    {{\n{decl}"
        else:
            head = f"    template <class T>\n    __device__ __forceinline__ void {fn}({sig} = 1)\n    {{\n"
        return "".join(out) + head + "".join(calls) + "    }\n"


    # ---- uniform tier (PROTOTYPE) ---------------------------------------------------------
    # PROTOTYPE(uniform): an experiment, not a finished tier. Every shortcut is tagged
    # PROTOTYPE(uniform) and listed in OPEN_TASKS.md ("Lane-uniform tier"); grep the tag to clean up.

    def uniform(self, sc: sched_mod.Schedule) -> tuple[str, int]:
        """One problem per warp, like ``warp``, but lanes run the SAME code: each stage's items are
        grouped by expression shape and each group is one straight-line statement over lanes
        ``i``, operands at ``s[base + stride * i]`` where the layout makes them affine and through
        a per-item index table where it does not. ``warp`` gives every lane its own ``case``, which
        SIMT runs one lane after another. Returns (code, scratch slots)."""
        g, name = self.g, self.p.name
        # Layout: the uniform tier allocates its own slots instead of the schedule's interval
        # colouring, which scatters a value's elements. Inputs first, then each group's results
        # in one contiguous block in member order, so operand columns come out affine in the
        # lane (``s[base + stride * i]``) and need no table. A block is freed (first-fit, merged
        # with free neighbours) once the last stage reading any of its members has passed; the
        # barrier before every stage makes that safe.
        slots: dict[int, int] = {}
        n = 0
        # PROTOTYPE(uniform): inputs are staged into scratch so every operand is one kind of load;
        # a real tier would read inputs in place.
        in_slot: dict[int, int] = {}
        for s in range(len(g.op)):
            if g.op[s] == "in":
                in_slot[s], n = n, n + 1

        def tree(s: int, task: int):
            op = g.op[s]
            if op == "const":
                return ("c", g.value[s])
            if op == "in" or sc.task_of.get(s, task) != task:
                return ("v", s)  # a scalar id; its slot is looked up at emission
            # PROTOTYPE(uniform): expands the item's DAG as a tree, so a value shared inside one
            # item is recomputed (same value, so still bit-identical) and deep sharing could
            # blow up. Emit shared nodes once per group instead.
            kids = [tree(a, task) for a in g.args[s]]
            if op in _COMMUTATIVE:  # IEEE add/mul/max/min are commutative: reordering is bit-safe
                kids.sort(key=lambda t: repr(_shape(t)))
            return (op, kids)

        # Every per-lane table is packed into one int array (operand / result slots) and one
        # constant array, copied into scratch at entry: a group's index load is then a shared
        # load, not a dependent global load (global tables measured ~9x slower than thread).
        # PROTOTYPE(uniform): the copy costs scratch and a prologue per problem; encode indices
        # in the instruction stream (or compute them from lane) in a real design.
        itab: list[int] = []
        ctab: list[float] = []

        def ints(vals) -> int:
            itab.extend(int(v) for v in vals)
            return len(itab) - len(vals)

        def consts(vals) -> int:
            ctab.extend(float(v) for v in vals)
            return len(ctab) - len(vals)

        body = []
        for nm, length in self.p.inputs:
            if not length:
                continue
            ids = [in_slot[g._memo[("in", nm, i)]] for i in range(length)]
            assert ids == list(range(ids[0], ids[0] + length))  # inputs were given consecutive slots
            body.append(f"        for (int i = lane; i < {length}; i += 32) s[{ids[0]} + i] = {nm}[i];")
        # Pass 1: each stage's groups and, per scalar, the last stage that reads it.
        stage_groups: list[list] = []
        last_read: dict[int, float] = {int(x): math.inf for x in self.p.flat_outputs}
        for st in range(1, sc.stages + 1):
            groups: dict = {}
            for it in sc.stage_items(st):
                if it.task >= 0:
                    t = tree(it.scalar, it.task)
                    groups.setdefault(repr(_shape(t)), []).append((it.scalar, t))
                    for kind, x in _leaves(t):
                        if kind == "v":
                            last_read[x] = max(last_read.get(x, 0), st)
            stage_groups.append(list(groups.values()))

        def where(x: int) -> int:
            return in_slot[x] if x in in_slot else slots[x]

        # Pass 2: allocate blocks (first fit over freed ranges) and emit.
        free: list[list[int]] = []  # [start, length], sorted, coalesced
        live: list[tuple[int, int, list[int]]] = []  # (base, count, member scalars)

        def alloc(count: int) -> int:
            nonlocal n
            for f in free:
                if f[1] >= count:
                    base = f[0]
                    f[0] += count
                    f[1] -= count
                    if not f[1]:
                        free.remove(f)
                    return base
            base, n = n, n + count
            return base

        def release(base: int, count: int):
            free.append([base, count])
            free.sort()
            merged = [free[0]]
            for f in free[1:]:
                if merged[-1][0] + merged[-1][1] == f[0]:
                    merged[-1][1] += f[1]
                else:
                    merged.append(f)
            free[:] = merged

        for st, groups in enumerate(stage_groups, 1):
            # PROTOTYPE(uniform): a barrier before every stage. Lanes are reassigned per group, so
            # sync_before (which assumes the LPT ranks) does not apply; derive the real need.
            body.append("        __syncwarp();")
            for blk in [b_ for b_ in live if all(last_read.get(x, 0) < st for x in b_[2])]:
                live.remove(blk)
                release(blk[0], blk[1])
            for members in groups:
                members = _order(members, where)
                count = len(members)
                base = alloc(count)
                live.append((base, count, [x for x, _ in members]))
                for i, (x, _) in enumerate(members):
                    slots[x] = base + i
                leaves = [_leaves(t) for _, t in members]
                names = []
                for j in range(len(leaves[0])):
                    kind = leaves[0][j][0]
                    vals = [lv[j][1] if kind == "c" else where(lv[j][1]) for lv in leaves]
                    stride = _affine(vals)
                    if kind == "c" and stride == 0:
                        names.append(literal(vals[0]))  # same constant in every lane: an immediate
                    elif kind == "c":  # converted to T once in the prologue, as T(literal) would be
                        names.append(f"cv[{consts(vals)} + i]")
                    elif stride is not None:
                        names.append(f"s[{_lin(vals[0], stride)}]")
                    else:
                        names.append(f"s[it[{ints(vals)} + i]]")
                # Straight-line, no loop or branch: a stage is one basic block, so ptxas can
                # interleave its independent groups and hide one group's latency behind another's.
                # Lanes past the group recompute its last member and write the same value to the
                # same slot (the split tier's trick for a partial batch), so the result is unchanged.
                for c in range(0, count, 32):
                    body.append(f"        {{ const int i = min(lane + {c}, {count - 1}); "
                                f"s[{_lin(base, 1)}] = {_render(members[0][1], iter(names))}; }}")
        body.append("        __syncwarp();")
        for nm, ids in self.p.outputs:
            src = []
            for i, x in enumerate(int(x) for x in ids):
                if g.op[x] == "const":
                    body.append(f"        if (lane == 0) {nm}[{i} * ys] = {literal(g.value[x])};")
                    src.append(-1)
                else:
                    src.append(where(x))
            off = ints(src)
            body.append(f"        for (int i = lane; i < {len(ids)}; i += 32) "
                        f"if (it[{off} + i] >= 0) {nm}[i * ys] = s[it[{off} + i]];")
        m, k = len(ctab), len(itab)
        head = [
            "        const int lane = threadIdx.x & 31;",
            f"        T* cv = s + {n};",
            f"        int* it = reinterpret_cast<int*>(cv + {m});  // {k} ints, reserved as {k} T",
            f"        for (int i = lane; i < {m}; i += 32) cv[i] = T({name}_u_ctab[i]);",
            f"        for (int i = lane; i < {k}; i += 32) it[i] = {name}_u_itab[i];",
        ]
        tables = _table("int", f"{name}_u_itab", itab) + _table("double", f"{name}_u_ctab", ctab)
        # PROTOTYPE(uniform): no __noinline__ stage split over the statement budget (the ptxas
        # guard the other tiers have); validate every build numerically until it is added.
        fn = (f"    template <class T>\n    __device__ __forceinline__ void {name}_uniform("
              f"{self.params}, T* __restrict__ s, int ys = 1)\n    {{\n" + "\n".join(head + body) + "\n    }\n")
        return tables + fn, n + m + k


_COMMUTATIVE = {"add", "mul", "max", "min"}


def _affine(vals) -> int | None:
    """The stride if ``vals[i] == vals[0] + stride * i``, else None (stride 0: every lane alike)."""
    if len(vals) < 2:
        return 0
    d = vals[1] - vals[0]
    return d if all(vals[i] - vals[i - 1] == d for i in range(2, len(vals))) else None


def _lin(base: int, stride: int) -> str:
    return str(base) if stride == 0 else f"{base} + i" if stride == 1 else f"{base} + {stride} * i"


def _order(members, where):
    """Members of one group in the order that makes the most operand columns affine: try sorting
    by each operand column as the key and keep the best (ties: the original order).
    PROTOTYPE(uniform): greedy per group; producers' orders are already fixed, so a global
    layout (choose producer orders for their consumers) could do better."""
    cols = list(zip(*[[where(lv[1]) for lv in _leaves(t) if lv[0] == "v"] for _, t in members]))

    def score(order):
        return sum(_affine([c[k] for k in order]) is not None for c in cols)
    best = list(range(len(members)))
    best_score = score(best)
    for c in cols:
        order = sorted(range(len(members)), key=lambda k: (c[k], k))
        sc_ = score(order)
        if sc_ > best_score:
            best, best_score = order, sc_
    return [members[k] for k in best]


def _shape(t):
    """An item tree with its operands abstracted: items of one shape run as one lane loop."""
    return t[0] if t[0] in ("c", "v") else (t[0], tuple(_shape(k) for k in t[1]))


def _leaves(t) -> list:
    return [t] if t[0] in ("c", "v") else [lv for k in t[1] for lv in _leaves(k)]


def _render(t, names) -> str:
    """C for a shape tree; operands are taken from ``names`` in leaf order."""
    if t[0] in ("c", "v"):
        return next(names)
    op, args = t[0], [_render(k, names) for k in t[1]]
    if op in _INFIX:
        return f"({args[0]} {_INFIX[op]} {args[1]})"
    if op == "neg":
        return f"(-{args[0]})"
    if op == "lt":
        return f"({args[0]} < {args[1]} ? T(1) : T(0))"
    return f"{_CALL.get(op, op)}({', '.join(args)})"


def _table(ctype: str, name: str, vals) -> str:
    # PROTOTYPE(uniform): tables are `static __device__` globals in the header, so each including
    # translation unit gets its own copy; move them into a per-robot .cu.
    body = ", ".join(repr(float(v)) if ctype == "double" else str(int(v)) for v in vals) or "0"
    return f"    static __device__ const {ctype} {name}[{max(len(vals), 1)}] = {{{body}}};\n"


def smem(sc: sched_mod.Schedule, tier: str) -> int:
    """Scratch, in elements of T: per warp (warp), per block (block, split)."""
    return max(sc.n_slots, 1) * (32 if tier == "split" else 1)


def emit(program: Program, variants: list[Variant], budget: int = BUDGET) -> str:
    """One algorithm at each tier in ``variants``, with its scratch and group-size constants."""
    em = _Emitter(program, budget)
    out, consts = [], []
    for v in variants:
        if v.tier == "thread":
            out.append(em.thread())
            continue
        sc = sched_mod.build(program, v.units, v.threshold, v.narrow)
        if v.tier == "uniform":  # PROTOTYPE(uniform)
            code, n_slots = em.uniform(sc)
            out.append(code)
            consts.append(f"    constexpr int {program.name}_uniform_smem = {max(n_slots, 1)};")
            consts.append(f"    constexpr int {program.name}_uniform_stages = {sc.stages};")
            continue
        out.append(em.cooperative(sc, v.tier))
        consts.append(f"    constexpr int {program.name}_{v.tier}_smem = {smem(sc, v.tier)};")
        if v.tier == "block":
            consts.append(f"    constexpr int {program.name}_block_threads = {v.units};")
        if v.tier == "split":
            consts.append(f"    constexpr int {program.name}_split_warps = {v.units};")
        consts.append(f"    constexpr int {program.name}_{v.tier}_stages = {sc.stages};")
    sizes = [f"    constexpr int {program.name}_{n}_size = {len(ids)};" for n, ids in program.outputs]
    return "\n".join(out) + "\n" + "\n".join(consts + sizes) + "\n"


def header(robot: Robot, sections: list[str], name: str | None = None) -> str:
    ns = (name or robot.name).lower().replace("-", "_")
    joints = '", "'.join(robot.joint_names)
    return f"""#pragma once

// Generated by crocket for robot "{robot.name}". Do not edit.
//
// Robot-specialized rigid-body kernels for CUDA, templated on T (float or double). Inputs are
// one problem's q / qd / qdd / tau (n_q each, ordered as `joint_names`); output element i is
// written at y[i * ys]. Gravity is -9.81 along world z.
//
// Define CROCKET_HD before including to override the thread-tier qualifiers.

#ifndef __CUDACC_RTC__  // NVRTC provides the math functions as builtins
#include <math.h>
#endif

#ifndef CROCKET_HD
#ifdef __CUDACC__
#define CROCKET_HD __host__ __device__ __forceinline__
#else
#define CROCKET_HD inline
#endif
#endif

// NOLINTBEGIN(*-magic-numbers)
namespace crocket::robots::{ns}
{{
    constexpr int n_q = {robot.n_q};
    constexpr const char *joint_names[n_q] = {{"{joints}"}};

{"".join(sections)}}}  // namespace crocket::robots::{ns}
// NOLINTEND(*-magic-numbers)
"""


DEFAULT_VARIANTS = (Variant("thread"), Variant("split", 8), Variant("warp", 32), Variant("block", 128))


def kinematics(robot: Robot, frames, end_effector: str | None, budget: int = BUDGET) -> str:
    """cricket's kinematics interface in float, for drop-in use: ``frame_poses(q, poses)`` (7 per
    frame), ``ee_pose(q, pose)`` and ``ee_pose_jacobian(q, pose, jacobian)``, poses as
    ``[qw, qx, qy, qz, tx, ty, tz]``, the Jacobian row-major ``6 x n_q`` LOCAL_WORLD_ALIGNED."""
    frames = list(frames)
    names = '", "'.join(frames)
    out = [f"    constexpr int n_frames = {len(frames)};\n",
           f'    constexpr const char *frame_names[n_frames > 0 ? n_frames : 1] = {{"{names}"}};\n',
           _PACK_POSE]
    poses = ALGORITHMS["frame_poses"].trace(robot, frames=tuple(frames))
    out.append(_Emitter(poses, budget).thread())
    out.append("""
    // World pose of every frame in `frame_names`, 7 floats each, in that order.
    CROCKET_HD void frame_poses(const float *q, float *poses)
    {
        float y[12 * (n_frames > 0 ? n_frames : 1)];
        frame_poses_thread<float>(q, y);
        for (int i = 0; i < n_frames; ++i)
        {
            detail::pack_pose(y + 12 * i, poses + 7 * i);
        }
    }
""")
    if end_effector is not None:
        jac = ALGORITHMS["frame_jacobian"].trace(robot, frame=end_effector)
        out.append(f'    constexpr const char *end_effector = "{end_effector}";\n')
        out.append(_Emitter(jac, budget).thread())
        out.append("""
    CROCKET_HD void ee_pose_jacobian(const float *q, float *pose, float *jacobian)
    {
        float y[12];
        frame_jacobian_thread<float>(q, y, jacobian);
        detail::pack_pose(y, pose);
    }

    CROCKET_HD void ee_pose(const float *q, float *pose)
    {
        float y[12], jacobian[6 * n_q];
        frame_jacobian_thread<float>(q, y, jacobian);  // the Jacobian's terms are dead and folded away
        detail::pack_pose(y, pose);
    }
""")
    return "".join(out)


_PACK_POSE = """    namespace detail
    {
        // y = [t (3), R column-major (9)] -> pose = [qw, qx, qy, qz, tx, ty, tz]
        CROCKET_HD void pack_pose(const float *y, float *pose)
        {
            const float r00 = y[3], r10 = y[4], r20 = y[5];
            const float r01 = y[6], r11 = y[7], r21 = y[8];
            const float r02 = y[9], r12 = y[10], r22 = y[11];

            const float trace = r00 + r11 + r22;
            if (trace > 0.f)
            {
                const float s = 0.5f / sqrtf(trace + 1.f);
                pose[0] = 0.25f / s;
                pose[1] = (r21 - r12) * s;
                pose[2] = (r02 - r20) * s;
                pose[3] = (r10 - r01) * s;
            }
            else if (r00 > r11 and r00 > r22)
            {
                const float s = 2.f * sqrtf(1.f + r00 - r11 - r22);
                pose[0] = (r21 - r12) / s;
                pose[1] = 0.25f * s;
                pose[2] = (r01 + r10) / s;
                pose[3] = (r02 + r20) / s;
            }
            else if (r11 > r22)
            {
                const float s = 2.f * sqrtf(1.f + r11 - r00 - r22);
                pose[0] = (r02 - r20) / s;
                pose[1] = (r01 + r10) / s;
                pose[2] = 0.25f * s;
                pose[3] = (r12 + r21) / s;
            }
            else
            {
                const float s = 2.f * sqrtf(1.f + r22 - r00 - r11);
                pose[0] = (r10 - r01) / s;
                pose[1] = (r02 + r20) / s;
                pose[2] = (r12 + r21) / s;
                pose[3] = 0.25f * s;
            }

            pose[4] = y[0];
            pose[5] = y[1];
            pose[6] = y[2];
        }
    }  // namespace detail

"""
