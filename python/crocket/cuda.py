"""CUDA emission: one header per robot, plain CUDA with the robot's constants as literals.

Every function is templated on ``T`` (float or double) and takes caller-owned pointers: no
allocation, no globals. Outputs are written at ``y[i * ys]`` so structure-of-arrays layouts
coalesce (GRiM's ``ys`` convention). Per algorithm ``<alg>``:

``<alg>_thread``  one problem per thread, straight-line (cricket's tier).
``<alg>_warp``    one warp (one rank) holding ``<alg>_warp_problems`` problems (P), 32 / P lanes
                  each; scratch ``<alg>_warp_smem`` per warp, several warps per block side by side.
``<alg>_block``   ``<alg>_block_threads / 32`` warps (the ranks) holding P problems;
                  ``<alg>_block_smem`` per block.

P (``Variant.problems``) splits each warp's lanes between problems and a problem's elements:
P = 1 is GLASS's contract (one problem per warp / block, lanes over its elements), P = 32 puts
one problem on each lane with the warps sharing each problem's stages (``cricket.tiered``'s
warp mode, formerly the ``split`` tier). Every thread passes its own problem's pointers. See
docs/tier_layouts.html.

The warp / block functions take ``(..., T* s, const T* tables, int ys)``: ``s`` is the scratch
(``<alg>_<tier>_smem`` elements per warp / per block), ``tables`` the robot's lookup tables
(``<alg>_<tier>_tables<T>()`` elements, once per block, shared by every warp: constants in T,
indices in 16 bits), filled once per launch
by ``<alg>_<tier>_load_tables(tables, rank, size)`` from every thread of the calling group,
which then syncs (``__syncwarp`` / ``__syncthreads``). Several warps loading the same tables
write identical values, so each warp may load them itself after an early return of others.

A unit over ``budget`` statements is emitted as ``__noinline__`` stage functions: large
straight-line functions are where ptxas has silently miscompiled cricket traces at -O2/-O3.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from . import schedule as sched_mod
from .robot import Robot
from .trace import ALGORITHMS, Program

TIERS = ("thread", "warp", "block")
WARPS = (1, 2, 4, 8)  # k: warps (ranks) per block; the warp tier is k = 1
PROBLEMS = (1, 2, 4, 8, 16, 32)  # P: problems per warp, 32 / P lanes each
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
    units: int = 1  # block: threads (a multiple of 32); warp: ignored (one warp)
    threshold: int = 8  # scalarisation threshold, see :mod:`crocket.schedule`
    placement: str = "element"  # what a rank is given: "element" or "task" (schedule.build step 3)
    problems: int = 1  # problems per warp (P): 1, 2, 4, 8, 16 or 32; lanes per problem 32 / P
    sync: str = "block"  # "block": __syncthreads; "named": only the ranks that exchange; "dataflow":
    #                      no barriers, ranks wait on producers' progress counters (split only)


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

    def cooperative(self, sc: sched_mod.Schedule, mode: str, fn: str | None = None,
                    tables: bool = False) -> str:
        """``mode="split"``: the warp / block tiers at P = 32, one problem per lane, the warps the
        ranks (each its own ``case`` per stage), named ``fn``. With one warp the rank is always 0,
        so packed warps each run their own 32 problems. ``mode="thread"`` is the same schedule run
        by one thread."""
        g, name = self.g, self.p.name
        rank, slot = {
            "split": ("threadIdx.x >> 5" if sc.units > 1 else "0", "s[{k} * 32 + (threadIdx.x & 31)]"),
            "thread": ("0", "s[{k}]"),
        }[mode]
        barrier = {"split": "__syncthreads();", "thread": None}[mode]

        assert not sc.dataflow or mode == "split", "dataflow progress counters are per warp: split only"
        counters = f"reinterpret_cast<volatile int*>(s + {max(sc.n_slots, 1) * 32})"  # after the slots
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
                lines = []
                if (st, r) in sc.waits:  # receive: spin until each producer published its stage
                    lines.append(f"volatile int* D = {counters};")
                    lines += [f"while (D[{p}] < {upto}) {{}}" for p, upto in sorted(sc.waits[st, r].items())]
                    lines.append("__threadfence_block();")
                lines += [f"const T t{s} = {expr(g, s, ref)};" for s in sorted(seen)]
                for it in by_rank[r]:
                    v = ref(it.scalar) if it.task < 0 else f"t{it.scalar}"
                    if it.scalar in sc.slots:
                        lines.append(f"{slot.format(k=sc.slots[it.scalar])} = {v};")
                    lines += self.store_outputs(it.scalar, v, sc.out_index)
                if (st, r) in sc.publish:  # send: every lane's writes, then the counter, no wait
                    lines += ["__syncwarp();", "__threadfence_block();",
                              f"if ((threadIdx.x & 31) == 0) {counters}[{r}] = {st};"]
                total += len(lines)
                body = "\n            ".join(lines)
                cases.append(f"        case {r}:\n        {{\n            {body}\n        }}\n        break;")
            stages.append((st, cases))

        split = total > self.budget
        fn = fn or f"{name}_{mode}"
        # the warp / block tiers share one signature; at P = 32 the tables are unused (none made)
        tab_sig, tab_arg = (", const T* __restrict__ tables", ", tables") if tables else ("", "")
        sig = f"{self.params}, T* __restrict__ s{tab_sig}, int ys"
        out, calls = [], []
        for st, cases in stages:
            sync = f"        {barrier}\n" if barrier and st in sc.sync_before else ""
            if st in sc.partial:  # a named barrier among the ranks that exchange values
                assert mode == "split", "named barriers count whole warps: split ranks only"
                bid, mask = sc.partial[st]
                sync = (f"        if (({mask}u >> (threadIdx.x >> 5)) & 1u) asm volatile(\"bar.sync %0, %1;\" "
                        f":: \"n\"({bid}), \"n\"({32 * bin(mask).count('1')}) : \"memory\");\n")
            sw = (f"        switch ({rank})\n        {{\n" + "\n".join(cases)
                  + "\n        default:\n            break;\n        }\n")
            if split:
                out.append(f"    template <class T>\n    __device__ __noinline__ void {fn}_s{st}({sig})\n"
                           f"    {{\n{sw}    }}\n")
                calls.append(sync + f"        {fn}_s{st}<T>({self.args}, s{tab_arg}, ys);\n")
            else:
                calls.append(sync + sw)
        if sc.dataflow:  # the counters must read 0 before any rank waits on them: one barrier
            calls.insert(0, f"        if (threadIdx.x < {sc.units}) {counters}[threadIdx.x] = 0;\n"
                            "        __syncthreads();\n")
        if mode == "thread":
            decl = f"        T s[{max(sc.n_slots, 1)}];\n"
            head = f"    template <class T>\n    CROCKET_HD void {fn}({self.params}, int ys = 1)\n    {{\n{decl}"
        else:
            head = f"    template <class T>\n    __device__ __forceinline__ void {fn}({sig} = 1)\n    {{\n"
        return "".join(out) + head + "".join(calls) + "    }\n"


    # ---- warp and block tiers --------------------------------------------------------------
    # The remaining shortcuts are tagged PROTOTYPE(uniform) and listed in OPEN_TASKS.md
    # ("Lane-uniform warp / block tiers"); grep the tag to clean up.

    def uniform(self, sc: sched_mod.Schedule, tier: str, problems: int = 1) -> tuple[str, int]:
        """The warp and block tiers: ``sc.units`` warps (the ranks; 1 for the warp tier) holding
        ``problems`` problems (P). Each warp's 32 lanes split into P groups of W = 32 / P lanes,
        group ``lane / W`` working on its own problem; lane ``i`` of every warp serves the same
        problem. P = 1 is GLASS's contract (one problem per warp / block); P = 32 puts one
        problem on each lane, as the split tier does. See docs/tier_layouts.html.

        The schedule places items (with ``placement="task"``, whole tasks) on warps, and each
        warp runs its own ``case`` per stage. Inside a warp every lane runs the SAME code: the
        warp's items of a stage are grouped by expression shape and each group is one
        straight-line statement, run in passes of W lanes, operands at ``sp[base + stride * i]``
        where the layout makes them affine and through a per-item index table where it does
        not. Every thread passes its own problem's input and output pointers. Returns (code,
        scratch elements)."""
        g, name = self.g, self.p.name
        if problems not in PROBLEMS:
            raise ValueError(f"problems={problems}: one of {PROBLEMS}")
        warps, P = sc.units, problems
        W = 32 // P  # lanes per problem
        threads = 32 * warps
        # The warp tier's warps share a block with other problems' warps: index by lane only.
        tid = "lane" if warps == 1 else "threadIdx.x"
        pre = f"{name}_{tier}"  # table names: a header can hold both tiers
        # A block barrier where a value crosses warps (step 4's sync_before), else a warp barrier:
        # lanes are reassigned for every group, so a warp always syncs its own lanes.
        block_bar = "__syncthreads();" if warps > 1 else "__syncwarp();"

        def full(st: int) -> bool:
            return warps == 1 or st in sc.sync_before
        # Layout: these tiers allocate their own slots instead of the schedule's interval
        # colouring, which scatters a value's elements. Inputs first, then each group's results
        # in one contiguous block in member order, so operand columns come out affine in the
        # lane (``sp[base + stride * i]``) and need no table. A block is freed (first-fit, merged
        # with free neighbours) once the last stage reading any of its members has passed; the
        # barrier before every stage makes that safe. Each problem has its own copy of the
        # slots, interleaved: slot x of problem p is ``s[x * P + p]`` (``sp = s + p``). A group's
        # W lanes read W consecutive slots of P problems, 32 consecutive words, so no bank
        # conflicts at any P; at P = 32 it is the split tier's ``s[slot * 32 + lane]``.
        # Contiguous per-problem regions measured 5x slower at P = 32 (bank conflicts).
        def at(e) -> str:
            return f"sp[{e}]" if P == 1 else f"sp[({e}) * {P}]"
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

        # Every per-lane table is packed into one constant array and one int array, held once
        # per block in shared memory that the caller passes as ``tables`` and fills once per
        # launch with ``<fn>_load_tables``: a group's index load is a shared load, not a dependent
        # global load (global tables measured ~9x slower than thread), and the tables are not in
        # each problem's scratch (they were 88% of g1's, which cut GRiM IK occupancy 1.7x).
        # With W = 1 the lane index is a constant, so no table is made: indices are literals.
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
            # each problem's lanes stage that problem's inputs (every warp: identical writes)
            body.append(f"        for (int i = sub; i < {length}; i += {W}) {at(f'{ids[0]} + i')} = {nm}[i];")
        # The prologue (tables, inputs) is copied by every thread of the block, so all warps
        # must pass a block barrier before stage 1 reads any of it.
        body.append(f"        {block_bar}")
        # Pass 1: each stage's groups per warp and, per scalar, the last stage that reads it.
        stage_groups: list[dict[int, list]] = []
        last_read: dict[int, float] = {int(x): math.inf for x in self.p.flat_outputs}
        for st in range(1, sc.stages + 1):
            groups: dict = {}
            for it in sc.stage_items(st):
                if it.task >= 0:
                    t = tree(it.scalar, it.task)
                    groups.setdefault(it.rank, {}).setdefault(repr(_shape(t)), []).append((it.scalar, t))
                    for kind, x in _leaves(t):
                        if kind == "v":
                            last_read[x] = max(last_read.get(x, 0), st)
            stage_groups.append({r: list(gs.values()) for r, gs in groups.items()})

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

        for st, by_warp in enumerate(stage_groups, 1):
            body.append(f"        {block_bar if full(st) else '__syncwarp();'}")
            # A block frees only at a block barrier: another warp may still be reading it.
            if full(st):
                for blk in [b_ for b_ in live if all(last_read.get(x, 0) < st for x in b_[2])]:
                    live.remove(blk)
                    release(blk[0], blk[1])
            if warps > 1:
                body.append("        switch (threadIdx.x >> 5)\n        {")
            for r, groups in sorted(by_warp.items()):
                if warps > 1:
                    body.append(f"        case {r}:\n        {{")
                self._uniform_groups(groups, W, at, where, alloc, live, slots, ints, consts, body)
                if warps > 1:
                    body.append("        }\n        break;")
            if warps > 1:
                body.append("        default:\n            break;\n        }")
        body.append(f"        {block_bar}")
        for nm, ids in self.p.outputs:
            src = []
            for i, x in enumerate(int(x) for x in ids):
                if g.op[x] == "const":
                    body.append(f"        if (sub == 0) {nm}[{i} * ys] = {literal(g.value[x])};")
                    src.append(0xFFFF)  # no slot: a constant output, written above
                else:
                    src.append(where(x))
            off = ints(src)
            body.append(f"        for (int i = sub; i < {len(ids)}; i += {W}) "
                        f"if (it[{off} + i] != 0xFFFF) {nm}[i * ys] = {at(f'it[{off} + i]')};")
        m, k = len(ctab), len(itab)
        head = [
            "        const int lane = threadIdx.x & 31;",
            f"        const int sub = lane & {W - 1};  // lane within this problem's group of {W}",
            f"        T* sp = s + lane / {W};  // this problem's slots, interleaved by {P}",
            "        const T* cv = tables;",
            f"        const unsigned short* it = reinterpret_cast<const unsigned short*>(tables + {m});"
            f"  // {k} indices",
        ]
        if any(0xFFFF <= i_ and i_ != 0xFFFF for i_ in itab) or any(i_ < 0 for i_ in itab):
            raise ValueError(f"{pre}: a slot index does not fit the 16-bit index tables (0xFFFF marks none)")
        tables = _table("unsigned short", f"{pre}_itab", itab) + _table("double", f"{pre}_ctab", ctab)
        loader = (f"    /** Fill ``tables`` ({pre}_tables<T>() elements of T) for {pre}: every thread of\n"
                  f"     *  the group calling it passes its rank and the group size; the group syncs after. */\n"
                  f"    template <class T>\n    __device__ __forceinline__ void {pre}_load_tables("
                  f"T* __restrict__ tables, int rank, int size)\n    {{\n"
                  f"        for (int i = rank; i < {m}; i += size) tables[i] = T({pre}_ctab[i]);\n"
                  f"        unsigned short* it = reinterpret_cast<unsigned short*>(tables + {m});\n"
                  f"        for (int i = rank; i < {k}; i += size) it[i] = {pre}_itab[i];\n    }}\n"
                  f"    /** ``tables`` for {pre}, in elements of T: {m} constants, then {k} 16-bit indices. */\n"
                  f"    template <class T>\n    __host__ __device__ constexpr int {pre}_tables()\n"
                  f"    {{\n        return {m} + ({k} * 2 + (int)sizeof(T) - 1) / (int)sizeof(T) + 1;\n    }}\n")
        # PROTOTYPE(uniform): no __noinline__ stage split over the statement budget (the ptxas
        # guard the other tiers have); validate every build numerically until it is added.
        fn = (f"    template <class T>\n    __device__ __forceinline__ void {name}_{tier}("
              f"{self.params}, T* __restrict__ s, const T* __restrict__ tables, int ys = 1)\n    {{\n"
              + "\n".join(head + body) + "\n    }\n")
        return tables + loader + fn, P * max(n, 1), m + k

    def _uniform_groups(self, groups, W, at, where, alloc, live, slots, ints, consts, body):
        """One warp's groups of one stage, each one lane-uniform statement (see ``uniform``)."""
        for members in groups:
            members = _order(members, where)
            count = len(members)
            base = alloc(count)
            live.append((base, count, [x for x, _ in members]))
            for i, (x, _) in enumerate(members):
                slots[x] = base + i
            leaves = [_leaves(t) for _, t in members]
            cols = []  # per operand: (kind, values over members)
            for j in range(len(leaves[0])):
                kind = leaves[0][j][0]
                cols.append((kind, [lv[j][1] if kind == "c" else where(lv[j][1]) for lv in leaves]))
            if W == 1:
                # One lane per problem: member c is a constant, so every index and constant is a
                # literal and the group is ``count`` plain statements (the split tier's code).
                for c in range(count):
                    names = [literal(v[c]) if kind == "c" else at(v[c]) for kind, v in cols]
                    body.append(f"        {at(base + c)} = {_render(members[0][1], iter(names))};")
                continue
            names = []
            for kind, vals in cols:
                stride = _affine(vals)
                if kind == "c" and stride == 0:
                    names.append(literal(vals[0]))  # same constant in every lane: an immediate
                elif kind == "c":  # converted to T once in the prologue, as T(literal) would be
                    names.append(f"cv[{consts(vals)} + i]")
                elif stride is not None:
                    names.append(at(_lin(vals[0], stride)))
                else:
                    names.append(at(f"it[{ints(vals)} + i]"))
            # Straight-line, no loop or branch: a stage is one basic block, so ptxas can
            # interleave its independent groups and hide one group's latency behind another's.
            # Lanes past the group recompute its last member and write the same value to the
            # same slot (the split tier's trick for a partial batch), so the result is unchanged.
            for c in range(0, count, W):
                body.append(f"        {{ const int i = min(sub + {c}, {count - 1}); "
                            f"{at(_lin(base, 1))} = {_render(members[0][1], iter(names))}; }}")


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


def _table(ctype: str, name: str, vals) -> str:  # ctype: int, unsigned short or double
    # PROTOTYPE(uniform): tables are `static __device__` globals in the header, so each including
    # translation unit gets its own copy; move them into a per-robot .cu.
    body = ", ".join(repr(float(v)) if ctype == "double" else str(int(v)) for v in vals) or "0"
    return f"    static __device__ const {ctype} {name}[{max(len(vals), 1)}] = {{{body}}};\n"


def smem(sc: sched_mod.Schedule, tier: str) -> int:
    """Scratch of a ``cooperative`` schedule, in elements of T (``tier="split"``: 32 problems)."""
    # dataflow adds one progress counter per rank (an int, reserved as one T) after the slots
    return max(sc.n_slots, 1) * (32 if tier == "split" else 1) + (sc.units if sc.dataflow else 0)


def emit(program: Program, variants: list[Variant], budget: int = BUDGET) -> str:
    """One algorithm at each tier in ``variants``, with its scratch and group-size constants."""
    em = _Emitter(program, budget)
    out, consts = [], []
    for v in variants:
        if v.tier == "thread":
            out.append(em.thread())
            continue
        if v.tier not in ("warp", "block"):
            raise ValueError(f"tier {v.tier!r}: one of {TIERS}")
        warps = 1 if v.tier == "warp" else v.units // 32
        if v.tier == "block" and (v.units % 32 or warps not in WARPS):
            raise ValueError(f"block tier: units={v.units} threads; k = units / 32 must be one of {WARPS}")
        if v.problems not in PROBLEMS:
            raise ValueError(f"problems={v.problems}: one of {PROBLEMS}")
        if v.sync != "block" and (v.problems != 32 or warps == 1):
            raise ValueError(f"sync={v.sync!r} needs P = 32 and several warps (block tier)")
        pre = f"{program.name}_{v.tier}"
        if v.problems == 32:  # one problem per lane: the per-lane trace, with CSE and registers
            sc = sched_mod.build(program, warps, v.threshold, placement=v.placement, sync=v.sync)
            code = em.cooperative(sc, "split", pre, tables=True) + (
                f"    template <class T>\n    __device__ __forceinline__ void {pre}_load_tables("
                "T* __restrict__, int, int) {}  // P = 32: no tables\n"
                f"    template <class T>\n    __host__ __device__ constexpr int {pre}_tables() {{ return 1; }}\n")
            n_slots, n_tables = smem(sc, "split"), 0
        else:
            sc = sched_mod.build(program, warps, v.threshold, placement=v.placement)
            code, n_slots, n_tables = em.uniform(sc, v.tier, v.problems)
        out.append(code)
        consts.append(f"    constexpr int {pre}_smem = {max(n_slots, 1)};  // per warp (warp) / block")
        consts.append(f"    constexpr int {program.name}_{v.tier}_problems = {v.problems};")
        if v.tier == "block":
            consts.append(f"    constexpr int {program.name}_block_threads = {v.units};")
        consts.append(f"    constexpr int {program.name}_{v.tier}_stages = {sc.stages};")
    sizes = [f"    constexpr int {program.name}_{n}_size = {len(ids)};" for n, ids in program.outputs]
    return "\n".join(out) + "\n" + "\n".join(consts + sizes) + "\n"


def layout(program: Program, v: Variant, size: int = 4) -> tuple[int, int]:
    """(scratch, tables) of a warp / block variant in elements of a ``size``-byte T: scratch per
    warp (warp tier) or per block, tables once per block (``<alg>_<tier>_tables<T>()``)."""
    src = emit(program, [v])
    pre = f"{program.name}_{v.tier}"
    scratch = int(src.split(f"{pre}_smem = ")[1].split(";")[0])
    if v.problems == 32:
        return scratch, 1
    m, k = (int(x) for x in re.search(rf"{pre}_tables\(\)\n    {{\n        return (\d+) \+ \((\d+) \*", src).groups())
    return scratch, m + (2 * k + size - 1) // size + 1


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


DEFAULT_VARIANTS = (Variant("thread"), Variant("warp", 32), Variant("block", 128))


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
