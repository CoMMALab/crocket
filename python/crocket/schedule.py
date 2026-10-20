"""Scheduling a traced program onto a group of lanes per problem.

1. **Coarsen.** A node whose folded cost is below ``threshold`` and that feeds exactly one other
   node is *scalarised*: it joins its consumer and is recomputed in the consumer's lanes.
   Every other node is a *task*; its elements that other tasks read (or that are outputs) are
   *items*, one lane each.
2. **Level.** A task's stage is one more than the latest task it reads. Tasks at the same depth
   of sibling subtrees land in one stage, which is GRiD's BFS-level parallelism.
3. **Assign lanes.** Each stage's items go to ``units`` ranks, longest first (as
   ``cricket.tiered``), now over far fewer and wider items.
4. **Insert syncs** at a stage boundary only when a later stage reads a value another rank wrote
   since the last sync.
5. **Place values.** An item read by a later stage gets a shared-memory slot, reused by interval
   colouring once a sync follows its last read.

Every output element is computed by one lane from one fixed expression, the same one the
thread tier computes, so tiers and group sizes agree bit for bit under ``-fmad=false``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .trace import Program


@dataclass
class Item:
    scalar: int
    task: int
    cost: int
    stage: int = 0
    rank: int = 0
    leaves: set[int] = field(default_factory=set)  # other tasks' items it reads


@dataclass
class Schedule:
    program: Program
    units: int
    threshold: int
    items: list[Item]
    stages: int
    sync_before: set[int]  # stages preceded by a barrier
    slots: dict[int, int]  # item scalar -> shared-memory slot
    n_slots: int
    task_of: dict[int, int]  # scalar -> owning task (materialized scalars only)
    out_index: dict[int, list[int]] = field(default_factory=dict)  # scalar -> flat output positions

    def stage_items(self, stage: int) -> list[Item]:
        return [it for it in self.items if it.stage == stage]


def walk(g, s: int, task: int, task_of: dict[int, int], seen: set[int], leaves: set[int]):
    """The in-lane statements for scalar ``s`` of ``task``: recurse until constants, inputs or
    another task's items. Collects the statements into ``seen`` and those items into ``leaves``."""
    stack = [s]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        for a in g.args[x]:
            if g.op[a] in ("const", "in") or a in seen:
                continue
            if task_of.get(a, task) != task:
                leaves.add(a)
            else:
                stack.append(a)


def build(program: Program, units: int, threshold: int = 8) -> Schedule:
    g = program.g
    outputs = program.flat_outputs
    live = g.live(outputs)
    out_index: dict[int, list[int]] = {}
    for k, s in enumerate(outputs):
        out_index.setdefault(int(s), []).append(k)

    # 1. coarsen: which nodes are tasks
    owner = np.asarray(g.owner)
    created = {n: [s for s in nd.created if live[s]] for n, nd in enumerate(g.nodes)}
    created = {n: c for n, c in created.items() if c}
    consumers: dict[int, set[int]] = {n: set() for n in created}
    for n, cs in created.items():
        for s in cs:
            for a in g.args[s]:
                o = owner[a]
                if o >= 0 and o != n:
                    consumers[o].add(n)
    is_task = {n: len(cs) >= threshold or len(consumers[n]) != 1 or any(s in out_index for s in cs)
               for n, cs in created.items()}
    task_of = {s: n for n, cs in created.items() if is_task[n] for s in cs}

    # 2. items, walking tasks latest first so every reader is known before its producer
    needed: dict[int, set[int]] = {n: set() for n in created if is_task[n]}
    for s in out_index:
        if s in task_of:
            needed[task_of[s]].add(s)
    deps: dict[int, set[int]] = {}
    items: list[Item] = []
    for n in sorted(needed, reverse=True):
        deps[n] = set()
        for s in sorted(needed[n]):
            seen, leaves = set(), set()
            walk(g, s, n, task_of, seen, leaves)
            items.append(Item(s, n, len(seen), leaves=leaves))
            for leaf in leaves:
                needed[task_of[leaf]].add(leaf)
                deps[n].add(task_of[leaf])
    level: dict[int, int] = {}
    for n in sorted(deps):
        level[n] = 1 + max((level[d] for d in deps[n]), default=0)
    for it in items:
        it.stage = level[it.task]
    stages = max(level.values(), default=0)
    # outputs that are constants or inputs: zero-cost copies in the first stage
    for s in sorted(out_index):
        if s not in task_of:
            items.append(Item(s, -1, 0, stage=1))
    stages = max(stages, 1 if items else 0)

    # 3. longest-processing-time lane assignment per stage
    for st in range(1, stages + 1):
        load = [0] * units
        for it in sorted((it for it in items if it.stage == st), key=lambda it: (-it.cost, it.scalar)):
            r = min(range(units), key=lambda u: (load[u], u))
            it.rank = r
            load[r] += it.cost
    items.sort(key=lambda it: (it.stage, it.rank, it.scalar))

    # 4. syncs: before a stage that reads another rank's unsynced write
    readers: dict[int, list[Item]] = {}
    for it in items:
        for leaf in it.leaves:
            readers.setdefault(leaf, []).append(it)
    sync_before: set[int] = set()
    pending: dict[int, int] = {}  # scalar -> writer rank since the last sync
    for st in range(1, stages + 1):
        stage_ranks = {it.scalar: it.rank for it in items if it.stage == st}
        for it in (it for it in items if it.stage == st):
            if any(leaf in pending and pending[leaf] != it.rank for leaf in it.leaves):
                sync_before.add(st)
                pending.clear()
                break
        pending.update({s: r for s, r in stage_ranks.items() if s in readers})

    # 5. slots by interval colouring; a slot frees at the first sync after its last read
    last_read = {s: max(r.stage for r in rs) for s, rs in readers.items()}
    syncs = sorted(sync_before)
    free_at = {s: next((b for b in syncs if b > last), stages + 1) for s, last in last_read.items()}
    slots: dict[int, int] = {}
    busy: list[tuple[int, int]] = []  # (free stage, slot)
    free: list[int] = []
    n_slots = 0
    for it in items:
        if it.scalar not in readers or it.scalar in slots:
            continue
        free += [slot for at, slot in busy if at <= it.stage]
        busy = [b for b in busy if b[0] > it.stage]
        if free:
            free.sort(reverse=True)
            slot = free.pop()
        else:
            slot, n_slots = n_slots, n_slots + 1
        slots[it.scalar] = slot
        busy.append((free_at[it.scalar], slot))
    return Schedule(program, units, threshold, items, stages, sync_before, slots, n_slots, task_of,
                    out_index)
