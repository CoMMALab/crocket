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
    # stage -> (named barrier id, rank bitmask) for a barrier among some ranks only; a stage in
    # sync_before but not here is a full barrier
    partial: dict[int, tuple[int, int]] = field(default_factory=dict)
    # sync="dataflow": no barriers. (stage, rank) -> {producer rank: stage} the rank must see
    # finished before running that stage's items, and the (stage, rank) points that publish
    # their progress. See build step 4.
    dataflow: bool = False
    waits: dict[tuple[int, int], dict[int, int]] = field(default_factory=dict)
    publish: set[tuple[int, int]] = field(default_factory=set)

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


def build(program: Program, units: int, threshold: int = 8, narrow: int = 0,
          placement: str = "element", sync: str = "block") -> Schedule:
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

    # REDESIGN(ranks): ``narrow`` only reduces barriers around the per-lane warp/block tiers'
    # serialised work; delete it with those tiers.
    # 3. rank assignment per stage; ``placement`` is what a rank is given. "element": items
    #    spread longest first (LPT), so one task's elements can land on several ranks. "task"
    #    (the rank model: ranks are warps): every task stays on one rank, longest task first, on
    #    the rank that wrote most of its leaves if that keeps the stage within LPT's makespan
    #    bound, else the least-loaded rank; step 4 then puts a barrier only where a value really
    #    crosses ranks. A stage of at most ``narrow`` items keeps tasks on one rank with no
    #    load bound.
    if placement not in ("element", "task"):
        raise ValueError(f"placement {placement!r}: 'element' or 'task'")
    by_task_placement = placement == "task"
    rank_of: dict[int, int] = {}
    for st in range(1, stages + 1):
        stage = [it for it in items if it.stage == st]
        load = [0] * units
        if by_task_placement or len(stage) <= narrow:
            by_task: dict[int, list[Item]] = {}
            for it in stage:
                by_task.setdefault(it.task, []).append(it)
            cost = {n: sum(it.cost for it in its) for n, its in by_task.items()}
            cap = (max(max(cost.values()), -(-sum(cost.values()) // units)) if by_task_placement and cost
                   else float("inf"))
            for task in sorted(by_task, key=lambda n: (-cost[n], n)):
                votes = [0] * units
                for it in by_task[task]:
                    for leaf in it.leaves:
                        if leaf in rank_of:
                            votes[rank_of[leaf]] += 1
                fits = [u for u in range(units) if votes[u] and load[u] + cost[task] <= cap]
                r = (min(fits, key=lambda u: (-votes[u], load[u], u)) if fits
                     else min(range(units), key=lambda u: (load[u], u)))
                for it in by_task[task]:
                    it.rank = r
                    load[r] += it.cost
        else:
            for it in sorted(stage, key=lambda it: (-it.cost, it.scalar)):
                r = min(range(units), key=lambda u: (load[u], u))
                it.rank = r
                load[r] += it.cost
        rank_of.update({it.scalar: it.rank for it in stage})
    items.sort(key=lambda it: (it.stage, it.rank, it.scalar))

    # 4. syncs: before a stage that reads another rank's unsynced write
    readers: dict[int, list[Item]] = {}
    for it in items:
        for leaf in it.leaves:
            readers.setdefault(leaf, []).append(it)
    sync_before: set[int] = set()
    partial: dict[int, tuple[int, int]] = {}
    if sync == "block":
        pending: dict[int, int] = {}  # scalar -> writer rank since the last sync
        for st in range(1, stages + 1):
            stage_ranks = {it.scalar: it.rank for it in items if it.stage == st}
            for it in (it for it in items if it.stage == st):
                if any(leaf in pending and pending[leaf] != it.rank for leaf in it.leaves):
                    sync_before.add(st)
                    pending.clear()
                    break
            pending.update({s: r for s, r in stage_ranks.items() if s in readers})
    elif sync == "named":
        # REDESIGN(ranks): correct but measured no faster than "block" (within +-5% at B=1,
        # worse at large B from lost slot reuse): the stage-levelled schedule leaves a skipped
        # rank nothing to run ahead to. Keep only if a dataflow schedule (step 2b) uses it.
        # Point to point, as an MPI receive: a stage's barrier holds only the ranks that read a
        # value they cannot see yet, and the ranks that can. visible[s] is the set of ranks a
        # write is visible to; a barrier among P makes every write visible to one member of P
        # visible to all of P (bar.sync orders shared memory among its participants, and that
        # ordering composes along chains of barriers).
        visible: dict[int, set[int]] = {}
        everyone = set(range(units))
        for st in range(1, stages + 1):
            stage = [it for it in items if it.stage == st]
            need: set[int] = set()
            for it in stage:
                for leaf in it.leaves:
                    if it.rank not in visible[leaf]:
                        need |= {it.rank} | visible[leaf]
            if need:
                sync_before.add(st)
                if need != everyone:
                    partial[st] = (0, sum(1 << r for r in need))  # id assigned below
                members = everyone if need == everyone else need
                for v in visible.values():
                    if v & members:
                        v |= members
            for it in stage:
                if it.scalar in readers:
                    visible[it.scalar] = {it.rank}
        # One named barrier id per distinct rank set (sm_80+: ids 1-15; 0 is __syncthreads). Only
        # its members ever use an id, and they reach its barriers in program order, so reusing it
        # is safe. Sets past 15 become full barriers: a superset, so still correct.
        ids: dict[int, int] = {}
        for st in sorted(partial):
            mask = partial[st][1]
            if mask not in ids and len(ids) < 15:
                ids[mask] = len(ids) + 1
            if mask in ids:
                partial[st] = (ids[mask], mask)
            else:
                del partial[st]
    elif sync == "dataflow":
        # REDESIGN(ranks): correct but measured 1.4-2.3x slower than "block" on RNEA: the
        # barrier stalls it removes were true dependency waits (the joint chain), so warps wait
        # on their producers about as long, and polling the counters costs more than a barrier.
        # Each rank runs its items in stage order (one global dependency order) and never meets
        # a barrier. A rank publishes "finished stage s" in its progress counter without waiting
        # (an MPI send); a reader waits only on the producer of a value it reads (a receive). The
        # waits are below: read-after-write here, write-after-read where step 5 reuses a slot.
        # Deadlock-free: a rank waits only for earlier stages, which every rank publishes
        # before it waits on anything later.
        pass
    else:
        raise ValueError(f"sync {sync!r}: 'block', 'named' or 'dataflow'")
    dataflow = sync == "dataflow"
    waits: dict[tuple[int, int], dict[int, int]] = {}

    def need(stage: int, rank: int, producer: int, upto: int):
        if producer != rank:
            w = waits.setdefault((stage, rank), {})
            w[producer] = max(w.get(producer, 0), upto)

    if dataflow:
        where = {it.scalar: it for it in items}
        for it in items:
            for leaf in it.leaves:
                need(it.stage, it.rank, where[leaf].rank, where[leaf].stage)

    # 5. slots by interval colouring; a slot frees at the first sync after its last read. Only a
    #    full barrier orders every last reader before the slot's next writer, so a partial one
    #    never frees a slot.
    #    Under dataflow a slot frees the stage after its last read, and its next writer waits
    #    for every other rank that read the old value to finish the stage it read it in.
    last_read = {s: max(r.stage for r in rs) for s, rs in readers.items()}
    syncs = sorted(sync_before - set(partial))
    free_at = {s: (last + 1 if dataflow else next((b for b in syncs if b > last), stages + 1))
               for s, last in last_read.items()}
    occupant: dict[int, int] = {}  # slot -> the scalar last stored in it
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
            if dataflow:
                for r in readers[occupant[slot]]:
                    need(it.stage, it.rank, r.rank, r.stage)
        else:
            slot, n_slots = n_slots, n_slots + 1
        slots[it.scalar] = slot
        occupant[slot] = it.scalar
        busy.append((free_at[it.scalar], slot))
    # A rank that already waited for producer p to reach stage s need not wait again for any
    # stage <= s; and a rank publishes only the (stage, rank) points somebody waits for.
    publish: set[tuple[int, int]] = set()
    if dataflow:
        known: dict[int, dict[int, int]] = {}
        for key in sorted(waits, key=lambda k: (k[0], k[1])):
            st, r = key
            seen_r = known.setdefault(r, {})
            w = {p: upto for p, upto in waits[key].items() if upto > seen_r.get(p, 0)}
            seen_r.update(w)
            waits[key] = w
            publish |= {(upto, p) for p, upto in w.items()}
        waits = {k: w for k, w in waits.items() if w}
    return Schedule(program, units, threshold, items, stages, sync_before, slots, n_slots, task_of,
                    out_index, partial, dataflow, waits, publish)
