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

from dataclasses import dataclass

from . import schedule as sched_mod
from .robot import Robot
from .trace import ALGORITHMS, Program

TIERS = ("thread", "split", "warp", "block")
BUDGET = 4000

_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def literal(v: float) -> str:
    return f"T({v!r})"


def expr(g, s: int, ref) -> str:
    op, args = g.op[s], [ref(a) for a in g.args[s]]
    if op in _INFIX:
        return f"{args[0]} {_INFIX[op]} {args[1]}"
    if op == "neg":
        return f"-{args[0]}"
    return f"{op}({args[0]})"


@dataclass
class Variant:
    """How one algorithm is emitted at one tier."""

    tier: str
    units: int = 1  # lanes per problem (warp: 32, block: threads), warps per block for split
    threshold: int = 8  # scalarisation threshold, see :mod:`cricket2.schedule`


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
            return self.cooperative(sched_mod.build(self.p, 1, threshold=1 << 30), "thread")
        ref = lambda a: self.ref_base(a) or f"t{a}"  # noqa: E731
        lines = [f"        const T t{s} = {expr(g, s, ref)};" for s in body]
        for k, s in enumerate(outputs):
            name, i = self.where[k]
            lines.append(f"        {name}[{i} * ys] = {ref(int(s))};")
        return (f"    template <class T>\n    CRICKET2_HD void {self.p.name}_thread({self.params}, int ys = 1)\n"
                "    {\n" + "\n".join(lines) + "\n    }\n")

    # ---- cooperative tiers ----------------------------------------------------------------

    def cooperative(self, sc: sched_mod.Schedule, mode: str) -> str:
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
            head = f"    template <class T>\n    CRICKET2_HD void {fn}({self.params}, int ys = 1)\n    {{\n{decl}"
        else:
            head = f"    template <class T>\n    __device__ __forceinline__ void {fn}({sig} = 1)\n    {{\n"
        return "".join(out) + head + "".join(calls) + "    }\n"


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
        sc = sched_mod.build(program, v.units, v.threshold)
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

// Generated by cricket2 for robot "{robot.name}". Do not edit.
//
// Robot-specialized rigid-body kernels for CUDA, templated on T (float or double). Inputs are
// one problem's q / qd / qdd / tau (n_q each, ordered as `joint_names`); output element i is
// written at y[i * ys]. Gravity is -9.81 along world z.
//
// Define CRICKET2_HD before including to override the thread-tier qualifiers.

#ifndef __CUDACC_RTC__  // NVRTC provides the math functions as builtins
#include <math.h>
#endif

#ifndef CRICKET2_HD
#ifdef __CUDACC__
#define CRICKET2_HD __host__ __device__ __forceinline__
#else
#define CRICKET2_HD inline
#endif
#endif

// NOLINTBEGIN(*-magic-numbers)
namespace cricket2::robots::{ns}
{{
    constexpr int n_q = {robot.n_q};
    constexpr const char *joint_names[n_q] = {{"{joints}"}};

{"".join(sections)}}}  // namespace cricket2::robots::{ns}
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
    CRICKET2_HD void frame_poses(const float *q, float *poses)
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
    CRICKET2_HD void ee_pose_jacobian(const float *q, float *pose, float *jacobian)
    {
        float y[12];
        frame_jacobian_thread<float>(q, y, jacobian);
        detail::pack_pose(y, pose);
    }

    CRICKET2_HD void ee_pose(const float *q, float *pose)
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
        CRICKET2_HD void pack_pose(const float *y, float *pose)
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
