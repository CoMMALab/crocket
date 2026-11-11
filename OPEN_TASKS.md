# Open tasks: crocket

Things that have been found but not fixed yet. Add an entry when you find one, and delete it
when it's fixed (the fix commit is the record). Format: **title**: symptom / evidence / where /
suggested fix. Items marked _(verify)_ were seeded from earlier session notes and haven't been
re-checked against current code.

## Bugs

- **ptxas miscompiles large straight-line traces** _(verify)_: under CUDA 13.3 at -O2/-O3, the
  G1 ID-gradient thread kernel is silently wrong. The workaround is `-Xptxas -O1` or the
  warp-split tier. A proper fix would split traces past a size threshold, and every build still
  needs numerical validation.

- **GRiM's MPPI IK is capped at 32 particles per seed**: `mppi_ik(n_particles=...)` above 32
  is rejected (`InvalidArgument` from the FFI handler); before, it was silently clamped to 32.
  The cap is `CK_MAX_EVAL` in GRiM's `kernels/ik_crocket/ck.cuh`: a seed's particle costs live
  in a fixed `real err[CK_MAX_EVAL]` / `real costs[CK_MAX_EVAL]`, in registers at the thread tier
  and in the per-seed shared `Work` slot at the warp and block tiers. Suggested fix: make the
  cap a build-time parameter (shared memory sized from it at the cooperative tiers), or stream
  particles through the 32 slots in batches with a running softmax (max-shifted sum) so any
  count works with fixed storage.

## Cooperative tier redesign (in progress)

- **Warps are the ranks, lanes are SIMD.** The `warp`/`block` tiers make lanes the ranks (`switch`
  on the lane), which SIMT serialises. The redesign: ranks are warps (only they have their own
  instruction stream), and lanes inside a warp do the same work, either on different problems
  (step 1, built on `split`) or on a task's elements (step 3, GLASS-style). Step 2 is
  point-to-point sync with named barriers (`bar.sync id, n`) instead of `__syncthreads`.
  Step 1 so far: `Variant(..., whole=True)` keeps every task whole on one rank (warp), placed
  on the rank that wrote most of its inputs within LPT's load bound (`schedule.build` step 3).
  Measured (us, B=1, idle A5000; thread / best split / best split whole): RNEA baxter
  17.7 / 34.4 / 16.0; RNEA panda 8.3 / 16.2 / 12.7; rnea_derivatives baxter 197 / 174 / 137
  (B=1024: 281 / 228 / 173); panda 76 / 98 / 84; traj_collision panda 164 / 109 / 109; fetch
  230 / 146 / 155. It halves cross-warp reads and cuts RNEA barriers 40-46%, but trajopt's
  barriers do not drop and its large uneven tasks balance worse, so `whole` is a per-robot
  choice: add it to `tune.candidates`. The thread tier still wins at B=8192 everywhere.
  Cleanup once step 3 lands, tagged `REDESIGN(ranks)` and `PROTOTYPE(uniform)` in code: delete
  the lane-switch `warp`/`block` codegen (`cuda._Emitter.cooperative`), `Variant.narrow`
  (`schedule.build`), and the `uniform` tier, or fold what step 3 reuses from it. Move GRiM's IK
  callers of `_warp`/`_block` (`ik_crocket/ck.cuh`, `region_ik/brownian_ik.cu`) to the new tiers.

## Lane-uniform tier (prototype)

- **`uniform` tier: correct, at best on par with `split`; likely delete.** Goal: within-problem
  parallelism. `warp`/`block` give each lane its own `switch` case, which SIMT runs one lane at a
  time (RNEA panda B=1: thread 8.3 us, warp 87 us). `Variant("uniform", 32)`
  (`cuda._Emitter.uniform`) groups each stage's items by expression shape and runs each group as
  one straight-line statement over lanes. Its own slot layout puts each group's results in one
  contiguous block, members ordered to make operand columns affine (`s[base + stride * i]`);
  the rest go through per-item index tables. Bit-identical to the thread tier
  (`test_group_size_invariance`).
  Measured (us, B=1, idle A5000, `harness` timing; thread / split4 / warp / uniform):
  traj_collision panda 164 / 127 / 442 / 135; fetch 234 / 166 / 511 / 169;
  RNEA panda 8.3 / 20 / 87 / 21; baxter 17.7 / 44 / 207 / 33;
  rnea_derivatives panda 76 / 92 / 467 / 156; baxter 197 / 179 / 760 / 386.
  At B=64+ uniform falls further behind (one problem per warp, large scratch).
  What each step bought: index tables in global memory first (RNEA panda 76 us); tables copied
  to scratch (54); own affine layout (38); straight-line stages instead of per-group loops (32);
  shared-typed scratch in the harness (21). Block reuse barely changes scratch: index tables
  are most of it (rnea_derivatives baxter 24.8k ints of 125 KB).
  Why it stalls there: 40-60% of operand columns stay non-affine. Folding gives related elements
  different shapes, so they land in different groups and blocks. A two-level pattern
  (`base + s0 * (i % w) + s1 * (i / w)`) covers only 10-30% of the remaining table entries.
  Fixing this needs a global layout (place producers for their consumers), with diminishing
  returns.
  Conclusion: `split` is already the lane-uniform design that works (lanes = problems, warps split
  stages). It matches or beats `uniform` at B=1 everywhere but RNEA baxter, and wins at
  larger B. Recommend deleting `uniform` (grep `PROTOTYPE(uniform)`) unless a global layout is
  attempted.
  Shortcuts, all tagged `PROTOTYPE(uniform)` in code: inputs staged into scratch; a `__syncwarp`
  before every stage; item DAGs expanded as trees (recomputation); greedy per-group member
  order; tables are `static __device__` globals in the header, copied into scratch per
  problem; no `__noinline__` budget split; launched as `warp` in `harness._kernel`; not in
  `tune.candidates`, `DEFAULT_VARIANTS` or `parallel_analysis`.

- **`split` stage functions lose the shared address space.** Over the statement budget,
  cooperative stages are `__noinline__` and take scratch as a generic `T*`, so loads are generic,
  not `LDS`. With the stages inlined (budget 10^9), split4 is 4-7% faster and still
  bit-identical (traj_collision panda/fetch, rnea_derivatives baxter). The noinline split
  guards against ptxas miscompiling large traces, so don't just drop it: pass scratch as a
  shared-space offset (or `__cvta_generic_to_shared` plus `ld.shared`) through the stage calls.

## Environment

- **NVRTC / driver version skew** _(verify)_: the NVRTC from conda can be newer than the driver,
  and the conda PATH can shadow the system nvcc.

## Branches

- **`benchmarks` has library changes `main` lacks.** `benchmarks` should be `main` plus the
  benchmark suite, but the IK work (`ik_geometry`, `analytic_ik_*`, SQP/LS traces, `rot_joint`,
  continuous joints, `tests/test_oracle.py`) landed only on `benchmarks`. Cherry-pick the
  non-`benchmarks/` commits onto `main`, then rebuild `benchmarks` on top. Until then GRiM pins
  its crocket submodule to `benchmarks` (`.gitmodules`).
