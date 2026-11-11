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
  Step 1 so far: `Variant(..., placement="task")` keeps every task on one rank (warp), placed
  on the rank that wrote most of its inputs within LPT's load bound (`schedule.build` step 3).
  Measured (us, B=1, idle A5000; thread / best split element / best split task): RNEA baxter
  17.7 / 34.4 / 16.0; RNEA panda 8.3 / 16.2 / 12.7; rnea_derivatives baxter 197 / 174 / 137
  (B=1024: 281 / 228 / 173); panda 76 / 98 / 84; traj_collision panda 164 / 109 / 109; fetch
  230 / 146 / 155. It halves cross-warp reads and cuts RNEA barriers 40-46%, but trajopt's
  barriers do not drop and its large uneven tasks balance worse, so `placement` is a per-robot
  choice, tuned in `tune.candidates`. The thread tier still wins at B=8192 everywhere.
  Step 2 (named barriers, `Variant(sync="named")`): only the ranks that exchange a value sync
  (`bar.sync id, n` per distinct rank set), with visibility tracked through chains of barriers.
  Correct (bit-identical) but no faster: within +-5% of `__syncthreads` at B=1 on RNEA,
  rnea_derivatives and traj_collision, and up to 1.7x slower at B=8192, because a slot is not
  reused after a partial barrier. The stage-levelled schedule leaves a skipped rank nothing to
  run ahead to; barrier waits are not the cost. Not in `tune.candidates`. Step 2b, if pursued:
  a dataflow schedule (each warp runs its tasks in dependency order, consumers wait on
  producers with `bar.arrive` (producer, non-blocking) / `bar.sync` (consumer), as an MPI
  send/recv) so a warp can run ahead; then decide whether `sync="named"` stays.
  Step 2b (dataflow, `Variant(sync="dataflow")`): no barriers; each warp publishes "finished
  stage s" in a shared progress counter (no wait) and spins only on the producers it reads
  from, plus write-after-read waits where a slot is reused. Correct (bit-identical) but 1.4-2.3x
  slower than `__syncthreads` on RNEA panda/baxter (split4 task, B=1: 13.0 -> 19.0 us). ncu
  (panda, split4 task): barrier stalls (1.7k cycles/warp) are replaced by spin-loop and wait
  stalls (2.1k + 1.9k): the time warps spent at barriers was dependency waiting along the joint
  chain, not imbalance a dataflow schedule removes, and polling costs more than a barrier
  wake. Conclusion of steps 2 and 2b: synchronization is not the cooperative tiers' cost; the
  dependency critical path is. `dataflow` vs `block` is kept as a split tuning choice
  (`tune.candidates`), since another graph shape may favour dataflow; `named` stays tagged for
  removal and untuned.
  Step 3 (lanes over a task's elements, `Variant("uniform", k, placement="task")`): one problem
  per block of k warps; warps are the ranks (tasks placed whole as in step 1, `__syncthreads`
  only where step 4 says a value crosses warps), and inside a warp the lanes run the warp's
  items as lane-uniform shape groups. Bit-identical to thread. Measured (us, B=1; thread / best
  split / best step 3, all best at 8 warps with task placement): traj_collision panda 164 / 109 /
  89.5, fetch 244 / 139 / 97.1; rnea_derivatives baxter 197 / 137 / 117, panda 76 / 82 / 71;
  RNEA panda 8.3 / 12.8 / 20.8, baxter 17.7 / 20.8 / 34.6. So large traces at small batch gain
  1.7-2.5x over thread; RNEA stays thread's. Task placement beat element placement in every
  step 3 case. Next: add it to `tune.candidates`, then the GRiM migration (tier renames:
  split -> block with lanes = problems, uniform -> block with lanes = elements, k = 1 is warp).
  Cleanup once step 3 lands, tagged `REDESIGN(ranks)` and `PROTOTYPE(uniform)` in code: delete
  the lane-switch `warp`/`block` codegen (`cuda._Emitter.cooperative`), `Variant.narrow`
  (`schedule.build`), and the `uniform` tier, or fold what step 3 reuses from it. Move GRiM's IK
  callers of `_warp`/`_block` (`ik_crocket/ck.cuh`, `region_ik/brownian_ik.cu`) to the new tiers.

## Lane-uniform tier (prototype)

- **`uniform` is now rank model step 3 (see above); its prototype shortcuts remain.** History of
  the single-warp version (RNEA panda B=1): index tables in global memory 76 us; tables copied
  to scratch 54; own affine slot layout 38; straight-line stages 32; shared-typed harness
  scratch 21. 40-60% of operand columns stay non-affine (folding gives related elements
  different shapes), and a two-level pattern `base + s0 * (i % w) + s1 * (i / w)` covers only
  10-30% of the rest, so the index tables are most of its scratch.
  Shortcuts, all tagged `PROTOTYPE(uniform)` in code: inputs staged into scratch; item DAGs
  expanded as trees (recomputation); greedy per-group member order; tables are `static
  __device__` globals in the header, copied into scratch per problem; no `__noinline__`
  budget split; not yet in `tune.candidates`, `DEFAULT_VARIANTS` or `parallel_analysis`.

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
