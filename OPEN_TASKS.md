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

## Environment

- **NVRTC / driver version skew** _(verify)_: the NVRTC from conda can be newer than the driver,
  and the conda PATH can shadow the system nvcc.

## Branches

- **`benchmarks` has library changes `main` lacks.** `benchmarks` should be `main` plus the
  benchmark suite, but the IK work (`ik_geometry`, `analytic_ik_*`, SQP/LS traces, `rot_joint`,
  continuous joints, `tests/test_oracle.py`) landed only on `benchmarks`. Cherry-pick the
  non-`benchmarks/` commits onto `main`, then rebuild `benchmarks` on top. Until then GRiM pins
  its crocket submodule to `benchmarks` (`.gitmodules`).
