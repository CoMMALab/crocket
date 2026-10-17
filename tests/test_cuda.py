"""Surface two: every compiled tier against the IR interpreter, and the tiers against each other.
Generated per (robot, algorithm, tier); needs nvcc and a GPU.

* thread tier, float and double, vs the float64 interpreter (GRiM's float bars, 1e-9 in double);
* split / warp / block tiers vs the thread tier: bit-identical under ``-fmad=false``;
* group-size invariance: the same outputs at every group size and batch size;
* large-trace guard: functions over the statement budget agree between ptxas -O0 and -O3.

    python -m pytest tests/test_cuda.py
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu_proof

import cricket2
from cricket2 import cuda, harness
from robots import SMALL, options, robot

ALGORITHMS = ["rnea", "crba", "aba", "rnea_derivatives", "frame_poses", "frame_jacobian"]
CASES = ([(r, a) for r in SMALL for a in ALGORITHMS if not (r == "fr3" and a == "aba")]
         + [("grid_g1", a) for a in ("rnea", "crba", "frame_jacobian")])
COOPERATIVE = [cuda.Variant("split", 8), cuda.Variant("warp", 32), cuda.Variant("block", 128)]


def _trace(robot_name, name):
    r = robot(robot_name)
    return cricket2.trace(r, name, **options(r, name))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("robot_name,name", CASES)
def test_tiers_match_interpreter(robot_name, name, dtype):
    p = _trace(robot_name, name)
    harness.validate(harness.build([(p, cuda.Variant("thread"))] + [(p, v) for v in COOPERATIVE], dtype))


@pytest.mark.parametrize("robot_name,name", CASES)
def test_tiers_bit_identical_without_fma(robot_name, name):
    p = _trace(robot_name, name)
    built = harness.build([(p, cuda.Variant("thread"))] + [(p, v) for v in COOPERATIVE],
                          flags=("-O3", "-fmad=false"))
    x = harness.sample_inputs(p, 97)
    ref, _ = built.run(0, x)
    for i in range(1, len(built.cases)):
        got, _ = built.run(i, x)
        for k in ref:
            np.testing.assert_array_equal(got[k], ref[k], err_msg=f"{built.cases[i][1]} {k}")


@pytest.mark.parametrize("robot_name,name", [("panda", "rnea"), ("grid_go2", "crba"), ("fr3", "rnea_derivatives")])
def test_group_size_invariance(robot_name, name):
    p = _trace(robot_name, name)
    variants = ([cuda.Variant("thread")] + [cuda.Variant("split", w) for w in (1, 2, 4, 8)]
                + [cuda.Variant("warp", 32)] + [cuda.Variant("block", t) for t in (32, 64, 128, 256)]
                + [cuda.Variant("block", 128, threshold=t) for t in (1, 64)])
    # one header per variant: the scratch constants are per tier
    built = [harness.build([(p, v)], flags=("-O3", "-fmad=false")) for v in variants]
    x = harness.sample_inputs(p, 4096)
    ref, _ = built[0].run(0, x, block=128)
    for b in built[1:]:
        for batch in (1, 7, 33, 4096):
            xb = {k: v[:batch] for k, v in x.items()}
            got, _ = b.run(0, xb, block=64)
            for k in ref:
                np.testing.assert_array_equal(got[k], ref[k][:batch], err_msg=f"{b.cases[0][1]} batch {batch}")
    for block in (32, 256):
        got, _ = built[0].run(0, x, block=block)
        for k in ref:
            np.testing.assert_array_equal(got[k], ref[k])


@pytest.mark.parametrize("robot_name,name", [("grid_g1", "rnea"), ("panda", "rnea_derivatives")])
def test_large_trace_guard(robot_name, name):
    """Over the statement budget, every tier is split into __noinline__ stage functions; those
    builds must agree between ptxas -O0 and -O3."""
    p = _trace(robot_name, name)
    cases = [(p, cuda.Variant("thread"))] + [(p, v) for v in COOPERATIVE]
    fast = harness.build(cases, flags=("-O3", "-fmad=false"), budget=200)
    slow = harness.build(cases, flags=("-O3", "-fmad=false", "-Xptxas=-O0"), budget=200)
    harness.validate(fast)
    x = harness.sample_inputs(p, 64)
    for i in range(len(cases)):
        a, _ = fast.run(i, x)
        b, _ = slow.run(i, x)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k], err_msg=f"{cases[i][1]} {k}")


def test_tune_picks_a_validated_layout_and_caches_it():
    from cricket2 import tune

    p = _trace("panda", "rnea")
    best = tune.tune(p, 256, reps=5, refresh=True)
    assert best.ms > 0 and best.tier in cuda.TIERS
    assert tune.tune(p, 256) == best  # second call is the cached config
    harness.validate(harness.build([(p, best.variant)]))
