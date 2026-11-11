"""Blackwell (sm_100 / sm_120) guards that need no GPU: see blackwellfix.md.

    python -m pytest tests/test_blackwell.py
"""

from __future__ import annotations

import re
import subprocess

import pytest

import crocket
from crocket import cuda, harness, tune
from robots import robot


def _stage_bodies(src: str, tier: str) -> list[str]:
    return re.findall(rf"__noinline__ void \w+_{tier}_s\d+\(.*?\)\n    \{{\n(.*?)\n    \}}\n", src, re.S)


@pytest.mark.parametrize("tier,units", [("warp", 32), ("block", 128)])
def test_cooperative_stages_reconverge(tier, units):
    """Every warp/block stage function reconverges before returning: without it ptxas for sm_120
    reuses uniform-register pointer arguments as scratch while other lanes still read them."""
    p = crocket.trace(robot("panda"), "rnea")
    src = cuda.emit(p, [cuda.Variant(tier, units)], budget=200)
    bodies = _stage_bodies(src, tier)
    assert bodies, "budget=200 should split rnea into stage functions"
    assert all(b.rstrip().endswith("__syncwarp();") for b in bodies)


def _fake_nvcc(monkeypatch, returncode: int, stdout: str = "", stderr: str = ""):
    def run(cmd, **kw):
        if "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "Cuda compilation tools\nBuild cuda_test\n", "")
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    monkeypatch.setattr(harness, "ARCH", "-arch=native")
    monkeypatch.setattr(harness.subprocess, "run", run)
    harness.arch_flags.cache_clear()
    harness.nvcc_version.cache_clear()


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    harness.arch_flags.cache_clear()
    harness.nvcc_version.cache_clear()


def test_native_resolves_to_concrete_arch(monkeypatch):
    _fake_nvcc(monkeypatch, 0, stdout='#$ cicc ... -arch compute_120 ...\n#$ ptxas -arch=sm_120 ...\n')
    assert harness.arch_flags() == ("-arch=sm_120",)


def test_native_with_two_gpus(monkeypatch):
    _fake_nvcc(monkeypatch, 0, stdout="-arch compute_86 ... -arch compute_120 ...")
    assert harness.arch_flags() == ("-gencode=arch=compute_120,code=sm_120",
                                    "-gencode=arch=compute_86,code=sm_86")


def test_old_nvcc_on_blackwell_is_a_clear_error(monkeypatch):
    _fake_nvcc(monkeypatch, 1, stderr="nvcc fatal   : Unsupported gpu architecture 'compute_120'\n")
    with pytest.raises(RuntimeError, match=r"sm_120.*CUDA 12\.8"):
        harness.arch_flags()


def test_explicit_arch_is_kept(monkeypatch):
    monkeypatch.setattr(harness, "ARCH", "-arch=sm_120")
    harness.arch_flags.cache_clear()
    assert harness.arch_flags() == ("-arch=sm_120",)


def test_tune_cache_key_names_the_gpu(monkeypatch):
    """A launch config tuned on one GPU is not reused on another."""
    p = crocket.trace(robot("panda"), "rnea")
    monkeypatch.setattr(harness, "nvcc_version", lambda: "Build cuda_test")
    keys = set()
    for arch in ("-arch=sm_86", "-arch=sm_120"):
        monkeypatch.setattr(harness, "arch_flags", lambda arch=arch: (arch,))
        keys.add(tune._key(p, 1024, "float32", {}))
    assert len(keys) == 2


def test_nvcc_failure_without_stderr_is_a_clear_error(monkeypatch):
    _fake_nvcc(monkeypatch, 1)
    with pytest.raises(RuntimeError, match="nvcc exited with status 1"):
        harness.arch_flags()
