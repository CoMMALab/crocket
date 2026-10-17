"""cricket2: a trace compiler for GRiD/GLASS rigid-body kernels."""

from __future__ import annotations

__version__ = "0.1.0"

from . import algorithms  # noqa: F401  (registers the algorithms)
from . import cuda
from .cuda import DEFAULT_VARIANTS, Variant
from .robot import Robot
from .trace import ALGORITHMS, Program, algorithm


def trace(robot: Robot, name: str, **options) -> Program:
    """Trace algorithm ``name`` against ``robot``."""
    return ALGORITHMS[name].trace(robot, **options)


def generate(robot: Robot, algorithms: dict[str, dict], variants=DEFAULT_VARIANTS, name: str | None = None,
             kinematics: dict | None = None, budget: int = cuda.BUDGET) -> str:
    """One CUDA header for ``robot``: each algorithm (name -> trace options) at each tier in
    ``variants`` (or name -> variants, e.g. from :func:`cricket2.tune.tune`), plus cricket's
    float kinematics interface when ``kinematics={"frames": [...], "end_effector": ...}``."""
    sections = []
    for alg, options in algorithms.items():
        vs = variants[alg] if isinstance(variants, dict) else variants
        sections.append(cuda.emit(trace(robot, alg, **options), list(vs), budget))
    if kinematics is not None:
        sections.append(cuda.kinematics(robot, kinematics.get("frames", ()), kinematics.get("end_effector"),
                                        budget))
    return cuda.header(robot, sections, name)


__all__ = ["ALGORITHMS", "DEFAULT_VARIANTS", "Program", "Robot", "Variant", "algorithm", "generate", "trace"]
