#!/usr/bin/env bash
# Every benchmark, every method, on one free GPU; then the tables in README.md.
#   bash benchmarks/run_all.sh            # GPU picked automatically (refuses if all are busy)
#   GPU=2 bash benchmarks/run_all.sh
# Envs: grim (JAX: GRiM, crocket, pyroki, frax, Pinocchio), curobo (PyTorch). pRRTC is built
# from PRRTC_DIR (default ~/Work/prrax/pRRTC) with nvcc.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRIM="$(cd "${HERE}/../../.." && pwd)"
CONDA="${CONDA_ROOT:-$HOME/miniconda3}/envs"
JAXPY="${CONDA}/grim/bin/python"
TORCHPY="${CONDA}/curobo/bin/python"
FRAX_DIR="${FRAX_DIR:-$HOME/Work/pyroffi/baselines/frax}"

if [[ -z "${GPU:-}" ]]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
        awk -F', ' '$2 < 500 && $3 < 5 {print $1; exit}')
  [[ -n "${GPU}" ]] || { echo "all GPUs busy:"; nvidia-smi; exit 1; }
fi
echo "GPU ${GPU}"
# True float32 matmuls: JAX's default TF32 on Ampere costs frax 1e-3..4e-2 relative error.
export CUDA_VISIBLE_DEVICES="${GPU}" XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONPATH="${GRIM}/bindings:${HERE}/../python:${FRAX_DIR}:${HERE}/fk"
cd "${HERE}"

JAX_PLATFORMS=cpu "${JAXPY}" fk/bench_fk.py prepare
JAX_PLATFORMS=cpu "${JAXPY}" fk/bench_fkcc.py prepare
JAX_PLATFORMS=cpu "${JAXPY}" ik/bench_ik.py prepare
JAX_PLATFORMS=cpu "${JAXPY}" dynamics/bench_dynamics.py prepare

for m in prrtc pyroki grim crocket; do "${JAXPY}" fk/bench_fk.py "$m"; done
"${TORCHPY}" fk/bench_fk.py curobo
for m in prrtc pyroki grim crocket crocket_fused; do "${JAXPY}" fk/bench_fkcc.py "$m"; done
"${TORCHPY}" fk/bench_fkcc.py curobo
for m in grim crocket_ls_mixed pyroki; do "${JAXPY}" ik/bench_ik.py "$m"; done
for m in grid crocket frax; do "${JAXPY}" dynamics/bench_dynamics.py "$m"; done

"${JAXPY}" tables.py
