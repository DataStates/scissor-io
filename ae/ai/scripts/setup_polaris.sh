#!/bin/bash
# setup_polaris.sh: environment for interactive runs on a Polaris compute node.
#
# Source from the ae/ai directory:
#   source scripts/setup_polaris.sh
#
# Variables (set before sourcing):
#   MODULES_SCRIPT        optional site script that loads compiler/MPI modules
#   VENV                  Python virtualenv with requirements.txt + dlio_benchmark
#   SSD_ROOT              node-local SSD directory (default /local/scratch/scissor_bench)
#   SCISSOR_MODEL_CONFIG  optional fitted models.conf for the SCISSOR regressor

AI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -n "$MODULES_SCRIPT" ]] && . "$MODULES_SCRIPT"
module load cray-python/3.11.7
[[ -n "$VENV" ]] && source "$VENV/bin/activate"
cd "$AI_DIR"
[[ -n "$SCISSOR_MODEL_CONFIG" ]] && export SCISSOR_MODEL_CONFIG
export MPICH_GPU_SUPPORT_ENABLED=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/lib
export SSD_ROOT=${SSD_ROOT:-/local/scratch/scissor_bench}
mkdir -p "$SSD_ROOT"
