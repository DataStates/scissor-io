#!/usr/bin/env bash
#PBS -l select=2
#PBS -l place=scatter:exclhost
#PBS -l walltime=07:00:00
#PBS -o breakdown.out
#PBS -e breakdown.err

# ========================================================================
# breakdown.sh -- SCISSOR with fixed coalescing gap and block size.
#
# Sweeps gap {0 1 2 4 8 16} x block size {128 256 512 1024 2048 4096}
# (block = N x chunk size x threads) at error tolerance 1e-7, with the
# performance model disabled (--tuning fixed).
#
# EXP_ID=np1000-01bil
# TYPE=offline | ssd
# CSIZE=1 (4/8 KB) | 2 (16/32 KB) | 3 (4/8/16/32 KB)
#
# qsub -A <account> -q <queue> -l filesystems=<filesystems> \
#      -v HACC_DIR=<dir>,STATEDIFF_BUILD=<dir>,EXP_ID=np1000-01bil,TYPE=offline,CSIZE=3 breakdown.sh
# ========================================================================

# PBS runs a copy of this script: common.sh is found in the submission
# directory (ae/hacc/scripts) or in SCRIPT_DIR
. "${SCRIPT_DIR:-${PBS_O_WORKDIR:-$(dirname "${BASH_SOURCE[0]}")}}/common.sh"

EXP_ID=${EXP_ID:-np1000-01bil}
TYPE=${TYPE:-offline}
CSIZE=${CSIZE:-3}

start_time=$SECONDS

echo "------------------------------------------------------------------------"
echo " Environment Setup "
echo "------------------------------------------------------------------------"
set_run_dirs $EXP_ID $TYPE

error_tols=(0.0000001)
comp_batch_sizes=(128 256 512 1024 2048 4096)
gaps=(0 1 2 4 8 16)

if [[ $CSIZE == "1" ]]; then
    chunk_sizes=(4096 8192)
elif [[ $CSIZE == "2" ]]; then
    chunk_sizes=(16384 32768)
else
    chunk_sizes=(4096 8192 16384 32768)
fi

outdir="$OUT_DIR/evaluation/breakdown"
outname="$outdir/breakbsgap"
mkdir -p $outdir
cmd_flags="--level 13 -t float -s $floc --tuning fixed $kokkos_flags"

var_name=combined
run0_full_files=( $(ls ${run0_dir}/m000p.mpirestart-${var_name}-*.dat) )
run1_full_files=( $(ls ${run1_dir}/m000p.mpirestart-${var_name}-*.dat) )
echo "Run 0 data full files: ${run0_full_files[@]}"
echo "Run 1 data full files: ${run1_full_files[@]}"

for chunk_size in "${chunk_sizes[@]}";
do
  for tol in "${error_tols[@]}";
  do
    echo "========================================================================"
    echo " Data type Float"
    echo " Chunk size $chunk_size"
    echo " Error tolerance $tol"
    echo "========================================================================"
    create_trees $chunk_size $tol $outname "$cmd_flags"
    for iter in $(seq 1 $niters)
    do
        for r_size in "${comp_batch_sizes[@]}"; do
            for gap in "${gaps[@]}";  do
                echo "------------------------------------------------------------------------"
                echo " ($iter) Break BS&GAP compare-tree : Compare Checkpoints (Gap = $gap, Worksize = $r_size)"
                echo "------------------------------------------------------------------------"
                run_cmd "$cmd_env $eval_statediff -c $chunk_size -g $gap -b $r_size --error $tol $data_files $tree_files -r $outname $cmd_flags"
            done
        done
    done
  done
done
rm -f ${run0_dir}/*.compare-tree ${run1_dir}/*.compare-tree

elapsed_time=$(( SECONDS - start_time ))
echo "Script executed in $elapsed_time seconds."
