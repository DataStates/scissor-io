# HACC reproducibility analytics: figures

This directory reproduces the HACC reproducibility-analytics results: read
throughput of the storage stack, the coalescing opportunity in HACC offsets, the
end-to-end throughput of SCISSOR against MMap and IOUring, the sensitivity to
coalescing gap, batch size and queue depths, the Regressor accuracy, weak scaling,
and the I/O and compute model fits.

The notebooks read the measurement logs shipped in `data/` and write the figures
to `figures/`. The executed notebooks, including their outputs, are committed.

## Figure map

| Result | Notebook | Section | Output (`figures/`) |
|---|---|---|---|
| Read throughput vs. number of IOPs | `nb_motivation.ipynb` | Plots (cells 14-15) | `PFS_io_bench.pdf` |
| Coalescing trade-off | `nb_motivation.ipynb` | Plots (cell 17) | `iop_wastedreads_plot.pdf` |
| Inter-offset distance | `nb_motivation.ipynb` | Plots (cell 16) | `np1000_violin_err_oft.pdf` |
| End-to-end throughput, dense ($10^{-7}$) | `nb_statediff.ipynb` | 3.1 Impact of DiffIO Engine on Performance | `e2e_plot_1e-7.pdf` |
| End-to-end throughput, sparse ($10^{-3}$) | `nb_statediff.ipynb` | 3.1 Impact of DiffIO Engine on Performance | `e2e_plot_1e-3.pdf` |
| Impact of the coalescing gap | `nb_statediff.ipynb` | 3.2 Impact of Gaps | `breakdowngap_multi_33554432.pdf` |
| Impact of the batch size | `nb_statediff.ipynb` | 3.3 Impact of Block Sizes | `breakdownbs_chunks_1.pdf` |
| Regressor vs. exhaustive search | `nb_statediff.ipynb` | 3.4 Regressor Error Margin | `regressor_margin_1e-7.pdf` |
| Submission and completion queue depth | `nb_statediff.ipynb` | 3.4 Queue Depth and Completion Queue Depth | `qdepth_diffio_vs_uring_cq512_4096.pdf`, `cqdepth_by_qdepth_dual_chunks_new.pdf` |
| Weak scaling | `nb_statediff.ipynb` | 3.5 Scalability Analysis | `nnode_scalability.pdf` |
| I/O and compute models | `nb_perf_model.ipynb` | 2.1 IO Model, 2.2 Compute Model | printed fit parameters and R² |

In the logs and notebooks, `DiffIO` is the name used for the SCISSOR engine.

## Running

Python 3.9 to 3.12. From the `ae/` directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cd hacc/notebooks
jupyter nbconvert --to notebook --execute --inplace \
    nb_motivation.ipynb nb_perf_model.ipynb nb_statediff.ipynb
```

All paths are relative to `notebooks/`. The three notebooks run in about a minute
on a laptop and regenerate every PDF in `figures/`. The compute-model fit in
`nb_perf_model.ipynb` keeps the best of 1000 random train/validation splits, so its
printed R² can vary in the third decimal between executions.

## Data

All measurements were collected on Polaris (ALCF) with Lustre as the parallel file
system. HACC simulations of 0.5B (`np796-500mil`), 1B (`np1000-01bil`) and 17B
(`np2592-17bil`) particles were used. The HACC checkpoints themselves are not
distributed. In the logs, the `File` column is relative to the HACC directory
(`hacc/<simulation>/<run>/...`).
`data/` contains only the files the notebooks read (≈26 MB).

| Path | Content | Used by |
|---|---|---|
| `data/motivation/benchmark_data/{contiguous,random}_io_{1,2}_{pfs,ssd}.csv` | Read microbenchmark: throughput vs. number of IOPs for 1 GB and 2 GB total reads, per storage target | `nb_motivation` |
| `data/motivation/benchmark_data/random_chks_io__{ssd,pfs,hyb}.csv` | I/O calibration points (IOP count × IOP size) | `nb_perf_model` |
| `data/motivation/benchmark_data/compute_benchmark.csv` | Compute-kernel timings vs. work size and thread count | `nb_perf_model` |
| `data/motivation/offsets/np1000/` | Offsets read by state-diff for one 1B-particle checkpoint pair, per IOP size and error margin | `nb_motivation` |
| `data/evaluation/e2e/{MMap,IOUring,DiffIO}/{np796-500mil,np1000-01bil}/` | Per-rank end-to-end state-diff logs, 0.5B and 1B particles | `nb_statediff` |
| `data/evaluation/breakdown/` | SCISSOR sweep over coalescing gap and batch size, 1B particles | `nb_statediff` |
| `data/evaluation/qdvar/{diffio,liburing}/` | Submission/completion queue-depth sweep, 1B particles | `nb_statediff` |
| `data/scalability/{DiffIO,IOUring}/node-N/` | Weak scaling on N nodes (2 ranks per node), 17B particles | `nb_statediff` |

## Rerunning experiments

The job scripts in `scripts/` run `eval_statediff` and the benchmarks of the
`scissor-io` branch of
[state-diff](https://github.com/DataStates/state-diff/tree/scissor-io), built
against scissor-io (see the state-diff README). They
write CSV logs in the layout of `data/` under `$OUT_DIR` (default `./logs`), so
the notebooks can read them by pointing their data directories there.

| Script | Results | Main variables |
|---|---|---|
| `scripts/e2e.sh` | End-to-end throughput | `ENGINE=scissor\|iouring\|mmap`, `EXP_ID`, `TYPE=offline\|ssd`, `CSIZE` |
| `scripts/breakdown.sh` | Coalescing gap × batch size, Regressor vs. exhaustive search | `EXP_ID`, `TYPE`, `CSIZE` |
| `scripts/qdepth.sh` | Submission and completion queue depth | `ENGINE=scissor\|iouring`, `EXP_ID`, `TYPE` |
| `scripts/scalability.sh` | Weak scaling (2 ranks per node) | `NNODES`, `EXP_ID` |
| `scripts/read_bench.sh` | Read throughput vs. number of IOPs (`SWEEP=io`), I/O model calibration points (`SWEEP=calib`) | `SWEEP`, `EXP_ID`, `TYPE=pfs\|ssd\|hyb`, `NREPS` |
| `scripts/compute_bench.sh` | Compute-kernel timings vs. work size and thread count | `THREADS`, `NREPS`, `MAX_CHUNKS` |
| `scripts/offsets.sh` | Offsets of the chunks flagged by the tree comparison, per chunk size and error margin | `EXP_ID`, `TYPE` |

All scripts read `scripts/common.sh`, which takes the site settings from the
environment: `HACC_DIR` (one directory per simulation, each with `run1/` and
`run2/` holding `m000p.mpirestart-combined-*.dat`), `STATEDIFF_BUILD`,
`OUT_DIR`, `VMTOUCH_DIR` (page-cache eviction between runs), `SCRATCH_DIR`
(`TYPE=ssd`), `MODULES_SCRIPT`, `LAUNCH` (MPI launch prefix), `NPROCS`,
`PROCS_PER_NODE`, `CORES_PER_RANK`, `THREADS_PER_RANK` and `NITERS`.

Submit from `ae/hacc/scripts`, giving the account, queue and filesystems on the
command line:

```bash
cd ae/hacc/scripts
qsub -A <account> -q <queue> -l filesystems=<filesystems> \
     -v HACC_DIR=<dir>,STATEDIFF_BUILD=<dir>,ENGINE=iouring,EXP_ID=np1000-01bil,TYPE=offline,CSIZE=3 \
     e2e.sh
```

The scripts also run outside PBS, e.g. `HACC_DIR=... STATEDIFF_BUILD=...
LAUNCH="mpirun -np 2" bash e2e.sh`.

### Running the scripts on synthetic checkpoints

The HACC checkpoints are not distributed. To exercise the scripts, generate
checkpoint pairs with the state-diff data generator and lay them out like a
simulation directory: `data_generator ... gen_` writes `gen_0.dat` (initial data)
and `gen_1.dat` (the same data with `--num-changes` perturbed elements), which
play the roles of run 1 and run 2.

```bash
export STATEDIFF_BUILD=<state-diff build directory>
export HACC_DIR=$PWD/synthetic-hacc
sim=$HACC_DIR/synthetic
mkdir -p $sim/run1 $sim/run2
# two checkpoints per run (simulation rank 0, checkpoints 10 and 20),
# 2^25 floats (128 MiB) each
for ckpt in 10 20; do
    $STATEDIFF_BUILD/src/tools/data_generator -t float --data-len 33554432 \
        -n 2 -s $ckpt -e 1e-4 --num-changes 3000 gen_
    mv gen_0.dat $sim/run1/m000p.mpirestart-combined-0-$ckpt.dat
    mv gen_1.dat $sim/run2/m000p.mpirestart-combined-0-$ckpt.dat
done

cd ae/hacc/scripts
export OUT_DIR=$PWD/logs LAUNCH="mpirun -np 1" NITERS=1 EXP_ID=synthetic
for engine in scissor iouring mmap; do
    ENGINE=$engine CSIZE=1 bash e2e.sh
done
CSIZE=1 bash breakdown.sh
ENGINE=scissor bash qdepth.sh
ENGINE=iouring bash qdepth.sh
bash offsets.sh
NREPS=1 CALIB_SIZES="16777216 33554432" SWEEP=calib TYPE=pfs bash read_bench.sh
NREPS=1 THREADS=2 MAX_CHUNKS=1024 bash compute_bench.sh
```

`scalability.sh` with `NNODES` nodes reads the checkpoints of simulation ranks
`0` to `NNODES-1` (`m000p.mpirestart-combined-<r>-<ckpt>.dat`) and spreads them
round-robin over its `NNODES × PROCS_PER_NODE` processes, so the synthetic
directory above already serves `NNODES=1`:
`NNODES=1 PROCS_PER_NODE=2 LAUNCH="mpirun -np 2" bash scalability.sh`.
Page-cache eviction between runs uses `vmtouch` (`VMTOUCH_DIR`).
