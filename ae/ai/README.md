# AI training data loading

End-to-end benchmark comparing I/O engines (MMap, io_uring, SCISSOR) on AI
training workloads (BERT, ResNet-50) whose samples are chosen by a
data-selection policy, producing small scattered reads.  A C++ driver
(`src/bench`) replays a read trace against a DLIO-generated dataset with
simulated compute between steps and reports throughput and IOP counts.

```
ai/
├── README.md
├── requirements.txt     Python packages for the pipeline scripts
├── src/                 bench driver and I/O engines (C++17, Makefile)
├── scripts/             data generation, trace extraction, benchmark orchestration
├── config/              workload configs (config/dlio/*.yaml for the DLIO pipeline)
├── data/
│   ├── results/dlio/        per-run JSON results read by the notebook
│   ├── results/dlio_sweep/  gap/batch sweep JSON results read by the notebook
│   └── traces/              sampled read traces (*_sampled.bin) read by the notebook
├── notebooks/nb_ai.ipynb
└── figures/             notebook output
```

## 1. Reproduce the figure from the shipped data

Requires the notebook environment described in [`../README.md`](../README.md)
(`ae/requirements.txt`).

```bash
cd ae/ai/notebooks
jupyter nbconvert --to notebook --execute --inplace nb_ai.ipynb
```

| Output | Content |
|---|---|
| `figures/ai_e2e_throughput.pdf` | Throughput of MMap, IOUring and SCISSOR (best `scissor_sr` gap) with IOP counts. BERT 0.08 / 0.15 / 0.35 GB/s, ResNet50 0.21 / 0.69 / 1.94 GB/s, 1.0M / 79K IOPs |
| `figures/ai_scissor_gap_sensitivity.pdf` | `scissor_sr` throughput vs. coalescing gap, with baselines |
| `figures/ai_scissor_gap_sensitivity_bars_auto.pdf` | `scissor_sr` throughput and unnecessary-read fraction vs. gap |
| `figures/ai_scissor_gap_sensitivity_bars.pdf` | gap sweep at fixed batch size (`data/results/dlio_sweep`) |
| `figures/ai_batch_sweep.pdf` | batch-size sweep at fixed gap (`data/results/dlio_sweep`) |

The notebook reads `../data/results/dlio` (`AI_RESULTS_DIR`), the sibling
`dlio_sweep` directory and `../data/traces`.  To plot newly measured results,
point `AI_RESULTS_DIR` at the new results directory (e.g. `../results/dlio`).

## 2. Prerequisites for rerunning the experiments

- Linux ≥ 5.11 (io_uring with `IORING_SETUP_SQPOLL` without extra privileges),
  GCC ≥ 10, CMake ≥ 3.14, make, rsync.
- liburing ≥ 2.6 (optional: the scissor-io build fetches and installs it when
  pkg-config does not find one), libaio (optional, enables the `libaio` engine).
- Python ≥ 3.11 and an MPI implementation providing `mpicc` (for `mpi4py`).
- Fast storage for the datasets: `bert_clustered` 16 GB, `resnet50_clustered`
  20 GB (`config/dlio/*.yaml`).  The test configs need about 10 GB each.
- Cold-cache runs: [vmtouch](https://github.com/hoytech/vmtouch), root, or
  passwordless `sudo` for `/proc/sys/vm/drop_caches`.

## 3. Build

### 3.1 scissor-io

From the repository root:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$PWD/_install -DSCISSOR_ENABLE_TESTS=OFF
cmake --build build -j
cmake --install build
export SCISSOR_PREFIX=$PWD/_install
```

The prefix provides `include/scissor/`, `lib/libscissor.a` and, when liburing
was built by scissor-io, `lib/liburing.a` and `include/liburing.h`.

### 3.2 bench driver

```bash
cd ae/ai/src
make -j SCISSOR_PREFIX=$SCISSOR_PREFIX
```

Expected:

```
[Makefile] libaio: ENABLED
[Makefile] io_uring: ENABLED (...)
[Makefile] scissor: ENABLED (.../lib/libscissor.a)
[Makefile] Built: bench
```

liburing is taken from `$SCISSOR_PREFIX/lib/liburing.a` if present, otherwise
from pkg-config (`PKG_CONFIG_PATH` for non-standard locations).  Without
`SCISSOR_PREFIX` the `scissor` and `scissor_sr` engines are not built.

Engines (`--engine`):

| Engine | I/O path |
|---|---|
| `mmap` | whole-file `mmap` + `madvise(MADV_RANDOM)`, thread pool copying from the mapping (page cache) |
| `io_uring` | liburing ring with `IORING_SETUP_SQPOLL` (`sq_thread_idle` 2000 ms), ring size = `--queue-depth`, one read per chunk |
| `scissor` | `scissor::Loader` over one `scissor::LiburingReader` per file, regressor-selected gap and batch size |
| `scissor_sr` | `scissor::Aggregator` + one shared `scissor::LiburingReader`, at most `--queue-depth` coalesced reads in flight |
| `pread`, `libaio` | thread-pool `pread` and Linux AIO |

`--o-direct` opens the dataset files with `O_DIRECT` (used for all engines
except `mmap` by `bench_dlio.py`).  `--force-gap N` fixes the coalescing gap
(in chunks) of the SCISSOR engines, and result names become `scissor_gN` /
`scissor_sr_gN`.

## 4. Python environment

```bash
cd ae/ai
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps "dlio_benchmark @ git+https://github.com/argonne-lcf/dlio_benchmark.git@8b280cb3562e2c3e250d406d1f4ce1680aa9b88d"
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

DLIO ([argonne-lcf/dlio_benchmark](https://github.com/argonne-lcf/dlio_benchmark))
is only used to generate the `indexed_binary` datasets. It
imports PyTorch (the CPU build is enough), not TensorFlow or DALI.  Check:

```bash
python3 -c "from dlio_benchmark.data_generator.indexed_binary_generator import IndexedBinaryGenerator; print('DLIO OK')"
```

`prepare_dlio.py` locates the DLIO Hydra configs in the installed package.
`DLIO_DIR=<dlio_benchmark checkout>` or `--dlio-config-path` override it.

## 5. Smoke test

Small synthetic workload (`config/test_small.yaml`: 4 files × 1 GB, 20 steps
× 64 reads), no DLIO required.  From `ae/ai`:

```bash
source .venv/bin/activate
SMOKE_DIR=/path/to/scratch/smoke_test          # 4 GB

python3 scripts/gen_trace.py   --config config/test_small.yaml
python3 scripts/gen_dataset.py --config config/test_small.yaml --outdir $SMOKE_DIR

for eng in mmap io_uring scissor scissor_sr; do
  src/bench \
      --trace    traces/test_small_trace.bin \
      --manifest $SMOKE_DIR/manifest.txt \
      --engine   $eng \
      --queue-depth 64 \
      --chunk-size 4096 \
      --compute-mean-ms 5.0 --compute-std-ms 1.0 --compute-dist normal \
      $( [ $eng != mmap ] && echo --o-direct ) \
      --report-json results/smoke_$eng.json
done

rm -rf $SMOKE_DIR
```

Each run prints a summary box (`Engine`, `Total requests`, `Throughput`,
`IOPS`, I/O-wait percentiles) and writes `results/smoke_<engine>.json`.

## 6. Full pipeline

The DLIO pipeline has four phases:

| Phase | Script | What it does |
|---|---|---|
| 1 | `prepare_dlio.py` | generate the DLIO `indexed_binary` dataset |
| 2 | `prepare_dlio.py` → `gen_trace_dlio.py` | extract the full read trace |
| 2.5 | `prepare_dlio.py` → `sample_trace.py` | apply the data-selection policy (`selection:` in the YAML) → `<workload>_sampled.bin` |
| 3 | `bench_dlio.py` | copy data to SSD (optional), evict the page cache, run `src/bench` per engine × run |
| 4 | `bench_dlio.py` | aggregate the JSON results, write summary plots to `plots/` |

Workloads (`config/dlio/`): `bert_clustered` (256 files × 8192 × 8 KB,
clustered selection), `resnet50_clustered` (32 files × 4096 × 160 KB,
per-class coreset selection).  `test_bert_clustered` and
`test_resnet50_clustered` are reduced versions for pipeline validation.
Queue depth, compute time per step and `O_DIRECT` come from the
`benchmark:` / `train:` sections of each YAML. The chunk size equals the
record size.

Paths are passed on the command line or through environment variables:

| Variable | Used by | Meaning |
|---|---|---|
| `PFS_ROOT` | `prepare_dlio.py --data-root`, `bench_dlio.py --pfs-root` | dataset root on shared storage |
| `SSD_ROOT` | `bench_dlio.py --ssd-root` | benchmark data directory (node-local SSD, or `$PFS_ROOT` with `--no-copy`) |
| `VMTOUCH_DIR` | `bench_dlio.py --vmtouch-dir` | directory containing `vmtouch` |
| `DLIO_DIR` | `prepare_dlio.py` | dlio_benchmark checkout (Hydra configs) |
| `SCISSOR_MODEL_CONFIG` | `scissor` engines | fitted `models.conf` for the regressor (see the scissor-io README) |

### 6.1 Generate data and traces

```bash
source .venv/bin/activate
export PFS_ROOT=/path/to/storage/scissor_data

python3 scripts/prepare_dlio.py \
    --workloads  bert_clustered resnet50_clustered \
    --data-root  $PFS_ROOT \
    --traces-dir $PFS_ROOT/traces
```

`--no-generate`, `--no-trace`, `--no-sample` skip phases whose outputs exist.

### 6.2 Benchmark

Engines `mmap`, `io_uring`, `scissor`, `scissor_sr`:

```bash
python3 scripts/bench_dlio.py \
    --workloads   bert_clustered resnet50_clustered \
    --pfs-root    $PFS_ROOT \
    --ssd-root    $SSD_ROOT \
    --traces-dir  $PFS_ROOT/traces \
    --vmtouch-dir $VMTOUCH_DIR \
    --runs 2
```

Gap sweep (`scissor_g{0,1,2,4,8,16}`, `scissor_sr_g{0,1,2,4,8,16}`), data
already in place:

```bash
python3 scripts/bench_dlio.py \
    --workloads  bert_clustered resnet50_clustered \
    --ssd-root   $SSD_ROOT --no-copy \
    --traces-dir $PFS_ROOT/traces \
    --vmtouch-dir $VMTOUCH_DIR \
    --runs 2 --gap-sweep
```

To benchmark directly on the shared storage, pass `--ssd-root $PFS_ROOT
--no-copy`.  Results go to `results/dlio/<workload>_<engine>_r<N>.json`
(`--results-dir`), summary plots to `plots/` (`--outdir`).  The fixed-batch
gap sweep (`--gap-batch-sweep`) and the fixed-gap batch sweep
(`--batch-sweep`) write `scissor_sr_b<M>mb_g<N>` and `scissor_sr_g<N>_b<M>mb`
results. Run them with `--results-dir results/dlio_sweep` for the notebook's
sweep figures.  `python3 scripts/bench_dlio.py --no-run` re-plots existing
results.

`scripts/run_resnet_validation.sh --data-root DIR [--workload NAME]
[--vmtouch PATH]` runs phases 1 to 3 for one workload and prints a summary table.

### 6.3 PBS (Polaris)

`scripts/submit_prod_polaris.sh` runs phases 1 to 4 in one job. Submit it from
`ae/ai` with the site account, queue and file systems on the command line:

```bash
qsub -A <account> -q <queue> -l filesystems=<fs> \
     -v PFS_ROOT=$PFS_ROOT,VENV=$PWD/.venv,VMTOUCH_DIR=$VMTOUCH_DIR,GAP_SWEEP=true \
     scripts/submit_prod_polaris.sh
```

The header of the script lists all variables (`PFS_BENCH`, `SKIP_PREPARE`,
`N_RUNS`, `SSD_ROOT`, `MODULES_SCRIPT`, sweep switches).
`scripts/setup_polaris.sh` sets up the same environment for interactive jobs.

### 6.4 Plot the new results

In the third code cell of `notebooks/nb_ai.ipynb` set
`AI_RESULTS_DIR = "../results/dlio"`, then:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace nb_ai.ipynb
```

The notebook reads the sweep results from `<results>/../dlio_sweep` and the
sampled traces from `<results>/../../traces`. With
`AI_RESULTS_DIR = "../results/dlio"` copy `$PFS_ROOT/traces/*_sampled.bin`
to `ae/ai/traces/`.

## 7. Other scripts

| Script | Purpose |
|---|---|
| `scripts/gen_trace.py`, `scripts/gen_dataset.py` | synthetic traces/datasets for `config/{test_small,bert,resnet50,unet3d}.yaml` |
| `scripts/run_all.sh` | synthetic workloads × `pread`/`libaio`/`io_uring`/`scissor` (`DATA_ROOT`, `N_RUNS`) |
| `scripts/gen_dlio_dataset.py` | DLIO-format dataset without DLIO (`config/dlio/bert_packed.yaml`) |
| `scripts/plot_results.py` | plots for `run_all.sh` results |
