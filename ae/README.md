# Artifact evaluation

This directory reproduces the SCISSOR evaluation results from the measurement
data shipped with the repository, and contains the code to rerun the experiments.

| Directory | Workload | Contents |
|---|---|---|
| [`hacc/`](hacc/README.md) | HPC reproducibility analytics (state-diff on HACC checkpoints) | notebooks, logs, figures |
| [`ai/`](ai/README.md) | AI training data loading | benchmark sources, scripts, configs, results, notebook, figures |

## Figure index

| Result | Notebook | Output |
|---|---|---|
| Read throughput vs. number of IOPs | `hacc/notebooks/nb_motivation.ipynb` | `hacc/figures/PFS_io_bench.pdf` |
| Coalescing trade-off (IOP count and extra data vs. gap) | `hacc/notebooks/nb_motivation.ipynb` | `hacc/figures/iop_wastedreads_plot.pdf` |
| Inter-offset distance by IOP size and error margin | `hacc/notebooks/nb_motivation.ipynb` | `hacc/figures/np1000_violin_err_oft.pdf` |
| End-to-end throughput, dense and sparse non-contiguous reads | `hacc/notebooks/nb_statediff.ipynb` | `hacc/figures/e2e_plot_1e-7.pdf`, `hacc/figures/e2e_plot_1e-3.pdf` |
| Impact of the coalescing gap | `hacc/notebooks/nb_statediff.ipynb` | `hacc/figures/breakdowngap_multi_33554432.pdf` |
| Impact of the batch size | `hacc/notebooks/nb_statediff.ipynb` | `hacc/figures/breakdownbs_chunks_1.pdf` |
| Regressor vs. exhaustive search | `hacc/notebooks/nb_statediff.ipynb` | `hacc/figures/regressor_margin_1e-7.pdf` |
| Submission and completion queue depth | `hacc/notebooks/nb_statediff.ipynb` | `hacc/figures/qdepth_diffio_vs_uring_cq512_4096.pdf`, `hacc/figures/cqdepth_by_qdepth_dual_chunks_new.pdf` |
| Weak scaling | `hacc/notebooks/nb_statediff.ipynb` | `hacc/figures/nnode_scalability.pdf` |
| I/O and compute models | `hacc/notebooks/nb_perf_model.ipynb` | printed fit parameters and R² |
| AI data loading throughput | `ai/notebooks/nb_ai.ipynb` | `ai/figures/ai_e2e_throughput.pdf` |

## Reproducing the figures

`requirements.txt` lists the pinned Python packages for all notebooks
(Python 3.9 to 3.12):

```bash
cd ae
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Then execute the notebooks as described in [`hacc/README.md`](hacc/README.md) and
[`ai/README.md`](ai/README.md).

## Rerunning experiments

- HACC: see [`ae/hacc/README.md`](hacc/README.md)
- AI: see [`ae/ai/README.md`](ai/README.md)
