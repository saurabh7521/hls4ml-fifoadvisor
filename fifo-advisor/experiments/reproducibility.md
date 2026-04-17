# Reproducing the FIFOAdvisor Experiments

This document outlines the steps to reproduce the results in the FIFOAdvisor paper (design artifacts, raw data, tables, figures). Commands are provided as
examples and should be launched from the repository root inside the conda environment described below.

## Prerequisites

- Linux host with Vitis HLS 2024.1 installed and licensed.
- Python environment defined in `environment.yml` (includes LightningSim Python
  bindings, pandas, pymoo, etc.).

## Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate fifo-advisor
```

Install FIFOAdvisor in editable mode to expose the CLI and package modules:

```bash
python -m pip install -e .
```

Copy the experiment `.env` template and fill in absolute paths to where you want the experiment scripts to set up the working directories with the test designs:

```bash
cp experiments/.env.template experiments/.env
```

Required keys:

- `DIR_PRE_SYNTH`: directory that will contain synthesized Vitis HLS projects, used for simulation and FIFO optimization.
- `DIR_COSIM`: directory where designs with co-simulation results will be created.

## Prepare the Test Designs (`experiments/pre_synth_all_designs`)

Run the pre-synthesis script to build Vitis HLS solutions for each required design.
This script copies each test case, runs C-simulation, and then HLS synthesis.

```bash
python experiments/pre_synth_all_designs/pre_synth.py
```

The resulting projects should live under `DIR_PRE_SYNTH`.

## 4. Cache LightningSim Traces

Most experiments reuse LightningSim trace caches for faster evaluation. After
pre-synthesis completes, run:

```bash
python experiments/setup_cached_traces/run.py
```

This script walks every project in `DIR_PRE_SYNTH`, runs LightningSim once, and
stores the resulting `trace.pkl` files inside each HLS solution directory.

## Main Optimizer Experiment (`experiments/main_run_v0`)

These scripts generate the Pareto trade-off data shown in the primary figures.
They assume cached traces are available.

1. Update `experiments/main_run_v0/exp.py` with any design filters or solver
   settings required for the paper snapshot (the defaults match the
   submission).
2. Execute the batch run:

```bash
python experiments/main_run_v0/exp.py
```

Raw CSV data and LaTeX tables are written to `experiments/main_run_v0/data/`.

Then run the analysis script to generate LaTeX tables (under `experiments/main_run_v0/data/`) and figures (under `experiments/main_run_v0/figures/`):

```bash
python experiments/main_run_v0/stats.py
python experiments/main_run_v0/stats_2.py
```

## Data-Dependent Control Flow Design Experiment with FlowGNN Designs (`experiments/main_run_v0_pna`)

This directory mirrors the main experiment but focuses on the HLS design from FlowGNN that implements a PNA graph neural network accelerator.

You will need to pre-synthesize the FlowGNN designs first using the one-off script:

```bash
python experiments/single_synth/synth.py
```

Then run the experiment:

```bash
python experiments/main_run_v0_pna/exp.py
```

Like before, raw CSV data and LaTeX tables are written to `experiments/main_run_v0_pna/data/`.

Then run the analysis script to generate LaTeX tables (under `experiments/main_run_v0_pna/data/`) and figures (under `experiments/main_run_v0_pna/figures/`):

```bash
python experiments/main_run_v0_pna/stats.py
python experiments/main_run_v0_pna/stats_2.py
```

### Co-Simulation vs. LightningSim Runtime Comparison

The `cosim_numbers_compare` folder produces the runtime comparison table and
related figures.

First run co-simulation on a subset of designs to collect the co-simulation runtime.

```bash
python experiments/cosim_numbers_compare/run.py
```

Then collect FIFOAdvisor runtime data as well.

```bash
python experiments/cosim_numbers_compare/collect_fifo_advisor_numbers.py
```

Finally, run the analysis script to generate the comparison in a LaTeX table.

```bash
python experiments/cosim_numbers_compare/analyze.py
```

The script produces a LaTeX table under `experiments/cosim_numbers_compare/figures/`.

### Runtime Scaling Study (`experiments/runtime_exp_v2`)

This experiment measures per-optimizer runtime for different FIFOAdvisor configurations.

```bash
python experiments/runtime_exp_v2/run.py
```

Then to analyze the results and generate figures:

```bash
python experiments/runtime_exp_v2/plot.py
```

The scripts generate data under `experiments/runtime_exp_v2/data/` and figures under `experiments/runtime_exp_v2/figures/`.
