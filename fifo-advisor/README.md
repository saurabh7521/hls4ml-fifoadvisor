# FIFOAdvisor: A DSE Framework for Automated FIFO Sizing of High-Level Synthesis Designs

FIFOAdvisor is a framework that optimizes FIFO depths in high-level synthesis designs, specifically targeting Vitis HLS. It can search for optimal FIFO depths in seconds to minimize design latency and BRAM usage in a way that also avoids deadlocks. FIFOAdvisor does this by using LightningSim, a fast cycle-accurate simulator for HLS designs, to perform runtime analysis and explore different FIFO size configurations in seconds to perform rapid design space exploration (DSE).

## Installation and Usage

FIFOAdvisor can be set up in a Conda environment to manage dependencies (including LightningSim).

Create and activate a conda environment using the supplied `environment.yml`, then
install FIFOAdvisor into that environment (from GitHub or a local checkout).

```bash
conda env create -f environment.yml
conda activate fifo-advisor

pip install --no-deps git+https://github.com/sharc-lab/fifo-advisor.git

# or, for a local checkout...
# pip install --no-deps .
```

We also recommend `mamba` or `micromamba` as a drop-in replacement for `conda` for better tooling.

From here, you should be able to run `fifo-advisor` in the activated environment.

```bash
fifo-advisor --help
```

```text
usage: fifo-advisor <solution_dir> [options]

A tool for optimizing FIFO depths in a high-level synthesis design using different optimization algorithms and fast co-simulation.

positional arguments:
  solution_dir

options:
  -h, --help            show this help message and exit
  --solver {random,group-random,heuristic,sa,group-sa}
                        Optimization strategy to run (default: random)
  --n-samples N_SAMPLES
                        Sample count for random-based solvers.
  --seed SEED           Random seed for stochastic solvers.
  --maxfun MAXFUN       Evaluation budget for simulated annealing solvers.
  --n-scaling-factors N_SCALING_FACTORS
                        Number of dual objective scaling factors for simulated annealing solvers.
  --round-type {floor,ceil,fix,trunc,round,rint}
                        Rounding mode for simulated annealing solvers.
  --init-with-largest   Start simulated annealing from the largest FIFO depths.
  --output OUTPUT       Write optimizer evaluations to this JSON file (default: fifo_advisor_results.json).
```

## Quick Start

To run FIFOAdvisor on a specific Vitis HLS design, you must first have synthesized the design with Vitis HLS and have the corresponding solution directory. The design also needs a testbench with a representative workload, marked as testbench files in the Vitis HLS project.

Then, you can run FIFOAdvisor on the solution directory as shown below with different solvers and options:

```bash
# Default random search (RandomSearchOptimizer) with n_samples=1000 and seed=7
fifo-advisor <solution_dir>

# Random search with explicit sample count and seed
fifo-advisor <solution_dir> --solver random --n-samples 1000 --seed 7

# Grouped random search (GroupRandomSearchOptimizer)
fifo-advisor <solution_dir> --solver group-random --n-samples 2000 --seed 1234

# Heuristic search (HeuristicOptimizer)
fifo-advisor <solution_dir> --solver heuristic

# Discrete simulated annealing (DiscreteSimulatedAnnealingOptimizer)
fifo-advisor <solution_dir> --solver sa --maxfun 100 --n-scaling-factors 8 --round-type rint

# Grouped discrete simulated annealing (GroupedDiscreteSimulatedAnnealingOptimizer)
fifo-advisor <solution_dir> --solver group-sa --maxfun 100 --n-scaling-factors 8 --round-type rint --init-with-largest --output custom_results.json
```

Every run emits the evaluated design points as JSON; by default the data is
written to `fifo_advisor_results.json`. Supply `--output <path>` for a custom data file path.

## Reproducibility of Paper Results

You can reproduce the results in our paper by following the instructions in the `experiments/reproducibility.md` file. This will guide you through running the scripts in the `experiments` directory to reproduce the results in our paper.

## Issues or Questions

If you encounter any issues or have questions, please open an issue on the FIFOAdvisor GitHub repository or reach out to the authors directly.

We are happy to help!

## Citation

If you use FIFOAdvisor in your work or research, please cite:

```bibtex
@inproceedings{fifoadvisor,
    title = {{FIFOAdvisor}: {A} {DSE} {Framework} for {Automated} {FIFO} {Sizing} of {High}-{Level} {Synthesis} {Designs}},
    shorttitle = {{FIFOAdvisor}},
    booktitle = {31st {Asia} and {South} {Pacific} {Design} {Automation} {Conference} ({ASP}-{DAC})},
    author = {Abi-Karam, Stefan and Sarkar, Rishov and Basalama, Suhail and Cong, Jason and Hao, Callie},
    month = jan,
    year = {2026},
}
```

```text
S. Abi-Karam, R. Sarkar, S. Basalama, J. Cong, and C. Hao, "FIFOAdvisor: A DSE Framework for Automated FIFO Sizing of High-Level Synthesis Designs," in 31st Asia and South Pacific Design Automation Conference (ASP-DAC), Jan. 2026.
```

## License

FIFOAdvisor is currently released under the AGPL-3.0 License. See the LICENSE file in the repo for more details.

Reach out to the authors directly if you have further inquiries about licensing.
