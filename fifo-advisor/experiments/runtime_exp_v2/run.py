import json
import time
from pathlib import Path
from pprint import pp

import numpy as np
from dotenv import dotenv_values
from fifo_advisor.automation import TestCase
from fifo_advisor.opt_env import EvalResult, LSEnv
from fifo_advisor.solvers import (
    DiscreteSimulatedAnnealingOptimizer,
    GroupedDiscreteSimulatedAnnealingOptimizer,
    GroupRandomSearchOptimizer,
    HeuristicOptimizer,
    RandomSearchOptimizer,
)
from matplotlib import pyplot as plt

### SETUP ###

DIR_CURRENT = Path(__file__).parent

ENV_FILE: Path = DIR_CURRENT.parent / ".env"
if ENV_FILE.exists():
    env_vars = dotenv_values(ENV_FILE)
else:
    raise FileNotFoundError(
        f"Environment file {ENV_FILE} not found. Please create it with the required variables."
    )
if "DIR_PRE_SYNTH" in env_vars:
    if env_vars["DIR_PRE_SYNTH"] is None:
        raise ValueError(
            "Environment variable 'DIR_PRE_SYNTH' is set to None. Please set it to a valid path."
        )
    DIR_PRE_SYNTH = Path(env_vars["DIR_PRE_SYNTH"])
else:
    raise KeyError(
        "Environment variable 'DIR_PRE_SYNTH' not found in .env file. Please add it."
    )


### LOAD DESIGN ###

selected_design_name = "k15mmtree"


designs_all_dirs: list[Path] = sorted(
    [d for d in DIR_PRE_SYNTH.glob("*") if d.is_dir()]
)


designs_all = [
    TestCase.from_dir(design_dir, design_dir.name.split("__")[0])
    for design_dir in designs_all_dirs
]

designs_to_ignore: list[str] = ["MultiHeadSelfAttention1"]
designs_all = [
    design
    for design in designs_all
    if not any(design.name.startswith(d) for d in designs_to_ignore)
]


extra_designs = [
    "adaptive_filter_bank",
    "scatter_gather",
    "cascaded_filter_bank",
    "fork_join_network",
    "scatter_gather_v2",
    "scatter_gather_v3",
]

designs_all = [
    design
    for design in designs_all
    if design.dir.name.endswith("__opt5") or design.dir.name in extra_designs
]

selected_design = next(
    (design for design in designs_all if design.name == selected_design_name), None
)
if selected_design is None:
    raise ValueError(f"Design {selected_design_name} not found in the list of designs.")


### BASELINE EVALUATION ###


def run_baseline_eval(design: TestCase) -> EvalResult:
    print(f"Running design: {design.dir}")
    prj_path = design.prj_path.resolve().absolute()

    sim_env = LSEnv(
        design.solution_dir,
        env_vars_extra={
            "PRJ_PATH": str(prj_path),
        },
    )
    fifo_sizes_base_not_none = {}
    for fifo_id, fifo_size in sim_env.fifo_sizes_base.items():
        if fifo_size is not None:
            fifo_sizes_base_not_none[fifo_id] = fifo_size
        else:
            raise ValueError(
                f"FIFO size for FIFO {fifo_id} is None. Please set a valid FIFO size."
            )

    print(fifo_sizes_base_not_none)

    baseline_results: EvalResult = sim_env.eval_solution_default()
    assert baseline_results.deadlock is False

    baseline_latency = baseline_results.latency
    baseline_bram_usage_total = baseline_results.bram_usage_total
    assert baseline_latency is not None
    assert baseline_bram_usage_total is not None

    return baseline_results


baseline_result = run_baseline_eval(selected_design)  # type: ignore


pp(baseline_result)


### RANDOM SAMPLE RUNTIME ###


MAX_NUM_SAMPLES = 140_000
N_STEPS = 20


n_sample_steps = np.linspace(
    20, MAX_NUM_SAMPLES, N_STEPS, endpoint=True, dtype=float
).tolist()
n_sample_steps = [int(round(x)) for x in n_sample_steps]


random_search_optimizers = {
    "RandomSearchOptimizer": RandomSearchOptimizer,
    "GroupRandomSearchOptimizer": GroupRandomSearchOptimizer,
}

random_search_results = {}
for opt in random_search_optimizers:
    random_search_results[opt] = {}

    for n_samples in n_sample_steps:
        prj_path = selected_design.prj_path.resolve().absolute()

        sim_env = LSEnv(
            selected_design.solution_dir,
            env_vars_extra={
                "PRJ_PATH": str(prj_path),
            },
        )

        opt_cls = random_search_optimizers[opt]
        opt_inst = opt_cls(
            sim_env,
            n_samples=n_samples,
        )

        t0 = time.perf_counter()
        try:
            results = opt_inst.solve()
            print(f"{len(results)} results found for optimizer {opt_cls.__name__}")
        except Exception as e:
            print(f"Error in design {selected_design.dir} with {opt_cls.__name__}: {e}")
            raise e
        t1 = time.perf_counter()
        dt = t1 - t0
        random_search_results[opt][n_samples] = {
            "n_samples": n_samples,
            "dt": dt,
            "results": results,
        }


### OTHER OPTIMIZERS ###

N_SAMPLES = 10_000
N_LEVELS_FOR_SA = 5

other_optimizers = {
    "DiscreteSimulatedAnnealingOptimizer": DiscreteSimulatedAnnealingOptimizer,
    "GroupedDiscreteSimulatedAnnealingOptimizer": GroupedDiscreteSimulatedAnnealingOptimizer,
    "HeuristicOptimizer": HeuristicOptimizer,
}


other_optimizer_results = {}
for opt_name, opt_cls in other_optimizers.items():
    other_optimizer_results[opt_name] = {}

    prj_path = selected_design.prj_path.resolve().absolute()

    sim_env = LSEnv(
        selected_design.solution_dir,
        env_vars_extra={
            "PRJ_PATH": str(prj_path),
        },
    )

    if "SimulatedAnnealing" in opt_name:
        opt_inst = opt_cls(
            sim_env,
            maxfun=N_SAMPLES // N_LEVELS_FOR_SA,
            n_scaling_factors=N_LEVELS_FOR_SA,
            init_with_largest=True,
        )
    else:
        opt_inst = opt_cls(sim_env)

    t0 = time.perf_counter()
    try:
        results = opt_inst.solve()
        print(f"{len(results)} results found for optimizer {opt_cls.__name__}")
    except Exception as e:
        print(f"Error in design {selected_design.dir} with {opt_cls.__name__}: {e}")
        raise e
    t1 = time.perf_counter()
    dt = t1 - t0
    other_optimizer_results[opt_name][N_SAMPLES] = {
        "n_samples": N_SAMPLES,
        "dt": dt,
        "results": results,
        "t0": t0,
    }

# pp(random_search_results)


### PLOTTING RESULTS ###

ALPHA = 0.7


def huristic_score(latency, bram, base_latency, base_bram, alpha=ALPHA):
    relative_latency = latency / base_latency
    if base_bram == 0:
        base_bram = 1

    relative_bram = bram / base_bram

    score = alpha * relative_latency + (1 - alpha) * relative_bram
    return score


data_to_cache_to_disk = {}

fig, ax = plt.subplots(figsize=(10, 6))

# y-axis is realtive latency to baseline
# x-axis is time


# draw a horizontal line at y=1
baseline_latency = baseline_result.latency
# ax.axhline(
#     y=huristic_score(
#         baseline_latency,
#         baseline_result.bram_usage_total,
#         baseline_latency,
#         baseline_result.bram_usage_total,
#         alpha=ALPHA,
#     ),
#     color="gray",
#     linestyle="--",
#     label="Baseline",
# )

data_to_cache_to_disk["baseline"] = {
    "latency": baseline_latency,
    "bram_usage_total": baseline_result.bram_usage_total,
}

# for the random search optimizers, we want to build a series of points from each n_smaples run and total time, for each n_samples we just pick the best scopre to plot
for opt_name, results in random_search_results.items():
    n_samples = []
    times = []
    scores = []

    for n_samples_key, result in results.items():
        n_samples.append(result["n_samples"])
        times.append(result["dt"])
        best_result = min(
            filter(
                lambda x: x.latency is not None and x.bram_usage_total is not None,
                result["results"],
            ),
            key=lambda x: huristic_score(
                x.latency,
                x.bram_usage_total,
                baseline_latency,
                baseline_result.bram_usage_total,
                alpha=ALPHA,
            ),
        )
        scores.append(
            huristic_score(
                best_result.latency,
                best_result.bram_usage_total,
                baseline_latency,
                baseline_result.bram_usage_total,
                alpha=ALPHA,
            )
        )

    data_to_cache_to_disk[opt_name] = {
        "times": times,
        "scores": scores,
    }

    # ax.plot(times, scores, label=opt_name)


# for the other optmizers we just read the timestmap from the results and subtract from the zero time
for opt_name, results_other in other_optimizer_results.items():
    times = []
    best_scores = []

    t0 = results_other[N_SAMPLES]["t0"]
    results_list = list(
        filter(
            lambda x: x.latency is not None and x.bram_usage_total is not None,
            results_other[N_SAMPLES]["results"],
        )
    )
    if not results_list:
        raise ValueError(
            f"No valid results found for optimizer {opt_name} with N_SAMPLES={N_SAMPLES}."
        )

    # sort results by timestamp
    results_list.sort(key=lambda x: x.timestamp)
    for i in range(len(results_list)):
        result = results_list[i]
        times.append(result.timestamp - t0)
        min_score = min(
            [
                huristic_score(
                    res.latency,
                    res.bram_usage_total,
                    baseline_latency,
                    baseline_result.bram_usage_total,
                    alpha=ALPHA,
                )
                for res in results_list[: i + 1]
                if res.latency is not None and res.bram_usage_total is not None
            ]
        )
        best_scores.append(min_score)

    data_to_cache_to_disk[opt_name] = {
        "times": times,
        "scores": best_scores,
    }

    # ax.plot(times, best_scores, label=opt_name)

# ax.set_xlabel("Time (s)")
# ax.set_ylabel(f"Heuristic Score (α={ALPHA}) Realtive to Baseline-Max")
# ax.set_title(f"Optimizer Runtime Comparison for {selected_design_name}")
# ax.legend()

# ax.set_ylim(bottom=0.5)  # Ensure y-axis starts at 0
# ax.set_xlim(left=0)  # Ensure x-axis starts at 0
# ax.grid(True)

# yticks = np.arange(0.6, 2.2, 0.2)
# ax.set_yticks(yticks)

# # x ticks every 0.25s
# # xticks = np.arange(0, 5.5, 0.5)
# # ax.set_xticks(xticks)

# plt.tight_layout()

# DIR_FIGURES = DIR_CURRENT / "figures"
# DIR_FIGURES.mkdir(parents=True, exist_ok=True)

# FP_FIGURE = DIR_FIGURES / "runtime_exp_v2.png"
# fig.savefig(FP_FIGURE, dpi=300)


DATA_DIR = DIR_CURRENT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FP_DATA = DATA_DIR / "runtime_exp_v2_data.json"
FP_DATA.write_text(
    json.dumps(data_to_cache_to_disk, indent=4),
)
