import json
from functools import partial
from pathlib import Path
from pprint import pp
from statistics import mean
from typing import Callable, Iterable

import pandas as pd

from fifo_advisor.automation import TestCase
from fifo_advisor.opt_env import EvalResult, LSEnv, MultiFIFOOptimizer
from fifo_advisor.solvers import (
    MultiDiscreteSimulatedAnnealingOptimizer,  # noqa F401
    MultiGroupedDiscreteSimulatedAnnealingOptimizer,
    MultiGroupRandomSearchOptimizer,
    MultiHeuristicOptimizer,
)

DIR_CURRENT = Path(__file__).parent

DIR_FIGURES = DIR_CURRENT / "figures"
if not DIR_FIGURES.exists():
    DIR_FIGURES.mkdir(exist_ok=True)

DIR_DATA = DIR_CURRENT / "data"
if not DIR_DATA.exists():
    DIR_DATA.mkdir(exist_ok=True)


DIR_PNA_PROJECTS = DIR_CURRENT / "pna_projects"

if not DIR_PNA_PROJECTS.exists():
    raise FileNotFoundError(
        f"PNA projects directory {DIR_PNA_PROJECTS} does not exist. Please run pre_synth.py first."
    )

project_dirs = sorted(DIR_PNA_PROJECTS.glob("*"))

test_cases = [
    TestCase.from_dir(
        dir_project,
    )
    for dir_project in project_dirs
]

test_cases = test_cases[:]  # limit to first design for faster testing

N_JOBS_OVER_ENVS = 64

fn_agg_latency: Callable[[Iterable[float]], float] = mean
fn_agg_bram: Callable[[Iterable[float]], float] = mean

optimizers: dict[str, Callable[..., MultiFIFOOptimizer]] = {
    "multi_group_random_search": partial(
        MultiGroupRandomSearchOptimizer,
        n_samples=5000,
        n_jobs_over_envs=N_JOBS_OVER_ENVS,
        fn_agg_latency=fn_agg_latency,
        fn_agg_bram=fn_agg_bram,
    ),
    "multi_heuristic": partial(
        MultiHeuristicOptimizer,
        n_jobs_over_envs=N_JOBS_OVER_ENVS,
        fn_agg_latency=fn_agg_latency,
        fn_agg_bram=fn_agg_bram,
    ),
    # "multi_discrete_simulated_annealing": partial(
    #     MultiDiscreteSimulatedAnnealingOptimizer,
    #     maxfun=5000 // 4,
    #     n_scaling_factors=4,
    #     init_with_largest=True,
    #     n_jobs_over_envs=N_JOBS_OVER_ENVS,
    #     fn_agg_latency=fn_agg_latency,
    #     fn_agg_bram=fn_agg_bram,
    # ),
    "multi_grouped_discrete_simulated_annealing": partial(
        MultiGroupedDiscreteSimulatedAnnealingOptimizer,
        maxfun=5000 // 4,
        n_scaling_factors=4,
        init_with_largest=True,
        n_jobs_over_envs=N_JOBS_OVER_ENVS,
        fn_agg_latency=fn_agg_latency,
        fn_agg_bram=fn_agg_bram,
    ),
}


def run_baseline_eval(design_cases: list[TestCase]):
    sim_envs = [
        LSEnv(
            design.solution_dir,
        )
        for design in design_cases
    ]

    # compute the baseline max

    baseline_results: list[EvalResult] = []
    for sim_env in sim_envs:
        result = sim_env.eval_solution_default()
        baseline_results.append(result)

    # pp(baseline_results)

    deadlocks = [res.deadlock for res in baseline_results]
    print(f"Baseline deadlocks: {deadlocks}")

    vals_latency = [res.latency for res in baseline_results if res.latency is not None]
    vals_bram = [
        res.bram_usage_total
        for res in baseline_results
        if res.bram_usage_total is not None
    ]
    print(f"Baseline latency: {vals_latency}")
    print(f"Baseline BRAM: {vals_bram}")

    val_avg_latency = mean(vals_latency)
    val_avg_bram = mean(vals_bram)
    return (
        baseline_results,
        val_avg_latency,
        val_avg_bram,
    )


baseline_results, val_avg_latency, val_avg_bram = run_baseline_eval(test_cases)
data_baseline = {
    "designs": [design.name for design in test_cases],
    "vals_latency": [res.latency for res in baseline_results],
    "vals_bram": [res.bram_usage_total for res in baseline_results],
    "avg_latency": val_avg_latency,
    "avg_bram": val_avg_bram,
}

fp_data_baseline = DIR_DATA / "data_baseline_multi_eval_pna.json"
fp_data_baseline.write_text(json.dumps(data_baseline, indent=4))
exit()


def run_single_eval(design_cases: list[TestCase], optimizer_name: str):
    print(
        f"Running design cases as single design for multi optimizer\n{design_cases}\nwith optimizer\n{optimizer_name}"
    )

    sim_envs = [
        LSEnv(
            design.solution_dir,
        )
        for design in design_cases
    ]

    optimizer_class = optimizers[optimizer_name]
    optimizer = optimizer_class(
        sim_envs,
    )

    try:
        results = optimizer.solve()
    except Exception as e:
        raise e

    data = []
    for design_case, results_case in zip(design_cases, results):
        for idx, result in enumerate(results_case):
            data_point = {
                "design": design_case.name,
                "optimizer_name": optimizer_name,
                "eval_index": idx,
                "latency": result.latency,
                "bram": result.bram_usage_total,
                "deadlock": result.deadlock,
            }
            data.append(data_point)
    return data


combos = [
    (test_cases, "multi_group_random_search"),
    (test_cases, "multi_heuristic"),
    (test_cases, "multi_grouped_discrete_simulated_annealing"),
]

# N_JOBS = 1
# with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
#     data_all = executor.map(lambda x: run_single_eval(*x), combos)

# use joblib
data_all = []
for combo in combos:
    data = run_single_eval(*combo)
    data_all.extend(data)


fp_data = DIR_DATA / "data_multi_eval_pna.csv"
df_multi_eval = pd.DataFrame(data_all)
df_multi_eval.to_csv(fp_data, index=False)
print(f"Saved multi-eval data to {fp_data}")
