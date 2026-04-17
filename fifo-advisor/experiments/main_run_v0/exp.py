import itertools
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

from fifo_advisor.automation import TestCase
from fifo_advisor.opt_env import LSEnv, is_pareto_efficient_simple
from fifo_advisor.solvers import (
    DiscreteSimulatedAnnealingOptimizer,
    GroupedDiscreteSimulatedAnnealingOptimizer,
    GroupRandomSearchOptimizer,
    HeuristicOptimizer,
    RandomSearchOptimizer,
    T_FIFOOptimizer,
)

DIR_CURRENT = Path(__file__).parent

DIR_FIGURES = DIR_CURRENT / "figures"
if not DIR_FIGURES.exists():
    DIR_FIGURES.mkdir(exist_ok=True)

DIR_DATA = DIR_CURRENT / "data"
if not DIR_DATA.exists():
    DIR_DATA.mkdir(exist_ok=True)

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


designs_all_dirs: list[Path] = sorted(
    [d for d in DIR_PRE_SYNTH.glob("*") if d.is_dir()]
)

designs_all = [
    TestCase.from_dir(design_dir, design_dir.name.split("__")[0])
    for design_dir in designs_all_dirs
]

designs_to_ignore = ["MultiHeadSelfAttention1"]
designs_all = [
    design
    for design in designs_all
    if not any(design.name.startswith(d) for d in designs_to_ignore)
]

designs_all = [design for design in designs_all if design.dir.name.endswith("__opt5")]

designs_all_filtered = designs_all[:]

data_baseline = []


def run_baseline_eval(design: TestCase):
    print(f"Running design: {design.dir}")
    prj_path = design.prj_path.resolve().absolute()

    sim_env = LSEnv(
        design.solution_dir,
        env_vars_extra={
            "PRJ_PATH": str(prj_path),
        },
    )

    baseline_results = sim_env.eval_solution_default()
    assert baseline_results.deadlock is False

    baseline_latency = baseline_results.latency
    baseline_bram_usage_total = baseline_results.bram_usage_total
    assert baseline_latency is not None
    assert baseline_bram_usage_total is not None

    baseline_latency_bram_product = baseline_latency * baseline_bram_usage_total

    data_baseline.append(
        {
            "latency": baseline_latency,
            "bram": baseline_bram_usage_total,
            "latency_bram_product": baseline_latency_bram_product,
            "design_name": design.dir.name,
        }
    )


for design in designs_all_filtered:
    run_baseline_eval(design)


df_baseline = pd.DataFrame(data_baseline)
df_baseline.to_csv(DIR_DATA / "data_baseline.csv", index=False)


def run_baseline_dumb_eval(design: TestCase):
    prj_path = design.prj_path.resolve().absolute()

    sim_env = LSEnv(
        design.solution_dir,
        env_vars_extra={
            "PRJ_PATH": str(prj_path),
        },
    )

    dumb_config = {}
    for fifo in sim_env.fifos:
        fifo_id = fifo.id
        assert isinstance(fifo_id, int)
        dumb_config[fifo_id] = 2

    eval_result = sim_env.eval_solution_single(dumb_config)

    print(f"Running design: {design.dir}")
    print(f"Deadlock: {eval_result.deadlock}")
    print(f"Latency: {eval_result.latency}")
    print(f"BRAM usage: {eval_result.bram_usage_total}")

    data = {
        "design_name": design.dir.name,
        "deadlock": eval_result.deadlock,
        "latency": eval_result.latency,
        "bram": eval_result.bram_usage_total,
    }

    return data


data_all_baseline_dumb = []
for design in designs_all_filtered:
    d = run_baseline_dumb_eval(design)
    data_all_baseline_dumb.append(d)

df_baseline_dumb = pd.DataFrame(data_all_baseline_dumb)
df_baseline_dumb.to_csv(DIR_DATA / "data_baseline_dumb.csv", index=False)


optimizers: dict[str, partial[T_FIFOOptimizer]] = {
    "random_search": partial(
        RandomSearchOptimizer,
        n_samples=1000,
    ),
    "group_random_search": partial(
        GroupRandomSearchOptimizer,
        n_samples=1000,
    ),
    "heuristic": partial(
        HeuristicOptimizer,
    ),
    "discrete_simulated_annealing": partial(
        DiscreteSimulatedAnnealingOptimizer,
        maxfun=1000,
    ),
    "grouped_discrete_simulated_annealing": partial(
        GroupedDiscreteSimulatedAnnealingOptimizer,
        maxfun=1000,
    ),
}


def run_single_eval(design: TestCase, optimizer_name: str):
    data_points = []
    data_search_counts = []

    print(f"Running design: {design.dir}")
    prj_path = design.prj_path.resolve().absolute()

    sim_env = LSEnv(
        design.solution_dir,
        env_vars_extra={
            "PRJ_PATH": str(prj_path),
        },
    )

    optimizer_class = optimizers[optimizer_name]
    optimizer = optimizer_class(
        sim_env,
    )

    try:
        results = optimizer.solve()
    except Exception as e:
        print(f"Error in design {design.dir}: {e}")
        return

    results_no_deadlock = [result for result in results if not result.deadlock]

    n_total = len(results)
    n_no_deadlock = len(results_no_deadlock)
    n_deadlock = len(results) - n_no_deadlock

    data_search_counts.append(
        {
            "design_name": design.dir.name,
            "optimizer_name": optimizer_name,
            "n_total": n_total,
            "n_no_deadlock": n_no_deadlock,
            "n_deadlock": n_deadlock,
        }
    )

    pareto_mask = is_pareto_efficient_simple(results_no_deadlock)

    vals_latency = [
        result.latency for result in results_no_deadlock if result.latency is not None
    ]
    vals_bram_usage_total = [
        result.bram_usage_total
        for result in results_no_deadlock
        if result.bram_usage_total is not None
    ]

    vals_latency_pareto = [
        latency
        for latency, is_efficient in zip(vals_latency, pareto_mask)
        if is_efficient
    ]

    vals_bram_usage_total_pareto = [
        bram_usage
        for bram_usage, is_efficient in zip(vals_bram_usage_total, pareto_mask)
        if is_efficient
    ]

    latency_bram_products = [
        latency * bram_usage
        for latency, bram_usage in zip(
            vals_latency_pareto, vals_bram_usage_total_pareto
        )
    ]

    points_to_add = []
    for latency, bram_usage, latency_bram_product in zip(
        vals_latency_pareto, vals_bram_usage_total_pareto, latency_bram_products
    ):
        points_to_add.append(
            {
                "latency": latency,
                "bram": bram_usage,
                "latency_bram_product": latency_bram_product,
                "design_name": design.dir.name,
                "optimizer_name": optimizer_name,
            }
        )

    data_points.extend(points_to_add)

    return data_points, data_search_counts


N_JOBS = 52
combos = list(itertools.product(designs_all_filtered, optimizers.keys()))

# data_all = Parallel(n_jobs=N_JOBS, backend="loky", timeout=300)(
#     delayed(run_single_eval)(design, optimizer_name)
#     for design, optimizer_name in combos
# )

# with multiprocessing.Pool(N_JOBS) as pool:
#     data_all = pool.starmap(
#         run_single_eval,
#         combos,
#         chunksize=1,
#     )

with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
    data_all = executor.map(lambda x: run_single_eval(*x), combos)

data_all = [result for result in data_all if result is not None]

all_data_points = []
all_data_search_counts = []
for data_points, data_search_counts in data_all:
    all_data_points.extend(data_points)
    all_data_search_counts.extend(data_search_counts)

df_points = pd.DataFrame(all_data_points)
df_search_counts = pd.DataFrame(all_data_search_counts)

df_points.to_csv(DIR_DATA / "data_points.csv", index=False)
df_search_counts.to_csv(DIR_DATA / "data_search_counts.csv", index=False)
