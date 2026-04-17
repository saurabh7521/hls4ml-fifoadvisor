import time
from copy import deepcopy
from functools import partial
from pathlib import Path

import pandas as pd
import tqdm
from dotenv import dotenv_values
from joblib import Parallel, delayed

from fifo_advisor.automation import TestCase
from fifo_advisor.opt_env import LSEnv
from fifo_advisor.solvers import (
    DiscreteSimulatedAnnealingOptimizer,
    GroupedDiscreteSimulatedAnnealingOptimizer,
    GroupRandomSearchOptimizer,
    HeuristicOptimizer,
    RandomSearchOptimizer,
    T_FIFOOptimizer,
)

N_SAMPLES = 1_000
N_LEVELS_FOR_SA = 5
N_JOBS = 16

# Plotting parameters
FIGURE_SIZE = (12, 8)
DPI = 300

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


optimizers_to_test = {
    "HeuristicOptimizer": HeuristicOptimizer,
    "RandomSearchOptimizer": partial(RandomSearchOptimizer, n_samples=N_SAMPLES),
    "GroupRandomSearchOptimizer": partial(
        GroupRandomSearchOptimizer, n_samples=N_SAMPLES
    ),
    "DiscreteSimulatedAnnealingOptimizer": partial(
        DiscreteSimulatedAnnealingOptimizer,
        maxfun=N_SAMPLES // N_LEVELS_FOR_SA,
        n_scaling_factors=N_LEVELS_FOR_SA,
        init_with_largest=True,
    ),
    "GroupedDiscreteSimulatedAnnealingOptimizer": partial(
        GroupedDiscreteSimulatedAnnealingOptimizer,
        maxfun=N_SAMPLES // N_LEVELS_FOR_SA,
        n_scaling_factors=N_LEVELS_FOR_SA,
        init_with_largest=True,
    ),
}


def run_optimizer_on_design(optimizer_name, optimizer_class, design):
    design_copy = deepcopy(design)
    print(f"Running design {design_copy.dir.name} with optimizer {optimizer_name}")
    prj_path = design_copy.prj_path.resolve().absolute()

    sim_env = LSEnv(
        design_copy.solution_dir,
        env_vars_extra={
            "PRJ_PATH": str(prj_path),
        },
    )

    optimizer: T_FIFOOptimizer = optimizer_class(sim_env)

    t0 = time.perf_counter()
    try:
        results = optimizer.solve()
        print(f"{len(results)} results found for optimizer {optimizer_name}")
    except Exception as e:
        print(f"Error in design {design_copy.dir} with {optimizer_name}: {e}")
        raise e
    t1 = time.perf_counter()
    dt = t1 - t0
    n_samples = len(results)

    return {
        "optimizer": optimizer_name,
        "test_case": design_copy.name,
        "elapsed_time": dt,
        "n_samples": n_samples,
    }


# Create list of all optimizer-design combinations
tasks = [
    (optimizer_name, optimizer_class, design)
    for optimizer_name, optimizer_class in optimizers_to_test.items()
    for design in designs_all
]

# Run in parallel
data = Parallel(n_jobs=N_JOBS)(
    delayed(run_optimizer_on_design)(optimizer_name, optimizer_class, design)
    for optimizer_name, optimizer_class, design in tqdm.tqdm(tasks)
)

df = pd.DataFrame(data)
df.to_csv(DIR_DATA / "fifo_advisor_runtimes.csv", index=False)
