from pathlib import Path

from joblib import Parallel, delayed

from fifo_advisor.automation import TestCase
from fifo_advisor.opt_env import LSEnv

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


def trace_test_case(test_case: TestCase) -> None:
    _sim_env = LSEnv(
        test_case.solution_dir,
    )


N_JOBS = 64
Parallel(n_jobs=N_JOBS, backend="threading")(
    delayed(trace_test_case)(test_case) for test_case in test_cases
)
