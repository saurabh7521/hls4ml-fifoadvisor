import shutil
from pathlib import Path
from pprint import pp

from dotenv import dotenv_values
from joblib import Parallel, delayed

from fifo_advisor.automation import TestCase

DIR_CURRENT = Path(__file__).parent


DIR_ROOT = DIR_CURRENT.parent.parent

DIR_TEST_CASES = DIR_ROOT / "test_cases_streamhls"
DIR_TEST_CASES_LARGE = DIR_ROOT / "test_cases_streamhls_large"

assert DIR_TEST_CASES.exists(), f"Test case dir {DIR_TEST_CASES} does not exist"
assert DIR_TEST_CASES_LARGE.exists(), (
    f"Test case dir {DIR_TEST_CASES_LARGE} does not exist"
)


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

if not DIR_PRE_SYNTH.exists():
    DIR_PRE_SYNTH.mkdir(exist_ok=True)


designs_to_test: list[str] = sorted(
    [d.name for d in DIR_TEST_CASES.glob("*")]
) + sorted([d.name for d in DIR_TEST_CASES_LARGE.glob("*")])
designs_to_ignore: list[str] = []
designs_to_test = [
    design for design in designs_to_test if design not in designs_to_ignore
]

pp(designs_to_test)

N_JOBS = 24


def synth_design(design_to_test: str):
    print(f"Test case: {design_to_test}")

    test_case_dir = DIR_PRE_SYNTH / design_to_test
    if test_case_dir.exists():
        shutil.rmtree(test_case_dir)
    test_case_dir.mkdir()

    print(f"Building test case dir: {test_case_dir}")

    test_case = TestCase.from_dir(
        DIR_TEST_CASES / design_to_test, design_to_test.split("__")[0]
    )
    test_case.copy_to(dest=test_case_dir)

    test_case.run_csim()
    test_case.run_synth()


Parallel(n_jobs=N_JOBS, backend="threading")(
    delayed(synth_design)(design) for design in designs_to_test
)
