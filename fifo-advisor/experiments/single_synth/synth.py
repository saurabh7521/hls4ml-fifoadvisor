from pathlib import Path

from fifo_advisor.automation import TestCase
from fifo_advisor.opt_env import LSEnv

DIR_CURRENT = Path(__file__).parent
DIR_ROOT = DIR_CURRENT.parent.parent
# DIR_TEST_CASES = DIR_ROOT / "test_cases_streamhls"
# DIR_TEST_CASES = DIR_ROOT / "test_cases_streamhls_large"
DIR_TEST_CASES = DIR_ROOT / "test_cases_flowgnn"

design_to_test = "pna"

test_case_dir = DIR_TEST_CASES / design_to_test
assert test_case_dir.exists(), f"Test case dir {test_case_dir} does not exist"
print(f"Test case dir: {test_case_dir}")

local_test_case_dir = DIR_CURRENT / "test_cases" / design_to_test
local_test_case_dir.mkdir(parents=True, exist_ok=True)

test_case = TestCase.from_dir(test_case_dir, design_to_test.split("__")[0])
test_case.copy_to(dest=local_test_case_dir)

test_case.run_csim()
test_case.run_synth()
# test_case.run_cosim()

prj_path = test_case.prj_path.resolve().absolute()

_sim_env = LSEnv(
    test_case.solution_dir,
    env_vars_extra={
        "PRJ_PATH": str(prj_path),
    },
)
