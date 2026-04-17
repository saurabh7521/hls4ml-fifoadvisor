from pathlib import Path
from pprint import pp

import pandas as pd
from dotenv import dotenv_values
from joblib import Parallel, delayed

from fifo_advisor.automation import TestCase

DIR_CURRENT = Path(__file__).parent

DIR_DATA = DIR_CURRENT / "data"
if not DIR_DATA.exists():
    DIR_DATA.mkdir(parents=True, exist_ok=True)

ENV_FILE: Path = DIR_CURRENT.parent / ".env"
if ENV_FILE.exists():
    env_vars = dotenv_values(ENV_FILE)
else:
    raise FileNotFoundError(
        f"Environment file {ENV_FILE} not found. Please create it with the required variables."
    )


if "DIR_OLD_COSIM" in env_vars:
    if env_vars["DIR_OLD_COSIM"] is None:
        raise ValueError(
            "Environment variable 'DIR_OLD_COSIM' is set to None. Please set it to a valid path."
        )
    DIR_OLD_COSIM = Path(env_vars["DIR_OLD_COSIM"])
else:
    raise KeyError(
        "Environment variable 'DIR_OLD_COSIM' not found in .env file. Please add it."
    )


# find any file named vhls_stdout.log in the directory DIR_OLD_COSIM

vhls_stdout_files = list(DIR_OLD_COSIM.glob("**/vhls_stdout.log"))
vhls_stdout_files = [f for f in vhls_stdout_files if "__opt5" in f.parent.name]
designs_to_ignore = ["MultiHeadSelfAttention1"]
vhls_stdout_files = [
    f
    for f in vhls_stdout_files
    if not any(d in f.parent.name for d in designs_to_ignore)
]
pp(vhls_stdout_files)

data = []
for file in vhls_stdout_files:
    print(f"Processing file: {file}")
    # find the line that contains the text "Finished Command cosim_design"
    txt = file.read_text()
    lines = txt.splitlines()

    dt = None
    for line in lines:
        if "Finished Command cosim_design" in line:
            # [2025-03-19 21:03:18.366445] INFO: [HLS 200-111] Finished Command cosim_design CPU user time: 13530.9 seconds. CPU system time: 14.01 seconds. Elapsed time: 16127.9 seconds; current allocated memory: 0.000 MB.
            # extract the elapsed time
            assert "Elapsed time: " in line, "Line does not contain 'Elapsed time'"
            elapsed_time_line = line.split("Elapsed time: ")[-1]
            elapsed_time = elapsed_time_line.split(" ")[0]
            dt = float(elapsed_time)
            break

    assert dt is not None, f"Did not find elapsed time in file {file}"

    # find the test case name
    test_case_name = file.parent.name

    data.append(
        {
            "test_case": test_case_name,
            "elapsed_time": dt,
        }
    )

df_cosim_runtimes = pd.DataFrame(data)
df_cosim_runtimes.to_csv(DIR_DATA / "cosim_runtimes.csv", index=False)
print(f"Saved data to {DIR_DATA / 'cosim_runtimes.csv'}")
