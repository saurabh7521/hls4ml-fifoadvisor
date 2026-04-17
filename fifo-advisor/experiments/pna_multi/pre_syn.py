import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from joblib import Parallel, delayed

from fifo_advisor.automation import TestCase

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

DIR_DESIGN_PNA_MULTI = DIR_CURRENT.parent.parent / "test_cases_flowgnn_multi" / "pna"
assert DIR_DESIGN_PNA_MULTI.exists()

dir_tb_data = DIR_DESIGN_PNA_MULTI / "tb_data"
tb_files = sorted(dir_tb_data.glob("*"))

# print(tb_files)


def group_graph_files(paths: list[Path]) -> list[dict[str, Any]]:
    """
    Group a flat list of file paths into a list of dicts grouped by graph index.

    Expected filenames follow:
        g<number>_edge_attr.bin
        g<number>_edge_list.bin
        g<number>_info.txt
        g<number>_node_feature.bin

    Returns a list of dicts like:
        {
            "index": 1,
            "edge_attr": "...",
            "edge_list": "...",
            "info": "...",
            "node_feature": "..."
        }
    """
    pattern = re.compile(r"g(\d+)_(edge_attr|edge_list|info|node_feature)")
    grouped: defaultdict[int, dict[str, str]] = defaultdict(dict)

    for p in paths:
        m = pattern.search(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        key = m.group(2)
        grouped[idx][key] = str(p)  # or `p` if you prefer Path object

    # Convert to sorted list of dicts
    result = [{"index": idx, **grouped[idx]} for idx in sorted(grouped.keys())]

    return result


grouped_tb_files = group_graph_files(tb_files)

# pp(grouped_tb_files)


DIR_PNA_PROJECTS = DIR_CURRENT / "pna_projects"


USE_CACHE = True

if USE_CACHE:
    if not DIR_PNA_PROJECTS.exists():
        DIR_PNA_PROJECTS.mkdir(exist_ok=True)
else:
    if DIR_PNA_PROJECTS.exists():
        shutil.rmtree(DIR_PNA_PROJECTS, ignore_errors=True)
    DIR_PNA_PROJECTS.mkdir(exist_ok=True)

project_dirs = []

for tb_case in grouped_tb_files:
    tb_index = tb_case["index"]
    dir_project_tb = DIR_PNA_PROJECTS / f"pna_multi__{str(tb_index)}"
    project_dirs.append(dir_project_tb)

    # if it exists skip
    if dir_project_tb.exists():
        print(f"Project {dir_project_tb} already exists. Skipping...")
        continue

    print(f"Creating project {dir_project_tb}...")

    shutil.copytree(
        DIR_DESIGN_PNA_MULTI,
        dir_project_tb,
        ignore=shutil.ignore_patterns("tb_data"),
        dirs_exist_ok=True,
    )

    # write the tb_case to a file tb_data_case.json in the project directory
    tb_data_case_file = dir_project_tb / "tb_data_case.json"
    tb_data_case_file.write_text(json.dumps(tb_case, indent=4))

    for key, file_path in tb_case.items():
        if key == "index":
            continue
        shutil.copy(file_path, dir_project_tb / Path(file_path).name)

    # rename the tb_data files just copied to change the number prefix to be 1
    # for file in dir_project_tb.glob("(g*_*.bin)|(g*_*.txt)"):
    for file in list(dir_project_tb.glob("g*_*.bin")) + list(
        dir_project_tb.glob("g*_*.txt")
    ):
        new_name = re.sub(r"g\d+_", "g1_", file.name)
        file.rename(dir_project_tb / new_name)


test_cases = [
    TestCase.from_dir(
        dir_project,
    )
    for dir_project in project_dirs
]


def synth_design(test_case: TestCase):
    print(f"Running csim: {test_case.name}")
    test_case.run_csim()
    print(f"Running synth: {test_case.name}")
    test_case.run_synth()


N_JOBS = 64
Parallel(n_jobs=N_JOBS, backend="threading")(
    delayed(synth_design)(test_case) for test_case in test_cases
)
