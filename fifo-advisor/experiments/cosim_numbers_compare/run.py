from pathlib import Path
from pprint import pp

from dotenv import dotenv_values
from joblib import Parallel, delayed

from fifo_advisor.automation import TestCase

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
if "DIR_COSIM" in env_vars:
    if env_vars["DIR_COSIM"] is None:
        raise ValueError(
            "Environment variable 'DIR_COSIM' is set to None. Please set it to a valid path."
        )
    DIR_COSIM = Path(env_vars["DIR_COSIM"])
else:
    raise KeyError(
        "Environment variable 'DIR_COSIM' not found in .env file. Please add it."
    )


designs_all_dirs = sorted([d for d in DIR_PRE_SYNTH.glob("*") if d.is_dir()])
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


pp([d.dir for d in designs_all])
print(f"Total designs found: {len(designs_all)}")

if not DIR_COSIM.exists():
    DIR_COSIM.mkdir(parents=True, exist_ok=True)

print(f"Copying synth designs to {DIR_COSIM}")


for design in designs_all:
    # check if the design directory exists in DIR_COSIM
    design_dir = DIR_COSIM / design.dir.name
    if design_dir.exists():
        print(f"Skipping {design.dir.name}, already exists in {DIR_COSIM}")
        continue

    # copy the design to DIR_COSIM
    design.copy_to(design_dir)

    # print the design details
    print(f"Copied {design.name} to {design_dir}")


designs_all_dirs_cosim = sorted([d for d in DIR_COSIM.glob("*") if d.is_dir()])
designs_all_cosim = [
    TestCase.from_dir(design_dir, design_dir.name.split("__")[0])
    for design_dir in designs_all_dirs_cosim
]

print(f"Total designs copied: {len(designs_all_cosim)}")
print("Designs copied:")

designs_cosim_to_run = []

for design in designs_all_cosim:
    # check if dir contains cosim_time.txt, if so skip
    cosim_time_fp = design.dir / "cosim_time.txt"
    if cosim_time_fp.exists():
        print(f"Skipping {design.name} - {design.dir}, already has cosim_time.txt")
        continue

    designs_cosim_to_run.append(design)

print(f"Total designs to run co-simulation: {len(designs_cosim_to_run)}")
print("Designs to run co-simulation:")
for design in designs_cosim_to_run:
    print(f"- {design.name} in {design.dir}")

N_JOBS = 32


def run_cosim_single(design: TestCase) -> None:
    print(f"Running co-simulation for design: {design.name} in {design.dir}")
    design.run_cosim()


Parallel(n_jobs=N_JOBS, backend="threading")(
    delayed(run_cosim_single)(design) for design in designs_cosim_to_run
)
