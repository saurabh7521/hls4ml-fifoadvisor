from pathlib import Path

import numpy as np
import pandas as pd

DIR_CURRENT = Path(__file__).parent

DIR_FIGURES = DIR_CURRENT / "figures"
if not DIR_FIGURES.exists():
    DIR_FIGURES.mkdir(exist_ok=True)

DIR_DATA = DIR_CURRENT / "data"
if not DIR_DATA.exists():
    DIR_DATA.mkdir(exist_ok=True)


FP_COSIM = DIR_DATA / "cosim_runtimes.csv"
FP_FIFO_ADVISOR = DIR_DATA / "fifo_advisor_runtimes.csv"


assert FP_COSIM.exists(), f"File {FP_COSIM} does not exist."
assert FP_FIFO_ADVISOR.exists(), f"File {FP_FIFO_ADVISOR} does not exist."


df_cosim = pd.read_csv(FP_COSIM)
df_fifo_advisor = pd.read_csv(FP_FIFO_ADVISOR)


print("Cosim DataFrame:")
print(df_cosim.head())

print("\nFIFO Advisor DataFrame:")
print(df_fifo_advisor.head())


# remove __opt5 from test_case in df_cosim
df_cosim["test_case"] = df_cosim["test_case"].str.replace("__opt5", "", regex=False)
# remove k15mmseq__opt5_test from df_cosim
df_cosim = df_cosim[~df_cosim["test_case"].str.contains("k15mmseq_test")]

# get the set of test cases in both DataFrames
test_cases_cosim = set(df_cosim["test_case"])
test_cases_fifo_advisor = set(df_fifo_advisor["test_case"])

print(f"Test cases in cosim: {len(test_cases_cosim)}")
print(f"Test cases in fifo_advisor: {len(test_cases_fifo_advisor)}")

assert test_cases_cosim == test_cases_fifo_advisor, (
    "Test cases in cosim and fifo_advisor do not match."
)

designs = sorted(test_cases_cosim, key=lambda x: x.lower())

optimizers = [
    "HeuristicOptimizer",
    "RandomSearchOptimizer",
    "GroupRandomSearchOptimizer",
    "DiscreteSimulatedAnnealingOptimizer",
    "GroupedDiscreteSimulatedAnnealingOptimizer",
]

MAP_OPTIMIZER_TO_SHORT_NAME = {
    "DiscreteSimulatedAnnealingOptimizer": "SA",
    "GroupRandomSearchOptimizer": "Grp. Rnd.",
    "GroupedDiscreteSimulatedAnnealingOptimizer": "Grp. SA",
    "HeuristicOptimizer": "Heuristic",
    "RandomSearchOptimizer": "Rnd.",
}


# I want to makea. multi level latex table with the following structure:

# Design | Co-Sim  |          FIFO Opt.                          |
#        |         | SA | Grp. Rnd. | Grp. SA | Heuristic | Rnd  |
# --------------------------------------------------------------------------
# design_1 | 0.123 | 0.456     | 0.789    | 0.101     | 0.112    |
# design_2 | 0.234 | 0.567     | 0.890    | 0.213     | 0.224     |
# design_3 | 0.345 | 0.678     | 0.901    | 0.324     | 0.335     |

N_SAMPLES = 1_000
N_PARALLEL_COSIM = 32

# Multi-level LaTeX table with two-level column headers

latex_txt = ""

latex_txt += "\\begin{table*}[ht]\n"
latex_txt += "\\centering\n"
latex_txt += "\\footnotesize\n"
latex_txt += (
    # "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l|c|"
    "\\begin{tabular}{l|c|" + "c" * len(optimizers) + "}\n"
)
latex_txt += "\\toprule\n"

# First level headers with multicolumn
latex_txt += (
    "\\multirow{2}{*}{\\textbf{Design}} & \\textbf{Co-Sim} & \\multicolumn{"
    + str(len(optimizers))
    + "}{c}{\\textbf{FIFO-Advisor}} \\\\\n"
)
latex_txt += "\\cmidrule{2-" + str(2 + len(optimizers)) + "}\n"

# Second level headers
latex_txt += (
    " & \\textbf{(PAR=32)} & "
    + " & ".join(
        f"\\textbf{{{MAP_OPTIMIZER_TO_SHORT_NAME[key]}}}" for key in optimizers
    )
    + " \\\\\n"
)
latex_txt += "\\midrule\n"


def format_scientific_notation(value):
    txt = f"{value:.2e}"
    base, exponent = txt.split("e")
    base = float(base)
    exponent = int(exponent)
    return f"${base:.2f}\\text{{e}}{exponent}$"


speedups = {}

for design in designs:
    cosim_runtime_single = df_cosim[df_cosim["test_case"] == design][
        "elapsed_time"
    ].values[0]
    cosim_runtime = cosim_runtime_single * N_SAMPLES

    optimizer_runtimes = []
    for optimizer in optimizers:
        fifo_advisor_runtime_single = df_fifo_advisor[
            (df_fifo_advisor["test_case"] == design)
            & (df_fifo_advisor["optimizer"] == optimizer)
        ]["elapsed_time"].values[0]
        fifo_advisor_runtime = fifo_advisor_runtime_single
        optimizer_runtimes.append(fifo_advisor_runtime)

    design_name = design.replace("_", r"\_")  # Escape underscores for LaTeX

    cosim_runtime_hours = cosim_runtime / 3600.0 / N_PARALLEL_COSIM
    cosim_runtime_days = cosim_runtime / (3600.0 * 24.0) / N_PARALLEL_COSIM

    speedups[design] = {
        optimizer: cosim_runtime / runtime
        for optimizer, runtime in zip(optimizers, optimizer_runtimes)
    }

    # latex_txt += (
    #     f"{design_name} & {cosim_runtime_days:.2f} days & "
    #     + " & ".join(
    #         f"{runtime:.2f} s. / {format_scientific_notation(cosim_runtime / runtime)}$\\times$"
    #         for runtime in optimizer_runtimes
    #     )
    #     + " \\\\\n"
    # )
    latex_txt += (
        f"{design_name} & {cosim_runtime_days:.2f} days & "
        + " & ".join(f"{runtime:.2f} s." for runtime in optimizer_runtimes)
        + " \\\\\n"
    )

latex_txt += "\\midrule\n"
# add a geomean for each optimizer speedup over co-sim
latex_txt += "\\textbf{FIFO-Advisor Speedup Geomean} & & "
geomean_values = []

for optimizer in optimizers:
    vals = []
    for design in designs:
        speedup = speedups[design][optimizer]
        vals.append(speedup)

    geomean = np.exp(np.mean(np.log(vals)))
    geomean_values.append(geomean)

geomean_values_exp = [np.log10(geomean) for geomean in geomean_values]


times_symbol = r"\pmb{\times}"

latex_txt += " & ".join(
    f"$\\bm{{10^{{{geomean_exp:.2f}}}{times_symbol}}}$"
    for geomean_exp in geomean_values_exp
)


latex_txt += " \\\\\n"


latex_txt += "\\bottomrule\n"
latex_txt += "\\end{tabular}\n"

latex_txt += "\\caption{Comparison of Co-Simulation and FIFO Optimization runtimes for different designs.}\n"
latex_txt += "\\label{tab:cosim_fifo_advisor_comparison}\n"
latex_txt += "\\end{table*}\n"

print(latex_txt)

fp_latex_table = DIR_FIGURES / "latex_table.txt"
fp_latex_table.write_text(latex_txt)
