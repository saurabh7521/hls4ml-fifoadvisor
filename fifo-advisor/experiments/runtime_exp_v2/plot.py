import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

DIR_CURRENT = Path(__file__).parent

DATA_DIR = DIR_CURRENT / "data"
assert DATA_DIR.exists(), f"Data directory {DATA_DIR} does not exist."

FP_DATA = DATA_DIR / "runtime_exp_v2_data.json"
assert FP_DATA.exists(), f"Data file {FP_DATA} does not exist."

### PLOTTING RESULTS ###

ALPHA = 0.7


def huristic_score(latency, bram, base_latency, base_bram, alpha=ALPHA):
    relative_latency = latency / base_latency
    if base_bram == 0:
        base_bram = 1

    relative_bram = bram / base_bram

    score = alpha * relative_latency + (1 - alpha) * relative_bram
    return score


data_to_cache_to_disk = json.loads(FP_DATA.read_text())

MAP_OPTIMIZER_TO_SHORT_NAME = {
    "baseline": "Baseline-Max",
    "DiscreteSimulatedAnnealingOptimizer": "SA",
    "GroupRandomSearchOptimizer": "Grp. Rnd.",
    "GroupedDiscreteSimulatedAnnealingOptimizer": "Grp. SA",
    "HeuristicOptimizer": "Heuristic",
    "RandomSearchOptimizer": "Rnd",
}

font = {"size": 10}
matplotlib.rc("font", **font)

fig, ax = plt.subplots(figsize=(6, 2.8))


baseline_latency = data_to_cache_to_disk["baseline"]["latency"]
baseline_bram_usage = data_to_cache_to_disk["baseline"]["bram_usage_total"]


ax.axhline(
    y=huristic_score(
        baseline_latency,
        baseline_bram_usage,
        baseline_latency,
        baseline_bram_usage,
        alpha=ALPHA,
    ),
    color="black",
    linestyle="-",
    label="Baseline",
    alpha=0.5,
)


opt_names = [
    "HeuristicOptimizer",
    "RandomSearchOptimizer",
    "GroupRandomSearchOptimizer",
    "DiscreteSimulatedAnnealingOptimizer",
    "GroupedDiscreteSimulatedAnnealingOptimizer",
]

MAP_COLORS = {
    "HeuristicOptimizer": "green",
    "RandomSearchOptimizer": "orange",
    "GroupRandomSearchOptimizer": "red",
    "DiscreteSimulatedAnnealingOptimizer": "blue",
    "GroupedDiscreteSimulatedAnnealingOptimizer": "purple",
}


for opt_name in opt_names:
    times = data_to_cache_to_disk[opt_name]["times"]
    scores = data_to_cache_to_disk[opt_name]["scores"]

    ax.plot(
        times,
        scores,
        label=MAP_OPTIMIZER_TO_SHORT_NAME.get(opt_name, opt_name),
        marker="o",
        markevery=[-1],
        # color=MAP_COLORS.get(opt_name, "black"),
    )


ax.set_xlabel("Time (s)")
ax.set_ylabel(f"Heuristic Score (α={ALPHA})\nRealtive to Baseline-Max")


ax.set_ylim(bottom=0.6, top=1.8)  # Ensure y-axis starts at 0
ax.set_xlim(left=0, right=10)  # Set x-axis limits
ax.grid(True, alpha=0.5, linestyle="--")

yticks = np.arange(0.6, 1.8, 0.2)
ax.set_yticks(yticks)

xticks = np.arange(0.0, 11.0, 1.0)
ax.set_xticks(xticks)

ax.legend(
    ncol=3,
    loc="upper center",
    fontsize=10,
    labelspacing=0.1,
)


selected_design_name = "k15mmtree"
ax.set_title(f'Optimizer Iso-Runtime Comparison for "{selected_design_name}"')


# ax.title.set_fontweight("bold")
# make axes labels bold
# ax.xaxis.label.set_fontweight("bold")
# ax.yaxis.label.set_fontweight("bold")


plt.tight_layout()

DIR_FIGURES = DIR_CURRENT / "figures"
DIR_FIGURES.mkdir(parents=True, exist_ok=True)

FP_FIGURE = DIR_FIGURES / "runtime_exp_v2.png"
fig.savefig(FP_FIGURE, dpi=300)
