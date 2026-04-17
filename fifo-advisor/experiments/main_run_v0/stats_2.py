import itertools
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

DIR_CURRENT = Path(__file__).parent

DIR_FIGURES = DIR_CURRENT / "figures"
if not DIR_FIGURES.exists():
    DIR_FIGURES.mkdir(exist_ok=True)

DIR_DATA = DIR_CURRENT / "data"
if not DIR_DATA.exists():
    DIR_DATA.mkdir(exist_ok=True)


df_baseline = pd.read_csv(DIR_DATA / "data_baseline.csv")
df_baseline_dumb = pd.read_csv(DIR_DATA / "data_baseline_dumb.csv")
df_points = pd.read_csv(DIR_DATA / "data_points.csv")
df_search_counts = pd.read_csv(DIR_DATA / "data_search_counts.csv")


designs_to_remove = ["ResMLP__opt5", "k7mmtree_balanced__opt5", "gesummv__opt5"]
# designs_to_remove = ["ResMLP__opt4", "ResMLP__opt5", "k7mmtree_balanced__opt5"]

for design in designs_to_remove:
    df_baseline = df_baseline[df_baseline["design_name"] != design]
    df_baseline_dumb = df_baseline_dumb[df_baseline_dumb["design_name"] != design]
    df_points = df_points[df_points["design_name"] != design]
    df_search_counts = df_search_counts[df_search_counts["design_name"] != design]


# if design_name ends in __opt5 remove the __opt5 suffix
df_baseline["design_name"] = df_baseline["design_name"].str.replace("__opt5", "")
df_baseline_dumb["design_name"] = df_baseline_dumb["design_name"].str.replace(
    "__opt5", ""
)
df_points["design_name"] = df_points["design_name"].str.replace("__opt5", "")
df_search_counts["design_name"] = df_search_counts["design_name"].str.replace(
    "__opt5", ""
)


designs = df_baseline["design_name"].unique().tolist()
optimizers = df_points["optimizer_name"].unique().tolist()
all_cases = itertools.product(designs, optimizers)


optimizer_to_color_map = {
    "random_search": "blue",
    "group_random_search": "green",
    "heuristic": "red",
    "init_simulated_annealing": "orange",
    "discrete_simulated_annealing": "purple",
    "grouped_discrete_simulated_annealing": "brown",
}

optimizer_to_name_map = {
    "random_search": "Random Search",
    "group_random_search": "Grouped Random Search",
    "heuristic": "Greedy Search",
    "init_simulated_annealing": "Seeded Simulated Annealing",
    "discrete_simulated_annealing": "Sim. Annealing",
    "grouped_discrete_simulated_annealing": "Grouped Sim. Annealing",
}


data_relative = []
for design, optimizer in all_cases:
    baseline_data_case = df_baseline[(df_baseline["design_name"] == design)].copy()
    assert len(baseline_data_case) == 1
    baseline_latency = baseline_data_case["latency"].values[0].item()
    baseline_bram = baseline_data_case["bram"].values[0].item()

    optimizer_data_case = df_points[
        (df_points["design_name"] == design)
        & (df_points["optimizer_name"] == optimizer)
    ].copy()

    # sort pereto front
    optimizer_data_case = optimizer_data_case.sort_values(
        ["latency", "bram"], ascending=[False, True]
    )

    def huristic_score(latency, bram, base_latency, base_bram, ALPHA=0.7):
        relative_latency = latency / base_latency
        if base_bram == 0:
            base_bram = 1

        relative_bram = bram / base_bram

        score = ALPHA * relative_latency + (1 - ALPHA) * relative_bram
        return score

    optimizer_data_case["heuristic_score"] = optimizer_data_case.apply(
        lambda row: huristic_score(
            row["latency"],
            row["bram"],
            base_latency=baseline_latency,
            base_bram=baseline_bram,
        ),
        axis=1,
    )

    best_heuristic_score = optimizer_data_case["heuristic_score"].min()
    best_heuristic_score_row = optimizer_data_case[
        optimizer_data_case["heuristic_score"] == best_heuristic_score
    ]
    assert len(best_heuristic_score_row) == 1

    baseline_dumb_data_case = df_baseline_dumb[
        (df_baseline_dumb["design_name"] == design)
    ].copy()
    assert len(baseline_dumb_data_case) == 1

    baseline_dumb_deadlock = baseline_dumb_data_case["deadlock"].tolist()[0]
    baseline_dumb_latency = baseline_dumb_data_case["latency"].tolist()[0]
    baseline_dumb_bram = baseline_dumb_data_case["bram"].tolist()[0]

    if baseline_dumb_deadlock is False:
        latency_reduction = (
            (best_heuristic_score_row["latency"].tolist()[0]) / baseline_dumb_latency
        )

        bram_increase = (
            # best_heuristic_score_row["bram"].tolist()[0] - baseline_dumb_bram
            best_heuristic_score_row["bram"].tolist()[0]
        )

        print(
            f"Design: {design}, Optimizer: {optimizer}, Latency Reduction: {latency_reduction}, BRAM Increase: {bram_increase}"
        )

        data = {
            "design_name": design,
            "optimizer_name": optimizer,
            "latency_reduction": latency_reduction,
            "bram_increase": bram_increase,
            "deadlock": baseline_dumb_deadlock,
        }

        data_relative.append(data)
    else:
        data = {
            "design_name": design,
            "optimizer_name": optimizer,
            "latency_reduction": 0,
            "bram_increase": bram_increase,
            "deadlock": baseline_dumb_deadlock,
        }
        data_relative.append(data)


df_dumb_improvement = pd.DataFrame(data_relative)

optimizers_to_plot = [
    "random_search",
    "discrete_simulated_annealing",
    "heuristic",
    "group_random_search",
    "grouped_discrete_simulated_annealing",
]

df_dumb_improvement = df_dumb_improvement[
    df_dumb_improvement["optimizer_name"].isin(optimizers_to_plot)
]

df_dumb_improvement.to_csv(DIR_DATA / "data_dumb_improvement.csv", index=False)


def geomean(x):
    return np.exp(np.log(x).mean())


avg_bram_increase_by_design = df_dumb_improvement.groupby("optimizer_name")[
    "bram_increase"
].mean()
avg_latency_reduction_by_design = (
    df_dumb_improvement[df_dumb_improvement["deadlock"] is False]
    .groupby("optimizer_name")["latency_reduction"]
    .agg(geomean)
)

report_txt = ""
report_txt += "=== Baseline-Min ===\n"
report_txt += "Average BRAM Increase by Optimizer (ABS):\n"
for optimizer, val in avg_bram_increase_by_design.items():
    report_txt += f"{optimizer}: {val:.5f}\n"
report_txt += "\n"
report_txt += "Average Relative Latency by Optimizer (GEO):\n"
for optimizer, val in avg_latency_reduction_by_design.items():
    report_txt += f"{optimizer}: {val:.5f}\n"

(DIR_DATA / "__report_baseline_min.txt").write_text(report_txt)


font = {"size": 17}
matplotlib.rc("font", **font)
fig, axs = plt.subplots(2, 1, figsize=(18, 5))
# chnage baseline font size for whole figure


n_designs = len(df_dumb_improvement["design_name"].unique())
# color_pallet = sns.color_palette("husl", n_designs)
color_pallet = sns.color_palette()
colors = {
    design: color_pallet[i % len(color_pallet)]
    for i, design in enumerate(df_dumb_improvement["design_name"].unique())
}

# design sored by latency reduction for the grouped discete simaulted annelaing cadse
designs_sorted = df_dumb_improvement[
    df_dumb_improvement["optimizer_name"] == "grouped_discrete_simulated_annealing"
]
print(designs_sorted)
designs_sorted = designs_sorted.sort_values(["latency_reduction"], ascending=False)
designs_sorted_list = designs_sorted["design_name"].unique().tolist()
print(designs_sorted_list)

ax_bram: Axes = axs[1]
ax_latency: Axes = axs[0]

# first axs is for the bram reduction
ax_bram.grid(which="both", linestyle="--", linewidth=0.5)
ax_bram.set_axisbelow(True)

sns.barplot(
    data=df_dumb_improvement,
    x="optimizer_name",
    y="bram_increase",
    hue="design_name",
    order=optimizers_to_plot,
    hue_order=designs_sorted_list,
    ax=ax_bram,
    legend=False,
    palette=colors,
)

ax_bram.set_xlabel(None)


# optimizer_to_name_map for x labels
ax_bram.set_xticklabels(
    [optimizer_to_name_map[x] for x in optimizers_to_plot],
)

optimizers_to_highlight = [
    "group_random_search",
    "grouped_discrete_simulated_annealing",
]

# get x tick locs
x_tick_locs = ax_bram.get_xticks()
for optimizer, loc in zip(optimizers_to_plot, x_tick_locs):
    if optimizer in optimizers_to_highlight:
        height = 125
        ax_bram.text(
            loc,
            height,
            "Minimal Memory Overhead",
            ha="center",
            va="center",
            color="green",
            fontsize=16,
            zorder=10,
        )
        ax_bram.text(
            loc,
            height + 30,
            "✓",
            ha="center",
            va="center",
            # color=bar.get_facecolor(),
            fontsize=36,
            zorder=10,
            color="green",
        )


MAX_BRAM_TO_PLOT = 250
ax_bram.set_ylim(0, top=MAX_BRAM_TO_PLOT)


for j, container in enumerate(ax_bram.containers):
    for i, bar in enumerate(container):
        if isinstance(bar, Rectangle):
            optimizer = optimizers_to_plot[i]
            print(optimizer)
            design = designs_sorted_list[j]
            print(design)
            deadlock = df_dumb_improvement[
                (df_dumb_improvement["optimizer_name"] == optimizer)
                & (df_dumb_improvement["design_name"] == design)
            ]["deadlock"].tolist()[0]
            print(deadlock)

            # if deadlock:
            #     ax_bram.text(
            #         bar.get_x() + bar.get_width() / 2,
            #         bar.get_height() + 0.01,
            #         "x",
            #         ha="center",
            #         va="bottom",
            #         # color=bar.get_facecolor(),
            #         fontsize=16,
            #         zorder=10,
            #         color="red",
            #     )

            bram_increase = df_dumb_improvement[
                (df_dumb_improvement["optimizer_name"] == optimizer)
                & (df_dumb_improvement["design_name"] == design)
            ]["bram_increase"].tolist()[0]

            # if bram_increase == 0 and not deadlock:
            if bram_increase == 0:
                ax_bram.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    "↓",
                    ha="center",
                    va="bottom",
                    # color=bar.get_facecolor(),
                    fontsize=16,
                    zorder=10,
                    color="black",
                )

            if bram_increase > MAX_BRAM_TO_PLOT:
                # add the number at the top of the plot over the bar
                ax_bram.text(
                    bar.get_x() + bar.get_width() / 2,
                    MAX_BRAM_TO_PLOT - 10,
                    f"{bram_increase}",
                    ha="center",
                    va="top",
                    # color=bar.get_facecolor(),
                    fontsize=12,
                    zorder=10,
                    color="black",
                    # add a backgorun box
                    bbox=dict(
                        facecolor="white",
                        # edgecolor="black",
                        boxstyle="round,pad=0.1",
                        alpha=0.9,
                    ),
                )


ax_bram.set_ylabel("BRAM Overhead\n(Baseline-Min)")


ax_latency.grid(which="both", linestyle="--", linewidth=0.5)
ax_latency.set_axisbelow(True)

sns.barplot(
    data=df_dumb_improvement,
    x="optimizer_name",
    y="latency_reduction",
    hue="design_name",
    order=optimizers_to_plot,
    hue_order=designs_sorted_list,
    ax=ax_latency,
    legend=False,
    palette=colors,
)

for j, container in enumerate(ax_latency.containers):
    for i, bar in enumerate(container):
        if isinstance(bar, Rectangle):
            optimizer = optimizers_to_plot[i]
            print(optimizer)
            design = designs_sorted_list[j]
            print(design)
            deadlock = df_dumb_improvement[
                (df_dumb_improvement["optimizer_name"] == optimizer)
                & (df_dumb_improvement["design_name"] == design)
            ]["deadlock"].tolist()[0]
            print(deadlock)

            if deadlock:
                # ax_latency.text(
                #     bar.get_x() + bar.get_width() / 2,
                #     bar.get_height() + 0.01,
                #     "x",
                #     ha="center",
                #     va="bottom",
                #     # color=bar.get_facecolor(),
                #     fontsize=16,
                #     zorder=10,
                #     color="red",
                # )
                # plot a chekc aabove an x
                ax_latency.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    "✓",
                    ha="center",
                    va="bottom",
                    # color=bar.get_facecolor(),
                    fontsize=24,
                    zorder=10,
                    color="green",
                )
                # up arrow
                ax_latency.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    "↑",
                    ha="center",
                    va="bottom",
                    # color=bar.get_facecolor(),
                    fontsize=16,
                    zorder=10,
                    color="black",
                )
                ax_latency.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    "x",
                    ha="center",
                    va="bottom",
                    # color=bar.get_facecolor(),
                    fontsize=16,
                    zorder=10,
                    color="red",
                )


ax_latency.set_ylim(0, top=1.5)
y_ticks = [0, 0.5, 1.0, 1.5]
ax_latency.set_yticks(y_ticks)
ax_latency.set_yticklabels([f"{val:.1f}x" for val in y_ticks])

ax_latency.set_ylabel("Relative Latency\n(Baseline-Min)")


ax_latency.set_xlabel(None)
# optimizer_to_name_map for x labels
ax_latency.set_xticklabels(
    [optimizer_to_name_map[x] for x in optimizers_to_plot],
)


ax_latency.autoscale(False)
ax_latency.hlines(
    1,
    xmin=ax_latency.get_xlim()[0],
    xmax=ax_latency.get_xlim()[1],
    color="black",
    linestyle="--",
    linewidth=2,
    zorder=-10,
)

legned_handels = [
    Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=2,
        label="Baseline-Min Latency",
    ),
]
ax_latency.legend(
    handles=legned_handels,
    # loc="upper right",
    # place in upper right corer using numbers
    loc="upper right",
    bbox_to_anchor=(1, 1.02),
    fontsize=14,
    facecolor="white",
    edgecolor="black",
    framealpha=1,
)

fig.suptitle(
    "(b) Relative to Baseline-Min",
    #  make bold
    weight="bold",
    # move sup title down
    y=0.92,
)

fig.tight_layout(h_pad=0.5)
fig.savefig(DIR_FIGURES / "__dumb_baseline.png", dpi=300, transparent=True)
