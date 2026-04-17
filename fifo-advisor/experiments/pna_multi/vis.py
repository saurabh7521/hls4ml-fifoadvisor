import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DIR_CURRENT = Path(__file__).parent

DIR_FIGURES = DIR_CURRENT / "figures"
if not DIR_FIGURES.exists():
    DIR_FIGURES.mkdir(exist_ok=True)

DIR_DATA = DIR_CURRENT / "data"
if not DIR_DATA.exists():
    DIR_DATA.mkdir(exist_ok=True)


data_baseline = json.loads((DIR_DATA / "data_baseline_multi_eval_pna.json").read_text())

data_baseline_avg_latency = data_baseline["avg_latency"]
data_baseline_avg_bram = data_baseline["avg_bram"]


df_data_all = pd.read_csv(DIR_DATA / "data_multi_eval_pna.csv")


df_data_all["design_index"] = df_data_all["design"].apply(
    lambda x: int(x.split("__")[-1])
)

print(df_data_all.head())


optmizers_to_plot = [
    "multi_group_random_search",
    "multi_heuristic",
    "multi_grouped_discrete_simulated_annealing",
]


optimizer_to_color_map = {
    "multi_group_random_search": "green",
    "multi_heuristic": "red",
    "multi_grouped_discrete_simulated_annealing": "brown",
}

optimizer_to_name_map = {
    "multi_group_random_search": "Multi Grouped Random Search",
    "multi_heuristic": "Multi Greedy Search",
    "multi_grouped_discrete_simulated_annealing": "Multi Grouped Sim. Annealing",
}


def compute_hypervolume(
    ref_point: tuple[int, int], x_vals: list[int], y_vals: list[int]
) -> tuple[float, list[tuple[int, int]]]:
    points: list[tuple[int, int]] = []
    points.append(ref_point)

    # first point is the point with the min x value but same y value as the ref point
    min_x = min(x_vals)
    min_y = ref_point[1]
    points.append((min_x, min_y))
    # then add the rest of the points
    for x, y in zip(x_vals, y_vals):
        points.append((x, y))

    # then the last point is the point with the same x value as the ref point but min y value
    max_x = ref_point[0]
    max_y = min(y_vals)
    points.append((max_x, max_y))
    # then the last point is the ref point
    points.append(ref_point)
    # compute the area
    area = 0.0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        area += (x2 - x1) * (y1 + y2) / 2.0
    area = abs(area)
    return area, points


font = {"size": 18}
matplotlib.rc("font", **font)
fig, ax = plt.subplots(figsize=(8, 5))

ax.grid(which="both", linestyle="--", linewidth=0.5)
ax.set_axisbelow(True)


dfs_for_plot_points = []


for optimizer_name in optmizers_to_plot:
    df_opt = df_data_all[df_data_all["optimizer_name"] == optimizer_name]
    good_eval_indices = []
    eval_indices = df_opt["eval_index"].unique()
    for eval_index in eval_indices:
        df_eval = df_opt[df_opt["eval_index"] == eval_index]
        if not df_eval["deadlock"].any():
            good_eval_indices.append(eval_index)
    df_filtered = df_opt[df_opt["eval_index"].isin(good_eval_indices)]
    df_by_design = {
        design_name: df_design
        for design_name, df_design in df_filtered.groupby("design")
    }
    df_agg = (
        df_filtered.groupby("eval_index")
        .agg({"bram": "mean", "latency": "mean"})
        .reset_index()
    )

    # for design_name, df_design in df_by_design.items():
    #     costs = df_design[["bram", "latency"]].to_numpy()
    #     is_efficient = np.ones(costs.shape[0], dtype=bool)
    #     for i, c in enumerate(costs):
    #         if is_efficient[i]:
    #             is_efficient[is_efficient] = np.any(
    #                 costs[is_efficient] < c, axis=1
    #             )  # Keep any point with a lower cost
    #             is_efficient[i] = True  # And keep self
    #     df_design_pareto = df_design[is_efficient]

    #     ax.plot(
    #         df_design_pareto["bram"],
    #         df_design_pareto["latency"],
    #         marker="x",
    #         linestyle="--",
    #         # label=design_name,
    #         color=color_map[optimizer_name],
    #         alpha=0.5,
    #     )

    costs_agg = df_agg[["bram", "latency"]].to_numpy()
    is_efficient_agg = np.ones(costs_agg.shape[0], dtype=bool)
    for i, c in enumerate(costs_agg):
        if is_efficient_agg[i]:
            is_efficient_agg[is_efficient_agg] = np.any(
                costs_agg[is_efficient_agg] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient_agg[i] = True  # And keep self
    df_agg_pareto = df_agg[is_efficient_agg]
    # sort to make front look nicer
    df_agg_pareto = df_agg_pareto.sort_values(by=["bram", "latency"])
    df_agg_pareto["optimizer_name"] = optimizer_name
    dfs_for_plot_points.append(df_agg_pareto)
    ax.plot(
        df_agg_pareto["bram"],
        df_agg_pareto["latency"],
        marker="o",
        linestyle="--",
        markersize=8,
        color=optimizer_to_color_map[optimizer_name],
        label=f"{optimizer_name}",
    )

ax.plot(
    data_baseline_avg_bram,
    data_baseline_avg_latency,
    marker="X",
    markersize=20,
    linestyle=None,
    label="Baseline-Max",
    color="orange",
    zorder=100,
)

df_plot_points = pd.concat(dfs_for_plot_points, ignore_index=True)
# df_plot_points add baseline point
df_plot_points = pd.concat(
    [
        df_plot_points,
        pd.DataFrame(
            {
                "bram": [data_baseline_avg_bram],
                "latency": [data_baseline_avg_latency],
                "optimizer_name": ["baseline"],
            }
        ),
    ],
    ignore_index=True,
)

max_y = df_plot_points["latency"].max()
max_x = df_plot_points["bram"].max()

max_x = max_x * 1.15
max_y = max_y * 1.003

# vol, points = compute_hypervolume(
#         (max_x, max_y),
#         bram_vals,
#         latency_vals,
#     )
#     print(f"hypervolume for {optimizer} {design}: {vol}")
#     poly_x, poly_y = zip(*points)
#     ax.fill(
#         poly_x,
#         poly_y,
#         alpha=0.08,
#         color=optimizer_to_color_map[optimizer],
#         # label=f"{optimizer} HV: {vol:.2f}",
#         zorder=0,
#     )

for optimizer in optmizers_to_plot:
    df_opt_points = df_plot_points[df_plot_points["optimizer_name"] == optimizer]
    vol, points = compute_hypervolume(
        (max_x, max_y),
        df_opt_points["bram"].tolist(),
        df_opt_points["latency"].tolist(),
    )
    print(f"hypervolume for {optimizer}: {vol}")
    poly_x, poly_y = zip(*points)
    ax.fill(
        poly_x,
        poly_y,
        alpha=0.1,
        color=optimizer_to_color_map[optimizer],
        # label=f"{optimizer} HV: {vol:.2f}",
        zorder=0,
    )


ax.set_xlim(0, max_x)
ax.set_ylim(None, max_y)
# get the current auto y lim
min_y_auto = ax.get_ylim()[0]
new_min_y = min_y_auto - 0.003 * (max_y)
# new_min_y = max(0, new_min_y)
ax.set_ylim(new_min_y, max_y)

y_tick_labels = ax.get_yticks()
y_tick_labels = [f"{int(x / 1000)}K" for x in y_tick_labels]
ax.set_yticklabels(y_tick_labels)

ax.text(
    # place in center relactive to axes
    0.5,
    0.07,
    "*Baseline-Min not shown due to deadlock",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=14,
    # make box around text
    bbox=dict(
        boxstyle="round,pad=0.3",
        edgecolor="black",
        facecolor="white",
        alpha=0.9,
    ),
)

ax.set_xlabel("Average Total FIFO BRAM Usage", labelpad=5)
ax.set_ylabel("Average Latency (Cycles)", labelpad=5)
ax.set_title('Multi-Input Optimization\nPareto Frontiers for "pna"', pad=10)

handels = [
    *[
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=optimizer_to_color_map[optimizer],
            markersize=10,
            # label=optimizer_to_name_map[optimizer].replace("\n", " "),
            label=optimizer_to_name_map[optimizer],
        )
        for optimizer in optmizers_to_plot
    ]
    + [
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="orange",
            markersize=14,
            label="Avg. Baseline-Max",
        ),
        # Line2D(
        #     [0],
        #     [0],
        #     marker="X",
        #     color="w",
        #     markerfacecolor="black",
        #     markersize=14,
        #     label="Baseline-Min",
        # ),
        #     # add a gray star
        #     Line2D(
        #         [0],
        #         [0],
        #         marker="*",
        #         color="w",
        #         markerfacecolor="gray",
        #         markersize=25,
        #         label="Highlighted Pareto Points",
        #     ),
    ]
]
ax.legend(
    handles=handels,
    # loc="lower center",
    loc="upper right",
    fontsize=12,
    facecolor="white",
    edgecolor="black",
    framealpha=1,
    # make line spacing tighter
    labelspacing=0.5,
)

fig.tight_layout()
fig.savefig(DIR_FIGURES / "pareto_fronts_multi_eval_pna_all_optimizers.png", dpi=300)


# make another plot like before but show the pareto frontier for each design separately
# and show all optimizer points for that design
# but only show the pareto frontier for each optimizer separately


fig, ax = plt.subplots(figsize=(8, 8))
ax.grid(which="both", linestyle="--", linewidth=0.5)
ax.set_axisbelow(True)

for optimizer_name in optmizers_to_plot:
    df_opt = df_data_all[df_data_all["optimizer_name"] == optimizer_name]
    good_eval_indices = []
    eval_indices = df_opt["eval_index"].unique()
    for eval_index in eval_indices:
        df_eval = df_opt[df_opt["eval_index"] == eval_index]
        if not df_eval["deadlock"].any():
            good_eval_indices.append(eval_index)
    df_filtered = df_opt[df_opt["eval_index"].isin(good_eval_indices)]
    df_by_design = {
        design_name: df_design
        for design_name, df_design in df_filtered.groupby("design")
    }

    for design_index, (design_name, df_design) in enumerate(df_by_design.items()):
        costs = df_design[["bram", "latency"]].to_numpy()
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] < c, axis=1
                )  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        df_design_pareto = df_design[is_efficient]
        # sort to make front look nicer
        df_design_pareto = df_design_pareto.sort_values(by=["bram", "latency"])

        ax.plot(
            df_design_pareto["bram"],
            df_design_pareto["latency"],
            marker="x",
            linestyle="--",
            markersize=6,
            color=optimizer_to_color_map[optimizer_name],
            label=f"{optimizer_name}" if design_index == 0 else None,
            alpha=0.4,
        )

    df_agg = (
        df_filtered.groupby("eval_index")
        .agg({"bram": "mean", "latency": "mean"})
        .reset_index()
    )

    costs_agg = df_agg[["bram", "latency"]].to_numpy()
    is_efficient_agg = np.ones(costs_agg.shape[0], dtype=bool)
    for i, c in enumerate(costs_agg):
        if is_efficient_agg[i]:
            is_efficient_agg[is_efficient_agg] = np.any(
                costs_agg[is_efficient_agg] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient_agg[i] = True  # And keep self
    df_agg_pareto = df_agg[is_efficient_agg]
    # sort to make front look nicer
    df_agg_pareto = df_agg_pareto.sort_values(by=["bram", "latency"])
    df_agg_pareto["optimizer_name"] = optimizer_name
    dfs_for_plot_points.append(df_agg_pareto)
    ax.plot(
        df_agg_pareto["bram"],
        df_agg_pareto["latency"],
        marker="o",
        linestyle="--",
        markersize=8,
        color=optimizer_to_color_map[optimizer_name],
        label=f"{optimizer_name}",
    )


for beaseline_latency, baseline_bram in zip(
    data_baseline["vals_latency"], data_baseline["vals_bram"]
):
    ax.plot(
        baseline_bram,
        beaseline_latency,
        marker="X",
        markersize=15,
        alpha=0.7,
        linestyle=None,
        label="Baseline-Max",
        color="orange",
        zorder=100,
    )

# ax.set_xlim(0, max_x)
# ax.set_ylim(new_min_y, max_y)
y_tick_labels = ax.get_yticks()
y_tick_labels = [f"{int(x / 1000)}K" for x in y_tick_labels]
ax.set_yticklabels(y_tick_labels)
ax.set_xlabel("Total FIFO BRAM Usage", labelpad=5)
ax.set_ylabel("Latency (Cycles)", labelpad=5)
ax.set_title('Pareto Frontiers of All Inputs for "pna" Kernel', pad=10)
handels = [
    *[
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=optimizer_to_color_map[optimizer],
            markersize=10,
            label=optimizer_to_name_map[optimizer],
        )
        for optimizer in optmizers_to_plot
    ]
    + [
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="orange",
            markersize=14,
            label="Baseline-Max",
        ),
    ]
]
ax.legend(
    handles=handels,
    loc="upper right",
    fontsize=12,
    facecolor="white",
    edgecolor="black",
    framealpha=1,
    labelspacing=0.5,
).set_zorder(10000)
fig.tight_layout()
fig.savefig(
    DIR_FIGURES / "pareto_fronts_multi_eval_pna_all_designs_separate.png", dpi=300
)
