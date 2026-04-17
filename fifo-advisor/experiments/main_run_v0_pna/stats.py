import itertools
from multiprocessing.pool import ThreadPool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    "init_simulated_annealing": "cyan",
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


def compute_area_under_curve(
    x_vals: list[int], y_vals: list[int]
) -> tuple[float, list[tuple[int, int]]]:
    # build a polygon
    points: list[tuple[int, int]] = []
    # start with (0, 0)
    points.append((0, 0))
    # get the y value of the first point
    y_first = y_vals[0]
    # add the first point
    points.append((0, y_first))
    # then add the rest of the points
    for x, y in zip(x_vals, y_vals):
        points.append((x, y))
    # add the last point
    points.append((x_vals[-1], 0))
    # add the origin point
    points.append((0, 0))
    # compute the area
    area = 0.0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        area += (x2 - x1) * (y1 + y2) / 2.0
    return area, points


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


best_pareto_solution = {}

# for each design and optmizer find the lowest latency bram product and compute the speed up oevr the baseline
data_improvement = []
data_hypervolume = []
for design, optimizer in all_cases:
    baseline_data_case = df_baseline[(df_baseline["design_name"] == design)].copy()
    assert len(baseline_data_case) == 1

    optimizer_data_case = df_points[
        (df_points["design_name"] == design)
        & (df_points["optimizer_name"] == optimizer)
    ].copy()

    # sort pereto front
    optimizer_data_case = optimizer_data_case.sort_values(
        ["latency", "bram"], ascending=[False, True]
    )

    baseline_latency = baseline_data_case["latency"].values[0].item()
    baseline_bram = baseline_data_case["bram"].values[0].item()
    baseline_modified_product = (baseline_bram + 1) * baseline_latency
    baseline_modified_distance = ((baseline_bram + 1) ** 2 + baseline_latency**2) ** 0.5

    optimizer_data_case["modified_product"] = (
        optimizer_data_case["bram"] + 1
    ) * optimizer_data_case["latency"]

    best_modified_product = optimizer_data_case["modified_product"].min()
    best_modified_product_row = optimizer_data_case[
        optimizer_data_case["modified_product"] == best_modified_product
    ]

    def euclidean_distance_modified(row):
        return ((row["bram"] + 1) ** 2 + row["latency"] ** 2) ** 0.5

    optimizer_data_case["modified_zero_distance"] = optimizer_data_case.apply(
        euclidean_distance_modified, axis=1
    )
    best_modified_distance = optimizer_data_case["modified_zero_distance"].min()
    best_modified_distance_row = optimizer_data_case[
        optimizer_data_case["modified_zero_distance"] == best_modified_distance
    ]

    def huristic_score(latency, bram, base_latency, base_bram, ALPHA=0.7):
        relative_latency = latency / base_latency
        if base_bram == 0:
            base_bram = 1
        # else:
        relative_bram = bram / base_bram

        score = ALPHA * relative_latency + (1 - ALPHA) * relative_bram
        return score

    optimizer_data_case["heuristic_score"] = optimizer_data_case.apply(
        lambda row: huristic_score(
            row["latency"],
            row["bram"],
            baseline_latency,
            baseline_bram,
        ),
        axis=1,
    )

    best_heuristic_score = optimizer_data_case["heuristic_score"].min()
    best_heuristic_score_row = optimizer_data_case[
        optimizer_data_case["heuristic_score"] == best_heuristic_score
    ]
    baseline_heuristic_score = huristic_score(
        baseline_latency,
        baseline_bram,
        baseline_latency,
        baseline_bram,
    )

    improvement_product = baseline_modified_product / best_modified_product
    improvement_distance = baseline_modified_distance / best_modified_distance
    improvement_heuristic = baseline_heuristic_score / best_heuristic_score

    df_points_for_this_design_all_optimizers = df_points[
        df_points["design_name"] == design
    ].copy()

    ref_y = df_points_for_this_design_all_optimizers["latency"].max() * 1.1
    ref_x = baseline_bram * 1.1
    ref_point = (ref_x, ref_y)
    vol, _points = compute_hypervolume(
        ref_point,
        optimizer_data_case["bram"].to_list(),
        optimizer_data_case["latency"].to_list(),
    )

    best_pareto_solution[(design, optimizer)] = {
        "modified_product": {
            "bram": best_modified_product_row["bram"].values[0].item(),
            "latency": best_modified_product_row["latency"].values[0].item(),
            "metric_val": best_modified_product,
            "baseline_improvement": improvement_product,
        },
        "modified_distance": {
            "bram": best_modified_distance_row["bram"].values[0].item(),
            "latency": best_modified_distance_row["latency"].values[0].item(),
            "metric_val": best_modified_distance,
            "baseline_improvement": improvement_distance,
        },
        "heuristic_score": {
            "bram": best_heuristic_score_row["bram"].values[0].item(),
            "latency": best_heuristic_score_row["latency"].values[0].item(),
            "metric_val": best_heuristic_score,
            "baseline_improvement": improvement_heuristic,
        },
    }

    improvement_row = {
        "design_name": design,
        "optimizer_name": optimizer,
        "improvement_product": improvement_product,
        "improvement_distance": improvement_distance,
        "best_modified_product": best_modified_product,
        "improvement_heuristic": improvement_heuristic,
    }

    data_improvement.append(improvement_row)

    hypervolume_row = {
        "design_name": design,
        "optimizer_name": optimizer,
        "hypervolume": vol,
    }
    data_hypervolume.append(hypervolume_row)

df_improvement = pd.DataFrame(data_improvement)
df_hypervolume = pd.DataFrame(data_hypervolume)


df_improvement_pivot = df_improvement.pivot(
    index="design_name", columns="optimizer_name", values="improvement_heuristic"
)
df_improvement_pivot_for_latex = df_improvement_pivot.copy()
df_improvement_pivot_for_latex = df_improvement_pivot_for_latex.rename(
    columns=optimizer_to_name_map
)
# df_improvement_pivot_for_latex = df_improvement_pivot_for_latex[
#     [
#         "Random Search",
#         "Group Random Search",
#         "Heuristic",
#         "Init Simulated Annealing",
#     ]
# ]
df_improvement_pivot_for_latex = df_improvement_pivot_for_latex.reset_index()
df_improvement_pivot_for_latex = df_improvement_pivot_for_latex.rename(
    columns={"design_name": "Design Name"}
)

txt_latex = df_improvement_pivot_for_latex.to_latex(
    index=False,
    float_format="%.2fx",
    column_format="l" + "c" * len(df_improvement_pivot_for_latex.columns),
    escape=True,
    label="tab:improvement",
    caption="Improvement of the best solution found by each optimizer over the baseline. The improvement is calculated as the ratio of the baseline modified product to the best modified product found by each optimizer. The modified product is defined as (bram + 1) * latency.",
    position="th",
)

with open(DIR_DATA / "tab_improvement.tex", "w") as f:
    f.write(txt_latex)

optimizers_to_plot = [
    # "random_search",
    "discrete_simulated_annealing",
    "heuristic",
    "group_random_search",
    "grouped_discrete_simulated_annealing",
]


improvement_plot_data = []
for (design, optimizer), data in best_pareto_solution.items():
    baseline_row = df_baseline[df_baseline["design_name"] == design].copy()
    assert len(baseline_row) == 1
    improvement_plot_data.append(
        {
            "design_name": design,
            "optimizer_name": optimizer,
            "bram": data["heuristic_score"]["bram"],
            "latency": data["heuristic_score"]["latency"],
            "slowdown": (data["heuristic_score"]["latency"])
            / baseline_row["latency"].values[0].item(),
            "relative_bram_usage": (
                (
                    data["heuristic_score"]["bram"]
                    / baseline_row["bram"].values[0].item()
                )
                if baseline_row["bram"].values[0].item() > 0
                else 0
            ),
        }
    )
df_improvement_plot = pd.DataFrame(improvement_plot_data)
# if optimizer_name is not in optimizers_to_plot remove it
df_improvement_plot = df_improvement_plot[
    df_improvement_plot["optimizer_name"].isin(optimizers_to_plot)
]


# design sored by latency reduction for the grouped discete simaulted annelaing cadse
designs_sorted = df_improvement_plot[
    df_improvement_plot["optimizer_name"] == "discrete_simulated_annealing"
]
print(designs_sorted)
designs_sorted = designs_sorted.sort_values(["slowdown"], ascending=False)
designs_sorted_list = designs_sorted["design_name"].unique().tolist()
print(designs_sorted_list)


def geomean(x):
    return np.exp(np.log(x).mean())


avg_bram_reduction_by_design = df_improvement_plot.groupby("optimizer_name")[
    "relative_bram_usage"
].mean()
avg_latency_reduction_by_design = df_improvement_plot.groupby("optimizer_name")[
    "slowdown"
].agg(geomean)

report_txt = ""
report_txt += "=== Baseline-Max ===\n"
report_txt += "Average BRAM Reduction by Optimizer (ABS):\n"
for optimizer, val in avg_bram_reduction_by_design.items():
    report_txt += f"{optimizer}: {val:.8f}\n"
report_txt += "\n"
report_txt += "Average Relative Latency by Optimizer (GEO):\n"
for optimizer, val in avg_latency_reduction_by_design.items():
    report_txt += f"{optimizer}: {val:.8f}\n"

(DIR_DATA / "__report_baseline_max.txt").write_text(report_txt)


font = {"size": 17}
matplotlib.rc("font", **font)
fig, axs = plt.subplots(2, 1, figsize=(18, 5.4))
# chnage baseline font size for whole figure


n_designs = len(df_improvement_plot["design_name"].unique())
# color_pallet = sns.color_palette("husl", n_designs)
color_pallet = sns.color_palette()
colors = {
    design: color_pallet[i % len(color_pallet)]
    for i, design in enumerate(df_improvement_plot["design_name"].unique())
}


ax_bram: Axes = axs[0]
ax_latency: Axes = axs[1]

# first axs is for the bram reduction
ax_bram.grid(which="both", linestyle="--", linewidth=0.5)
ax_bram.set_axisbelow(True)


sns.barplot(
    data=df_improvement_plot,
    x="optimizer_name",
    y="relative_bram_usage",
    hue="design_name",
    order=optimizers_to_plot,
    hue_order=designs_sorted_list,
    # palette=optimizer_to_color_map.values(),
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

optimizer_to_height = {
    "group_random_search": 0.5,
    "grouped_discrete_simulated_annealing": 0.4,
}

x_tic_locs = ax_bram.get_xticks()
# get the x tick labels
for optimizer, loc in zip(optimizers_to_plot, x_tic_locs):
    if optimizer in optimizers_to_highlight:
        height = optimizer_to_height[optimizer]
        ax_bram.text(
            loc,
            height,
            "Best Memory Reduction",
            ha="center",
            va="center",
            color="green",
            fontsize=16,
            zorder=10,
        )
        ax_bram.text(
            loc,
            height + 0.15,
            "✓",
            ha="center",
            va="center",
            # color=bar.get_facecolor(),
            fontsize=36,
            zorder=10,
            color="green",
        )


for j, container in enumerate(ax_bram.containers):
    for i, bar in enumerate(container):
        if isinstance(bar, Rectangle):
            if bar.get_height() == 0:
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

ax_bram.autoscale(False)
ax_bram.hlines(
    1,
    xmin=ax_bram.get_xlim()[0],
    xmax=ax_bram.get_xlim()[1],
    color="black",
    linestyle="--",
    linewidth=2,
)

legned_handels = [
    Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=2,
        label="Baseline BRAM Usage",
    ),
]
ax_bram.legend(
    handles=legned_handels,
    loc="upper right",
    # shift down a bti
    # bbox_to_anchor=(1.0, 0.9),
    fontsize=16,
    facecolor="white",
    edgecolor="black",
    framealpha=1,
)


ax_bram.set_ylim(0, 1.1)
ax_bram.set_yticks(np.arange(0, 1.2, 0.2))
ax_bram.set_yticklabels(
    [f"{int(x * 100)}%" if x > 0 else "0%" for x in np.arange(0, 1.2, 0.2)]
)
# ax_bram.set_title("Relative FIFO BRAM Usage Compared to Baseline Design")
# ax_bram.set_ylabel("BRAM Reduction\n(Baseline-Max)")
ax_bram.set_ylabel("BRAM Usage\n(Baseline-Max)")

ax_latency.grid(which="both", linestyle="--", linewidth=0.5)
ax_latency.set_axisbelow(True)


sns.barplot(
    data=df_improvement_plot,
    x="optimizer_name",
    y="slowdown",
    hue="design_name",
    order=optimizers_to_plot,
    hue_order=designs_sorted_list,
    # palette=optimizer_to_color_map.values(),
    ax=ax_latency,
    legend=False,
    palette=colors,
)

ax_latency.set_xticklabels(
    [optimizer_to_name_map[x] for x in optimizers_to_plot],
)
ax_latency.set_xlabel(None)

# get location of x ticks
opts_to_highlight = [
    "heuristic",
    "group_random_search",
    "grouped_discrete_simulated_annealing",
]
x_ticks = ax_latency.get_xticks()
for optimizer, loc in zip(optimizers_to_plot, x_ticks):
    if optimizer in opts_to_highlight:
        # highlight the bar
        ax_latency.text(
            loc,
            1.08,
            "Minimal Latency Overhead",
            ha="center",
            va="bottom",
            color="green",
            fontsize=16,
            zorder=10,
        )
        ax_latency.text(
            loc,
            1.18,
            "✓",
            ha="center",
            va="bottom",
            # color=bar.get_facecolor(),
            fontsize=36,
            zorder=10,
            color="green",
        )

# set ticks between 1 and 3 in 0.5 increments
ax_latency.set_yticks(np.arange(0.5, 3.5, 0.5))
ax_latency.set_yticklabels([f"{x:.1f}x" for x in np.arange(0.5, 3.5, 0.5)])
# ax_latency.set_yticks(np.arange(1, 3, 0.5))


ax_latency.set_ylim(0.5, None)
ax_latency.set_ylabel("Relative Latency\n(Baseline-Max)")


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
        label="Baseline Latency",
    ),
]
ax_latency.legend(
    handles=legned_handels,
    loc="upper right",
    fontsize=16,
    # make solid white background
    facecolor="white",
    edgecolor="black",
    framealpha=1,
)

# ax_latency.set_title("Relative Latency Slowdown Compared to Baseline Design")

# ax_latency.autoscale(enable=True)
# # set the y axis to log scale
# ax_bram.autoscale(enable=True)
# fig.tight_layout()
# reduce space between top and botton
# fig.suptitle(
#     "Relative FIFO BRAM Usage and Latency Slowdown of Optimized Point Compared to Baseline Design"
# )

fig.suptitle(
    "(a) Relative to Baseline-Max",
    #  make bold
    weight="bold",
    # move sup title down
    y=0.92,
)

fig.tight_layout(h_pad=0.12)
fig.savefig(DIR_FIGURES / "__baseline_comparison.png", dpi=300, transparent=True)

# exit()

################################################

# df_hypervolume_pivot = df_hypervolume.pivot(
#     index="design_name", columns="optimizer_name", values="hypervolume"
# )

# df_hypervolume_pivot = df_hypervolume_pivot.div(
#     df_hypervolume_pivot["random_search"], axis=0
# )

# # print(df_hypervolume_pivot)


# df_hypervolume_unpivot = df_hypervolume_pivot.reset_index().melt(
#     id_vars="design_name", var_name="optimizer_name", value_name="hypervolume"
# )

# # df_hypervolume_unpivot = df_hypervolume_unpivot[
# #     df_hypervolume_unpivot["optimizer_name"] != "random_search"
# # ]

# # print(df_hypervolume_unpivot)

# optimizers_to_plot_hyper = [o for o in optimizers_to_plot if o != "random_search"]


# # remove those rows
# df_hypervolume_unpivot = df_hypervolume_unpivot[
#     df_hypervolume_unpivot["optimizer_name"].isin(optimizers_to_plot_hyper)
# ]

# # font size
# font = {"size": 20}
# matplotlib.rc("font", **font)
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.grid(which="both", linestyle="--", linewidth=0.5)
# ax.set_axisbelow(True)

# sns.barplot(
#     data=df_hypervolume_unpivot,
#     x="optimizer_name",
#     y="hypervolume",
#     hue="design_name",
#     order=optimizers_to_plot_hyper,
#     palette=colors,
#     ax=ax,
#     legend=False,
# )


# ax.set_xticklabels(
#     [optimizer_to_name_map[x] for x in optimizers_to_plot_hyper],
# )
# ax.set_xlabel(None)

# ax.set_yticks(np.arange(0, 6, 0.5))
# ax.set_yticklabels([f"{x:.1f}x" for x in np.arange(0, 6, 0.5)])
# ax.set_ylim(0, 5.5)
# ax.set_ylabel("Pareto Frontier Hypervolume\n(Rel. to Random Search)", labelpad=15)


# fig.tight_layout()
# fig.savefig(DIR_FIGURES / "__hypervolume.png", dpi=300)


# def geo_mean(iterable):
#     a = np.array(iterable)
#     return a.prod() ** (1.0 / len(a))


# # compute the geomean for each colum
# geomeans = {}
# for col in df_hypervolume_pivot.columns:
#     geomeans[col] = geo_mean(df_hypervolume_pivot[col])

# df_hypervolume_pivot_for_latex = df_hypervolume_pivot.copy()
# df_hypervolume_pivot_for_latex = df_hypervolume_pivot_for_latex.rename(
#     columns=optimizer_to_name_map
# )

# df_hypervolume_pivot_for_latex = df_hypervolume_pivot_for_latex.reset_index()
# df_hypervolume_pivot_for_latex = df_hypervolume_pivot_for_latex.rename(
#     columns={"design_name": "Design Name"}
# )
# txt_latex = df_hypervolume_pivot_for_latex.to_latex(
#     index=False,
#     float_format="%.2fx",
#     column_format="l|" + "c" * len(df_hypervolume_pivot_for_latex.columns),
#     escape=True,
#     label="tab:hypervolume",
#     caption='Relative hypervolume of the pareto front found by each optimizer relative to the baseline "Random Search" optimizer for all designs. '
#     "The higher the hypervolume the better the discovered pareto front. "
#     "The reference point used for the hypervolume calculation is $(1.1*\\text{bram}_\\text{baseline}, 1.1*\\text{latency}_\\text{max})$, where $\\text{bram}_\\text{baseline}$ is the BRAM usage of the baseline design and $\\text{latency}_\\text{max}$ is the maximum latency of all designs.",
#     position="th",
# )

# # create a new row with the geomeans
# row_geomean_txt = ""
# row_geomean_txt += "Geomean & "
# print(geomeans)
# for col in df_hypervolume_pivot.columns[:]:
#     print(f"{col}: {geomeans[col]:.2f}x")
#     row_geomean_txt += f"{geomeans[col]:.2f}x & "
# row_geomean_txt += "\\\\\n"


# txt_latex = txt_latex.replace(
#     "\\bottomrule", "\\hline\n" + row_geomean_txt + "\\bottomrule\n"
# )

# (DIR_DATA / "tab_hypervolume.tex").write_text(txt_latex)


# plot frontiers per solution space
def plot_design(design):
    # plot the baseline
    baseline_data_case = df_baseline[(df_baseline["design_name"] == design)].copy()
    assert len(baseline_data_case) == 1
    baseline_bram = baseline_data_case["bram"].values[0].item()
    baseline_latency = baseline_data_case["latency"].values[0].item()

    baseline_dumb_data_case = df_baseline_dumb[
        (df_baseline_dumb["design_name"] == design)
    ].copy()
    assert len(baseline_dumb_data_case) == 1
    baseline_dumb_bram = baseline_dumb_data_case["bram"].values[0].item()
    baseline_dumb_latency = baseline_dumb_data_case["latency"].values[0].item()
    baseline_dumb_deadlock = baseline_dumb_data_case["deadlock"].values[0].item()

    df_points_for_this_design = df_points[df_points["design_name"] == design].copy()

    # fomat font size
    font = {"size": 18}
    matplotlib.rc("font", **font)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.grid(which="both", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    for optimizer in optimizers_to_plot:
        optimizer_data_case = df_points[
            (df_points["design_name"] == design)
            & (df_points["optimizer_name"] == optimizer)
        ].copy()

        optimizer_data_case = optimizer_data_case.sort_values(
            ["latency", "bram"], ascending=[False, True]
        )
        bram_vals = optimizer_data_case["bram"].tolist()
        latency_vals = optimizer_data_case["latency"].tolist()

        # plot the points
        ax.plot(
            bram_vals,
            latency_vals,
            marker="o",
            linestyle="--",
            markersize=10,
            label=optimizer,
            color=optimizer_to_color_map[optimizer],
            zorder=10,
        )

        # plot the best product point
        metric = "heuristic_score"
        best_point = best_pareto_solution[(design, optimizer)]
        best_bram = best_point[metric]["bram"]
        best_latency = best_point[metric]["latency"]

        ax.plot(
            best_bram,
            best_latency,
            marker="*",
            markersize=20,
            linewidth=12,
            linestyle=None,
            color=optimizer_to_color_map[optimizer],
            zorder=20,
        )

        max_y = df_points_for_this_design["latency"].max()
        max_x = baseline_bram

        if not baseline_dumb_deadlock:
            max_y = max(max_y, baseline_dumb_latency)

        max_x = max_x * 1.15
        max_y = max_y * 1.01

        vol, points = compute_hypervolume(
            (max_x, max_y),
            bram_vals,
            latency_vals,
        )
        print(f"hypervolume for {optimizer} {design}: {vol}")
        poly_x, poly_y = zip(*points)
        ax.fill(
            poly_x,
            poly_y,
            alpha=0.08,
            color=optimizer_to_color_map[optimizer],
            # label=f"{optimizer} HV: {vol:.2f}",
            zorder=0,
        )

    ax.plot(
        baseline_bram,
        baseline_latency,
        marker="X",
        markersize=20,
        linestyle=None,
        label="Baseline-Max",
        color="orange",
        zorder=100,
    )

    if not baseline_dumb_deadlock:
        ax.plot(
            baseline_dumb_bram,
            baseline_dumb_latency,
            marker="X",
            markersize=20,
            linestyle=None,
            label="Baseline-Min",
            color="black",
            zorder=15,
        )

    if baseline_dumb_deadlock:
        # x_lim = ax.get_xlim()
        # x_middle = (x_lim[0] + x_lim[1]) / 2
        # y_lim = ax.get_ylim()
        # y_middle = (y_lim[0] + y_lim[1]) / 2
        ax.text(
            # place in center relactive to axes
            0.5,
            0.05,
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

    ax.set_xlim(0, max_x)
    ax.set_ylim(None, max_y)
    # get the current auto y lim
    min_y_auto = ax.get_ylim()[0]
    new_min_y = min_y_auto - 0.005 * (max_y)
    # new_min_y = max(0, new_min_y)
    ax.set_ylim(new_min_y, max_y)

    # set tick at origin
    # ax.set_yticks(np.arange(15000, max_y + 1, 10000))

    # format y ticks to be in therms of K as in thousands
    # current y tick labels
    y_tick_labels = ax.get_yticks()
    y_tick_labels = [f"{int(x / 1000)}K" for x in y_tick_labels]
    ax.set_yticklabels(y_tick_labels)

    ax.set_xlabel("Total FIFO BRAM Usage", labelpad=5)
    ax.set_ylabel("Latency (Cycles)", labelpad=5)
    ax.set_title(f'Pareto Frontiers for "{design}"', pad=1)

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
            for optimizer in optimizers_to_plot
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
            Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                markerfacecolor="black",
                markersize=14,
                label="Baseline-Min",
            ),
            # add a gray star
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="gray",
                markersize=25,
                label="Highlighted Pareto Points",
            ),
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

    return fig


def parallel_fn_plot_design(design):
    fig = plot_design(design)
    fig.savefig(DIR_FIGURES / f"{design}.png", dpi=300, pad_inches=0)
    plt.close(fig)


N_JOBS = 32


with ThreadPool(N_JOBS) as pool:
    figs = pool.map(parallel_fn_plot_design, designs)
