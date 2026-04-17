from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fifo_advisor.opt_env import (
    EvalResult,
    FIFOOptimizer,
    LSEnv,
    is_pareto_efficient_simple,
)
from fifo_advisor.solvers import (
    ROUND_TYPE,
    DiscreteSimulatedAnnealingOptimizer,
    GroupedDiscreteSimulatedAnnealingOptimizer,
    GroupRandomSearchOptimizer,
    HeuristicOptimizer,
    RandomSearchOptimizer,
)


@dataclass(frozen=True)
class SolverSpec:
    cls: type[FIFOOptimizer]
    allowed_kwargs: set[str]


SOLVER_SPECS: dict[str, SolverSpec] = {
    "random": SolverSpec(
        cls=RandomSearchOptimizer,
        allowed_kwargs={"n_samples", "seed"},
    ),
    "group-random": SolverSpec(
        cls=GroupRandomSearchOptimizer,
        allowed_kwargs={"n_samples", "seed"},
    ),
    "heuristic": SolverSpec(
        cls=HeuristicOptimizer,
        allowed_kwargs=set(),
    ),
    "sa": SolverSpec(
        cls=DiscreteSimulatedAnnealingOptimizer,
        allowed_kwargs={
            "maxfun",
            "n_scaling_factors",
            "round_type",
            "init_with_largest",
        },
    ),
    "group-sa": SolverSpec(
        cls=GroupedDiscreteSimulatedAnnealingOptimizer,
        allowed_kwargs={
            "maxfun",
            "n_scaling_factors",
            "round_type",
            "init_with_largest",
        },
    ),
}

ROUND_TYPE_CHOICES = {
    name.lower(): member for name, member in ROUND_TYPE.__members__.items()
}
DEFAULT_OUTPUT_PATH = Path("fifo_advisor_results.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fifo-advisor",
        description=(
            "A tool for optimizing FIFO depths in a high-level synthesis design using"
            " different optimization algorithms and fast co-simulation."
        ),
        usage="fifo-advisor <solution_dir> [options]",
        allow_abbrev=False,
    )
    parser.add_argument("solution_dir", type=Path)
    parser.add_argument(
        "--solver",
        choices=list(SOLVER_SPECS.keys()),
        default="random",
        help="Optimization strategy to run (default: random)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Sample count for random-based solvers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for stochastic solvers.",
    )
    parser.add_argument(
        "--maxfun",
        type=int,
        default=None,
        help="Evaluation budget for simulated annealing solvers.",
    )
    parser.add_argument(
        "--n-scaling-factors",
        type=int,
        default=None,
        help="Number of dual objective scaling factors for simulated annealing solvers.",
    )
    parser.add_argument(
        "--round-type",
        choices=list(ROUND_TYPE_CHOICES.keys()),
        default=None,
        help="Rounding mode for simulated annealing solvers.",
    )
    parser.add_argument(
        "--init-with-largest",
        action="store_true",
        help="Start simulated annealing from the largest FIFO depths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fifo_advisor_results.json"),
        help="Write optimizer evaluations to this JSON file (default: fifo_advisor_results.json).",
    )
    return parser


def collect_solver_kwargs(
    args: argparse.Namespace,
) -> tuple[type[FIFOOptimizer], dict[str, Any]]:
    solver_key = args.solver
    solver_spec = SOLVER_SPECS[solver_key]

    provided_values: dict[str, Any] = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "maxfun": args.maxfun,
        "n_scaling_factors": args.n_scaling_factors,
        "round_type": args.round_type,
    }

    solver_kwargs: dict[str, Any] = {}
    for param, value in provided_values.items():
        if value is None:
            continue
        if param not in solver_spec.allowed_kwargs:
            raise ValueError(
                f"Parameter '{param}' is not supported by solver '{solver_key}'."
            )
        if param == "round_type":
            solver_kwargs[param] = ROUND_TYPE_CHOICES[value]
        else:
            solver_kwargs[param] = value

    if args.init_with_largest:
        if "init_with_largest" not in solver_spec.allowed_kwargs:
            raise ValueError(
                f"Parameter 'init_with_largest' is not supported by solver '{solver_key}'."
            )
        solver_kwargs["init_with_largest"] = True

    return solver_spec.cls, solver_kwargs


def fifo_id_to_name_map_from_env(sim_env: LSEnv) -> dict[int, str]:
    fifo_id_to_name: dict[int, str] = {}
    set_ids = set()
    set_names = set()
    for fifo in sim_env.fifos:
        # fifo_id_to_name[fifo.id] = fifo.name
        fifo_id = fifo.id
        fifo_name = fifo.name
        # check that fifo_id is not already in the map
        if fifo_id in set_ids:
            raise ValueError(f"Duplicate FIFO ID found: {fifo_id}")
        # check that fifo_name is not already in the map
        if fifo_name in set_names:
            raise ValueError(f"Duplicate FIFO name found: {fifo_name}")
        fifo_id_to_name[fifo_id] = fifo_name
        set_ids.add(fifo_id)
        set_names.add(fifo_name)
    return fifo_id_to_name


def main(args: argparse.Namespace) -> None:
    run_with_args(args)


def run_with_args(args: argparse.Namespace) -> dict[str, Any]:
    solution_dir: Path = args.solution_dir
    sim_env = LSEnv(solution_dir)
    fifo_id_to_name_map = fifo_id_to_name_map_from_env(sim_env)
    solver_cls, solver_kwargs = collect_solver_kwargs(args)
    optimizer = solver_cls(sim_env, **solver_kwargs)
    results = optimizer.solve()
    serialized = serialize_eval_results(results, fifo_id_to_name_map)
    serialized["solution_dir"] = str(solution_dir)
    serialized["solver"] = args.solver
    serialized["output"] = str(args.output)
    emit_results(serialized, args.output)
    return serialized


def fifo_config_to_inline_pragma(fifo_name: str, depth: int) -> str:
    return f"#pragma HLS STREAM variable={fifo_name} depth={depth}"


def fifo_config_to_tcl_config(fifo_name: str, depth: int) -> str:
    return (
        f"set_directive_stream -depth {depth} -type fifo {{{{location}}}} {fifo_name}"
    )


def huristic_score(latency, bram, base_latency, base_bram, alpha):
    relative_latency = latency / base_latency
    if base_bram == 0:
        base_bram = 1

    relative_bram = bram / base_bram

    score = alpha * relative_latency + (1 - alpha) * relative_bram
    return score


def serialize_eval_results(
    results: list[EvalResult], fifo_id_to_name_map: dict[int, str]
) -> dict[str, Any]:
    payload = {}
    payload["fifo_id_to_name_map"] = {
        str(fifo_id): name for fifo_id, name in fifo_id_to_name_map.items()
    }
    payload["evaluations"] = []

    pareto_results_mask = is_pareto_efficient_simple(results)

    _pareto_results = [
        result for result, is_pareto in zip(results, pareto_results_mask) if is_pareto
    ]
    # pareto_results_huristic_scores = [
    #     huristic_score(
    #         result.latency,
    #         result.bram_usage_total,
    #         PLACEHOLDER,  # TODO: fix baseline values to be baseline max
    #         PLACEHOLDER,  # TODO: fix baseline values to be baseline max
    #         alpha=0.5,
    #     )
    #     for result in pareto_results
    # ]

    for result, is_pareto in zip(results, pareto_results_mask):
        payload["evaluations"].append(
            {
                "fifo_sizes": {
                    str(fifo_id): depth for fifo_id, depth in result.fifo_sizes.items()
                },
                "fifo_config_inline_pragmas": {
                    fifo_id_to_name_map[fifo_id]: fifo_config_to_inline_pragma(
                        fifo_id_to_name_map[fifo_id], depth
                    )
                    for fifo_id, depth in result.fifo_sizes.items()
                },
                "fifo_config_tcl_commands": {
                    fifo_id_to_name_map[fifo_id]: fifo_config_to_tcl_config(
                        fifo_id_to_name_map[fifo_id], depth
                    )
                    for fifo_id, depth in result.fifo_sizes.items()
                },
                "deadlock": result.deadlock,
                "latency": result.latency,
                "bram_usage_total": result.bram_usage_total,
                "timestamp": result.timestamp,
                "is_pareto_optimal": is_pareto,
            }
        )
    return payload


def emit_results(payload: dict[str, Any], output_path: Path) -> None:
    json_blob = json.dumps(payload, indent=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_blob)


def run_fifo_advisor(
    solution_dir: Path | str,
    solver: str = "random",
    n_samples: int | None = None,
    seed: int | None = None,
    maxfun: int | None = None,
    n_scaling_factors: int | None = None,
    round_type: str | ROUND_TYPE | None = None,
    init_with_largest: bool = False,
    output: Path | str | None = None,
) -> dict[str, Any]:
    if isinstance(round_type, ROUND_TYPE):
        round_type = round_type.name.lower()

    args = argparse.Namespace(
        solution_dir=Path(solution_dir),
        solver=solver,
        n_samples=n_samples,
        seed=seed,
        maxfun=maxfun,
        n_scaling_factors=n_scaling_factors,
        round_type=round_type,
        init_with_largest=init_with_largest,
        output=Path(output) if output is not None else DEFAULT_OUTPUT_PATH,
    )
    return run_with_args(args)


def cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_with_args(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    cli()
