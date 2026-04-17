import asyncio
import os
import pickle
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
from lightningsim.model import Solution
from lightningsim.runner import Runner, RunnerStep
from lightningsim.trace_file import ResolvedTrace


@dataclass
class EvalResult:
    fifo_sizes: dict[int, int]
    deadlock: bool

    latency: float | None
    bram_usage_total: int | None
    # bram_usage_per_fifo: dict[int, int] | None

    timestamp: float | None = None


EvalResults = list[EvalResult]
MultiEvalResults = list[EvalResults]


class LSEnv:
    def __init__(
        self,
        vitis_hls_solution_dir: Path,
        min_fifo_size: int = 2,
        env_vars_extra: dict[str, str] = {},
    ):
        self.vitis_hls_solution_dir = vitis_hls_solution_dir
        self.min_fifo_size = min_fifo_size
        self.env_vars_extra = env_vars_extra

        for key, value in env_vars_extra.items():
            os.environ[key] = value

        try:
            with open(os.path.join(vitis_hls_solution_dir, "trace.pkl"), "rb") as f:
                self.trace_base: ResolvedTrace = pickle.load(f)
                print("Loaded trace from pickle file.")

        except FileNotFoundError:
            solution = Solution(self.vitis_hls_solution_dir)
            runner = Runner(solution)

            runner.steps[RunnerStep.ANALYZING_PROJECT].on_start(
                lambda _: print("Analyzing project...")
            )
            runner.steps[RunnerStep.WAITING_FOR_BITCODE].on_start(
                lambda _: print("Waiting for bitcode to be generated...")
            )
            runner.steps[RunnerStep.GENERATING_SUPPORT_CODE].on_start(
                lambda _: print("Generating support code...")
            )
            runner.steps[RunnerStep.LINKING_BITCODE].on_start(
                lambda _: print("Linking bitcode...")
            )
            runner.steps[RunnerStep.COMPILING_BITCODE].on_start(
                lambda _: print("Compiling bitcode...")
            )
            runner.steps[RunnerStep.LINKING_TESTBENCH].on_start(
                lambda _: print("Linking testbench...")
            )
            runner.steps[RunnerStep.RUNNING_TESTBENCH].on_start(
                lambda _: print("Running testbench...")
            )
            runner.steps[RunnerStep.PARSING_SCHEDULE_DATA].on_start(
                lambda _: print("Parsing schedule data from C synthesis...")
            )
            runner.steps[RunnerStep.RESOLVING_TRACE].on_start(
                lambda _: print("Resolving dynamic schedule from trace...")
            )

            sys.setrecursionlimit(10_000)

            self.trace_base: ResolvedTrace = asyncio.run(runner.run())  # type: ignore
            with open(os.path.join(vitis_hls_solution_dir, "trace.pkl"), "wb") as f:
                pickle.dump(self.trace_base, f)
                print("Saved trace to pickle file.")

        self.simulation_base = self.trace_base.compiled.execute(self.trace_base.params)

        self.fifos = self.trace_base.fifos
        self.num_fifos = len(self.trace_base.fifos)

        self.fifo_sizes_base: dict[int, int | None] = {}
        for fifo in self.fifos:
            fifo_id: int = fifo.id
            fifo_depth: int | None = self.trace_base.params.fifo_depths[fifo_id]
            self.fifo_sizes_base[fifo_id] = fifo_depth

    def eval_solution_single(self, x: dict[int, int]) -> EvalResult:
        base_params = self.trace_base.params

        design_points = [x]

        dse_results = self.trace_base.compiled.dse(base_params, design_points)
        t = time.perf_counter()
        assert len(dse_results) == 1
        dse_result = dse_results[0]

        fifo_sizes = x
        if dse_result.latency is None:
            deadlock = True
            latency = None
            bram_usage_total = None
        else:
            deadlock = False
            latency = dse_result.latency
            bram_usage_total = dse_result.bram_count

        return EvalResult(
            fifo_sizes=fifo_sizes,
            deadlock=deadlock,
            latency=latency,
            bram_usage_total=bram_usage_total,
            timestamp=t,
        )

    def eval_solution_parallel(
        self, x_multiple: list[dict[int, int]]
    ) -> list[EvalResult]:
        base_params = self.trace_base.params

        design_points = x_multiple
        dse_results = self.trace_base.compiled.dse(base_params, design_points)
        t = time.perf_counter()

        results = []
        for dse_result, design_point in zip(dse_results, design_points):
            fifo_sizes = design_point
            if dse_result.latency is None:
                deadlock = True
                latency = None
                bram_usage_total = None
            else:
                deadlock = False
                latency = dse_result.latency
                bram_usage_total = dse_result.bram_count

            results.append(
                EvalResult(
                    fifo_sizes=fifo_sizes,
                    deadlock=deadlock,
                    latency=latency,
                    bram_usage_total=bram_usage_total,
                    timestamp=t,
                )
            )

        return results

    def eval_solution_default(self) -> EvalResult:
        fifo_sizes_base_not_none = {}
        for fifo_id, fifo_size in self.fifo_sizes_base.items():
            if fifo_size is not None:
                fifo_sizes_base_not_none[fifo_id] = fifo_size
            else:
                raise ValueError(
                    f"FIFO size for FIFO {fifo_id} is None. Please set a valid FIFO size."
                )
        return self.eval_solution_single(fifo_sizes_base_not_none)


class FIFOOptimizer(ABC):
    def __init__(self, sim_env: LSEnv):
        self.sim_env: LSEnv = sim_env

    @abstractmethod
    def solve(self) -> list[EvalResult]:
        pass


T_agg_list_val = TypeVar("T_agg_list_val", int, float)

T_fn_agg = Callable[[list[T_agg_list_val]], T_agg_list_val]


class MultiFIFOOptimizer(ABC):
    def __init__(
        self,
        sim_envs: list[LSEnv],
        n_jobs_over_envs: int = 1,
        fn_agg_latency: T_fn_agg = max,
        fn_agg_bram: T_fn_agg = max,
    ):
        self.sim_envs: list[LSEnv] = sim_envs
        self.n_jobs_over_envs = n_jobs_over_envs
        self.fn_agg_latency = fn_agg_latency
        self.fn_agg_bram = fn_agg_bram

    @abstractmethod
    def solve(self) -> MultiEvalResults:
        pass


# class DummyFIFOOptimizer(FIFOOptimizer):
#     def solve(self) -> list[EvalResult]:
#         return []


# def is_pareto_efficient_simple(costs: np.ndarray) -> np.ndarray:
def is_pareto_efficient_simple(eval_results: list[EvalResult]) -> list[bool]:
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    costs = np.zeros((len(eval_results), 2))
    for i, result in enumerate(eval_results):
        if result.deadlock:
            costs[i, 0] = np.inf
            costs[i, 1] = np.inf
        else:
            costs[i, 0] = result.latency
            costs[i, 1] = result.bram_usage_total

    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self

    is_efficient_list = is_efficient.tolist()
    is_efficient_list_bool = [bool(v) for v in is_efficient_list]
    return is_efficient_list_bool
