import enum
import itertools
import random
from collections import defaultdict
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from typing import Union

import numpy as np
from lightningsim.trace_file import ResolvedStream
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize as minimize_pymoo
from scipy.optimize import Bounds, dual_annealing
from scipy.optimize._optimize import OptimizeResult

from fifo_advisor.opt_env import (
    EvalResult,
    FIFOOptimizer,
    LSEnv,
    MultiEvalResults,
    MultiFIFOOptimizer,
    T_fn_agg,
    is_pareto_efficient_simple,
)


class ROUND_TYPE(enum.Enum):
    FLOOR = enum.auto()
    CEIL = enum.auto()
    FIX = enum.auto()
    TRUNC = enum.auto()
    ROUND = enum.auto()
    RINT = enum.auto()


def round(x: np.ndarray, round_type: ROUND_TYPE) -> np.ndarray:
    match round_type:
        case ROUND_TYPE.FLOOR:
            return np.floor(x)
        case ROUND_TYPE.CEIL:
            return np.ceil(x)
        case ROUND_TYPE.FIX:
            return np.fix(x)
        case ROUND_TYPE.TRUNC:
            return np.trunc(x)
        case ROUND_TYPE.ROUND:
            return np.round(x)
        case ROUND_TYPE.RINT:
            return np.rint(x)
        case _:
            raise ValueError(f"Unknown rounding type: {round_type}")


def compute_dual_obj_scaling_factors(N_points: int) -> np.ndarray:
    first_row = np.linspace(0, 1, N_points, endpoint=True)
    second_row = first_row[::-1]
    return np.vstack((first_row, second_row)).T


def count_configs(sim_env) -> int:
    fifos_dse_space = {}
    for fifo in sim_env.fifos:
        fifo_id = fifo.id
        fifo_depths = sim_env.trace_base.compiled.get_fifo_design_space(
            [fifo_id], fifo.width
        )
        fifos_dse_space[fifo_id] = fifo_depths

    design_space_size = 1
    for fifo_id, fifo_depths in fifos_dse_space.items():
        design_space_size *= len(fifo_depths)
    return design_space_size


def count_configs_grouped(sim_env) -> int:
    fifo_groups = defaultdict(list)
    for fifo in sim_env.fifos:
        fifo_groups[fifo.get_display_name()].append(fifo)

    fifo_groups_depths = {}
    for fifo_group, fifos in fifo_groups.items():
        fifo_groups_depths[fifo_group] = (
            sim_env.trace_base.compiled.get_fifo_design_space(
                [fifo.id for fifo in fifos], fifos[0].width
            )
        )

    design_space_size = 1
    for fifo_id, fifo_depths in fifo_groups_depths.items():
        design_space_size *= len(fifo_depths)
    return design_space_size


def all_same_fifos(sim_envs: list[LSEnv], throw_error=True) -> bool:
    base_fifo_ids = set(fifo.id for fifo in sim_envs[0].fifos)
    for sim_env in sim_envs[1:]:
        current_fifo_ids = set(fifo.id for fifo in sim_env.fifos)
        if current_fifo_ids != base_fifo_ids:
            if throw_error:
                raise ValueError(
                    "FIFO IDs do not match across all simulation environments."
                )
            return False
    return True


def get_fifo_id_to_name_map(sim_env: LSEnv) -> dict[int, str]:
    fifo_id_to_name = {}
    for fifo in sim_env.fifos:
        fifo_id_to_name[fifo.id] = fifo.get_display_name()
    return fifo_id_to_name


# def eval_solution_parallel_over_envs(
#     sim_envs: list[LSEnv], sampled_configs: list[dict[int, int]], n_jobs: int = 1
# ) -> MultiEvalResults:
#     all_results: MultiEvalResults = []

#     def worker(env: LSEnv) -> list[EvalResult]:
#         # print(f"Evaluating in env: {env.vitis_hls_solution_dir}")
#         return env.eval_solution_parallel(sampled_configs)

#     with ThreadPool(processes=n_jobs) as pool:
#         results = pool.map(worker, sim_envs, chunksize=1)
#         all_results.extend(results)

#     return all_results


# def eval_solution_single_over_envs(
#     sim_envs: list[LSEnv], sampled_config: dict[int, int], n_jobs: int = 1
# ) -> list[EvalResult]:
#     all_results: list[EvalResult] = []

#     def worker(env: LSEnv) -> EvalResult:
#         # print(f"Evaluating in env: {env.vitis_hls_solution_dir}")
#         return env.eval_solution_single(sampled_config)

#     with ThreadPool(processes=n_jobs) as pool:
#         results = pool.map(worker, sim_envs, chunksize=1)
#         all_results.extend(results)
#     return all_results


def eval_solution_single_over_envs(
    pool: ThreadPool,
    sim_envs: list[LSEnv],
    sampled_config: dict[int, int],
) -> list[EvalResult]:
    def worker(env: LSEnv) -> EvalResult:
        # print(f"Evaluating in env: {env.vitis_hls_solution_dir}")
        return env.eval_solution_single(sampled_config)

    all_results = pool.map(worker, sim_envs, chunksize=1)
    return all_results


def eval_solution_parallel_over_envs(
    pool: ThreadPool,
    sim_envs: list[LSEnv],
    sampled_configs: list[dict[int, int]],
) -> MultiEvalResults:
    def worker(env: LSEnv) -> list[EvalResult]:
        # print(f"Evaluating in env: {env.vitis_hls_solution_dir}")
        return env.eval_solution_parallel(sampled_configs)

    all_results = pool.map(worker, sim_envs, chunksize=1)
    return all_results


class RandomSearchOptimizer(FIFOOptimizer):
    def __init__(self, sim_env: LSEnv, n_samples: int = 100, seed: int = 7):
        super().__init__(sim_env)

        self.n_samples = n_samples
        self.seed = seed
        self.r = random.Random(seed)

    def solve(self) -> list[EvalResult]:
        fifos_dse_space = {}
        for fifo in self.sim_env.fifos:
            fifo_id = fifo.id
            fifo_depths = self.sim_env.trace_base.compiled.get_fifo_design_space(
                [fifo_id], fifo.width
            )
            fifos_dse_space[fifo_id] = fifo_depths
        # print(fifos_dse_space)

        sampled_configs = []
        for _ in range(self.n_samples):
            sample: dict[int, int] = {}
            for fifo_id, fifo_depths in fifos_dse_space.items():
                sample[fifo_id] = self.r.choice(fifo_depths)
            sampled_configs.append(sample)

        results: list[EvalResult] = self.sim_env.eval_solution_parallel(sampled_configs)

        return results


class MultiRandomSearchOptimizer(MultiFIFOOptimizer):
    def __init__(
        self,
        sim_envs: list[LSEnv],
        n_samples: int = 100,
        seed: int = 7,
        n_jobs_over_envs: int = 1,
        fn_agg_latency: T_fn_agg = max,
        fn_agg_bram: T_fn_agg = max,
    ):
        super().__init__(sim_envs)

        self.sim_envs = sim_envs
        self.n_samples = n_samples
        self.seed = seed
        self.r = random.Random(seed)

        self.n_jobs_over_envs = n_jobs_over_envs
        self.fn_agg_latency = fn_agg_latency
        self.fn_agg_bram = fn_agg_bram

    def solve(self) -> list[list[EvalResult]]:
        thread_pool = ThreadPool(processes=self.n_jobs_over_envs)

        all_same_fifos(self.sim_envs, throw_error=True)

        fifo_dse_spaces = []
        for sim_env in self.sim_envs:
            fifos_dse_space = {}
            for fifo in sim_env.fifos:
                fifo_id = fifo.id
                fifo_depths = sim_env.trace_base.compiled.get_fifo_design_space(
                    [fifo_id], fifo.width
                )
                fifos_dse_space[fifo_id] = fifo_depths
            fifo_dse_spaces.append(fifos_dse_space)

        fifo_dse_space_merged_sets = defaultdict(set)
        for fifo_dse_space in fifo_dse_spaces:
            for fifo_id, fifo_depths in fifo_dse_space.items():
                fifo_dse_space_merged_sets[fifo_id].update(fifo_depths)
        fifo_dse_space_merged = {
            fifo_id: sorted(list(depths))
            for fifo_id, depths in fifo_dse_space_merged_sets.items()
        }

        sampled_configs = []
        for _ in range(self.n_samples):
            sample: dict[int, int] = {}
            for fifo_id, fifo_depths in fifo_dse_space_merged.items():
                sample[fifo_id] = self.r.choice(fifo_depths)
            sampled_configs.append(sample)

        all_results: list[list[EvalResult]] = []

        # for sim_env in self.sim_envs:
        #     results: list[EvalResult] = sim_env.eval_solution_parallel(sampled_configs)
        #     all_results.append(results)

        all_results = eval_solution_parallel_over_envs(
            thread_pool, self.sim_envs, sampled_configs
        )

        thread_pool.close()
        thread_pool.join()

        return all_results


class GroupRandomSearchOptimizer(FIFOOptimizer):
    def __init__(
        self,
        sim_env: LSEnv,
        n_samples: int = 100,
        seed: int = 7,
    ):
        super().__init__(sim_env)

        self.n_samples = n_samples
        self.seed = seed
        self.r = random.Random(seed)

    def solve(self) -> list[EvalResult]:
        fifo_groups = defaultdict(list)
        for fifo in self.sim_env.fifos:
            fifo_groups[fifo.get_display_name()].append(fifo)

        fifo_groups_depths = {}
        for fifo_group, fifos in fifo_groups.items():
            fifo_groups_depths[fifo_group] = (
                self.sim_env.trace_base.compiled.get_fifo_design_space(
                    [fifo.id for fifo in fifos], fifos[0].width
                )
            )

        design_space_size = 1
        for fifo_id, fifo_depths in fifo_groups_depths.items():
            design_space_size *= len(fifo_depths)
        # print("NAME: ", self.sim_env.vitis_hls_solution_dir.parent.name)
        # print("SIZE: ", design_space_size)
        # print(
        #     f"NAME: {self.sim_env.vitis_hls_solution_dir.parent.name}, SIZE: {design_space_size}"
        # )

        sampled_configs = []
        for _ in range(self.n_samples):
            sample: dict[int, int] = {}
            for fifo_group, fifo_depths in fifo_groups_depths.items():
                group_depths = self.r.choice(fifo_depths)
                for fifo in fifo_groups[fifo_group]:
                    sample[fifo.id] = group_depths
            sampled_configs.append(sample)

        results = self.sim_env.eval_solution_parallel(sampled_configs)

        return results


class MultiGroupRandomSearchOptimizer(MultiFIFOOptimizer):
    def __init__(
        self,
        sim_envs: list[LSEnv],
        n_samples: int = 100,
        seed: int = 7,
        n_jobs_over_envs: int = 1,
        fn_agg_latency: T_fn_agg = max,
        fn_agg_bram: T_fn_agg = max,
    ):
        super().__init__(sim_envs)

        self.sim_envs = sim_envs
        self.n_samples = n_samples
        self.seed = seed
        self.r = random.Random(seed)

        self.n_jobs_over_envs = n_jobs_over_envs
        self.fn_agg_latency = fn_agg_latency
        self.fn_agg_bram = fn_agg_bram

    def solve(self) -> MultiEvalResults:
        thread_pool = ThreadPool(processes=self.n_jobs_over_envs)

        all_same_fifos(self.sim_envs, throw_error=True)

        fifo_groups_all: list[defaultdict[str, list[ResolvedStream]]] = []
        for sim_env in self.sim_envs:
            fifo_groups = defaultdict(list)
            for fifo in sim_env.fifos:
                fifo_groups[fifo.get_display_name()].append(fifo)
            fifo_groups_all.append(fifo_groups)

        base_fifo_groups = fifo_groups_all[0]
        for fifo_groups in fifo_groups_all[1:]:
            if sorted(fifo_groups.keys()) != sorted(base_fifo_groups.keys()):
                raise ValueError("FIFO group keys do not match across all sim envs.")
            for key in fifo_groups.keys():
                base_fifo_ids = sorted([fifo.id for fifo in base_fifo_groups[key]])
                current_fifo_ids = sorted([fifo.id for fifo in fifo_groups[key]])
                if base_fifo_ids != current_fifo_ids:
                    raise ValueError(
                        f"FIFO IDs for group {key} do not match across all sim envs."
                    )

        fifo_groups_depths_all: list[dict[str, list[int]]] = []
        for sim_env in self.sim_envs:
            fifo_groups_depths = {}
            for fifo_group, fifos in fifo_groups_all[0].items():
                fifo_groups_depths[fifo_group] = (
                    sim_env.trace_base.compiled.get_fifo_design_space(
                        [fifo.id for fifo in fifos], fifos[0].width
                    )
                )
            fifo_groups_depths_all.append(fifo_groups_depths)

        fifo_groups_depths_merged = {}
        for fifo_group in fifo_groups_all[0].keys():
            merged_depths_set = set()
            for fifo_groups_depths in fifo_groups_depths_all:
                merged_depths_set.update(fifo_groups_depths[fifo_group])
            fifo_groups_depths_merged[fifo_group] = sorted(list(merged_depths_set))

        sampled_configs = []
        for _ in range(self.n_samples):
            sample: dict[int, int] = {}
            for fifo_group, fifo_depths in fifo_groups_depths_merged.items():
                group_depths = self.r.choice(fifo_depths)
                for fifo in fifo_groups_all[0][fifo_group]:
                    sample[fifo.id] = group_depths
            sampled_configs.append(sample)

        results_all_envs = eval_solution_parallel_over_envs(
            thread_pool, self.sim_envs, sampled_configs
        )
        thread_pool.close()
        thread_pool.join()

        return results_all_envs


class DiscreteSimulatedAnnealingOptimizer(FIFOOptimizer):
    def __init__(
        self,
        sim_env: LSEnv,
        maxfun: int = 100,
        n_scaling_factors: int = 8,
        round_type: ROUND_TYPE = ROUND_TYPE.RINT,
        init_with_largest: bool = False,
    ):
        super().__init__(sim_env)
        self.maxfun = maxfun
        self.round_type = round_type
        self.init_with_largest = init_with_largest

        self.fifo_ids = [fifo.id for fifo in self.sim_env.fifos]
        self.n_scaling_factors = n_scaling_factors
        self.dual_objective_scaling_factors = compute_dual_obj_scaling_factors(
            self.n_scaling_factors
        )

        self.fifos_dse_space = {}

        for fifo in self.sim_env.fifos:
            fifo_id = fifo.id
            try:
                fifo_depths = self.sim_env.trace_base.compiled.get_fifo_design_space(
                    [fifo_id], fifo.width
                )
                if fifo_depths == [2]:
                    fifo_depths = [2, 64 * fifo.width]
                print(f"FIFO {fifo_id} design space: {fifo_depths}")
            except Exception:
                fifo_depths = [2, 64 * fifo.width]
            self.fifos_dse_space[fifo_id] = fifo_depths

        self.fifos_dse_space_bounds = []
        for fifo_id in self.fifo_ids:
            bounds = (0, len(self.fifos_dse_space[fifo_id]) - 1)
            self.fifos_dse_space_bounds.append(bounds)

    def solve(self) -> list[EvalResult]:
        results = []

        results_all = []

        for idx in range(self.n_scaling_factors):
            scaling_factor_latency = self.dual_objective_scaling_factors[idx, 0]
            scaling_factor_bram = self.dual_objective_scaling_factors[idx, 1]

            def objective_function(x: np.ndarray) -> float:
                x = round(x, round_type=self.round_type).astype(int)
                # sample = dict(zip(self.fifo_ids, x))  # Directly construct dictionary
                sample = {}
                for fifo_id, x_val in zip(self.fifo_ids, x):
                    selected_fifo_size = self.fifos_dse_space[fifo_id][x_val]
                    sample[fifo_id] = selected_fifo_size

                y = self.sim_env.eval_solution_single(sample)
                results_all.append(y)

                if y.deadlock:
                    return np.inf

                return (
                    scaling_factor_latency * y.latency
                    + scaling_factor_bram * y.bram_usage_total
                )

            bounds = Bounds(
                lb=[lower for lower, _upper in self.fifos_dse_space_bounds],  # type: ignore
                ub=[upper for _lower, upper in self.fifos_dse_space_bounds],  # type: ignore
            )

            x0 = None
            if self.init_with_largest:
                x0 = np.array(
                    [
                        len(self.fifos_dse_space[fifo_id]) - 1
                        for fifo_id in self.fifo_ids
                    ]
                )

            result = dual_annealing(
                objective_function,
                bounds=bounds,
                maxfun=self.maxfun,
                no_local_search=True,
                rng=7,
                x0=x0,
            )
            x_rounded = round(result.x, self.round_type)
            x_python = x_rounded.tolist()
            x_python_int = [int(x) for x in x_python]

            sol_sample = {
                fifo_id: size for fifo_id, size in zip(self.fifo_ids, x_python_int)
            }
            sol_eval_results = self.sim_env.eval_solution_single(sol_sample)
            results.append(sol_eval_results)

        return results_all


class MultiDiscreteSimulatedAnnealingOptimizer(MultiFIFOOptimizer):
    def __init__(
        self,
        sim_envs: list[LSEnv],
        maxfun: int = 100,
        n_scaling_factors: int = 8,
        round_type: ROUND_TYPE = ROUND_TYPE.RINT,
        init_with_largest: bool = False,
        n_jobs_over_envs: int = 1,
        fn_agg_latency: T_fn_agg = max,
        fn_agg_bram: T_fn_agg = max,
    ):
        super().__init__(sim_envs)
        self.sim_envs = sim_envs
        self.maxfun = maxfun
        self.round_type = round_type
        self.init_with_largest = init_with_largest

        self.n_jobs_over_envs = n_jobs_over_envs
        self.fn_agg_latency = fn_agg_latency
        self.fn_agg_bram = fn_agg_bram

        # Ensure all FIFO sets are identical across envs
        all_same_fifos(self.sim_envs, throw_error=True)

        # Use FIFO ids from the first env as the canonical ordering
        self.fifo_ids = [fifo.id for fifo in self.sim_envs[0].fifos]

        self.n_scaling_factors = n_scaling_factors
        self.dual_objective_scaling_factors = compute_dual_obj_scaling_factors(
            self.n_scaling_factors
        )

        # Build a unified discrete design space for each FIFO id
        # by taking the union of all depths across all envs
        self.fifos_dse_space: dict[int, list[int]] = {}
        for fifo in self.sim_envs[0].fifos:
            fifo_id = fifo.id
            fifo_width = fifo.width

            depths_union: set[int] = set()
            for sim_env in self.sim_envs:
                try:
                    ds = sim_env.trace_base.compiled.get_fifo_design_space(
                        [fifo_id], fifo_width
                    )
                    if ds == [2]:
                        ds = [2, 64 * fifo_width]
                except Exception:
                    ds = [2, 64 * fifo_width]
                depths_union.update(ds)

            self.fifos_dse_space[fifo_id] = sorted(depths_union)
            print(
                f"[MultiDiscreteSA] FIFO {fifo_id} unified design space: "
                f"{self.fifos_dse_space[fifo_id]}"
            )

        # Bounds are over indices into each fifo's discrete design space
        self.fifos_dse_space_bounds: list[tuple[int, int]] = []
        for fifo_id in self.fifo_ids:
            bounds = (0, len(self.fifos_dse_space[fifo_id]) - 1)
            self.fifos_dse_space_bounds.append(bounds)

    def solve(self) -> MultiEvalResults:
        thread_pool = ThreadPool(processes=self.n_jobs_over_envs)

        # One list of EvalResults per env
        results_all_envs: MultiEvalResults = [[] for _ in self.sim_envs]

        for idx in range(self.n_scaling_factors):
            scaling_factor_latency = self.dual_objective_scaling_factors[idx, 0]
            scaling_factor_bram = self.dual_objective_scaling_factors[idx, 1]

            def objective_function(x: np.ndarray) -> float:
                # x are continuous indices; round to nearest integer and clamp
                x = round(x, round_type=self.round_type).astype(int)

                # Build FIFO size sample by mapping index -> actual depth
                sample: dict[int, int] = {}
                for fifo_id, x_val in zip(self.fifo_ids, x):
                    # Defensive clamp (should already be within bounds due to Bounds)
                    x_idx = max(0, min(x_val, len(self.fifos_dse_space[fifo_id]) - 1))
                    selected_fifo_size = self.fifos_dse_space[fifo_id][x_idx]
                    sample[fifo_id] = selected_fifo_size

                # Evaluate this sample on all sim envs
                # per_env_results: list[EvalResult] = []
                # for env in self.sim_envs:
                #     res = env.eval_solution_single(sample)
                #     per_env_results.append(res)

                per_env_results = eval_solution_single_over_envs(
                    thread_pool, self.sim_envs, sample
                )

                # Log results per env
                for env_idx, res in enumerate(per_env_results):
                    results_all_envs[env_idx].append(res)

                # If ANY env deadlocks, solution is invalid
                if any(res.deadlock for res in per_env_results):
                    return np.inf

                # Aggregate latency and resource usage as max across envs
                latencies: list[float] = []
                brams: list[int] = []
                for res in per_env_results:
                    # If any metric is missing, treat as invalid
                    if res.latency is None or res.bram_usage_total is None:
                        return np.inf
                    latencies.append(res.latency)
                    brams.append(res.bram_usage_total)
                brams_float = [float(b) for b in brams]

                # max_latency = max(latencies)
                # max_bram = max(brams)
                agg_latency = self.fn_agg_latency(latencies)
                agg_bram = self.fn_agg_bram(brams_float)

                return (
                    scaling_factor_latency * agg_latency
                    + scaling_factor_bram * agg_bram
                )

            bounds = Bounds(
                lb=[lower for (lower, _upper) in self.fifos_dse_space_bounds],  # type: ignore
                ub=[upper for (_lower, upper) in self.fifos_dse_space_bounds],  # type: ignore
            )

            x0 = None
            if self.init_with_largest:
                # Initialize with the largest discrete option index for each FIFO
                x0 = np.array(
                    [
                        len(self.fifos_dse_space[fifo_id]) - 1
                        for fifo_id in self.fifo_ids
                    ]
                )

            # Run dual annealing over the discrete index space
            _result: OptimizeResult = dual_annealing(
                objective_function,
                bounds=bounds,
                maxfun=self.maxfun,
                no_local_search=True,
                rng=7,
                x0=x0,
            )

        thread_pool.close()
        thread_pool.join()

        # We return all evaluations per env (consistent with MultiRandomSearch style)
        return results_all_envs


class GroupedDiscreteSimulatedAnnealingOptimizer(FIFOOptimizer):
    def __init__(
        self,
        sim_env: LSEnv,
        maxfun: int = 100,
        n_scaling_factors: int = 8,
        round_type: ROUND_TYPE = ROUND_TYPE.RINT,
        init_with_largest: bool = False,
    ):
        super().__init__(sim_env)
        self.maxfun = maxfun
        self.round_type = round_type
        self.init_with_largest = init_with_largest

        self.fifo_ids = [fifo.id for fifo in self.sim_env.fifos]
        self.n_scaling_factors = n_scaling_factors
        self.dual_objective_scaling_factors = compute_dual_obj_scaling_factors(
            self.n_scaling_factors
        )

        # self.fifos_dse_space = {}
        # for fifo in self.sim_env.fifos:
        #     fifo_id = fifo.id
        #     fifo_depths = self.sim_env.trace_base.compiled.get_fifo_design_space(
        #         [fifo_id], fifo.width
        #     )
        #     self.fifos_dse_space[fifo_id] = fifo_depths

        # self.fifos_dse_space_bounds = []
        # for fifo_id in self.fifo_ids:
        #     bounds = (0, len(self.fifos_dse_space[fifo_id]) - 1)
        #     self.fifos_dse_space_bounds.append(bounds)

        self.fifo_groups: dict[str, list[ResolvedStream]] = defaultdict(list)
        for fifo in self.sim_env.fifos:
            self.fifo_groups[fifo.get_display_name()].append(fifo)

        self.fifo_group_names = list(self.fifo_groups.keys())

        self.fifo_ids_groups = defaultdict(list)
        for k, v in self.fifo_groups.items():
            for fifo in v:
                self.fifo_ids_groups[k].append(fifo.id)

        self.grouped_fifos_dse_space = {}
        for fifo_group, fifos in self.fifo_groups.items():
            # self.grouped_fifos_dse_space[fifo_group] = (
            #     self.sim_env.trace_base.compiled.get_fifo_design_space(
            #         [fifo.id for fifo in fifos], fifos[0].width
            #     )
            # )
            ds = self.sim_env.trace_base.compiled.get_fifo_design_space(
                [fifo.id for fifo in fifos], fifos[0].width
            )
            if ds == [2]:
                ds: list[int] = [2, 64 * fifos[0].width]  # type: ignore
            self.grouped_fifos_dse_space[fifo_group] = ds

        self.fifo_group_bounds: list[tuple[int, int]] = []
        for fifo_group in self.fifo_group_names:
            bounds = (0, len(self.grouped_fifos_dse_space[fifo_group]) - 1)
            self.fifo_group_bounds.append(bounds)

        # print(self.fifo_group_bounds)

    def solve(self) -> list[EvalResult]:
        results = []

        results_all = []

        for idx in range(self.n_scaling_factors):
            scaling_factor_latency = self.dual_objective_scaling_factors[idx, 0]
            scaling_factor_bram = self.dual_objective_scaling_factors[idx, 1]

            def objective_function(x: np.ndarray) -> float:
                x = round(x, round_type=self.round_type).astype(int)

                sample = {}
                for fifo_group, x_val in zip(self.fifo_group_names, x):
                    selected_fifo_size = self.grouped_fifos_dse_space[fifo_group][x_val]
                    for fifo_id in self.fifo_ids_groups[fifo_group]:
                        sample[fifo_id] = selected_fifo_size

                y = self.sim_env.eval_solution_single(sample)
                results_all.append(y)

                if y.deadlock:
                    return np.inf

                return (
                    scaling_factor_latency * y.latency
                    + scaling_factor_bram * y.bram_usage_total
                )

            bounds = Bounds(
                lb=[lower for lower, _upper in self.fifo_group_bounds],  # type: ignore
                ub=[upper for _lower, upper in self.fifo_group_bounds],  # type: ignore
            )

            x0 = None
            if self.init_with_largest:
                x0 = np.array(
                    [
                        len(self.grouped_fifos_dse_space[fifo_group]) - 1
                        for fifo_group in self.fifo_group_names
                    ]
                )

            result: OptimizeResult = dual_annealing(
                objective_function,
                bounds=bounds,
                maxfun=self.maxfun,
                no_local_search=True,
                rng=7,
                x0=x0,
            )
            x_rounded = round(result.x, self.round_type)
            x_python = x_rounded.tolist()
            x_python_int = [int(x) for x in x_python]

            sol_sample = {
                fifo_id: size for fifo_id, size in zip(self.fifo_ids, x_python_int)
            }
            sol_eval_results = self.sim_env.eval_solution_single(sol_sample)
            results.append(sol_eval_results)

        return results_all


class MultiGroupedDiscreteSimulatedAnnealingOptimizer(MultiFIFOOptimizer):
    def __init__(
        self,
        sim_envs: list[LSEnv],
        maxfun: int = 100,
        n_scaling_factors: int = 8,
        round_type: ROUND_TYPE = ROUND_TYPE.RINT,
        init_with_largest: bool = False,
        n_jobs_over_envs: int = 1,
        fn_agg_latency: T_fn_agg = max,
        fn_agg_bram: T_fn_agg = max,
    ):
        super().__init__(sim_envs)
        self.sim_envs = sim_envs
        self.maxfun = maxfun
        self.round_type = round_type
        self.init_with_largest = init_with_largest
        self.n_jobs_over_envs = n_jobs_over_envs
        self.fn_agg_latency = fn_agg_latency
        self.fn_agg_bram = fn_agg_bram

        # Sanity: same FIFO sets across envs
        all_same_fifos(self.sim_envs, throw_error=True)

        self.n_scaling_factors = n_scaling_factors
        self.dual_objective_scaling_factors = compute_dual_obj_scaling_factors(
            self.n_scaling_factors
        )

        # --------------------------------------------------
        # 1. Build FIFO groups per env and check consistency
        # --------------------------------------------------
        fifo_groups_all: list[defaultdict[str, list[ResolvedStream]]] = []
        for env in self.sim_envs:
            groups: defaultdict[str, list[ResolvedStream]] = defaultdict(list)
            for fifo in env.fifos:
                groups[fifo.get_display_name()].append(fifo)
            fifo_groups_all.append(groups)

        base_fifo_groups = fifo_groups_all[0]

        # Check that group names and membership are consistent across envs
        for groups in fifo_groups_all[1:]:
            if sorted(groups.keys()) != sorted(base_fifo_groups.keys()):
                raise ValueError(
                    "FIFO group keys do not match across all simulation environments."
                )
            for key in groups.keys():
                base_fifo_ids = sorted([fifo.id for fifo in base_fifo_groups[key]])
                current_fifo_ids = sorted([fifo.id for fifo in groups[key]])
                if base_fifo_ids != current_fifo_ids:
                    raise ValueError(
                        f"FIFO IDs for group {key} do not match across all sim envs."
                    )

        # Use first env's grouping as canonical
        self.fifo_groups: dict[str, list[ResolvedStream]] = base_fifo_groups
        self.fifo_group_names: list[str] = list(self.fifo_groups.keys())

        # Map group -> list of FIFO IDs
        self.fifo_ids_groups: dict[str, list[int]] = {}
        for group_name, fifos in self.fifo_groups.items():
            self.fifo_ids_groups[group_name] = [fifo.id for fifo in fifos]

        # --------------------------------------------------
        # 2. Build a unified discrete design space per group
        #    (union of depths across all envs)
        # --------------------------------------------------
        self.grouped_fifos_dse_space: dict[str, list[int]] = {}
        for group_name in self.fifo_group_names:
            depths_union: set[int] = set()
            # same canonical group membership for all envs
            fifos = self.fifo_groups[group_name]
            fifo_ids = [fifo.id for fifo in fifos]
            fifo_width = fifos[0].width

            for env in self.sim_envs:
                try:
                    ds = env.trace_base.compiled.get_fifo_design_space(
                        fifo_ids, fifo_width
                    )
                    if ds == [2]:
                        ds = [2, 64 * fifo_width]
                except Exception:
                    ds = [2, 64 * fifo_width]

                depths_union.update(ds)

            self.grouped_fifos_dse_space[group_name] = sorted(depths_union)
            print(
                f"[MultiGroupedSA] Group '{group_name}' unified design space: "
                f"{self.grouped_fifos_dse_space[group_name]}"
            )

        # Bounds over indices in each group's discrete design space
        self.fifo_group_bounds: list[tuple[int, int]] = []
        for group_name in self.fifo_group_names:
            bounds = (0, len(self.grouped_fifos_dse_space[group_name]) - 1)
            self.fifo_group_bounds.append(bounds)

    def solve(self) -> MultiEvalResults:
        thread_pool = ThreadPool(processes=self.n_jobs_over_envs)

        # One list of EvalResults per env
        results_all_envs: MultiEvalResults = [[] for _ in self.sim_envs]

        for idx in range(self.n_scaling_factors):
            scaling_factor_latency = self.dual_objective_scaling_factors[idx, 0]
            scaling_factor_bram = self.dual_objective_scaling_factors[idx, 1]

            def objective_function(x: np.ndarray) -> float:
                # x are continuous indices; discretize
                x = round(x, round_type=self.round_type).astype(int)

                # Build config: shared group depth across all envs
                sample: dict[int, int] = {}
                for group_name, x_val in zip(self.fifo_group_names, x):
                    # Clamp defensively
                    ds = self.grouped_fifos_dse_space[group_name]
                    x_idx = max(0, min(x_val, len(ds) - 1))
                    selected_fifo_size = ds[x_idx]
                    for fifo_id in self.fifo_ids_groups[group_name]:
                        sample[fifo_id] = selected_fifo_size

                # Evaluate this sample across all envs
                per_env_results = eval_solution_single_over_envs(
                    thread_pool, self.sim_envs, sample
                )

                # Log all EvalResults
                for env_idx, res in enumerate(per_env_results):
                    results_all_envs[env_idx].append(res)

                # If ANY env deadlocks, solution is invalid
                if any(res.deadlock for res in per_env_results):
                    return np.inf

                latencies: list[float] = []
                brams: list[int] = []
                for res in per_env_results:
                    if res.latency is None or res.bram_usage_total is None:
                        return np.inf
                    latencies.append(res.latency)
                    brams.append(res.bram_usage_total)
                brams_float = [float(b) for b in brams]

                agg_latency = self.fn_agg_latency(latencies)
                agg_bram = self.fn_agg_bram(brams_float)

                return (
                    scaling_factor_latency * agg_latency
                    + scaling_factor_bram * agg_bram
                )

            bounds = Bounds(
                lb=[lower for (lower, _upper) in self.fifo_group_bounds],  # type: ignore
                ub=[upper for (_lower, upper) in self.fifo_group_bounds],  # type: ignore
            )

            x0 = None
            if self.init_with_largest:
                x0 = np.array(
                    [
                        len(self.grouped_fifos_dse_space[group_name]) - 1
                        for group_name in self.fifo_group_names
                    ]
                )

            _result: OptimizeResult = dual_annealing(
                objective_function,
                bounds=bounds,
                maxfun=self.maxfun,
                no_local_search=True,
                rng=7,
                x0=x0,
            )

        thread_pool.close()
        thread_pool.join()

        # Return all evaluations per env (same style as MultiDiscreteSimulatedAnnealingOptimizer)
        return results_all_envs


class HeuristicOptimizer(FIFOOptimizer):
    level_sets = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    def solve(self) -> list[EvalResult]:
        all_evals = []
        for level in self.level_sets:
            print(f"Running heuristic optimization for level: {level}...")

            base_depths = {}
            for fifo, fifo_io in self.sim_env.simulation_base.fifo_io.items():
                fifo_id = fifo.id
                base_depths[fifo_id] = max(fifo_io.get_observed_depth(), 2)

            eval_results = self.sim_env.eval_solution_single(base_depths)
            all_evals.append(eval_results)
            assert not eval_results.deadlock

            base_latency = eval_results.latency
            base_bram_usage_total = eval_results.bram_usage_total

            assert base_latency is not None
            assert base_bram_usage_total is not None

            fifo_ids_sorted_by_depth = sorted(
                base_depths.keys(), key=lambda fifo_id: base_depths[fifo_id]
            )

            fifo_ids_larger_than_two = [
                fifo_id
                for fifo_id in fifo_ids_sorted_by_depth
                if base_depths[fifo_id] > 2
            ]

            working_set_of_depths = deepcopy(base_depths)

            for fifo_id in fifo_ids_larger_than_two:
                new_sample = deepcopy(working_set_of_depths)
                new_sample[fifo_id] = 2
                eval_results_case = self.sim_env.eval_solution_single(new_sample)
                all_evals.append(eval_results_case)
                if eval_results_case.deadlock:
                    continue
                assert eval_results_case.latency is not None
                if eval_results_case.latency > base_latency * 1.01:
                    continue

                working_set_of_depths[fifo_id] = 2

            eval_results_final = self.sim_env.eval_solution_single(
                working_set_of_depths
            )
            all_evals.append(eval_results_final)
            assert not eval_results_final.deadlock

        return all_evals


class MultiHeuristicOptimizer(MultiFIFOOptimizer):
    level_sets = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    def _get_fifo_io_by_id(self, env: LSEnv, fifo_id: int):
        # Correct: match FIFO ID inside fifo_io dict, which is keyed by real Fifo objects
        for fifo_obj, fifo_io in env.simulation_base.fifo_io.items():
            if fifo_obj.id == fifo_id:
                return fifo_obj, fifo_io
        raise KeyError(f"Fifo id {fifo_id} not found in simulation_base.fifo_io")

    def solve(self) -> MultiEvalResults:
        thread_pool = ThreadPool(processes=self.n_jobs_over_envs)

        all_same_fifos(self.sim_envs, throw_error=True)

        fifo_ids = [fifo.id for fifo in self.sim_envs[0].fifos]
        results_all_envs: MultiEvalResults = [[] for _ in self.sim_envs]

        for level in self.level_sets:
            print(f"[MultiHeuristic] Running heuristic level={level}")

            # -----------------------------------------------
            # 1. Aggregate observed depths across ALL envs
            # -----------------------------------------------
            aggregated_observed_depth: dict[int, int] = {}

            for fifo_id in fifo_ids:
                max_depth = 0
                for env in self.sim_envs:
                    fifo_obj, fifo_io = self._get_fifo_io_by_id(env, fifo_id)
                    observed = fifo_io.get_observed_depth()
                    max_depth = max(max_depth, observed)

                aggregated_observed_depth[fifo_id] = max(max_depth, 2)

            # -----------------------------------------------
            # 2. Evaluate baseline across all envs
            # -----------------------------------------------
            baseline_results = []
            deadlock = False
            # for env_idx, env in enumerate(self.sim_envs):
            #     r = env.eval_solution_single(aggregated_observed_depth)
            #     results_all_envs[env_idx].append(r)
            #     baseline_results.append(r)
            #     if r.deadlock:
            #         deadlock = True

            baseline_results = eval_solution_single_over_envs(
                thread_pool, self.sim_envs, aggregated_observed_depth
            )
            for env_idx, r in enumerate(baseline_results):
                results_all_envs[env_idx].append(r)
                if r.deadlock:
                    deadlock = True

            if deadlock:
                print("[MultiHeuristic] Baseline deadlocked; skipping level.")
                continue

            for r in baseline_results:
                assert not r.deadlock
                assert r.latency is not None

            latencies = [r.latency for r in baseline_results if r.latency is not None]
            # base_latency = max(latencies)
            base_latency = self.fn_agg_latency(latencies)

            # -----------------------------------------------
            # 3. Sort FIFOs by baseline depth
            # -----------------------------------------------
            fifo_ids_sorted = sorted(
                fifo_ids, key=lambda fid: aggregated_observed_depth[fid]
            )

            fifo_reduce_list = [
                fid for fid in fifo_ids_sorted if aggregated_observed_depth[fid] > 2
            ]

            working_depths = deepcopy(aggregated_observed_depth)

            # -----------------------------------------------
            # 4. Attempt depth reductions
            # -----------------------------------------------
            for fid in fifo_reduce_list:
                trial = deepcopy(working_depths)
                trial[fid] = 2

                per_env_results = []
                deadlock = False
                # for env_idx, env in enumerate(self.sim_envs):
                #     r = env.eval_solution_single(trial)
                #     results_all_envs[env_idx].append(r)
                #     per_env_results.append(r)
                #     if r.deadlock:
                #         deadlock = True
                per_env_results = eval_solution_single_over_envs(
                    thread_pool, self.sim_envs, trial
                )
                for env_idx, r in enumerate(per_env_results):
                    results_all_envs[env_idx].append(r)
                    if r.deadlock:
                        deadlock = True

                if deadlock:
                    continue

                for r in per_env_results:
                    assert not r.deadlock
                    assert r.latency is not None

                latencies = [
                    r.latency for r in per_env_results if r.latency is not None
                ]
                # trial_latency = max(latencies)
                trial_latency = self.fn_agg_latency(latencies)

                if trial_latency > base_latency * (1 + level):
                    continue

                working_depths[fid] = 2

            # -----------------------------------------------
            # 5. Final evaluation
            # -----------------------------------------------
            # for env_idx, env in enumerate(self.sim_envs):
            #     r = env.eval_solution_single(working_depths)
            #     results_all_envs[env_idx].append(r)
            final_results = eval_solution_single_over_envs(
                thread_pool, self.sim_envs, working_depths
            )
            for env_idx, r in enumerate(final_results):
                results_all_envs[env_idx].append(r)

        thread_pool.close()
        thread_pool.join()

        return results_all_envs


T_FIFOOptimizer = Union[
    RandomSearchOptimizer,
    GroupRandomSearchOptimizer,
    HeuristicOptimizer,
    DiscreteSimulatedAnnealingOptimizer,
    GroupedDiscreteSimulatedAnnealingOptimizer,
]

T_MultiFIFOOptimizer = Union[
    MultiRandomSearchOptimizer,
    MultiHeuristicOptimizer,
    MultiDiscreteSimulatedAnnealingOptimizer,
    MultiGroupedDiscreteSimulatedAnnealingOptimizer,
]


############ Other Experimental Optimizers ############


class GroupExhaustiveOptimizer(FIFOOptimizer):
    def __init__(self, sim_env: LSEnv, size_limit: int = 10_000, seed: int = 7):
        super().__init__(sim_env)

        self.size_limit = size_limit
        self.seed = seed
        self.r = random.Random(seed)

    def solve(self) -> list[EvalResult]:
        fifo_groups = defaultdict(list)
        for fifo in self.sim_env.fifos:
            fifo_groups[fifo.get_display_name()].append(fifo)

        fifo_groups_depths = {}
        for fifo_group, fifos in fifo_groups.items():
            fifo_groups_depths[fifo_group] = (
                self.sim_env.trace_base.compiled.get_fifo_design_space(
                    [fifo.id for fifo in fifos], fifos[0].width
                )
            )

        design_space_size = 1
        for fifo_group, fifo_depths in fifo_groups_depths.items():
            design_space_size *= len(fifo_depths)

        if design_space_size > self.size_limit:
            raise ValueError(
                f"Design space size {design_space_size} exceeds limit {self.size_limit}. Use a larger size limit or different optimizer."
            )

        fifo_groups_keys = list(fifo_groups_depths.keys())
        fifo_groups_values = list(fifo_groups_depths.values())

        combos = itertools.product(
            *fifo_groups_values,
        )

        samples = []
        for combo in combos:
            sample: dict[int, int] = {}
            for fifo_group, fifo_depths in zip(fifo_groups_keys, combo):
                for fifo in fifo_groups[fifo_group]:
                    sample[fifo.id] = fifo_depths
            samples.append(sample)

        assert len(samples) == design_space_size, (
            "mismatch in computed design space size and samples generated size"
        )

        results = self.sim_env.eval_solution_parallel(samples)

        return results


class SimulatedAnnealingOptimizer(FIFOOptimizer):
    def __init__(
        self,
        sim_env: LSEnv,
        maxfun: int = 100,
        round_type: ROUND_TYPE = ROUND_TYPE.RINT,
        init_pool: list[EvalResult] | None = None,
    ):
        super().__init__(sim_env)
        self.maxfun = maxfun
        self.round_type = round_type
        self.init_pool = init_pool

        self.fifo_ids = [fifo.id for fifo in self.sim_env.fifos]
        self.n_scaling_factors = 8
        self.dual_objective_scaling_factors = compute_dual_obj_scaling_factors(
            self.n_scaling_factors
        )

    def solve(self) -> list[EvalResult]:
        results = []

        results_all = []

        sampled_x0: list[list[int]] | list[None] = []
        if self.init_pool is not None:
            sampled_eval_results = random.choices(
                self.init_pool, k=self.n_scaling_factors
            )
            sampled_configs = [r.fifo_sizes for r in sampled_eval_results]
            sampled_x0 = []
            for sample_config in sampled_configs:
                x0 = [sample_config[fifo_id] for fifo_id in self.fifo_ids]
                sampled_x0.append(x0)  # type: ignore
        else:
            sampled_x0 = [None for _ in range(self.n_scaling_factors)]

        for idx in range(self.n_scaling_factors):
            scaling_factor_latency = self.dual_objective_scaling_factors[idx, 0]
            scaling_factor_bram = self.dual_objective_scaling_factors[idx, 1]

            def objective_function(x: np.ndarray) -> float:
                x = round(x, self.round_type).astype(int)
                sample = dict(zip(self.fifo_ids, x))  # Directly construct dictionary

                y = self.sim_env.eval_solution_single(sample)
                results_all.append(y)

                if y.deadlock:
                    return np.inf

                return (
                    scaling_factor_latency * y.latency
                    + scaling_factor_bram * y.bram_usage_total
                )

            bounds = Bounds(
                lb=np.full_like((self.sim_env.num_fifos,), self.sim_env.min_fifo_size),  # type: ignore
                ub=np.array(
                    [self.sim_env.fifo_sizes_base[fifo_id] for fifo_id in self.fifo_ids]  # type: ignore
                ),
            )

            result = dual_annealing(
                objective_function,
                bounds=bounds,
                maxfun=self.maxfun,
                no_local_search=True,
                rng=7,
                x0=sampled_x0[idx],  # type: ignore
            )
            x_rounded = round(result.x, self.round_type)
            x_python = x_rounded.tolist()
            x_python_int = [int(x) for x in x_python]

            sol_sample = {
                fifo_id: size for fifo_id, size in zip(self.fifo_ids, x_python_int)
            }
            sol_eval_results = self.sim_env.eval_solution_single(sol_sample)
            results.append(sol_eval_results)

        return results_all


class FIFOOptProblemInt(Problem):
    def __init__(
        self,
        fifo_optmizer_obj: FIFOOptimizer,
        n_fifos: int,
        fifo_ids: list[int],
        fifo_upper_bounds: dict[int, int],
    ):
        self.fifo_optmizer_obj = fifo_optmizer_obj
        self.n_fifos = n_fifos
        self.fifo_ids = fifo_ids
        self.fifo_upper_bounds = fifo_upper_bounds
        assert len(set(fifo_upper_bounds.keys())) == n_fifos, (
            "Must have a fifo upper bound for each fifo"
        )

        fifo_upper_bounds_ordered = [fifo_upper_bounds[fifo_id] for fifo_id in fifo_ids]

        super().__init__(
            n_var=self.n_fifos,
            n_obj=2,
            n_ieq_constr=1,
            xl=2 * np.ones(self.n_fifos),
            xu=np.array(fifo_upper_bounds_ordered),
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        fifo_sizes = []
        for solution in x:
            fifo_sizes.append(
                {fifo_id: size for fifo_id, size in zip(self.fifo_ids, solution)}
            )
        results = self.fifo_optmizer_obj.sim_env.eval_solution_parallel(fifo_sizes)
        F = np.zeros((len(results), 2))
        G = np.zeros((len(results), 1))
        for i, result in enumerate(results):
            if result.deadlock:
                F[i] = [np.inf, np.inf]
                G[i][0] = 1
            else:
                F[i] = [result.latency, result.bram_usage_total]
                G[i][0] = -1
        out["F"] = F
        out["G"] = G


class ResultsHistoryTracker(Callback):
    def __init__(self, fifo_optmizer_obj: FIFOOptimizer, fifo_ids: list[int]):
        super().__init__()
        self.fifo_optmizer_obj = fifo_optmizer_obj
        self.fifo_ids = fifo_ids
        self.all_results: list[EvalResult] = []

    def notify(self, algorithm):
        X = algorithm.pop.get("X")
        fifo_configs = []
        for x in X:
            fifo_sizes = {fifo_id: size for fifo_id, size in zip(self.fifo_ids, x)}
            fifo_configs.append(fifo_sizes)
        results = self.fifo_optmizer_obj.sim_env.eval_solution_parallel(fifo_configs)
        self.all_results.extend(results)


class GAOptimizer(FIFOOptimizer):
    def __init__(
        self,
        sim_env: LSEnv,
        seed: int = 7,
        n_gen: int = 10,
        pop_size: int = 100,
    ):
        super().__init__(
            sim_env,
        )

        self.seed = seed
        self.n_gen = n_gen
        self.pop_size = pop_size

        # check that all values in fifo_sizes_base are not none
        if any(
            fifo_size is None for fifo_size in self.sim_env.fifo_sizes_base.values()
        ):
            raise ValueError(
                "All fifo sizes must have a default value to have some kind of upper bound for the optimization."
            )

        self.fifo_ids = [fifo.id for fifo in self.sim_env.fifos]

        self.problem = FIFOOptProblemInt(
            self,
            n_fifos=self.sim_env.num_fifos,
            fifo_ids=self.fifo_ids,
            fifo_upper_bounds=self.sim_env.fifo_sizes_base,  # type: ignore
        )

        self.algorithm = NSGA2(
            pop_size=self.pop_size,
            eliminate_duplicates=True,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.5, eta=15, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=0.5, eta=20, vtype=float, repair=RoundingRepair()),
        )

    def solve(self) -> list[EvalResult]:
        results_history_tracker = ResultsHistoryTracker(self, self.fifo_ids)

        _res = minimize_pymoo(
            self.problem,
            self.algorithm,
            termination=("n_gen", self.n_gen),
            seed=self.seed,
            save_history=False,
            callback=results_history_tracker,
            verbose=True,
        )

        return results_history_tracker.all_results


class PSOptimizer(FIFOOptimizer):
    def solve(self) -> list[EvalResult]:
        raise NotImplementedError


class BayesianOptimizer(FIFOOptimizer):
    def solve(self) -> list[EvalResult]:
        raise NotImplementedError


class GroupRandomInitializedSimulatedAnnealingOptimizer(FIFOOptimizer):
    def __init__(
        self,
        sim_env: LSEnv,
        maxfun: int = 50,
        round_type: ROUND_TYPE = ROUND_TYPE.RINT,
        n_samples: int = 1000,
    ):
        super().__init__(sim_env)
        self.maxfun = maxfun
        self.round_type = round_type
        self.n_samples = n_samples

    def solve(self) -> list[EvalResult]:
        self.opt_grouped_random_search = GroupRandomSearchOptimizer(
            self.sim_env,
            n_samples=self.n_samples,
            seed=7,
        )
        results_random_opt = self.opt_grouped_random_search.solve()
        # filter out any deadlock results
        results_random_opt = [
            result for result in results_random_opt if not result.deadlock
        ]
        assert len(results_random_opt) > 0, (
            "No valid results found in random search optimization."
        )

        pareto_mask = is_pareto_efficient_simple(results_random_opt)
        results_random_opt = [
            result for result, mask in zip(results_random_opt, pareto_mask) if mask
        ]

        all_results: list[EvalResult] = []
        for r in results_random_opt:
            assert not r.deadlock
            assert r.latency is not None
            assert r.bram_usage_total is not None

            self.opt_simulated_annealing = SimulatedAnnealingOptimizer(
                self.sim_env,
                maxfun=self.maxfun,
                round_type=self.round_type,
                init_pool=[r],
            )
            results_simulated_annealing = self.opt_simulated_annealing.solve()
            all_results.extend(results_simulated_annealing)
        return all_results
