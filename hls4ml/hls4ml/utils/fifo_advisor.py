from __future__ import annotations

import copy
import importlib
import warnings
from collections.abc import Mapping
from pathlib import Path

FIFO_ADVISOR_ENABLED_KEY = 'FIFOAdvisor'
FIFO_ADVISOR_CONFIG_KEY = 'FIFOAdvisorConfig'
FIFO_ADVISOR_DEFAULT_OUTPUT = 'fifo_advisor_results.json'
FIFO_DEPTH_OPTIMIZATION_FLOW = 'fifo_depth_optimization'
FIFO_ADVISOR_ALLOWED_OPTIONS = {
    'solver',
    'n_samples',
    'seed',
    'maxfun',
    'n_scaling_factors',
    'round_type',
    'init_with_largest',
    'output',
}


def _validate_fifo_advisor_config(config, *, source):
    if config is None:
        return {}

    if not isinstance(config, Mapping):
        raise TypeError(f'{source} must be a mapping, got {type(config).__name__}.')

    normalized = copy.deepcopy(dict(config))
    invalid_keys = sorted(set(normalized) - FIFO_ADVISOR_ALLOWED_OPTIONS)
    if invalid_keys:
        allowed = ', '.join(sorted(FIFO_ADVISOR_ALLOWED_OPTIONS))
        raise ValueError(f'Unsupported FIFOAdvisor options in {source}: {invalid_keys}. Allowed keys: {allowed}.')

    return normalized


def _resolve_fifo_advisor_output(output_dir, output):
    if output is None:
        return output_dir / FIFO_ADVISOR_DEFAULT_OUTPUT

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = output_dir / output_path

    return output_path


def is_fifo_advisor_enabled(model, fifo_advisor=None):
    enabled = model.config.get_config_value(FIFO_ADVISOR_ENABLED_KEY, False) if fifo_advisor is None else fifo_advisor
    if enabled is None or enabled is False:
        return False

    if not isinstance(enabled, bool):
        raise TypeError(f'fifo_advisor must be a boolean, got {type(enabled).__name__}.')

    return True


def has_hls4ml_fifo_depth_optimization(model):
    flows = getattr(model.config, 'flows', None) or []
    return any(str(flow) == FIFO_DEPTH_OPTIMIZATION_FLOW or str(flow).endswith(f':{FIFO_DEPTH_OPTIMIZATION_FLOW}') for flow in flows)


def warn_if_fifo_advisor_runs_after_hls4ml_fifo_opt(model, fifo_opt=False, fifo_advisor=None):
    if not is_fifo_advisor_enabled(model, fifo_advisor=fifo_advisor):
        return

    if model.config.get_config_value('Backend') != 'Vitis':
        return

    if not (fifo_opt or has_hls4ml_fifo_depth_optimization(model)):
        return

    if getattr(model, '_fifo_advisor_fifo_opt_warning_emitted', False):
        return

    warnings.warn(
        'Both hls4ml FIFO optimization and FIFOAdvisor are enabled. hls4ml will finish its FIFO profiling/optimization '
        'first, and FIFOAdvisor will run once afterward on the final build.',
        stacklevel=3,
    )
    model._fifo_advisor_fifo_opt_warning_emitted = True


def get_fifo_advisor_settings(model, fifo_advisor=None, fifo_advisor_config=None):
    if not is_fifo_advisor_enabled(model, fifo_advisor=fifo_advisor):
        return None

    backend = model.config.get_config_value('Backend')
    if backend != 'Vitis':
        raise RuntimeError('FIFOAdvisor integration currently supports only the Vitis backend.')

    if model.config.get_config_value('IOType') != 'io_stream':
        raise RuntimeError('FIFOAdvisor requires IOType="io_stream".')

    stored_config = _validate_fifo_advisor_config(
        model.config.get_config_value(FIFO_ADVISOR_CONFIG_KEY, {}),
        source=FIFO_ADVISOR_CONFIG_KEY,
    )
    override_config = _validate_fifo_advisor_config(fifo_advisor_config, source='fifo_advisor_config')

    merged_config = stored_config
    merged_config.update(override_config)

    output_dir = Path(model.config.get_output_dir()).resolve()
    solution_dir = output_dir / model.config.get_project_dir() / 'solution1'
    if not solution_dir.exists():
        raise FileNotFoundError(f'FIFOAdvisor expected a Vitis solution at "{solution_dir}", but it was not found.')

    output_path = _resolve_fifo_advisor_output(output_dir, merged_config.pop('output', None))

    return solution_dir, merged_config, output_path


def _load_fifo_advisor_runner():
    try:
        module = importlib.import_module('fifo_advisor.main')
    except ModuleNotFoundError as exc:
        if exc.name == 'fifo_advisor':
            raise ModuleNotFoundError(
                'FIFOAdvisor was enabled, but the "fifo_advisor" package is not installed in the active environment.'
            ) from exc
        raise ModuleNotFoundError(
            f'FIFOAdvisor was enabled, but a required dependency is missing: {exc.name}.'
        ) from exc

    run_fifo_advisor = getattr(module, 'run_fifo_advisor', None)
    if run_fifo_advisor is None:
        raise ImportError('The installed fifo_advisor package does not expose run_fifo_advisor().')

    return run_fifo_advisor


def run_fifo_advisor_if_enabled(model, report=None, fifo_advisor=None, fifo_advisor_config=None):
    settings = get_fifo_advisor_settings(model, fifo_advisor=fifo_advisor, fifo_advisor_config=fifo_advisor_config)
    if settings is None:
        return report

    solution_dir, advisor_config, output_path = settings
    run_fifo_advisor = _load_fifo_advisor_runner()
    payload = run_fifo_advisor(solution_dir=solution_dir, output=output_path, **advisor_config)

    evaluations = payload.get('evaluations', [])
    pareto_count = sum(1 for evaluation in evaluations if evaluation.get('is_pareto_optimal'))

    report_payload = {} if report is None else dict(report)
    report_payload['FIFOAdvisorReport'] = {
        'SolutionDir': str(solution_dir),
        'OutputPath': str(output_path),
        'Solver': payload.get('solver', advisor_config.get('solver', 'random')),
        'Evaluations': len(evaluations),
        'ParetoOptimal': pareto_count,
    }
    return report_payload
