import sys
import types
import importlib.metadata as importlib_metadata

import pytest

if 'h5py' not in sys.modules:
    h5py_stub = types.ModuleType('h5py')

    class _UnusedH5File:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('h5py stub should not be used in test_fifo_advisor_integration.')

    h5py_stub.File = _UnusedH5File
    sys.modules['h5py'] = h5py_stub

if 'quantizers' not in sys.modules:
    quantizers_stub = types.ModuleType('quantizers')
    quantizers_stub.get_fixed_quantizer_np = lambda *args, **kwargs: (lambda data, *qargs, **qkwargs: data)
    sys.modules['quantizers'] = quantizers_stub

if 'hls4ml._version' not in sys.modules:
    version_stub = types.ModuleType('hls4ml._version')
    version_stub.version = '0.0.test'
    version_stub.version_tuple = (0, 0, 'test')
    sys.modules['hls4ml._version'] = version_stub


class _DummyMetadata:
    _headers = [
        ('Requires-Dist', 'calmjs-parse; extra == "quartus-report"'),
        ('Requires-Dist', 'tabulate; extra == "quartus-report"'),
        ('Requires-Dist', 'da4ml>=0.5.2,<0.6; extra == "da"'),
        ('Requires-Dist', 'onnx>=1.4; extra == "onnx"'),
        ('Requires-Dist', 'sympy>=1.13.1; extra == "sr"'),
    ]


importlib_metadata.metadata = lambda name: _DummyMetadata()

from hls4ml.backends import get_backend
from hls4ml.backends.vitis.passes.fifo_depth_optimization import execute_cosim_to_profile_fifos
from hls4ml.utils.config import create_config
from hls4ml.utils.fifo_advisor import get_fifo_advisor_settings


class DummyConfig:
    def __init__(
        self,
        output_dir,
        project_name='myproject',
        project_dir=None,
        backend='Vitis',
        io_type='io_stream',
        flows=None,
        extra_config=None,
    ):
        self._config = {
            'OutputDir': str(output_dir),
            'ProjectName': project_name,
            'ProjectDir': project_dir or f'{project_name}_prj',
            'Backend': backend,
            'IOType': io_type,
        }
        self.flows = flows or []
        if extra_config is not None:
            self._config.update(extra_config)

    def get_config_value(self, key, default=None):
        return self._config.get(key, default)

    def get_output_dir(self):
        return self._config['OutputDir']

    def get_project_name(self):
        return self._config['ProjectName']

    def get_project_dir(self):
        return self._config['ProjectDir']


class DummyModel:
    def __init__(self, config):
        self.config = config


class DummyProcess:
    def __init__(self, *args, **kwargs):
        self.returncode = 0

    def communicate(self):
        return None


def test_create_config_stores_fifo_advisor_settings():
    config = create_config(
        backend='Vitis',
        fifo_advisor=True,
        fifo_advisor_config={'solver': 'heuristic', 'seed': 7},
    )

    assert config['FIFOAdvisor'] is True
    assert config['FIFOAdvisorConfig'] == {'solver': 'heuristic', 'seed': 7}


def test_get_fifo_advisor_settings_uses_defaults_and_relative_output(tmp_path):
    solution_dir = tmp_path / 'myproject_prj' / 'solution1'
    solution_dir.mkdir(parents=True)
    model = DummyModel(
        DummyConfig(
            tmp_path,
            extra_config={
                'FIFOAdvisor': True,
                'FIFOAdvisorConfig': {'solver': 'heuristic', 'output': 'custom_results.json'},
            },
        )
    )

    resolved_solution_dir, advisor_config, output_path = get_fifo_advisor_settings(model)

    assert resolved_solution_dir == solution_dir
    assert advisor_config == {'solver': 'heuristic'}
    assert output_path == tmp_path / 'custom_results.json'


def test_get_fifo_advisor_settings_merges_build_time_overrides(tmp_path):
    solution_dir = tmp_path / 'myproject_prj' / 'solution1'
    solution_dir.mkdir(parents=True)
    model = DummyModel(
        DummyConfig(
            tmp_path,
            extra_config={
                'FIFOAdvisor': True,
                'FIFOAdvisorConfig': {'solver': 'random', 'seed': 7},
            },
        )
    )

    _, advisor_config, output_path = get_fifo_advisor_settings(
        model,
        fifo_advisor_config={'solver': 'sa', 'maxfun': 25, 'output': 'sa_results.json'},
    )

    assert advisor_config == {'solver': 'sa', 'seed': 7, 'maxfun': 25}
    assert output_path == tmp_path / 'sa_results.json'


def test_get_fifo_advisor_settings_requires_vitis_io_stream(tmp_path):
    solution_dir = tmp_path / 'myproject_prj' / 'solution1'
    solution_dir.mkdir(parents=True)

    model = DummyModel(
        DummyConfig(
            tmp_path,
            backend='Vivado',
            extra_config={'FIFOAdvisor': True},
        )
    )
    try:
        get_fifo_advisor_settings(model)
    except RuntimeError as exc:
        assert 'Vitis backend' in str(exc)
    else:
        raise AssertionError('Expected FIFOAdvisor to reject non-Vitis backends.')

    model = DummyModel(
        DummyConfig(
            tmp_path,
            io_type='io_parallel',
            extra_config={'FIFOAdvisor': True},
        )
    )
    try:
        get_fifo_advisor_settings(model)
    except RuntimeError as exc:
        assert 'IOType="io_stream"' in str(exc)
    else:
        raise AssertionError('Expected FIFOAdvisor to reject non-stream designs.')


def test_vitis_build_invokes_fifo_advisor_hook(monkeypatch, tmp_path):
    output_dir = tmp_path / 'build'
    output_dir.mkdir()
    model = DummyModel(DummyConfig(output_dir))
    backend = get_backend('Vitis')
    hook_calls = []

    def fake_run_fifo_advisor_if_enabled(model_arg, report, fifo_advisor=None, fifo_advisor_config=None):
        hook_calls.append((model_arg, report, fifo_advisor, fifo_advisor_config))
        updated_report = dict(report)
        updated_report['FIFOAdvisorReport'] = {'Solver': fifo_advisor_config['solver']}
        return updated_report

    monkeypatch.setattr('hls4ml.backends.vitis.vitis_backend.os.system', lambda command: 0)
    monkeypatch.setattr('hls4ml.backends.vitis.vitis_backend.subprocess.Popen', DummyProcess)
    monkeypatch.setattr('hls4ml.backends.vitis.vitis_backend.parse_vivado_report', lambda _: {'CSynthesisReport': {}})
    monkeypatch.setattr(
        'hls4ml.backends.vitis.vitis_backend.run_fifo_advisor_if_enabled',
        fake_run_fifo_advisor_if_enabled,
    )

    report = backend.build(
        model,
        csim=False,
        synth=False,
        cosim=False,
        export=False,
        vsynth=False,
        fifo_advisor=True,
        fifo_advisor_config={'solver': 'heuristic'},
    )

    assert len(hook_calls) == 1
    assert hook_calls[0][0] is model
    assert hook_calls[0][2] is True
    assert hook_calls[0][3] == {'solver': 'heuristic'}
    assert report['FIFOAdvisorReport']['Solver'] == 'heuristic'


def test_vitis_build_warns_when_fifo_advisor_and_hls4ml_fifo_opt_are_both_enabled(monkeypatch, tmp_path):
    output_dir = tmp_path / 'build'
    output_dir.mkdir()
    model = DummyModel(DummyConfig(output_dir, flows=['vitis:fifo_depth_optimization']))
    backend = get_backend('Vitis')

    monkeypatch.setattr('hls4ml.backends.vitis.vitis_backend.os.system', lambda command: 0)
    monkeypatch.setattr('hls4ml.backends.vitis.vitis_backend.subprocess.Popen', DummyProcess)
    monkeypatch.setattr('hls4ml.backends.vitis.vitis_backend.parse_vivado_report', lambda _: {'CSynthesisReport': {}})
    monkeypatch.setattr(
        'hls4ml.backends.vitis.vitis_backend.run_fifo_advisor_if_enabled',
        lambda *args, **kwargs: {'FIFOAdvisorReport': {'Solver': 'heuristic'}},
    )

    with pytest.warns(UserWarning, match='Both hls4ml FIFO optimization and FIFOAdvisor are enabled'):
        backend.build(
            model,
            csim=False,
            synth=False,
            cosim=False,
            export=False,
            vsynth=False,
            fifo_advisor=True,
            fifo_advisor_config={'solver': 'heuristic'},
        )


def test_internal_vitis_fifo_opt_build_disables_fifo_advisor():
    build_calls = []

    class ProfilingModel:
        def write(self):
            return None

        def build(self, **kwargs):
            build_calls.append(kwargs)
            return {}

    execute_cosim_to_profile_fifos(ProfilingModel())

    assert len(build_calls) == 1
    assert build_calls[0]['fifo_opt'] is True
    assert build_calls[0]['fifo_advisor'] is False
