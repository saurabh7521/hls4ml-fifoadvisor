# hls4ml FIFOAdvisor Integration Study Notes

This document explains the changes made to integrate `hls4ml` and `fifo-advisor` in this repository. The goal is to make it easy to study the implementation file by file and understand how the final workflow behaves.

## High-Level Goal

The target user flow is:

1. Build an `hls4ml` project with `backend='Vitis'` and `io_type='io_stream'`.
2. Let `hls4ml` run its normal HLS build flow.
3. After the final HLS build finishes, automatically invoke FIFOAdvisor on the generated Vitis HLS solution directory.
4. Return a normal `hls4ml` build report, augmented with a small FIFOAdvisor summary.

The solution directory used by the integration is:

```text
<output_dir>/<project_name>_prj/solution1
```

## End-to-End Control Flow

The main control flow now looks like this:

1. User passes `fifo_advisor=True` and optionally `fifo_advisor_config={...}` to an `hls4ml` conversion function.
2. Those settings are stored in the generated `hls4ml` config.
3. The user calls `hls_model.build(...)`.
4. The Vitis backend performs the HLS build as usual.
5. After the build completes, `hls4ml` resolves the Vitis solution path automatically.
6. `hls4ml` imports the local installed `fifo_advisor` Python package and calls it programmatically.
7. FIFOAdvisor writes its JSON results file.
8. `hls4ml` returns its original report plus a `FIFOAdvisorReport` summary.

If the user also enables hls4ml’s built-in FIFO depth optimization flow, the internal profiling builds explicitly disable FIFOAdvisor so that FIFOAdvisor does not run recursively. FIFOAdvisor then runs only once at the end of the final outer build.

## File-By-File Breakdown

### 1. [fifo-advisor/fifo_advisor/main.py](./fifo-advisor/fifo_advisor/main.py)

Context:
This is FIFOAdvisor’s main CLI entry point. Before the integration, it mainly supported command-line usage like:

```bash
fifo-advisor <solution_dir> --solver heuristic
```

Implemented changes:

- Added `DEFAULT_OUTPUT_PATH`.
- Refactored the main execution logic into `run_with_args(args)`.
- Added `run_fifo_advisor(...)` as a Python-callable API.
- Extended the emitted JSON payload to also store:
  - `solution_dir`
  - `solver`
  - `output`
- Updated `main()` and `cli()` to use `run_with_args(...)`.
- Removed the unused `from matplotlib.pylab import pareto` import.

Why:

- The refactor keeps the CLI behavior and the programmatic behavior aligned by funneling both through the same execution path.

Key integration idea:

- `hls4ml` calls `fifo_advisor.main.run_fifo_advisor(...)`.
- This means FIFOAdvisor can be driven directly from the `hls_model.build(...)` path.

### 2. [hls4ml/hls4ml/utils/fifo_advisor.py](./hls4ml/hls4ml/utils/fifo_advisor.py)

Context:
This is the new integration helper module on the `hls4ml` side. It acts as the bridge between `hls4ml` and `fifo-advisor`.

Implemented changes:

- Created a new helper module for all FIFOAdvisor-specific logic.
- Added config key constants:
  - `FIFOAdvisor`
  - `FIFOAdvisorConfig`
- Added validation for supported `fifo_advisor_config` keys.
- Added output-path resolution logic.
- Added `is_fifo_advisor_enabled(...)`.
- Added `has_hls4ml_fifo_depth_optimization(...)`.
- Added `warn_if_fifo_advisor_runs_after_hls4ml_fifo_opt(...)`.
- Added `get_fifo_advisor_settings(...)`.
- Added `_load_fifo_advisor_runner(...)`.
- Added `run_fifo_advisor_if_enabled(...)`.

Why:

- Without a dedicated helper module, the backend build code would become cluttered and hard to maintain.
- This file centralizes:
  - validation
  - path resolution
  - warning logic
  - runtime import of `fifo_advisor`
  - conversion of FIFOAdvisor output into `hls4ml` report format

Important design choices:

- FIFOAdvisor is currently restricted to `Backend == 'Vitis'` and  `IOType == 'io_stream'`.
- The warning is only emitted once per model instance.


### 3. [hls4ml/hls4ml/utils/config.py](./hls4ml/hls4ml/utils/config.py)

Context:
This is where `hls4ml` creates the initial project configuration dictionary used throughout conversion and build.

Implemented changes:

- Extended `create_config(...)` with:
  - `fifo_advisor=None`
  - `fifo_advisor_config=None`
- Stored these in the config dictionary as:
  - `config['FIFOAdvisor']`
  - `config['FIFOAdvisorConfig']`
- Used `copy.deepcopy(...)` when storing `fifo_advisor_config`.

Why:

- The integration needs a place to persist FIFOAdvisor settings onto the model config.
- Once stored in the model config, the backend can still access those settings later during `build(...)`.
- Deep-copying avoids accidental mutation of the user’s original config dictionary.


### 4. [hls4ml/hls4ml/converters/__init__.py](./hls4ml/hls4ml/converters/__init__.py)

Context:
This file exposes the top-level conversion entry points such as:

- `convert_from_keras_model(...)`
- `convert_from_pytorch_model(...)`
- `convert_from_onnx_model(...)`

Implemented changes:

- Added `fifo_advisor` and `fifo_advisor_config` parameters to:
  - `convert_from_keras_model(...)`
  - `convert_from_pytorch_model(...)`
  - `convert_from_onnx_model(...)`
- Forwarded those parameters into `create_config(...)`.
- Updated the docstrings to describe the new parameters.

Why:

- This makes the user-facing API ergonomic.
- The user can now write:

```python
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    backend='Vitis',
    io_type='io_stream',
    fifo_advisor=True,
    fifo_advisor_config={'solver': 'heuristic'},
)
```

instead of having to mutate hidden internal config objects manually.


### 5. [hls4ml/hls4ml/backends/vitis/vitis_backend.py](./hls4ml/hls4ml/backends/vitis/vitis_backend.py)

Context:
This file is the actual Vitis backend implementation. Its `build(...)` method is the exact place where Vitis HLS is launched.

Implemented changes:

- Imported:
  - `run_fifo_advisor_if_enabled`
  - `warn_if_fifo_advisor_runs_after_hls4ml_fifo_opt`
- Extended `build(...)` with:
  - `fifo_advisor=None`
  - `fifo_advisor_config=None`
- Added a warning call before the HLS build begins.
- After `parse_vivado_report(...)`, routed the result through `run_fifo_advisor_if_enabled(...)`.

Why:

- This is a clean place to hook FIFOAdvisor in.
- The Vitis build is what creates the `solution1` directory that FIFOAdvisor needs.


### 6. [hls4ml/hls4ml/backends/vivado/vivado_backend.py](./hls4ml/hls4ml/backends/vivado/vivado_backend.py)

Context:
This is the Vivado backend implementation.

Implemented changes:

- Added the same FIFOAdvisor-related imports used in the Vitis backend.
- Extended `build(...)` with:
  - `fifo_advisor=None`
  - `fifo_advisor_config=None`
- Added the same warning call before build.
- Added the same post-build routing through `run_fifo_advisor_if_enabled(...)`.

Why:

- This keeps the backend API  consistent.
- Even though FIFOAdvisor is currently restricted to Vitis in the helper logic, having the same signature in both backends avoids API divergence.
- The helper will reject non-Vitis FIFOAdvisor runs centrally.

### 7. [hls4ml/hls4ml/backends/vitis/passes/fifo_depth_optimization.py](./hls4ml/hls4ml/backends/vitis/passes/fifo_depth_optimization.py)

Context:
This file implements hls4ml’s built-in Vitis FIFO depth optimization pass.

Implemented changes:

- In the internal profiling build call inside `execute_cosim_to_profile_fifos(...)`, added:
  - `fifo_advisor=False`
- Added comments explaining why this is necessary.

Why:

- The built-in flow uses an internal `model.build(...)` call to perform profiling.
- If FIFOAdvisor were left enabled here, it would fire during the internal profiling build, which is not intended.
- The correct behavior is:
  - internal hls4ml profiling build runs first
  - final outer build finishes
  - FIFOAdvisor runs once at the end to override FIFO optimization by vitis


### 8. [hls4ml/hls4ml/backends/vivado/passes/fifo_depth_optimization.py](./hls4ml/hls4ml/backends/vivado/passes/fifo_depth_optimization.py)

Context:
This is the equivalent built-in FIFO depth optimization pass for the Vivado backend.

Implemented changes:

- In the internal `model.build(...)` call inside `get_vcd_data(...)`, added:
  - `fifo_advisor=False`

Why:

- This mirrors the same suppression pattern used in the Vitis pass.
- It keeps internal profiling builds from accidentally triggering FIFOAdvisor if the user enabled it.

### 9. [hls4ml/docs/advanced/fifo_depth.rst](./hls4ml/docs/advanced/fifo_depth.rst)

Context:
This is the hls4ml advanced documentation page about FIFO depth optimization.

Implemented changes:

- Added a new “FIFOAdvisor Integration” section.
- Documented:
  - Vitis-only support
  - `io_stream` requirement
  - the real target directory: `<output_dir>/<project_name>_prj/solution1`
  - example usage with `fifo_advisor=True`
  - example usage with `fifo_advisor_config={...}`
  - interaction with hls4ml’s built-in FIFO optimization

Why:

- Users studying FIFO-related workflows will naturally land on this page.
- This is the most natural place to explain how FIFOAdvisor differs from hls4ml’s built-in `fifo_opt` flow.


### 10. [hls4ml/test/pytest/test_fifo_advisor_integration.py](./hls4ml/test/pytest/test_fifo_advisor_integration.py)

Context:
This is the focused unit test file added for the new integration.

Implemented changes:

- Added tests for:
  - config storage of FIFOAdvisor settings
  - automatic solution path resolution
  - config override merging
  - backend and `io_stream` validation
  - Vitis build hook invocation
  - warning emission when both hls4ml FIFO optimization and FIFOAdvisor are enabled
  - suppression of FIFOAdvisor inside internal Vitis profiling builds
- Added lightweight stubs for import-time dependencies:
  - `h5py`
  - `quantizers`
  - `hls4ml._version`
  - package metadata lookup

Why:

- The test is used to validate the integration
- The stubs let the test run in a lighter environment without requiring the entire frontend stack during collection.

### 12. [environment.yml](./environment.yml)

Context:
This is the root Conda environment definition for the integrated workspace.

Implemented changes:

- Created a single root environment file.
- Named the environment:
  - `hls4ml-fifoadvisor`
- Added:
  - `python=3.12`
  - `pip`
  - `lightningsim=0.2.6`
- Added pip installation of the root `requirements.txt`.

Why:

- This is what gives the workspace a single shared environment.
- `lightningsim` must be available since FIFOAdvisor depends on it for fast HLS trace analysis.

## Design Decisions and Rationale

### Why use a Python API for FIFOAdvisor instead of invoking the CLI with subprocess?

- A Python API is easier to validate.
- It avoids shell quoting and subprocess management.
- It makes it easier to return structured data directly.

### Why store FIFOAdvisor settings on the `hls4ml` model config?

- The user sets options at conversion time, but execution happens at build time.
- The model config is the natural bridge between those two phases.

### Why only support Vitis?

- FIFOAdvisor is built around Vitis HLS solution structure and LightningSim’s expectations.

### Why suppress FIFOAdvisor during internal profiling builds instead of disabling hls4ml FIFO optimization entirely?

- hls4ml built-in FIFO optimization and FIFOAdvisor are conceptually distinct.
- Users may want both:
  - first let hls4ml do its own built-in sizing flow
  - then let FIFOAdvisor run on the final solution
- Suppression during internal profiling builds preserves both tools without recursive behavior.

### Why emit a warning when both are enabled?

- The warning tells the user exactly what order of operations will happen.

## What the Integration Does Not Yet Do

The current integration:

- automatically launches FIFOAdvisor
- automatically resolves the Vitis solution path
- writes FIFOAdvisor JSON output
- augments the `hls4ml` build report with a summary

It does not yet:

- choose one Pareto point automatically
- rewrite selected FIFO depths back into the `hls4ml` project
- rebuild the project using a chosen FIFOAdvisor result

That would be the next stage if a fully closed-loop optimization flow is needed.

## Verification Performed

Focused integration tests were added and run:

```bash
python -m pytest -q test/pytest/test_fifo_advisor_integration.py
```

Observed result:

```text
7 passed in 0.22s
```

## Practical Summary

The integration is built around three ideas:

1. Accept FIFOAdvisor settings at conversion time.
2. Resolve and invoke FIFOAdvisor automatically at Vitis build time.
3. Prevent FIFOAdvisor from firing during hls4ml’s internal FIFO profiling builds.

For a more in depth study, start with studying the following files:

1. [hls4ml/hls4ml/utils/fifo_advisor.py](./hls4ml/hls4ml/utils/fifo_advisor.py)
2. [hls4ml/hls4ml/backends/vitis/vitis_backend.py](./hls4ml/hls4ml/backends/vitis/vitis_backend.py)
3. [fifo-advisor/fifo_advisor/main.py](./fifo-advisor/fifo_advisor/main.py)
4. [hls4ml/hls4ml/backends/vitis/passes/fifo_depth_optimization.py](./hls4ml/hls4ml/backends/vitis/passes/fifo_depth_optimization.py)
