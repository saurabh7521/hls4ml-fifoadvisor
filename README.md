# hls4ml-fifoadvisor-integration

This workspace packages `hls4ml` and `fifo-advisor` together so an `io_stream` Vitis HLS build can automatically launch FIFOAdvisor after `hls_model.build()`.

## Setup

Create one environment for both projects:

```bash
conda env create -f environment.yml
conda activate hls4ml-fifoadvisor
```

The environment installs both local packages in editable mode from this checkout.

## Automatic FIFOAdvisor Runs

FIFOAdvisor integration is intended for `backend='Vitis'` and `io_type='io_stream'`. The automatic hook runs on the generated HLS solution directory:

```text
<output_dir>/<project_name>_prj/solution1
```

It does not run on Vivado accelerator `project_1` block-design directories.

Default FIFOAdvisor execution:

```python
import hls4ml

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='hls4ml_prj_io_stream',
    backend='Vitis',
    io_type='io_stream',
    fifo_advisor=True,
)

hls_model.write()
hls_model.build(csim=True, synth=True, cosim=True)
```

Custom FIFOAdvisor settings:

```python
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='hls4ml_prj_io_stream',
    backend='Vitis',
    io_type='io_stream',
    fifo_advisor=True,
    fifo_advisor_config={
        'solver': 'group-sa',
        'maxfun': 100,
        'n_scaling_factors': 8,
        'round_type': 'rint',
        'init_with_largest': True,
        'output': 'fifo_advisor_group_sa.json',
    },
)
```

When enabled, the build report gets a `FIFOAdvisorReport` summary and the full optimizer results are written to JSON.

If you also enable hls4ml's built-in FIFO depth optimization flow, its internal profiling builds will not trigger
FIFOAdvisor. FIFOAdvisor runs only once at the end of the final outer build.
