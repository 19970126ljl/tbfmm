# GPU task coarsening comparison

This opt-in research example compares an asynchronous single-task baseline with
M2L grouping, P2P grouping, or both, using the same FP64 rotation kernel at order
12 (91 complex coefficients per cell). M2L and P2P run on GPUs; the other FMM
operators run on CPU workers in every mode. The mathematical operations and
submission order are shared. The original P2P device kernels are retained.

## Build

Requirements: Linux, CMake >= 3.18, a CUDA C++17 toolchain and an extended
StarPU 1.4 runtime from
[task-coarsening](https://github.com/19970126ljl/task-coarsening).
The required interface is `starpu_codelet::coarsened_variant`,
`starpu_coarsening_configure`, and `starpu_coarsening_get_stats`.
CMake checks the selected headers and library together. A stock StarPU build
can be used for the existing examples; this additional benchmark requires the
extension.

The integration was checked with GCC 12.1, CUDA 12.9 and the runtime ABI header
at task-coarsening commit `f188caa2a1e2603c06e39d78262ec79335c9f625`
(header SHA256 `eb25b055bcb1ef3eb8fe12a3dc2fb266db28e756fbc94f960cf989c18dbdcec3`).
Build the runtime separately using that repository's explicit performance
purpose and an installation prefix:

```bash
# Run in the task-coarsening repository, with the CUDA toolchain configured.
TC_PROTO2_BUILD_ROOT="$PWD/build/tbfmm-runtime" \
  python3 -m scripts.proto2.build --profile performance --purpose performance \
  --execution single --install-prefix "$PWD/build/tbfmm-runtime-install"
```

Then configure this repository with that prefix. Use a fresh build directory
when changing the compiler or runtime. Choose CUDA architectures for the GPUs
on which the executable will run.

```bash
cmake -S . -B build-coarsening -DCMAKE_BUILD_TYPE=Release \
  -DTBFMM_ENABLE_SPECX=OFF -DTBFMM_ENABLE_INASTEMP=OFF \
  -DTBFMM_ENABLE_OPENMP=OFF -DTBFMM_ENABLE_FFTW=OFF \
  -DTBFMM_USE_STARPU=/path/to/extended-starpu-install -DTBFMM_USE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DTBFMM_BUILD_GPU_COARSENING=ON
cmake --build build-coarsening --target gpu-coarsening-benchmark gpu-coarsening-validate -j2
```

For a runtime with separate source/configuration include directories, pass
`-DTBFMM_STARPU_INCLUDE_DIRS='/path/to/include/source;/path/to/include/build'`.
`gpu-coarsening-validate` contains the sequential CPU reference and numerical
checks. `gpu-coarsening-benchmark` compiles those checks out; use its timings
for performance comparisons. Both targets use the same application code.

## Run

Run inside a scheduled allocation. Keep its `CUDA_VISIBLE_DEVICES` unchanged
and set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Select distinct allocated physical
cores for the main thread, each GPU driver and each CPU worker. Verify that
all selected GPUs/cores belong to the same NUMA node and bind host memory to
that node. The program checks allocation membership and actual worker CPU
bindings; the launcher is responsible for the physical-core/NUMA checks.

Required arguments are `--aggregation`, `--output`, `--model-directory`,
`--main-cpu` and `--worker-cpus`. The worker list contains GPU-driver cores
first, followed by CPU-worker cores. Here is a one-GPU example, assuming
CPUs 0 through 5 and GPU 0 have been verified to be allocated and local to
NUMA node 0. Adapt these IDs to the allocation.

```bash
mkdir -p models
exe=build-coarsening/benchmarks/gpu-coarsening
numactl --membind=0 "$exe/gpu-coarsening-benchmark" \
  --aggregation single --particles 1024 --height 4 --block 8 \
  --gpus 1 --cpu-workers 4 --main-cpu 0 --worker-cpus 1,2,3,4,5 \
  --model-directory "$PWD/models" --prepare-models 1 --output models-prepared.json
numactl --membind=0 "$exe/gpu-coarsening-validate" \
  --aggregation both --particles 4096 --height 4 --block 8 \
  --gpus 1 --cpu-workers 4 --main-cpu 0 --worker-cpus 1,2,3,4,5 \
  --model-directory "$PWD/models" --warmup 0 --repeats 2 --output checked.json
numactl --membind=0 "$exe/gpu-coarsening-benchmark" \
  --aggregation single --particles 1000000 --height 6 --block 32 \
  --gpus 1 --cpu-workers 4 --main-cpu 0 --worker-cpus 1,2,3,4,5 \
  --model-directory "$PWD/models" --warmup 1 --repeats 4 --output single.json
numactl --membind=0 "$exe/gpu-coarsening-benchmark" \
  --aggregation both --particles 1000000 --height 6 --block 32 \
  --gpus 1 --cpu-workers 4 --main-cpu 0 --worker-cpus 1,2,3,4,5 \
  --model-directory "$PWD/models" --warmup 1 --repeats 4 --output both.json
```

`--prepare-models 1` performs bus-only initialization, with calibration threads
inheriting the bound main CPU, then exits without executing FMM. Prepare the
model directory on the selected hardware before validation/timing and keep it
frozen for comparisons. For two GPUs, prepare with `--gpus 2` and provide two
GPU-driver cores followed by the same four CPU-worker cores.

Modes are `single`, `m2l`, `p2p`, and `both`. Defaults are one GPU, four CPU
workers, CUDA pipeline 2, prio scheduling, prefetch enabled, scan limit 256,
minimum group size 2, cross M2L maximum 16, cross P2P maximum 8, and inside
maximum 4. CLI flags expose these values (`--help` lists them); pipeline must
be 0, 1, or 2 because each worker has two protected workspace slots. Inputs
default to one million uniform particles, height 6, block size 32, seed 0,
one warmup and four measured iterations. Block size counts leaf cells, not
particles. Unsupported names/values fail explicitly.

The benchmark replaces inherited `STARPU_*` settings with its CLI configuration
and the preset above. All modes use the same asynchronous single callbacks,
per-execution interaction preparation and workspace reuse. Grouping attaches
the corresponding variants; it does not cache tree interaction structure.
Compatibility is checked on a complete proposal; no application `can_append`
callback repeatedly scans prefixes. Workspace cleanup runs on the owning
CUDA workers before runtime shutdown.

## Measurement scope

`execute_seconds` covers the complete `algorithm.execute`: data registration,
task preparation/submission, execution, output retrieval and unregistering.
Tree construction, runtime/model initialization, CPU reference comparison and
JSON output are outside that timer. This is an application execution time;
it is separate from the direct speedup of a fixed number of grouped kernels.
The single baseline already includes the common asynchronous integration.

Keep every raw iteration, exclude the explicitly marked warmups, alternate
mode order across independent process blocks, and compute paired speedups
within the same configuration. Record GPU identities, toolchain, runtime and
application revisions, bindings and model fingerprints alongside the JSON.
The final group/member counters are process totals, including warmups, across
all participating GPU workers. They are not per-operator coverage or FLOPs.
Numerical checks compare all four output components to the sequential FMM
with relative L2 tolerance `1e-11`; they do not measure FMM approximation error
against a direct all-pairs solve. Multi-GPU scaling requires its own evidence.
