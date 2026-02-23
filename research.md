# QuickRunCUDA - Codebase Research Report

## Overview

QuickRunCUDA is a CUDA kernel microbenchmarking framework designed for rapid iteration. It was used to build the winning project at the SemiAnalysis GPU Hackathon ("Optimizing NVIDIA Blackwell's Split L2"). The framework compiles CUDA kernel source files at runtime using NVRTC, executes them via the CUDA Driver API, and measures performance with configurable timing.

---

## File Inventory

```
QuickRunCUDA/
  QuickRunCUDA.cpp           658 lines   Main application
  Makefile                    94 lines   Build system
  default_kernel.cu            3 lines   Default "hello world" kernel
  _run.sh                      1 line    NSight Compute profiling command
  README.md                   16 lines   Project overview
  .gitignore                  10 lines   Ignores *.ptx, *.cubin, binary, logs

  utils/
    cuda_helper.h             322 lines  CUDA compilation & execution helpers
    ipc_helper.h               61 lines  Named pipe IPC for server mode
    nvmlClass.h               372 lines  NVML wrapper (clock control, stats, fan)
    CLI11.hpp                  ~7K lines Header-only CLI argument parser (vendored)
    cuda_controller.py         43 lines  Python client for server mode
    git_and_run.py            327 lines  Directory watcher + auto-commit tool

  tests/
    vector_add.cu                        FP16 vector add with cluster dims
    1_RELU.cu                            Simple ReLU kernel
    1_LLMC.cu                            Packed128 operations from LLM.c
    icache.cu                            Instruction cache benchmarking
    icache_sweep.py                      GPC discovery automation
    side_aware.cu                        L2 side-aware reduction (hackathon winner)
    0_WIP.cu                             Work-in-progress experiments
```

---

## Build System (Makefile)

- Uses `nvcc` as the compiler driver with `g++` as the host compiler.
- C++17, 64-bit, `--threads 0` for parallel compilation.
- Automatic GPU compute capability detection via `nvidia-smi --query-gpu=compute_cap`.
- Special suffix handling: SM 90 becomes `sm_90a`, SM 100 becomes `sm_100a`.
- Links against: `libcuda` (driver API), `libnvrtc` (runtime compilation), `libcupti` (profiling), `libnvidia-ml` (NVML), `lnvperf_host`/`lnvperf_target`, `libcurand`.
- Outputs a single binary: `QuickRunCUDA`.

---

## Architecture: Main Application (QuickRunCUDA.cpp)

### Entry Point & Flow

1. Parse CLI args via CLI11
2. Initialize CUDA: `cuInit()` -> `cuDeviceGet()` -> `cuCtxCreate()`
3. Optionally set GPU clock via NVML
4. Either run in **server mode** (IPC loop) or **normal mode** (single run)
5. `run_cuda_test(args)` handles everything from compilation to execution

### CmdLineArgs Structure

All configuration lives in a single struct with these categories:

- **Array sizes**: `arrayDwordsA/B/C` (default 64M dwords = 256MB each)
- **Random init**: `randomArrayA`, `randomArrayB`, `randomArraysBitMask`, `randomSeed`
- **Kernel config**: `threadsPerBlockX`, `numBlocksX`, `persistentBlocks`, `sharedMemoryBlockBytes`, `sharedMemoryCarveoutBytes`, `runInitKernel`
- **Benchmark**: `timedRuns`, `perfMultiplier`, `perfMultiplierPerThread`, `perfSpeedOfLight`, `perfMultiplier_unit`, `listIndividualTimes`
- **Compilation**: `kernel_filename`, `header`, `reuse_cubin`
- **Modes**: `server_mode`, `clock_speed`
- **Array I/O**: `dump_c_array`, `dump_c_format`, `load_c_array`, `reference_c_filename`, `compare_tolerance`
- **L2 flush**: enum `L2FlushMode` (NO_FLUSH=0, FLUSH_AT_START=1, FLUSH_EVERY_RUN=2)

### run_cuda_test() Flow

1. **Allocate GPU memory**: `cuMemAlloc` for arrays A, B, C
2. **Handle persistent blocks**: if `-p`, set `numBlocksX` = SM count
3. **Compile or load kernel**:
   - If `reuse_cubin`: read `output.cubin` from disk
   - Otherwise: call `CUDA.compileFileToCUBIN()` and write result to `output.cubin`
4. **Load module**: `cuModuleLoadData` + `cuModuleGetFunction` for "kernel" (and "init" if `-i`)
5. **Initialize arrays**: zeros, random (parallel CPU RNG + H2D copy), or loaded from file
6. **Prepare kernel args**: `{&d_A, &d_B, &d_C, &arg0, &arg1, &arg2}`
7. **Run init kernel** (if `-i`)
8. **Warm-up run**: launch kernel once + synchronize
9. **Timed runs**: event-based timing with optional L2 flush and individual per-run events
10. **Output results**: average time, performance metric, speed-of-light percentage
11. **Array I/O**: dump C to file (raw/csv), compare against reference
12. **Cleanup**: free GPU memory, unload module

---

## Compilation Pipeline (cuda_helper.h :: compileFileToCUBIN)

This is the core compilation function. Current flow:

1. Read kernel `.cu` source file into memory
2. Prepend optional header string (from `-H` flag)
3. Detect compute capability: `cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR/MINOR)`
4. Build NVRTC compile options:
   - `--generate-line-info`
   - `-use_fast_math`
   - `--std=c++17`
   - `--gpu-architecture=sm_XY[a]` (auto-detected, with `a` suffix for SM 90/100)
   - `-I/usr/local/cuda/include/`
5. Create NVRTC program: `nvrtcCreateProgram(&prog, source, filename, 0, NULL, NULL)`
6. Compile: `nvrtcCompileProgram(prog, numOptions, options)`
7. Get/print compilation log (errors/warnings)
8. **Extract CUBIN directly**: `nvrtcGetCUBINSize()` + `nvrtcGetCUBIN()`
9. Return the CUBIN binary

### Critical Observation: No PTX Step

The current code compiles **directly to CUBIN** via NVRTC. It never generates or saves PTX. The NVRTC API supports both:
- `nvrtcGetCUBIN()` / `nvrtcGetCUBINSize()` - get compiled CUBIN (current)
- `nvrtcGetPTX()` / `nvrtcGetPTXSize()` - get PTX intermediate representation (not used)

To compile PTX to CUBIN separately, the CUDA Driver API provides:
- `cuModuleLoadDataEx()` with PTX input and JIT options
- Or use `nvrtcCompileProgram` with `--gpu-architecture=compute_XY` (yields PTX, not CUBIN)

### Module Loading (loadCUBIN)

Simple wrapper around `cuModuleLoadData(&module, cubin)`. Frees the cubin buffer after loading.

---

## Features Identified for Removal

### 1. Server Mode (QuickRunCUDA.cpp + ipc_helper.h + cuda_controller.py)

- Named pipe IPC via `/tmp/quickruncuda_cmd` and `/tmp/quickruncuda_resp`
- Persistent CUDA context across multiple runs (avoids ~1-2s init overhead)
- Stdout capture via pipe redirection (`dup2`)
- Command string parsing with single-quote support
- Python client class `CUDAController`
- **Lines affected**: ~50 lines in main(), ipc_helper.h (61 lines), cuda_controller.py (43 lines)

### 2. Clock Control (nvmlClass.h)

- Full NVML wrapper (372 lines)
- Lock/unlock GPU and memory clocks
- Fan speed control (set to 100%, reset to auto)
- Background stats collection thread (1ms samples)
- Power, temperature, utilization, clock monitoring
- CSV output capability
- Thread-safe sample access with pause/resume
- **Used in main()**: 3 lines to optionally create `nvmlClass` object
- **Links**: `libnvidia-ml`, `lnvperf_host`, `lnvperf_target`

### 3. git_and_run.py

- Directory watcher with debounced file change detection
- Auto-commit with optional OpenAI-generated summaries
- Subprocess server mode integration
- **Dependencies**: `watchdog`, `gitpython`, `openai`

### 4. _run.sh (NSight Compute script)

- Single-line NCU profiling command + nvdisasm
- Not core functionality

---

## Features to Keep (Core)

### 1. Compilation Pipeline (to be modified)
- NVRTC compilation (change from CUBIN-only to PTX-first)
- Header injection (`-H` flag)
- Compute capability detection
- Module loading

### 2. Kernel Execution
- Driver API kernel launch (`cuLaunchKernel`)
- Init kernel support (`-i` flag)
- Persistent blocks (`-p` flag)
- Shared memory configuration
- Kernel integer arguments (`-0`, `-1`, `-2`)

### 3. Memory Management
- Array allocation (A, B, C with configurable sizes)
- Random initialization (parallel CPU RNG)
- Array I/O (dump, load, reference comparison)

### 4. Timing & Benchmarking
- CUDA event-based timing
- Per-run individual events
- Performance metric calculation
- Speed-of-light percentage
- L2 cache flush

### 5. CLI Argument Parsing (CLI11)

---

## Key Specificities & Edge Cases

### SM Architecture Suffix
The code adds an 'a' suffix for SM 90 and SM 100 (for tensor float support). This is done both in the Makefile (for host compilation) and in `cuda_helper.h` (for NVRTC). Currently only handles SM 90 in `cuda_helper.h`:
```cpp
snprintf(..., "%s%d%d%s", ..., major, minor, (major == 9 && minor == 0) ? "a" : "");
```
The Makefile also handles SM 100:
```makefile
SM_COMPUTE_CAPABILITY := $(GPU_COMPUTE_CAPABILITY)$(if $(filter 90 100,$(GPU_COMPUTE_CAPABILITY)),a,)
```

### Kernel Signature Convention
All kernels must follow this exact signature:
```cuda
extern "C" __global__ void kernel(float *A, float *B, float *C,
                                   int arg0, int arg1, int arg2)
```
The `extern "C"` is critical for NVRTC to find the function by name.

### Random Number Generation
Uses parallel CPU-side MT19937-64 with OpenMP, chunked into 1MB pieces. Each chunk gets seed = `baseSeed + chunkIndex`. Supports bit masking for power analysis experiments.

### Bit Mask Parsing
Custom transform on `--randomMask` to handle hex (`0x`) and binary (`0b`) prefixes.

### L2 Flush
200MB buffer allocated once globally, written to zero with `cuMemsetD8`. Larger than most GPU L2 caches to guarantee full eviction.

### Error Handling
`checkCudaErrors` macro handles CUresult, cudaError_t, and nvrtcResult types with automatic error string lookup. Exits on any error.

### CudaHelper Class Extras (not used by main app)
The `CudaHelper` class has several methods used only by standalone programs (not by `run_cuda_test`):
- `setKernelSize()` / `getKernelSize()` - stored kernel dimensions
- `launch()` / `launchCooperative()` - wrapper around cuLaunchKernel
- `startTimerGPU()` / `endTimerGPU()` - CUDA event timing
- `deltaTimerCPU()` - CPU high-resolution timing
- `launchSimpleTest()` - single kernel launch with timing
- 4 CUDA streams with priority
- Device property queries

The main application only uses:
- `compileFileToCUBIN()` (via `CUDA` object constructed with `CudaHelper CUDA(cuDeviceGlobal)`)
- `loadCUBIN()`

### Global State
Three global variables: `cuDeviceGlobal`, `cuContextGlobal`, `d_flush` (L2 flush buffer). The context is created once in `main()` and shared across all runs.

---

## Compilation Output Files (Current)

Currently writes to a single fixed filename:
- `output.cubin` - the compiled CUBIN binary (overwritten each run)

The `--reuse-cubin` flag reads from `output.cubin` instead of recompiling.

---

## Dependencies Summary

**Required (keep)**:
- CUDA Driver API (`libcuda`)
- NVRTC (`libnvrtc`)
- CUDA Runtime (`cuda_runtime.h` - used by CudaHelper for device props)
- OpenMP (for parallel random init)
- CLI11 (vendored header-only)
- Standard C++17

**Optional (remove)**:
- NVML (`libnvidia-ml`) - clock control
- CUPTI (`libcupti`) - profiling (not actually used in code, only linked)
- nvperf (`lnvperf_host`, `lnvperf_target`) - performance API (not used in code, only linked)
- curand (`libcurand`) - device RNG (commented out in cuda_helper.h, not used)
- Python packages: `watchdog`, `gitpython`, `openai`
