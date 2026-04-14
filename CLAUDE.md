# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

QuickRunCUDA is a microbenchmarking harness for CUDA kernels. A single C++ host binary (`QuickRunCUDA.cpp`) compiles a user-supplied `.cu` file to CUBIN at runtime via NVRTC, allocates three device buffers, launches the kernel once (optionally with an `init` kernel first), then optionally runs N timed iterations with CUDA events. This lets you iterate on a kernel without rebuilding the host every time.

## Build & run

```bash
make                                   # builds ./QuickRunCUDA; auto-detects SM arch via nvidia-smi
make dbg=1                             # -g -G debug build
make clean
./QuickRunCUDA <kernel.cu> [flags]     # kernel path can also be passed via -f
```

The Makefile appends `a` to SM ≥ 90 (e.g. `sm_90a`, `sm_100a`) so PTX that needs the architecture-specific variants compiles. After each successful compile, the host writes `output.cubin` and also dumps SASS to `sass/<kernel_basename>[_<headerhash>].sass` (plus a matching `.cubin`). `--reuse-cubin` skips recompilation and loads `output.cubin` directly.

## Kernel contract

Every kernel file must expose:

```cpp
extern "C" __global__ void kernel(float* A, float* B, float* C,
                                  int arg0, int arg1, int arg2);
```

And, if `-i` is passed, an `init` kernel with the same signature. `A`, `B`, `C` are the three device buffers whose sizes come from `-A/-B/-C` (in dwords). `arg0/arg1/arg2` come from `-0/-1/-2`. The types in the signature are nominal — many kernels reinterpret as `int*`, `uint4*`, etc.

`default_kernel.cu` is the minimal example; `tests/` holds the real kernels.

## Important flags

- `-t` threads/block, `-b` blocks, `-p` persistent (gridDim = SM count)
- `-A/-B/-C` buffer size in dwords (default 64M dwords = 256 MiB each)
- `-r` / `--randomB` fill A/B with random data (`--randomMask`, `--randomSeed`); otherwise zeroed
- `-T N` run N timed iterations; `-P` multiplier + `-U` unit + `-L` speed-of-light give `ops/s` and `% SOL` output (`-N` is per-thread multiplier)
- `--l2flush {0,1,2}` none / at start / every run (mode 2 forces per-launch events)
- `--timesPerRun` prints every iteration time
- `-H "<string>"` prepends text to the kernel source before NVRTC — used to inject `#define UNROLL 8` etc. The header string is hashed into the SASS filename so different `-H` variants don't clobber each other.
- `--dump-c`, `--load-c`, `--reference-c`, `--compare-tolerance` for golden-output testing (`raw`/`int_csv`/`float_csv` formats)
- `--clock-speed <MHz>` locks GPU clock via NVML (0 = no force, 1 = unlocked; **does not reset after run**)

## Server mode

`./QuickRunCUDA --server` keeps the CUDA context alive and listens on FIFOs `/tmp/quickruncuda_cmd` and `/tmp/quickruncuda_resp`. Each line written to the command pipe is re-parsed as a fresh argv (via `parseCommandString`) and run; stdout is captured and written back. `utils/cuda_controller.py` is the reference Python client. Use this for sweeps — CUDA init is ~hundreds of ms so avoiding re-init dominates iteration speed.

## Repository layout

- `QuickRunCUDA.cpp` — the whole host (~680 lines). `main` → `run_cuda_test` does allocation, NVRTC compile, event-based timing, optional dump/compare.
- `utils/cuda_helper.h` — NVRTC wrapper (`compileFileToCUBIN`, `loadCUBIN`), error macros. Also defines `GPU_SM_COUNT=132` etc. — these constants are **H100/H200 defaults** and are not auto-detected.
- `utils/nvmlClass.h` — clock locking via NVML.
- `utils/ipc_helper.h` — named-pipe glue for server mode.
- `utils/CLI11.hpp` — third-party CLI parser (header-only).
- `tests/` — kernels. `side_aware.cu` is the L2-side-aware reduction (the hackathon project). `bench_*.cu` are the Blackwell CVT/MUFU/FMA microbenches; `run_microbench.sh` drives the whole suite and `MICROBENCH_RESULTS.md` has B300 numbers. `test_cvt_correctness.cu` is the correctness harness for the narrow-format conversions.
- `sass/` — auto-populated on every compile; safe to delete.
- `_run.sh` — example NSight Compute profiling invocation.

## Conventions when writing benchmark kernels

- Make inputs depend on `threadIdx.x` and keep a loop-carried dependency out to `C[]` (under an impossible `if`) to defeat LICM / DCE — see `bench_fp32_fma.cu` for the pattern.
- `#ifndef UNROLL` guard so the outer runner can inject it through `-H`.
- For correctness tests, write results to `C` and compare with `--reference-c` + `--compare-tolerance`.
