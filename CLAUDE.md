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

---

## CRITICAL: B300 Benchmarking Methodology

This repo was bitten repeatedly by unreliable measurements. Before reporting ANY number:

### 1. Sanity-check against hardware spec FIRST

Before claiming peak throughput, ALWAYS compute the theoretical first. If measured >> theoretical, the test is broken.

**B300 SXM6 (sm_103a) theoretical peaks** (at 2032 MHz boost):
- **FP32 FFMA: 76.96 TFLOPS** = 148 SMs × 128 FP32 cores/SM × 2 op/FMA × 2.032 GHz
  - **B300 has 128 FP32 cores per SM, NOT 256.** Same as Hopper H100.
  - The 4 SMSPs × 32 lanes = 128 cores is standard; there is no FP32 dual-issue that doubles this.
  - Claims of "154 TFLOPS FP32" are wrong (usually a 2× formula error).
- **FP64 DFMA: 1.20 TFLOPS** (ratio 1:64 per `SingleToDoublePrecisionPerfRatio`)
- **FP16/BF16 mma.sync m16n8k16: ~540-580 TFLOPS** (legacy tensor path)
- **BF16 tensor via tcgen05.mma / cuBLAS: ~1980 TFLOPS** (Blackwell path, used internally by cuBLAS)
- **FP8 tensor via cuBLAS (tcgen05 internal): ~4500 TFLOPS** (verified, 91% MFU at M=N=K=8192)
- **HBM3E: ~7-7.5 TB/s** read peak (matches 8 TB/s spec)
- **L2 bandwidth: depends heavily on access pattern; ranges 10-36 TB/s reported**
- **Shared memory: 38.5 TB/s theoretical** (32 banks × 4 bytes × 2.032 GHz × 148 SMs)
- **NVLink (2× B300): 757 GB/s unidirectional / 1503 GB/s bidirectional** (NVLink v7)

### 2. Clock frequency: 1920 vs 2032 MHz

**Default (no nvidia-smi lock): boost to 2032 MHz under sustained FFMA load.**
**`nvidia-smi -lgc 2032` paradoxically pins to 1920 MHz** (base clock), NOT 2032.

ALL TFLOPS claims must state which clock state:
- "Default boost" → 2032 MHz
- "Locked" → 1920 MHz (6% lower)

The catalog has many numbers at both clocks mixed together, causing ~6% noise.

### 3. Common measurement pitfalls to AVOID

**Pitfall: Formulas presented as measurements.** If a test reports `TFLOPS = cores × ops × clock / time`, that's often just theoretical math, not measured throughput. To actually MEASURE TFLOPS:
- Count ACTUAL FLOPS executed (instructions × iters × op-count-per-instruction × threads)
- Measure actual wall-clock time
- Divide: measured_FLOPS / time = measured TFLOPS
- Then compare to theoretical as a sanity check

**Pitfall: DCE (Dead Code Elimination).** Compiler will eliminate any loop whose output isn't used. Signs:
- Measured time doesn't scale with iteration count
- Measured BW is > theoretical peak
- 0.001 ms runtime on "massive" test

Defenses:
- Unconditional write of final result to global
- Use loop index in addressing
- Make loop values depend on runtime inputs (not compile-time constants)
- Check kernel runtime >= 1 ms

**Pitfall: Self-op chains (fma a,a,a,0).** Inflates latency 2× due to register port pressure. Use distinct sources: `fma.rn.f32 d, a, b, c` with a, b, c from different registers.

**Pitfall: Launch-overhead-dominated tests.** Kernels running <100 µs have >10% launch overhead. For peak throughput measurements, ensure runtime >= 10 ms. For latency, use clock64 inside the kernel to exclude launch.

**Pitfall: Self-consistency not checked.** Before reporting, sanity-check:
- Does measured % of theoretical make architectural sense?
- Do related numbers (e.g., latency × throughput) match?
- Do cross-sections of the catalog agree?

### 4. Recommended verification workflow

For every B300 measurement:

1. **State the theoretical maximum first.** "Theoretical peak = X TFLOPS at clock Y MHz."
2. **State the measured number.** "Measured Z TFLOPS = Z/X of theoretical."
3. **If Z > X**: STOP. Test is broken. Look for DCE, formula bugs, clock mismatch.
4. **If Z > 1.5× theoretical**: almost certainly DCE.
5. **If Z < 0.5× theoretical**: under-saturated or methodology issue. Check ILP, occupancy, anti-DCE.
6. **If Z in [0.5×, 1.0×]**: plausible, but verify SASS has expected instruction count, check ncu metrics.
7. **SASS-verify**: `nvcc -keep` and look at the .sass — count the expected instructions.
8. **Cross-check with ncu** where available: `pipe_fma.avg.pct_of_peak_sustained_active` for FFMA, etc.

### 5. This catalog has many known inaccuracies

See `CRITIQUE.md` (698 lines) and `AUDIT_NOTES.md` (490 lines) for known issues with `B300_PIPE_CATALOG.md`. Trust order:
1. AUDIT_NOTES.md > CRITIQUE.md > B300_PIPE_CATALOG.md
2. Any number in catalog should be verified before citing.
3. "HIGH confidence" in AUDIT_NOTES.md typically means cross-checked; others need re-verification.

### 6. When spawning sub-agents for investigations

Sub-agent outputs are NOT authoritative without verification. Common failure modes:
- Agent presents a formula result as measured throughput
- Agent uses wrong constants (e.g., "256 cores/SM" when B300 has 128)
- Agent trusts compiler-emitted code without SASS verification
- Agent runs test too short (<1 ms) and measures noise

**Always apply step 3 above to sub-agent findings**: is the reported number plausible vs theoretical?

### 7. Common catalog errors to watch for

- **Confusing "FFMA" and "FMA-ops":** an FFMA is ONE instruction that does 2 FLOPS. Don't double-count.
- **Mixing up "cores" vs "SMSPs":** B300 has 148 SMs × 4 SMSPs × 32 FP32 lanes = 18,944 FP32 cores total. Per SM: 128 cores.
- **"Dual-issue" vs separate pipes:** Hopper/Blackwell CAN issue 1 FFMA + 1 INT32 + 1 FP64 per cycle per SMSP (different pipes), but this is NOT "2 FFMAs per cycle".
- **Spec TFLOPS ambiguity:** NVIDIA quotes peak "with sparsity" for tensor ops. Dense is 2× less. Check context.
