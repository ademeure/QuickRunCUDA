# QuickRunCUDA — Harness Internals

A complete, opinionated walk-through of the host binary, helpers, kernel contract, and conventions, written for a future Claude session that's never seen the repo before. The companion top-level `CLAUDE.md` is a 30-second index; this is the 30-minute version.

Last full rewrite: 2026-05-16 (Opus 4.7). Verify against the code before relying on any specific line number.

> **Scope note (added when integrated onto the corpus branch):** §2–§13 describe the **host/framework**, which is current and accurate — those host fixes (the §8 "fixed" list) are applied on this branch. The **research-corpus** sections (§1, §9, §14, §15) were written against a snapshot that *predates the late-April-2026 B300 audit sprint*. For rigor-verified numbers and the live research backlog, the authoritative sources are now **`b300_clean/B300_TRUE_REFERENCE.md`** and **`b300_clean/CURIOSITY_LIST_V4.md`** (plus the `ULTRA_DENSE`/`DENSE`/`JUSTIFIED_B300_PIPE_CATALOG.md` audit catalogs and `justifications/`). See also the methodology section in the top-level `CLAUDE.md`. Treat any corpus-size/commit-count number below as approximate.

---

## 1. What this repo actually is

Two layered things in one tree:

1. **The framework** (~700 lines of host C++ + ~600 lines of helpers). A tiny NVRTC-driven launcher: load `.cu`, compile to CUBIN at runtime, allocate 3 buffers (A/B/C), optionally run an `init` kernel once, then run the main `kernel` either once or `N` timed iterations with CUDA-event timing. CLI-driven, single-file `QuickRunCUDA.cpp`. Also supports a server mode (FIFO IPC) for sweeps.

2. **A B300 / Blackwell characterization corpus** (~780 `bench_*.cu` files in `tests/` out of ~1,130 `.cu` total, dozens of top-level `.md` research docs, ~20 shell drivers). This is the user's microbenchmarking corpus — pipe catalog, F2FP/CVT/MUFU deep-dives, NVFP4 quantization, L2-side-aware reductions. Most of these were added in one big April 9, 2026 commit (`a9421df`) plus a long tail of incremental additions, including a late-April-2026 multi-agent audit sprint (~1,500 commits) that produced the `b300_clean/` rigor corpus (~290 docs), the `ULTRA_DENSE`/`DENSE`/`JUSTIFIED` catalog variants, and `justifications/`.

Treat the framework as **slow-changing infrastructure** — the host binary was last meaningfully edited in April 2026 to add `sm_103a` support and auto-SASS dump. The corpus is **research-velocity** — kernels and docs are added daily.

There are also `multigpu/MGFenceBench.cpp` (a stripped-down standalone P2P fence harness, ~160 lines, builds separately) and a sibling Python script (`utils/git_and_run.py`, watchdog + auto-commit with optional GPT-4o summaries — a research workflow tool, not part of the core harness).

### Branches you should know about

- `main`: the published baseline (last meaningful host change: Oct 2025).
- `f2fp-deep-dive` (default working branch): same framework as main + the entire B300 corpus + the F2FP deep dive.
- `wip` (origin only, not local): an **alternative simplified** rewrite of the host (Feb 2026, `aced16a` and follow-ups). Drops server/IPC/NVML/clock, switches to a PTX→CUBIN two-stage pipeline with files saved to `PTX/` and `CUBIN/` directories, adds `--ptx-input` / `--cubin-input`. Roughly 488 lines vs the current 680. **We are NOT on this branch.** If the user asks to "simplify", consider cherry-picking ideas, not replacing wholesale — the corpus on `f2fp-deep-dive` depends on the current CLI shape.

---

## 2. The host binary — `QuickRunCUDA.cpp` (line-level guide)

Single TU, ~680 lines. Layout (line numbers as of 2026-05-16, **verify with `Read` before quoting**):

| Range | What's there |
|---|---|
| 1–62 | License header, includes |
| 63–66 | Globals: `cuDeviceGlobal`, `cuContextGlobal`, `d_flush` |
| 68–125 | `struct CmdLineArgs` — every CLI flag with default |
| 127–195 | `setupCommandLineParser` — CLI11 wiring of every flag |
| 197–252 | `parseCommandString` — string→argv for server mode |
| 254–258 | `flushL2Cache` — lazy-alloc 200 MiB scratch + memset |
| 268–347 | `main` — CUDA init + clock lock + dispatch (single run vs server loop) |
| 354–680 | `run_cuda_test` — the actual work |

### 2.1 `main()` lifecycle

1. Parse CLI (`CLI11_PARSE`). If a positional arg was given, it overrides `--kernel-filename`.
2. `cuInit(0)`, `cuDeviceGet`, `cuCtxCreate`. **Single device, device 0 only.** Multi-GPU is via `CUDA_VISIBLE_DEVICES`, not CLI.
3. If `--clock-speed > 0`, instantiate `nvmlClass` to lock SM clocks. The destructor would re-set fan speed but **does not reset the clock** — the clock-lock persists across runs and the next process inherits it. See §6.3 for caveats.
4. Single-run mode: call `run_cuda_test(args)`. **Its return code is currently dropped** (line ~297). The process exits 0 unless CLI parse failed. (Bug — see §8.)
5. Server mode: mkfifo `/tmp/quickruncuda_cmd` and `/tmp/quickruncuda_resp`, loop reading commands, exec each in-process. See §5.

### 2.2 `run_cuda_test()` flow

```
allocate d_A, d_B, d_C (cuMemAlloc, sizes from -A/-B/-C in dwords)
malloc h_C
if --persistentBlocks: numBlocksX := SM count (queried once); clear the flag
if --reuse-cubin:
    read output.cubin (binary) into a `new char[]` buffer
else:
    nvrtcCompile(args.kernel_filename, header=args.header)  → cubin in mem
    write output.cubin
    write sass/<basename>[_<headerhash>].sass via `nvdisasm`
    write sass/<basename>[_<headerhash>].cubin (copy)
cuModuleLoadData(cubin); cuModuleGetFunction("kernel"); optionally "init"
load h_C from --load-c (raw bytes), else memset zero, then HtoD
init A/B: random (CPU OpenMP, splitmix-ish via mt19937_64) or zeroed
prepare kernel_args[] = {&d_A, &d_B, &d_C, &i0, &i1, &i2}
if -i: launch init kernel once
if l2flush ≥ 1: flushL2Cache  (200 MiB cuMemsetD8)
launch main kernel once  (warmup / sole launch if -T 0)
if l2flush == 1: flushL2Cache  (a SECOND flush — see §4.4)
cuCtxSynchronize
if -T > 0:
    decide individual_events := (l2flush == 2 || --timesPerRun)
    create overall_start/stop + (optional) per-iter events
    record overall_start
    for i in [0..N):
        if l2flush == 2: flushL2Cache
        if individual_events: record start_events[i]
        cuLaunchKernel(kernel_addr, ...)
        if individual_events: record stop_events[i]
    record overall_stop
    cuEventSynchronize(overall_stop)
    avg_time := (sum-of-per-iter | overall) / N
    print "  ms" and optionally GOps/% SOL
cuCtxSynchronize
if --dump-c: DtoH and write file (raw|int_csv|float_csv)
if --reference-c: DtoH and compare (with optional tolerance for f32 diff)
free everything; cuMemFree(d_flush) (one-shot global)
cuModuleUnload
return 0
```

Important: this function **calls `exit(EXIT_FAILURE)` on file-I/O errors** (lines ~388, 444, 643). In server mode that kills the whole daemon. (Bug — see §8.)

### 2.3 The three buffers

`A`, `B`, `C` are flat `uint32_t` device buffers whose sizes come from `-A/-B/-C` (in **dwords = 4 bytes**, default `64 * 1024 * 1024` dwords = 256 MiB each). The host always passes them as `float*` in the kernel-args array, but the **kernel is free to reinterpret as `int*`, `uint4*`, half2*, etc.** — see `bench_dram_bw.cu` casting through `int4*`.

Conventions in practice (not enforced):
- `A` and `B` are inputs; `C` is the output that `--dump-c` / `--reference-c` operate on.
- The init kernel, if any, is given the same pointers and arg ints. Most init kernels write metadata into `B` (e.g., `side_aware.cu` writes per-SM side info into `B`).

### 2.4 Kernel int args

Three `int` slots, populated from `-0/-1/-2`. Default zero. The kernel signature must declare them by name and use them; unused slots can be named `unused_N`. There is no way to pass `float` or `int64_t` directly — encode through dwords if you need them (`__int_as_float`, etc.).

---

## 3. The kernel contract

```cpp
extern "C" __global__ void kernel(float* A, float* B, float* C,
                                  int arg0, int arg1, int arg2);

// Optional, only loaded if -i is passed:
extern "C" __global__ void init  (float* A, float* B, float* C,
                                  int arg0, int arg1, int arg2);
```

Why `extern "C"`: NVRTC mangles names by default; we look up the function by the literal string `"kernel"` / `"init"`.

**Both kernels share the same arg layout.** This is intentional — the init kernel typically populates metadata into one of the buffers that the main kernel then reads.

If you need a different effective signature (e.g., interpret `A` as `uint4*`), cast at the top of the kernel — see any of the `bench_*.cu` files for the pattern.

### 3.1 The two-pass init pattern

Used by `side_aware.cu`, the quantize kernels, and any benchmark that needs per-SM setup. The init kernel typically:
1. Writes SM-side or per-page metadata into `B`.
2. Uses `atomicInc` on a counter so only the last SM finalizes the global state.
3. Optionally prints a one-time `printf` summary (under a `DEBUG_PRINTF` guard).

The main kernel reads that metadata and proceeds. With `-T N`, init runs once and the main kernel runs `N+1` times (1 warmup + N timed, all sharing the metadata `B` populated by init).

---

## 4. CLI flags — what they do, when to use, gotchas

Grouped as the CLI11 option groups in the source.

### 4.1 Operational modes

- `--server` — start the FIFO daemon (§5). Mutually exclusive with normal mode.
- `--clock-speed <MHz>` — NVML clock lock. **0 = leave alone, 1 = unlock (reset to default), >1 = lock SM clock to that MHz.** Does NOT reset on exit — your next run inherits the lock. To unlock, run with `--clock-speed 1` or `nvidia-smi -rgc`.
- `--reuse-cubin` — load `output.cubin` from CWD instead of compiling. Skips NVRTC entirely. Header (`-H`) is ignored. Use for fast iteration when you only changed CLI args.

### 4.2 Kernel execution

- `-t / --threadsPerBlock N` — blockDim.x (default 32). y/z are always 1.
- `-b / --blocksPerGrid N` — gridDim.x (default 1).
- `-p / --persistentBlocks` — overrides `-b` with the SM count of device 0 (queried via `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`). Use for benchmarks that want one CTA per SM. Sets `args.persistentBlocks = false` after applying so re-runs in server mode don't double-apply.
- `-s / --sharedMemoryBlockBytes N` — sets `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` for the kernel. Required for kernels using `extern __shared__` ≥ 48 KB.
- `-o / --sharedMemoryCarveoutBytes N` — sets `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`. Note: the CUDA driver takes this as a *percentage*, not bytes; the field name is misleading. Hopper/Blackwell carveout is 0–100 (% L1 vs. smem).
- `-i / --runInitKernel` — also look up and launch the `init` kernel once before the main kernel. Without this, `init` is not even resolved.
- `--l2flush {0,1,2}` — see §4.4.

### 4.3 Performance measurement

- `-T / --timedRuns N` — run the main kernel N more times after the warmup, with CUDA-event timing. Setting `-T 0` (default) skips the timed loop entirely; you just get the warmup launch.
- `-P / --perfMultiplier X` — operations to divide by elapsed seconds. The printed number is `X / (avg_time_s)` in `args.perfMultiplier_unit`. Used for fixed-op benchmarks (e.g., `-P` = total GBytes transferred for a memcpy → unit = `GB/s`).
- `-N / --perfMultiplierPerThread X` — alternative: per-thread multiplier. Final multiplier = `X * threadsPerBlock * numBlocks`. Used for compute-bound benchmarks (e.g., `-N` = TFLOPs per thread → unit = `TFLOPS`). Mutually exclusive with `-P` (P is overridden if N > 0).
- `-U / --perfMultiplier-unit STR` — label only. Default `"ops/s"`.
- `-L / --perfSpeedOfLight X` — if set, also prints `==> X.XXX%` (achieved / SOL).
- `--timesPerRun` — print every per-iteration time. Forces per-iter event creation (see §4.5).

### 4.4 L2 flush modes

This is subtle. `flushL2Cache()` lazily allocates a 200 MiB scratch buffer (`d_flush`, global) and `cuMemsetD8`s it to 0. The buffer is freed at the end of `run_cuda_test` (so per-server-command alloc + free; one-shot for normal mode).

| Mode | Before warmup | Between warmup & timed loop | Per timed iter | Forces per-iter events |
|---:|:---:|:---:|:---:|:---:|
| 0 (none) | no | no | no | no |
| 1 (at start) | yes | yes (second flush!) | no | no |
| 2 (every run) | yes | no | yes | yes |

The second flush in mode 1 is intentional — the warmup launch fills the L2, and mode 1's purpose is to start the timed loop with a cold L2. Mode 2 doesn't need it because every iteration flushes anyway.

200 MiB exceeds the L2 of every GPU through Blackwell (H100 = 50 MB, H200 = 50 MB, B200 = 100 MB, B300 = 126 MB). Don't size it down without checking the GPU you're on.

### 4.5 Event timing modes

`run_cuda_test` decides between two timing strategies:

- **Overall-only** (default when `l2flush ≤ 1` and `--timesPerRun` is off): one `CUevent` pair around the entire loop. Avoids per-iter event-record overhead (which can dominate when kernels are short — sub-µs).
- **Per-iter events**: forced when `l2flush == 2` (so we can subtract the flush time from each iter) or `--timesPerRun` (so we can print them). Allocates `2*N` events, sums per-iter times for the average.

In per-iter mode the print also shows the overall time (which includes L2 flushes) for sanity.

### 4.6 Array configuration

- `-A / -B / -C N` — buffer size in **dwords** (4 bytes), default `64M = 67108864` dwords = 256 MiB.
- `-r / --randomA` — fill `A` with random uint32 (CPU `std::mt19937_64`, OpenMP chunked, seed = `randomSeed + chunk_index`).
- `--randomB` — same for `B`. C is always zeroed (unless `--load-c`).
- `--randomMask 0x...` — bitwise AND mask applied to each random value (e.g., `0x3F800000` to keep only the f32 mantissa region). Hex/binary literals accepted via a `transform()` on the CLI option.
- `--randomSeed N` — base for the per-chunk RNG. Different seed → different random stream.

If neither `-r` nor `--randomB` is given, A and B are `cuMemsetD8(0)`. Faster than CPU init.

### 4.7 Kernel arguments

`-0 / -1 / -2 N` — the three int args. No mnemonic. Convention varies per kernel; check the kernel's `kernel(...)` signature for what they mean.

### 4.8 Kernel source & compilation

- `-f / --kernel-filename PATH` — kernel `.cu` to compile. Same as positional.
- `-H / --header STR` — string prepended to the source before NVRTC. Used to inject `#define UNROLL 8` etc. **The header is hashed (`hash = hash*31 + c`) into the SASS filename** so different `-H` variants don't clobber each other in `sass/`. The hash is **not** in `output.cubin` — every compile overwrites `output.cubin`, so `--reuse-cubin` only safely reuses the *most recent* compilation.
- `--reuse-cubin` — see §4.1.

### 4.9 Array I/O (compare / dump / load)

- `--dump-c PATH` — after the run, DtoH `C` and write to disk.
- `--dump-c-format {raw,int_csv,float_csv}` — raw binary, comma-separated uint32 (skips zeros — see §8.4), or comma-separated f32 with `%.2f`.
- `--load-c PATH` — before the run, read raw bytes (must match `sizeC`) and HtoD `C`. Useful for "seed the output buffer" benchmarks.
- `--reference-c PATH` — after the run, compare DtoH `C` to a raw reference file. Prints the first mismatching index.
- `--compare-tolerance F` — if > 0, compare as f32 with absolute tolerance; else exact uint32 compare.

---

## 5. Server mode (FIFO IPC)

`./QuickRunCUDA --server` enters a loop that:

1. `IPCHelper` constructor `mkfifo`s `/tmp/quickruncuda_cmd` and `/tmp/quickruncuda_resp` (0666). Both unlinked on destructor.
2. `waitForCommand` blocking-`open()` on the command pipe (read), reads up to 4096 bytes, returns the string. The pipe blocks until a writer connects.
3. If the command is `exit`, break. Otherwise:
4. Redirect stdout to a pipe pair (so we can capture it), `parseCommandString` the input, call `run_cuda_test`, restore stdout, read captured output, send via the response pipe.

Caveats baked in (updated after the 2026-05-16 host pass):
- **`parseCommandString` accepts both `'` and `"`** as grouping quotes (the closing quote must match the opener; the other kind embeds literally; no backslash escapes). `-H "#define UNROLL 8"` and `-H '#define UNROLL 8'` both work now.
- **Recoverable errors throw instead of `exit()`** — file-I/O failures and NVRTC compile errors are caught per-command, appended to the response, and the daemon continues. **BUT** `checkCudaErrors` still `exit()`s, so *driver* errors (a missing `extern "C" kernel`/`init` symbol, bad launch geometry, OOM, a faulting kernel) still take the daemon down with no response — the most common "bad kernel" cases are **not yet survivable** (see §8 open items). Also, a thrown recoverable error currently **leaks** the run's device buffers + module in server mode (no RAII cleanup on the throw path — §8).
- **`d_flush` persists across commands** — allocated once, reused, reclaimed at process exit. No more per-command 200 MiB alloc/free.
- **Command pipe read is 4096-byte capped.** Very long `-H` strings would truncate.
- **>64 KiB of captured stdout self-deadlocks** — the reader only drains the pipe after `run_cuda_test` returns, and the write end is already closed, so a test that prints more than the pipe buffer blocks forever. Keep kernel output small.
- **Single-client only** — both pipes are non-FIFO-queue, and there's no per-client framing. Concurrent writers will interleave bytes.

The Python reference client is `utils/cuda_controller.py`. It `Popen`s the binary, waits for the pipes to appear, and `open(pipe, "w")` per command. `__del__` sends `exit`. There's a near-identical class in `utils/git_and_run.py` for filesystem-event-driven runs.

---

## 6. Helpers in `utils/`

### 6.1 `cuda_helper.h` — NVRTC wrapper + macros

Stripped to its used surface on 2026-05-16 (~139 lines; was a 325-line `CudaHelper` class that was ~50% dead). Now exactly:
- `checkCudaErrors(err)` macro — type-dispatched (cudaError_t / CUresult / nvrtcResult). Prints the error string and `exit(EXIT_FAILURE)`. (It still `exit()`s rather than throwing — that's why driver errors are not survivable in server mode; see §8.)
- `compileFileToCUBIN(...)` and `loadCUBIN(...)` — now **free functions** (no class, no per-call stream/event/timer state being created and thrown away).

Gone (and confirmed unused by the host before removal): the `CudaHelper` class with its 4 priority streams, CPU/GPU timer pairs, kernel-launch wrappers, `deviceProp()`, the commented-out `rng_on_device` (curand), and the stale `GPU_SM_COUNT=132 / FLOPS_PER_SM / DEFAULT_GPU_CLOCK / MAX_FLOPS_PER_CLOCK` macros (H100 defaults, never referenced — kernels that need the SM count use `-p` / `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`).

#### 6.1.1 `compileFileToCUBIN` details

- Reads the file, prepends `header + '\n'` to its content, NVRTC-compiles with options:
  - `--generate-line-info`
  - `-use_fast_math` (hardcoded — see §8 — affects `expf` / `sinf` / etc.)
  - `--std=c++17`
  - `--gpu-architecture=sm_<major><minor>[a]` — `a` suffix when major ≥ 9 (so `sm_90a`, `sm_100a`, `sm_103a`). Needed for architecture-specific instructions (cluster, tcgen05, narrow CVT, etc.).
  - `-I<path>` — from the `cudaIncludePath` arg, else the `CUDA_INCLUDE_PATH` env var, else `/usr/local/cuda/include/`.
- Returns `*cubinResult = code` where `code` is a `new char[codeSize]` buffer the **caller must `delete[]`** (or hand to `loadCUBIN`, which `delete[]`s it — matched allocator; the old `new[]/free` UB is fixed). The source buffer and `nvrtcProgram` are released on every path now (the old per-compile leak is fixed).
- The compile log is printed to stderr when non-empty.

#### 6.1.2 `loadCUBIN` details

```cpp
CUmodule loadCUBIN(char *cubin) {       // takes ownership: delete[]s cubin
    CUmodule module;
    checkCudaErrors(cuModuleLoadData(&module, cubin));
    delete[] cubin;
    return module;
}
```

The old 3-arg signature (`CUcontext`, `CUdevice` — both ignored) and the `free(cubin)` allocator mismatch are gone.

### 6.2 `ipc_helper.h` — FIFO IPC for server mode

~60 lines. `mkfifo` two named pipes in `/tmp/`, open-read on demand to wait for commands, open-write on demand to send responses. Each command opens fresh fds. Pipes are unlinked in the destructor.

The static `const std::string COMMAND_PIPE_NAME = "/tmp/quickruncuda_cmd"` defined in the header is fine because this file is included exactly once (in `QuickRunCUDA.cpp`).

### 6.3 `nvmlClass.h` — clock locking + stats sampling

Big class (~370 lines) with two main features:
- **Clock locking** — `nvmlDeviceSetGpuLockedClocks(min=max)`. Called from the constructor based on the `force_clock` param. `force_clock == 1` calls `nvmlDeviceResetGpuLockedClocks` (unlock); `> 1` locks both min and max to that value. **Does not unlock in the destructor** — by design, since the binary is short-lived and the user might want the lock to persist for a subsequent profile.
- **Sampling thread** — when `gather_stats=true`, spawns a background thread polling power/temp/util/memory/clocks at ~1 kHz (limited by NVML to ~10 Hz for power on most cards regardless of the sleep). Writes a CSV (`gpuStats.csv` by default) on destruction.

`QuickRunCUDA.cpp` only uses the clock-lock side: `nvmlClass nvml(0, args.clock_speed, false, false, false);` — fan unforced, no stats, no CSV. The full sampling path is unused by the harness but documented for anyone writing a higher-level driver.

NVML init / shutdown is not free (~5 ms). Avoid `--clock-speed` if you don't actually need it.

### 6.4 `CLI11.hpp` — third-party header-only CLI parser

Vendored copy. ~420 KB. Don't touch.

### 6.5 `cuda_controller.py` — Python reference client

Minimal: `Popen` the binary with `--server`, `open()` each pipe per command, send command as " ".join(args), read response. No timeout handling, no protocol versioning, no concurrency. Sends `exit` in `__del__`.

### 6.6 `git_and_run.py` — research workflow tool

A `watchdog`-based file watcher that runs a command on every `.cu` save and optionally auto-commits with a GPT-4o-generated summary. Used as a "save → bench → commit" loop during the user's microbench research sessions. Independent of the harness — orchestrates `./QuickRunCUDA` via the server-mode IPC (same protocol as `cuda_controller.py`).

---

## 7. Build & Makefile

```
make            # auto-detects GPU arch via nvidia-smi, compiles with -O2 default
make dbg=1      # adds -g -G (device debug)
make clean      # removes binary, output.cubin
```

Notes from `Makefile`:
- `GPU_COMPUTE_CAPABILITY` is read from `nvidia-smi --query-gpu=compute_cap`, sorted ascending, lowest taken. So on a mixed-GPU machine the build targets the older card.
- For arch ≥ 90, an `a` suffix is appended (`sm_90a`, `sm_100a`, `sm_103a`) so the gencode matches what NVRTC also emits at runtime. Without this, statically-compiled host code would be `sm_90` while runtime cubins are `sm_90a` — they happen to be compatible but you get a warning.
- Libraries pulled in: `-lcuda -lnvidia-ml -lnvrtc` (trimmed 2026-05-16 — the previously-linked `cupti`, `nvperf_host`, `nvperf_target`, `curand` were unused by the host; re-add them here if you reintroduce profile-API or cuRAND code).
- C++17, `-Xcompiler` pass-through, no warnings flags set (consider adding `-Wall -Wextra` — still open, §8 item 27).
- Builds in-tree — no `build/` dir. `output.cubin` ends up in CWD.

---

## 8. Known issues / bugs / cleanup targets

Sourced from the 2026-05-16 review pass. Items marked ✅ were fixed in that pass — on this branch they are the 4 host commits `175e9a0..5abbdb3` (re-authored, with new hashes, from the original `dbdf483..ec23cda` work that lived on the pre-rewrite local lineage). Items marked ⏳ are still open.

### Fixed in 2026-05-16 pass

1. ✅ **`new[]/free` mismatch** in `loadCUBIN`: `compileFileToCUBIN` returned a `new char[]` buffer that `loadCUBIN` released with `free()`. Now `delete[]`. Same applies to the `--reuse-cubin` path.

2. ✅ **`run_cuda_test` return value dropped** in `main`. Now propagated as the process exit code.

3. ✅ **`exit()` inside `run_cuda_test` and `compileFileToCUBIN` killed the server**. File-I/O failures, missing-reference, cubin open/write errors, and NVRTC compile failures now throw `std::runtime_error`. Server mode catches per-command, appends the error to the response, and continues. Non-server mode catches at `main` and exits 1.

4. ✅ **`parseCommandString` only accepted single quotes**. Now accepts both `'` and `"` (closing must match the opener). Docstring updated to be honest about what's supported (no backslash escapes).

5. ✅ **Dead `CudaHelper` class code** (~50% of `cuda_helper.h`: streams, timers, kernel-launch wrappers, properties, commented-out cuRAND). Stripped. `compileFileToCUBIN` and `loadCUBIN` are now free functions.

6. ✅ **Stale macros** `GPU_SM_COUNT=132`, `FLOPS_PER_SM=256`, `DEFAULT_GPU_CLOCK=735`, `MAX_FLOPS_PER_CLOCK` (referenced H100 SXM5, never used in host code, never auto-detected). Deleted.

7. ✅ **Hardcoded `/usr/local/cuda/include/`**. Now `CUDA_INCLUDE_PATH` env var overrides; same fallback if unset.

11. ✅ **`d_flush` alloc/free per `run_cuda_test`**. Now allocated once, reused across server-mode commands, reclaimed at process exit.

12. ✅ **Linker pulled in unused libs** (`-lcupti -lnvperf_host -lnvperf_target -lcurand`). Trimmed; kept `-lcuda -lnvidia-ml -lnvrtc`.

17. ✅ **Redundant `int kernel_int_args[3]` local copy**. Pointers now go directly into `args.kernel_int_args`.

19. ✅ **No range check on `--l2flush`**. Now `->check(CLI::Range(0, 2))`.

21. ✅ **NVRTC compile failure used to `exit()`** via `checkCudaErrors(res)`. Now throws.

Also: ✅ leaked `memBlock` and `nvrtcProgram` in `compileFileToCUBIN`. ✅ leaked fds + stuck stdout if `run_cuda_test` threw inside the server-mode dup2 block. ✅ `nvmlClass` initializer-list-order warning + two signed/unsigned compares. ✅ `cuCtxCreate` second arg was `NULL` (gives a warning on CUDA 12.5+ where it's a `CUctxCreateParams*`); now `nullptr`.

### Still open

8. ⏳ **Always-on `-use_fast_math`** in NVRTC options. Silently rewrites `expf` → `__expf` etc., which changes microbench results for any kernel that uses `math.h` calls. Existing kernels avoid this by using inline PTX `asm volatile` blocks, but it's a foot-gun for anyone writing a new bench. **Fix:** add a `--no-fast-math` flag, keep the default to avoid breaking replication of existing results.

9. ⏳ **`dump_c_format == "int_csv"` empty-cell sparse output** (`QuickRunCUDA.cpp:~625`). The trailing comma still emits when the value is zero, producing `10,,20,,,5` for sparse-looking output. Surprising behavior; likely intentional as a sparse compressor but unconventional. **Fix:** either drop the zero check, or skip both value and comma.

14. ⏳ **`-T 0` still allocates the overall_start/stop events** (`QuickRunCUDA.cpp:~531`). Cheap (~µs) but unnecessary.

15. ⏳ **`float_csv` precision** uses `%.2f` — loses precision. **Fix:** make configurable, or use `%.7g`.

16. ⏳ **`--persistentBlocks` mutates `args.persistentBlocks = false`** so subsequent server calls don't re-apply. Re-reading the same args in a sweep gets the wrong block count.

20. ⏳ **OpenMP random init alloc waste** — `h_A` and `h_B` both allocated even if only one is random; the unused one is still memset'd via `cuMemsetD8`. Minor.

22. ⏳ **No graceful Ctrl-C in server mode** — `open()` on the FIFO blocks; SIGINT tears the process down but `~IPCHelper` may not run, leaving FIFOs in `/tmp/`.

23. ⏳ **`--reuse-cubin` lives in the "Operational Modes" CLI group** but is conceptually "Kernel Source and Compilation". Cosmetic.

24. ⏳ **No stream support** — only the default stream is used. Acceptable for microbenchmarks but no async kernel overlap.

25. ⏳ **`output.cubin` in CWD races** if two `QuickRunCUDA` processes share a directory. **Fix:** PID-suffixed or tempdir output.

26. ⏳ **`bench_*.cu` triage** — `tests/0_WIP.cu`, `tests/icache.cu` + `tests/icache_sweep.py`, `tests/vector_add.cu` may be stale. Check the user's `tmp/qrc_results/` logs before deleting.

27. ⏳ **`Makefile` has no `-Wall -Wextra`** as compile flags. The 2026-05-16 pass squashed all `-Wall -Wextra` warnings under a stub-header clang syntax check, but they're not enforced at build time. **Fix:** add `CCFLAGS += -Wall -Wextra` (then pass `-Xcompiler -Wall` via `ALL_CCFLAGS`).

28. ⏳ **`sharedMemoryCarveoutBytes` is misleadingly named** — `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT` takes a percentage (0–100), not bytes. The flag would be wrong if anyone passed actual bytes.

29. ⏳ **`checkCudaErrors` still `exit()`s — server daemon-survival is only half-done.** The 2026-05-16 pass made file-I/O and NVRTC-compile errors throw (so the daemon survives those), but driver errors routed through `checkCudaErrors` (missing `extern "C" kernel`/`init` symbol, bad launch geometry, OOM, a faulting kernel) still hard-`exit()` — and they do so while stdout is redirected into the capture pipe, so the client just sees a closed pipe with no error. These are the *most common* "bad kernel" cases, so the daemon is not yet robust to a typo'd `-t`/`-b` or a missing kernel symbol. **Fix:** make `checkCudaErrors` throw `std::runtime_error` instead of `exit()` (then item 30 below becomes mandatory).

30. ⏳ **Recoverable-error throw path leaks device buffers in server mode (regression introduced 2026-05-16).** `run_cuda_test` allocates `d_A/d_B/d_C`, `h_C`, and a `module` with no RAII; the cleanup at the end is only reached on the success path. When a recoverable error now throws (e.g. a bad `--load-c`/`--reference-c` path or size mismatch), the server loop catches it and continues, leaking ~3× buffer (up to 768 MiB at default sizes) + the module per failed command. Before the pass these were `exit()` → process death → OS reclaim, so it's a genuine new leak. Bounded (needs repeated malformed commands; restart clears it) but real. **Fix:** wrap the body in RAII or a `try{…}catch{ cleanup; rethrow; }` — do this together with item 29 so driver errors don't leak either.

### Local syntax-check workflow (no GPU needed)

When this file is touched, a quick sanity check that the host still compiles cleanly can be done without nvcc:

```bash
mkdir -p /tmp/cuda_stubs
# Hand-rolled stub headers for cuda.h, cuda_runtime.h, nvrtc.h, nvml.h, omp.h
# (see the -Wall squash commit 5abbdb3 for the minimal set used)
clang++ -std=c++17 -fsyntax-only -isystem /tmp/cuda_stubs -I . -Wall -Wextra \
  -Wno-unused-parameter -Wno-deprecated-declarations QuickRunCUDA.cpp
```

This catches signed/unsigned comparisons, init-list order issues, missing includes, etc. It does NOT catch CUDA-specific issues like wrong cuLaunchKernel arg counts (the stubs are loose).

---

## 9. The `tests/` corpus

~1,130 `.cu` files (of which ~780 are `bench_*.cu`) — it grew a lot during the late-April-2026 audit sprint. Roughly five clusters by purpose:

### 9.1 The L2-side-aware hackathon project
- `side_aware.cu` (~400 lines) — the actual project that won the SemiAnalysis Blackwell hackathon. FP32 absmax reduction that splits L2 traffic by side. Has `init` (figures out per-SM side + per-2MiB-page side from latency probes) and `kernel` (reduction).
- `default_kernel.cu` (top-level) — minimal "Hello world" kernel. Doesn't reference A/B/C; just prints thread/block IDs.

### 9.2 The F2FP / CVT / MUFU pipe catalog (`bench_f2fp_*`, `bench_cvt_*`, `bench_mufu_*`)
Comprehensive microbenchmarks for the SFU pipe. See `F2FP_DEEP_DIVE.md` for the master findings doc. Key files:
- `bench_latency_calib.cu` — calibration harness; produces L_op = 4.00 cy for FFMA/IMAD/FMUL (validates the diff-of-depths method)
- `bench_f2fp_pure.cu` / `_oneway.cu` / `_pack_variants.cu` — throughput ceilings
- `bench_f2fp_depth.cu` — K-deep dependency chains for latency
- `bench_f2fp_port_hypothesis.cu` — the rigorous discrimination between MERGE_C / PACK_AB / UNPACK_B hypotheses
- `test_cvt_correctness.cu` — golden-output verifier for narrow CVT (uses `--reference-c` pattern but inlines the expected values via `printf`, not via reference comparison; intended to be eyeballed)
- `sass_map.cu` — minimal kernel that emits every CVT variant once for SASS reference

### 9.3 The B300 pipe catalog (`bench_pipe_*`, `bench_balanced_pair_*`, `bench_triple_*`, `bench_redux_*`, etc.)
Pair-contention sweeps used to derive the B300 pipe model. Driven by `run_pipe_matrix.sh`. See `B300_PIPE_CATALOG.md` (~19,700 lines) for findings — but prefer the audited `b300_clean/B300_TRUE_REFERENCE.md` for numbers (§14).

### 9.4 Narrow-format quantization (`quantize_*.cu`, `four_six_fp4.cu`)
- `quantize_bf16_to_nvfp4.cu` — flat BF16→NVFP4, 93% of DRAM copy BW
- `quantize_row_bf16_to_nvfp4.cu` — row-wise version with absmax-per-row scales, e4m3 microscales
- `four_six_fp4.cu` (~535 lines) — production quantize variants (no longer the largest test; the `bench_tcgen05_bf16_*` kernels now top out ~800–890 LOC)

### 9.5 Misc / DRAM / atomics / latency
Everything else: `bench_dram_*` (HBM BW variants), `bench_atom*` (atomic contention), `bench_ldg_*` / `bench_lds_*` (load throughput), `bench_tma*` / `bench_tcgen05_*` (Blackwell-only paths), `bench_imma_*` / `bench_hmma_*` (tensor cores).

### 9.6 Drivers
- `tests/run_microbench.sh` — DRAM + FP32 FMA + the full CVT narrow-format throughput sweep. Read this if you want to know the canonical command line for any benchmark.
- `tests/run_pipe_matrix.sh` — SOLO + PAIR sweeps over the pipe-catalog matrix.
- `tests/sweep_cvt.sh` — exhaustive CVT format sweep.
- `tests/sweep_mix_e2m1.sh` — e2m1x2 + companion-instruction interactions.

### 9.7 Supporting data
- `tests/reference/` — older HBM BW reference kernels.
- `tests/golden/` — golden `.bin` files for `--reference-c` comparison (NVFP4 outputs).
- `tests/sass_tools/` — Python scripts (`patch_lop3.py`, `rename_lop3.py`) for SASS rewriting. `SASS_MODIFICATION.md` documents the workflow.

### 9.8 Stale-looking files (review before deleting)
- `tests/0_WIP.cu` — looks abandoned.
- `tests/1_LLMC.cu`, `tests/1_RELU.cu` — old test kernels.
- `tests/vector_add.cu` — GPU MODE leaderboard kernel (referenced in a Feb 2026 commit message).
- `tests/icache.cu` + `tests/icache_sweep.py` — i-cache probe, not in any driver.

Don't delete these without checking they're not still referenced from a research log.

### 9.9 Conventions when adding a new bench kernel

Cribbed from `bench_fp32_fma.cu` and applied across the F2FP / CVT bench files:

1. **Make inputs depend on `threadIdx.x`** (e.g., `float tid_f = __int_as_float(threadIdx.x | 0x3F800000u)`). Defeats cross-thread constant folding.
2. **Keep a loop-carried dependency to `C[]` under an impossible `if`**: `if (threadIdx.x >= blockDim.x) C[threadIdx.x] = r0 + r1 + ...;` — the compiler can't prove the branch is dead at compile time, so it can't DCE the accumulator chain.
3. **`#ifndef UNROLL` guard** so the runner can inject `-H "#define UNROLL 8"`.
4. **Inline `asm volatile` blocks** for the hot loop so ptxas can't reorder / fold.
5. **`__launch_bounds__(threads, blocks_per_sm, min_blocks_per_sm)`** when the kernel needs a specific register budget or occupancy (see `side_aware.cu` line 257).
6. **`#pragma unroll 1` outer + `#pragma unroll` inner** when you want unroll only at one level.

---

## 10. Performance metric mechanics

The "ops/s" and "% SOL" output is computed as:

```
multiplier := args.perfMultiplier            // from -P
if args.perfMultiplierPerThread > 0:
    multiplier := args.perfMultiplierPerThread * threadsPerBlockX * numBlocksX
perf := multiplier / (avg_time_ms / 1000)    // → ops per second
if perfSpeedOfLight > 0:
    sol_pct := 100.0 * perf / perfSpeedOfLight
```

So `-P` is "total operations executed in one launch" — used for memcpy-like benchmarks where the work is the same regardless of grid size. `-N` is "operations per thread per launch" — used for compute-bound benchmarks where the work scales with the grid.

Setting `-U "GB/s"` and `-P` to total bytes / 1e9 gives bandwidth. Setting `-U "TFLOPS"` and `-N` to TFLOPS-per-thread (e.g., `ITERS * 8_fmas_per_iter * 2_flops_per_fma / 1e12`) gives compute throughput.

---

## 11. Compilation pipeline & SASS

1. Source + header → NVRTC → CUBIN in memory.
2. CUBIN → `output.cubin` (CWD).
3. `nvdisasm output.cubin --print-code` → `sass/<basename>[_<headerhash>].sass`.
4. `cp output.cubin sass/<basename>[_<headerhash>].cubin`.
5. `cuModuleLoadData(cubin)` → `cuModuleGetFunction("kernel")`.

The SASS dump is the most important output for any non-trivial benchmark — it's how you confirm the compiler emitted the SASS opcodes you expected. The hash mechanism for `-H` makes header variants distinguishable in the same `sass/` directory.

If `nvdisasm` is missing, the SASS dump silently fails (`2>/dev/null` in the `system()` call). The CUBIN copy still happens.

---

## 12. Multi-GPU (`multigpu/MGFenceBench.cpp`)

Standalone (not linked into `QuickRunCUDA`). Same NVRTC + 3-buffer model, but:
- Allocates `A` and/or `B` on a *remote* GPU via P2P (`cuCtxEnablePeerAccess`), with `--remote-a` / `--b-remote` flags.
- `C` always on primary for readback.
- No L2 flush, no init kernel, no compile cache. Just a tight loop with overall timing.

Drop-in for benchmarks like cross-GPU fence cost (the file name). Build with:
```
nvcc -O2 -std=c++17 -lcuda -lnvrtc multigpu/MGFenceBench.cpp -o MGFenceBench
```
(There is no Makefile target — add one if you start iterating.)

The `.gitignore` in `multigpu/` excludes `MGFenceBench`, so the binary won't accidentally land in commits.

---

## 13. Quick recipes

### Compute-bound: FP32 FMA TFLOPS
```
./QuickRunCUDA tests/bench_fp32_fma.cu \
  -H "#define UNROLL 8" \
  -p -t 256 -0 8192 \
  -T 100 -N $(python3 -c "print(8192*8*2/1e12)") -U TFLOPS -L 76.9
```
- `-p`: one CTA per SM
- `-N`: TFLOPs per thread = `iters * fmas_per_iter * 2_flops`
- `-L`: B300 SOL ≈ 76.9 TFLOPS

### Memory-bound: HBM copy GB/s
```
./QuickRunCUDA tests/bench_dram_bw.cu \
  -A $((64*1024*1024)) -C $((64*1024*1024)) \
  -t 256 -b $((64*1024*1024 / 4 / 256)) \
  -T 100 -P $(python3 -c "print(64*1024*1024*4*2/1e9)") -U GB/s -L 8184 \
  --l2flush 1
```
- `-l2flush 1`: cold-L2 each run-group

### L2-side-aware reduction
```
./QuickRunCUDA tests/side_aware.cu \
  -i -p -t 1024 -A 1000000000 -0 1000000000 \
  -T 100 -P 4.0 -U GB/s
```
- `-i`: run init kernel first (latency-probe per SM)
- `-P 4.0`: 4 bytes per input dword, so GB/s = total bytes / elapsed

### Quick sanity (no benchmarking)
```
./QuickRunCUDA tests/test_cvt_correctness.cu -t 1 -b 1
```

### Sweep via server mode (Python)
```python
from utils.cuda_controller import CUDAController
ctl = CUDAController()
for unroll in (1, 2, 4, 8, 16):
    print(unroll, ctl.send_command([
        'tests/bench_fp32_fma.cu',
        '-H', f"'#define UNROLL {unroll}'",  # NB single quotes
        '-p', '-t', 256, '-0', 8192,
        '-T', 100, '-N', 0.131, '-U', 'TFLOPS'
    ]))
```

---

## 14. What's actively in motion

Active branch is `f2fp-deep-dive`. The corpus is being extended; the framework is stable.

Open research threads visible in repo state:
- **Rigor-verified reference (start here):** `b300_clean/B300_TRUE_REFERENCE.md` — the master summary from the multi-method (wall-clock + ncu + SASS) audit sprint, every number citing the commit that produced it. Category breakdowns in `b300_clean/01_*.md`..`17_*.md`; per-claim re-verification in `b300_clean/M3_REVERIFY_LOG.md`; new-task backlog in `b300_clean/CURIOSITY_LIST_V4.md`. Run `utils/rigor_run.sh ./binary` to get 3-method verification for a new measurement.
- F2FP / SFU pipe characterization (`F2FP_DEEP_DIVE.md`, ongoing)
- B300 pipe catalog (`B300_PIPE_CATALOG.md` — 18K lines, legacy/unverified) and its audited rewrites `ULTRA_DENSE`/`DENSE`/`JUSTIFIED_B300_PIPE_CATALOG.md` + `AUDIT_NOTES.md` / `CRITIQUE.md` / `STATUS_OF_REPLICATION.md`. Trust order is documented in the top-level `CLAUDE.md`.
- Future ideas in `FUTURE_IDEAS.md` (tcgen05 peak, MX block-scaled, multi-GPU TMA, etc.)
- Skeptical-list retractions in `SKEPTICAL_LIST.md` (things claimed too confidently — see for what's been walked back)

The `MICROBENCH_RESULTS_GPU1.md` (top-level) and `tests/MICROBENCH_RESULTS.md` are run logs for the two GPUs in the test environment.

---

## 15. Index of key files (`ag`-friendly)

```
QuickRunCUDA.cpp                  680 LOC   main host
default_kernel.cu                   3 LOC   minimal kernel template
Makefile                            93 LOC  build, arch auto-detect
utils/cuda_helper.h               ~139 LOC  NVRTC wrapper (compileFileToCUBIN/loadCUBIN), post-2026-05-16 strip
utils/ipc_helper.h                  60 LOC  FIFO IPC
utils/nvmlClass.h                  370 LOC  clock lock + sampling
utils/CLI11.hpp                  vendored   CLI parser
utils/cuda_controller.py            42 LOC  Python client (server mode)
utils/git_and_run.py               325 LOC  watchdog + auto-commit research tool
tests/side_aware.cu                400 LOC  hackathon project (only kernel with init+main)
tests/run_microbench.sh            230 LOC  canonical CVT/DRAM/FMA driver
tests/run_pipe_matrix.sh           212 LOC  SOLO + PAIR pipe-catalog driver
F2FP_DEEP_DIVE.md                  411 lines current focus doc
B300_PIPE_CATALOG.md           ~19,700 lines legacy research doc (prefer b300_clean/)
MICROBENCH_RESULTS_GPU1.md         548 lines per-GPU run log
docs/REPORT_*.md                12 reports   historical pipe-audit reports
docs/SFU_*.md                      3 notes   SFU investigation notes
docs/HARNESS.md                this file     framework reference
b300_clean/B300_TRUE_REFERENCE.md  rigor-verified master summary (START HERE for numbers)
b300_clean/CURIOSITY_LIST_V4.md    new-investigation backlog
b300_clean/                     ~290 docs   audited per-category breakdowns + rigor proofs (incl. 01_*..17_*)
ULTRA_DENSE/DENSE/JUSTIFIED_B300_PIPE_CATALOG.md  audited catalog rewrites
justifications/                 ~95 entries  per-claim justification artifacts
utils/rigor_run.sh                          3-method (wall-clock+ncu+SASS) verification driver
```
