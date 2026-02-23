# QuickRunCUDA Simplification Plan

## Goal

Strip QuickRunCUDA down to its core: compile CUDA source, run it, measure performance. Replace the current direct-to-CUBIN compilation with a two-step pipeline (source -> PTX -> CUBIN) with saved intermediate files and the ability to skip steps.

## Summary of Changes

| Action | What |
|--------|------|
| **Remove** | Server mode (IPC pipes, command loop, stdout capture) |
| **Remove** | Clock control (nvmlClass.h, NVML init, fan control, stats thread) |
| **Remove** | `ipc_helper.h`, `nvmlClass.h`, `cuda_controller.py`, `git_and_run.py`, `_run.sh` |
| **Remove** | Unused linked libraries (`libcupti`, `libnvidia-ml`, `lnvperf_*`, `libcurand`) |
| **Remove** | Unused CudaHelper class (streams, timers, launch wrappers, device props) |
| **Remove** | `-i`/`--runInitKernel` flag (auto-detect init function instead) |
| **Remove** | `-N`/`--perfMultiplierPerThread` and `-U`/`--perfMultiplier-unit` |
| **Modify** | Compilation pipeline: source -> PTX (saved) -> CUBIN (saved) |
| **Modify** | CLI args: remove server/clock/reuse-cubin, add PTX/CUBIN input options |
| **Modify** | Makefile: remove unnecessary library links |
| **Modernize** | Replace `malloc`/`free` with `std::vector`, `NULL` with `nullptr`, `uint` with `uint32_t`, consistent I/O style, cleaner string building |
| **Add** | `/PTX` and `/CUBIN` directories for saved artifacts |
| **Add** | `--ptx-input` and `--cubin-input` flags to skip compilation steps |
| **Keep** | All kernel execution, timing, memory, array I/O, L2 flush functionality |

---

## New Compilation Pipeline

```
Option A (default): Source .cu file
  1. NVRTC: source -> PTX (with --generate-line-info)
  2. Save PTX to PTX/<datetime>_<charcount>.ptx
  3. Driver API: PTX -> CUBIN (via cuLinkCreate)
  4. Save CUBIN to CUBIN/<datetime>_<charcount>.cubin
  5. Load CUBIN module, get kernel function

Option B: --ptx-input <file.ptx>
  1. Read PTX from file (skip NVRTC compilation)
  2. Driver API: PTX -> CUBIN
  3. Save CUBIN to CUBIN/<datetime>_<charcount>.cubin
  4. Load CUBIN module, get kernel function

Option C: --cubin-input <file.cubin>
  1. Read CUBIN from file (skip all compilation)
  2. Load CUBIN module, get kernel function
```

### NVRTC PTX Generation

Change from `nvrtcGetCUBIN` to `nvrtcGetPTX`. This requires changing the `--gpu-architecture` from `sm_XY` to `compute_XY` (PTX virtual architecture):

```cpp
// BEFORE (generates CUBIN):
"--gpu-architecture=sm_90a"
nvrtcGetCUBINSize(prog, &codeSize);
nvrtcGetCUBIN(prog, code);

// AFTER (generates PTX):
"--gpu-architecture=compute_90a"
nvrtcGetPTXSize(prog, &codeSize);
nvrtcGetPTX(prog, code);
```

### PTX to CUBIN via Linker API

Use `cuLinkCreate` / `cuLinkAddData` / `cuLinkComplete` to JIT PTX to CUBIN. This gives us the CUBIN binary to save before loading the module:

```cpp
CUlinkState linkState;
void *cubinData;
size_t cubinSize;

char infoLog[4096] = {0}, errorLog[4096] = {0};
CUjit_option options[] = {
    CU_JIT_GENERATE_LINE_INFO,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER
};
void *optionValues[] = {
    (void*)(uintptr_t)1,
    (void*)(uintptr_t)sizeof(infoLog), infoLog,
    (void*)(uintptr_t)sizeof(errorLog), errorLog
};

checkCudaErrors(cuLinkCreate(5, options, optionValues, &linkState));
checkCudaErrors(cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
    (void*)ptx, ptxSize, "kernel.ptx", 0, nullptr, nullptr));
checkCudaErrors(cuLinkComplete(linkState, &cubinData, &cubinSize));

// Save CUBIN, load module, THEN destroy linker (cubinData invalidated by destroy)
saveFile("CUBIN", "cubin", cubinData, cubinSize, charCount);
CUmodule module;
checkCudaErrors(cuModuleLoadData(&module, cubinData));
checkCudaErrors(cuLinkDestroy(linkState));
```

### Filename Format

```
PTX/20260223_143052_12847.ptx
CUBIN/20260223_143052_12847.cubin

Format: YYYYMMDD_HHMMSS_<charcount>.<ext>
```

Where `<charcount>` is the number of characters in the original CUDA source (or PTX source if `--ptx-input` is used). Same charcount for PTX and CUBIN when generated in the same run.

---

## Auto-Detect Init Kernel

Remove the `-i`/`--runInitKernel` flag. Instead, probe the module for an `init` function after loading:

```cpp
CUfunction kernel_addr, init_addr;
checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "kernel"));

// Auto-detect init function (no flag needed)
bool has_init = (cuModuleGetFunction(&init_addr, module, "init") == CUDA_SUCCESS);
if (has_init) {
    printf("Found init() kernel, running it first\n");
}
```

`cuModuleGetFunction` returns `CUDA_ERROR_NOT_FOUND` if the function doesn't exist, which is not fatal. We just check the return value directly instead of wrapping in `checkCudaErrors`.

---

## Remove perfMultiplierPerThread and perfMultiplier-unit

The performance output section simplifies to:

```cpp
if (args.perfMultiplier > 0.0f) {
    float perf = args.perfMultiplier / (avg_time / 1000.f);
    printf(" ==> %.4f", perf);
    if (args.perfSpeedOfLight > 0.0f) {
        printf(" ==> %.3f%%", 100.0f * perf / args.perfSpeedOfLight);
    }
}
```

No more `perfMultiplierPerThread`, no more `perfMultiplier_unit` string. The user knows what units they're working with from `-P`.

---

## Code Modernization

Targeted cleanup of the worst "old C" patterns. Not a rewrite, just making it less painful.

### 1. `malloc`/`free` -> `std::vector`

```cpp
// BEFORE:
uint *h_C = reinterpret_cast<uint *>(malloc(sizeC));
// ... use h_C ...
free(h_C);

// AFTER:
std::vector<uint32_t> h_C(args.arrayDwordsC, 0);
// ... use h_C.data() for CUDA calls ...
// automatic cleanup
```

Same for `h_A`, `h_B`, `ref_C`.

### 2. `NULL` -> `nullptr`

Throughout the file. Also `0` used as null pointer in `cuLaunchKernel(..., 0)` becomes `nullptr`.

### 3. `uint` -> `uint32_t`

More explicit, more portable. The `uint` typedef is a Linux-ism.

### 4. Consistent error output

Pick `fprintf(stderr, ...)` everywhere (it's already the dominant style). Remove the lone `std::cerr` usages.

### 5. Cleaner string building for NVRTC options

The old code mallocs each compile option individually. Use stack-allocated strings:

```cpp
// BEFORE (cuda_helper.h):
std::string other_options_0 = "--generate-line-info";
compileParams[0] = reinterpret_cast<char *>(malloc(sizeof(char) * (other_options_0.length() + 1)));
strcpy(compileParams[0], other_options_0.c_str());
// ... repeat 4 more times, then free each one ...

// AFTER (inline in QuickRunCUDA.cpp):
char archBuf[32];
snprintf(archBuf, sizeof(archBuf), "--gpu-architecture=compute_%d%d%s", major, minor,
         ((major == 9 && minor == 0) || (major == 10 && minor == 0)) ? "a" : "");

const char *opts[] = {
    "--generate-line-info", "-use_fast_math", "--std=c++17",
    archBuf, "-I/usr/local/cuda/include/"
};
```

### 6. File I/O helper

The pattern "open binary, seek end, get size, seek start, read" appears 3+ times. Extract it:

```cpp
std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    size_t size = f.tellg();
    std::vector<char> data(size + 1);
    f.seekg(0);
    f.read(data.data(), size);
    data[size] = '\0';
    return data;
}
```

### 7. Extract `launchKernel` helper

The `cuFuncSetAttribute` + `cuLaunchKernel` pattern repeats for init and main kernel. Extract:

```cpp
void launchKernel(CUfunction func, const CmdLineArgs& args, void** kernel_args) {
    checkCudaErrors(cuFuncSetAttribute(func,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, args.sharedMemoryBlockBytes));
    checkCudaErrors(cuFuncSetAttribute(func,
        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, args.sharedMemoryCarveoutBytes));
    checkCudaErrors(cuLaunchKernel(func,
        args.numBlocksX, 1, 1, args.threadsPerBlockX, 1, 1,
        args.sharedMemoryBlockBytes, nullptr, kernel_args, nullptr));
}
```

---

## Detailed File-by-File Changes

### 1. QuickRunCUDA.cpp

#### CmdLineArgs (final state):
```cpp
struct CmdLineArgs {
    // Array sizes (in 32-bit dwords)
    size_t arrayDwordsA = 64 * 1024 * 1024;
    size_t arrayDwordsB = 64 * 1024 * 1024;
    size_t arrayDwordsC = 64 * 1024 * 1024;

    // Array initialization
    bool randomArrayA = false;
    bool randomArrayB = false;
    uint32_t randomArraysBitMask = 0xFFFFFFFF;
    uint32_t randomSeed = 1234;

    // Kernel config
    int kernel_int_args[3] = {0};
    int threadsPerBlockX = 32;
    int numBlocksX = 1;
    bool persistentBlocks = false;
    int sharedMemoryBlockBytes = 0;
    int sharedMemoryCarveoutBytes = 0;

    // Benchmark
    int timedRuns = 0;
    float perfMultiplier = 0.0f;
    float perfSpeedOfLight = 0.0f;
    bool listIndividualTimes = false;

    // L2 flush
    enum L2FlushMode { NO_FLUSH = 0, FLUSH_AT_START = 1, FLUSH_EVERY_RUN = 2 };
    L2FlushMode l2FlushMode = NO_FLUSH;

    // Compilation
    std::string kernel_filename = "default_kernel.cu";
    std::string header;
    std::string ptx_input;     // NEW: skip NVRTC, load PTX directly
    std::string cubin_input;   // NEW: skip all compilation, load CUBIN directly

    // Array I/O
    std::string dump_c_array;
    std::string dump_c_format = "raw";
    std::string load_c_array;
    std::string reference_c_filename;
    float compare_tolerance = 0.0f;

    // Positional
    std::vector<std::string> positional_args;
};
```

**Removed fields**: `server_mode`, `clock_speed`, `reuse_cubin`, `runInitKernel`, `perfMultiplierPerThread`, `perfMultiplier_unit`

**Added fields**: `ptx_input`, `cubin_input`

#### setupCommandLineParser() changes:

Remove:
- `--server`, `--clock-speed`, `--reuse-cubin`
- `-i, --runInitKernel`
- `-N, --perfMultiplierPerThread`
- `-U, --perfMultiplier-unit`
- The `modes_group` entirely (nothing left in it)

Add:
```cpp
auto compile_group = app.add_option_group("Compilation Input");
compile_group->add_option("--ptx-input", args.ptx_input, "Load PTX directly (skip NVRTC)")
    ->check(CLI::ExistingFile);
compile_group->add_option("--cubin-input", args.cubin_input, "Load CUBIN directly (skip all compilation)")
    ->check(CLI::ExistingFile);
```

#### Remove entirely:
- `parseCommandString()` function (~50 lines)
- Server mode block in `main()` (~45 lines)
- Clock control in `main()` (~3 lines)
- `#include "utils/ipc_helper.h"` and `#include "utils/nvmlClass.h"`

#### Simplify main():
```cpp
int main(int argc, char **argv) {
    CLI::App app{"QuickRunCUDA: Fast iteration for CUDA microbenchmarking"};
    app.set_help_all_flag("--help-all", "Show all help options");
    CmdLineArgs args;
    setupCommandLineParser(app, args);

    try {
        CLI11_PARSE(app, argc, argv);
        if (!args.positional_args.empty())
            args.kernel_filename = args.positional_args[0];

        checkCudaErrors(cuInit(0));
        checkCudaErrors(cuDeviceGet(&cuDeviceGlobal, 0));
        checkCudaErrors(cuCtxCreate(&cuContextGlobal, 0, cuDeviceGlobal));

        return run_cuda_test(args);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }
}
```

#### New helper functions:

```cpp
// --- File I/O ---

std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    size_t size = f.tellg();
    std::vector<char> data(size + 1);
    f.seekg(0);
    f.read(data.data(), size);
    data[size] = '\0';
    return data;
}

void ensureDir(const char* dir) {
    struct stat st;
    if (stat(dir, &st) != 0) mkdir(dir, 0755);
}

std::string makeTimestampedPath(const char* dir, const char* ext, size_t charCount) {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct tm tm;
    localtime_r(&t, &tm);
    char buf[128];
    snprintf(buf, sizeof(buf), "%s/%04d%02d%02d_%02d%02d%02d_%zu.%s",
             dir, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec, charCount, ext);
    return buf;
}

void saveFile(const char* dir, const char* ext, const void* data, size_t size, size_t charCount) {
    ensureDir(dir);
    std::string path = makeTimestampedPath(dir, ext, charCount);
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)data, size);
    printf("%s saved: %s\n", ext, path.c_str());
}

// --- Compilation ---

std::vector<char> compileSourceToPTX(CUdevice device, const char* filename,
                                      const char* header, size_t& sourceCharCount) {
    // Read source file
    std::ifstream inputFile(filename, std::ios::binary | std::ios::ate);
    if (!inputFile) {
        fprintf(stderr, "Error: unable to open %s\n", filename);
        exit(EXIT_FAILURE);
    }
    size_t inputSize = inputFile.tellg();
    size_t headerSize = header ? strlen(header) : 0;
    sourceCharCount = inputSize + headerSize + 1;

    std::vector<char> source(sourceCharCount + 1);
    memcpy(source.data(), header, headerSize);
    source[headerSize] = '\n';
    inputFile.seekg(0);
    inputFile.read(source.data() + headerSize + 1, inputSize);
    source[sourceCharCount] = '\0';

    // Detect architecture
    int major = 0, minor = 0;
    checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    char archBuf[32];
    snprintf(archBuf, sizeof(archBuf), "--gpu-architecture=compute_%d%d%s",
             major, minor,
             ((major == 9 && minor == 0) || (major == 10 && minor == 0)) ? "a" : "");

    const char* opts[] = {
        "--generate-line-info", "-use_fast_math", "--std=c++17",
        archBuf, "-I/usr/local/cuda/include/"
    };

    nvrtcProgram prog;
    checkCudaErrors(nvrtcCreateProgram(&prog, source.data(), filename, 0, nullptr, nullptr));
    nvrtcResult res = nvrtcCompileProgram(prog, 5, opts);

    // Print log on error
    size_t logSize;
    checkCudaErrors(nvrtcGetProgramLogSize(prog, &logSize));
    if (logSize > 1) {
        std::vector<char> log(logSize + 1);
        checkCudaErrors(nvrtcGetProgramLog(prog, log.data()));
        fprintf(stderr, "\n------- COMPILATION LOG -------\n%s\n------- END LOG -------\n", log.data());
    }
    checkCudaErrors(res);

    // Extract PTX
    size_t ptxSize;
    checkCudaErrors(nvrtcGetPTXSize(prog, &ptxSize));
    std::vector<char> ptx(ptxSize);
    checkCudaErrors(nvrtcGetPTX(prog, ptx.data()));
    checkCudaErrors(nvrtcDestroyProgram(&prog));
    return ptx;
}

CUmodule compilePTXtoCUBIN(const char* ptx, size_t ptxSize, size_t charCount) {
    CUlinkState linkState;
    void *cubinData;
    size_t cubinSize;

    char infoLog[4096] = {0}, errorLog[4096] = {0};
    CUjit_option options[] = {
        CU_JIT_GENERATE_LINE_INFO,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER
    };
    void *optionValues[] = {
        (void*)(uintptr_t)1,
        (void*)(uintptr_t)sizeof(infoLog), infoLog,
        (void*)(uintptr_t)sizeof(errorLog), errorLog
    };

    checkCudaErrors(cuLinkCreate(5, options, optionValues, &linkState));
    CUresult linkRes = cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
                                      (void*)ptx, ptxSize, "kernel.ptx",
                                      0, nullptr, nullptr);
    if (linkRes != CUDA_SUCCESS) {
        fprintf(stderr, "PTX link error:\n%s\n", errorLog);
        exit(EXIT_FAILURE);
    }
    checkCudaErrors(cuLinkComplete(linkState, &cubinData, &cubinSize));

    // Save CUBIN and load module before destroying linker
    saveFile("CUBIN", "cubin", cubinData, cubinSize, charCount);
    CUmodule module;
    checkCudaErrors(cuModuleLoadData(&module, cubinData));
    checkCudaErrors(cuLinkDestroy(linkState));
    return module;
}

// --- Kernel launch ---

void launchKernel(CUfunction func, const CmdLineArgs& args, void** kernel_args) {
    checkCudaErrors(cuFuncSetAttribute(func,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, args.sharedMemoryBlockBytes));
    checkCudaErrors(cuFuncSetAttribute(func,
        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, args.sharedMemoryCarveoutBytes));
    checkCudaErrors(cuLaunchKernel(func,
        args.numBlocksX, 1, 1, args.threadsPerBlockX, 1, 1,
        args.sharedMemoryBlockBytes, nullptr, kernel_args, nullptr));
}
```

#### Modified run_cuda_test() (full rewrite):

```cpp
int run_cuda_test(CmdLineArgs& args) {
    // --- Persistent blocks ---
    if (args.persistentBlocks) {
        checkCudaErrors(cuDeviceGetAttribute(&args.numBlocksX,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDeviceGlobal));
        printf("Using persistent blocks (%d = 1 per SM)\n", args.numBlocksX);
        args.persistentBlocks = false;
    }

    // --- Compilation pipeline ---
    CUmodule module;
    if (!args.cubin_input.empty()) {
        // Path C: CUBIN directly
        auto cubin = readFile(args.cubin_input);
        checkCudaErrors(cuModuleLoadData(&module, cubin.data()));
    } else if (!args.ptx_input.empty()) {
        // Path B: PTX -> CUBIN
        auto ptx = readFile(args.ptx_input);
        module = compilePTXtoCUBIN(ptx.data(), ptx.size(), ptx.size());
    } else {
        // Path A: Source -> PTX -> CUBIN
        size_t sourceCharCount;
        auto ptx = compileSourceToPTX(cuDeviceGlobal, args.kernel_filename.c_str(),
                                       args.header.c_str(), sourceCharCount);
        saveFile("PTX", "ptx", ptx.data(), ptx.size(), sourceCharCount);
        module = compilePTXtoCUBIN(ptx.data(), ptx.size(), sourceCharCount);
    }

    // --- Get kernel functions ---
    CUfunction kernel_addr;
    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "kernel"));

    CUfunction init_addr;
    bool has_init = (cuModuleGetFunction(&init_addr, module, "init") == CUDA_SUCCESS);

    // --- Allocate GPU memory ---
    CUdeviceptr d_A, d_B, d_C;
    size_t sizeA = args.arrayDwordsA * sizeof(uint32_t);
    size_t sizeB = args.arrayDwordsB * sizeof(uint32_t);
    size_t sizeC = args.arrayDwordsC * sizeof(uint32_t);
    checkCudaErrors(cuMemAlloc(&d_A, sizeA));
    checkCudaErrors(cuMemAlloc(&d_B, sizeB));
    checkCudaErrors(cuMemAlloc(&d_C, sizeC));

    // --- Initialize arrays ---
    std::vector<uint32_t> h_C(args.arrayDwordsC, 0);
    if (!args.load_c_array.empty()) {
        auto data = readFile(args.load_c_array);
        if (data.size() - 1 != sizeC) {  // -1 for null terminator added by readFile
            fprintf(stderr, "File size mismatch for --load-c\n");
            exit(EXIT_FAILURE);
        }
        memcpy(h_C.data(), data.data(), sizeC);
        checkCudaErrors(cuMemcpyHtoD(d_C, h_C.data(), sizeC));
    } else {
        checkCudaErrors(cuMemsetD8(d_C, 0, sizeC));
    }

    if (args.randomArrayA || args.randomArrayB) {
        std::vector<uint32_t> h_A(args.arrayDwordsA), h_B(args.arrayDwordsB);
        constexpr size_t chunk_size = 1024 * 1024;
        size_t num_chunks = (std::max(args.arrayDwordsA, args.arrayDwordsB) + chunk_size - 1) / chunk_size;
        #pragma omp parallel for schedule(static)
        for (size_t chunk = 0; chunk < num_chunks; chunk++) {
            std::mt19937_64 rng(args.randomSeed + chunk);
            std::uniform_int_distribution<uint32_t> dist;
            if (args.randomArrayA) {
                size_t end = std::min((chunk + 1) * chunk_size, args.arrayDwordsA);
                for (size_t i = chunk * chunk_size; i < end; ++i)
                    h_A[i] = dist(rng) & args.randomArraysBitMask;
            }
            if (args.randomArrayB) {
                size_t end = std::min((chunk + 1) * chunk_size, args.arrayDwordsB);
                for (size_t i = chunk * chunk_size; i < end; ++i)
                    h_B[i] = dist(rng) & args.randomArraysBitMask;
            }
        }
        if (args.randomArrayA) checkCudaErrors(cuMemcpyHtoD(d_A, h_A.data(), sizeA));
        if (args.randomArrayB) checkCudaErrors(cuMemcpyHtoD(d_B, h_B.data(), sizeB));
    }
    if (!args.randomArrayA) checkCudaErrors(cuMemsetD8(d_A, 0, sizeA));
    if (!args.randomArrayB) checkCudaErrors(cuMemsetD8(d_B, 0, sizeB));

    // --- Prepare kernel arguments ---
    int kernel_int_args[3] = {args.kernel_int_args[0], args.kernel_int_args[1], args.kernel_int_args[2]};
    void *kernel_args[] = { &d_A, &d_B, &d_C,
                            &kernel_int_args[0], &kernel_int_args[1], &kernel_int_args[2] };

    // --- Init kernel (auto-detected) ---
    if (has_init) {
        printf("Running init() kernel\n");
        launchKernel(init_addr, args, kernel_args);
    }
    if (args.l2FlushMode >= CmdLineArgs::FLUSH_AT_START) flushL2Cache();

    // --- Warm-up run ---
    launchKernel(kernel_addr, args, kernel_args);
    if (args.l2FlushMode == CmdLineArgs::FLUSH_AT_START) flushL2Cache();
    checkCudaErrors(cuCtxSynchronize());

    // --- Timed runs ---
    if (args.timedRuns > 0) {
        bool individual_events = (args.l2FlushMode >= CmdLineArgs::FLUSH_EVERY_RUN
                                  || args.listIndividualTimes);

        CUevent overall_start, overall_stop;
        checkCudaErrors(cuEventCreate(&overall_start, CU_EVENT_DEFAULT));
        checkCudaErrors(cuEventCreate(&overall_stop, CU_EVENT_DEFAULT));

        std::vector<CUevent> start_events, stop_events;
        std::vector<float> run_times(args.timedRuns);
        if (individual_events) {
            start_events.resize(args.timedRuns);
            stop_events.resize(args.timedRuns);
            for (int i = 0; i < args.timedRuns; i++) {
                checkCudaErrors(cuEventCreate(&start_events[i], CU_EVENT_DEFAULT));
                checkCudaErrors(cuEventCreate(&stop_events[i], CU_EVENT_DEFAULT));
            }
        }

        checkCudaErrors(cuEventRecord(overall_start, nullptr));
        for (int i = 0; i < args.timedRuns; i++) {
            if (args.l2FlushMode == CmdLineArgs::FLUSH_EVERY_RUN) flushL2Cache();
            if (individual_events) checkCudaErrors(cuEventRecord(start_events[i], nullptr));
            launchKernel(kernel_addr, args, kernel_args);
            if (individual_events) checkCudaErrors(cuEventRecord(stop_events[i], nullptr));
        }
        checkCudaErrors(cuEventRecord(overall_stop, nullptr));
        checkCudaErrors(cuEventSynchronize(overall_stop));

        float total_time = 0.f;
        if (individual_events) {
            for (int i = 0; i < args.timedRuns; i++) {
                checkCudaErrors(cuEventElapsedTime(&run_times[i], start_events[i], stop_events[i]));
                total_time += run_times[i];
            }
        }

        float overall_time = 0.f;
        checkCudaErrors(cuEventElapsedTime(&overall_time, overall_start, overall_stop));

        if (args.listIndividualTimes) {
            printf("Individual runtimes: ");
            for (int i = 0; i < args.timedRuns; i++)
                printf("%.5f%s", run_times[i], i < args.timedRuns - 1 ? " / " : "\n");
        }

        float avg_time = (individual_events ? total_time : overall_time) / args.timedRuns;
        printf("\n%.5f ms", avg_time);
        if (individual_events)
            printf(" (%.5f ms including L2 flushes)", overall_time / args.timedRuns);

        if (args.perfMultiplier > 0.0f) {
            float perf = args.perfMultiplier / (avg_time / 1000.f);
            printf(" ==> %.4f", perf);
            if (args.perfSpeedOfLight > 0.0f)
                printf(" ==> %.3f%%", 100.0f * perf / args.perfSpeedOfLight);
        }
        printf("\n\n");

        // Cleanup events
        checkCudaErrors(cuEventDestroy(overall_start));
        checkCudaErrors(cuEventDestroy(overall_stop));
        for (auto& e : start_events) checkCudaErrors(cuEventDestroy(e));
        for (auto& e : stop_events) checkCudaErrors(cuEventDestroy(e));
    }

    checkCudaErrors(cuCtxSynchronize());

    // --- Array dump ---
    if (!args.dump_c_array.empty()) {
        checkCudaErrors(cuMemcpyDtoH(h_C.data(), d_C, sizeC));
        if (args.dump_c_format == "raw") {
            std::ofstream f(args.dump_c_array, std::ios::binary);
            f.write(reinterpret_cast<char*>(h_C.data()), sizeC);
        } else {
            std::ofstream f(args.dump_c_array);
            for (size_t i = 0; i < args.arrayDwordsC; i++) {
                if (h_C[i] != 0) {
                    if (args.dump_c_format == "int_csv")
                        f << h_C[i];
                    else if (args.dump_c_format == "float_csv")
                        f << std::fixed << std::setprecision(2)
                          << *reinterpret_cast<float*>(&h_C[i]);
                }
                if (i < args.arrayDwordsC - 1) f << ",";
            }
        }
    }

    // --- Reference comparison ---
    if (!args.reference_c_filename.empty()) {
        auto ref_data = readFile(args.reference_c_filename);
        if (ref_data.size() - 1 != sizeC) {
            fprintf(stderr, "Reference file wrong size\n");
            exit(EXIT_FAILURE);
        }
        auto* ref_C = reinterpret_cast<uint32_t*>(ref_data.data());
        checkCudaErrors(cuMemcpyDtoH(h_C.data(), d_C, sizeC));

        for (size_t i = 0; i < args.arrayDwordsC; i++) {
            if (args.compare_tolerance > 0.0f) {
                float a = *reinterpret_cast<float*>(&h_C[i]);
                float b = *reinterpret_cast<float*>(&ref_C[i]);
                float diff = std::abs(a - b);
                if (diff > args.compare_tolerance) {
                    printf("First difference at %zu: %.8f vs %.8f (diff %.8f)\n", i, a, b, diff);
                    break;
                }
            } else if (h_C[i] != ref_C[i]) {
                printf("First difference at %zu: %u vs %u (hex: %08x vs %08x, fp32: %.4f vs %.4f)\n",
                       i, h_C[i], ref_C[i], h_C[i], ref_C[i],
                       *reinterpret_cast<float*>(&h_C[i]), *reinterpret_cast<float*>(&ref_C[i]));
                break;
            }
        }
    }

    // --- Cleanup ---
    checkCudaErrors(cuMemFree(d_A));
    checkCudaErrors(cuMemFree(d_B));
    checkCudaErrors(cuMemFree(d_C));
    if (d_flush) { checkCudaErrors(cuMemFree(d_flush)); d_flush = 0; }
    checkCudaErrors(cuModuleUnload(module));
    return 0;
}
```

### 2. utils/cuda_helper.h

Replace the entire 322-line file with just the error-checking macro:

```cpp
#pragma once

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <cuda.h>
#include <nvrtc.h>

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

template <typename T>
inline void __checkCudaErrors(T err, const char *file, const int line) {
    if (err != 0) {
        const char *errorStr = "";
        if constexpr (std::is_same<T, CUresult>::value)
            cuGetErrorString(err, &errorStr);
        else if constexpr (std::is_same<T, nvrtcResult>::value)
            errorStr = nvrtcGetErrorString(err);
        fprintf(stderr, "CUDA error = %04d \"%s\" at %s:%d\n",
                (int)err, errorStr, file, line);
        exit(EXIT_FAILURE);
    }
}
#endif
```

Note: Remove `cudaError_t` handling and `cuda_runtime.h` include since we only use the Driver API now.

### 3. Makefile

Remove unnecessary library links. Final libraries section:
```makefile
LIBRARIES += -L$(CUDALIB) -lcuda
LIBRARIES += -lnvrtc

# ADD OpenMP flags
ALL_CCFLAGS += -Xcompiler -fopenmp
ALL_LDFLAGS += -Xcompiler -fopenmp
```

Remove `cuda_runtime.h` dependency (we can use `nvcc` as compiler driver but only link Driver API + NVRTC).

Clean target updated:
```makefile
clean:
	rm -f QuickRunCUDA QuickRunCUDA.o
```

### 4. Files to Delete

```
utils/ipc_helper.h
utils/nvmlClass.h
utils/cuda_controller.py
utils/git_and_run.py
_run.sh
```

### 5. .gitignore

```
*.i
*.ii
*.gpu
*.fatbin
QuickRunCUDA
out.txt
directory_watcher.log
PTX/
CUBIN/
```

Remove `*.ptx` and `*.cubin` (now saved in tracked subdirectories if desired).

---

## CLI Arguments (Final State)

### Kept:
```
-t, --threadsPerBlock       blockDim.x (default: 32)
-b, --blocksPerGrid         gridDim.x (default: 1)
-p, --persistentBlocks      gridDim.x = SM count
-s, --sharedMemoryBlockBytes
-o, --sharedMemoryCarveoutBytes
--l2flush                   0=none, 1=start, 2=every run
-T, --timedRuns             Number of timed runs
-P, --perfMultiplier
-L, --perfSpeedOfLight
--timesPerRun               Print individual times
-A, --arrayDwordsA
-B, --arrayDwordsB
-C, --arrayDwordsC
-r, --randomA
--randomB
--randomMask
--randomSeed
-0, -1, -2                  Kernel integer args
-f, --kernel-filename
-H, --header
--dump-c, --dump-c-format
--load-c
--reference-c, --compare-tolerance
[positional]                Kernel filename
```

### Removed:
```
--server                    (server mode)
--clock-speed               (NVML clock control)
--reuse-cubin               (replaced by --cubin-input)
-i, --runInitKernel         (auto-detected now)
-N, --perfMultiplierPerThread
-U, --perfMultiplier-unit
```

### Added:
```
--ptx-input FILE            Load PTX directly (skip NVRTC)
--cubin-input FILE          Load CUBIN directly (skip all compilation)
```

---

## Implementation Order

1. **Delete files**: `ipc_helper.h`, `nvmlClass.h`, `cuda_controller.py`, `git_and_run.py`, `_run.sh`
2. **Rewrite `cuda_helper.h`**: Just the `checkCudaErrors` macro
3. **Rewrite `QuickRunCUDA.cpp`**:
   a. Remove deleted includes
   b. Remove server mode, clock control, `parseCommandString()`
   c. Remove `-i`, `-N`, `-U` from struct and parser
   d. Add helper functions: `readFile`, `ensureDir`, `makeTimestampedPath`, `saveFile`, `compileSourceToPTX`, `compilePTXtoCUBIN`, `launchKernel`
   e. Replace compilation section with three-path pipeline
   f. Add auto-detection of init kernel
   g. Simplify `main()`
   h. Modernize: `malloc`->`vector`, `NULL`->`nullptr`, `uint`->`uint32_t`
4. **Update Makefile**: Remove unnecessary libs, add OpenMP flags
5. **Update `.gitignore`**: `PTX/` and `CUBIN/` directories
6. **Test**: Compile and run with `default_kernel.cu`
