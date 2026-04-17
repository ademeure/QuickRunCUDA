// NVRTC compile options exploration - what affects compile time and output?
#include <nvrtc.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <string>

int main() {
    cuInit(0);

    // Medium kernel
    std::string src;
    src = "extern \"C\" __global__ void k(float *out) { float a = threadIdx.x;\n";
    for (int i = 0; i < 200; i++) src += "  a = a*1.0001f + 0.0001f;\n";
    src += "  out[threadIdx.x] = a; }";

    auto compile = [&](const std::vector<const char*> &opts) {
        nvrtcProgram prog;
        nvrtcCreateProgram(&prog, src.c_str(), "k.cu", 0, nullptr, nullptr);

        auto t0 = std::chrono::high_resolution_clock::now();
        nvrtcResult r = nvrtcCompileProgram(prog, opts.size(), opts.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1-t0).count();

        size_t cubin_sz = 0;
        if (r == NVRTC_SUCCESS) {
            nvrtcGetCUBINSize(prog, &cubin_sz);
        } else {
            size_t logsz; nvrtcGetProgramLogSize(prog, &logsz);
            std::string log(logsz, 0); nvrtcGetProgramLog(prog, log.data());
            printf("    LOG: %s\n", log.c_str());
        }
        nvrtcDestroyProgram(&prog);
        return std::pair<float, size_t>{us, cubin_sz};
    };

    printf("# B300 NVRTC option exploration (medium 200-FMA kernel)\n");
    printf("# %-50s %-12s %-12s\n", "options", "time_us", "cubin_B");

    auto run = [&](const std::vector<const char*> opts, const char *label) {
        // Best of 3
        float best = 1e30; size_t sz = 0;
        for (int i = 0; i < 3; i++) {
            auto [t, s] = compile(opts);
            if (t < best) { best = t; sz = s; }
        }
        printf("  %-50s %-12.0f %-12zu\n", label, best, sz);
    };

    // Baseline
    run({"--gpu-architecture=sm_103a"}, "sm_103a default");
    run({"--gpu-architecture=sm_103a", "-O0"}, "+ -O0");
    run({"--gpu-architecture=sm_103a", "-O1"}, "+ -O1");
    run({"--gpu-architecture=sm_103a", "-O2"}, "+ -O2");
    run({"--gpu-architecture=sm_103a", "-O3"}, "+ -O3");
    run({"--gpu-architecture=sm_103a", "--maxrregcount=32"}, "+ maxrregcount=32");
    run({"--gpu-architecture=sm_103a", "--maxrregcount=64"}, "+ maxrregcount=64");
    run({"--gpu-architecture=sm_103a", "-G"}, "+ debug -G");
    run({"--gpu-architecture=sm_103a", "-lineinfo"}, "+ lineinfo");
    run({"--gpu-architecture=sm_103a", "--use_fast_math"}, "+ fast math");
    run({"--gpu-architecture=sm_103a", "--ptxas-options=-v"}, "+ ptxas verbose");
    run({"--gpu-architecture=sm_103a", "--device-as-default-execution-space"}, "+ device default");
    run({"--gpu-architecture=sm_103a", "--restrict"}, "+ restrict");
    run({"--gpu-architecture=sm_103a", "--default-device"}, "+ default-device");
    run({"--gpu-architecture=sm_103a", "--no-source-include"}, "+ no-source-include");
    run({"--gpu-architecture=sm_103a", "--device-debug"}, "+ device-debug");

    // Output type
    run({"--gpu-architecture=compute_103"}, "compute_103 (PTX only)");

    return 0;
}
