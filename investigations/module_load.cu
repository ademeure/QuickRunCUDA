// cuModuleLoad / cuModuleLoadData: cubin load cost vs PTX JIT cost
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>

int main() {
    cudaSetDevice(0);  // forces ctx
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);

    auto compile_to_cubin = [](const std::string &src, std::vector<char> &cubin) {
        nvrtcProgram prog;
        nvrtcCreateProgram(&prog, src.c_str(), "k.cu", 0, nullptr, nullptr);
        const char *opts[1] = {"--gpu-architecture=sm_103a"};
        nvrtcCompileProgram(prog, 1, opts);
        size_t sz; nvrtcGetCUBINSize(prog, &sz);
        cubin.resize(sz);
        nvrtcGetCUBIN(prog, cubin.data());
        nvrtcDestroyProgram(&prog);
    };

    auto compile_to_ptx = [](const std::string &src, std::string &ptx) {
        nvrtcProgram prog;
        nvrtcCreateProgram(&prog, src.c_str(), "k.cu", 0, nullptr, nullptr);
        const char *opts[1] = {"--gpu-architecture=compute_103"};
        nvrtcCompileProgram(prog, 1, opts);
        size_t sz; nvrtcGetPTXSize(prog, &sz);
        ptx.resize(sz);
        nvrtcGetPTX(prog, ptx.data());
        nvrtcDestroyProgram(&prog);
    };

    auto bench = [&](auto fn, int trials = 50) {
        for (int i = 0; i < 3; i++) fn();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("# B300 cuModule load times: cubin vs PTX JIT\n\n");
    printf("# %-25s %-12s %-15s %-15s %-15s\n",
           "kernel", "cubin_B", "ptx_B", "load_cubin_us", "load_ptx_us");

    // Test sizes
    std::string srcs[] = {
        // Tiny
        R"(extern "C" __global__ void k(float *o) { o[threadIdx.x] = 1.0f; })",
        // Medium - 100 FMAs
        []{ std::string s = "extern \"C\" __global__ void k(float *o) { float a = threadIdx.x;\n";
            for (int i = 0; i < 100; i++) s += "  a = a*1.0001f+0.0001f;\n";
            s += "o[threadIdx.x]=a; }"; return s; }(),
        // Large - 2000 FMAs
        []{ std::string s = "extern \"C\" __global__ void k(float *o) { float a = threadIdx.x;\n";
            for (int i = 0; i < 2000; i++) s += "  a = a*1.0001f+0.0001f;\n";
            s += "o[threadIdx.x]=a; }"; return s; }(),
    };
    const char *names[] = {"tiny", "100-FMA", "2000-FMA"};

    for (int i = 0; i < 3; i++) {
        std::vector<char> cubin;
        std::string ptx;
        compile_to_cubin(srcs[i], cubin);
        compile_to_ptx(srcs[i], ptx);

        float t_cubin = bench([&]{
            CUmodule m; cuModuleLoadData(&m, cubin.data()); cuModuleUnload(m);
        });
        float t_ptx = bench([&]{
            CUmodule m; cuModuleLoadData(&m, ptx.c_str()); cuModuleUnload(m);
        });

        printf("  %-25s %-12zu %-15zu %-15.1f %-15.1f\n",
               names[i], cubin.size(), ptx.size(), t_cubin, t_ptx);
    }

    // cuModuleGetFunction cost (after load)
    printf("\n## cuModuleGetFunction cost (post-load)\n");
    {
        std::vector<char> cubin;
        compile_to_cubin(srcs[1], cubin);
        CUmodule m; cuModuleLoadData(&m, cubin.data());

        float t = bench([&]{
            CUfunction f; cuModuleGetFunction(&f, m, "k");
        }, 1000);
        printf("  Lookup time: %.3f us\n", t);
        cuModuleUnload(m);
    }

    return 0;
}
