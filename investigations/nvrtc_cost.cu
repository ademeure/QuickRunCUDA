// NVRTC compile cost as function of source size, optimization level, and arch
#include <nvrtc.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>
#include <string>

#define NVRTC_CHECK(x) do { auto r = (x); if (r != NVRTC_SUCCESS) { printf("NVRTC err: %s\n", nvrtcGetErrorString(r)); exit(1); } } while(0)

float compile_it(const std::string &src, const char *arch, const char *opt) {
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), "k.cu", 0, nullptr, nullptr));
    const char *opts[3] = {arch, opt, nullptr};
    int n_opts = opt[0] ? 2 : 1;

    auto t0 = std::chrono::high_resolution_clock::now();
    nvrtcResult r = nvrtcCompileProgram(prog, n_opts, opts);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (r != NVRTC_SUCCESS) {
        size_t logsz; nvrtcGetProgramLogSize(prog, &logsz);
        std::string log(logsz, 0); nvrtcGetProgramLog(prog, log.data());
        printf("FAIL: %s\n", log.c_str());
        return -1;
    }
    nvrtcDestroyProgram(&prog);
    return std::chrono::duration<float, std::milli>(t1-t0).count();
}

int main() {
    cuInit(0);

    // Tiny kernel
    std::string tiny = R"(
extern "C" __global__ void k(float *out) {
    out[threadIdx.x] = threadIdx.x * 1.5f;
}
)";

    // Medium kernel: 100 FMAs
    std::string medium = "extern \"C\" __global__ void k(float *out) { float a = threadIdx.x;\n";
    for (int i = 0; i < 100; i++) medium += "  a = a * 1.0001f + 0.0001f;\n";
    medium += "  out[threadIdx.x] = a;\n}\n";

    // Large: 5000 FMAs
    std::string large = "extern \"C\" __global__ void k(float *out) { float a = threadIdx.x;\n";
    for (int i = 0; i < 5000; i++) large += "  a = a * 1.0001f + 0.0001f;\n";
    large += "  out[threadIdx.x] = a;\n}\n";

    // Kernel with many functions
    std::string many_fn;
    for (int i = 0; i < 50; i++)
        many_fn += "__device__ float f" + std::to_string(i) + "(float x) { return x*1.0001f + " + std::to_string(i) + ".0f; }\n";
    many_fn += "extern \"C\" __global__ void k(float *out) { float a = threadIdx.x;\n";
    for (int i = 0; i < 50; i++) many_fn += "  a = f" + std::to_string(i) + "(a);\n";
    many_fn += "  out[threadIdx.x] = a;\n}\n";

    printf("# B300 NVRTC compile cost (sm_103a unless noted)\n");
    printf("# %-25s %-10s %-12s %s\n", "kernel", "size_B", "compile_ms", "notes");

    auto run = [&](const char *name, const std::string &s) {
        // Run 3x, take median (NVRTC has caches in some versions)
        float t1 = compile_it(s, "--gpu-architecture=sm_103a", "");
        float t2 = compile_it(s, "--gpu-architecture=sm_103a", "");
        float t3 = compile_it(s, "--gpu-architecture=sm_103a", "");
        float min_t = t1 < t2 ? (t1 < t3 ? t1 : t3) : (t2 < t3 ? t2 : t3);
        printf("  %-25s %-10zu %-12.2f O3 default\n", name, s.size(), min_t);
    };

    run("tiny", tiny);
    run("medium 100 FMA", medium);
    run("large 5000 FMA", large);
    run("50-fn many_fn", many_fn);

    // Optimization level scan
    printf("\n## Optimization level on medium kernel\n");
    for (const char *opt : {"-G", "--device-debug", "", ""}) {
        const char *o = opt[0] ? opt : "";
        float t = compile_it(medium, "--gpu-architecture=sm_103a", o);
        printf("  opt=%-20s %.2f ms\n", opt[0] ? opt : "(default)", t);
    }

    // Arch comparison
    printf("\n## Arch comparison on medium kernel\n");
    for (const char *arch : {"--gpu-architecture=sm_70", "--gpu-architecture=sm_80", "--gpu-architecture=sm_90a", "--gpu-architecture=sm_100a", "--gpu-architecture=sm_103a"}) {
        float t = compile_it(medium, arch, "");
        printf("  %-30s %.2f ms\n", arch, t);
    }

    // Cubin size
    printf("\n## Output sizes (cubin & ptx)\n");
    const char *names[3] = {"tiny", "medium", "large"};
    std::string srcs[3] = {tiny, medium, large};
    for (int i = 0; i < 3; i++) {
        nvrtcProgram prog;
        nvrtcCreateProgram(&prog, srcs[i].c_str(), "k.cu", 0, nullptr, nullptr);
        const char *opts[1] = {"--gpu-architecture=sm_103a"};
        nvrtcCompileProgram(prog, 1, opts);
        size_t cubsz; nvrtcGetCUBINSize(prog, &cubsz);
        size_t ptxsz; nvrtcGetPTXSize(prog, &ptxsz);
        printf("  %-10s cubin=%zu B, ptx=%zu B\n", names[i], cubsz, ptxsz);
        nvrtcDestroyProgram(&prog);
    }

    return 0;
}
