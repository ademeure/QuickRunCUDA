// MGFenceBench — minimal multi-GPU fence benchmark harness.
// Allocates 3 buffers (A, B, C), where A can optionally be on a REMOTE GPU.
// Compiles kernel via NVRTC (same contract as QuickRunCUDA: extern "C" __global__ void kernel(...)).
// Launches on PRIMARY GPU, which may write to REMOTE GPU's A via P2P.
//
// Usage:
//   ./MGFenceBench <kernel.cu> --primary 0 --remote 1 [--remote-a] -t THREADS -b BLOCKS [-H "#define ..."]
//   --remote-a   : put A on remote GPU (forces NVLink writes)
//   --b-remote   : put B on remote GPU
// Outputs: kernel time per iter + basic stats. Writes C[0..N] to stdout.

#include <cuda.h>
#include <nvrtc.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_CUDA(x) do { CUresult r = (x); if (r) { const char* s; cuGetErrorString(r, &s); fprintf(stderr,"CUDA err %d: %s at %s:%d\n", r, s, __FILE__,__LINE__); exit(1);} } while(0)
#define CHECK_NVRTC(x) do { nvrtcResult r = (x); if (r) { fprintf(stderr,"NVRTC err %d at %s:%d\n", r, __FILE__,__LINE__); exit(1);} } while(0)

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <kernel.cu> [--primary 0] [--remote 1] [--remote-a] [--b-remote]\n", argv[0]);
        fprintf(stderr, "       [-t threads] [-b blocks] [-A dwords] [-B dwords] [-C dwords] [-H header] [-T iters]\n");
        return 1;
    }
    const char* kernel_file = argv[1];
    int primary_gpu = 0, remote_gpu = 1;
    bool a_remote = false, b_remote = false;
    int threads = 32, blocks = 1;
    int arr_a = 1<<24, arr_b = 1<<20, arr_c = 1<<10;
    int iters = 1;
    std::string header;

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--primary") primary_gpu = atoi(argv[++i]);
        else if (a == "--remote") remote_gpu = atoi(argv[++i]);
        else if (a == "--remote-a") a_remote = true;
        else if (a == "--b-remote") b_remote = true;
        else if (a == "-t") threads = atoi(argv[++i]);
        else if (a == "-b") blocks = atoi(argv[++i]);
        else if (a == "-A") arr_a = atoi(argv[++i]);
        else if (a == "-B") arr_b = atoi(argv[++i]);
        else if (a == "-C") arr_c = atoi(argv[++i]);
        else if (a == "-T") iters = atoi(argv[++i]);
        else if (a == "-H") header = argv[++i];
    }

    CHECK_CUDA(cuInit(0));
    CUdevice dev_pri, dev_rem;
    CHECK_CUDA(cuDeviceGet(&dev_pri, primary_gpu));
    CHECK_CUDA(cuDeviceGet(&dev_rem, remote_gpu));
    CUcontext ctx_pri, ctx_rem;
    CHECK_CUDA(cuCtxCreate(&ctx_pri, NULL, 0, dev_pri));
    CHECK_CUDA(cuCtxCreate(&ctx_rem, NULL, 0, dev_rem));

    // Enable P2P both ways
    CHECK_CUDA(cuCtxSetCurrent(ctx_pri));
    int can_pri_rem;
    CHECK_CUDA(cuDeviceCanAccessPeer(&can_pri_rem, dev_pri, dev_rem));
    fprintf(stderr, "P2P primary->remote: %d\n", can_pri_rem);
    if (can_pri_rem) CHECK_CUDA(cuCtxEnablePeerAccess(ctx_rem, 0));
    CHECK_CUDA(cuCtxSetCurrent(ctx_rem));
    int can_rem_pri;
    CHECK_CUDA(cuDeviceCanAccessPeer(&can_rem_pri, dev_rem, dev_pri));
    fprintf(stderr, "P2P remote->primary: %d\n", can_rem_pri);
    if (can_rem_pri) CHECK_CUDA(cuCtxEnablePeerAccess(ctx_pri, 0));
    CHECK_CUDA(cuCtxSetCurrent(ctx_pri));

    // Allocate buffers
    size_t sizeA = (size_t)arr_a * 4, sizeB = (size_t)arr_b * 4, sizeC = (size_t)arr_c * 4;
    CUdeviceptr d_A, d_B, d_C;

    if (a_remote) {
        CHECK_CUDA(cuCtxSetCurrent(ctx_rem));
        CHECK_CUDA(cuMemAlloc(&d_A, sizeA));
        CHECK_CUDA(cuCtxSetCurrent(ctx_pri));
    } else {
        CHECK_CUDA(cuMemAlloc(&d_A, sizeA));
    }
    if (b_remote) {
        CHECK_CUDA(cuCtxSetCurrent(ctx_rem));
        CHECK_CUDA(cuMemAlloc(&d_B, sizeB));
        CHECK_CUDA(cuCtxSetCurrent(ctx_pri));
    } else {
        CHECK_CUDA(cuMemAlloc(&d_B, sizeB));
    }
    CHECK_CUDA(cuMemAlloc(&d_C, sizeC));  // C always on primary for result readback

    // Only memset local buffers (cuMemsetD32 may fail on P2P-mapped remote)
    if (!a_remote) CHECK_CUDA(cuMemsetD32(d_A, 0, sizeA/4));
    if (!b_remote) CHECK_CUDA(cuMemsetD32(d_B, 0, sizeB/4));
    CHECK_CUDA(cuMemsetD32(d_C, 0, sizeC/4));

    // Read kernel file
    std::ifstream f(kernel_file);
    if (!f.is_open()) { fprintf(stderr, "Can't open %s\n", kernel_file); return 1; }
    std::stringstream ss;
    ss << header << "\n" << f.rdbuf();
    std::string src = ss.str();

    // Compile via NVRTC
    nvrtcProgram prog;
    CHECK_NVRTC(nvrtcCreateProgram(&prog, src.c_str(), kernel_file, 0, nullptr, nullptr));
    // Detect arch
    int major, minor;
    CHECK_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev_pri));
    CHECK_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev_pri));
    char arch_opt[64];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=sm_%d%da", major, minor);
    const char* opts[] = { arch_opt, "--std=c++17", "--device-c" };
    nvrtcResult r = nvrtcCompileProgram(prog, 3, opts);
    if (r != NVRTC_SUCCESS) {
        size_t log_size; nvrtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, 0);
        nvrtcGetProgramLog(prog, &log[0]);
        fprintf(stderr, "NVRTC compile failed:\n%s\n", log.c_str());
        return 1;
    }
    size_t cubin_size;
    CHECK_NVRTC(nvrtcGetCUBINSize(prog, &cubin_size));
    std::vector<char> cubin(cubin_size);
    CHECK_NVRTC(nvrtcGetCUBIN(prog, cubin.data()));

    // Load module
    CUmodule mod;
    CHECK_CUDA(cuModuleLoadData(&mod, cubin.data()));
    CUfunction fn;
    CHECK_CUDA(cuModuleGetFunction(&fn, mod, "kernel"));

    // Launch
    void* args[] = { &d_A, &d_B, &d_C, (void*)&arr_a /*u0*/, (void*)&iters /*seed*/, (void*)&arr_b /*u2*/ };
    int u0 = 0, u1 = 42, u2 = 0;
    args[3] = &u0; args[4] = &u1; args[5] = &u2;

    CUevent e_start, e_stop;
    CHECK_CUDA(cuEventCreate(&e_start, 0));
    CHECK_CUDA(cuEventCreate(&e_stop, 0));
    CHECK_CUDA(cuEventRecord(e_start, 0));
    for (int i = 0; i < (iters > 0 ? iters : 1); i++) {
        CHECK_CUDA(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));
    }
    CHECK_CUDA(cuEventRecord(e_stop, 0));
    CHECK_CUDA(cuEventSynchronize(e_stop));
    float ms = 0;
    CHECK_CUDA(cuEventElapsedTime(&ms, e_start, e_stop));
    fprintf(stderr, "Wall: %.3f ms  (iters=%d)\n", ms, iters);

    // Read back C
    std::vector<unsigned> h_C(arr_c);
    CHECK_CUDA(cuMemcpyDtoH(h_C.data(), d_C, sizeC));
    // Dump raw
    fwrite(h_C.data(), 4, arr_c, stdout);

    return 0;
}
