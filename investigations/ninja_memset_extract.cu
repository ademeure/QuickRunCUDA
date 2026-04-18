// B4: try various tactics to identify cudaMemset's internal kernel
//
// Approaches:
//   1. cudaFuncGetAttributes on candidate function pointers
//   2. nvprof/ncu API to enumerate launched kernel names
//   3. Hook cuLaunchKernel
//   4. Set up CUPTI callback for kernel launches
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <dlfcn.h>

int main() {
    cudaSetDevice(0);
    cuInit(0);

    void *d_buf; cudaMalloc(&d_buf, 1024 * 1024 * 1024);

    // Approach 1: query CU function attributes near memset entry
    void* memset_addr = (void*)cudaMemset;
    printf("# cudaMemset host wrapper at: %p\n", memset_addr);

    Dl_info info;
    if (dladdr(memset_addr, &info)) {
        printf("# Library: %s\n", info.dli_fname ? info.dli_fname : "?");
        printf("# Symbol: %s\n", info.dli_sname ? info.dli_sname : "?");
    }

    // Approach 2: scan loaded modules for kernel-like symbols
    // Try cuModuleEnumerateFunctions on the runtime module
    CUmodule mod = nullptr;
    CUresult res = cuModuleLoad(&mod, "");  // attempt to get currently-bound module
    printf("# cuModuleLoad result: %d\n", res);

    // Approach 3: trigger memset and immediately query lastError details
    cudaMemset(d_buf, 0xAB, 1024);
    cudaDeviceSynchronize();

    // Approach 4: time + workload signature (size scaling)
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    printf("# cudaMemset timing vs size (workload signature):\n");
    for (size_t bytes : {1024UL, 64*1024UL, 1024*1024UL, 16*1024*1024UL,
                         256*1024*1024UL, 1024*1024*1024UL}) {
        for (int i = 0; i < 3; i++) cudaMemset(d_buf, 0xAB, bytes);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            cudaMemset(d_buf, 0xAB, bytes);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double bw = (double)bytes / (best/1000.0) / 1e9;
        printf("  size=%5ld MB  best=%.4f ms  %.1f GB/s\n",
               bytes / (1024*1024), best, bw);
    }

    // Approach 5: try ncu --print-summary to find memset kernel name
    // (this is what we'd run externally)
    printf("\n# Would need: /usr/local/cuda/bin/ncu --print-summary per-kernel ./this_binary\n");
    printf("# OR: dlsym for known internal kernel names like:\n");
    printf("#     'memset_block_kernel', '_ZN13memset_kernel...', 'cudaMemset_internal'\n");

    void *handle = dlopen("libcudart.so.13", RTLD_NOW | RTLD_NOLOAD);
    if (!handle) handle = dlopen("libcudart.so", RTLD_NOW | RTLD_NOLOAD);
    if (handle) {
        for (const char* name : {"memset_kernel", "_Z13memset_kerneli", "cuMemsetD8Async",
                                  "memset_block_kernel", "__internal_memset_kernel"}) {
            void *sym = dlsym(handle, name);
            printf("  dlsym('%s'): %p\n", name, sym);
        }
        dlclose(handle);
    } else {
        printf("  dlopen failed\n");
    }

    return 0;
}
