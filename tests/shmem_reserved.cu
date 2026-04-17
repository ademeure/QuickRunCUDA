// DEEP investigation of the driver-reserved 1 KiB shared memory per block
// Goal: understand what it's for, and whether/how user code can reclaim it
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// Static shared memory declarations - test how much we can declare
#define SH_DECL(NAME, BYTES) extern "C" __global__ void NAME() { \
    __shared__ char buf[BYTES]; \
    buf[threadIdx.x % BYTES] = (char)threadIdx.x; \
    if (threadIdx.x == 0) printf("k_%d: ok\n", BYTES); \
}

SH_DECL(k_static_48k, 49152)            // exactly 48 KB
SH_DECL(k_static_48k_minus, 49151)      // 48 KB - 1
SH_DECL(k_static_47k, 48128)            // 48 KB - 1024 (47 KB)

// dynamic shmem version
extern "C" __global__ void k_dynamic(int dyn_size) {
    extern __shared__ char buf[];
    if (threadIdx.x == 0) buf[0] = 0;
}

// kernel with both static and dynamic
extern "C" __global__ void k_both_1024(int dyn_size) {
    __shared__ char st[1024];
    extern __shared__ char dyn[];
    if (threadIdx.x == 0) { st[0] = 0; dyn[0] = 0; }
}
extern "C" __global__ void k_both_4096(int dyn_size) {
    __shared__ char st[4096];
    extern __shared__ char dyn[];
    if (threadIdx.x == 0) { st[0] = 0; dyn[0] = 0; }
}
extern "C" __global__ void k_both_47k(int dyn_size) {
    __shared__ char st[48128];
    extern __shared__ char dyn[];
    if (threadIdx.x == 0) { st[0] = 0; dyn[0] = 0; }
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("# B300 driver-reserved shared memory analysis\n");
    printf("# Reported reservedSharedMemPerBlock: %zu bytes (%zu KB)\n",
           prop.reservedSharedMemPerBlock, prop.reservedSharedMemPerBlock/1024);
    printf("# Default shmem per block:           %zu bytes (%.1f KB)\n",
           prop.sharedMemPerBlock, prop.sharedMemPerBlock/1024.0);
    printf("# Opt-in shmem per block:            %zu bytes (%.1f KB)\n",
           prop.sharedMemPerBlockOptin, prop.sharedMemPerBlockOptin/1024.0);
    printf("# Per-SM total shmem:                %zu bytes (%.1f KB)\n\n",
           prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor/1024.0);

    // ===== Test 1: Function attributes for each kernel =====
    printf("## Function attributes (cudaFuncAttribute)\n");
    cudaFuncAttributes fa;
    void *kernels[] = {
        (void*)k_static_48k, (void*)k_static_48k_minus, (void*)k_static_47k,
        (void*)k_dynamic, (void*)k_both_1024, (void*)k_both_4096, (void*)k_both_47k
    };
    const char *names[] = {
        "k_static_48k (49152 bytes)",
        "k_static_48k_minus (49151)",
        "k_static_47k (48128)",
        "k_dynamic (no static)",
        "k_both_1024 (static)",
        "k_both_4096 (static)",
        "k_both_47k (static)"
    };

    for (int i = 0; i < 7; i++) {
        cudaFuncGetAttributes(&fa, kernels[i]);
        printf("  %-30s: shared=%zu, max_dyn=%d\n",
               names[i], fa.sharedSizeBytes, fa.maxDynamicSharedSizeBytes);
    }

    // ===== Test 2: Try to launch each kernel =====
    printf("\n## Launch test\n");
    for (int i = 0; i < 7; i++) {
        // Try launching
        cudaError_t r;
        // Reset error state
        cudaGetLastError();

        switch (i) {
            case 0: case 1: case 2:
                // Static-only kernels
                ((void(*)())kernels[i])();  // can't call __global__ from host
                break;
        }
        // Use cudaLaunchKernel
        if (i < 3) {
            cudaLaunchKernel(kernels[i], dim3(1), dim3(32), nullptr, 0, 0);
        } else {
            int dyn = 0;
            void *args[] = {&dyn};
            cudaLaunchKernel(kernels[i], dim3(1), dim3(32), args, 0, 0);
        }
        r = cudaGetLastError();
        printf("  %-30s: %s\n", names[i], r==cudaSuccess ? "OK" : cudaGetErrorString(r));
    }
    cudaDeviceSynchronize();

    // ===== Test 3: cudaFuncSetAttribute - opt-in shared memory =====
    printf("\n## Opt-in shared memory (cudaFuncAttributeMaxDynamicSharedMemorySize)\n");
    int dyn_arr[] = {0, 1024, 4096, 16384, 32768, 49152, 65536, 100000, 196608, 200000, 230000, 232448};
    for (int dyn : dyn_arr) {
        cudaError_t r = cudaFuncSetAttribute((void*)k_dynamic,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             dyn);
        if (r != cudaSuccess) {
            printf("  set %-7d : %s\n", dyn, cudaGetErrorString(r));
        } else {
            // Now try to launch with that much dyn shmem
            int args_dyn = dyn;
            void *args[] = {&args_dyn};
            cudaLaunchKernel((void*)k_dynamic, dim3(1), dim3(32), args, dyn, 0);
            cudaError_t lr = cudaGetLastError();
            printf("  set %-7d : OK (launch %s)\n", dyn,
                   lr==cudaSuccess ? "OK" : cudaGetErrorString(lr));
        }
    }

    // ===== Test 4: How does opt-in affect occupancy? =====
    printf("\n## Occupancy with opt-in (cudaOccupancyMaxActiveBlocksPerMultiprocessor)\n");
    cudaFuncSetAttribute((void*)k_dynamic, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)prop.sharedMemPerBlockOptin);
    int act;
    int dyn_test[] = {0, 1024, 16384, 32768, 49152, 65536, 100000, 200000, 232448};
    for (int dyn : dyn_test) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act, (void*)k_dynamic, 32, dyn);
        // Total shmem used = 1024 (reserved) + dyn
        printf("  dyn=%-7d (total %zu bytes/block): %d blocks/SM (= %d total threads/SM)\n",
               dyn, dyn + prop.reservedSharedMemPerBlock, act, act * 32);
    }

    return 0;
}
