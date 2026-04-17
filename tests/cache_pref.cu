// Test legacy cudaFuncCachePreferShared / cudaFuncSetAttribute() variations
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void k_dummy() {
    extern __shared__ char buf[];
    if (threadIdx.x == 0) buf[0] = 0;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

    printf("# Legacy cudaFuncCachePreference test\n");
    printf("# Default per-SM shmem: %zu, opt-in: %zu\n\n",
           prop.sharedMemPerMultiprocessor, prop.sharedMemPerBlockOptin);

    // Set various cache preferences and check if max dyn shmem changes
    cudaFuncCache prefs[] = {
        cudaFuncCachePreferNone,
        cudaFuncCachePreferShared,
        cudaFuncCachePreferL1,
        cudaFuncCachePreferEqual
    };
    const char *pref_names[] = {"None", "Shared", "L1", "Equal"};

    for (int i = 0; i < 4; i++) {
        cudaFuncSetCacheConfig((void*)k_dummy, prefs[i]);
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, (void*)k_dummy);
        // Try setting opt-in to maximum + 1
        for (int sz : {232448, 232449, 250000, 1024*1024}) {
            cudaError_t r = cudaFuncSetAttribute((void*)k_dummy,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize, sz);
            printf("  cachePref=%-8s, MaxDynShmem=%-7d : %s\n",
                   pref_names[i], sz, r == cudaSuccess ? "OK" : cudaGetErrorString(r));
        }
        printf("\n");
    }

    // Try cudaFuncSetSharedMemConfig (old bank size config)
    printf("# cudaFuncSetSharedMemConfig (bank size):\n");
    cudaSharedMemConfig sm_configs[] = {
        cudaSharedMemBankSizeDefault,
        cudaSharedMemBankSizeFourByte,
        cudaSharedMemBankSizeEightByte
    };
    const char *sm_names[] = {"Default", "4-byte banks", "8-byte banks"};
    for (int i = 0; i < 3; i++) {
        cudaError_t r = cudaFuncSetSharedMemConfig((void*)k_dummy, sm_configs[i]);
        printf("  %s: %s\n", sm_names[i], r == cudaSuccess ? "OK" : cudaGetErrorString(r));
    }

    return 0;
}
