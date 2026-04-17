// Isolated test: ONLY 'steal reserved' + PDL signal — measure exact corruption
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void k_steal_pdl(int *check, unsigned int *stolen_after) {
    extern __shared__ char buf[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Steal reserved space (all 256 words)
    unsigned int magic = 0x12340000 | bid;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(offset), "r"(magic + i) : "memory");
    }
    __syncthreads();

    // Read back BEFORE any PDL op — verify steal worked
    int corrupt_before = 0;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
        if (val != magic + i) corrupt_before++;
    }
    __syncthreads();

    // PDL signal
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    __syncthreads();

    // Read back AFTER PDL signal
    int corrupt_after = 0;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
        if (val != magic + i) corrupt_after++;
        if (tid == 0 && val != magic + i)
            stolen_after[i] = val;  // record the corrupted value
        else if (tid == 0)
            stolen_after[i] = 0;  // OK marker
    }

    if (tid == 0) {
        check[bid * 2] = corrupt_before;
        check[bid * 2 + 1] = corrupt_after;
    }
}

int main() {
    cudaSetDevice(0);

    int *d_check;
    unsigned int *d_stolen;
    cudaMalloc(&d_check, 16);
    cudaMalloc(&d_stolen, 256 * 4);
    cudaMemset(d_check, 0, 16);
    cudaMemset(d_stolen, 0, 256 * 4);

    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {dim3(1), dim3(32), 4096, 0, &attr, 1};
    cudaFuncSetAttribute((void*)k_steal_pdl,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 4096);

    void *args[] = {&d_check, &d_stolen};
    cudaError_t r = cudaLaunchKernelExC(&cfg, (void*)k_steal_pdl, args);
    cudaDeviceSynchronize();
    cudaError_t r2 = cudaGetLastError();
    printf("Launch: %s, sync: %s\n",
           r == cudaSuccess ? "OK" : cudaGetErrorString(r),
           r2 == cudaSuccess ? "OK" : cudaGetErrorString(r2));

    int check[16];
    cudaMemcpy(check, d_check, 16, cudaMemcpyDeviceToHost);
    unsigned int stolen[256];
    cudaMemcpy(stolen, d_stolen, 256 * 4, cudaMemcpyDeviceToHost);

    printf("Block 0: corruption before PDL=%d, after PDL=%d\n", check[0], check[1]);
    if (check[1] > 0) {
        printf("\nCorrupted words after PDL signal (expected magic=0x12340000+i):\n");
        for (int i = 0; i < 256; i++) {
            if (stolen[i] != 0) {
                printf("  [0x%03x] = 0x%08x (expected 0x%08x)\n", i*4, stolen[i], 0x12340000 + i);
            }
        }
    }

    cudaFree(d_check); cudaFree(d_stolen);
    return 0;
}
