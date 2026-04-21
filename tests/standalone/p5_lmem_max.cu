// P5: Max LMEM size per thread
// LMEM (local memory) is per-thread "stack" allocated in DRAM
// Test allocation sizes from 1 KB to 16 MB per thread; find max
#include <cuda_runtime.h>
#include <cstdio>

template <int N_KB>
__global__ void touch_lmem(int* out, int u2) {
    int local[N_KB * 256];  // N_KB * 1024 bytes
    for (int i = 0; i < N_KB * 256; i++) local[i] = i + u2;
    int sum = 0;
    for (int i = 0; i < N_KB * 256; i++) sum += local[i];
    if (sum == 12345) out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

template <int N_KB>
int test() {
    int* dev_out;
    cudaMalloc(&dev_out, 4096);

    cudaFuncAttributes attrs;
    cudaError_t err = cudaFuncGetAttributes(&attrs, touch_lmem<N_KB>);
    if (err != cudaSuccess) {
        printf("N_KB=%d: cudaFuncGetAttributes failed: %s\n", N_KB, cudaGetErrorString(err));
        return -1;
    }

    touch_lmem<N_KB><<<1, 32>>>(dev_out, 7);
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("N_KB=%-4d (%6d B/thread): launch FAILED: %s [shared=%zu B]\n",
               N_KB, N_KB*1024, cudaGetErrorString(err), attrs.sharedSizeBytes);
    } else {
        printf("N_KB=%-4d (%6d B/thread): OK [local=%zu B/thread, shared=%zu B]\n",
               N_KB, N_KB*1024, attrs.localSizeBytes, attrs.sharedSizeBytes);
    }
    cudaFree(dev_out);
    return 0;
}

int main() {
    test<1>();    // 1 KB
    test<4>();    // 4 KB
    test<16>();   // 16 KB
    test<64>();   // 64 KB
    test<128>();  // 128 KB
    test<256>();  // 256 KB
    test<512>();  // 512 KB (NVIDIA documents 512 KB max)
    test<1024>();  // 1 MB
    test<2048>();  // 2 MB
    test<4096>();  // 4 MB
    test<8192>();  // 8 MB
    test<16384>(); // 16 MB
    test<65536>(); // 64 MB
    test<131072>(); // 128 MB
    test<524288>(); // 512 MB
    test<1048576>(); // 1 GB
    return 0;
}
