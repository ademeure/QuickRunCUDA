// Read RAW bytes at the driver-reserved shmem offsets 0..1023
// This is "forbidden" but lets us see what's there
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void dump_reserved(unsigned int *out_words) {
    if (threadIdx.x != 0) return;
    // Use raw PTX to read from shmem offset 0..1020 (256 words)
    for (int i = 0; i < 256; i++) {
        unsigned int v;
        unsigned int offset = i * 4;
        // Use ld.shared with explicit offset 0
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(offset));
        out_words[blockIdx.x * 256 + i] = v;
    }
}

extern "C" __global__ void dump_reserved_with_cluster() {
    if (threadIdx.x != 0) return;
    extern __shared__ unsigned int dummy[];
    if (blockIdx.x == 0) {
        printf("# Cluster kernel reserved bytes:\n");
        for (int i = 0; i < 32; i++) {  // first 128 bytes
            unsigned int v;
            unsigned int offset = i * 4;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(offset));
            printf("  [0x%03x] = 0x%08x\n", offset, v);
        }
    }
}

extern "C" __global__ void __cluster_dims__(2,1,1) dump_reserved_cluster() {
    if (threadIdx.x != 0) return;
    if (blockIdx.x == 0) {
        printf("# Cluster=2 kernel reserved bytes:\n");
        for (int i = 0; i < 32; i++) {
            unsigned int v;
            unsigned int offset = i * 4;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(offset));
            printf("  [0x%03x] = 0x%08x\n", offset, v);
        }
    }
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

    unsigned int *d_out;
    cudaMalloc(&d_out, 4 * 256 * 4);  // 4 blocks * 256 words

    printf("# Dumping driver-reserved 1 KiB shmem (4 blocks of simple kernel)\n");
    dump_reserved<<<4, 32>>>(d_out);
    cudaDeviceSynchronize();

    unsigned int *h_out = new unsigned int[4 * 256];
    cudaMemcpy(h_out, d_out, 4 * 256 * 4, cudaMemcpyDeviceToHost);

    for (int b = 0; b < 4; b++) {
        printf("\nBlock %d reserved bytes (first 64 words):\n", b);
        for (int i = 0; i < 64; i++) {
            printf("[0x%03x]=%08x ", i*4, h_out[b * 256 + i]);
            if (i % 4 == 3) printf("\n");
        }
    }

    // Check non-zero pattern
    printf("\nNon-zero word counts per block:\n");
    for (int b = 0; b < 4; b++) {
        int nz = 0;
        for (int i = 0; i < 256; i++) if (h_out[b * 256 + i] != 0) nz++;
        printf("  Block %d: %d/256 non-zero words\n", b, nz);
    }

    // Now test with cluster kernel
    printf("\n## Testing kernel with __cluster_dims__\n");
    dump_reserved_cluster<<<4, 32>>>();
    cudaDeviceSynchronize();

    delete[] h_out;
    cudaFree(d_out);
    return 0;
}
