// J1 RIGOR: pageable memory CPU↔GPU coherence bug repro
// THEORETICAL: pageable memory accessed by GPU on B300 with PageableMemoryAccess=1
// migrates pages on first touch. After migration, CPU writes may not be visible to GPU.
//
// Test: malloc → CPU init A → GPU read X1 → CPU write A → GPU read X2.
// Bug if X1 != X2 (CPU's second write not visible to GPU).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

__global__ void read_one(unsigned *p, unsigned *out, int idx) {
    out[0] = p[idx];
}

int main() {
    cudaSetDevice(0);
    int dev_attr;
    cudaDeviceGetAttribute(&dev_attr, cudaDevAttrPageableMemoryAccess, 0);
    printf("# PageableMemoryAccess support: %d\n", dev_attr);

    unsigned *out_d;
    cudaMalloc(&out_d, 4);

    // 64 KB pageable buffer (multiple OS pages)
    int N = 16384;
    unsigned *h_buf = (unsigned*)malloc(N * sizeof(unsigned));
    for (int i = 0; i < N; i++) h_buf[i] = 0xAAAA0000 + i;

    int idx = 100;
    printf("# Initial CPU write: h_buf[%d] = 0x%08x\n", idx, h_buf[idx]);

    // GPU read 1
    read_one<<<1, 1>>>(h_buf, out_d, idx);
    cudaDeviceSynchronize();
    unsigned x1; cudaMemcpy(&x1, out_d, 4, cudaMemcpyDeviceToHost);
    printf("# GPU read 1 (after first GPU touch): 0x%08x\n", x1);

    // CPU writes new value WITHOUT any explicit sync
    h_buf[idx] = 0xBBBB1111;
    printf("# CPU re-write: h_buf[%d] = 0x%08x\n", idx, h_buf[idx]);

    // GPU read 2 — does it see the new value?
    read_one<<<1, 1>>>(h_buf, out_d, idx);
    cudaDeviceSynchronize();
    unsigned x2; cudaMemcpy(&x2, out_d, 4, cudaMemcpyDeviceToHost);
    printf("# GPU read 2 (after CPU re-write): 0x%08x\n", x2);

    if (x2 == 0xBBBB1111) {
        printf("# COHERENT: GPU saw the CPU re-write\n");
    } else {
        printf("# STALE: GPU did NOT see CPU re-write — coherence bug!\n");
    }

    // What does CPU see now?
    printf("# CPU read after both: h_buf[%d] = 0x%08x\n", idx, h_buf[idx]);

    free(h_buf);
    cudaFree(out_d);
    return 0;
}
