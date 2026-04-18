// J1 v2: Try harder — large buffer, GPU-touches all pages, then CPU rewrites
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void touch_all(unsigned *p, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        // Read each page's first word — forces migration of all pages
        if ((i & 1023) == 0) p[i] = p[i];  // RMW touch
    }
}

__global__ void check(unsigned *p, unsigned *out, int idx) {
    out[0] = p[idx];
}

int main() {
    cudaSetDevice(0);
    int N = 1024 * 1024;  // 4 MB
    unsigned *h = (unsigned*)malloc(N * sizeof(unsigned));
    for (int i = 0; i < N; i++) h[i] = i;

    unsigned *d_out; cudaMalloc(&d_out, 4);

    // Force migration of ALL pages by touching them from GPU
    touch_all<<<148, 256>>>(h, N);
    cudaDeviceSynchronize();

    int test_idx = 12345;
    printf("# After GPU first-touch: h[%d] should still be %d (originally written by CPU)\n", test_idx, test_idx);

    // CPU re-writes some pages
    h[test_idx] = 0xCAFE1234;
    h[12345 + 4096] = 0xDEAD5678;  // different page
    printf("# CPU re-wrote h[%d] = 0xCAFE1234\n", test_idx);

    // GPU reads
    check<<<1, 1>>>(h, d_out, test_idx);
    cudaDeviceSynchronize();
    unsigned x;
    cudaMemcpy(&x, d_out, 4, cudaMemcpyDeviceToHost);
    printf("# GPU sees h[%d] = 0x%08x\n", test_idx, x);

    if (x == 0xCAFE1234) printf("# COHERENT\n");
    else printf("# STALE — GPU got 0x%08x instead of 0xCAFE1234\n", x);

    // Try MANY rewrites in a loop, see if any get lost
    int errors = 0;
    for (int i = 0; i < 100; i++) {
        unsigned val = 0xF0000000 | i;
        h[test_idx] = val;
        check<<<1, 1>>>(h, d_out, test_idx);
        cudaDeviceSynchronize();
        cudaMemcpy(&x, d_out, 4, cudaMemcpyDeviceToHost);
        if (x != val) {
            printf("# Iter %d: wrote 0x%08x, GPU sees 0x%08x\n", i, val, x);
            errors++;
        }
    }
    printf("# Coherence test: %d errors out of 100\n", errors);

    free(h);
    cudaFree(d_out);
    return 0;
}
