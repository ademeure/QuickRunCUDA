// Per-SM: large permutation chase, .cg bypasses L1, see if 8/140 holds
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
__global__ void chase_init(uint32_t *p, uint64_t N) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    p[tid] = (tid * 65537ULL + 7919ULL) % N;
}
__global__ void per_sm_dram_chase(uint32_t *p, uint64_t *out, uint64_t N, int max_sms) {
    if (threadIdx.x != 0) return;
    int sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    uint32_t cur = sm_id * 4097 % N;
    long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 200; i++) {
        uint32_t v;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p + cur));
        cur = v;
    }
    long long t1 = clock64();
    if (sm_id < max_sms) {
        atomicCAS((unsigned long long*)&out[sm_id*2], 0ULL, (unsigned long long)(t1 - t0));
        out[sm_id*2 + 1] = cur;
    }
}
int main() {
    cudaSetDevice(0);
    uint64_t N = 256ULL * 1024 * 1024;
    uint32_t *d_p; cudaMalloc(&d_p, N * 4);
    chase_init<<<(N+255)/256, 256>>>(d_p, N);
    cudaDeviceSynchronize();
    int max_sms = 200;
    uint64_t *d_out; cudaMalloc(&d_out, max_sms * 16);
    cudaMemset(d_out, 0, max_sms * 16);
    per_sm_dram_chase<<<2000, 32>>>(d_p, d_out, N, max_sms);
    cudaDeviceSynchronize();
    uint64_t h[200 * 2];
    cudaMemcpy(h, d_out, 200 * 16, cudaMemcpyDeviceToHost);
    int n = 0; long long mn = 1LL<<60, mx = 0, total = 0;
    long lat[200];
    for (int i = 0; i < max_sms; i++) {
        if (h[i*2] > 0) { lat[n++] = h[i*2]; if ((long long)h[i*2]<mn) mn=h[i*2]; if ((long long)h[i*2]>mx) mx=h[i*2]; total += h[i*2]; }
    }
    if (!n) { printf("No SMs\n"); return 1; }
    long mid = (mn + mx) / 2;
    int near = 0, far = 0; long sn = 0, sf = 0;
    for (int i = 0; i < n; i++) { if (lat[i] < mid) { near++; sn += lat[i]; } else { far++; sf += lat[i]; } }
    printf("DRAM-chase per SM:  Min=%.1f Mean=%.1f Max=%.1f cy/load  Range=%.2fx  Near=%d (%.1f cy=%.1f ns)  Far=%d (%.1f cy=%.1f ns)\n",
        (double)mn/200, (double)total/n/200, (double)mx/200, (double)mx/mn,
        near, near?(double)sn/near/200:0, near?(double)sn/near/200/2.032:0,
        far,  far?(double)sf/far/200:0,   far?(double)sf/far/200/2.032:0);
    return 0;
}
