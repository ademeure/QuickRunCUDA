// V45: SMEM throughput-bound bank conflict measurement (Rule 9 verification of V44)
// V44 used dep chain — chain may have masked conflict cost.
// V45: independent loads, max occupancy, measure pure throughput.

#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 1024  // 4 KB

template<int STRIDE, int ILP, int N_ITERS>
__global__ __launch_bounds__(256, 2)
void smem_bw(unsigned* out) {
    __shared__ unsigned smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 256) smem[i] = i * 7 + 3;
    __syncthreads();

    unsigned base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    // Each thread reads ILP independent addresses, all conflict pattern STRIDE
    unsigned addr[16];
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        // Lane (tid%32) × STRIDE shifted by k for ILP slots
        unsigned a = ((tid * STRIDE) + (k * 4 * 256)) & (SMEM_W * 4 - 4);
        addr[k] = base + a;
    }

    unsigned acc[16] = {0};
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            unsigned v;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr[k]) : "memory");
            acc[k] += v;
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (tid == 0) {
        unsigned x = 0;
        for (int k = 0; k < ILP; k++) x ^= acc[k];
        out[blockIdx.x] = x + (unsigned)(t1 - t0);
    }
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    unsigned* d_out;
    CK(cudaMalloc(&d_out, 4 * prop.multiProcessorCount * 4));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_ITERS = 5000;
    const int BLOCKS = 1184;
    const int ILP = 8;

    printf("=== V45 SMEM throughput-bound conflict cost (Rule 9 verify of V44) ===\n");
    printf("256 thr/CTA × 2 CTAs/SM × 1184 blocks, ILP=%d, N_ITERS=%d\n\n", ILP, N_ITERS);
    printf("STRIDE  pattern        wall_ms   load_inst   Glanes/s   ratio\n");

    auto run = [&](const char* lbl, int op) {
        if (op == 0) smem_bw<4, ILP, N_ITERS><<<BLOCKS, 256>>>(d_out);
        if (op == 1) smem_bw<8, ILP, N_ITERS><<<BLOCKS, 256>>>(d_out);
        if (op == 2) smem_bw<16, ILP, N_ITERS><<<BLOCKS, 256>>>(d_out);
        if (op == 3) smem_bw<32, ILP, N_ITERS><<<BLOCKS, 256>>>(d_out);
        if (op == 4) smem_bw<64, ILP, N_ITERS><<<BLOCKS, 256>>>(d_out);
        if (op == 5) smem_bw<128, ILP, N_ITERS><<<BLOCKS, 256>>>(d_out);
        cudaError_t e = cudaDeviceSynchronize();
        if (e) { printf("%-20s FAIL\n", lbl); cudaGetLastError(); return 0.0; }

        int RUNS = 3;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            if (op == 0) smem_bw<4, ILP, N_ITERS><<<BLOCKS, 256>>>(d_out);
            if (op == 1) smem_bw<8, ILP, N_ITERS><<<BLOCKS, 256>>>(d_out);
            if (op == 2) smem_bw<16, ILP, N_ITERS><<<BLOCKS, 256>>>(d_out);
            if (op == 3) smem_bw<32, ILP, N_ITERS><<<BLOCKS, 256>>>(d_out);
            if (op == 4) smem_bw<64, ILP, N_ITERS><<<BLOCKS, 256>>>(d_out);
            if (op == 5) smem_bw<128, ILP, N_ITERS><<<BLOCKS, 256>>>(d_out);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        double total = (double)BLOCKS * 256 * ILP * N_ITERS;
        double glanes = total / (avg_ms / 1e3) / 1e9;
        return (double)glanes;
    };

    double base = 0;
    base = run("STRIDE=4   1-way",    0);
    printf("4       no-conflict     %.3f       %d × inst   %.1f      1.00× (BASELINE)\n",
           BLOCKS * 256 * ILP * N_ITERS / 1e9 / (base/1e9), ILP, base);

    double r;
    r = run("STRIDE=8   2-way",    1); printf("8       2-way         %.1f      %.2f×\n", r, base/r);
    r = run("STRIDE=16  4-way",    2); printf("16      4-way         %.1f      %.2f×\n", r, base/r);
    r = run("STRIDE=32  8-way",    3); printf("32      8-way         %.1f      %.2f×\n", r, base/r);
    r = run("STRIDE=64  16-way",   4); printf("64      16-way        %.1f      %.2f×\n", r, base/r);
    r = run("STRIDE=128 32-way",   5); printf("128     32-way        %.1f      %.2f×\n", r, base/r);

    cudaFree(d_out);
    return 0;
}
