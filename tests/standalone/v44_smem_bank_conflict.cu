// V44: SMEM bank conflict cost matrix (rigorous)
// Theoretical: SMEM has 32 banks × 4B; N-way conflict serializes N cycles
// Plus broadcast (all threads same address) is FREE (1 cycle).
//
// Method: 32 threads × different stride patterns × dep chain → measure cy/load

#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 1024  // 4 KB

// Each lane reads from address: base + (lane * STRIDE) % SMEM_W
// STRIDE=4 → all different banks (no conflict)
// STRIDE=8 → 2-way conflict (banks 0,2,4,...,30 → 16 unique banks, 2 lanes per bank)
// STRIDE=16 → 4-way
// STRIDE=32 → 8-way
// STRIDE=64 → 16-way
// STRIDE=128 → 32-way (one bank, all lanes — broadcast IF all addresses same; conflict IF different)
template<int STRIDE, int CL>
__global__ __launch_bounds__(32, 1)
void smem_lat(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    // Init: each slot stores a byte offset that points back to itself*stride
    for (int i = tid; i < SMEM_W; i += 32) {
        unsigned next = ((i * 37u + seed) % SMEM_W);
        smem[i] = next * 4u;
    }
    __syncthreads();

    unsigned base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    // Each lane has its own dep-chain starting at its assigned bank
    unsigned cur = (tid * STRIDE) & (SMEM_W * 4 - 4);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        // Load from base+cur, advance cur to next slot at same lane's bank
        // To preserve N-way conflict pattern across iterations, OR with bank mask
        unsigned next;
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(next) : "r"(base + cur) : "memory");
        // Force next address to keep the same conflict-pattern
        cur = ((next & (SMEM_W*4 - 4)) + (tid * STRIDE)) & (SMEM_W*4 - 4);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = cur;
    }
}

// Pure broadcast: all 32 lanes read SAME address
template<int CL>
__global__ __launch_bounds__(32, 1)
void smem_broadcast(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) {
        unsigned next = ((i * 37u + seed) % SMEM_W);
        smem[i] = next * 4u;
    }
    __syncthreads();

    unsigned base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur = 0;  // all lanes read same address

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        unsigned next;
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(next) : "r"(base + cur) : "memory");
        cur = next & (SMEM_W*4 - 4);  // shared across lanes
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = cur;
    }
}

template<typename K>
static int run_avg(K kernel, int N, double* cy_out) {
    unsigned long long* d_out;
    cudaMalloc(&d_out, 16);
    double sum = 0; int got = 0;
    for (int r = 0; r < N; r++) {
        kernel<<<1, 32>>>(d_out, 42u + r);
        cudaError_t e = cudaDeviceSynchronize();
        if (e) { cudaGetLastError(); continue; }
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        sum += (double)cy;
        got++;
    }
    cudaFree(d_out);
    if (got > 0) { *cy_out = sum / got; return got; }
    return 0;
}

int main() {
    CK(cudaSetDevice(0));
    double cy;

    printf("=== V44 SMEM bank conflict cost matrix (1 warp, dep chain) ===\n");
    printf("Theoretical: 1-way=1cy, 2-way=2cy, ..., 32-way=32cy serialization\n\n");
    printf("Pattern              CL=200 cy_per_load    expected_factor\n");

    int n = 0;
    n = run_avg(smem_lat<4, 200>, 30, &cy);
    if (n > 0) printf("STRIDE=4   (no conflict)   %.2f          1×\n", cy/200);
    double cy_baseline = cy / 200;

    n = run_avg(smem_lat<8, 200>, 30, &cy);
    if (n > 0) printf("STRIDE=8   (2-way)         %.2f          %.2f×\n", cy/200, cy/200/cy_baseline);

    n = run_avg(smem_lat<16, 200>, 30, &cy);
    if (n > 0) printf("STRIDE=16  (4-way)         %.2f          %.2f×\n", cy/200, cy/200/cy_baseline);

    n = run_avg(smem_lat<32, 200>, 30, &cy);
    if (n > 0) printf("STRIDE=32  (8-way)         %.2f          %.2f×\n", cy/200, cy/200/cy_baseline);

    n = run_avg(smem_lat<64, 200>, 30, &cy);
    if (n > 0) printf("STRIDE=64  (16-way)        %.2f          %.2f×\n", cy/200, cy/200/cy_baseline);

    n = run_avg(smem_lat<128, 200>, 30, &cy);
    if (n > 0) printf("STRIDE=128 (32-way same-bank diff-addr) %.2f   %.2f×\n", cy/200, cy/200/cy_baseline);

    n = run_avg(smem_broadcast<200>, 30, &cy);
    if (n > 0) printf("BROADCAST  (all same addr) %.2f          %.2f×\n", cy/200, cy/200/cy_baseline);

    return 0;
}
