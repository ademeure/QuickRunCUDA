// Reproduce catalog "Memory hierarchy knees" table conditions exactly:
// - bs=1024, mb=2 (296 blocks), ITERS=32768, UNROLL=16
// - v4.u32 .cg and .ca loads
// - modulo addressing (power-of-2 WS)

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 1024
#define BLOCKS_PER_SM 2
#define UNROLL 16

__global__ __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
void bench_cg_iters(const uint32_t* __restrict__ A, uint32_t* __restrict__ C,
                    int ITERS, uint32_t WS_BYTES) {
    uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nthr = gridDim.x * blockDim.x;
    uint32_t mask = WS_BYTES - 1u;
    uint64_t base = (uint64_t)(uintptr_t)A;
    uint32_t acc0=0, acc1=0, acc2=0, acc3=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            uint32_t off = (uint32_t)(((uint64_t)tid*16ULL + (uint64_t)(i+j)*(uint64_t)nthr*16ULL) & mask) & ~15u;
            uint64_t addr = base + off;
            uint32_t x0,x1,x2,x3;
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];"
                :"=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3):"l"(addr):"memory");
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
        }
    }
    C[tid] = acc0^acc1^acc2^acc3;
}

__global__ __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
void bench_ca_iters(const uint32_t* __restrict__ A, uint32_t* __restrict__ C,
                    int ITERS, uint32_t WS_BYTES) {
    uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nthr = gridDim.x * blockDim.x;
    uint32_t mask = WS_BYTES - 1u;
    uint64_t base = (uint64_t)(uintptr_t)A;
    uint32_t acc0=0, acc1=0, acc2=0, acc3=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            uint32_t off = (uint32_t)(((uint64_t)tid*16ULL + (uint64_t)(i+j)*(uint64_t)nthr*16ULL) & mask) & ~15u;
            uint64_t addr = base + off;
            uint32_t x0,x1,x2,x3;
            asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3},[%4];"
                :"=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3):"l"(addr):"memory");
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
        }
    }
    C[tid] = acc0^acc1^acc2^acc3;
}

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA err %s\n",cudaGetErrorString(e));exit(1);}} while(0)

int main(int argc, char** argv) {
    CHECK(cudaSetDevice(0));
    int nsm=0;
    CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    int nb = nsm * BLOCKS_PER_SM;
    int bs = BLOCK_SIZE;
    long long nthr = (long long)nb * bs;
    
    printf("GPU: %d SMs, %d blocks, %d threads/block = %lld total threads\n", nsm, nb, bs, nthr);

    // Test with catalog-exact ITERS and multiple ITERS values
    int iters_list[] = {32, 64, 256, 1024, 4096, 16384, 32768, 131072, 524288};
    int niters = sizeof(iters_list)/sizeof(iters_list[0]);
    
    size_t max_ws = 256ULL * 1024 * 1024;
    size_t c_sz = (size_t)nthr * 4;
    uint32_t *d_A, *d_C;
    CHECK(cudaMalloc(&d_A, max_ws));
    CHECK(cudaMalloc(&d_C, c_sz));
    CHECK(cudaMemset(d_A, 0, max_ws));
    
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    
    // Test at 32 MB WS (should be purely L2-resident)
    size_t ws = 32ULL * 1024 * 1024;
    uint32_t ws32 = (uint32_t)ws;
    dim3 grid(nb), block(bs);
    
    printf("\n=== 32 MB WS, varying ITERS ===\n");
    printf("%-12s  %12s  %12s  %10s\n", "ITERS", "BW .cg", "BW .ca", "Time.cg(ms)");
    printf("%-12s  %12s  %12s  %10s\n", "------------", "------------", "------------", "----------");
    
    for (int k = 0; k < niters; k++) {
        int iters = iters_list[k];
        
        // Warmup
        bench_cg_iters<<<grid,block>>>(d_A, d_C, iters, ws32);
        bench_cg_iters<<<grid,block>>>(d_A, d_C, iters, ws32);
        CHECK(cudaDeviceSynchronize());
        
        // CG timing
        float ms_cg = 0, ms_ca = 0;
        CHECK(cudaEventRecord(t0));
        for (int r=0; r<5; r++) bench_cg_iters<<<grid,block>>>(d_A, d_C, iters, ws32);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        CHECK(cudaEventElapsedTime(&ms_cg, t0, t1));
        ms_cg /= 5.0f;
        
        // CA timing
        CHECK(cudaEventRecord(t0));
        for (int r=0; r<5; r++) bench_ca_iters<<<grid,block>>>(d_A, d_C, iters, ws32);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        CHECK(cudaEventElapsedTime(&ms_ca, t0, t1));
        ms_ca /= 5.0f;
        
        double bytes = (double)nthr * iters * 16.0;
        double bw_cg = bytes / (ms_cg * 1e-3) / 1e12;
        double bw_ca = bytes / (ms_ca * 1e-3) / 1e12;
        printf("%-12d  %10.2f TB/s  %10.2f TB/s  %10.2f\n", iters, bw_cg, bw_ca, ms_cg);
    }
    
    // Also check WS sweep at catalog ITERS=32768
    printf("\n=== WS sweep at ITERS=32768 (catalog value) ===\n");
    printf("%-10s  %12s  %12s\n", "WS", "BW .cg", "BW .ca");
    printf("%-10s  %12s  %12s\n", "----------", "------------", "------------");
    
    int iters = 32768;
    int ws_list_mb[] = {1, 4, 16, 32, 64, 128, 256};
    int nws = sizeof(ws_list_mb)/sizeof(ws_list_mb[0]);
    
    for (int k = 0; k < nws; k++) {
        size_t wss = (size_t)ws_list_mb[k] * 1024 * 1024;
        uint32_t wss32 = (uint32_t)wss;
        
        // Warmup
        bench_cg_iters<<<grid,block>>>(d_A, d_C, iters, wss32);
        bench_ca_iters<<<grid,block>>>(d_A, d_C, iters, wss32);
        CHECK(cudaDeviceSynchronize());
        
        float ms_cg=0, ms_ca=0;
        CHECK(cudaEventRecord(t0));
        for (int r=0;r<10;r++) bench_cg_iters<<<grid,block>>>(d_A, d_C, iters, wss32);
        CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
        CHECK(cudaEventElapsedTime(&ms_cg, t0, t1)); ms_cg/=10.0f;
        
        CHECK(cudaEventRecord(t0));
        for (int r=0;r<10;r++) bench_ca_iters<<<grid,block>>>(d_A, d_C, iters, wss32);
        CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
        CHECK(cudaEventElapsedTime(&ms_ca, t0, t1)); ms_ca/=10.0f;
        
        double bytes = (double)nthr * iters * 16.0;
        double bw_cg = bytes / (ms_cg * 1e-3) / 1e12;
        double bw_ca = bytes / (ms_ca * 1e-3) / 1e12;
        printf("%3d MB      %10.2f TB/s  %10.2f TB/s\n", ws_list_mb[k], bw_cg, bw_ca);
    }
    
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    return 0;
}
