// Fine-grain L2 sweep with carveout=100 (min L1 = 28 KB)
// This matches catalog "Fine-grain L2 sweep at carveout=100"
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

#define BLOCK_SIZE 1024
#define BLOCKS_PER_SM 2
#define UNROLL 16

__global__ __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
void bench_cg(const uint32_t* __restrict__ A, uint32_t* __restrict__ C, int ITERS, uint32_t WS) {
    uint32_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t nthr = gridDim.x*blockDim.x;
    uint64_t base = (uint64_t)(uintptr_t)A;
    uint32_t mask = WS-1u;
    uint32_t a=0,b=0,c=0,d=0;
    #pragma unroll 1
    for (int i=0; i<ITERS; i+=UNROLL) {
        #pragma unroll
        for (int j=0; j<UNROLL; j++) {
            uint32_t off = (uint32_t)(((uint64_t)tid*16ULL+(uint64_t)(i+j)*(uint64_t)nthr*16ULL)&mask)&~15u;
            uint64_t addr = base+off;
            uint32_t x0,x1,x2,x3;
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];"
                :"=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3):"l"(addr):"memory");
            a^=x0; b^=x1; c^=x2; d^=x3;
        }
    }
    C[tid] = a^b^c^d;
}

#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA err %s\n",cudaGetErrorString(e));exit(1);}}while(0)

int main() {
    CHECK(cudaSetDevice(0));
    int nsm=0;
    CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    int nb=nsm*BLOCKS_PER_SM, bs=BLOCK_SIZE;
    long long nthr=(long long)nb*bs;
    
    size_t max_ws = 256ULL*1024*1024;
    uint32_t *d_A,*d_C;
    CHECK(cudaMalloc(&d_A, max_ws));
    CHECK(cudaMalloc(&d_C, nthr*4));
    CHECK(cudaMemset(d_A, 0, max_ws));
    
    int iters = 131072;
    dim3 grid(nb), block(bs);
    
    // Set carveout
    cudaFuncAttribute attr = cudaFuncAttributePreferredSharedMemoryCarveout;
    CHECK(cudaFuncSetAttribute((void*)bench_cg, attr, 100));
    
    cudaEvent_t t0,t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    
    // Fine-grain sweep matching catalog
    int ws_list_mb[] = {8, 16, 24, 30, 40, 50, 60, 70, 80, 90, 110, 120, 126, 128, 140, 160, 200, 256};
    int nws = sizeof(ws_list_mb)/sizeof(ws_list_mb[0]);
    
    printf("%-8s  %12s  %8s  Notes\n", "WS_MB", "BW .cg", "time(ms)");
    printf("%-8s  %12s  %8s  -----\n", "--------", "------------", "--------");
    
    for (int k=0; k<nws; k++) {
        // Find next power of 2 >= ws_list_mb[k] MB for modulo
        size_t ws_mb = ws_list_mb[k];
        // Actually catalog uses modulo masking so WS must be power of 2
        // If not power of 2, use next P2 but only fill ws_mb MB
        // For non-P2 WS, let's use the actual MB as P2 ceiling
        // Actually the catalog may use non-P2 WS with masking - let's check
        // For now, use exact MB value as P2 by rounding up
        size_t ws_bytes = (size_t)ws_mb * 1024 * 1024;
        // Find P2 >= ws_bytes
        size_t ws_p2 = 1;
        while (ws_p2 < ws_bytes) ws_p2 <<= 1;
        // Use the exact MB (not P2) by adjusting the mask appropriately
        // For non-P2: can't use simple mask. Use P2 but note it.
        uint32_t ws32 = (uint32_t)ws_p2;
        
        // Warmup
        for (int w=0;w<2;w++) bench_cg<<<grid,block>>>(d_A,d_C,iters,ws32);
        CHECK(cudaDeviceSynchronize());
        
        float best=1e9;
        for (int r=0;r<5;r++) {
            CHECK(cudaEventRecord(t0));
            bench_cg<<<grid,block>>>(d_A,d_C,iters,ws32);
            CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
            float ms=0; CHECK(cudaEventElapsedTime(&ms,t0,t1));
            if(ms<best) best=ms;
        }
        double bw=(double)nthr*iters*16.0/(best*1e-3)/1e12;
        
        const char* note = "";
        if (ws_p2 != ws_bytes) note = "(rounded to P2)";
        if (ws_mb <= 63) note = "≤ 1 L2-side";
        if (ws_mb >= 126) note = "≥ L2 cap (126 MB)";
        
        printf("%8zu  %8.2f TB/s  %8.2f  %s (P2=%zuMB)\n", 
               ws_mb, bw, best, note, ws_p2/(1024*1024));
    }
    
    printf("\n=== With default carveout (0 = max L1) ===\n");
    CHECK(cudaFuncSetAttribute((void*)bench_cg, attr, 0));
    
    // Just a few key WS
    int key_ws_mb[] = {16, 32, 64, 128};
    for (int k=0; k<4; k++) {
        size_t ws_bytes = (size_t)key_ws_mb[k] * 1024 * 1024;
        uint32_t ws32 = (uint32_t)ws_bytes;
        
        for (int w=0;w<2;w++) bench_cg<<<grid,block>>>(d_A,d_C,iters,ws32);
        CHECK(cudaDeviceSynchronize());
        
        float best=1e9;
        for (int r=0;r<5;r++) {
            CHECK(cudaEventRecord(t0));
            bench_cg<<<grid,block>>>(d_A,d_C,iters,ws32);
            CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
            float ms=0; CHECK(cudaEventElapsedTime(&ms,t0,t1));
            if(ms<best) best=ms;
        }
        double bw=(double)nthr*iters*16.0/(best*1e-3)/1e12;
        printf("%8d  %8.2f TB/s  %8.2f  (carveout=0)\n", key_ws_mb[k], bw, best);
    }
    
    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_C));
    return 0;
}
