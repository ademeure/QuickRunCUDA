extern "C" __global__ void init(float* A, float* B, float* C, int u0, int u1, int ws_bytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n_words = ws_bytes / 4;
    unsigned* p = (unsigned*)A;
    for (int i = idx; i < n_words; i += stride) p[i] = 0u;
}
extern "C" __global__ __launch_bounds__(512, 1)
void kernel(float* A, float* B, float* C, int iters, int u1, int ws_bytes) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned wid = tid / 32;
    unsigned lid = tid & 31;
    unsigned total_warps = (gridDim.x * blockDim.x) / 32;
    unsigned per_warp_bytes = (ws_bytes / total_warps) & ~31u;
    unsigned long long warp_base = (unsigned long long)wid * per_warp_bytes;
    unsigned mask = per_warp_bytes - 1;
    unsigned long long v0 = tid, v1 = tid+1, v2 = tid+2, v3 = tid+3;

    #pragma unroll 1
    for (int i = 0; i < iters; i += 32) {
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            // Each warp-cycle writes 32 threads × 32B = 1024B = one cache line
            unsigned off = (lid * 32 + (i + j) * 1024) & mask;
            unsigned long long addr = (unsigned long long)A + warp_base + off;
            asm volatile("st.global.cg.v4.b64 [%0], {%1,%2,%3,%4};"
                : : "l"(addr), "l"(v0),"l"(v1),"l"(v2),"l"(v3));
        }
    }
}
