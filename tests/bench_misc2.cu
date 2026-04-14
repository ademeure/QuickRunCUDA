// Miscellaneous probes: vector loads/stores, LDGSTS variants, stmatrix, f64 cvt.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int tid = threadIdx.x;
    unsigned int base_s = (unsigned)__cvta_generic_to_shared(&smem[tid * 8]);
    unsigned long long base_g = (unsigned long long)A + (blockIdx.x * blockDim.x + tid) * 32;
    if (tid < BLOCK_SIZE*8) smem[tid] = tid;
    __syncthreads();
    unsigned int v0,v1,v2,v3;
    float f0=1.0f,f1=2.0f;
    double d0 = 1.5;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // ld.global.v4.u32 (vector 128-bit)
            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3) : "l"(base_g));
#elif OP == 1  // ld.global.v2.u32
            asm volatile("ld.global.v2.u32 {%0,%1}, [%2];" : "=r"(v0),"=r"(v1) : "l"(base_g));
#elif OP == 2  // st.global.v4.u32
            v0=tid; v1=tid+1; v2=tid+2; v3=tid+3;
            asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(base_g), "r"(v0),"r"(v1),"r"(v2),"r"(v3));
#elif OP == 3  // ld.shared.v4.u32
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3) : "r"(base_s));
#elif OP == 4  // st.shared.v4.u32
            v0=tid; v1=tid+1; v2=tid+2; v3=tid+3;
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(base_s), "r"(v0),"r"(v1),"r"(v2),"r"(v3));
#elif OP == 5  // cvt.rn.f32.f64
            { float f;
              asm volatile("cvt.rn.f32.f64 %0, %1;" : "=f"(f) : "d"(d0)); f0 = f;
              d0 = (double)f + 0.001; }
#elif OP == 6  // cvt.f64.f32
            asm volatile("cvt.f64.f32 %0, %1;" : "=d"(d0) : "f"(f0)); f0 += 0.0001f;
#elif OP == 7  // cp.async.cg.shared.global (128-bit)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(base_s), "l"(base_g));
#elif OP == 8  // cp.async.ca.shared.global (4-byte with cache-at-all-levels)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(base_s), "l"(base_g));
#elif OP == 9  // stmatrix.sync.aligned.m8n8.x1
            asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};" :: "r"(base_s), "r"(v0));
#elif OP == 10  // stmatrix.sync.aligned.m8n8.x4
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(base_s), "r"(v0),"r"(v1),"r"(v2),"r"(v3));
#elif OP == 11  // multimem.ld_reduce (only on multi-GPU, should fail here)
            asm volatile("multimem.ld_reduce.acquire.sys.global.add.u32 %0, [%1];" : "=r"(v0) : "l"(base_g));
#elif OP == 12  // fence.sc.gpu
            asm volatile("fence.sc.gpu;");
#elif OP == 13  // fence.acq_rel.cta
            asm volatile("fence.acq_rel.cta;");
#elif OP == 14  // fence.acquire.cluster
            asm volatile("fence.acquire.cluster;");
#elif OP == 15  // f64 compare
            { bool p;
              asm volatile("{.reg .pred p; setp.lt.f64 p, %1, %2; selp.b32 %0, 1, 0, p;}" : "=r"(v0) : "d"(d0), "d"(d0+0.1));
              d0 = d0 + (double)v0 * 1e-9; }
#endif
        }
    }
    unsigned int acc = v0 ^ v1 ^ v2 ^ v3 ^ __float_as_int(f0) ^ (unsigned)d0;
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
