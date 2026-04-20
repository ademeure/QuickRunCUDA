// When does compiler emit STG.NA (non-temporal store)?
// Test patterns:
//   Mode 0: simple store
//   Mode 1: PTX st.global.wt (write-through)
//   Mode 2: PTX st.global.cs (cache-streaming)
//   Mode 3: PTX st.global.wb (write-back)
//   Mode 4: streaming write loop pattern (no later read)

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int* Au = (unsigned int*)A;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FFF;
    unsigned int val = (unsigned)idx + (unsigned)u2;

#if MODE == 0
    Au[idx] = val;
#elif MODE == 1
    asm("st.global.wt.u32 [%0], %1;" :: "l"(Au + idx), "r"(val));
#elif MODE == 2
    asm("st.global.cs.u32 [%0], %1;" :: "l"(Au + idx), "r"(val));
#elif MODE == 3
    asm("st.global.wb.u32 [%0], %1;" :: "l"(Au + idx), "r"(val));
#elif MODE == 4
    // Loop with multiple distinct stores per thread (streaming)
    for (int i = 0; i < 16; i++) {
        Au[idx + i * 1024] = val + i;
    }
#endif
}
