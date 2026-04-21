// V9: Register spill (LMEM) cost
// Force spills via high live-var count + launch_bounds capping regs.
// Compare FFMA chain with spilled accumulators vs register-resident.
#ifndef SPILL_VARS
#define SPILL_VARS 32
#endif
#ifndef THREADS
#define THREADS 256
#endif

extern "C" __global__ __launch_bounds__(THREADS, 8)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Array of SPILL_VARS floats — compiler may spill to LMEM under pressure
    float v[SPILL_VARS];
    float b = (float)(threadIdx.x + 1) * 0.001f;

    // Initialize — values must vary to prevent CSE
    #pragma unroll
    for (int k = 0; k < SPILL_VARS; k++) {
        v[k] = (float)(threadIdx.x + k) * 0.001f;
    }

    // FFMA chain on every element — forces live-ness of all vars
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < SPILL_VARS; k++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(v[k]) : "f"(b));
            }
        }
    }

    // Anti-DCE: sum and write all
    float acc = 0;
    #pragma unroll
    for (int k = 0; k < SPILL_VARS; k++) acc += v[k];
    if ((int)acc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
