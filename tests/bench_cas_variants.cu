// CAS half-rate irrespective of compare result?
// OP=0: CAS always succeeds (compare matches memory)
// OP=1: CAS always fails (compare never matches)
// OP=2: baseline atom.shared.add (for reference)
// OP=3: CAS but different write value each time

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
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
    // Each thread owns 8 slots — initialize to known value
    unsigned int base = (unsigned)__cvta_generic_to_shared(&smem[threadIdx.x * 8]);
    for (int k = 0; k < 8; k++) smem[threadIdx.x * 8 + k] = 0x12345678u;
    __syncthreads();

    unsigned int v0=0,v1=0,v2=0,v3=0,v4=0,v5=0,v6=0,v7=0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // CAS always succeeds: compare = actual mem value (0x12345678), swap writes same
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v0) : "r"(base + 0*4), "r"(0x12345678u), "r"(0x12345678u));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v1) : "r"(base + 1*4), "r"(0x12345678u), "r"(0x12345678u));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v2) : "r"(base + 2*4), "r"(0x12345678u), "r"(0x12345678u));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v3) : "r"(base + 3*4), "r"(0x12345678u), "r"(0x12345678u));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v4) : "r"(base + 4*4), "r"(0x12345678u), "r"(0x12345678u));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v5) : "r"(base + 5*4), "r"(0x12345678u), "r"(0x12345678u));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v6) : "r"(base + 6*4), "r"(0x12345678u), "r"(0x12345678u));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v7) : "r"(base + 7*4), "r"(0x12345678u), "r"(0x12345678u));
#elif OP == 1  // CAS always fails: compare is wrong value, memory stays 0x12345678
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v0) : "r"(base + 0*4), "r"(0xDEADBEEFu), "r"(0xABABABABu));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v1) : "r"(base + 1*4), "r"(0xDEADBEEFu), "r"(0xABABABABu));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v2) : "r"(base + 2*4), "r"(0xDEADBEEFu), "r"(0xABABABABu));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v3) : "r"(base + 3*4), "r"(0xDEADBEEFu), "r"(0xABABABABu));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v4) : "r"(base + 4*4), "r"(0xDEADBEEFu), "r"(0xABABABABu));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v5) : "r"(base + 5*4), "r"(0xDEADBEEFu), "r"(0xABABABABu));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v6) : "r"(base + 6*4), "r"(0xDEADBEEFu), "r"(0xABABABABu));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v7) : "r"(base + 7*4), "r"(0xDEADBEEFu), "r"(0xABABABABu));
#elif OP == 2  // baseline: atom.shared.add
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v0) : "r"(base + 0*4), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v1) : "r"(base + 1*4), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v2) : "r"(base + 2*4), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v3) : "r"(base + 3*4), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v4) : "r"(base + 4*4), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v5) : "r"(base + 5*4), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v6) : "r"(base + 6*4), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v7) : "r"(base + 7*4), "r"(1u));
#endif
        }
    }
    unsigned int acc = v0^v1^v2^v3^v4^v5^v6^v7;
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
