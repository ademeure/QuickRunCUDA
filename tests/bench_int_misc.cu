// Less-common integer ops: BMSK, ISCADD, IABS, LEA, byte extracts, etc.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[8];
    #pragma unroll
    for (int k=0;k<8;k++) v[k] = threadIdx.x * 131 + k * 17;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k=0;k<8;k++) {
                unsigned int nxt = v[(k+1) & 7];
#if OP == 0  // bmsk (bitfield mask)
                asm volatile("bmsk.b32 %0, %0, %1;" : "+r"(v[k]) : "r"(nxt & 0x1F));
#elif OP == 1  // shf.l + IADD (emulate scaled-add)
                asm volatile("shf.l.clamp.b32 %0, %0, %0, %1;" : "+r"(v[k]) : "r"(2u));
                v[k] = v[k] + nxt;
#elif OP == 2  // abs.s32
                int x = (int)v[k];
                asm volatile("abs.s32 %0, %0;" : "+r"(x));
                v[k] = (unsigned int)x - nxt;
#elif OP == 3  // neg.s32
                int x = (int)v[k];
                asm volatile("neg.s32 %0, %0;" : "+r"(x));
                v[k] = (unsigned int)x + nxt;
#elif OP == 4  // cnot.b32 (bitwise not + 1)
                asm volatile("cnot.b32 %0, %0;" : "+r"(v[k]));
                v[k] ^= nxt;
#elif OP == 5  // popc.b64
                unsigned long long x = ((unsigned long long)v[k] << 32) | nxt;
                unsigned int r;
                asm volatile("popc.b64 %0, %1;" : "=r"(r) : "l"(x));
                v[k] = r ^ nxt;
#elif OP == 6  // clz.b64
                unsigned long long x = ((unsigned long long)v[k] << 32) | (nxt | 1);
                unsigned int r;
                asm volatile("clz.b64 %0, %1;" : "=r"(r) : "l"(x));
                v[k] = r ^ nxt;
#elif OP == 7  // lz + bfe pattern (bit-field extract explicit)
                asm volatile("bfe.u32 %0, %0, 4, 16;" : "+r"(v[k]));
                v[k] += nxt;
#elif OP == 8  // dp4a.s32.u32 (mixed signed)
                asm volatile("dp4a.s32.u32 %0, %0, %1, %0;" : "+r"(v[k]) : "r"(nxt));
#elif OP == 9  // dp4a.u32.s32
                asm volatile("dp4a.u32.s32 %0, %0, %1, %0;" : "+r"(v[k]) : "r"(nxt));
#elif OP == 10  // Full-saturation uniform LOP3 (3-input XOR)
                asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v[k]) : "r"(nxt), "r"(v[(k+2)&7]));
#elif OP == 11  // LOP3 with majority LUT (0xE8)
                asm volatile("lop3.b32 %0, %0, %1, %2, 0xE8;" : "+r"(v[k]) : "r"(nxt), "r"(v[(k+2)&7]));
#elif OP == 12  // szext
                asm volatile("{.reg .b32 z; szext.wrap.s32 z, %0, %1; add.u32 %0, z, 0;}" : "+r"(v[k]) : "r"(16u));
#endif
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k=0;k<8;k++) acc ^= v[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
