// Legacy mma.sync m16n8k16 BF16 power microbench
// Tests if data-dependent power asymmetry (A vs B) exists in legacy tensor path
// (not tcgen05).
//
// mma.sync: each warp issues 1 instruction with A in regs (8 bf16), B in regs (4 bf16)
// We loop many times with same A, B across iterations, varying their content via mode.
//
// mode 0: A const +1.0, B const +1.0 (Tier B baseline)
// mode 1: A const, B random
// mode 2: A random, B const
// mode 3: A random, B random
// mode 4: A const, B all-zero (Tier A maybe)
// mode 5: A all-zero, B all-zero

extern "C" __global__ __launch_bounds__(128, 8)
void kernel(float* A, float* B, float* C, int iters, int mode, int verify) {
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int lane = tid % 32;

    // 4 BF16 each (packed in unsigned)
    unsigned a0, a1, a2, a3;
    unsigned b0, b1;

    // Initialize register A operand (m16n8k16 needs 4 unsigned for A: 8 BF16 total per thread)
    auto rand_word = [&](int seed_off) -> unsigned {
        unsigned r = (tid + seed_off + warpId * 32) * 0xCAFEBABEu;
        r ^= r >> 13;
        r *= 0x9E3779B1u;
        r ^= r >> 16;
        return r;
    };

    if (mode == 0 || mode == 1 || mode == 4) {
        a0 = a1 = a2 = a3 = 0x3F803F80u;  // A = +1.0 packed
    } else if (mode == 5) {
        a0 = a1 = a2 = a3 = 0u;  // A = 0
    } else {
        a0 = rand_word(0);
        a1 = rand_word(1);
        a2 = rand_word(2);
        a3 = rand_word(3);
    }

    if (mode == 0 || mode == 2) {
        b0 = b1 = 0x3F803F80u;  // B = +1.0
    } else if (mode == 4 || mode == 5) {
        b0 = b1 = 0u;  // B = 0
    } else {
        b0 = rand_word(100);
        b1 = rand_word(101);
    }

    // Output accumulator
    float c0=0, c1=0, c2=0, c3=0;

    unsigned long long t0=0, t1=0;
    if (lane == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 0; i < iters; i++) {
        // m16n8k16 BF16 mma.sync.aligned with .row.col layout
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};\n\t"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1));
    }

    if (lane == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // Force write so compiler doesn't DCE
    if (verify) {
        C[tid] = c0 + c1 + c2 + c3;
    }

    if (tid == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        printf("MMA sync mode=%d iters=%d cy/inst=%.2f\n", mode, iters, (double)(t1-t0)/iters);
    }
}
