// Legacy mma.sync m16n8k32 FP8 e4m3 power test
// Tests if dedup mechanism extends to FP8 mma.sync (legacy path)

extern "C" __global__ __launch_bounds__(128, 8)
void kernel(float* A, float* B, float* C, int iters, int mode, int verify) {
    int tid = threadIdx.x;
    int warpId = tid / 32;

    // FP8 mma m16n8k32: A needs 16 FP8 per thread = 4 unsigned
    //                    B needs 8 FP8 per thread = 2 unsigned
    unsigned a0, a1, a2, a3;
    unsigned b0, b1;

    auto rand_word = [&](int seed_off) -> unsigned {
        unsigned r = (tid + seed_off + warpId * 32) * 0xCAFEBABEu;
        r ^= r >> 13;
        r *= 0x9E3779B1u;
        r ^= r >> 16;
        return r;
    };

    if (mode == 0 || mode == 1 || mode == 4) {
        a0 = a1 = a2 = a3 = 0x38383838u;  // FP8 e4m3 = +1.0 packed
    } else if (mode == 5) {
        a0 = a1 = a2 = a3 = 0u;
    } else {
        a0 = rand_word(0);
        a1 = rand_word(1);
        a2 = rand_word(2);
        a3 = rand_word(3);
    }

    if (mode == 0 || mode == 2) {
        b0 = b1 = 0x38383838u;
    } else if (mode == 4 || mode == 5) {
        b0 = b1 = 0u;
    } else {
        b0 = rand_word(100);
        b1 = rand_word(101);
    }

    float c0=0, c1=0, c2=0, c3=0;

    unsigned long long t0=0, t1=0;
    if (tid % 32 == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 0; i < iters; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};\n\t"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1));
    }

    if (tid % 32 == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (verify) {
        C[tid] = c0 + c1 + c2 + c3;
    }

    if (tid == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        printf("MMA sync FP8 mode=%d iters=%d cy/inst=%.2f\n", mode, iters, (double)(t1-t0)/iters);
    }
}
