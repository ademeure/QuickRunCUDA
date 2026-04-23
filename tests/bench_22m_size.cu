// Kernel-size sweep for §22m. Total static FFMA count = N_INSTS, controllable
// at NVRTC time via -H "#define N_INSTS <K>" to vary cubin size.
//
// Implementation: one thread (threadIdx.x==0 / blockIdx.x==0) does N_INSTS
// independent FFMAs in a fully-unrolled `#pragma unroll` loop. The loop body
// contains EXACTLY one fma.rn.f32 PTX instruction so SASS instruction count
// == N_INSTS (plus a fixed prologue/epilogue ≈ 8 SASS lines).
//
// Anti-DCE: the final accumulator is written to C[] under an `if(threadIdx.x>=blockDim.x)`
// guard which is always false at runtime but the compiler can't prove. So the
// FFMAs are kept, but at runtime each block does ZERO useful memory traffic.
//
// We launch with -t 32 -b 1 (1 warp, 1 block) so kernel runtime is dominated
// by the FFMA chain on warp lane 0 (and identical work on lanes 1..31, but
// hardware issues all 32 lanes simultaneously per FFMA insts ⇒ same time).

#ifndef N_INSTS
#define N_INSTS 10
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int u0, int u1, int u2) {
    float tid_f = __int_as_float(threadIdx.x | 0x3F800000u);
    float a = 1.0000001f;
    float b = 0.9999999f;
    float r = tid_f;

    #pragma unroll
    for (int i = 0; i < N_INSTS; i++) {
        asm volatile("fma.rn.f32 %0, %1, %2, %0;\n\t" : "+f"(r) : "f"(a), "f"(b));
    }

    if (threadIdx.x >= blockDim.x) {
        C[threadIdx.x] = r;
    }
}
