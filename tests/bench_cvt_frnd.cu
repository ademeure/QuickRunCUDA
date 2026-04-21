// cvt.frnd.f32.f32 (round float to float — for floor/ceil/round)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 0.5f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // cvt.rni.f32.f32 (round-to-nearest-int as float)
        asm("cvt.rni.f32.f32 %0, %0;" : "+f"(fv));
#elif MODE == 1
        // cvt.rzi.f32.f32 (truncate / floor of positive)
        asm("cvt.rzi.f32.f32 %0, %0;" : "+f"(fv));
#elif MODE == 2
        // cvt.rmi.f32.f32 (floor)
        asm("cvt.rmi.f32.f32 %0, %0;" : "+f"(fv));
#elif MODE == 3
        // cvt.rpi.f32.f32 (ceil)
        asm("cvt.rpi.f32.f32 %0, %0;" : "+f"(fv));
#elif MODE == 4
        // floorf intrinsic
        fv = floorf(fv);
#elif MODE == 5
        // truncf intrinsic
        fv = truncf(fv);
#elif MODE == 6
        // roundf intrinsic
        fv = roundf(fv);
#elif MODE == 7
        // rintf intrinsic (banker's round)
        fv = rintf(fv);
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
