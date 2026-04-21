// I6: Block-to-SM scheduling order
// Each block reads %smid; observe block→SM mapping
extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    ((unsigned int*)C)[blockIdx.x] = smid;
}
