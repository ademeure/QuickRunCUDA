// V8 SMEM-BW via ldmatrix: theoretical better than plain LDS due to wider fetch.
// ldmatrix.sync.aligned.m8n8.x4 loads 8×8 × 4 = 256 bytes per warp per issue.
// 4 SMSPs × 256 B = 1024 B/cy/SM theoretical? Actually bank-limited to 128 B.
#ifndef SMEM_BYTES
#define SMEM_BYTES 32768
#endif

extern "C" __global__ __launch_bounds__(128, 8)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ alignas(128) float smem[SMEM_BYTES/4];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    // Init SMEM
    #pragma unroll
    for (int i = tid; i < SMEM_BYTES/4; i += blockDim.x) {
        smem[i] = A[(gtid + i) & (SMEM_BYTES/4 - 1)];
    }
    __syncthreads();

    // ldmatrix.sync.aligned.m8n8.x4 — each thread gets 4 × 32-bit = 16 B per issue
    // 32 threads × 16 B = 512 B per warp per issue
    unsigned int acc_x = 0, acc_y = 0, acc_z = 0, acc_w = 0;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int warp_off = warp_id * 256;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        int iter_off = (i * 4 * 32) & (SMEM_BYTES/4 - 1);
        int local = (warp_off / 4 + (lane & 7) * 8 + iter_off) & (SMEM_BYTES/4 - 1);
        unsigned int ptr = __cvta_generic_to_shared(&smem[local]);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(acc_x), "=r"(acc_y), "=r"(acc_z), "=r"(acc_w)
            : "r"(ptr)
        );
    }

    // Anti-DCE
    float sum = (float)(acc_x + acc_y + acc_z + acc_w);
    if (sum == 1.234567e-30f) C[gtid] = sum;
}
