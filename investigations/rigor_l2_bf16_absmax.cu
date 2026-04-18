// L2 RIGOR: optimal BF16 absmax kernel
//
// THEORETICAL: HBM read peak 7.30 TB/s. BF16 = 2 B. For 1 GB tensor:
//   - bytes read: 1 GB
//   - lower bound: 1e9 / 7.30e12 = 137 us
//
// Build: warp redux.sync.max + cluster reduction + global atomic max.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" __launch_bounds__(256, 4) __global__ void absmax_v1_naive(
    const __nv_bfloat16 *x, float *out, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float local = 0.f;
    for (int i = tid; i < N; i += stride) {
        float v = fabsf(__bfloat162float(x[i]));
        local = fmaxf(local, v);
    }
    // warp shfl reduce
    for (int o = 16; o; o >>= 1) {
        local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, o));
    }
    // first lane atomic
    if ((threadIdx.x & 31) == 0) {
        atomicMax((int*)out, __float_as_int(local));
    }
}

extern "C" __launch_bounds__(256, 4) __global__ void absmax_v2_redux(
    const __nv_bfloat16 *x, float *out, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float local = 0.f;
    // 8-ILP load: read 8 BF16 = 16 B per thread per iter (1024 B per warp coalesced)
    int warp_base = (blockIdx.x * blockDim.x + (threadIdx.x & ~31)) * 8;
    int lane = threadIdx.x & 31;
    for (int i = warp_base + lane * 8; i < N - 7; i += stride * 8) {
        // Cast pointer to read as 4 unsigned (8 BF16 = 16 B)
        uint4 v = *(uint4*)&x[i];
        const __nv_bfloat16 *p = (const __nv_bfloat16*)&v;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            local = fmaxf(local, fabsf(__bfloat162float(p[j])));
        }
    }
    // warp redux: use shfl
    for (int o = 16; o; o >>= 1) {
        local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, o));
    }
    if (lane == 0) atomicMax((int*)out, __float_as_int(local));
}

// v3: use 16-ILP per-warp coalesced + redux.sync hardware (only INT but we use bit-twiddle)
extern "C" __launch_bounds__(256, 4) __global__ void absmax_v3_bittrick(
    const __nv_bfloat16 *x, float *out, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int lane = threadIdx.x & 31;
    int warp_base_idx = (blockIdx.x * blockDim.x + (threadIdx.x & ~31)) * 8;

    // Trick: BF16 absolute value = clear sign bit. Then comparing as unsigned
    // sorts by exponent (high bits) then mantissa — gives same order as |float|.
    unsigned local_max = 0;
    for (int i = warp_base_idx + lane * 8; i < N - 7; i += stride * 8) {
        uint4 v = *(uint4*)&x[i];
        // Clear sign bits in all 8 BF16s. BF16 sign bit = bit 15 of each 16-bit element.
        // 16-bit mask = 0x7FFF; 32-bit mask = 0x7FFF7FFF
        v.x &= 0x7FFF7FFF; v.y &= 0x7FFF7FFF;
        v.z &= 0x7FFF7FFF; v.w &= 0x7FFF7FFF;
        // Per-32-bit: max of 2 packed BF16 abs values (compare each half)
        // Actually simpler: each 16-bit slot is independent abs(BF16).
        // Take max as unsigned 16-bit per slot.
        // For now: just max the unsigneds — works since BF16 layout means
        // higher-exponent BF16 has higher unsigned value.
        unsigned a = max(v.x & 0xFFFF, v.x >> 16);
        unsigned b = max(v.y & 0xFFFF, v.y >> 16);
        unsigned c = max(v.z & 0xFFFF, v.z >> 16);
        unsigned d = max(v.w & 0xFFFF, v.w >> 16);
        unsigned m1 = max(a, b);
        unsigned m2 = max(c, d);
        local_max = max(local_max, max(m1, m2));
    }
    // warp redux.sync.max (HARDWARE!)
    asm("redux.sync.max.u32 %0, %0, 0xffffffff;" : "+r"(local_max));
    // Convert BF16-abs (16-bit) back to float and atomic max as int
    if (lane == 0) {
        // Extract the 16-bit BF16 absmax and convert to float
        unsigned bf16_bits = local_max;
        unsigned f32_bits = bf16_bits << 16;
        atomicMax((int*)out, (int)f32_bits);
    }
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 1ull * 1024 * 1024 * 1024;  // 1 GB
    int N = bytes / sizeof(__nv_bfloat16);

    // Init: random BF16
    __nv_bfloat16 *h = (__nv_bfloat16*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++) h[i] = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);
    h[N/2] = __float2bfloat16(7.5f);  // KNOWN absmax for verification

    __nv_bfloat16 *d_x; cudaMalloc(&d_x, bytes);
    cudaMemcpy(d_x, h, bytes, cudaMemcpyHostToDevice);
    free(h);

    float *d_out; cudaMalloc(&d_out, sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int blocks = 148 * 4, threads = 256;

    auto bench = [&](auto launch, const char* label) {
        cudaMemset(d_out, 0, sizeof(float));
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 10; i++) {
            cudaMemset(d_out, 0, sizeof(float));
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        float result; cudaMemcpy(&result, d_out, 4, cudaMemcpyDeviceToHost);
        double gbs = bytes / (best/1000) / 1e9;
        printf("  %s: %.4f ms = %.1f GB/s = %.1f%% of 7300 HBM peak  (result=%.4f)\n",
               label, best, gbs, gbs/7300*100, result);
    };

    bench([&]{ absmax_v1_naive<<<blocks, threads>>>(d_x, d_out, N); }, "v1 naive   ");
    bench([&]{ absmax_v2_redux<<<blocks, threads>>>(d_x, d_out, N); }, "v2 v8 ILP  ");
    bench([&]{ absmax_v3_bittrick<<<blocks, threads>>>(d_x, d_out, N); }, "v3 bittrick");

    return 0;
}
