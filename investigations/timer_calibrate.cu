// Quick calibration: compare globaltimer with CUDA event timing
// to understand if globaltimer == wall-clock ns or SM-cycle-based ns
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); \
    } \
} while(0)

__global__ void timer_kernel(unsigned long long* out, int ms_target) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    // spin for ~ms_target ms worth of ns
    unsigned long long target_ns = (unsigned long long)ms_target * 1000000ULL;
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    } while ((t1 - t0) < target_ns);
    out[0] = t0;
    out[1] = t1;
    out[2] = t1 - t0;
}

int main() {
    CHECK_CUDA(cudaSetDevice(0));
    unsigned long long *d_out, h_out[3];
    CHECK_CUDA(cudaMalloc(&d_out, 3*sizeof(unsigned long long)));

    // Warmup
    timer_kernel<<<1,32>>>(d_out, 10);
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int ms : {100, 200, 500}) {
        cudaEvent_t ev0, ev1;
        CHECK_CUDA(cudaEventCreate(&ev0));
        CHECK_CUDA(cudaEventCreate(&ev1));
        CHECK_CUDA(cudaEventRecord(ev0));
        timer_kernel<<<1,32>>>(d_out, ms);
        CHECK_CUDA(cudaEventRecord(ev1));
        CHECK_CUDA(cudaEventSynchronize(ev1));
        float cuda_ms;
        CHECK_CUDA(cudaEventElapsedTime(&cuda_ms, ev0, ev1));
        CHECK_CUDA(cudaMemcpy(h_out, d_out, 3*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        double gt_ms = h_out[2] / 1e6;
        printf("target=%dms  CUDA_events=%.2fms  globaltimer=%.2fms  "
               "ratio=%.4f  globaltimer_tick_rate=%.4f GHz\n",
               ms, cuda_ms, gt_ms, cuda_ms/gt_ms, h_out[2]/(cuda_ms*1e6));
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    }
    cudaFree(d_out);
    return 0;
}
