// cuStreamWaitValue / WriteValue: GPU-side polling primitives
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}

extern "C" __global__ void delay_then_write(uint32_t *flag, int delay_iters, uint32_t val) {
    float a = (float)threadIdx.x;
    for (int i = 0; i < delay_iters; i++) a = a*1.0001f + 0.0001f;
    if (a < -1e30f) flag[blockIdx.x+1] = 1;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        *flag = val;
    }
}

int main() {
    cudaSetDevice(0);
    cuInit(0);

    uint32_t *d_flag;
    cudaMalloc(&d_flag, 1024 * sizeof(uint32_t));
    cudaMemset(d_flag, 0, 1024 * sizeof(uint32_t));

    cudaStream_t s_prod, s_cons;
    cudaStreamCreate(&s_prod);
    cudaStreamCreate(&s_cons);

    printf("# B300 cuStreamWaitValue/WriteValue: GPU-side polling primitives\n\n");

    // Test 1: WriteValue alone
    printf("## Test 1: WriteValue cost\n");
    {
        uint32_t val = 42;
        // Warmup
        for (int i = 0; i < 5; i++) cuStreamWriteValue32(s_prod, (CUdeviceptr)d_flag, val++, 0);
        cudaStreamSynchronize(s_prod);

        float best = 1e30f;
        for (int i = 0; i < 100; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cuStreamWriteValue32(s_prod, (CUdeviceptr)d_flag, val++, 0);
            cudaStreamSynchronize(s_prod);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        printf("  WriteValue + sync: %.2f us\n", best);
    }

    // Test 2: WriteValue then WaitValue same stream (should be ~0 cost)
    printf("\n## Test 2: Write + Wait same stream (already-satisfied)\n");
    {
        uint32_t target = 1000;
        // Pre-set
        cuStreamWriteValue32(s_prod, (CUdeviceptr)d_flag, target, 0);
        cudaStreamSynchronize(s_prod);

        float best = 1e30f;
        for (int i = 0; i < 100; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            // Wait on already-set value
            cuStreamWaitValue32(s_prod, (CUdeviceptr)d_flag, target, CU_STREAM_WAIT_VALUE_EQ);
            cudaStreamSynchronize(s_prod);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        printf("  Wait on satisfied value: %.2f us\n", best);
    }

    // Test 3: Cross-stream wait + kernel signaling (no pre-arming)
    printf("\n## Test 3: Kernel write → cross-stream wait\n");
    {
        // Producer kernel writes to flag, consumer waits, then runs noop
        uint32_t target = 7777;
        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            cudaMemsetAsync(d_flag, 0, 4, s_prod);
            cudaStreamSynchronize(s_prod);
            cudaStreamSynchronize(s_cons);

            auto t0 = std::chrono::high_resolution_clock::now();
            // Producer
            delay_then_write<<<1, 32, 0, s_prod>>>(d_flag, 5000, target);
            // Consumer waits then runs noop
            cuStreamWaitValue32(s_cons, (CUdeviceptr)d_flag, target, CU_STREAM_WAIT_VALUE_EQ);
            noop<<<1, 32, 0, s_cons>>>();
            cudaStreamSynchronize(s_cons);
            cudaStreamSynchronize(s_prod);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        printf("  Wait/Write chain (5000 iter delay):  %.2f us\n", best);

        // Compare to using event sync
        cudaEvent_t e; cudaEventCreate(&e);
        float ev_best = 1e30f;
        for (int i = 0; i < 30; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            delay_then_write<<<1, 32, 0, s_prod>>>(d_flag, 5000, target);
            cudaEventRecord(e, s_prod);
            cudaStreamWaitEvent(s_cons, e, 0);
            noop<<<1, 32, 0, s_cons>>>();
            cudaStreamSynchronize(s_cons);
            cudaStreamSynchronize(s_prod);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < ev_best) ev_best = us;
        }
        printf("  Event-based equivalent:              %.2f us\n", ev_best);
        printf("  WaitValue is %.2f us %s than event sync\n",
               fabsf(best - ev_best), best < ev_best ? "FASTER" : "slower");
    }

    return 0;
}
