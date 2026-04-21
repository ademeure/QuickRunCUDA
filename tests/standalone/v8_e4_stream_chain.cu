// V8 E4: Multi-stream serialization via WriteValue/WaitValue chains
// Compare (a) single-stream sequential (b) multi-stream chain w/ semaphores
// (c) multi-stream chain overlapping pre-compute on stage i+1 with stage i write
//
// Pattern A (baseline): 4 kernels on 1 stream → naturally serialized
// Pattern B (chain): 4 streams, each waits for prev's semaphore, signals own
// Pattern C (pipeline): stage i's output fed to stage i+1 via sem — can stages
//                        advance in "pipeline" fashion if kernel has phases?
//
// For kernels of duration D, what's the latency-per-stage overhead of the
// semaphore chain vs native stream dependencies?
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// Tiny kernel (short duration — amplifies overhead)
__global__ void tiny(int* counter, int iters) {
    int v = 0;
    for (int i = 0; i < iters; i++) v = v*1664525 + 1013904223;
    if (v == 0xdeadbeef) counter[0] = v;
}

// Longer kernel (scaled)
__global__ void longer(int* counter, int iters) {
    int v = 0;
    for (int i = 0; i < iters; i++) v = v*1664525 + 1013904223;
    if (v == 0xdeadbeef) counter[0] = v;
}

int main() {
    cudaSetDevice(0);
    cuInit(0);

    int N_STAGES = 4;
    int N_REPEATS = 200;
    int* d_out;
    CK(cudaMalloc(&d_out, 64));

    // Allocate semaphores (managed for host visibility but will be touched only by GPU + cuStreamWait)
    unsigned int* sems;
    CK(cudaMallocManaged((void**)&sems, sizeof(unsigned int) * N_STAGES));
    CUdeviceptr dev_sems[8];
    for (int i = 0; i < N_STAGES; i++) dev_sems[i] = (CUdeviceptr)(&sems[i]);

    cudaStream_t streams[8];
    for (int i = 0; i < N_STAGES; i++) cudaStreamCreate(&streams[i]);

    // Warmup
    for (int i = 0; i < 3; i++) {
        tiny<<<132, 32, 0, streams[0]>>>(d_out, 100);
        cudaStreamSynchronize(streams[0]);
    }

    for (int iters_per_stage : {100, 1000, 10000}) {
        fprintf(stderr, "\n=== iters_per_stage = %d (short:100, med:1000, long:10000) ===\n", iters_per_stage);

        // (A) Single-stream sequential
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < N_REPEATS; r++) {
            for (int s = 0; s < N_STAGES; s++) {
                tiny<<<132, 32, 0, streams[0]>>>(d_out, iters_per_stage);
            }
        }
        cudaStreamSynchronize(streams[0]);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us_A = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_REPEATS;

        // (B) Multi-stream chain via WriteValue/WaitValue
        // Reset semaphores
        for (int i = 0; i < N_STAGES; i++) sems[i] = 0;
        cudaDeviceSynchronize();

        auto t2 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < N_REPEATS; r++) {
            unsigned int target = (unsigned)(r + 1);
            // Stage 0 — launch freely
            tiny<<<132, 32, 0, streams[0]>>>(d_out, iters_per_stage);
            cuStreamWriteValue32(streams[0], dev_sems[0], target, 0);
            // Stages 1..N-1 — wait for prev, run, signal
            for (int s = 1; s < N_STAGES; s++) {
                cuStreamWaitValue32(streams[s], dev_sems[s-1], target, CU_STREAM_WAIT_VALUE_GEQ);
                tiny<<<132, 32, 0, streams[s]>>>(d_out, iters_per_stage);
                cuStreamWriteValue32(streams[s], dev_sems[s], target, 0);
            }
        }
        // Wait final stream
        cudaStreamSynchronize(streams[N_STAGES-1]);
        auto t3 = std::chrono::high_resolution_clock::now();
        double us_B = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_REPEATS;

        // (C) Multi-stream with cudaStreamWaitEvent (native CUDA sync)
        cudaEvent_t events[8];
        for (int i = 0; i < N_STAGES; i++) cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);

        cudaDeviceSynchronize();
        auto t4 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < N_REPEATS; r++) {
            // Stage 0
            tiny<<<132, 32, 0, streams[0]>>>(d_out, iters_per_stage);
            cudaEventRecord(events[0], streams[0]);
            for (int s = 1; s < N_STAGES; s++) {
                cudaStreamWaitEvent(streams[s], events[s-1], 0);
                tiny<<<132, 32, 0, streams[s]>>>(d_out, iters_per_stage);
                cudaEventRecord(events[s], streams[s]);
            }
        }
        cudaStreamSynchronize(streams[N_STAGES-1]);
        auto t5 = std::chrono::high_resolution_clock::now();
        double us_C = std::chrono::duration<double, std::micro>(t5 - t4).count() / N_REPEATS;

        fprintf(stderr, "  (A) single-stream sequential 4 kernels: %.2f us/iter (%.2f us/stage)\n", us_A, us_A / N_STAGES);
        fprintf(stderr, "  (B) multi-stream WriteValue chain:      %.2f us/iter (%.2f us/stage)\n", us_B, us_B / N_STAGES);
        fprintf(stderr, "  (C) multi-stream Event chain:           %.2f us/iter (%.2f us/stage)\n", us_C, us_C / N_STAGES);
        fprintf(stderr, "  chain overhead: (B) vs (A) = %+.2f us/iter; (C) vs (A) = %+.2f us/iter\n",
                us_B - us_A, us_C - us_A);

        for (int i = 0; i < N_STAGES; i++) cudaEventDestroy(events[i]);
    }

    fprintf(stderr, "\n(A) = natural stream ordering — baseline\n");
    fprintf(stderr, "(B) = WriteValue/WaitValue semaphore chain across streams\n");
    fprintf(stderr, "(C) = Event record/wait dependency chain across streams\n");
    fprintf(stderr, "If (B) or (C) < (A) at short-kernel sizes: async dispatch overlap wins vs in-order serialization\n");
    fprintf(stderr, "If (B) or (C) > (A) at long-kernel sizes: signaling overhead dominates kernel time, sequential better\n");

    for (int i = 0; i < N_STAGES; i++) cudaStreamDestroy(streams[i]);
    cudaFree(d_out);
    cudaFree(sems);
    return 0;
}
