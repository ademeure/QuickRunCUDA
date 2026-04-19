// D2: Persistent kernel template — 1 block per SM, atomic work distribution
//
// Pattern:
//   - Launch 148 blocks (1 per SM) once
//   - Each block grabs work units from a global atomic counter
//   - Communicates with host via mapped memory or device-side commands
//   - Eliminates per-launch overhead (0.5-5 us → 0 amortized)
//
// USE CASES:
//   - Streaming inference: many small kernels back-to-back → just enqueue
//   - Producer/consumer between GPUs
//   - Server-mode computation (e.g. matmul service)
//
// USAGE:
//   1. Host fills work queue via cudaMemcpyAsync
//   2. Host signals "go" via mapped memory
//   3. Kernel loops: pop work item → process → write result → signal done
//   4. Host signals "stop" via mapped memory; kernel exits
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

constexpr int N_SMS = 148;

// Per-SM persistent kernel state
struct PersistentState {
    volatile int signal;        // 0=run, 1=stop
    volatile int work_counter;  // atomic work index
    volatile int done_count;    // count of completed work items
    int N_work;                 // total work items
};

// Demo kernel: each work item = compute sum-of-squares of D=4096 BF16 elements
__launch_bounds__(256, 4) __global__
void persistent_kernel(PersistentState *state, const float *input, float *output, int D) {
    while (true) {
        // Cooperative work-stealing
        int my_work;
        if (threadIdx.x == 0) {
            // Wait for signal
            while (true) {
                int s = state->signal;
                if (s == 1) { my_work = -1; break; }   // stop
                int w = atomicAdd((int*)&state->work_counter, 1);
                if (w < state->N_work) { my_work = w; break; }
                __nanosleep(100);
            }
        }
        my_work = __shfl_sync(0xFFFFFFFF, my_work, 0);
        // Broadcast across block via SMEM
        __shared__ int shared_work;
        if (threadIdx.x == 0) shared_work = my_work;
        __syncthreads();
        my_work = shared_work;

        if (my_work < 0) return;

        // Process work item: sum-of-squares of input[my_work * D : (my_work+1) * D]
        const float *row = input + my_work * D;
        float acc = 0;
        for (int j = threadIdx.x; j < D; j += blockDim.x) {
            float v = row[j];
            acc += v * v;
        }
        // Block reduce
        __shared__ float smem[8];  // 8 warps × float
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xFFFFFFFF, acc, s);
        if (lane == 0) smem[warp] = acc;
        __syncthreads();
        if (warp == 0) {
            acc = (lane < blockDim.x/32) ? smem[lane] : 0;
            for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xFFFFFFFF, acc, s);
            if (lane == 0) {
                output[my_work] = acc;
                __threadfence_system();
                atomicAdd((int*)&state->done_count, 1);
            }
        }
    }
}

int main() {
    cudaSetDevice(0);
    constexpr int N_WORK = 1024;
    constexpr int D = 4096;

    // Mapped state for host-device sync
    PersistentState *h_state, *d_state;
    cudaHostAlloc(&h_state, sizeof(PersistentState), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_state, h_state, 0);
    h_state->signal = 0;
    h_state->work_counter = 0;
    h_state->done_count = 0;
    h_state->N_work = N_WORK;

    float *d_input, *d_output;
    cudaMalloc(&d_input, (size_t)N_WORK * D * sizeof(float));
    cudaMalloc(&d_output, N_WORK * sizeof(float));
    cudaMemset(d_input, 0x40, (size_t)N_WORK * D * sizeof(float));

    cudaStream_t s; cudaStreamCreate(&s);

    // Launch persistent kernel: 1 block per SM, runs until signal=1
    persistent_kernel<<<N_SMS, 256, 0, s>>>(d_state, d_input, d_output, D);

    // Wait for kernel to complete all work (poll done_count)
    auto t0 = std::chrono::high_resolution_clock::now();
    while (h_state->done_count < N_WORK) {
        // busy wait or sleep briefly
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("# Persistent kernel processed %d work items in %.3f ms (%.1f us/item)\n",
           N_WORK, ms, ms*1000/N_WORK);

    // Reset and run again (no kernel launch overhead!)
    h_state->done_count = 0;
    h_state->work_counter = 0;
    auto t2 = std::chrono::high_resolution_clock::now();
    while (h_state->done_count < N_WORK) {}
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms2 = std::chrono::duration<double, std::milli>(t3 - t2).count();
    printf("# Second run (no relaunch):  %d items in %.3f ms (%.1f us/item)\n",
           N_WORK, ms2, ms2*1000/N_WORK);

    // Compare to per-launch baseline: relaunch fresh kernel each time
    h_state->signal = 1;  // stop persistent
    cudaStreamSynchronize(s);

    // Single-shot baseline
    h_state->signal = 0;
    h_state->done_count = 0;
    h_state->work_counter = 0;
    auto t4 = std::chrono::high_resolution_clock::now();
    persistent_kernel<<<N_SMS, 256, 0, s>>>(d_state, d_input, d_output, D);
    while (h_state->done_count < N_WORK) {}
    auto t5 = std::chrono::high_resolution_clock::now();
    h_state->signal = 1;
    cudaStreamSynchronize(s);
    double ms3 = std::chrono::duration<double, std::milli>(t5 - t4).count();
    printf("# Cold launch + %d items:    %.3f ms (relaunch overhead = %.1f us)\n",
           N_WORK, ms3, (ms3 - ms2) * 1000);

    cudaFreeHost(h_state);
    return 0;
}
