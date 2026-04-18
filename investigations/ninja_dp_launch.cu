// GPU-side dynamic parallelism (DP) launch overhead
//
// Theoretical:
//   CPU launch: 1.85 us per kernel (catalog)
//   Persistent handoff (relaxed.sys): 2.03 us round-trip
//   DP from device: should avoid PCIe; expect <1 us if hot
//
// Method: parent kernel launches N child kernels in a loop and waits
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

extern "C" __global__ void child_noop() {}

extern "C" __global__ void child_signal(int *flag) {
    if (threadIdx.x == 0) atomicAdd(flag, 1);
}

// Parent launches N children, waits via streamSync per child
extern "C" __global__ void parent_dp_serial(int N, int *count) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < N; i++) {
        child_signal<<<1, 32, 0, cudaStreamFireAndForget>>>(count);
    }
}

// Parent launches N children, waits ONCE at end
extern "C" __global__ void parent_dp_batch(int N, int *count) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < N; i++) {
        child_signal<<<1, 32, 0, cudaStreamFireAndForget>>>(count);
    }
}

// Parent does no DP — just runs a no-op N times itself (control)
extern "C" __global__ void parent_no_dp(int N, int *count) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < N; i++) {
        atomicAdd(count, 1);
    }
}

int main(int argc, char** argv) {
    cudaSetDevice(0);
    int N = 1000;

    int *d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    // Bench DP serial (sync after each launch)
    cudaMemset(d_count, 0, sizeof(int));
    parent_dp_serial<<<1, 32>>>(N, d_count);
    cudaDeviceSynchronize();  // warmup
    cudaMemset(d_count, 0, sizeof(int));
    cudaEventRecord(e0);
    parent_dp_serial<<<1, 32>>>(N, d_count);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    int h; cudaMemcpy(&h, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("DP serial (sync each):  %d launches in %.3f ms = %.2f us/launch  count=%d\n",
           N, ms, ms * 1000.0 / N, h);

    // Bench DP batch (sync once at end)
    cudaMemset(d_count, 0, sizeof(int));
    parent_dp_batch<<<1, 32>>>(N, d_count);
    cudaDeviceSynchronize();
    cudaMemset(d_count, 0, sizeof(int));
    cudaEventRecord(e0);
    parent_dp_batch<<<1, 32>>>(N, d_count);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    cudaMemcpy(&h, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("DP batch (sync end):    %d launches in %.3f ms = %.2f us/launch  count=%d\n",
           N, ms, ms * 1000.0 / N, h);

    // Control: no DP, just an atomic loop
    cudaMemset(d_count, 0, sizeof(int));
    cudaEventRecord(e0);
    parent_no_dp<<<1, 32>>>(N, d_count);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    cudaMemcpy(&h, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Control (no DP, atom):  %d ops in %.3f ms = %.2f us/op  count=%d\n",
           N, ms, ms * 1000.0 / N, h);

    // CPU baseline: launch N empty kernels from host
    cudaEventRecord(e0);
    for (int i = 0; i < N; i++) {
        child_signal<<<1, 32>>>(d_count);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    printf("CPU async launch x%d:    %.3f ms = %.2f us/launch (no per-call sync)\n",
           N, ms, ms * 1000.0 / N);

    return 0;
}
