// H6: Idle SM static power vs active SM
// Vary number of blocks (= active SMs); measure power
// 1 block, 2, ..., 148 blocks (1 per SM each)
// Subtract idle to find per-SM static + dynamic
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <thread>

__global__ __launch_bounds__(256, 1)
void busy(unsigned long long delay_iters) {
    if (threadIdx.x == 0) {
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < delay_iters);
    }
}

int main(int argc, char** argv) {
    int n_blocks = (argc > 1) ? atoi(argv[1]) : 1;
    unsigned long long delay = 4500000000ULL;  // 3 sec at 1500 MHz
    busy<<<n_blocks, 256>>>(delay);
    cudaDeviceSynchronize();
    return 0;
}
