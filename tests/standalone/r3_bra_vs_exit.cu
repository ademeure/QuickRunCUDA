// R3: SM in BRA loop vs EXIT — does power differ?
// Run kernel on N SMs:
// MODE 0: spin in BRA loop (SM still has active warp)
// MODE 1: launch then immediate EXIT (SM idle, but launched)
// MODE 2: never launch (true idle)
#include <cuda_runtime.h>
#include <cstdio>

__global__ __launch_bounds__(32, 1)
void spin_loop(unsigned long long delay) {
    if (threadIdx.x == 0) {
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < delay);
    }
}

__global__ __launch_bounds__(32, 1)
void exit_immediately() {
    // Immediate exit
}

int main(int argc, char** argv) {
    int mode = (argc > 1) ? atoi(argv[1]) : 0;
    int n_blocks = 148;
    unsigned long long delay = 4500000000ULL;  // 3 sec at 1500 MHz

    if (mode == 0) {
        spin_loop<<<n_blocks, 32>>>(delay);
    } else if (mode == 1) {
        // Launch many empty kernels in a tight loop to keep SMs "occupied" but idle
        for (int i = 0; i < 1000; i++) {
            exit_immediately<<<n_blocks, 32>>>();
        }
    } else {
        // No launch, just sleep
        cudaDeviceSynchronize();
        struct timespec ts = {3, 0};
        nanosleep(&ts, nullptr);
    }
    cudaDeviceSynchronize();
    return 0;
}
