// R2: Idle SM "active" cost — kernel waiting on a flag (no work done)
// Compare:
// MODE 0: kernel that exits immediately (truly idle SMs)
// MODE 1: kernel waiting on managed-mem flag via spin (active warp, no compute)
// MODE 2: kernel waiting via __syncthreads loop (active barrier)
// MODE 3: kernel waiting via mbarrier.try_wait (HW barrier wait state)
#include <cuda_runtime.h>
#include <cstdio>
#include <thread>
#include <chrono>

__global__ __launch_bounds__(32, 1)
void exit_immediately() {}

__global__ __launch_bounds__(32, 1)
void spin_until_flag(volatile unsigned int* flag) {
    if (threadIdx.x == 0) {
        while (*flag == 0) { /* spin */ }
    }
}

__global__ __launch_bounds__(32, 1)
void sync_loop_until_flag(volatile unsigned int* flag) {
    while (*flag == 0) {
        __syncthreads();
    }
}

__global__ __launch_bounds__(32, 1)
void mbarrier_wait_until_flag(volatile unsigned int* flag) {
    __shared__ __align__(8) unsigned long long bar;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;"
            :: "r"((unsigned)__cvta_generic_to_shared(&bar)));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    while (*flag == 0) {
        // arrive then try_wait
        unsigned int bar_addr = (unsigned)__cvta_generic_to_shared(&bar);
        asm volatile("{ .reg .b64 state;\n"
                     "  mbarrier.arrive.shared::cta.b64 state, [%0]; }"
                     :: "r"(bar_addr));
        asm volatile("{ .reg .pred p;\n"
                     "L_w_%=: mbarrier.try_wait.shared::cta.b64 p, [%0], 0;\n"
                     "  @!p bra L_w_%=; }"
                     :: "r"(bar_addr));
    }
}

int main(int argc, char** argv) {
    int mode = (argc > 1) ? atoi(argv[1]) : 0;
    cudaSetDevice(0);

    volatile unsigned int* flag;
    cudaMallocManaged((void**)&flag, sizeof(unsigned int));
    *flag = 0;

    int n_blocks = 148;  // 1 per SM
    cudaStream_t s;
    cudaStreamCreate(&s);

    // Launch the chosen waiter
    if (mode == 0) {
        exit_immediately<<<n_blocks, 32, 0, s>>>();
    } else if (mode == 1) {
        spin_until_flag<<<n_blocks, 32, 0, s>>>(flag);
    } else if (mode == 2) {
        sync_loop_until_flag<<<n_blocks, 32, 0, s>>>(flag);
    } else if (mode == 3) {
        mbarrier_wait_until_flag<<<n_blocks, 32, 0, s>>>(flag);
    }

    // Sleep 3 sec to allow power sample
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Tell the kernel to exit
    *flag = 1;
    cudaStreamSynchronize(s);
    return 0;
}
