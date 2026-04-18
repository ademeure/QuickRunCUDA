// I3 RIGOR: GPU clock during FFMA vs tensor core load
// Theoretical: SM clock should be same; both pipes are in same clock domain.
// But tensor cores draw MORE power, so thermal/power throttle could reduce clock.
//
// Method: long-running FFMA vs long-running mma.sync. Sample NVML clock.
// Cross-check: clock64 vs globaltimer ratio inside each kernel.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <nvml.h>
#include <chrono>
#include <thread>
#include <atomic>

extern "C" __launch_bounds__(256, 8) __global__ void ffma_long(float *out, float a, int iters, unsigned long long *clk_out) {
    float r0=0.5f, r1=1.5f, r2=2.5f, r3=3.5f, r4=4.5f, r5=5.5f, r6=6.5f, r7=7.5f;
    float b = a + 1, c = a + 2;
    unsigned long long c0 = clock64();
    unsigned long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        r0 = r0*a+b; r1 = r1*a+c; r2 = r2*a+b; r3 = r3*a+c;
        r4 = r4*a+b; r5 = r5*a+c; r6 = r6*a+b; r7 = r7*a+c;
    }
    unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    unsigned long long c1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        clk_out[0] = c1 - c0;
        clk_out[1] = g1 - g0;
    }
    float s = r0+r1+r2+r3+r4+r5+r6+r7;
    if (s < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = s;
}

extern "C" __launch_bounds__(256, 4) __global__ void mma_long(float *out, int iters, unsigned long long *clk_out) {
    int laneId = threadIdx.x & 31;
    unsigned a0 = laneId, a1 = laneId+1, a2 = laneId+2, a3 = laneId+3;
    unsigned b0 = laneId*7, b1 = laneId*11;
    float c0=0,c1=0,c2=0,c3=0;
    unsigned long long t0 = clock64();
    unsigned long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        clk_out[0] = t1 - t0;
        clk_out[1] = g1 - g0;
    }
    if (c0+c1+c2+c3 < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = c0+c1+c2+c3;
}

int main() {
    cudaSetDevice(0);
    nvmlInit();
    nvmlDevice_t nvml_dev;
    nvmlDeviceGetHandleByIndex(0, &nvml_dev);

    float *d_out; cudaMalloc(&d_out, 1<<24);
    unsigned long long *d_clk; cudaMalloc(&d_clk, 16 * sizeof(unsigned long long));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](auto launch, const char* label) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();

        // Sample power+clock during run
        std::atomic<bool> done{false};
        std::atomic<long long> sum_mw{0};
        std::atomic<long long> sum_mhz{0};
        std::atomic<int> n{0};
        std::thread t([&]() {
            while (!done) {
                unsigned mw, mhz;
                if (nvmlDeviceGetPowerUsage(nvml_dev, &mw) == NVML_SUCCESS &&
                    nvmlDeviceGetClockInfo(nvml_dev, NVML_CLOCK_SM, &mhz) == NVML_SUCCESS) {
                    sum_mw += mw; sum_mhz += mhz; n++;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });

        cudaEventRecord(e0);
        for (int i = 0; i < 30; i++) launch();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        done = true; t.join();
        float ms; cudaEventElapsedTime(&ms, e0, e1);

        unsigned long long h_clk[2];
        cudaMemcpy(h_clk, d_clk, 2*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        double sm_clock_via_clk64 = (double)h_clk[0] / h_clk[1] * 1e9;  // cycles/s

        printf("  %s:\n", label);
        printf("    Time: %.1f ms\n", ms);
        printf("    NVML  : %.0f W avg, %.0f MHz avg over %d samples\n",
               (double)sum_mw/n/1000, (double)sum_mhz/n, n.load());
        printf("    clock64/globaltimer: SM clock = %.1f MHz\n", sm_clock_via_clk64/1e6);
    };

    bench([&]{ ffma_long<<<148*8, 256>>>(d_out, 1.5f, 100000, d_clk); }, "FFMA (heavy)    ");
    bench([&]{ mma_long<<<148*4, 256>>>(d_out, 100000, d_clk); }, "mma.sync BF16   ");

    nvmlShutdown();
    return 0;
}
