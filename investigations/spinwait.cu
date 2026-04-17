// Spin-wait patterns: loads vs CAS vs nanosleep
#include <cuda_runtime.h>
#include <cstdio>

// Producer kernel: writes flag after delay
extern "C" __global__ void producer(unsigned *flag, unsigned val, int delay_iters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float a = 0;
        for (int i = 0; i < delay_iters; i++) a = a*1.0001f + 0.0001f;
        if (a > 1e30f) *flag = 0xdead;
        __threadfence_system();
        *flag = val;
    }
}

// Consumer pattern 1: spin on regular load
extern "C" __global__ void cons_load(unsigned *flag, unsigned long long *out, unsigned target) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (*((volatile unsigned*)flag) != target) {}
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

// Consumer pattern 2: spin with nanosleep
extern "C" __global__ void cons_nanosleep(unsigned *flag, unsigned long long *out, unsigned target) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (*((volatile unsigned*)flag) != target) {
            asm volatile("nanosleep.u32 %0;" :: "r"(1000u));  // 1us
        }
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

// Consumer pattern 3: ld.acquire.gpu
extern "C" __global__ void cons_acquire(unsigned *flag, unsigned long long *out, unsigned target) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        unsigned cur;
        do {
            asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(cur) : "l"(flag));
        } while (cur != target);
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

// Pure cost (no production): same kernel but flag pre-set
extern "C" __global__ void cons_load_satisfied(unsigned *flag, unsigned long long *out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        unsigned cur = *((volatile unsigned*)flag);
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
        out[1] = cur;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned *d_flag; cudaMalloc(&d_flag, sizeof(unsigned));
    unsigned long long *d_out; cudaMalloc(&d_out, 16*sizeof(unsigned long long));

    cudaStream_t s_prod, s_cons;
    cudaStreamCreate(&s_prod);
    cudaStreamCreate(&s_cons);

    printf("# B300 spin-wait patterns: cycles to detect flag transition\n");
    printf("# Producer delays for ~100us, then writes flag\n\n");

    int delay = 100000;  // ~100 us of FMAs
    unsigned target = 42;

    auto run = [&](auto cons_fn, const char *name) {
        // Reset flag
        cudaMemset(d_flag, 0, sizeof(unsigned));
        cudaDeviceSynchronize();

        // Launch producer (will signal after delay)
        producer<<<1, 1, 0, s_prod>>>(d_flag, target, delay);
        // Launch consumer concurrently on different stream
        cons_fn<<<1, 1, 0, s_cons>>>(d_flag, d_out, target);
        cudaDeviceSynchronize();

        unsigned long long cyc;
        cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        printf("  %-25s %llu cyc = %.1f us\n", name, (unsigned long long)cyc, cyc/2.032/1000);
    };

    run(cons_load,      "spin on volatile load");
    run(cons_acquire,   "spin on ld.acquire.gpu");
    run(cons_nanosleep, "spin + nanosleep(1us)");

    // Pure load latency (uncontended)
    cudaMemset(d_flag, 0, sizeof(unsigned));
    {
        unsigned val = 99;
        cudaMemcpy(d_flag, &val, sizeof(unsigned), cudaMemcpyHostToDevice);
        cons_load_satisfied<<<1, 1>>>(d_flag, d_out);
        cudaDeviceSynchronize();
        unsigned long long cyc; cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        printf("\n  Pure ld.volatile latency: %llu cyc = %.1f ns\n",
               (unsigned long long)cyc, cyc/2.032);
    }

    return 0;
}
