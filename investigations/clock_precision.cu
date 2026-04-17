// clock64 precision and overhead
#include <cuda_runtime.h>
#include <cstdio>

__global__ void clock_overhead(unsigned long long *out) {
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1 - t0;
}

__global__ void clock_5x(unsigned long long *out) {
    unsigned long long t[6];
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[0]));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[1]));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[2]));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[3]));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[4]));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[5]));
    for (int i = 0; i < 5; i++) out[i] = t[i+1] - t[i];
}

__global__ void clock_vs_globaltimer(unsigned long long *out) {
    unsigned long long c0, c1;
    unsigned long long g0, g1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c0));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
    // Sleep ~1us
    for (volatile int i = 0; i < 1000; i++);
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c1));

    out[0] = c1 - c0;       // SM clock cycles
    out[1] = g1 - g0;       // ns
    // clock64/globaltimer ratio = MHz/1000
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out; cudaMalloc(&d_out, 32 * sizeof(unsigned long long));

    printf("# B300 clock64 precision and overhead\n\n");

    // Test 1: minimum measurable interval
    printf("## clock64-clock64 interval (back-to-back reads):\n");
    {
        unsigned long long deltas[100];
        for (int i = 0; i < 100; i++) {
            clock_overhead<<<1, 1>>>(d_out);
            cudaDeviceSynchronize();
            cudaMemcpy(&deltas[i], d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        }
        unsigned long long min = ~0ull, max = 0;
        for (int i = 0; i < 100; i++) {
            if (deltas[i] < min) min = deltas[i];
            if (deltas[i] > max) max = deltas[i];
        }
        printf("  Min: %llu cyc, Max: %llu cyc (across 100 launches)\n", min, max);
    }

    // Test 2: 5 sequential clock reads, all deltas
    printf("\n## 5 sequential clock64 reads:\n");
    {
        clock_5x<<<1, 1>>>(d_out);
        cudaDeviceSynchronize();
        unsigned long long deltas[5];
        cudaMemcpy(deltas, d_out, 5*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        printf("  Deltas: %llu, %llu, %llu, %llu, %llu cyc\n",
               deltas[0], deltas[1], deltas[2], deltas[3], deltas[4]);
    }

    // Test 3: clock vs globaltimer
    printf("\n## clock64 vs globaltimer ratio (gives actual SM MHz):\n");
    {
        clock_vs_globaltimer<<<1, 1>>>(d_out);
        cudaDeviceSynchronize();
        unsigned long long results[2];
        cudaMemcpy(results, d_out, 2*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        printf("  SM clock: %llu cycles, globaltimer: %llu ns\n", results[0], results[1]);
        printf("  Implied SM clock: %.1f MHz\n", (double)results[0] / results[1] * 1000);
    }

    return 0;
}
