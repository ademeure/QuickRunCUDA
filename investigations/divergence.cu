// Branch divergence cost on B300
#include <cuda_runtime.h>
#include <cstdio>

#define ITERS 10000

// No divergence - all threads take same path
extern "C" __global__ void no_div(int *out) {
    int tid = threadIdx.x;
    float a = (float)tid;
    unsigned long long start, end;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        if (i < ITERS) {  // always true, no div
            a = a * 1.0001f + 0.0001f;
        } else {
            a = a * 1.0001f - 0.0001f;
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[0] = (int)a;
        out[1] = (int)(end - start);
    }
}

// 2-way divergence - half of warp each way
extern "C" __global__ void div_2way(int *out) {
    int tid = threadIdx.x;
    float a = (float)tid;
    unsigned long long start, end;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        if (tid & 1) {
            a = a * 1.0001f + 0.0001f;
        } else {
            a = a * 1.0002f + 0.0002f;
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[0] = (int)a;
        out[1] = (int)(end - start);
    }
}

// 4-way divergence
extern "C" __global__ void div_4way(int *out) {
    int tid = threadIdx.x;
    float a = (float)tid;
    unsigned long long start, end;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        switch (tid & 3) {
            case 0: a = a * 1.0001f + 0.0001f; break;
            case 1: a = a * 1.0002f + 0.0002f; break;
            case 2: a = a * 1.0003f + 0.0003f; break;
            case 3: a = a * 1.0004f + 0.0004f; break;
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[0] = (int)a;
        out[1] = (int)(end - start);
    }
}

// 32-way divergence (each lane different)
extern "C" __global__ void div_32way(int *out) {
    int tid = threadIdx.x;
    float a = (float)tid;
    unsigned long long start, end;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        switch (tid & 31) {
            case 0: a = a * 1.001f; break;
            case 1: a = a + 0.001f; break;
            case 2: a = a * 2.0f; break;
            case 3: a = a / 3.0f; break;
            case 4: a = a - 1.0f; break;
            case 5: a = a + 2.0f; break;
            case 6: a = a * 0.5f; break;
            case 7: a = a + 3.0f; break;
            default:
                a = a + (float)(tid & 31) * 0.01f;
                break;
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[0] = (int)a;
        out[1] = (int)(end - start);
    }
}

int main() {
    cudaSetDevice(0);
    int *d_out;
    cudaMalloc(&d_out, 4 * sizeof(int));

    auto run_test = [&](const char *name, void (*fn)(int*)) {
        fn<<<1, 32>>>(d_out);
        cudaDeviceSynchronize();
        int h[2];
        cudaMemcpy(h, d_out, 2*sizeof(int), cudaMemcpyDeviceToHost);
        printf("  %-12s : %d cy / %d iters = %.2f cy/iter\n",
               name, h[1], ITERS, (double)h[1]/ITERS);
    };

    printf("# B300 branch divergence cost (1 warp)\n\n");
    run_test("no div",    no_div);
    run_test("2-way div", div_2way);
    run_test("4-way div", div_4way);
    run_test("32-way div", div_32way);

    cudaFree(d_out);
    return 0;
}
