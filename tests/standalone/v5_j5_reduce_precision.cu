// V5 J5: Block reduction precision (pairwise vs serial vs Kahan)
// Sum 1024 floats; compare to FP64 reference
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifndef MODE
#define MODE 0
#endif

__global__ void serial_sum(float* in, float* out, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s = 0;
    for (int i = 0; i < n; i++) s += in[i];
    *out = s;
}

__global__ void pairwise_sum(float* in, float* out, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // Pairwise tree reduction (in registers / SMEM)
    __shared__ float buf[1024];
    for (int i = 0; i < n; i++) buf[i] = in[i];
    __syncthreads();
    for (int s = n / 2; s > 0; s >>= 1) {
        for (int i = 0; i < s; i++) buf[i] += buf[i + s];
    }
    *out = buf[0];
}

__global__ void kahan_sum(float* in, float* out, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s = 0, c = 0;
    for (int i = 0; i < n; i++) {
        float y = in[i] - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    *out = s;
}

int main() {
    cudaSetDevice(0);
    int n = 1024;

    float* dev_in;
    float* dev_out_serial;
    float* dev_out_pairwise;
    float* dev_out_kahan;
    cudaMallocManaged(&dev_in, n * sizeof(float));
    cudaMallocManaged(&dev_out_serial, sizeof(float));
    cudaMallocManaged(&dev_out_pairwise, sizeof(float));
    cudaMallocManaged(&dev_out_kahan, sizeof(float));

    // Adversarial: big initial value + many small adds where the small ones matter
    // sum should be 1e8 + 1024 * 1.0 = 1e8 + 1024
    dev_in[0] = 1e8f;
    for (int i = 1; i < n; i++) dev_in[i] = 1.0f;

    serial_sum<<<1, 1>>>(dev_in, dev_out_serial, n);
    pairwise_sum<<<1, 1024>>>(dev_in, dev_out_pairwise, n);
    kahan_sum<<<1, 1>>>(dev_in, dev_out_kahan, n);
    cudaDeviceSynchronize();

    // FP64 reference
    double ref = 0;
    for (int i = 0; i < n; i++) ref += dev_in[i];

    printf("Sum 1024 floats (mixed magnitude):\n");
    printf("  FP64 reference:   %.10g\n", ref);
    printf("  Serial FP32:      %.10g  err = %.6e\n", *dev_out_serial, fabs(*dev_out_serial - ref));
    printf("  Pairwise FP32:    %.10g  err = %.6e\n", *dev_out_pairwise, fabs(*dev_out_pairwise - ref));
    printf("  Kahan FP32:       %.10g  err = %.6e\n", *dev_out_kahan, fabs(*dev_out_kahan - ref));

    return 0;
}
