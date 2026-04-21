// V6 K3: cudaLaunchCooperativeKernelMultiDevice — still supported on B300?
#include <cuda_runtime.h>
#include <cstdio>

__global__ void multi_dev_kernel(unsigned int* buf) {
    if (threadIdx.x == 0) buf[0] = blockIdx.x;
}

int main() {
    int n_devices;
    cudaGetDeviceCount(&n_devices);
    printf("Devices: %d\n", n_devices);
    if (n_devices < 2) {
        printf("Need 2+ devices for multi-device test\n");
        return 0;
    }

    // Try the deprecated cudaLaunchCooperativeKernelMultiDevice API
    cudaLaunchParams params[2];
    unsigned int* dev_bufs[2];

    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaMalloc(&dev_bufs[i], 1024);
        params[i].func = (void*)multi_dev_kernel;
        params[i].gridDim = dim3(2);
        params[i].blockDim = dim3(32);
        params[i].sharedMem = 0;
        params[i].stream = 0;
        static void* args[2][1];  // per-device args
        args[i][0] = &dev_bufs[i];
        params[i].args = args[i];
    }

    cudaError_t err = cudaLaunchCooperativeKernelMultiDevice(params, 2, 0);
    if (err != cudaSuccess) {
        printf("cudaLaunchCooperativeKernelMultiDevice ERROR: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaLaunchCooperativeKernelMultiDevice OK\n");
    }

    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaFree(dev_bufs[i]);
    }
    return 0;
}
