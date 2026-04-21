// L4: cuLaunchKernel vs cuLaunchKernelEx
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void empty() {}

extern "C" int main() {
    cuInit(0);

    CUcontext ctx;
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    cuDevicePrimaryCtxRetain(&ctx, dev);
    cuCtxSetCurrent(ctx);

    // Get function handle
    CUfunction func;
    CUmodule mod;
    // Use the runtime kernel launch first to ensure module loaded
    empty<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Use cudaGetFuncBySymbol or cudaModuleGetFunction equivalent
    // Easier: use cudaLaunchKernel directly which goes through driver
    cudaFuncAttributes attrs;
    cudaFuncGetAttributes(&attrs, (void*)empty);

    int N = 10000;

    // Test 1: cudaLaunchKernel (runtime API)
    cudaStream_t s;
    cudaStreamCreate(&s);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaLaunchKernel((void*)empty, dim3(1), dim3(1), nullptr, 0, s);
    }
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ns_per = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;
    printf("cudaLaunchKernel: %.1f ns/launch\n", ns_per);

    // Test 2: cudaLaunchKernelEx (with attributes)
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeIgnore;  // null attr just to use Ex API
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = s;
    cfg.numAttrs = 0;  // no attrs
    cfg.attrs = nullptr;

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaLaunchKernelEx(&cfg, empty);
    }
    cudaStreamSynchronize(s);
    auto t3 = std::chrono::high_resolution_clock::now();
    double ns_per_ex = std::chrono::duration<double, std::nano>(t3 - t2).count() / N;
    printf("cudaLaunchKernelEx (no attrs): %.1f ns/launch\n", ns_per_ex);

    // Test 3: cudaLaunchKernelEx with cluster dim attr
    cudaLaunchAttribute attr_cluster;
    attr_cluster.id = cudaLaunchAttributeClusterDimension;
    attr_cluster.val.clusterDim.x = 1;
    attr_cluster.val.clusterDim.y = 1;
    attr_cluster.val.clusterDim.z = 1;
    cfg.numAttrs = 1;
    cfg.attrs = &attr_cluster;

    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaLaunchKernelEx(&cfg, empty);
    }
    cudaStreamSynchronize(s);
    auto t5 = std::chrono::high_resolution_clock::now();
    double ns_per_cluster = std::chrono::duration<double, std::nano>(t5 - t4).count() / N;
    printf("cudaLaunchKernelEx (1 attr): %.1f ns/launch\n", ns_per_cluster);

    return 0;
}
