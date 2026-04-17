// Test cudaLaunchAttributeNvlinkUtilCentricScheduling
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void p2p_read(float4 *src_remote, float *dst_local, int N4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float4 acc = make_float4(0,0,0,0);
    for (int i = tid; i < N4; i += stride) {
        float4 v = src_remote[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x + acc.y + acc.z + acc.w == -42.0f) dst_local[tid] = acc.x;
}

int main() {
    CK(cudaSetDevice(0));
    cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1); cudaDeviceEnablePeerAccess(0, 0);
    cudaSetDevice(0);

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;

    int N = 64 << 20;
    float *d0_dst, *d1_buf;
    cudaSetDevice(0); CK(cudaMalloc(&d0_dst, sm_count * 1024 * sizeof(float)));
    cudaSetDevice(1); CK(cudaMalloc(&d1_buf, N * sizeof(float)));
    cudaSetDevice(1); cudaMemset(d1_buf, 0x40, N * sizeof(float));
    cudaSetDevice(0);

    cudaStream_t s; CK(cudaStreamCreate(&s));

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    int N4 = N / 4;

    int b_arr[] = {148, 296, 592};
    int t_arr[] = {128, 256, 512, 1024};
    printf("# B300 NvlinkUtilCentricScheduling test (kernel reads remote GPU memory)\n");
    printf("# %-12s %-8s %-15s %-15s\n", "blocks×thr", "config", "no_hint_GB/s", "with_hint_GB/s");

    for (int b : b_arr) for (int t : t_arr) {
        float t_no = bench([&]{
            p2p_read<<<b, t, 0, s>>>((float4*)d1_buf, d0_dst, N4);
        });

        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeNvlinkUtilCentricScheduling;
        attr.val.nvlinkUtilCentricScheduling = 1;
        cudaLaunchConfig_t cfg = {dim3(b), dim3(t), 0, s, &attr, 1};

        float t_with = bench([&]{
            int n4 = N4;
            void *args[] = {&d1_buf, &d0_dst, &n4};
            cudaLaunchKernelExC(&cfg, (void*)p2p_read, args);
        });

        float bw_no = (size_t)N*4 / (t_no/1e3) / 1e9;
        float bw_with = (size_t)N*4 / (t_with/1e3) / 1e9;
        printf("  %dx%-8d %s  %.1f         %.1f\n",
               b, t, b*t < 75000 ? "small" : "big",
               bw_no, bw_with);
    }

    cudaFree(d0_dst);
    cudaSetDevice(1); cudaFree(d1_buf);
    cudaSetDevice(0);
    return 0;
}
