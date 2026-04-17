// Test P2P kernel-side access between 2 B300s
// (kernel on GPU 0 reads/writes GPU 1's memory directly via NVLink)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void p2p_read(float *src_remote, float *dst_local, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0;
    for (int i = tid; i < N; i += stride) {
        acc += src_remote[i];
    }
    if (acc == -42.0f) dst_local[tid] = acc;
}

extern "C" __global__ void p2p_write(float *src_local, float *dst_remote, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        dst_remote[i] = src_local[i] * 2.0f;
    }
}

extern "C" __global__ void atomic_remote(unsigned int *remote_counter, int n_ops) {
    for (int i = 0; i < n_ops; i++)
        atomicAdd(remote_counter, 1);
}

int main() {
    CK(cudaSetDevice(0));
    cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1); cudaDeviceEnablePeerAccess(0, 0);

    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount, threads = 256;

    float *d0_in, *d1_buf;
    int N = 64 << 20;  // 256 MB
    cudaSetDevice(0); CK(cudaMalloc(&d0_in, N * sizeof(float)));
    cudaSetDevice(1); CK(cudaMalloc(&d1_buf, N * sizeof(float)));
    cudaSetDevice(0);

    cudaMemset(d0_in, 0x40, N * sizeof(float));
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

    // ===== Test 1: Local read (GPU 0 → GPU 0) =====
    float *d0_dst;
    cudaMalloc(&d0_dst, blocks * threads * sizeof(float));

    float t_local = bench([&]{
        p2p_read<<<blocks, threads, 0, s>>>(d0_in, d0_dst, N);
    });
    printf("# B300 P2P kernel-side access\n\n");
    printf("Local read (GPU 0 reads GPU 0):  %.3f ms (%.1f GB/s)\n",
           t_local, (size_t)N*4 / (t_local/1e3) / 1e9);

    // ===== Test 2: Remote read (GPU 0 reads GPU 1) =====
    float t_remote = bench([&]{
        p2p_read<<<blocks, threads, 0, s>>>(d1_buf, d0_dst, N);
    });
    printf("Remote read (GPU 0 reads GPU 1): %.3f ms (%.1f GB/s)\n",
           t_remote, (size_t)N*4 / (t_remote/1e3) / 1e9);

    // ===== Test 3: Remote write (GPU 0 writes GPU 1) =====
    float t_rwrite = bench([&]{
        p2p_write<<<blocks, threads, 0, s>>>(d0_in, d1_buf, N);
    });
    printf("Remote write (GPU 0 writes GPU 1): %.3f ms (%.1f GB/s)\n",
           t_rwrite, (size_t)N*4 / (t_rwrite/1e3) / 1e9);

    // ===== Test 4: Remote atomic =====
    {
        unsigned int *d1_counter;
        cudaSetDevice(1); cudaMalloc(&d1_counter, sizeof(unsigned int));
        cudaSetDevice(0);

        for (int n_ops : {100, 1000, 10000}) {
            cudaSetDevice(1); cudaMemset(d1_counter, 0, sizeof(unsigned int));
            cudaSetDevice(0);

            float t = bench([&]{
                atomic_remote<<<blocks, threads, 0, s>>>(d1_counter, n_ops);
            });
            // Each thread does n_ops atomics on remote counter
            int total_atomics = blocks * threads * n_ops;
            float gops = total_atomics / (t/1e3) / 1e9;
            printf("Remote atomicAdd (%d threads × %d ops): %.3f ms = %.2f Gatomic/s\n",
                   blocks*threads, n_ops, t, gops);
        }

        cudaSetDevice(1); cudaFree(d1_counter);
        cudaSetDevice(0);
    }

    cudaFree(d0_in);
    cudaFree(d0_dst);
    cudaSetDevice(1); cudaFree(d1_buf);
    cudaSetDevice(0);
    return 0;
}
