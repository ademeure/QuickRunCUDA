// Multi-GPU NVLink test on 2× B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

int main() {
    int n_gpus;
    CK(cudaGetDeviceCount(&n_gpus));
    printf("# Found %d GPUs\n", n_gpus);

    if (n_gpus < 2) {
        printf("Need 2+ GPUs for this test\n");
        return 0;
    }

    // Print info per GPU
    for (int i = 0; i < n_gpus; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        printf("# GPU %d: %s, %d SMs, %.1f GB\n",
               i, p.name, p.multiProcessorCount, p.totalGlobalMem/(1024.f*1024.f*1024.f));
    }

    // Check P2P access
    int can_access_01, can_access_10;
    cudaDeviceCanAccessPeer(&can_access_01, 0, 1);
    cudaDeviceCanAccessPeer(&can_access_10, 1, 0);
    printf("# P2P 0→1: %s, P2P 1→0: %s\n", can_access_01?"YES":"NO", can_access_10?"YES":"NO");

    if (!can_access_01) return 1;

    // Enable peer access
    cudaSetDevice(0);
    CK(cudaDeviceEnablePeerAccess(1, 0));
    cudaSetDevice(1);
    CK(cudaDeviceEnablePeerAccess(0, 0));

    // Allocate on GPU 0 and GPU 1
    float *d0, *d1;
    size_t bytes = 256 << 20;  // 256 MB
    cudaSetDevice(0); CK(cudaMalloc(&d0, bytes));
    cudaSetDevice(1); CK(cudaMalloc(&d1, bytes));

    cudaSetDevice(0);
    cudaStream_t s; CK(cudaStreamCreate(&s));

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

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

    // ===== Test 1: P2P bandwidth (GPU 0 → GPU 1) =====
    printf("\n## P2P bandwidth (cudaMemcpyPeer)\n");
    {
        float t = bench([&]{
            cudaMemcpyPeerAsync(d1, 1, d0, 0, bytes, s);
        });
        printf("  GPU0→GPU1 (%zu MB): %.3f ms = %.1f GB/s\n",
               bytes >> 20, t, bytes / (t/1e3f) / 1e9f);
    }

    // ===== Test 2: P2P read latency (small transfers) =====
    printf("\n## P2P transfer latency vs size\n");
    int size_arr[] = {1024, 4096, 65536, (1<<20), (4<<20), (16<<20), (64<<20), (256<<20)};
    for (int sz : size_arr) {
        float t = bench([&]{
            cudaMemcpyPeerAsync(d1, 1, d0, 0, sz, s);
        });
        printf("  %-10d B (%-6.1f KB): %.3f ms = %.1f GB/s\n",
               sz, sz/1024.f, t, sz / (t/1e3f) / 1e9f);
    }

    // ===== Test 3: Simultaneous bidirectional =====
    printf("\n## Simultaneous bidirectional P2P\n");
    {
        cudaStream_t s2;
        cudaSetDevice(1); CK(cudaStreamCreate(&s2));

        cudaSetDevice(0);
        float t_uni = bench([&]{
            cudaMemcpyPeerAsync(d1, 1, d0, 0, bytes, s);
        });

        float t_bi = bench([&]{
            cudaMemcpyPeerAsync(d1, 1, d0, 0, bytes, s);
            cudaSetDevice(1);
            cudaMemcpyPeerAsync(d0, 0, d1, 1, bytes, s2);
            cudaSetDevice(0);
        });
        printf("  unidir: %.3f ms (%.1f GB/s)\n", t_uni, bytes/(t_uni/1e3)/1e9);
        printf("  bidir:  %.3f ms (%.1f GB/s aggregate, %.2fx unidir)\n",
               t_bi, 2*bytes/(t_bi/1e3)/1e9, t_uni/t_bi*2);

        cudaStreamDestroy(s2);
    }

    // ===== Test 4: cudaMemcpyAsync vs cudaMemcpyPeerAsync =====
    printf("\n## cudaMemcpyAsync (auto-detect peer) vs cudaMemcpyPeerAsync\n");
    {
        cudaSetDevice(0);
        float t_peer = bench([&]{
            cudaMemcpyPeerAsync(d1, 1, d0, 0, bytes, s);
        });
        float t_default = bench([&]{
            cudaMemcpyAsync(d1, d0, bytes, cudaMemcpyDefault, s);
        });
        printf("  cudaMemcpyPeerAsync: %.3f ms (%.1f GB/s)\n", t_peer, bytes/(t_peer/1e3)/1e9);
        printf("  cudaMemcpyAsync:     %.3f ms (%.1f GB/s)\n", t_default, bytes/(t_default/1e3)/1e9);
    }

    cudaSetDevice(0); cudaFree(d0);
    cudaSetDevice(1); cudaFree(d1);
    return 0;
}
