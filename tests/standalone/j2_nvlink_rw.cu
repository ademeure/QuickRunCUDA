// J2: NVLink read vs write asymmetry (per-link bandwidth)
// MODE 0: GPU 0 reads from GPU 1 (cudaMemcpy P2P, dev1→dev0)
// MODE 1: GPU 0 writes to GPU 1 (cudaMemcpy P2P, dev0→dev1)
// MODE 2: GPU 0 kernel-write to GPU 1 (uint4 store via peer pointer)
// MODE 3: GPU 0 kernel-read from GPU 1 (uint4 load via peer pointer)
#include <cuda_runtime.h>
#include <cstdio>

__global__ void write_peer(uint4* peer_buf, int n_uint4_per_thread) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    uint4 v = make_uint4(gtid, gtid+1, gtid+2, gtid+3);
    for (int i = 0; i < n_uint4_per_thread; i++) {
        peer_buf[i * total + gtid] = v;
    }
}

__global__ void read_peer(const uint4* peer_buf, uint4* sink, int n_uint4_per_thread) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0,0,0,0);
    for (int i = 0; i < n_uint4_per_thread; i++) {
        uint4 v = peer_buf[i * total + gtid];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0xDEADBEEF) sink[gtid] = acc;
}

int main() {
    size_t bytes = 256ull * 1024 * 1024;  // 256 MB
    int blocks = 296;
    int threads = 256;
    int total = blocks * threads;
    int n_uint4_per_thread = bytes / total / 16;
    size_t actual_bytes = (size_t)n_uint4_per_thread * total * 16;

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    uint4* gpu0_sink;
    cudaMalloc(&gpu0_sink, sizeof(uint4) * total);

    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);
    uint4* gpu1_buf;
    cudaMalloc(&gpu1_buf, actual_bytes);
    cudaMemset(gpu1_buf, 0, actual_bytes);

    cudaEvent_t s, e;
    cudaSetDevice(0);
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // === Test 1: cudaMemcpy P2P read (dev1 → dev0)
    void* dev0_temp;
    cudaMalloc(&dev0_temp, actual_bytes);
    cudaMemcpyPeer(dev0_temp, 0, gpu1_buf, 1, actual_bytes); // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    cudaMemcpyPeer(dev0_temp, 0, gpu1_buf, 1, actual_bytes);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float read_ms;
    cudaEventElapsedTime(&read_ms, s, e);
    printf("P2P cudaMemcpy READ  (dev1→dev0): %.3f ms = %.1f GB/s\n", read_ms, actual_bytes / read_ms / 1e6);

    // === Test 2: cudaMemcpy P2P write (dev0 → dev1)
    cudaMemcpyPeer(gpu1_buf, 1, dev0_temp, 0, actual_bytes); // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    cudaMemcpyPeer(gpu1_buf, 1, dev0_temp, 0, actual_bytes);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float write_ms;
    cudaEventElapsedTime(&write_ms, s, e);
    printf("P2P cudaMemcpy WRITE (dev0→dev1): %.3f ms = %.1f GB/s\n", write_ms, actual_bytes / write_ms / 1e6);

    // === Test 3: kernel write GPU 0 → GPU 1 (peer pointer)
    cudaSetDevice(0);
    write_peer<<<blocks, threads>>>(gpu1_buf, n_uint4_per_thread); // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    write_peer<<<blocks, threads>>>(gpu1_buf, n_uint4_per_thread);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float kw_ms;
    cudaEventElapsedTime(&kw_ms, s, e);
    printf("Kernel WRITE (peer ptr): %.3f ms = %.1f GB/s\n", kw_ms, actual_bytes / kw_ms / 1e6);

    // === Test 4: kernel read GPU 1 → GPU 0 (peer pointer)
    cudaSetDevice(0);
    read_peer<<<blocks, threads>>>(gpu1_buf, gpu0_sink, n_uint4_per_thread); // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    read_peer<<<blocks, threads>>>(gpu1_buf, gpu0_sink, n_uint4_per_thread);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float kr_ms;
    cudaEventElapsedTime(&kr_ms, s, e);
    printf("Kernel READ  (peer ptr): %.3f ms = %.1f GB/s\n", kr_ms, actual_bytes / kr_ms / 1e6);

    return 0;
}
