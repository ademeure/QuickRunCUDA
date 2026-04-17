// Deep P2P kernel-side bandwidth investigation on B300 NVLink
// Sweep: thread count, block count, vector loads, ILP, L2 hit rate
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// Scalar load (4-byte LDG)
extern "C" __global__ void p2p_scalar(float *src_remote, float *dst_local, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0;
    for (int i = tid; i < N; i += stride) {
        acc += src_remote[i];
    }
    if (acc == -42.0f) dst_local[tid] = acc;
}

// Vector load 4 (float4 = 16-byte LDG.128)
extern "C" __global__ void p2p_vec4(float4 *src_remote, float *dst_local, int N4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float4 acc = make_float4(0,0,0,0);
    for (int i = tid; i < N4; i += stride) {
        float4 v = src_remote[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x + acc.y + acc.z + acc.w == -42.0f) dst_local[tid] = acc.x;
}

// Vector with ILP=4 (issue 4 outstanding loads per thread)
extern "C" __global__ void p2p_vec4_ilp4(float4 *src_remote, float *dst_local, int N4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float4 a0 = make_float4(0,0,0,0), a1 = a0, a2 = a0, a3 = a0;
    for (int i = tid; i < N4; i += stride * 4) {
        if (i + 0*stride < N4) { float4 v = src_remote[i + 0*stride]; a0.x += v.x; a0.y += v.y; }
        if (i + 1*stride < N4) { float4 v = src_remote[i + 1*stride]; a1.x += v.x; a1.y += v.y; }
        if (i + 2*stride < N4) { float4 v = src_remote[i + 2*stride]; a2.x += v.x; a2.y += v.y; }
        if (i + 3*stride < N4) { float4 v = src_remote[i + 3*stride]; a3.x += v.x; a3.y += v.y; }
    }
    float total = a0.x+a0.y+a1.x+a1.y+a2.x+a2.y+a3.x+a3.y;
    if (total == -42.0f) dst_local[tid] = total;
}

// LDG.128 with cache hint (.E - explicit, ca = cache-all = L1+L2, cg = L2 only)
extern "C" __global__ void p2p_ldg_cg(float *src_remote, float *dst_local, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0;
    for (int i = tid; i < N; i += stride) {
        float v;
        asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(&src_remote[i]));
        acc += v;
    }
    if (acc == -42.0f) dst_local[tid] = acc;
}

int main() {
    CK(cudaSetDevice(0));
    cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1); cudaDeviceEnablePeerAccess(0, 0);
    cudaSetDevice(0);

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;

    int N = 64 << 20;  // 256 MB worth of floats
    float *d0_in, *d1_buf, *d0_dst;
    cudaSetDevice(0); CK(cudaMalloc(&d0_in, N * sizeof(float)));
    cudaSetDevice(1); CK(cudaMalloc(&d1_buf, N * sizeof(float)));
    cudaSetDevice(0); CK(cudaMalloc(&d0_dst, sm_count * 1024 * sizeof(float)));

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

    printf("# B300 P2P kernel-side BW deep dive (256 MB remote read)\n");
    printf("# Reading from GPU 1 across NVLink into GPU 0 SMs\n\n");

    // ===== Test 1: Block × Thread sweep with scalar loads =====
    printf("## Scalar loads (LDG f32) — blocks × threads sweep\n");
    int b_arr[] = {37, 74, 148, 296, 592, 1184};
    int t_arr[] = {32, 64, 128, 256, 512, 1024};

    printf("%-10s ", "blocks↓ threads→");
    for (int t : t_arr) printf("%9d ", t);
    printf("\n");

    for (int b : b_arr) {
        printf("%-10d ", b);
        for (int t : t_arr) {
            if (b * t > 4 * 1024 * 1024) { printf("%9s ", "skip"); continue; }
            float ms = bench([&]{
                p2p_scalar<<<b, t, 0, s>>>(d1_buf, d0_dst, N);
            });
            float bw = (size_t)N * 4 / (ms / 1e3) / 1e9;
            printf("%9.0f ", bw);
        }
        printf("(GB/s)\n");
    }

    // ===== Test 2: Vector loads =====
    printf("\n## float4 vector loads (LDG.128) — blocks × threads sweep\n");
    printf("%-10s ", "blocks↓ threads→");
    for (int t : t_arr) printf("%9d ", t);
    printf("\n");

    for (int b : b_arr) {
        printf("%-10d ", b);
        for (int t : t_arr) {
            if (b * t > 4 * 1024 * 1024) { printf("%9s ", "skip"); continue; }
            float ms = bench([&]{
                p2p_vec4<<<b, t, 0, s>>>((float4*)d1_buf, d0_dst, N/4);
            });
            float bw = (size_t)N * 4 / (ms / 1e3) / 1e9;
            printf("%9.0f ", bw);
        }
        printf("(GB/s)\n");
    }

    // ===== Test 3: Vector + ILP4 =====
    printf("\n## float4 + ILP4 (4 outstanding loads/thread)\n");
    for (int b : b_arr) {
        printf("%-10d ", b);
        for (int t : t_arr) {
            if (b * t > 4 * 1024 * 1024) { printf("%9s ", "skip"); continue; }
            float ms = bench([&]{
                p2p_vec4_ilp4<<<b, t, 0, s>>>((float4*)d1_buf, d0_dst, N/4);
            });
            float bw = (size_t)N * 4 / (ms / 1e3) / 1e9;
            printf("%9.0f ", bw);
        }
        printf("(GB/s)\n");
    }

    // ===== Test 4: Compare to local memory peak =====
    printf("\n## Reference: local memory bandwidth (same kernels reading GPU 0 → GPU 0)\n");
    int b_local = 296, t_local = 512;
    {
        float t_scalar = bench([&]{
            p2p_scalar<<<b_local, t_local, 0, s>>>(d0_in, d0_dst, N);
        });
        float t_vec = bench([&]{
            p2p_vec4<<<b_local, t_local, 0, s>>>((float4*)d0_in, d0_dst, N/4);
        });
        float t_ilp = bench([&]{
            p2p_vec4_ilp4<<<b_local, t_local, 0, s>>>((float4*)d0_in, d0_dst, N/4);
        });
        printf("  scalar local:   %.1f GB/s\n", (size_t)N*4/(t_scalar/1e3)/1e9);
        printf("  float4 local:   %.1f GB/s\n", (size_t)N*4/(t_vec/1e3)/1e9);
        printf("  float4 ILP4:    %.1f GB/s\n", (size_t)N*4/(t_ilp/1e3)/1e9);
    }

    // ===== Test 5: cg cache hint vs default =====
    printf("\n## Cache hint comparison (296 blocks × 512 threads)\n");
    {
        float t_default = bench([&]{
            p2p_scalar<<<b_local, t_local, 0, s>>>(d1_buf, d0_dst, N);
        });
        float t_cg = bench([&]{
            p2p_ldg_cg<<<b_local, t_local, 0, s>>>(d1_buf, d0_dst, N);
        });
        printf("  scalar default: %.1f GB/s\n", (size_t)N*4/(t_default/1e3)/1e9);
        printf("  scalar .cg:     %.1f GB/s\n", (size_t)N*4/(t_cg/1e3)/1e9);
    }

    cudaFree(d0_in); cudaFree(d0_dst);
    cudaSetDevice(1); cudaFree(d1_buf);
    cudaSetDevice(0);
    return 0;
}
