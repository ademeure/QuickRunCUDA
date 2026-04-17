// Surface objects: read-write from cudaArray
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void surf_write(cudaSurfaceObject_t s, int W, int H, float val) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        surf2Dwrite(val + x * 0.001f, s, x * sizeof(float), y);
    }
}

extern "C" __global__ void surf_read(cudaSurfaceObject_t s, float *out, int W, int H, int iters) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float a = 0;
    for (int i = 0; i < iters; i++) {
        float v;
        surf2Dread(&v, s, x * sizeof(float), y);
        a += v;
    }
    if (a < -1e30f) out[y * W + x] = a;
}

extern "C" __global__ void global_read(const float *data, float *out, int W, int H, int iters) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float a = 0;
    for (int i = 0; i < iters; i++) {
        a += data[y * W + x];
    }
    if (a < -1e30f) out[y * W + x] = a;
}

int main() {
    cudaSetDevice(0);
    int W = 4096, H = 4096;
    size_t bytes = (size_t)W * H * sizeof(float);

    // Create cudaArray + surface
    cudaArray_t arr;
    cudaChannelFormatDesc chan = cudaCreateChannelDesc<float>();
    cudaMallocArray(&arr, &chan, W, H, cudaArraySurfaceLoadStore);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;
    cudaSurfaceObject_t surf;
    cudaCreateSurfaceObject(&surf, &resDesc);

    // Linear memory for comparison
    float *d_lin; cudaMalloc(&d_lin, bytes);
    float *d_out; cudaMalloc(&d_out, bytes);
    cudaMemset(d_lin, 0, bytes);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 3; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    int iters = 10;
    dim3 block(32, 8);
    dim3 grid((W + 31) / 32, (H + 7) / 8);

    // Init surface
    surf_write<<<grid, block>>>(surf, W, H, 1.0f);
    cudaDeviceSynchronize();

    // Read benchmark
    float t_surf = bench([&]{ surf_read<<<grid, block>>>(surf, d_out, W, H, iters); });
    float t_glo = bench([&]{ global_read<<<grid, block>>>(d_lin, d_out, W, H, iters); });

    double bw_surf = (double)bytes * iters / (t_surf/1000) / 1e9;
    double bw_glo = (double)bytes * iters / (t_glo/1000) / 1e9;

    printf("# B300 surface object vs linear memory bandwidth\n");
    printf("# %d × %d float = %zu MB; %d iterations\n\n", W, H, bytes/1024/1024, iters);
    printf("  surface (read):  %.3f ms = %.0f GB/s\n", t_surf, bw_surf);
    printf("  linear (read):   %.3f ms = %.0f GB/s\n", t_glo, bw_glo);
    printf("  surface vs linear: %.2fx %s\n",
           bw_glo / bw_surf, bw_glo > bw_surf ? "LINEAR FASTER" : "SURFACE FASTER");

    return 0;
}
