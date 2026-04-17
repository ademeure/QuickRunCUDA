// Exotic integer ops: umulhi, sad, viaddmin etc
#include <cuda_runtime.h>
#include <cstdio>

template<typename Op>
__global__ void run_op(unsigned *out, int iters, unsigned k, Op op) {
    unsigned a = threadIdx.x + 1;
    unsigned b = threadIdx.x * 7 + 1;
    unsigned c = threadIdx.x * 13 + 1;
    unsigned d = threadIdx.x * 31 + 1;
    for (int i = 0; i < iters; i++) {
        a = op(a, k); b = op(b, k); c = op(c, k); d = op(d, k);
    }
    if (a + b + c + d == 0xdeadbeef) out[blockIdx.x] = a + b + c + d;
}

int main() {
    cudaSetDevice(0);
    unsigned *d_out; cudaMalloc(&d_out, 1024 * sizeof(unsigned));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 128;
    long total_ops = (long)blocks * threads * iters * 4;
    unsigned k = 17;

    auto bench = [&](auto launch, const char *name) {
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
        double gops = total_ops / (best/1000.0) / 1e9;
        printf("  %-30s %.3f ms  %.0f Gops/s\n", name, best, gops);
    };

    printf("# B300 exotic integer ops (4-ILP, 148 × 128 thr)\n\n");

    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters, k,
        [] __device__ (unsigned x, unsigned y) { return __umulhi(x, y); }); }, "umulhi");
    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters, k,
        [] __device__ (unsigned x, unsigned y) { return __usad(x, y, 0); }); }, "usad (|x-y|)");
    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters, k,
        [] __device__ (unsigned x, unsigned y) {
            unsigned r;
            asm("vmad.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(y), "r"(0u));
            return r;
        }); }, "vmad u32");
    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters, k,
        [] __device__ (unsigned x, unsigned y) {
            unsigned r;
            asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(y), "r"(7u));
            return r;
        }); }, "shf.l.wrap (funnel)");
    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters, k,
        [] __device__ (unsigned x, unsigned y) {
            unsigned r;
            asm("lop3.b32 %0, %1, %2, %3, 0xa6;" : "=r"(r) : "r"(x), "r"(y), "r"(0xdeadbeefu));
            return r;
        }); }, "lop3 (3-input boolean)");
    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters, k,
        [] __device__ (unsigned x, unsigned y) {
            unsigned r;
            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(y), "r"(0x4321u));
            return r;
        }); }, "prmt (byte permute)");

    return 0;
}
