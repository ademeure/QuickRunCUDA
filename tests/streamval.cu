// Driver-level stream signaling: cuStreamWaitValue / cuStreamWriteValue
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)
#define DK(c) do { CUresult r=(c); if(r!=CUDA_SUCCESS){ \
    const char *m; cuGetErrorString(r, &m); fprintf(stderr,"DRV %d: %s\n", r, m); exit(1);} } while(0)

extern "C" __global__ void compute(float *out, int iters, int k) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + k * 0.00001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    int can_64bit, can_nor;
    cuDeviceGetAttribute(&can_64bit, (CUdevice_attribute)122, 0);  // CAN_USE_64_BIT_STREAM_MEM_OPS
    cuDeviceGetAttribute(&can_nor, (CUdevice_attribute)123, 0);   // CAN_USE_STREAM_WAIT_VALUE_NOR
    printf("# B300 cuStreamWaitValue/cuStreamWriteValue support:\n");
    printf("#   64-bit memops:    %s\n", can_64bit ? "YES" : "NO");
    printf("#   WAIT_VALUE_NOR:   %s\n", can_nor ? "YES" : "NO");

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * sizeof(float)));
    CK(cudaMemset(d_out, 0, blocks * sizeof(float)));

    // Allocate signal flag - must be zero-init device memory
    unsigned int *d_flag;
    CK(cudaMalloc(&d_flag, sizeof(unsigned int)));
    CK(cudaMemset(d_flag, 0, sizeof(unsigned int)));

    cudaStream_t s0, s1;
    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) {
            CK(cudaMemset(d_flag, 0, sizeof(unsigned int)));
            fn();
            cudaDeviceSynchronize();
        }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            CK(cudaMemset(d_flag, 0, sizeof(unsigned int)));
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    int iters = 5000;

    // ===== Test 1: cuStreamWaitValue cross-stream sync =====
    printf("\n## Cross-stream sync via cuStreamWaitValue/WriteValue (vs event)\n");
    {
        // Method 1: Event-based sync
        cudaEvent_t evt; CK(cudaEventCreate(&evt));
        float t_event = bench([&]{
            compute<<<blocks,threads,0,s0>>>(d_out, iters, 0);
            cudaEventRecord(evt, s0);
            cudaStreamWaitEvent(s1, evt, 0);
            compute<<<blocks,threads,0,s1>>>(d_out, iters, 1);
        });
        printf("  event-based sync:        %.4f ms\n", t_event);

        // Method 2: cuStreamWriteValue + cuStreamWaitValue
        // s0: kernel + WriteValue=1
        // s1: WaitValue(value=1) + kernel
        float t_streamval = bench([&]{
            compute<<<blocks,threads,0,s0>>>(d_out, iters, 0);
            cuStreamWriteValue32((CUstream)s0, (CUdeviceptr)d_flag, 1, 0);
            cuStreamWaitValue32((CUstream)s1, (CUdeviceptr)d_flag, 1, CU_STREAM_WAIT_VALUE_GEQ);
            compute<<<blocks,threads,0,s1>>>(d_out, iters, 1);
        });
        printf("  StreamValue sync:        %.4f ms (vs event %+.4f)\n",
               t_streamval, t_streamval - t_event);

        // Method 3: cuStreamBatchMemOp (combine write+wait)
        // Set up the batch op
        // Producer + write atomically
        float t_batch = bench([&]{
            compute<<<blocks,threads,0,s0>>>(d_out, iters, 0);

            CUstreamBatchMemOpParams ops[1];
            ops[0].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
            ops[0].writeValue.address = (CUdeviceptr)d_flag;
            ops[0].writeValue.value = 1;
            ops[0].writeValue.flags = 0;
            cuStreamBatchMemOp((CUstream)s0, 1, ops, 0);

            CUstreamBatchMemOpParams wait_ops[1];
            wait_ops[0].operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
            wait_ops[0].waitValue.address = (CUdeviceptr)d_flag;
            wait_ops[0].waitValue.value = 1;
            wait_ops[0].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
            cuStreamBatchMemOp((CUstream)s1, 1, wait_ops, 0);

            compute<<<blocks,threads,0,s1>>>(d_out, iters, 1);
        });
        printf("  Batch MemOp sync:        %.4f ms\n", t_batch);

        cudaEventDestroy(evt);
    }

    // ===== Test 2: Producer-consumer with semaphore =====
    printf("\n## Stream WaitValue as a producer-consumer semaphore (fine-grain)\n");
    {
        // Producer writes flag to N, consumer waits for >= N
        for (int N : {1, 4, 16, 64, 256}) {
            CK(cudaMemset(d_flag, 0, sizeof(unsigned int)));
            CK(cudaDeviceSynchronize());

            float t = bench([&]{
                // Producer launches N kernels and writes flag
                for (int k = 0; k < N; k++) {
                    compute<<<1,32,0,s0>>>(d_out, iters/100, k);
                }
                cuStreamWriteValue32((CUstream)s0, (CUdeviceptr)d_flag, N, 0);

                // Consumer waits for flag >= N
                cuStreamWaitValue32((CUstream)s1, (CUdeviceptr)d_flag, N, CU_STREAM_WAIT_VALUE_GEQ);
                compute<<<1,32,0,s1>>>(d_out, iters/100, 999);
            });
            printf("  N=%-4d producer kernels + sync: %.4f ms\n", N, t);
        }
    }

    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);
    cudaFree(d_out);
    cudaFree(d_flag);
    return 0;
}
