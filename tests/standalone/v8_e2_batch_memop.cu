// V8 E2: cuStreamBatchMemOp — batch multiple write/wait ops
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cuInit(0);
    CUcontext ctx; cuCtxGetCurrent(&ctx);

    unsigned int* sig;
    cudaMallocManaged((void**)&sig, 16 * sizeof(unsigned int));
    for (int i = 0; i < 16; i++) sig[i] = 0;
    CUdeviceptr dev_sig = (CUdeviceptr)sig;

    cudaStream_t s;
    cudaStreamCreate(&s);

    int N_RUNS = 1000;

    // Warmup
    cuStreamWriteValue32(s, dev_sig, 1, 0);
    cudaStreamSynchronize(s);

    // Test 1: individual cuStreamWriteValue × 4
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        cuStreamWriteValue32(s, dev_sig + 0, (unsigned)i, 0);
        cuStreamWriteValue32(s, dev_sig + 1, (unsigned)(i+1), 0);
        cuStreamWriteValue32(s, dev_sig + 2, (unsigned)(i+2), 0);
        cuStreamWriteValue32(s, dev_sig + 3, (unsigned)(i+3), 0);
    }
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::high_resolution_clock::now();
    double individual_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

    // Test 2: cuStreamBatchMemOp (4 ops in one call)
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        CUstreamBatchMemOpParams params[4];
        for (int k = 0; k < 4; k++) {
            params[k].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
            params[k].writeValue.address = dev_sig + k;
            params[k].writeValue.value = (unsigned)(i + k);
            params[k].writeValue.flags = 0;
        }
        cuStreamBatchMemOp(s, 4, params, 0);
    }
    cudaStreamSynchronize(s);
    auto t3 = std::chrono::high_resolution_clock::now();
    double batch_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_RUNS;

    printf("4× WriteValue per iter:\n");
    printf("  Individual cuStreamWriteValue: %.2f us/iter (%.2f us/op)\n", individual_us, individual_us / 4);
    printf("  cuStreamBatchMemOp (4 ops):    %.2f us/iter (%.2f us/op)\n", batch_us, batch_us / 4);
    printf("  Speedup: %.2fx\n", individual_us / batch_us);

    return 0;
}
