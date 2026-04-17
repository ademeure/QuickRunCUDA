// NVTX overhead measurement
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);

    const int N = 100000;

    // Cost of nvtxRangePush/Pop
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            nvtxRangePush("test");
            nvtxRangePop();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("nvtxRangePush/Pop: %.2f ns/pair\n",
               std::chrono::duration<float, std::nano>(t1-t0).count() / N);
    }

    // Cost of nvtxMark
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) nvtxMark("mark");
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("nvtxMark:          %.2f ns/call\n",
               std::chrono::duration<float, std::nano>(t1-t0).count() / N);
    }

    // RangeStart/End (with returned id)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            nvtxRangeId_t id = nvtxRangeStart("range");
            nvtxRangeEnd(id);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("nvtxRangeStart/End: %.2f ns/pair\n",
               std::chrono::duration<float, std::nano>(t1-t0).count() / N);
    }

    return 0;
}
