// NVTX range overhead
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials = 1000) {
        for (int i = 0; i < 5; i++) fn();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ns = std::chrono::duration<float, std::nano>(t1-t0).count();
            if (ns < best) best = ns;
        }
        return best;
    };

    printf("# B300 NVTX overhead (when no profiler attached)\n\n");

    {
        float t = bench([&]{ nvtxRangePushA("test"); nvtxRangePop(); });
        printf("  nvtxRangePushA + Pop:        %.0f ns\n", t);
    }
    {
        float t = bench([&]{ nvtxRangeId_t r = nvtxRangeStartA("t"); nvtxRangeEnd(r); });
        printf("  nvtxRangeStart + End:        %.0f ns\n", t);
    }
    {
        float t = bench([&]{ nvtxMarkA("mark"); });
        printf("  nvtxMarkA:                   %.0f ns\n", t);
    }
    {
        nvtxEventAttributes_t ev = {0};
        ev.version = NVTX_VERSION;
        ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        ev.colorType = NVTX_COLOR_ARGB;
        ev.color = 0xFF00FF00;
        ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
        ev.message.ascii = "test";
        float t = bench([&]{ nvtxRangePushEx(&ev); nvtxRangePop(); });
        printf("  nvtxRangePushEx + Pop:       %.0f ns\n", t);
    }

    // Compare to bare CPU clock
    {
        volatile int x = 0;
        float t = bench([&]{ x++; });
        printf("\n  (baseline: int++):           %.0f ns\n", t);
    }

    return 0;
}
