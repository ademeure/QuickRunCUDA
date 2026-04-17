#include <cuda.h>
#include <cstdio>
#include <chrono>

int main() {
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);

    auto bench = [&](auto fn, int trials = 100) {
        for (int i = 0; i < 5; i++) fn();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("# B300 CUDA context APIs\n\n");

    // Primary context retain/release
    {
        // First retain to make sure it exists
        CUcontext ctx;
        cuDevicePrimaryCtxRetain(&ctx, dev);
        cuDevicePrimaryCtxRelease(dev);

        float t = bench([&]{
            CUcontext c;
            cuDevicePrimaryCtxRetain(&c, dev);
            cuDevicePrimaryCtxRelease(dev);
        });
        printf("  PrimaryCtx Retain+Release:  %.2f us\n", t);
    }

    // Just GetCurrent
    {
        CUcontext c;
        cuDevicePrimaryCtxRetain(&c, dev);
        cuCtxSetCurrent(c);

        float t = bench([&]{
            CUcontext cc;
            cuCtxGetCurrent(&cc);
        }, 1000);
        printf("  cuCtxGetCurrent:            %.2f us\n", t);

        cuDevicePrimaryCtxRelease(dev);
    }

    // Push/pop
    {
        CUcontext c;
        cuDevicePrimaryCtxRetain(&c, dev);
        // Pop it
        CUcontext popped;
        cuCtxPopCurrent(&popped);

        float t = bench([&]{
            cuCtxPushCurrent(c);
            CUcontext p;
            cuCtxPopCurrent(&p);
        }, 200);
        printf("  cuCtxPush + cuCtxPop:       %.2f us\n", t);

        cuDevicePrimaryCtxRelease(dev);
    }

    return 0;
}
