// NVML query overhead
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    nvmlReturn_t nr = nvmlInit_v2();
    if (nr != NVML_SUCCESS) { printf("init: %s\n", nvmlErrorString(nr)); return 1; }

    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex_v2(0, &dev);

    auto bench = [&](auto fn, int trials = 1000) {
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

    printf("# B300 NVML query overhead\n\n");

    // Various queries
    {
        unsigned int rate;
        printf("  nvmlDeviceGetClockInfo (SM):  %.2f us\n",
               bench([&]{ nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &rate); }));
    }
    {
        unsigned int p;
        printf("  nvmlDeviceGetPowerUsage:      %.2f us\n",
               bench([&]{ nvmlDeviceGetPowerUsage(dev, &p); }));
    }
    {
        unsigned int t;
        printf("  nvmlDeviceGetTemperature:     %.2f us\n",
               bench([&]{ nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &t); }));
    }
    {
        nvmlMemory_t m;
        printf("  nvmlDeviceGetMemoryInfo:      %.2f us\n",
               bench([&]{ nvmlDeviceGetMemoryInfo(dev, &m); }));
    }
    {
        nvmlUtilization_t u;
        printf("  nvmlDeviceGetUtilizationRates:%.2f us\n",
               bench([&]{ nvmlDeviceGetUtilizationRates(dev, &u); }));
    }
    {
        nvmlPstates_t p;
        printf("  nvmlDeviceGetPowerState:      %.2f us\n",
               bench([&]{ nvmlDeviceGetPowerState(dev, &p); }));
    }
    {
        unsigned int u;
        printf("  nvmlDeviceGetEnforcedPowerLimit: %.2f us\n",
               bench([&]{ nvmlDeviceGetEnforcedPowerLimit(dev, &u); }));
    }
    {
        char name[128];
        printf("  nvmlDeviceGetName:            %.2f us\n",
               bench([&]{ nvmlDeviceGetName(dev, name, sizeof(name)); }));
    }
    {
        unsigned int rates[3];
        printf("  nvmlDeviceGetCurrPcieLinkGeneration: %.2f us\n",
               bench([&]{ nvmlDeviceGetCurrPcieLinkGeneration(dev, rates); }));
    }

    // What if we query very fast - is there caching?
    {
        unsigned int rate;
        printf("\n  nvmlDeviceGetClockInfo back-to-back 1000x:\n");
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; i++) nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &rate);
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1-t0).count();
        printf("    Total %.0f us = %.2f us/call\n", us, us/1000);
    }

    nvmlShutdown();
    return 0;
}
