// utils/power_sampler.cpp — high-rate power sampler using NVML library directly
// Build: g++ -O2 -lnvidia-ml -ldl -o power_sampler power_sampler.cpp
// Usage: ./power_sampler [duration_sec] [target_hz]
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <nvml.h>

int main(int argc, char** argv) {
    int duration_sec = (argc > 1) ? atoi(argv[1]) : 1;
    int target_hz = (argc > 2) ? atoi(argv[2]) : 1000;

    nvmlReturn_t ret = nvmlInit();
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "nvmlInit failed: %s\n", nvmlErrorString(ret));
        return 1;
    }
    nvmlDevice_t dev;
    ret = nvmlDeviceGetHandleByIndex(0, &dev);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "nvmlDeviceGetHandleByIndex failed: %s\n", nvmlErrorString(ret));
        return 1;
    }

    // Sample at target rate
    int target_samples = duration_sec * target_hz;
    long long sleep_us = 1000000 / target_hz;
    if (sleep_us < 1) sleep_us = 1;

    long long unique_count = 0;
    long long total_count = 0;
    unsigned int last_mw = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < target_samples; i++) {
        unsigned int mw;
        ret = nvmlDeviceGetPowerUsage(dev, &mw);
        if (ret != NVML_SUCCESS) {
            fprintf(stderr, "GetPowerUsage failed: %s\n", nvmlErrorString(ret));
            break;
        }
        if (mw != last_mw) unique_count++;
        last_mw = mw;
        total_count++;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    double actual_rate = (double)total_count * 1e6 / elapsed_us;
    double unique_rate = (double)unique_count * 1e6 / elapsed_us;
    printf("NVML power sampler:\n");
    printf("  Target rate: %d Hz\n", target_hz);
    printf("  Actual samples: %lld in %.3f sec = %.1f Hz\n", total_count, elapsed_us / 1e6, actual_rate);
    printf("  Unique values: %lld = %.1f Hz unique rate\n", unique_count, unique_rate);
    printf("  Last power: %u mW = %.2f W\n", last_mw, last_mw / 1000.0);

    nvmlShutdown();
    return 0;
}
