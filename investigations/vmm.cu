// Virtual Memory Management API: cuMemCreate / cuMemMap / cuMemSetAccess
// Goal: characterize cost of each step + verify access modes
#include <cuda.h>
#include <cstdio>
#include <chrono>
#include <vector>

int main() {
    cudaSetDevice(0);
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);

    auto bench_n = [&](auto fn, int trials = 100) {
        for (int i = 0; i < 3; i++) fn();
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

    printf("# B300 VMM API: per-step costs vs allocation size\n\n");
    printf("# %-12s %-12s %-12s %-12s %-12s %-12s\n",
           "size", "alloc_us", "reserve_us", "map_us", "setacc_us", "total_us");

    // Get allocation granularity
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;

    size_t gran;
    cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    printf("# Minimum granularity: %zu B (= %zu KB)\n", gran, gran/1024);

    cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    printf("# Recommended granularity: %zu B (= %zu KB)\n\n", gran, gran/1024);

    for (size_t mb : {2, 16, 256, 1024}) {
        size_t sz = mb * 1024 * 1024;

        // Step 1: cuMemCreate (physical alloc)
        float t_alloc = bench_n([&]{
            CUmemGenericAllocationHandle h;
            cuMemCreate(&h, sz, &prop, 0);
            cuMemRelease(h);
        }, 30);

        // Step 2: cuMemAddressReserve (virtual address)
        float t_reserve = bench_n([&]{
            CUdeviceptr ptr;
            cuMemAddressReserve(&ptr, sz, 0, 0, 0);
            cuMemAddressFree(ptr, sz);
        }, 100);

        // Step 3: cuMemMap (associate VA with physical)
        // Need to keep handle and reservation alive
        CUmemGenericAllocationHandle h;
        cuMemCreate(&h, sz, &prop, 0);
        CUdeviceptr ptr;
        cuMemAddressReserve(&ptr, sz, 0, 0, 0);

        float t_map = bench_n([&]{
            cuMemMap(ptr, sz, 0, h, 0);
            cuMemUnmap(ptr, sz);
        }, 30);

        // Step 4: cuMemSetAccess
        cuMemMap(ptr, sz, 0, h, 0);
        CUmemAccessDesc desc = {};
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id = 0;
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        float t_acc = bench_n([&]{
            cuMemSetAccess(ptr, sz, &desc, 1);
        }, 100);

        cuMemUnmap(ptr, sz);
        cuMemAddressFree(ptr, sz);
        cuMemRelease(h);

        printf("  %-12zu %-12.1f %-12.1f %-12.1f %-12.1f %-12.1f\n",
               sz, t_alloc, t_reserve, t_map, t_acc,
               t_alloc + t_reserve + t_map + t_acc);
    }

    // Test growable allocation: reserve big virtual range, map physical chunks one by one
    printf("\n## Growable allocation pattern: 4 MB physical chunks into 1 GB VA reservation\n");
    {
        size_t chunk = 4 * 1024 * 1024;
        size_t total = 1024 * 1024 * 1024;
        CUdeviceptr ptr;
        cuMemAddressReserve(&ptr, total, 0, 0, 0);

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<CUmemGenericAllocationHandle> handles;
        for (size_t off = 0; off < total; off += chunk) {
            CUmemGenericAllocationHandle h;
            cuMemCreate(&h, chunk, &prop, 0);
            cuMemMap(ptr + off, chunk, 0, h, 0);
            handles.push_back(h);
        }
        CUmemAccessDesc desc = {};
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id = 0;
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuMemSetAccess(ptr, total, &desc, 1);
        auto t1 = std::chrono::high_resolution_clock::now();

        float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
        printf("  Total grow: %.2f ms = %zu chunks × %.1f us each\n",
               ms, handles.size(), ms*1000/handles.size());

        for (auto h : handles) cuMemRelease(h);
        cuMemUnmap(ptr, total);
        cuMemAddressFree(ptr, total);
    }

    return 0;
}
