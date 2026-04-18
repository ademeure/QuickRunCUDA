// J3 RIGOR: VMM physical-mem aliasing — does access via alias A populate
// alias B's cache lookup?
//
// Theoretical: L1/L2 are PHYSICALLY tagged on B300 (typical for GPUs). Any
// alias access populates physical-cache, hits regardless of which virtual.
//
// Method: cuMemCreate one allocation, cuMemMap at TWO virtual addresses.
// Touch via A, time via B. If physical-tagged, B hits cache.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { const char *s; cuGetErrorString(e, &s); printf("ERR %d (%s) at %s:%d\n", (int)e, s, __FILE__, __LINE__); return 1; }} while(0)

__global__ void touch(unsigned *p, int n, unsigned *out) {
    unsigned acc = 0;
    for (int i = 0; i < n; i++) acc ^= p[i];
    if (acc == 0xdeadbeef) out[0] = acc;
}

__global__ void chase_lat(unsigned *p, unsigned *clk_out, int n, int mask) {
    unsigned acc = 0;
    unsigned long long t0 = clock64();
    for (int i = 0; i < n; i++) acc ^= p[i & mask];
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) {
        clk_out[0] = (unsigned)(t1 - t0);
        clk_out[1] = acc;
    }
}

int main() {
    // Use runtime API to create context
    cudaFree(0);
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxGetCurrent(&ctx);

    // 16 KB physical alloc
    size_t sz = 16ull * 1024 * 1024;  // 16 MB allocation, but probe with 16 KB working set

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;

    size_t gran;
    CHECK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    printf("# Allocation granularity: %zu B\n", gran);

    sz = ((sz + gran - 1) / gran) * gran;

    CUmemGenericAllocationHandle handle;
    CHECK(cuMemCreate(&handle, sz, &prop, 0));

    // Reserve 2 virtual ranges
    CUdeviceptr va_a, va_b;
    CHECK(cuMemAddressReserve(&va_a, sz, 0, 0, 0));
    CHECK(cuMemAddressReserve(&va_b, sz, 0, 0, 0));

    // Map both to the SAME physical handle
    CHECK(cuMemMap(va_a, sz, 0, handle, 0));
    CHECK(cuMemMap(va_b, sz, 0, handle, 0));

    CUmemAccessDesc adesc = {};
    adesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    adesc.location.id = 0;
    adesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK(cuMemSetAccess(va_a, sz, &adesc, 1));
    CHECK(cuMemSetAccess(va_b, sz, &adesc, 1));

    printf("# Aliases: A=0x%llx  B=0x%llx  (delta=%lld bytes)\n",
           (unsigned long long)va_a, (unsigned long long)va_b,
           (long long)va_b - (long long)va_a);

    cudaMemset((void*)va_a, 0xab, sz);

    unsigned *d_clk; cudaMalloc(&d_clk, 16);
    unsigned *d_out; cudaMalloc(&d_out, 4);

    int n_chase = 65536;
    int mask = (1 << 20) - 1;  // 1 MB working set in 4-B units

    // Flush L2 by streaming through 256 MB unrelated buffer
    void *flush_buf; cudaMalloc(&flush_buf, 256ull * 1024 * 1024);
    cudaMemset(flush_buf, 0xff, 256ull * 1024 * 1024);

    // Cold via A
    cudaDeviceSynchronize();
    chase_lat<<<1, 1>>>((unsigned*)va_a, d_clk, n_chase, mask);
    cudaDeviceSynchronize();
    unsigned cold_a; cudaMemcpy(&cold_a, d_clk, 4, cudaMemcpyDeviceToHost);

    // Warm via A again
    chase_lat<<<1, 1>>>((unsigned*)va_a, d_clk, n_chase, mask);
    cudaDeviceSynchronize();
    unsigned warm_a; cudaMemcpy(&warm_a, d_clk, 4, cudaMemcpyDeviceToHost);

    // Flush L2 then via B (to confirm it's actually L2-cached)
    cudaMemset(flush_buf, 0xee, 256ull * 1024 * 1024);
    cudaDeviceSynchronize();

    chase_lat<<<1, 1>>>((unsigned*)va_b, d_clk, n_chase, mask);
    cudaDeviceSynchronize();
    unsigned via_b_cold; cudaMemcpy(&via_b_cold, d_clk, 4, cudaMemcpyDeviceToHost);

    // Now warm via A again, then via B WITHOUT flushing
    chase_lat<<<1, 1>>>((unsigned*)va_a, d_clk, n_chase, mask);
    cudaDeviceSynchronize();
    chase_lat<<<1, 1>>>((unsigned*)va_b, d_clk, n_chase, mask);
    cudaDeviceSynchronize();
    unsigned via_b_after_a; cudaMemcpy(&via_b_after_a, d_clk, 4, cudaMemcpyDeviceToHost);

    printf("# Chase results (cycles for %d loads, 1 MB working set):\n", n_chase);
    printf("  Cold via A (after flush):       %u cy = %.1f ns/load\n",
           cold_a, (float)cold_a / n_chase / 1.92);
    printf("  Warm via A (re-access):         %u cy = %.1f ns/load\n",
           warm_a, (float)warm_a / n_chase / 1.92);
    printf("  Cold via B (after flush):       %u cy = %.1f ns/load\n",
           via_b_cold, (float)via_b_cold / n_chase / 1.92);
    printf("  Via B after warming via A:      %u cy = %.1f ns/load\n",
           via_b_after_a, (float)via_b_after_a / n_chase / 1.92);

    if (via_b_after_a < via_b_cold * 0.7) {
        printf("# PHYSICAL-TAGGED: via B benefits from A's cache fills\n");
    } else {
        printf("# Cold-via-B and warm-via-A-then-B similar: may be L1 effect or test issue\n");
    }

    cuMemUnmap(va_a, sz);
    cuMemUnmap(va_b, sz);
    cuMemAddressFree(va_a, sz);
    cuMemAddressFree(va_b, sz);
    cuMemRelease(handle);
    return 0;
}
