// N=2 atomic anomaly investigation: vary offset between the two addresses.
// Catalog L8466 claims N=2 hotspot is 20× worse than N=1 — speculates "both
// addresses hash to same L2 slice". This test sweeps OFFSET between the 2
// addresses to map the address→partition hash function.
//
// Setup: 148 CTAs × 128 threads. Each thread atomicAdds to one of 2 addresses,
// chosen by (blockIdx & 1). Address pair = (base, base + OFFSET_BYTES).
// Measure aggregate throughput.
//
// -H "#define OFFSET_BYTES <n>"  -- offset between the two addresses (in BYTES)
// -H "#define ITERS <n>"          -- atomicAdds per thread (default 1000)

#ifndef OFFSET_BYTES
#define OFFSET_BYTES 4
#endif
#ifndef ITERS
#define ITERS 1000
#endif

extern "C" __global__ __launch_bounds__(128, 1)
void kernel(unsigned int* A, float* B, float* C, int seed, int u1, int u2) {
    // Use slot 0 (base) and slot OFFSET_BYTES/4 (base + OFFSET_BYTES bytes)
    unsigned int* base = A;
    unsigned int* addr0 = base;
    unsigned int* addr1 = base + (OFFSET_BYTES / 4);

    // Each CTA picks one of the two addresses (alternating)
    unsigned int* myaddr = (blockIdx.x & 1) ? addr1 : addr0;

    unsigned int acc = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // atomicAdd with return value (forces ATOM not REDG)
        acc += atomicAdd(myaddr, 1u);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    // Anti-DCE
    if (acc == 0xDEADBEEF) C[blockIdx.x] = (float)acc;

    // Each CTA leader writes its cycle count
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[1024 + blockIdx.x] = t1 - t0;
    }
}
