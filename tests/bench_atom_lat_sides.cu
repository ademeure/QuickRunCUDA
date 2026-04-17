// Measure atom.global.add.u32 round-trip latency at a specific address offset,
// to expose near-side vs far-side L2 partition latency on B300.
//
// Usage (caller): pass the byte offset into A via arg `-1` (seed), ITERS via `-0`.
//   ./QuickRunCUDA tests/bench_atom_lat_sides.cu -t 1 -b 1 -0 <ITERS> -1 <OFFSET_BYTES>
//
// The kernel does a serial atom.add chain on (A + offset_bytes/4), dependency through
// the return value so each atomic must complete before the next issues.
// C[0:1] = cycle count, C[2] = final accumulator.
//
// B300 has 2 L2 partitions. L2-side hashing flips roughly every 4KB. If an address
// lands on the SM's "near" side, latency is low; on the "far" side, it's higher due
// to XBAR cross. Sweep offsets 0..many MB in caller.
extern "C" __global__ __launch_bounds__(1, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int ITERS, int offset_bytes, int u2) {
    // Convert byte offset to u32 element offset
    unsigned* p = (unsigned*)((char*)A + (size_t)(unsigned)offset_bytes);
    unsigned v = 1u;
    unsigned long long t0, t1;

    // Warm up: touch once so the line is resident (still we want L2 hit, not compulsory miss)
    asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v) : "l"(p), "r"(v));

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned r;
        asm volatile("atom.global.add.u32 %0, [%1], %2;"
                     : "=r"(r) : "l"(p), "r"(v));
        v = r + 1;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    ((unsigned long long*)C)[0] = t1 - t0;
    C[2] = v;
}
