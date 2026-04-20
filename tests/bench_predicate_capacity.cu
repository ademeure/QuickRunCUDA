// Predicate register file capacity test.
// Allocate N PTX predicates, see if compiler spills (uses extra inst) or fails.
// SM has 7 PT registers historically (P0..P6 + PT). PTX-level allocation may
// be virtual but physical pressure shows as more inst.

#ifndef N_PRED
#define N_PRED 4
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)(threadIdx.x + (unsigned)u2);
    unsigned int a = (unsigned)threadIdx.x * 131;
    unsigned int b = (unsigned)threadIdx.x * 271;

    // Compute N predicates simultaneously and use them all
    asm volatile(
#if N_PRED == 4
        "{ .reg .pred p0,p1,p2,p3;"
        "  setp.gt.u32 p0, %1, 0;"
        "  setp.gt.u32 p1, %1, 1;"
        "  setp.gt.u32 p2, %1, 2;"
        "  setp.gt.u32 p3, %1, 3;"
        "  @p0 add.u32 %0, %0, 1;"
        "  @p1 add.u32 %0, %0, 2;"
        "  @p2 add.u32 %0, %0, 3;"
        "  @p3 add.u32 %0, %0, 4; }"
#elif N_PRED == 8
        "{ .reg .pred p0,p1,p2,p3,p4,p5,p6,p7;"
        "  setp.gt.u32 p0, %1, 0;"
        "  setp.gt.u32 p1, %1, 1;"
        "  setp.gt.u32 p2, %1, 2;"
        "  setp.gt.u32 p3, %1, 3;"
        "  setp.gt.u32 p4, %1, 4;"
        "  setp.gt.u32 p5, %1, 5;"
        "  setp.gt.u32 p6, %1, 6;"
        "  setp.gt.u32 p7, %1, 7;"
        "  @p0 add.u32 %0, %0, 1;"
        "  @p1 add.u32 %0, %0, 2;"
        "  @p2 add.u32 %0, %0, 3;"
        "  @p3 add.u32 %0, %0, 4;"
        "  @p4 add.u32 %0, %0, 5;"
        "  @p5 add.u32 %0, %0, 6;"
        "  @p6 add.u32 %0, %0, 7;"
        "  @p7 add.u32 %0, %0, 8; }"
#elif N_PRED == 16
        "{ .reg .pred p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15;"
        "  setp.gt.u32 p0, %1, 0;"
        "  setp.gt.u32 p1, %1, 1;"
        "  setp.gt.u32 p2, %1, 2;"
        "  setp.gt.u32 p3, %1, 3;"
        "  setp.gt.u32 p4, %1, 4;"
        "  setp.gt.u32 p5, %1, 5;"
        "  setp.gt.u32 p6, %1, 6;"
        "  setp.gt.u32 p7, %1, 7;"
        "  setp.gt.u32 p8, %1, 8;"
        "  setp.gt.u32 p9, %1, 9;"
        "  setp.gt.u32 p10, %1, 10;"
        "  setp.gt.u32 p11, %1, 11;"
        "  setp.gt.u32 p12, %1, 12;"
        "  setp.gt.u32 p13, %1, 13;"
        "  setp.gt.u32 p14, %1, 14;"
        "  setp.gt.u32 p15, %1, 15;"
        "  @p0 add.u32 %0, %0, 1;"
        "  @p1 add.u32 %0, %0, 2;"
        "  @p2 add.u32 %0, %0, 3;"
        "  @p3 add.u32 %0, %0, 4;"
        "  @p4 add.u32 %0, %0, 5;"
        "  @p5 add.u32 %0, %0, 6;"
        "  @p6 add.u32 %0, %0, 7;"
        "  @p7 add.u32 %0, %0, 8;"
        "  @p8 add.u32 %0, %0, 9;"
        "  @p9 add.u32 %0, %0, 10;"
        "  @p10 add.u32 %0, %0, 11;"
        "  @p11 add.u32 %0, %0, 12;"
        "  @p12 add.u32 %0, %0, 13;"
        "  @p13 add.u32 %0, %0, 14;"
        "  @p14 add.u32 %0, %0, 15;"
        "  @p15 add.u32 %0, %0, 16; }"
#endif
        : "+r"(v) : "r"(a));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
