// Try tcgen05.alloc via NVRTC (which QuickRunCUDA uses) — different from static ptxas
extern "C" __global__  void kernel(float *A, float *B, float *C, int arg0, int arg1, int arg2) {
    if (threadIdx.x != 0) return;

    __shared__ unsigned int tmem_addr;

    asm volatile(
        "{ .reg .b32 col;\n\t"
        "  mov.u32 col, 32;\n\t"
        "  tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], col;\n\t"
        "}\n"
        :: "l"((unsigned long)&tmem_addr) : "memory"
    );

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("TMEM alloc: addr = 0x%x\n", tmem_addr);
    }

    asm volatile(
        "{ .reg .b32 col, off;\n\t"
        "  ld.shared.u32 off, [%0];\n\t"
        "  mov.u32 col, 32;\n\t"
        "  tcgen05.dealloc.cta_group::1.sync.aligned.b32 off, col;\n\t"
        "}\n"
        :: "l"((unsigned long)&tmem_addr) : "memory"
    );
}
