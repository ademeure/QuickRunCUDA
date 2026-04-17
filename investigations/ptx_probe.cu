// Probe which Blackwell PTX features sm_103a supports via static nvcc/ptxas
#include <cuda_runtime.h>
#include <cstdio>

// 1. mbarrier
extern "C" __global__ void probe_mbarrier() {
    __shared__ alignas(8) unsigned long long mbar;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "l"(&mbar));
    }
    __syncthreads();
    unsigned long long token;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];"
                 : "=l"(token) : "l"(&mbar));
}

// 2. TMA (cp.async.bulk)
extern "C" __global__ void probe_tma() {
    extern __shared__ char smem[];
    __shared__ alignas(8) unsigned long long mbar;
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "l"(smem), "l"((void*)0x1000), "r"(1024u), "l"(&mbar));
}

// 3. cp.async.bulk.tensor
extern "C" __global__ void probe_tma_tensor() {
    extern __shared__ char smem[];
    __shared__ alignas(8) unsigned long long mbar;
    // This needs a tensor map descriptor — probably won't compile without
}

// 4-7: REJECTED by ptxas sm_103a
// wgmma.fence — Hopper, rejected
// tcgen05.alloc — rejected (works via NVRTC!)
// tcgen05.fence — rejected
// barrier.cluster.sync — syntax error

// 8. ld.global.nc (non-coherent)
extern "C" __global__ void probe_ldnc(int *a) {
    int x;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(x) : "l"(a));
    if (threadIdx.x == 0 && x < 0) a[0] = x;
}

// 9. atomic.acq_rel.sys
extern "C" __global__ void probe_atomic_acq_rel_sys(int *a) {
    int v = 1;
    asm volatile("atom.acq_rel.sys.global.add.u32 %0, [%1], %2;"
                 : "=r"(v) : "l"(a), "r"(1));
    if (threadIdx.x == 0 && v < 0) a[0] = v;
}

// 10. atomic.seq_cst.sys — REJECTED (not supported on sm_103a)

int main() {
    printf("# Static nvcc sm_103a PTX feature support:\n");
    printf("# If the test compiled, the feature is ACCESSIBLE via static ptxas.\n");
    printf("# (Various features may still work via NVRTC even if static rejects.)\n");
    return 0;
}
