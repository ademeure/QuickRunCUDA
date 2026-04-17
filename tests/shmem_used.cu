// What ACTUALLY uses the reserved 1 KiB?
// Read bytes before/after cluster.sync(), mbarrier ops, etc.
#include <cuda_runtime.h>
#include <cstdio>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/std/utility>

namespace cg = cooperative_groups;

extern "C" __global__ void __cluster_dims__(2,1,1) probe_cluster_sync() {
    if (threadIdx.x != 0) return;
    auto cluster = cg::this_cluster();

    if (blockIdx.x == 0) {
        unsigned int v_before, off0 = 0;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v_before) : "r"(off0));
        printf("Block %d: before cluster.sync(), shmem[0]=0x%x\n", blockIdx.x, v_before);
    }

    cluster.sync();

    if (blockIdx.x == 0) {
        unsigned int v_after, off1 = 0;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v_after) : "r"(off1));
        printf("Block %d: after cluster.sync(), shmem[0]=0x%x\n", blockIdx.x, v_after);

        // Read first 16 words to see pattern
        for (int i = 0; i < 16; i++) {
            unsigned int v;
            unsigned int o = i * 4;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(o));
            printf("  [0x%03x] = 0x%08x\n", o, v);
        }
    }
}

extern "C" __global__ void probe_mbarrier_init() {
    if (threadIdx.x != 0) return;

    // Print the first 16 words BEFORE doing anything
    printf("Before any op:\n");
    for (int i = 0; i < 16; i++) {
        unsigned int v;
        unsigned int o = i * 4;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(o));
        printf("  [0x%03x] = 0x%08x\n", o, v);
    }
}

// Use mbarrier explicitly
extern "C" __global__ void probe_with_mbar() {
    if (threadIdx.x == 0) {
        // Initialize an mbarrier in shmem at offset 1024 (start of user shmem)
        __shared__ alignas(8) unsigned long long mbar_obj;
        unsigned long long *mbar_ptr = &mbar_obj;
        // Init mbarrier with 1 thread
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((unsigned)((unsigned long)mbar_ptr & 0xFFFFFF)), "r"(1u));

        // Read reserved space before arrive
        printf("Before mbarrier.arrive:\n");
        for (int i = 0; i < 16; i++) {
            unsigned int v;
            unsigned int o = i * 4;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(o));
            printf("  [0x%03x] = 0x%08x\n", o, v);
        }

        unsigned long long token;
        asm volatile("mbarrier.arrive.shared.b64 %0, [%1];"
                     : "=l"(token) : "r"((unsigned)((unsigned long)mbar_ptr & 0xFFFFFF)));

        // Read after arrive
        printf("\nAfter mbarrier.arrive (token=0x%llx):\n", token);
        for (int i = 0; i < 16; i++) {
            unsigned int v;
            unsigned int o = i * 4;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(o));
            printf("  [0x%03x] = 0x%08x\n", o, v);
        }
    }
}

int main() {
    cudaSetDevice(0);

    printf("# Probe: simple kernel reserved bytes (no cluster, no mbarrier)\n");
    probe_mbarrier_init<<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("\n# Probe: kernel with __cluster_dims__ + cluster.sync()\n");
    probe_cluster_sync<<<2, 32>>>();
    cudaDeviceSynchronize();

    printf("\n# Probe: kernel with explicit mbarrier\n");
    probe_with_mbar<<<1, 32>>>();
    cudaDeviceSynchronize();

    return 0;
}
