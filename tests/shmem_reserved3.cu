// Final test: where is the reserved 1 KiB located? Beginning or end?
// Also: what's the exact behavior with maximum static declaration?
#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// Probe shmem address range using PTX
extern "C" __global__ void probe_shmem_addrs() {
    if (threadIdx.x != 0) return;
    extern __shared__ int dyn_buf_int[];
    __shared__ int st_buf[64];

    // Get the addresses
    long st_addr = (long)st_buf;
    long dyn_addr = (long)dyn_buf_int;

    printf("# Block %d:\n", blockIdx.x);
    printf("  static_buf addr: 0x%lx (offset from shmem base: %ld)\n", st_addr, st_addr & 0xFFFFFF);
    printf("  dyn_buf addr:    0x%lx (offset from shmem base: %ld)\n", dyn_addr, dyn_addr & 0xFFFFFF);
}

// Use cuda-c shmem layout intrinsic
extern "C" __global__ void probe_with_asm() {
    if (threadIdx.x != 0) return;
    extern __shared__ char dyn_buf_c[];

    // PTX %dynamic_smem_size: gives dynamic smem allocated
    // Use SR intrinsic or ld.shared from a base address
    unsigned dyn_size;
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_size));
    unsigned tot_size;
    asm volatile("mov.u32 %0, %%total_smem_size;" : "=r"(tot_size));

    printf("# Block %d: dyn_smem_size=%u, total_smem_size=%u, dyn-tot=%d\n",
           blockIdx.x, dyn_size, tot_size, (int)(tot_size - dyn_size));
}

// Test what happens with maximum static declaration
extern "C" __global__ void k_static_47k() {
    __shared__ char st[48128];  // exactly 47 KB
    st[threadIdx.x % 48128] = (char)threadIdx.x;
    if (threadIdx.x == 0) printf("47k_static: ok\n");
}

extern "C" __global__ void k_static_max() {
    __shared__ char st[49152];  // exactly 48 KB (= default limit)
    st[threadIdx.x % 49152] = (char)threadIdx.x;
    if (threadIdx.x == 0) printf("max_static: ok\n");
}

// Mixed: static + dyn = 48 KB total
extern "C" __global__ void k_mixed_48k(int dyn_size) {
    __shared__ char st[24576];  // 24 KB
    extern __shared__ char dyn[];
    st[threadIdx.x % 24576] = 1;
    if (threadIdx.x < dyn_size) dyn[threadIdx.x] = 2;
    if (threadIdx.x == 0) printf("mixed: static_done\n");
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("# Reserved per block: %zu, default per block: %zu, opt-in max: %zu\n\n",
           prop.reservedSharedMemPerBlock, prop.sharedMemPerBlock, prop.sharedMemPerBlockOptin);

    // Run probe with no dyn
    printf("## probe_with_asm: dyn=0 bytes\n");
    probe_with_asm<<<1, 32, 0>>>();
    cudaDeviceSynchronize();

    printf("\n## probe_with_asm: dyn=4096 bytes\n");
    probe_with_asm<<<1, 32, 4096>>>();
    cudaDeviceSynchronize();

    printf("\n## probe_with_asm: dyn=49152 bytes (max default)\n");
    probe_with_asm<<<1, 32, 49152>>>();
    cudaDeviceSynchronize();

    // Try opt-in max
    cudaFuncSetAttribute((void*)probe_with_asm, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)prop.sharedMemPerBlockOptin);
    printf("\n## probe_with_asm: dyn=232448 bytes (max opt-in)\n");
    probe_with_asm<<<1, 32, prop.sharedMemPerBlockOptin>>>();
    cudaDeviceSynchronize();

    // Static probe
    printf("\n## probe_shmem_addrs: dyn=4096\n");
    probe_shmem_addrs<<<1, 32, 4096>>>();
    cudaDeviceSynchronize();

    // Try to launch 47k static + 1k dyn (should fit)
    printf("\n## k_static_47k\n");
    cudaError_t r;
    k_static_47k<<<1, 32>>>();
    cudaDeviceSynchronize();
    r = cudaGetLastError();
    printf("  status: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

    // Try 48k static (= default per-block limit)
    printf("\n## k_static_max (48 KB static)\n");
    k_static_max<<<1, 32>>>();
    cudaDeviceSynchronize();
    r = cudaGetLastError();
    printf("  status: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

    // Test cudaFuncAttributePreferredSharedMemoryCarveout (newer attribute)
    printf("\n## PreferredSharedMemoryCarveout effect on max\n");
    int carve_arr[] = {0, 25, 50, 75, 100};
    for (int c : carve_arr) {
        cudaFuncSetAttribute((void*)probe_with_asm, cudaFuncAttributePreferredSharedMemoryCarveout, c);
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, (void*)probe_with_asm);
        printf("  carveout=%-3d : sharedSizeBytes=%zu, maxDynShmem=%d\n",
               c, fa.sharedSizeBytes, fa.maxDynamicSharedSizeBytes);
    }

    return 0;
}
