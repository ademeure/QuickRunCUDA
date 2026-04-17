// Find all SM IDs used and infer GPC topology from cluster
#include <cuda_runtime.h>
#include <cstdio>

__device__ unsigned get_smid() { unsigned s; asm("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
__device__ unsigned get_nsmid() { unsigned s; asm("mov.u32 %0, %%nsmid;" : "=r"(s)); return s; }

__global__ void enum_smids(unsigned *smids, unsigned *nsmids) {
    if (threadIdx.x == 0) {
        smids[blockIdx.x] = get_smid();
        if (blockIdx.x == 0) *nsmids = get_nsmid();
    }
}

int main() {
    cudaSetDevice(0);

    unsigned *d_smids; cudaMalloc(&d_smids, 1024 * sizeof(unsigned));
    unsigned *d_nsmid; cudaMalloc(&d_nsmid, sizeof(unsigned));

    // Launch enough blocks to fill all SMs
    int n_blocks = 296;  // 2 per SM
    cudaMemset(d_smids, 0xff, 1024 * sizeof(unsigned));
    enum_smids<<<n_blocks, 32>>>(d_smids, d_nsmid);
    cudaDeviceSynchronize();

    unsigned smids[1024], nsmid;
    cudaMemcpy(smids, d_smids, n_blocks * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(&nsmid, d_nsmid, sizeof(unsigned), cudaMemcpyDeviceToHost);

    printf("# B300 SM enumeration\n");
    printf("# %%nsmid (max SM count): %u\n\n", nsmid);

    // Find unique SM IDs
    bool seen[300] = {};
    int unique = 0;
    unsigned max_id = 0;
    for (int i = 0; i < n_blocks; i++) {
        if (!seen[smids[i]]) {
            seen[smids[i]] = true;
            unique++;
            if (smids[i] > max_id) max_id = smids[i];
        }
    }
    printf("# Unique SMs seen: %d (max ID: %u)\n", unique, max_id);
    printf("# Max possible SM ID range: 0 to %u\n\n", nsmid - 1);

    printf("# SM presence map (X = present, . = absent):\n");
    for (int row = 0; row < (nsmid + 31) / 32; row++) {
        printf("  SM %3d-%3d: ", row * 32, std::min((row+1)*32-1, (int)nsmid-1));
        for (int col = 0; col < 32 && row*32 + col < nsmid; col++) {
            printf("%c", seen[row*32 + col] ? 'X' : '.');
        }
        printf("\n");
    }

    return 0;
}
