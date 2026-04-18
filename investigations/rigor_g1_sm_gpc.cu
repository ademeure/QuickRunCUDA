// G1 RIGOR: SM->GPC mapping via SM clock-skew "boot groups"
//
// THEORETICAL: B300 has 148 SMs, 8 GPCs per cluster placement evidence.
// SMs in same GPC will have very close clock64 startup values (boot
// together); SMs in different GPCs will have larger offsets.
//
// Method: launch one block per SM with a fixed delay; record clock64
// at start. Cluster SMs by clock value into groups.

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

__global__ void probe(unsigned long long *clk_out, unsigned *sm_out) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned long long c = clock64();
    if (threadIdx.x == 0) {
        clk_out[blockIdx.x] = c;
        sm_out[blockIdx.x] = smid;
    }
}

int main() {
    cudaSetDevice(0);
    int n_sms = 148;

    // Launch many more blocks than SMs to ensure each SM gets a block
    int blocks = n_sms * 4;  // 592 blocks
    int threads = 32;
    unsigned long long *d_clk; cudaMalloc(&d_clk, blocks * sizeof(unsigned long long));
    unsigned *d_sm; cudaMalloc(&d_sm, blocks * sizeof(unsigned));

    probe<<<blocks, threads>>>(d_clk, d_sm);
    cudaDeviceSynchronize();

    std::vector<unsigned long long> h_clk(blocks);
    std::vector<unsigned> h_sm(blocks);
    cudaMemcpy(h_clk.data(), d_clk, blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sm.data(), d_sm, blocks * sizeof(unsigned), cudaMemcpyDeviceToHost);

    // For each SM, take the EARLIEST clock value (first block dispatched)
    std::vector<unsigned long long> sm_first_clk(n_sms, ~0ULL);
    for (int b = 0; b < blocks; b++) {
        unsigned smid = h_sm[b];
        if (smid < (unsigned)n_sms && h_clk[b] < sm_first_clk[smid]) {
            sm_first_clk[smid] = h_clk[b];
        }
    }

    // Find min clock to subtract baseline
    unsigned long long min_c = *std::min_element(sm_first_clk.begin(), sm_first_clk.end());

    // Sort SMs by their offset from min
    std::vector<std::pair<long long, int>> sm_offset;
    for (int sm = 0; sm < n_sms; sm++) {
        if (sm_first_clk[sm] == ~0ULL) continue;
        long long off = (long long)sm_first_clk[sm] - (long long)min_c;
        sm_offset.push_back({off, sm});
    }
    std::sort(sm_offset.begin(), sm_offset.end());

    // Print sorted offsets
    printf("# SM-startup-clock offsets (relative to earliest SM)\n");
    printf("# sorted ascending — gaps indicate group boundaries\n");
    long long prev = 0;
    int group = 0;
    int sms_in_group = 0;
    for (auto& [off, sm] : sm_offset) {
        long long gap = off - prev;
        if (gap > 1000000) { // 1M cycles = clear group boundary
            printf("\n--- group %d -> group %d (gap %lld cy) ---\n", group, group+1, gap);
            group++;
            sms_in_group = 0;
        }
        printf("  SM%3d: offset = %12lld cy\n", sm, off);
        sms_in_group++;
        prev = off;
    }
    printf("\nTotal groups detected: %d\n", group + 1);

    return 0;
}
