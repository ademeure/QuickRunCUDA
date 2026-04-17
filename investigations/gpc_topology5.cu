/*
 * GPC Topology Part 5 — B300 sm_103a
 *
 * Key finding from parts 1-4:
 *   - 148 SMs total, SM IDs 0-147 contiguous
 *   - TPC pairs: {0,1}, {2,3}, ..., {146,147} — 74 TPCs
 *   - Cluster-2: always places 2 blocks on the same TPC (stride-1 pairs)
 *   - Cluster-8: blocks can be placed on any 4 TPCs (no fixed GPC constraint observed)
 *   - Cluster-16: FAILED ("misconfiguration") — max cluster = 8?
 *   - SM 142-147 never scheduled in cluster-8 with 60 rounds — suspicious
 *
 * New strategy:
 *   1. Check occupancy/max-cluster-size API
 *   2. Look at SMs 130-147 specifically — partial GPC?
 *   3. Try to force all 148 SMs into a cluster-4 round-robin and
 *      check which SMs are NEVER scheduled (disabled SMs in partial GPC?)
 *   4. Probe cluster-4 patterns for SMs 128-147 specifically
 *   5. Use cudaOccupancyMaxPotentialClusterSize to get the theoretical limit
 *
 * Compile: nvcc -arch=sm_103a -O3 -o gpc_topology5 gpc_topology5.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// Simple smid-per-block without cluster (to get accurate SM coverage)
__global__ void kernel_smid_simple(unsigned* out, int n) {
    int bid = (int)blockIdx.x;
    if (bid >= n) return;
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid) :);
    if (threadIdx.x == 0) out[bid] = smid;
}

template<int CS>
__global__ void __cluster_dims__(CS, 1, 1) kernel_smid_rank(unsigned* out, int /*unused*/)
{
    cg::cluster_group cluster = cg::this_cluster();
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid) :);
    unsigned rank = cluster.block_rank();
    int bid = (int)blockIdx.x;
    if (threadIdx.x == 0) {
        out[bid * 2 + 0] = smid;
        out[bid * 2 + 1] = rank;
    }
}

template<int CS>
static std::vector<std::set<unsigned>> gather_clusters(int sm_count, int n_rounds)
{
    int n_clusters = (sm_count / CS) * n_rounds;
    int n_blocks   = n_clusters * CS;

    unsigned* d_out;
    CHECK(cudaMalloc(&d_out, n_blocks * 2 * sizeof(unsigned)));
    cudaMemset(d_out, 0xff, n_blocks * 2 * sizeof(unsigned));

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim  = dim3(n_blocks, 1, 1);
    cfg.blockDim = dim3(1, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = 0;

    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim = {(unsigned)CS, 1, 1};
    cfg.attrs    = attr;
    cfg.numAttrs = 1;

    int dummy = n_blocks;
    void* kargs[] = {&d_out, &dummy};
    cudaError_t err = cudaLaunchKernelExC(&cfg, (void*)kernel_smid_rank<CS>, kargs);
    if (err != cudaSuccess) {
        printf("  [CS=%d] launch failed: %s\n", CS, cudaGetErrorString(err));
        cudaFree(d_out);
        return {};
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  [CS=%d] sync failed: %s\n", CS, cudaGetErrorString(err));
        cudaFree(d_out);
        return {};
    }

    std::vector<unsigned> h(n_blocks * 2);
    CHECK(cudaMemcpy(h.data(), d_out, n_blocks * 2 * sizeof(unsigned), cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    std::vector<std::set<unsigned>> result;
    for (int c = 0; c < n_clusters; c++) {
        std::vector<unsigned> smids(CS, 0xffffffff);
        bool ok = true;
        for (int b = c * CS; b < (c+1) * CS; b++) {
            unsigned smid = h[b * 2 + 0];
            unsigned rank = h[b * 2 + 1];
            if (smid >= 1024 || rank >= (unsigned)CS) { ok = false; break; }
            smids[rank] = smid;
        }
        if (!ok) continue;
        std::set<unsigned> grp;
        bool all_valid = true;
        for (auto s : smids) {
            if (s >= 1024) { all_valid = false; break; }
            grp.insert(s);
        }
        if (!all_valid || (int)grp.size() != CS) continue;
        result.push_back(grp);
    }
    printf("  [CS=%d] collected %zu valid cluster observations\n", CS, result.size());
    return result;
}

int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  SM=%d  Compute=%d.%d\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    int sm_count = prop.multiProcessorCount;
    // maxBlocksPerCluster not in this CUDA SDK version, use API instead

    // ================================================================
    // Test which cluster sizes are actually valid
    // ================================================================
    printf("\n=== Valid cluster sizes ===\n");
    for (int cs : {1, 2, 4, 8, 16, 32}) {
        // Use cudaOccupancyMaxPotentialClusterSize
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim  = dim3(cs, 1, 1);
        cfg.blockDim = dim3(1, 1, 1);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = 0;

        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim = {(unsigned)cs, 1, 1};
        cfg.attrs    = attr;
        cfg.numAttrs = 1;

        int max_clusters = 0;
        cudaError_t err = cudaOccupancyMaxActiveClusters(&max_clusters,
                                                          (void*)kernel_smid_simple, &cfg);
        if (err == cudaSuccess) {
            printf("  cluster_size=%2d: valid, max_active_clusters=%d\n", cs, max_clusters);
        } else {
            printf("  cluster_size=%2d: INVALID (%s)\n", cs, cudaGetErrorString(err));
            cudaGetLastError();  // clear
        }
    }

    // ================================================================
    // Verify SM coverage with many blocks (no cluster)
    // ================================================================
    printf("\n=== SM coverage (no cluster, 4096 blocks) ===\n");
    {
        int n = 4096;
        unsigned* d; CHECK(cudaMalloc(&d, n * sizeof(unsigned)));
        cudaMemset(d, 0xff, n * sizeof(unsigned));
        kernel_smid_simple<<<n, 1>>>(d, n);
        CHECK(cudaDeviceSynchronize());
        std::vector<unsigned> h(n);
        CHECK(cudaMemcpy(h.data(), d, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
        cudaFree(d);

        std::set<unsigned> seen;
        for (auto s : h) if (s < 1024) seen.insert(s);
        printf("  Unique SMs seen: %zu\n", seen.size());
        printf("  Max SM ID: %u\n", *seen.rbegin());

        // Which SMs in [128,147] appear?
        printf("  SMs 128-147 seen: {");
        for (auto s : seen) if (s >= 128) printf("%u,", s);
        printf("}\n");

        // Check for gaps
        unsigned prev = *seen.begin();
        bool has_gap = false;
        for (auto s : seen) {
            if (s > prev + 1 && s != *seen.begin()) {
                printf("  GAP: [%u, %u)\n", prev+1, s);
                has_gap = true;
            }
            prev = s;
        }
        if (!has_gap) printf("  No gaps in SM IDs\n");
    }

    // ================================================================
    // cluster-4 analysis focused on SMs 128-147 (partial GPC region)
    // ================================================================
    printf("\n=== cluster-4 analysis focused on SMs 128-147 ===\n");
    {
        auto clusters = gather_clusters<4>(sm_count, 200);

        // Filter to clusters involving SMs 128-147
        std::set<std::set<unsigned>> high_sm_clusters;
        std::map<std::set<unsigned>, int> freq;
        for (auto& g : clusters) {
            freq[g]++;
            for (auto s : g) {
                if (s >= 128) { high_sm_clusters.insert(g); break; }
            }
        }

        printf("  Clusters involving SM >= 128:\n");
        // Sort by min SM in group
        std::vector<std::set<unsigned>> sorted_groups(high_sm_clusters.begin(), high_sm_clusters.end());
        std::sort(sorted_groups.begin(), sorted_groups.end(),
                  [](auto& a, auto& b){
                      return *a.begin() < *b.begin();
                  });
        for (auto& g : sorted_groups) {
            printf("    freq=%3d: {", freq[g]);
            for (auto s : g) printf("%u,", s);
            printf("}\n");
        }

        // Find the most common cluster patterns for SMs 128-147
        printf("\n  Most common groups containing SMs >= 128:\n");
        std::vector<std::pair<int,std::set<unsigned>>> grp_freq;
        for (auto& g : high_sm_clusters) grp_freq.push_back({freq[g], g});
        std::sort(grp_freq.rbegin(), grp_freq.rend());
        for (int i = 0; i < (int)std::min((size_t)20, grp_freq.size()); i++) {
            printf("    freq=%3d: {", grp_freq[i].first);
            for (auto s : grp_freq[i].second) printf("%u,", s);
            printf("}\n");
        }
    }

    // ================================================================
    // cluster-4: find the consistent same-GPC pairs of TPCs
    // Use mutual frequency: if {A,B,C,D} appears N times, the pairs
    // (A,B), (A,C) etc. always co-schedule within cluster-4.
    // Hypothesis: GPCs are {0..15}, {16..31}, ... {128..143}, {144..147}
    // i.e., 9 full GPCs of 16 SMs + 1 partial GPC of 4 SMs.
    // ================================================================
    printf("\n=== Hypothesis test: are GPCs groups of 16 SMs? ===\n");
    {
        auto clusters = gather_clusters<4>(sm_count, 200);

        // Count how many times two TPCs appear in the SAME cluster-4
        // A TPC is identified by its pair {2k, 2k+1} -> TPC id = sm/2
        // So 74 TPCs: TPC 0={0,1}, TPC 1={2,3}, ..., TPC 73={146,147}
        // A cluster-4 has 2 TPCs
        std::map<std::pair<unsigned,unsigned>, int> tpc_pair_freq;
        for (auto& g : clusters) {
            std::vector<unsigned> smids(g.begin(), g.end());
            // Get TPC IDs
            std::set<unsigned> tpcs;
            for (auto s : smids) tpcs.insert(s / 2);
            std::vector<unsigned> tpc_list(tpcs.begin(), tpcs.end());
            if (tpc_list.size() == 2) {
                auto p = std::make_pair(tpc_list[0], tpc_list[1]);
                tpc_pair_freq[p]++;
            }
        }

        // For each TPC, which other TPC does it MOST frequently pair with?
        std::map<unsigned, std::pair<unsigned,int>> best_pair;  // TPC -> (best partner, freq)
        for (auto& [pair, freq] : tpc_pair_freq) {
            auto [a, b] = pair;
            if (!best_pair.count(a) || best_pair[a].second < freq)
                best_pair[a] = {b, freq};
            if (!best_pair.count(b) || best_pair[b].second < freq)
                best_pair[b] = {a, freq};
        }

        printf("  TPC pair affinities (most frequent cluster partner):\n");
        for (unsigned tpc = 0; tpc < 74; tpc++) {
            if (best_pair.count(tpc)) {
                auto [partner, freq] = best_pair[tpc];
                // Check if partner agrees (mutual)
                bool mutual = best_pair.count(partner) && best_pair[partner].first == tpc;
                printf("    TPC %2u (SMs %3u,%3u) -> TPC %2u (SMs %3u,%3u)  freq=%d %s\n",
                       tpc, tpc*2, tpc*2+1,
                       partner, partner*2, partner*2+1,
                       freq, mutual ? "[MUTUAL]" : "");
            }
        }

        // Group TPCs into GPC hypothesis: groups of 8 TPCs (16 SMs)
        // Check if cluster-4 placements respect GPC boundaries of 16 SMs
        printf("\n  Cross-GPC-16 cluster-4 placements:\n");
        int within = 0, cross = 0;
        for (auto& g : clusters) {
            // Find GPC id of each SM (GPC = SM / 16)
            std::set<unsigned> gpcs;
            for (auto s : g) gpcs.insert(s / 16);
            if (gpcs.size() == 1) within++;
            else cross++;
        }
        printf("  Within-GPC-16: %d, Cross-GPC-16: %d (total %zu)\n",
               within, cross, clusters.size());
        printf("  Cross-GPC-16 fraction: %.1f%%\n",
               100.0 * cross / clusters.size());
    }

    // ================================================================
    // What is the TPC-group structure within a GPC?
    // Check cluster-4 patterns within each hypothetical GPC of 16 SMs
    // ================================================================
    printf("\n=== Within-GPC-16 cluster-4 analysis ===\n");
    {
        auto clusters = gather_clusters<4>(sm_count, 100);
        // Count within-GPC vs cross-GPC by group of 16
        std::map<std::set<unsigned>, int> freq;
        for (auto& g : clusters) freq[g]++;

        // Within GPC-16 clusters only
        std::map<unsigned, std::set<std::set<unsigned>>> gpc_internal;  // gpc -> set of observed patterns
        for (auto& g : clusters) {
            std::set<unsigned> gpcs;
            for (auto s : g) gpcs.insert(s / 16);
            if (gpcs.size() == 1) {
                gpc_internal[*gpcs.begin()].insert(g);
            }
        }

        for (auto& [gpc, patterns] : gpc_internal) {
            unsigned base = gpc * 16;
            printf("  GPC-%u (SMs %u-%u): %zu internal cluster-4 patterns\n",
                   gpc, base, base + 15, patterns.size());
            for (auto& p : patterns) {
                printf("    {");
                for (auto s : p) printf("%u,", s);
                printf("} (freq=%d)\n", freq[p]);
            }
        }
    }

    printf("\nDone.\n");
    return 0;
}
