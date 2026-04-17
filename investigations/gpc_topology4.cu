/*
 * GPC Topology Part 4 — B300 sm_103a
 *
 * Print actual peer groups for cluster-4 and cluster-8.
 * The previous run showed peer sizes of 10, 18, 20 for cluster-8 —
 * meaning the hardware schedules cluster-8 blocks to 8 different SMs
 * but the "group" seen in peer intersections spans more SMs (they can
 * be placed on different GPCs across runs).
 *
 * Key insight needed: the UNION of all cluster placements per SM tells
 * us the "reachable" SMs, not the GPC.  Instead, we need the
 * INTERSECTION of peer sets across rounds.
 *
 * Better strategy:
 *   For each SM x, find the intersection of all cluster-8 groups that
 *   contained SM x.  That intersection (minus x itself) = guaranteed
 *   co-schedulable SMs = the GPC.
 *
 * Even better: a GPC shows up as a cluster that appears REPEATEDLY
 * with the same set of SMs.  Find frequently occurring clusters.
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

template<int CS>
__global__ void __cluster_dims__(CS, 1, 1) kernel_smid_rank(
    unsigned* out, int /*unused*/)
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
    int valid = 0, invalid = 0;
    for (int c = 0; c < n_clusters; c++) {
        std::vector<unsigned> smids(CS, 0xffffffff);
        bool ok = true;
        for (int b = c * CS; b < (c+1) * CS; b++) {
            unsigned smid = h[b * 2 + 0];
            unsigned rank = h[b * 2 + 1];
            if (smid >= 1024 || rank >= (unsigned)CS) { ok = false; break; }
            smids[rank] = smid;
        }
        if (!ok) { invalid++; continue; }
        bool all_valid = true;
        std::set<unsigned> grp;
        for (auto s : smids) {
            if (s >= 1024) { all_valid = false; break; }
            grp.insert(s);
        }
        if (!all_valid || (int)grp.size() != CS) { invalid++; continue; }
        valid++;
        result.push_back(grp);
    }
    printf("  [CS=%d] valid clusters: %d, invalid: %d\n", CS, valid, invalid);
    return result;
}

int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    int sm_count = prop.multiProcessorCount;

    // ================================================================
    // cluster-8: find most frequently recurring groups
    // ================================================================
    printf("\n=== Cluster-8 frequency analysis ===\n");
    {
        auto clusters = gather_clusters<8>(sm_count, 60);

        // Count frequency of each unique group
        std::map<std::set<unsigned>, int> freq;
        for (auto& g : clusters) freq[g]++;

        // Sort by frequency descending
        std::vector<std::pair<int, std::set<unsigned>>> sorted;
        for (auto& [g, f] : freq) sorted.push_back({f, g});
        std::sort(sorted.rbegin(), sorted.rend());

        printf("  Total unique cluster-8 groups seen: %zu\n", freq.size());
        printf("  Top 40 most frequent groups:\n");
        int shown = 0;
        for (auto& [f, g] : sorted) {
            if (shown++ >= 40) break;
            printf("    freq=%3d: {", f);
            for (auto s : g) printf("%u,", s);
            printf("}\n");
        }

        // Groups that appear very frequently are the "natural" GPCs
        // Groups that appear rarely are cross-GPC placements
        int high_freq_threshold = (int)(sorted[0].first * 0.7);  // 70% of max
        printf("\n  High-frequency groups (freq >= %d):\n", high_freq_threshold);
        int gpc_idx = 0;
        std::set<unsigned> all_in_stable_gpcs;
        for (auto& [f, g] : sorted) {
            if (f < high_freq_threshold) break;
            printf("    GPC-candidate-%d (freq=%d): {", gpc_idx++, f);
            for (auto s : g) printf("%u,", s);
            printf("}\n");
            for (auto s : g) all_in_stable_gpcs.insert(s);
        }
        printf("  SMs in stable GPCs: %zu / %d\n", all_in_stable_gpcs.size(), sm_count);
    }

    // ================================================================
    // cluster-4: find most frequently recurring groups
    // ================================================================
    printf("\n=== Cluster-4 frequency analysis ===\n");
    {
        auto clusters = gather_clusters<4>(sm_count, 60);

        std::map<std::set<unsigned>, int> freq;
        for (auto& g : clusters) freq[g]++;

        std::vector<std::pair<int, std::set<unsigned>>> sorted;
        for (auto& [g, f] : freq) sorted.push_back({f, g});
        std::sort(sorted.rbegin(), sorted.rend());

        printf("  Total unique cluster-4 groups seen: %zu\n", freq.size());
        printf("  Top 50 most frequent groups:\n");
        int shown = 0;
        for (auto& [f, g] : sorted) {
            if (shown++ >= 50) break;
            printf("    freq=%3d: {", f);
            for (auto s : g) printf("%u,", s);
            printf("}\n");
        }

        int high_freq_threshold = (int)(sorted[0].first * 0.6);
        printf("\n  High-frequency groups (freq >= %d):\n", high_freq_threshold);
        for (auto& [f, g] : sorted) {
            if (f < high_freq_threshold) break;
            printf("    freq=%d: {", f);
            for (auto s : g) printf("%u,", s);
            printf("}\n");
        }
    }

    // ================================================================
    // cluster-2: verify TPC pairs are always consistent
    // ================================================================
    printf("\n=== Cluster-2 consistency check ===\n");
    {
        auto clusters = gather_clusters<2>(sm_count, 20);
        std::map<std::set<unsigned>, int> freq;
        for (auto& g : clusters) freq[g]++;
        printf("  Unique pairs: %zu\n", freq.size());
        // Check if every pair is exactly {2k, 2k+1}
        bool all_consecutive = true;
        for (auto& [g, f] : freq) {
            auto it = g.begin();
            unsigned a = *it++, b = *it;
            if (b != a + 1) { all_consecutive = false; break; }
        }
        printf("  All pairs consecutive (stride-1): %s\n", all_consecutive ? "YES" : "NO");
        if (!all_consecutive) {
            for (auto& [g, f] : freq) {
                auto it = g.begin();
                unsigned a = *it++, b = *it;
                if (b != a + 1)
                    printf("  NON-CONSECUTIVE pair: {%u,%u}\n", a, b);
            }
        }
    }

    // ================================================================
    // Analysis: for each SM, what's the intersection of all cluster-8
    // groups that contained it?  That's the "always co-scheduled" set.
    // ================================================================
    printf("\n=== Per-SM intersection analysis (cluster-8) ===\n");
    {
        auto clusters = gather_clusters<8>(sm_count, 60);

        // For each SM, collect all cluster groups it appeared in
        std::map<unsigned, std::vector<std::set<unsigned>>> sm_appearances;
        for (auto& g : clusters)
            for (auto s : g)
                sm_appearances[s].push_back(g);

        printf("  SM -> intersection of all its cluster groups:\n");
        for (int sm = 0; sm < sm_count; sm++) {
            auto it = sm_appearances.find(sm);
            if (it == sm_appearances.end()) {
                printf("  SM %3d: never seen!\n", sm);
                continue;
            }
            // Compute intersection
            std::set<unsigned> inter = it->second[0];
            for (size_t i = 1; i < it->second.size(); i++) {
                std::set<unsigned> tmp;
                for (auto x : inter)
                    if (it->second[i].count(x)) tmp.insert(x);
                inter = tmp;
            }
            // Only print if intersection is interesting (smaller than full set)
            if ((int)inter.size() < 8) {
                printf("  SM %3d: intersection size=%zu: {", sm, inter.size());
                for (auto s : inter) printf("%u,", s);
                printf("}  appearances=%zu\n", it->second.size());
            }
        }

        // Find SMs whose intersection is exactly size 8 (always same co-group)
        std::set<std::set<unsigned>> stable_gpcs;
        std::set<unsigned> stable_sms;
        for (int sm = 0; sm < sm_count; sm++) {
            auto it = sm_appearances.find(sm);
            if (it == sm_appearances.end()) continue;
            std::set<unsigned> inter = it->second[0];
            for (size_t i = 1; i < it->second.size(); i++) {
                std::set<unsigned> tmp;
                for (auto x : inter)
                    if (it->second[i].count(x)) tmp.insert(x);
                inter = tmp;
            }
            if ((int)inter.size() == 8) {
                stable_gpcs.insert(inter);
                for (auto s : inter) stable_sms.insert(s);
            }
        }
        printf("\n  Stable GPCs (intersection = exactly 8 SMs):\n");
        int gi = 0;
        for (auto& g : stable_gpcs) {
            printf("    GPC-%d: {", gi++);
            for (auto s : g) printf("%u,", s);
            printf("}\n");
        }
        printf("  Total stable GPCs: %zu, SMs covered: %zu / %d\n",
               stable_gpcs.size(), stable_sms.size(), sm_count);
    }

    printf("\nDone.\n");
    return 0;
}
