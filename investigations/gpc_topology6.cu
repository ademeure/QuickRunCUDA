/*
 * GPC Topology Part 6 — B300 sm_103a  (DEFINITIVE)
 *
 * Key findings so far:
 *  - 148 SMs, IDs 0-147 contiguous
 *  - 74 TPCs, each TPC = consecutive SM pair {2k, 2k+1}
 *  - Max cluster size = 8 (cluster-16 is invalid)
 *  - cluster-4: 95% of placements are cross-GPC-16
 *  - The "within-GPC" restriction must be at a LARGER granularity than 16
 *    or the driver doesn't respect GPC boundaries for cluster placement
 *
 * New insight from TPC affinity analysis:
 *  TPC 0 ↔ TPC 8 (SMs 0,1 ↔ SMs 16,17) — very strong (freq 106/109)
 *  TPC 8 ↔ TPC 16 (SMs 16,17 ↔ SMs 32,33) — very strong (freq 109)
 *  TPC 24 ↔ TPC 16 (SMs 48,49 ↔ SMs 32,33) — strong (freq 102)
 *  => TPC affinity chain: 0-8-16-24 (SM stride 16)
 *
 *  TPC 65 ↔ TPC 68 (SMs 130,131 ↔ SMs 136,137) — strong (68)
 *  TPC 66 ↔ TPC 69 (SMs 132,133 ↔ SMs 138,139) — strong (56)
 *  TPC 67 ↔ TPC 70 (SMs 134,135 ↔ SMs 140,141) — strong (56)
 *  => Partial GPC: TPCs 65-70 (SMs 130-141) + what about 142-147?
 *
 * This program does:
 *  1. Build complete TPC co-occurrence graph from large cluster-4 run
 *  2. Find connected components with strong affinity
 *  3. Determine GPC structure from the graph
 *  4. Check SMs 142-147 specifically
 *
 * Compile: nvcc -arch=sm_103a -O3 -o gpc_topology6 gpc_topology6.cu
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <map>
#include <set>
#include <queue>
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
    printf("  [CS=%d] %zu valid cluster observations (from %d rounds)\n", CS, result.size(), n_rounds);
    return result;
}

int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  SM=%d\n", prop.name, prop.multiProcessorCount);
    int sm_count = prop.multiProcessorCount;

    // ================================================================
    // 1. Check SMs 142-147 using simple (no-cluster) launches
    // ================================================================
    printf("\n=== SM coverage check (simple, persistent launch) ===\n");
    {
        // Large grid, 1 thread per block to force good coverage
        int n = sm_count * 64;
        unsigned* d; CHECK(cudaMalloc(&d, n * sizeof(unsigned)));
        cudaMemset(d, 0xff, n * sizeof(unsigned));
        kernel_smid_simple<<<n, 1>>>(d, n);
        CHECK(cudaDeviceSynchronize());
        std::vector<unsigned> h(n);
        CHECK(cudaMemcpy(h.data(), d, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
        cudaFree(d);
        std::set<unsigned> seen;
        for (auto s : h) if (s < 1024) seen.insert(s);
        printf("  Unique SMs: %zu, last 20: {", seen.size());
        auto it = seen.end();
        int show = 0;
        while (show < 20 && it != seen.begin()) { --it; show++; }
        for (; it != seen.end(); ++it) printf("%u,", *it);
        printf("}\n");
    }

    // ================================================================
    // 2. Very large cluster-4 run — build TPC co-occurrence matrix
    // ================================================================
    printf("\n=== Large cluster-4 run (500 rounds) — TPC affinity matrix ===\n");
    auto clusters4 = gather_clusters<4>(sm_count, 500);

    // Build TPC-pair frequency table
    // TPC(sm) = sm / 2
    // n_tpc = 74 (148/2)
    int n_tpc = sm_count / 2;
    std::vector<std::vector<int>> tpc_cofreq(n_tpc, std::vector<int>(n_tpc, 0));
    for (auto& g : clusters4) {
        // Extract TPC IDs
        std::set<unsigned> tpcs;
        for (auto s : g) tpcs.insert(s / 2);
        std::vector<unsigned> tpc_list(tpcs.begin(), tpcs.end());
        for (size_t i = 0; i < tpc_list.size(); i++)
            for (size_t j = i+1; j < tpc_list.size(); j++) {
                tpc_cofreq[tpc_list[i]][tpc_list[j]]++;
                tpc_cofreq[tpc_list[j]][tpc_list[i]]++;
            }
    }

    // For each TPC, find its N strongest partners
    printf("  TPC affinity (top-3 partners per TPC):\n");
    for (int t = 0; t < n_tpc; t++) {
        std::vector<std::pair<int,int>> partners;
        for (int p = 0; p < n_tpc; p++) {
            if (p != t && tpc_cofreq[t][p] > 0)
                partners.push_back({tpc_cofreq[t][p], p});
        }
        std::sort(partners.rbegin(), partners.rend());
        printf("    TPC%2d (SM%3d,%3d):", t, t*2, t*2+1);
        for (int k = 0; k < (int)std::min((size_t)3, partners.size()); k++) {
            printf(" TPC%2d(f=%d)", partners[k].second, partners[k].first);
        }
        printf("\n");
    }

    // ================================================================
    // 3. Build GPC graph using strong affinity threshold
    // ================================================================
    printf("\n=== GPC graph construction ===\n");
    {
        // Total observations per TPC
        std::vector<int> tpc_obs(n_tpc, 0);
        for (auto& g : clusters4) {
            std::set<unsigned> tpcs;
            for (auto s : g) tpcs.insert(s / 2);
            for (auto t : tpcs) tpc_obs[t]++;
        }

        // For each TPC, its best partner's frequency / own_obs = fraction
        // High fraction = very likely same GPC
        printf("  Building adjacency with strong edges (>15%% of obs):\n");
        std::vector<std::set<int>> adj(n_tpc);
        for (int t = 0; t < n_tpc; t++) {
            for (int p = 0; p < n_tpc; p++) {
                if (p == t) continue;
                double frac = (double)tpc_cofreq[t][p] / tpc_obs[t];
                if (frac > 0.15) {
                    adj[t].insert(p);
                }
            }
        }

        // Find connected components
        std::vector<int> comp(n_tpc, -1);
        int n_comp = 0;
        for (int start = 0; start < n_tpc; start++) {
            if (comp[start] != -1) continue;
            std::queue<int> q;
            q.push(start);
            comp[start] = n_comp;
            while (!q.empty()) {
                int cur = q.front(); q.pop();
                for (auto nb : adj[cur]) {
                    if (comp[nb] == -1) {
                        comp[nb] = n_comp;
                        q.push(nb);
                    }
                }
            }
            n_comp++;
        }

        // Group TPCs by component
        std::map<int, std::set<int>> components;
        for (int t = 0; t < n_tpc; t++) components[comp[t]].insert(t);

        printf("  Connected components (threshold 15%%):\n");
        for (auto& [cid, tpcs] : components) {
            printf("    Component %d: %zu TPCs = {", cid, tpcs.size());
            for (auto t : tpcs) printf("TPC%d,", t);
            printf("}\n    SMs: {");
            for (auto t : tpcs) printf("%d,%d,", t*2, t*2+1);
            printf("}\n");
        }
        printf("  Total components: %d\n", n_comp);
    }

    // ================================================================
    // 4. Try different thresholds to find GPC boundaries
    // ================================================================
    printf("\n=== Threshold sweep for TPC affinity graph ===\n");
    for (double thresh : {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}) {
        std::vector<int> tpc_obs(n_tpc, 0);
        for (auto& g : clusters4) {
            std::set<unsigned> tpcs;
            for (auto s : g) tpcs.insert(s / 2);
            for (auto t : tpcs) tpc_obs[t]++;
        }

        std::vector<std::set<int>> adj(n_tpc);
        for (int t = 0; t < n_tpc; t++) {
            for (int p = 0; p < n_tpc; p++) {
                if (p == t) continue;
                double frac = (tpc_obs[t] > 0) ?
                    (double)tpc_cofreq[t][p] / tpc_obs[t] : 0;
                if (frac > thresh) adj[t].insert(p);
            }
        }

        std::vector<int> comp(n_tpc, -1);
        int n_comp = 0;
        std::map<int,int> comp_size;
        for (int start = 0; start < n_tpc; start++) {
            if (comp[start] != -1) continue;
            std::queue<int> q;
            q.push(start); comp[start] = n_comp;
            int sz = 0;
            while (!q.empty()) {
                int cur = q.front(); q.pop(); sz++;
                for (auto nb : adj[cur]) {
                    if (comp[nb] == -1) { comp[nb] = n_comp; q.push(nb); }
                }
            }
            comp_size[n_comp] = sz;
            n_comp++;
        }

        // Size histogram
        std::map<int,int> size_hist;
        for (auto& [c, sz] : comp_size) size_hist[sz]++;

        printf("  thresh=%.2f: %d components, sizes={", thresh, n_comp);
        for (auto& [sz, cnt] : size_hist) printf("%dTPCs×%d,", sz, cnt);
        printf("}\n");
    }

    // ================================================================
    // 5. cluster-8 patterns for SMs 128-147 specifically
    // ================================================================
    printf("\n=== cluster-8: patterns involving SMs 128-147 ===\n");
    {
        auto clusters8 = gather_clusters<8>(sm_count, 200);

        std::map<std::set<unsigned>, int> freq;
        for (auto& g : clusters8) freq[g]++;

        std::vector<std::pair<int,std::set<unsigned>>> sorted;
        for (auto& [g, f] : freq) {
            bool has_high = false;
            for (auto s : g) if (s >= 128) { has_high = true; break; }
            if (has_high) sorted.push_back({f, g});
        }
        std::sort(sorted.rbegin(), sorted.rend());

        printf("  Top cluster-8 groups involving SMs >= 128:\n");
        for (int i = 0; i < (int)std::min((size_t)30, sorted.size()); i++) {
            printf("    freq=%3d: {", sorted[i].first);
            for (auto s : sorted[i].second) printf("%u,", s);
            printf("}\n");
        }

        // Find which SMs appear with SMs 142-147
        printf("\n  SMs that cluster-8 with SMs 142/143/144/145/146/147:\n");
        std::map<unsigned, std::map<unsigned, int>> cofreq;
        for (auto& [g, f] : freq) {
            for (auto s : g) {
                if (s >= 142) {
                    for (auto p : g) {
                        if (p != s) cofreq[s][p] += f;
                    }
                }
            }
        }
        for (auto& [sm, partners] : cofreq) {
            printf("  SM %u appears with: {", sm);
            for (auto& [p, f] : partners) printf("%u(f=%d),", p, f);
            printf("}\n");
        }
    }

    // ================================================================
    // 6. SMs 142-147 never appear in cluster-8 with 60 rounds —
    //    check if they're truly unreachable in cluster mode
    // ================================================================
    printf("\n=== Direct check: SM 142-147 reachability ===\n");
    {
        // Run 1000 rounds of cluster-2 focused on high SM IDs
        // by launching large grids
        auto c2 = gather_clusters<2>(sm_count, 100);
        std::set<unsigned> seen_in_c2;
        for (auto& g : c2) for (auto s : g) seen_in_c2.insert(s);
        printf("  cluster-2: SMs seen: %zu\n", seen_in_c2.size());
        printf("  SMs 140-147 in cluster-2: {");
        for (unsigned s = 140; s <= 147; s++)
            if (seen_in_c2.count(s)) printf("%u,", s);
        printf("}\n");

        auto c8 = gather_clusters<8>(sm_count, 100);
        std::set<unsigned> seen_in_c8;
        for (auto& g : c8) for (auto s : g) seen_in_c8.insert(s);
        printf("  cluster-8: SMs seen: %zu\n", seen_in_c8.size());
        printf("  SMs 140-147 in cluster-8: {");
        for (unsigned s = 140; s <= 147; s++)
            if (seen_in_c8.count(s)) printf("%u,", s);
        printf("}\n");
    }

    printf("\nDone.\n");
    return 0;
}
