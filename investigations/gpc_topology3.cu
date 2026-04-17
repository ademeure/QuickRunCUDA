/*
 * GPC Topology Part 3 — B300 sm_103a
 *
 * Approach: launch many separate cluster-8 jobs, each small enough
 * to be reliable, and accumulate peer relationships.
 * Key insight from first run:
 *   - cluster-2 always gives {2k, 2k+1} — consecutive pairs (TPCs)
 *   - cluster-4 shows {.., .+16, ..} structure
 *   - cluster-8 shows {.., .+16, .+32, .+48} or similar
 *
 * This version uses a simple per-block smid write (no dsmem)
 * combined with cluster_id to reconstruct groups.
 *
 * Compile: nvcc -arch=sm_103a -O3 -o gpc_topology3 gpc_topology3.cu
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

// ---------------------------------------------------------------
// Simple SMID recorder — each block records its smid and
// its cluster block rank.  Output layout:
//   out[blockIdx.x * 3 + 0] = smid
//   out[blockIdx.x * 3 + 1] = cluster block rank
//   out[blockIdx.x * 3 + 2] = blockIdx.x (sanity)
// ---------------------------------------------------------------
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
        out[bid * 3 + 0] = smid;
        out[bid * 3 + 1] = rank;
        out[bid * 3 + 2] = (unsigned)bid;
    }
}

template<int CS>
static bool launch_cluster_smid(int sm_count, int n_rounds,
                                 std::map<unsigned, std::set<unsigned>>& sm_peers_out)
{
    int n_clusters  = (sm_count / CS) * n_rounds;
    int n_blocks    = n_clusters * CS;

    unsigned* d_out;
    if (cudaMalloc(&d_out, n_blocks * 3 * sizeof(unsigned)) != cudaSuccess) return false;
    cudaMemset(d_out, 0xff, n_blocks * 3 * sizeof(unsigned));

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
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  [CS=%d] sync failed: %s\n", CS, cudaGetErrorString(err));
        cudaFree(d_out);
        return false;
    }

    std::vector<unsigned> h(n_blocks * 3);
    CHECK(cudaMemcpy(h.data(), d_out, n_blocks * 3 * sizeof(unsigned), cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    // Reconstruct clusters
    int valid = 0;
    for (int c = 0; c < n_clusters; c++) {
        // Collect smids for this cluster
        std::vector<unsigned> smids_in_cluster(CS, 0xffffffff);
        bool ok = true;
        for (int b = c * CS; b < (c+1) * CS; b++) {
            unsigned smid = h[b * 3 + 0];
            unsigned rank = h[b * 3 + 1];
            if (smid >= 1024 || rank >= (unsigned)CS) { ok = false; break; }
            smids_in_cluster[rank] = smid;
        }
        if (!ok) continue;
        for (auto s : smids_in_cluster) if (s >= 1024) { ok = false; break; }
        if (!ok) continue;

        valid++;
        std::set<unsigned> grp(smids_in_cluster.begin(), smids_in_cluster.end());
        if ((int)grp.size() != CS) continue;  // duplicates = something wrong
        for (auto a : grp)
            for (auto b : grp)
                if (a != b) sm_peers_out[a].insert(b);
    }
    printf("  [CS=%d] valid clusters: %d / %d\n", CS, valid, n_clusters);
    return valid > 0;
}

// ---------------------------------------------------------------
// Extract peer groups from sm_peers map
// ---------------------------------------------------------------
static std::vector<std::set<unsigned>> extract_groups(
    const std::map<unsigned, std::set<unsigned>>& sm_peers, int expected_size)
{
    std::set<unsigned> assigned;
    std::vector<std::set<unsigned>> groups;

    for (auto& [sm, peers] : sm_peers) {
        if (assigned.count(sm)) continue;
        std::set<unsigned> grp = peers;
        grp.insert(sm);
        if ((int)grp.size() >= expected_size) {
            // Truncate if larger (shouldn't happen)
            while ((int)grp.size() > expected_size) grp.erase(grp.end());
        }
        groups.push_back(grp);
        for (auto s : grp) assigned.insert(s);
    }

    std::sort(groups.begin(), groups.end(),
              [](auto& a, auto& b){ return *a.begin() < *b.begin(); });
    return groups;
}

int main() {
    CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    int sm_count = prop.multiProcessorCount;

    // ================================================================
    // Part A: Cluster-2 to identify TPCs (consecutive pairs)
    // ================================================================
    printf("\n=== Part A: cluster-2 — TPC pairs ===\n");
    {
        std::map<unsigned, std::set<unsigned>> peers;
        launch_cluster_smid<2>(sm_count, 20, peers);
        auto groups = extract_groups(peers, 2);
        printf("  %zu TPC pairs:\n", groups.size());
        for (size_t i = 0; i < groups.size(); i++) {
            printf("    TPC-%zu: {", i);
            for (auto s : groups[i]) printf("%u,", s);
            printf("}  size=%zu\n", groups[i].size());
        }
    }

    // ================================================================
    // Part B: Cluster-4 — half-GPC groupings (2 TPCs per half-GPC)
    // ================================================================
    printf("\n=== Part B: cluster-4 — group-of-4 SM clusters ===\n");
    {
        std::map<unsigned, std::set<unsigned>> peers;
        launch_cluster_smid<4>(sm_count, 30, peers);

        // Show peer-set-size histogram
        std::map<int, int> hist;
        for (auto& [sm, p] : peers) hist[(int)p.size() + 1]++;
        printf("  Peer group size distribution:\n");
        for (auto& [sz, cnt] : hist)
            printf("    size=%d: %d SMs\n", sz, cnt);

        // Show unique group-of-4 patterns
        std::set<std::set<unsigned>> unique4;
        for (auto& [sm, p] : peers) {
            std::set<unsigned> g = p; g.insert(sm);
            if ((int)g.size() == 4) unique4.insert(g);
        }
        printf("  Unique cluster-4 patterns: %zu\n", unique4.size());
        int i = 0;
        for (auto& g : unique4) {
            printf("    [%d] {", i++);
            for (auto s : g) printf("%u,", s);
            printf("}\n");
        }
    }

    // ================================================================
    // Part C: Cluster-8 — full GPC groupings (4 TPCs per GPC)
    // ================================================================
    printf("\n=== Part C: cluster-8 — full GPC groupings ===\n");
    {
        std::map<unsigned, std::set<unsigned>> peers;
        // Run many rounds — need enough to saturate all (sm/8) distinct clusters
        launch_cluster_smid<8>(sm_count, 60, peers);

        std::map<int, int> hist;
        for (auto& [sm, p] : peers) hist[(int)p.size() + 1]++;
        printf("  Peer group size distribution:\n");
        for (auto& [sz, cnt] : hist)
            printf("    size=%d: %d SMs\n", sz, cnt);

        // Find SMs whose peer set has exactly 7 others (= a solid GPC group)
        std::set<std::set<unsigned>> unique8;
        int incomplete = 0;
        for (auto& [sm, p] : peers) {
            std::set<unsigned> g = p; g.insert(sm);
            if ((int)g.size() == 8) unique8.insert(g);
            else incomplete++;
        }
        printf("  Unique cluster-8 patterns (complete GPCs): %zu\n", unique8.size());
        printf("  SMs with incomplete peer set: %d\n", incomplete);
        int i = 0;
        for (auto& g : unique8) {
            printf("    GPC-%d: {", i++);
            for (auto s : g) printf("%u,", s);
            printf("}\n");
        }

        // Determine unassigned SMs (partial GPC)
        std::set<unsigned> assigned;
        for (auto& g : unique8)
            for (auto s : g) assigned.insert(s);
        std::set<unsigned> unassigned;
        for (int s = 0; s < sm_count; s++)
            if (!assigned.count(s)) unassigned.insert(s);
        if (!unassigned.empty()) {
            printf("  Unassigned SMs (partial GPC or not seen): {");
            for (auto s : unassigned) printf("%u,", s);
            printf("}\n");
        }

        printf("\n  SUMMARY:\n");
        printf("    Full GPCs (8 SMs): %zu\n", unique8.size());
        printf("    Unassigned SMs:    %zu\n", unassigned.size());
        printf("    Total SM count:    %d\n", sm_count);
        if (!unassigned.empty()) {
            printf("    Partial GPC size:  %zu\n", unassigned.size());
            printf("    Total GPC count:   %zu\n", unique8.size() + 1);
        } else {
            printf("    Total GPC count:   %zu\n", unique8.size());
        }
    }

    printf("\nDone.\n");
    return 0;
}
