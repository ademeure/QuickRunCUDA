/*
 * GPC Topology Investigation Part 2 — B300 sm_103a
 *
 * Focus: use cluster-8 exhaustively to find which groups of 8 SMs
 *        are physically co-located (= within one GPC).
 *
 * Also: try cluster-4 exhaustively and check TPC (pairs) structure.
 *
 * Compile: nvcc -arch=sm_103a -O3 -o gpc_topology2 gpc_topology2.cu
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

// ============================================================
// Cluster kernel — gathers SMIDs from all blocks in a cluster
// Uses distributed shared memory approach.
// ============================================================
template<int CLUSTER_SIZE>
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) kernel_cluster_smid(
    unsigned* out,   // [n_clusters * CLUSTER_SIZE]
    int n_blocks_total)
{
    cg::cluster_group cluster = cg::this_cluster();

    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid) :);

    __shared__ unsigned my_smid;
    if (threadIdx.x == 0) my_smid = smid;
    __syncthreads();

    cluster.sync();

    // Only thread 0 of each block gathers data
    if (threadIdx.x == 0) {
        int cluster_id = (int)(blockIdx.x / CLUSTER_SIZE);
        // Each block reads all peer smids
        // We'll have block rank 0 write all of them
        unsigned my_rank = cluster.block_rank();
        if (my_rank == 0) {
            int base = cluster_id * CLUSTER_SIZE;
            for (int r = 0; r < CLUSTER_SIZE; r++) {
                if (base + r < n_blocks_total) {
                    unsigned* peer_smid = cluster.map_shared_rank(&my_smid, r);
                    out[base + r] = *peer_smid;
                }
            }
        }
    }
}

// ============================================================
// Run many rounds of cluster launches to gather GPC groupings
// ============================================================
template<int CS>
static bool run_cluster_rounds(int sm_count, int n_rounds,
                                std::map<unsigned, std::set<unsigned>>& sm_peers_out)
{
    int n_clusters = (sm_count / CS) * n_rounds;
    int n_blocks   = n_clusters * CS;

    unsigned* d_out;
    if (cudaMalloc(&d_out, n_blocks * sizeof(unsigned)) != cudaSuccess) return false;
    cudaMemset(d_out, 0xff, n_blocks * sizeof(unsigned));

    cudaLaunchConfig_t config = {};
    config.gridDim  = dim3(n_blocks, 1, 1);
    config.blockDim = dim3(1, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = 0;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CS;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    void* kargs[] = {&d_out, &n_blocks};
    cudaError_t err = cudaLaunchKernelExC(&config, (void*)kernel_cluster_smid<CS>, kargs);
    if (err != cudaSuccess) {
        printf("  launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  sync failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return false;
    }

    std::vector<unsigned> h(n_blocks);
    cudaMemcpy(h.data(), d_out, n_blocks * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    // Build peer map
    int valid_clusters = 0;
    for (int c = 0; c < n_clusters; c++) {
        std::set<unsigned> grp;
        bool ok = true;
        for (int r = 0; r < CS; r++) {
            unsigned s = h[c * CS + r];
            if (s >= 1024) { ok = false; break; }
            grp.insert(s);
        }
        if (ok && (int)grp.size() == CS) {
            valid_clusters++;
            for (auto a : grp)
                for (auto b : grp)
                    if (a != b) sm_peers_out[a].insert(b);
        }
    }
    printf("  Valid clusters: %d / %d\n", valid_clusters, n_clusters);
    return true;
}

// ============================================================
// Partition SM set into groups based on peer relationships
// ============================================================
static std::vector<std::set<unsigned>> extract_groups(
    const std::map<unsigned, std::set<unsigned>>& sm_peers, int expected_size)
{
    std::vector<std::set<unsigned>> groups;
    std::set<unsigned> assigned;

    for (auto& [sm, peers] : sm_peers) {
        if (assigned.count(sm)) continue;
        std::set<unsigned> grp = peers;
        grp.insert(sm);
        if ((int)grp.size() == expected_size) {
            groups.push_back(grp);
            for (auto s : grp) assigned.insert(s);
        }
    }

    // Any unassigned SMs form singleton or partial groups
    for (auto& [sm, peers] : sm_peers) {
        if (!assigned.count(sm)) {
            std::set<unsigned> grp = peers;
            grp.insert(sm);
            groups.push_back(grp);
            for (auto s : grp) assigned.insert(s);
        }
    }

    std::sort(groups.begin(), groups.end(),
              [](auto& a, auto& b){ return *a.begin() < *b.begin(); });
    return groups;
}

int main() {
    int device = 0;
    CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    int sm_count = prop.multiProcessorCount;

    // ================================================================
    // Section A: TPC identification via cluster-2
    // ================================================================
    printf("\n=== Section A: TPC pairs via cluster-2 (%d rounds) ===\n", 20);
    {
        std::map<unsigned, std::set<unsigned>> sm_peers;
        if (run_cluster_rounds<2>(sm_count, 20, sm_peers)) {
            auto groups = extract_groups(sm_peers, 2);
            printf("  Found %zu TPC pairs:\n", groups.size());
            int i = 0;
            for (auto& g : groups) {
                printf("    TPC-%d: {", i++);
                for (auto s : g) printf("%u,", s);
                printf("}\n");
            }
        }
    }

    // ================================================================
    // Section B: GPC identification via cluster-4
    // ================================================================
    printf("\n=== Section B: GPC identification via cluster-4 (%d rounds) ===\n", 30);
    {
        std::map<unsigned, std::set<unsigned>> sm_peers;
        if (run_cluster_rounds<4>(sm_count, 30, sm_peers)) {
            // Find most common group size
            std::map<int, int> size_hist;
            for (auto& [sm, peers] : sm_peers) {
                size_hist[(int)peers.size() + 1]++;
            }
            printf("  Peer group size histogram:\n");
            for (auto& [sz, cnt] : size_hist) {
                printf("    size=%d: %d SMs\n", sz, cnt);
            }

            auto groups = extract_groups(sm_peers, 4);
            printf("  Found %zu groups-of-4 (2 TPCs per GPC):\n", groups.size());
            int i = 0;
            for (auto& g : groups) {
                printf("    group4-%d: {", i++);
                for (auto s : g) printf("%u,", s);
                printf("}  size=%zu\n", g.size());
            }
        }
    }

    // ================================================================
    // Section C: GPC identification via cluster-8
    // Each GPC on Blackwell-class has 8 SMs (4 TPCs × 2 SMs/TPC)
    // ================================================================
    printf("\n=== Section C: GPC identification via cluster-8 (%d rounds) ===\n", 40);
    {
        std::map<unsigned, std::set<unsigned>> sm_peers;
        if (run_cluster_rounds<8>(sm_count, 40, sm_peers)) {
            // Find most common peer-set size
            std::map<int, int> size_hist;
            for (auto& [sm, peers] : sm_peers) {
                size_hist[(int)peers.size() + 1]++;
            }
            printf("  Peer group size histogram (size = #peers+1):\n");
            for (auto& [sz, cnt] : size_hist) {
                printf("    size=%d: %d SMs\n", sz, cnt);
            }

            // Extract groups of 8
            auto groups = extract_groups(sm_peers, 8);
            printf("\n  Found %zu groups-of-8 (= GPCs if hardware GPC = 8 SMs):\n", groups.size());
            int i = 0;
            for (auto& g : groups) {
                if ((int)g.size() == 8) {
                    printf("    GPC-%d: {", i);
                    for (auto s : g) printf("%u,", s);
                    printf("}\n");
                } else {
                    printf("    partial-%d (size=%zu): {", i, g.size());
                    for (auto s : g) printf("%u,", s);
                    printf("}\n");
                }
                i++;
            }

            // Count full GPCs vs partial
            int full = 0, partial = 0;
            for (auto& g : groups) {
                if ((int)g.size() == 8) full++;
                else partial++;
            }
            printf("  Full GPCs (8 SMs): %d, partial: %d\n", full, partial);
            printf("  Total SMs covered: %d\n", sm_count);
            printf("\n  Interpretation:\n");
            printf("    If full=%d and partial=%d:\n", full, partial);
            printf("      => %d GPCs total (%d×8 + %d×smaller)\n",
                   full + partial, full, partial);
        }
    }

    // ================================================================
    // Section D: Verify with cluster-4 which SMs share a "half-GPC"
    //            The cluster assignment reveals TPC groupings within GPC
    // ================================================================
    printf("\n=== Section D: Detailed cluster-4 pattern analysis ===\n");
    {
        // Run fewer rounds but print each unique cluster pattern once
        std::map<unsigned, std::set<unsigned>> sm_peers;
        run_cluster_rounds<4>(sm_count, 10, sm_peers);

        // Build unique groups
        std::set<std::set<unsigned>> unique_groups;
        for (auto& [sm, peers] : sm_peers) {
            std::set<unsigned> grp = peers;
            grp.insert(sm);
            if ((int)grp.size() == 4) unique_groups.insert(grp);
        }

        printf("  Unique cluster-4 groups seen: %zu\n", unique_groups.size());
        int i = 0;
        for (auto& g : unique_groups) {
            printf("    [%d] {", i++);
            for (auto s : g) printf("%u,", s);
            printf("}\n");
        }
    }

    printf("\nDone.\n");
    return 0;
}
