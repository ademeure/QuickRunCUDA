/*
 * GPC Topology Investigation for B300 sm_103a (SXM6 AC)
 *
 * Strategy:
 *   1. Launch many blocks (>=296), read %smid and %nsmid from each block.
 *   2. Use cooperative groups clusters (sizes 2,4,8,16) to see which SMIDs
 *      co-locate within a cluster — that reveals GPC (or TPC) grouping.
 *   3. Analyze grouping patterns to determine GPC count and SM-per-GPC.
 *
 * Compile:
 *   nvcc -arch=sm_103a -O3 -o gpc_topology gpc_topology.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Kernel 1: simple smid + nsmid read
// ============================================================
__global__ void kernel_smid(unsigned* out_smid, int n_blocks) {
    int bid = blockIdx.x;
    if (bid >= n_blocks) return;
    unsigned smid, nsmid;
    asm volatile("mov.u32 %0, %%smid;"  : "=r"(smid)  :);
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsmid) :);
    out_smid[bid * 2 + 0] = smid;
    out_smid[bid * 2 + 1] = nsmid;
}

// ============================================================
// Kernel 2: cluster-aware smid collection
// Blocks within the same cluster write their smid to a shared array
// ============================================================
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template<int CLUSTER_SIZE>
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) kernel_cluster_smid(
    unsigned* out,   // [gridDim blocks][CLUSTER_SIZE]
    int n_blocks_total)
{
    cg::cluster_group cluster = cg::this_cluster();

    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid) :);

    // Rank within cluster
    unsigned rank = cluster.thread_rank() / blockDim.x;  // block rank in cluster

    int cluster_id = (int)(blockIdx.x / CLUSTER_SIZE);
    // Use distributed shared memory to gather smids across cluster
    __shared__ unsigned my_smid;
    my_smid = smid;

    // Synchronize cluster so all blocks have written their smid
    cluster.sync();

    // Each block in the cluster reads all peer smids via cluster.map_shared_rank
    if (threadIdx.x == 0) {
        int base = cluster_id * CLUSTER_SIZE;
        for (int r = 0; r < CLUSTER_SIZE; r++) {
            unsigned* peer_smid = cluster.map_shared_rank(&my_smid, r);
            if (base + r < n_blocks_total) {
                out[base + r] = *peer_smid;
            }
        }
    }
}

// ============================================================
// Kernel 3: store (smid, blockIdx.x) pairs — very simple
// Launch many more blocks than SMs so each SM gets hit multiple times
// ============================================================
__global__ void kernel_smid_simple(unsigned* out, int total) {
    int bid = blockIdx.x;
    if (bid >= total) return;
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid) :);
    out[bid] = smid;
}

// ============================================================
// Analysis helpers
// ============================================================
static void print_smid_histogram(const std::vector<unsigned>& smids) {
    std::map<unsigned, int> hist;
    for (auto s : smids) hist[s]++;
    unsigned maxsm = hist.rbegin()->first;
    printf("  SM ID histogram (id: count):\n");
    // Print in rows of 16
    for (unsigned id = 0; id <= maxsm; id++) {
        int cnt = hist.count(id) ? hist[id] : 0;
        if (id % 16 == 0) printf("  [%3u-%3u]:", id, std::min(id+15, maxsm));
        printf(" %2d", cnt);
        if ((id % 16) == 15 || id == maxsm) printf("\n");
    }
}

static void analyze_cluster_grouping(
    const std::vector<unsigned>& cluster_smids,  // length = n_clusters * cluster_size
    int cluster_size, int n_clusters)
{
    printf("\n  Cluster SMID groupings (cluster_size=%d):\n", cluster_size);

    // For each cluster, gather the unique SMIDs and find stride/grouping
    std::map<std::set<unsigned>, int> pattern_count;
    for (int c = 0; c < n_clusters; c++) {
        std::set<unsigned> group;
        for (int r = 0; r < cluster_size; r++) {
            unsigned sid = cluster_smids[c * cluster_size + r];
            // Filter out invalid (0xffffffff can appear if kernel not reached)
            if (sid < 1024) group.insert(sid);
        }
        if (!group.empty()) {
            pattern_count[group]++;
        }
    }

    // Show unique cluster patterns
    int shown = 0;
    std::vector<unsigned> first_sm_of_each_cluster;
    std::map<unsigned, std::set<unsigned>> gpc_guess; // first_sm -> set of all peers

    for (auto& [grp, cnt] : pattern_count) {
        if (shown < 30) {
            printf("    pattern x%d: {", cnt);
            for (auto s : grp) printf("%u,", s);
            printf("}\n");
            shown++;
        }
        unsigned first = *grp.begin();
        for (auto s : grp) gpc_guess[first].insert(s);
    }

    if (shown >= 30) printf("    ... (showing first 30 of %zu patterns)\n", pattern_count.size());

    // Compute SM stride within clusters
    printf("\n  SM stride analysis:\n");
    std::map<unsigned, int> stride_hist;
    for (auto& [grp, cnt] : pattern_count) {
        std::vector<unsigned> sorted_grp(grp.begin(), grp.end());
        for (size_t i = 1; i < sorted_grp.size(); i++) {
            unsigned stride = sorted_grp[i] - sorted_grp[0];
            stride_hist[stride] += cnt;
        }
    }
    for (auto& [stride, cnt] : stride_hist) {
        printf("    stride=%u appears in %d cluster instances\n", stride, cnt);
    }
}

int main() {
    int device = 0;
    CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Compute: %d.%d\n", prop.major, prop.minor);
    int sm_count = prop.multiProcessorCount;

    // ================================================================
    // Test 1: Basic smid/nsmid collection
    // ================================================================
    printf("\n=== Test 1: Basic smid/nsmid (1 block per SM) ===\n");
    {
        int n_blocks = sm_count;  // ideally 1 per SM
        unsigned* d_out;
        CHECK(cudaMalloc(&d_out, n_blocks * 2 * sizeof(unsigned)));
        CHECK(cudaMemset(d_out, 0xff, n_blocks * 2 * sizeof(unsigned)));

        kernel_smid<<<n_blocks, 1>>>(d_out, n_blocks);
        CHECK(cudaDeviceSynchronize());

        std::vector<unsigned> h(n_blocks * 2);
        CHECK(cudaMemcpy(h.data(), d_out, n_blocks * 2 * sizeof(unsigned), cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_out));

        std::set<unsigned> unique_smids;
        unsigned max_nsmid = 0;
        for (int i = 0; i < n_blocks; i++) {
            unique_smids.insert(h[i*2+0]);
            max_nsmid = std::max(max_nsmid, h[i*2+1]);
        }
        printf("  Unique SMIDs seen: %zu\n", unique_smids.size());
        printf("  Max nsmid reported: %u\n", max_nsmid);
        printf("  Min SMID: %u, Max SMID: %u\n", *unique_smids.begin(), *unique_smids.rbegin());
        printf("  SM IDs:");
        int idx = 0;
        for (auto s : unique_smids) {
            if (idx % 16 == 0) printf("\n    ");
            printf("%3u", s);
            idx++;
        }
        printf("\n");
    }

    // ================================================================
    // Test 2: Large launch — saturate all SMs multiple times
    // ================================================================
    printf("\n=== Test 2: Large launch (%d blocks) to see all SM IDs ===\n", sm_count * 4);
    {
        int n_blocks = sm_count * 4;
        unsigned* d_out;
        CHECK(cudaMalloc(&d_out, n_blocks * sizeof(unsigned)));
        CHECK(cudaMemset(d_out, 0xff, n_blocks * sizeof(unsigned)));

        kernel_smid_simple<<<n_blocks, 32>>>(d_out, n_blocks);
        CHECK(cudaDeviceSynchronize());

        std::vector<unsigned> h(n_blocks);
        CHECK(cudaMemcpy(h.data(), d_out, n_blocks * sizeof(unsigned), cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_out));

        std::set<unsigned> unique_smids;
        for (auto s : h) if (s < 1024) unique_smids.insert(s);
        printf("  Unique SMIDs seen: %zu\n", unique_smids.size());
        printf("  SM IDs:");
        int idx = 0;
        for (auto s : unique_smids) {
            if (idx % 16 == 0) printf("\n    ");
            printf("%3u ", s);
            idx++;
        }
        printf("\n");

        // Print histogram
        print_smid_histogram(std::vector<unsigned>(h.begin(), h.end()));
    }

    // ================================================================
    // Test 3: Cluster sizes to understand GPC grouping
    // ================================================================
    // Try cluster sizes 2, 4, 8, 16 — valid cluster sizes on sm_103a
    for (int cs : {2, 4, 8, 16}) {
        printf("\n=== Test 3: Cluster size %d (to probe GPC grouping) ===\n", cs);

        // Need n_clusters such that we cover all SMs; each cluster occupies cs SMs
        // Use enough clusters to see patterns
        int n_clusters = (sm_count / cs) * 4;  // 4 rounds
        if (n_clusters < 4) n_clusters = 4;
        int n_blocks = n_clusters * cs;

        unsigned* d_out;
        CHECK(cudaMalloc(&d_out, n_blocks * sizeof(unsigned)));
        CHECK(cudaMemset(d_out, 0xff, n_blocks * sizeof(unsigned)));

        cudaLaunchConfig_t config = {};
        config.gridDim  = dim3(n_blocks, 1, 1);
        config.blockDim = dim3(1, 1, 1);
        config.dynamicSmemBytes = 0;
        config.stream = 0;

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = cs;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        config.attrs = attrs;
        config.numAttrs = 1;

        void* kargs[] = {&d_out, &n_blocks};
        cudaError_t err = cudaErrorInvalidValue;
        if (cs == 2)       err = cudaLaunchKernelExC(&config, (void*)kernel_cluster_smid<2>,  kargs);
        else if (cs == 4)  err = cudaLaunchKernelExC(&config, (void*)kernel_cluster_smid<4>,  kargs);
        else if (cs == 8)  err = cudaLaunchKernelExC(&config, (void*)kernel_cluster_smid<8>,  kargs);
        else if (cs == 16) err = cudaLaunchKernelExC(&config, (void*)kernel_cluster_smid<16>, kargs);

        if (err != cudaSuccess) {
            printf("  cluster size %d: FAILED (%s)\n", cs, cudaGetErrorString(err));
            CHECK(cudaFree(d_out));
            continue;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  cluster size %d sync: FAILED (%s)\n", cs, cudaGetErrorString(err));
            CHECK(cudaFree(d_out));
            continue;
        }

        std::vector<unsigned> h(n_blocks);
        CHECK(cudaMemcpy(h.data(), d_out, n_blocks * sizeof(unsigned), cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_out));

        // Filter valid entries
        int valid = 0;
        for (auto s : h) if (s < 1024) valid++;
        printf("  Valid entries: %d / %d\n", valid, n_blocks);

        analyze_cluster_grouping(h, cs, n_clusters);
    }

    // ================================================================
    // Test 4: Run 1024 blocks and map ALL smid occurrences
    // ================================================================
    printf("\n=== Test 4: 1024-block sweep — full smid distribution ===\n");
    {
        int n_blocks = 1024;
        unsigned* d_out;
        CHECK(cudaMalloc(&d_out, n_blocks * sizeof(unsigned)));
        CHECK(cudaMemset(d_out, 0xff, n_blocks * sizeof(unsigned)));

        kernel_smid_simple<<<n_blocks, 32>>>(d_out, n_blocks);
        CHECK(cudaDeviceSynchronize());

        std::vector<unsigned> h(n_blocks);
        CHECK(cudaMemcpy(h.data(), d_out, n_blocks * sizeof(unsigned), cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_out));

        std::set<unsigned> unique_smids;
        for (auto s : h) if (s < 1024) unique_smids.insert(s);
        printf("  Unique SMIDs: %zu\n", unique_smids.size());
        unsigned maxsm = *unique_smids.rbegin();
        printf("  Max SMID: %u\n", maxsm);

        // Print the full smid list grouped by 16
        printf("  Full SMID list:\n");
        int idx = 0;
        for (auto s : unique_smids) {
            if (idx % 16 == 0) printf("  [group %d, offset %u]:", idx/16, s);
            printf(" %u", s);
            idx++;
            if (idx % 16 == 0) printf("\n");
        }
        printf("\n");

        // Count gaps in the smid space
        printf("  Gaps in SM ID space:\n");
        unsigned prev = *unique_smids.begin();
        bool had_gaps = false;
        for (auto s : unique_smids) {
            if (s > prev + 1 && s != *unique_smids.begin()) {
                printf("    gap: [%u .. %u) missing %u IDs\n", prev+1, s, s - prev - 1);
                had_gaps = true;
            }
            prev = s;
        }
        if (!had_gaps) printf("    none — SM IDs are contiguous\n");

        // Group by 16 to infer GPC structure
        printf("\n  SM count per group-of-16:\n");
        std::map<unsigned, int> group16;
        for (auto s : unique_smids) group16[s / 16]++;
        for (auto& [g, cnt] : group16) {
            printf("    group [%u*16=%u .. %u*16+15=%u]: %d SMs\n",
                   g, g*16, g, g*16+15, cnt);
        }

        // Group by 8
        printf("\n  SM count per group-of-8:\n");
        std::map<unsigned, int> group8;
        for (auto s : unique_smids) group8[s / 8]++;
        for (auto& [g, cnt] : group8) {
            printf("    group-8 [%u]: SMs %u-%u => %d\n", g, g*8, g*8+7, cnt);
        }
    }

    // ================================================================
    // Test 5: Cluster size 8 — exhaustive round-robin to pin down
    //         which groups of 8 SMs share a cluster (= GPC or slice)
    // ================================================================
    printf("\n=== Test 5: Cluster-8 exhaustive — which 8 SMs co-cluster? ===\n");
    {
        int cs = 8;
        // Launch enough rounds to fill each SM many times
        int n_clusters = (sm_count / cs) * 8;
        int n_blocks = n_clusters * cs;

        unsigned* d_out;
        CHECK(cudaMalloc(&d_out, n_blocks * sizeof(unsigned)));
        CHECK(cudaMemset(d_out, 0xff, n_blocks * sizeof(unsigned)));

        cudaLaunchConfig_t config = {};
        config.gridDim  = dim3(n_blocks, 1, 1);
        config.blockDim = dim3(1, 1, 1);
        config.dynamicSmemBytes = 0;
        config.stream = 0;

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = cs;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        config.attrs = attrs;
        config.numAttrs = 1;

        void* kargs5[] = {&d_out, &n_blocks};
        cudaError_t err = cudaLaunchKernelExC(&config,
            (void*)kernel_cluster_smid<8>, kargs5);

        if (err != cudaSuccess) {
            printf("  FAILED: %s\n", cudaGetErrorString(err));
            CHECK(cudaFree(d_out));
        } else {
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("  sync FAILED: %s\n", cudaGetErrorString(err));
                CHECK(cudaFree(d_out));
            } else {
                std::vector<unsigned> h(n_blocks);
                CHECK(cudaMemcpy(h.data(), d_out, n_blocks * sizeof(unsigned), cudaMemcpyDeviceToHost));
                CHECK(cudaFree(d_out));

                // Build map: for each SM, which other SMs appear in same cluster?
                std::map<unsigned, std::set<unsigned>> sm_peers;
                for (int c = 0; c < n_clusters; c++) {
                    std::set<unsigned> grp;
                    for (int r = 0; r < cs; r++) {
                        unsigned sid = h[c * cs + r];
                        if (sid < 1024) grp.insert(sid);
                    }
                    if ((int)grp.size() == cs) {
                        for (auto a : grp)
                            for (auto b : grp)
                                if (a != b) sm_peers[a].insert(b);
                    }
                }

                // Find unique peer-sets (= GPCs or TPC groups)
                std::map<std::set<unsigned>, unsigned> gpc_groups;  // peer_set -> representative_sm
                for (auto& [sm, peers] : sm_peers) {
                    std::set<unsigned> full_grp = peers;
                    full_grp.insert(sm);
                    gpc_groups[full_grp] = *full_grp.begin();
                }

                printf("  Distinct SM groups that always cluster together:\n");
                int gi = 0;
                for (auto& [grp, rep] : gpc_groups) {
                    printf("  GPC-%d: {", gi++);
                    for (auto s : grp) printf("%u,", s);
                    printf("}  size=%zu\n", grp.size());
                }
                printf("  => %zu distinct groups\n", gpc_groups.size());
            }
        }
    }

    printf("\nDone.\n");
    return 0;
}
