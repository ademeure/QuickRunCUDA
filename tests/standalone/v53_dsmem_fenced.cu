// V53: Settle whether DSMEM write BW is ISSUE-RATE or COMPLETION-RATE.
//
// V21 measured 560 GB/s aggregate WRITE ceiling, but its `push_ring_wr` did
// `clock64; for (st.shared::cluster) ; clock64;` with NO fence between the
// stores and the closing clock64. PTX `st.shared::cluster.u32` is fire-and-forget
// — the timer ends as soon as the last store enters the queue.
//
// This test runs TWO variants side-by-side, identical except for the fence:
//   UNFENCED: stores ... clock64-end                  (reproduces V21)
//   FENCED:   stores ... fence.sc.cluster
//                      ... barrier.cluster.arrive
//                      ... barrier.cluster.wait
//                      ... clock64-end                (true completion timing)
//
// Sweeps ILP={4,8,16} and TILE={1KB,4KB,16KB,64KB} per CTA per outer iter.
// 144 CTAs (18 clusters × 8) leaving 4 SMs idle. __launch_bounds__(128, 1).
//
// Anti-DCE: derived value of written region is XOR-collapsed and written to global.
// Anti-LICM: store addresses depend on threadIdx AND iteration index `it`.
//
// Build: nvcc -arch=sm_103a -O3 v53_dsmem_fenced.cu -o /tmp/v53

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define CLUSTER 8
#define THREADS 128
#define SMEM_DW (16 * 1024)   // 64 KB / 4 = 16384 u32; biggest tile fits

// Templated DSMEM write kernel.
//   FENCED=0 reproduces V21's missing-fence pattern.
//   FENCED=1 inserts fence.sc.cluster + barrier.cluster.arrive/wait BEFORE end-clock.
//   ILP    = u32 stores per inner-k unroll
//   INNER  = inner passes per outer iter (TILE_DW = INNER * THREADS * ILP)
//   N_ITER = outer iters
template<int FENCED, int ILP, int INNER, int N_ITER>
__global__ __cluster_dims__(CLUSTER, 1, 1) __launch_bounds__(THREADS, 1)
void v53_write(unsigned long long* out_cy, unsigned* out_sink, unsigned seed) {
    __shared__ __align__(16) unsigned smem[SMEM_DW];

    int tid = threadIdx.x;

    // Touch the local smem to force allocation and zero it.
    #pragma unroll 1
    for (int i = tid; i < SMEM_DW; i += THREADS) smem[i] = 0;
    __syncthreads();

    // Cluster info
    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));

    // Initial cluster sync so all peers' SMEM is allocated before we mapa.
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    // Map peer (ring: my_cta -> (my_cta+1) mod CLUSTER)
    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned target_cta = (my_cta + 1u) % CLUSTER;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    constexpr int CHUNK_DWS = THREADS * ILP;          // dwords per inner-k loop pass
    constexpr int TILE_DW   = INNER * CHUNK_DWS;      // total dwords/CTA/outer-iter
    static_assert(TILE_DW <= SMEM_DW, "TILE_DW must fit in SMEM_DW");
    constexpr unsigned MASK = (SMEM_DW - 1);          // SMEM_DW is power of 2 (16384)

    // One sync barrier so all CTAs start the timed region together.
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITER; it++) {
        #pragma unroll 1
        for (int j = 0; j < INNER; j++) {
            // Anti-LICM: address depends on tid AND (it,j)
            unsigned base_off = (unsigned)((tid + j * THREADS + it * 31u) & MASK);
            #pragma unroll
            for (int k = 0; k < ILP; k++) {
                unsigned off  = (base_off + (unsigned)(k * THREADS)) & MASK;
                unsigned addr = peer_base + off * 4u;
                unsigned val  = seed + (unsigned)(it * 17u + tid * 131u + k * 7u + j * 257u);
                asm volatile("st.shared::cluster.u32 [%0], %1;"
                             :: "r"(addr), "r"(val) : "memory");
            }
        }
    }

    // Conditional completion fence + cluster barrier
    if (FENCED) {
        asm volatile("fence.sc.cluster;" ::: "memory");
        asm volatile("barrier.cluster.arrive;" ::: "memory");
        asm volatile("barrier.cluster.wait;"  ::: "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // Final sync
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    // Anti-DCE: XOR-collapse local smem and stash to global.
    unsigned acc = 0;
    #pragma unroll 1
    for (int i = tid; i < SMEM_DW; i += THREADS) acc ^= smem[i];
    // Reduce within block via simple shuffle
    for (int o = 16; o > 0; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
    if ((tid & 31) == 0) {
        atomicXor(&out_sink[blockIdx.x], acc);
    }
    if (tid == 0) out_cy[blockIdx.x] = (t1 - t0);
}

// ------------------- host bench harness -------------------

template<typename Kernel>
static double run_one(Kernel kernel, int n_blocks, int n_runs,
                      double clock_ghz, double bytes_total,
                      unsigned long long* d_cy, unsigned* d_sink, unsigned seed_base,
                      double* out_avg_cy = nullptr) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(n_blocks, 1, 1);
    cfg.blockDim = dim3(THREADS, 1, 1);
    cfg.stream = 0;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    // Warm-up
    cudaLaunchKernelEx(&cfg, kernel, d_cy, d_sink, 1u);
    cudaDeviceSynchronize();
    cudaGetLastError();

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    double best_ms = 1e30;
    double cy_sum = 0;
    int got_cy = 0;
    for (int r = 0; r < n_runs; r++) {
        cudaEventRecord(e0);
        cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, d_cy, d_sink, seed_base + (unsigned)r);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        if (err) { cudaGetLastError(); continue; }
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
        // Pull cy from CTA 0 of cluster 0
        unsigned long long cy0;
        cudaMemcpy(&cy0, d_cy, sizeof(cy0), cudaMemcpyDeviceToHost);
        cy_sum += (double)cy0;
        got_cy++;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (out_avg_cy && got_cy > 0) *out_avg_cy = cy_sum / got_cy;

    return best_ms;  // best wall-time across runs
}

template<int ILP, int INNER, int N_ITER>
static void bench_pair(const char* size_label, int n_blocks, int n_runs,
                       double clock_ghz,
                       unsigned long long* d_cy, unsigned* d_sink) {
    constexpr int TILE_DW = INNER * THREADS * ILP;
    double bytes_per_cta = (double)TILE_DW * 4.0 * (double)N_ITER;
    double bytes_total   = bytes_per_cta * (double)n_blocks;

    double cy_uf = 0, cy_f = 0;
    double ms_uf = run_one(v53_write<0, ILP, INNER, N_ITER>, n_blocks, n_runs,
                           clock_ghz, bytes_total, d_cy, d_sink, 100u, &cy_uf);
    double ms_f  = run_one(v53_write<1, ILP, INNER, N_ITER>, n_blocks, n_runs,
                           clock_ghz, bytes_total, d_cy, d_sink, 200u, &cy_f);

    double bw_uf_agg = bytes_total / (ms_uf / 1e3) / 1e9; // GB/s aggregate cluster-write
    double bw_f_agg  = bytes_total / (ms_f  / 1e3) / 1e9;
    double bw_uf_clu = bw_uf_agg / (n_blocks / CLUSTER);  // per-cluster
    double bw_f_clu  = bw_f_agg  / (n_blocks / CLUSTER);
    double ratio = (bw_f_agg > 0) ? bw_uf_agg / bw_f_agg : 0.0;

    // V21-style clock64 per-CTA BW: stores per CTA / cy_per_CTA / clock_ghz, × 4 B/store, × CLUSTER
    double stores_per_cta = (double)TILE_DW * (double)N_ITER;
    double bw_uf_cy_per_cta = (cy_uf > 0) ? stores_per_cta * 4.0 * clock_ghz / cy_uf : 0;
    double bw_f_cy_per_cta  = (cy_f  > 0) ? stores_per_cta * 4.0 * clock_ghz / cy_f  : 0;

    printf("  ILP=%-2d %-6s  uf wall=%6.2fms %5.1f GB/s/CTA (%5.0f agg)  cy=%9.0f cyBW=%5.1f GB/s/CTA   "
           "|  f wall=%6.2fms %5.1f GB/s/CTA (%5.0f agg)  cy=%9.0f cyBW=%5.1f GB/s/CTA   wallR=%.2fx\n",
           ILP, size_label,
           ms_uf, bw_uf_clu / CLUSTER, bw_uf_agg, cy_uf, bw_uf_cy_per_cta,
           ms_f , bw_f_clu  / CLUSTER, bw_f_agg , cy_f , bw_f_cy_per_cta,
           ratio);
}

int main() {
    CK(cudaSetDevice(0));

    int dev;
    cudaGetDevice(&dev);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    int clk_khz = 0;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev);
    double clock_ghz = clk_khz / 1.0e6;

    const int N_CLUSTERS = 18;            // 144 CTAs, leaves 4 SMs idle
    const int n_blocks   = N_CLUSTERS * CLUSTER;
    const int n_runs     = 5;

    printf("=== V53 DSMEM write SoL: UNFENCED vs FENCED (B300, sm_103a) ===\n");
    printf("SMs=%d, clusters=%d (CLUSTER=%d), CTAs=%d, threads/CTA=%d, peak clock=%.3f GHz\n",
           sm_count, N_CLUSTERS, CLUSTER, n_blocks, THREADS, clock_ghz);
    printf("Each row: same TILE/ILP run twice — uf=no fence (V21 mode), f=fence.sc.cluster + barrier.cluster\n");
    printf("Bytes counted = TILE_DW * 4 * N_ITER per CTA, summed across all 144 CTAs\n\n");

    unsigned long long* d_cy;
    unsigned* d_sink;
    CK(cudaMalloc(&d_cy, sizeof(*d_cy) * n_blocks));
    CK(cudaMalloc(&d_sink, sizeof(*d_sink) * n_blocks));
    CK(cudaMemset(d_cy, 0, sizeof(*d_cy) * n_blocks));
    CK(cudaMemset(d_sink, 0, sizeof(*d_sink) * n_blocks));

    // ---- TILE size sweep, ILP sweep ----
    // CHUNK_DWS = THREADS * ILP, TILE_DW = INNER * CHUNK_DWS, byte-tile = 4*TILE_DW
    //   ILP=4  : chunk = 512 dw = 2 KB
    //   ILP=8  : chunk = 1024 dw = 4 KB
    //   ILP=16 : chunk = 2048 dw = 8 KB
    // Pick N_ITER large enough for ~10-30 ms wall.

    printf("=== TILE = 2 KB / CTA / iter ===\n");
    bench_pair<4 , 1, 50000>("2KB", n_blocks, n_runs, clock_ghz, d_cy, d_sink);

    printf("\n=== TILE = 4 KB / CTA / iter ===\n");
    bench_pair<4 , 2, 30000>("4KB", n_blocks, n_runs, clock_ghz, d_cy, d_sink);
    bench_pair<8 , 1, 30000>("4KB", n_blocks, n_runs, clock_ghz, d_cy, d_sink);

    printf("\n=== TILE = 8 KB / CTA / iter ===\n");
    bench_pair<4 , 4, 15000>("8KB", n_blocks, n_runs, clock_ghz, d_cy, d_sink);
    bench_pair<8 , 2, 15000>("8KB", n_blocks, n_runs, clock_ghz, d_cy, d_sink);
    bench_pair<16, 1, 15000>("8KB", n_blocks, n_runs, clock_ghz, d_cy, d_sink);

    printf("\n=== TILE = 16 KB / CTA / iter ===\n");
    bench_pair<4 , 8,  8000>("16KB", n_blocks, n_runs, clock_ghz, d_cy, d_sink);
    bench_pair<8 , 4,  8000>("16KB", n_blocks, n_runs, clock_ghz, d_cy, d_sink);
    bench_pair<16, 2,  8000>("16KB", n_blocks, n_runs, clock_ghz, d_cy, d_sink);

    printf("\n=== TILE = 64 KB / CTA / iter (== SMEM region) ===\n");
    bench_pair<4 , 32, 2000>("64KB", n_blocks, n_runs, clock_ghz, d_cy, d_sink);
    bench_pair<8 , 16, 2000>("64KB", n_blocks, n_runs, clock_ghz, d_cy, d_sink);
    bench_pair<16, 8 , 2000>("64KB", n_blocks, n_runs, clock_ghz, d_cy, d_sink);

    // ===== V21-style burst test: small N_ITER on 1 cluster =====
    // Reproduce V21's exact "560 GB/s" geometry: 4 warps × ILP=4 × CL=5 outer.
    // Then run the SAME geometry FENCED to expose true completion BW.
    printf("\n=== V21-style burst (1 cluster, 4 warps × ILP=4, 5 outer iters) ===\n");
    bench_pair<4, 1, 5>("burst5", CLUSTER, n_runs * 50, clock_ghz, d_cy, d_sink);
    bench_pair<4, 1, 50>("brst50", CLUSTER, n_runs * 20, clock_ghz, d_cy, d_sink);
    bench_pair<4, 1, 500>("brst5h", CLUSTER, n_runs * 5, clock_ghz, d_cy, d_sink);
    bench_pair<4, 1, 5000>("brst5k", CLUSTER, n_runs, clock_ghz, d_cy, d_sink);

    // Sink to keep optimizer honest
    unsigned h_sink = 0;
    cudaMemcpy(&h_sink, d_sink, 4, cudaMemcpyDeviceToHost);
    printf("\n[sink=%u]\n", h_sink);

    cudaFree(d_cy);
    cudaFree(d_sink);
    return 0;
}
