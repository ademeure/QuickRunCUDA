// V15: Every SM-pair latency within 8-wide cluster (28 ordered pairs tested with src,dst)
//
// Methodology per 04_dsmem:
//   - cluster_dims(8), all 8 CTAs launched for SMEM lifetime
//   - Only CTA = SRC does the loads (reads from peer CTA = DST)
//   - All other CTAs spin on a trailing cluster barrier so SMEM stays alive
//   - Dep pointer chain (CL=5) for DCE-immunity, crash-safe
//   - 1 thread per CTA to eliminate port contention
//
// Output: 8×8 matrix (SRC=row, DST=col) of cy/load.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256
#define CX 8
#define CL 5

// Single-thread dep chain. SRC reads peer DST; all other CTAs spin idle.
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void pair_lat(unsigned long long* out, unsigned src, unsigned dst, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (SMEM_W - 1)) * 4u;
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(dst));

    unsigned long long t0 = 0, t1 = 0;
    unsigned cur = 0;

    if (my_cta == src && tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(cur) : "r"(peer_base + cur) : "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }

    // CRITICAL: all CTAs must survive until SRC finishes.
    // Trailing cluster barrier guarantees this.
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == src && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = cur;
    }
}

static int run_pair(unsigned src, unsigned dst, unsigned long long* d_out, double* avg_cy, int N) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CX, 1, 1);
    cfg.blockDim = dim3(32, 1, 1);
    cfg.stream = 0;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CX;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    double sum = 0;
    int got = 0, crashes = 0;
    for (int r = 0; r < N * 3 && got < N; r++) {
        cudaError_t e = cudaLaunchKernelEx(&cfg, pair_lat, d_out, src, dst, 42u + r);
        if (e) { crashes++; cudaGetLastError(); continue; }
        e = cudaDeviceSynchronize();
        if (e) { crashes++; cudaGetLastError(); continue; }
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        sum += (double)cy;
        got++;
    }
    if (got == 0) { *avg_cy = 0; return -1; }
    *avg_cy = sum / got / CL;
    return got;
}

int main() {
    CK(cudaSetDevice(0));
    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 16));

    printf("=== V15 DSMEM all SM-pairs (cluster=8, 1 thread, CL=5, 1920 MHz) ===\n\n");

    double mat[CX][CX] = {{0}};
    int ok[CX][CX] = {{0}};

    for (int s = 0; s < CX; s++) {
        for (int d = 0; d < CX; d++) {
            if (s == d) { mat[s][d] = 0; ok[s][d] = 1; continue; }
            double avg;
            int got = run_pair(s, d, d_out, &avg, 15);
            if (got > 0) { mat[s][d] = avg; ok[s][d] = 1; }
            else { fprintf(stderr, "  pair src=%d dst=%d: all crashed\n", s, d); }
        }
    }

    printf("     ");
    for (int d = 0; d < CX; d++) printf("  DST=%d ", d);
    printf("\n");
    for (int s = 0; s < CX; s++) {
        printf("SRC=%d ", s);
        for (int d = 0; d < CX; d++) {
            if (s == d) printf("   --   ");
            else if (ok[s][d]) printf("%6.2f  ", mat[s][d]);
            else printf("CRASH   ");
        }
        printf("\n");
    }

    // Check symmetry: is (s->d) latency = (d->s) latency?
    printf("\n--- Asymmetry (s->d - d->s), cy/load ---\n");
    printf("     ");
    for (int d = 0; d < CX; d++) printf("  DST=%d ", d);
    printf("\n");
    for (int s = 0; s < CX; s++) {
        printf("SRC=%d ", s);
        for (int d = 0; d < CX; d++) {
            if (s == d) printf("   --   ");
            else if (ok[s][d] && ok[d][s]) printf("%+6.2f  ", mat[s][d] - mat[d][s]);
            else printf("   ?    ");
        }
        printf("\n");
    }

    // Range per source
    printf("\n--- Per-source latency range (min/max/spread) ---\n");
    for (int s = 0; s < CX; s++) {
        double mn = 1e30, mx = 0, sum = 0; int n = 0;
        for (int d = 0; d < CX; d++) {
            if (s != d && ok[s][d]) {
                if (mat[s][d] < mn) mn = mat[s][d];
                if (mat[s][d] > mx) mx = mat[s][d];
                sum += mat[s][d]; n++;
            }
        }
        if (n > 0) printf("  SRC=%d: min %.2f, max %.2f, avg %.2f, spread %.2f\n",
                          s, mn, mx, sum/n, mx-mn);
    }

    cudaFree(d_out);
    return 0;
}
