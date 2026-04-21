// V16: DSMEM + SM-ID topology correlation
//
// A) Capture SM_ID of each CTA across many launches (is placement stable?)
// B) Pair-sweep with SM_ID tags so we can see which physical SMs are "close"
// C) Bonus: cluster_ctaid and GPC id if exposable

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256
#define CX 8
#define CL 5

__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void capture_smids(unsigned int* out) {
    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    if (threadIdx.x != 0) return;

    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    out[my_cta] = smid;
}

__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void pair_lat_smid(unsigned long long* out, unsigned int* smids_out,
                   unsigned src, unsigned dst, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (SMEM_W - 1)) * 4u;
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    if (tid == 0) {
        unsigned smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        smids_out[my_cta] = smid;
    }
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

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == src && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = cur;
    }
}

int main() {
    CK(cudaSetDevice(0));
    unsigned long long* d_out;
    unsigned int* d_smids;
    CK(cudaMalloc(&d_out, 16));
    CK(cudaMalloc(&d_smids, 8 * 4));

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

    printf("=== V16 DSMEM + SM-ID topology (CX=8) ===\n\n");

    // A) Is CTA→SM placement stable?
    printf("A) CTA→SM_ID across 5 launches:\n");
    for (int r = 0; r < 5; r++) {
        CK(cudaLaunchKernelEx(&cfg, capture_smids, d_smids));
        CK(cudaDeviceSynchronize());
        unsigned int smids[8];
        cudaMemcpy(smids, d_smids, 8*4, cudaMemcpyDeviceToHost);
        printf("  Run %d: ", r);
        for (int i = 0; i < 8; i++) printf("CTA%d→SM%-3u  ", i, smids[i]);
        printf("\n");
    }

    // B) Pair-sweep with SM-ID capture on same run
    printf("\nB) Pair-sweep with SM-ID tags:\n");
    double mat[CX][CX] = {{0}};
    int mat_ok[CX][CX] = {{0}};
    unsigned int smid_sum[CX] = {0};
    int smid_n[CX] = {0};

    for (int s = 0; s < CX; s++) {
        for (int d = 0; d < CX; d++) {
            if (s == d) continue;
            double sum = 0; int got = 0;
            unsigned int smids[8] = {0};
            for (int r = 0; r < 20; r++) {
                cudaError_t e = cudaLaunchKernelEx(&cfg, pair_lat_smid, d_out, d_smids, s, d, 42u + r);
                if (e) { cudaGetLastError(); continue; }
                e = cudaDeviceSynchronize();
                if (e) { cudaGetLastError(); continue; }
                unsigned long long cy;
                cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
                cudaMemcpy(smids, d_smids, 8*4, cudaMemcpyDeviceToHost);
                sum += (double)cy;
                for (int i = 0; i < 8; i++) { smid_sum[i] += smids[i]; smid_n[i]++; }
                got++;
                if (got >= 10) break;
            }
            if (got > 0) { mat[s][d] = sum / got / CL; mat_ok[s][d] = 1; }
        }
    }

    // Average SM_ID per CTA (since placement should be stable, this is just the SM)
    printf("  Avg CTA→SM_ID (over pair runs):\n    ");
    for (int i = 0; i < 8; i++) {
        if (smid_n[i] > 0) printf("CTA%d→SM%.1f  ", i, (double)smid_sum[i]/smid_n[i]);
    }
    printf("\n");

    // Print matrix
    printf("\n  Latency matrix (cy/load) with CTA→SM assignment in headers:\n       ");
    for (int d = 0; d < CX; d++) {
        double smid = smid_n[d] > 0 ? (double)smid_sum[d]/smid_n[d] : -1;
        printf(" D%d=SM%-3.0f", d, smid);
    }
    printf("\n");
    for (int s = 0; s < CX; s++) {
        double smid = smid_n[s] > 0 ? (double)smid_sum[s]/smid_n[s] : -1;
        printf("  S%d=SM%-3.0f ", s, smid);
        for (int d = 0; d < CX; d++) {
            if (s == d) printf("   --   ");
            else if (mat_ok[s][d]) printf(" %6.2f ", mat[s][d]);
            else printf(" CRASH  ");
        }
        printf("\n");
    }

    // Group by SMID difference
    printf("\n  Latency vs |SM_dst - SM_src|:\n");
    for (int s = 0; s < CX; s++) {
        for (int d = 0; d < CX; d++) {
            if (s != d && mat_ok[s][d]) {
                int src_smid = smid_n[s] > 0 ? smid_sum[s]/smid_n[s] : 0;
                int dst_smid = smid_n[d] > 0 ? smid_sum[d]/smid_n[d] : 0;
                int diff = abs((int)dst_smid - (int)src_smid);
                // don't print all — too many; just print a sample
                (void)diff;
            }
        }
    }

    // Summary by SM distance bucket
    double bucket_sum[200] = {0}; int bucket_n[200] = {0};
    for (int s = 0; s < CX; s++) {
        for (int d = 0; d < CX; d++) {
            if (s != d && mat_ok[s][d]) {
                int src_smid = smid_n[s] > 0 ? smid_sum[s]/smid_n[s] : 0;
                int dst_smid = smid_n[d] > 0 ? smid_sum[d]/smid_n[d] : 0;
                int diff = abs((int)dst_smid - (int)src_smid);
                if (diff < 200) { bucket_sum[diff] += mat[s][d]; bucket_n[diff]++; }
            }
        }
    }
    for (int k = 0; k < 200; k++) {
        if (bucket_n[k] > 0) printf("    |ΔSM|=%d: avg %.2f cy (n=%d)\n", k, bucket_sum[k]/bucket_n[k], bucket_n[k]);
    }

    cudaFree(d_out);
    cudaFree(d_smids);
    return 0;
}
