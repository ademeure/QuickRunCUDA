// ILP sweep for ALL ops at 1/2/4 SMSPs
// Pure ops only (no extra IADD/ALU contending with op being measured)
// ILP = independent register chains (each warp runs ILP chains in parallel)
#include <cuda_runtime.h>
#include <cstdio>

// SHFL constant-offset, no extra ALU
#define KSHFL_ILP(NAME, ILP) \
template <int SMSP_MASK> __launch_bounds__(1024, 1) __global__ void NAME(int *out, int N) { \
    int warp_id = threadIdx.x >> 5; \
    int smsp = warp_id & 3; \
    if (((SMSP_MASK >> smsp) & 1) == 0) return; \
    int v[ILP]; \
    _Pragma("unroll") for (int i = 0; i < ILP; i++) v[i] = (threadIdx.x ^ i) + 0xCAFE; \
    for (int i = 0; i < N; i++) { \
        _Pragma("unroll") for (int j = 0; j < ILP; j++) v[j] = __shfl_xor_sync(0xFFFFFFFF, v[j], 1); \
    } \
    int sum = 0; _Pragma("unroll") for (int i = 0; i < ILP; i++) sum ^= v[i]; \
    if (sum == 0xDEADBEEF && N < 0) out[blockIdx.x] = sum; \
}

// REDUX.SUM (MIO)
#define KREDUX_ILP(NAME, ILP) \
template <int SMSP_MASK> __launch_bounds__(1024, 1) __global__ void NAME(int *out, int N) { \
    int warp_id = threadIdx.x >> 5; \
    int smsp = warp_id & 3; \
    if (((SMSP_MASK >> smsp) & 1) == 0) return; \
    unsigned v[ILP]; \
    _Pragma("unroll") for (int i = 0; i < ILP; i++) v[i] = (threadIdx.x ^ i) + 0xCAFE; \
    for (int i = 0; i < N; i++) { \
        _Pragma("unroll") for (int j = 0; j < ILP; j++) { \
            unsigned r; asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v[j])); \
            v[j] = r; \
        } \
    } \
    unsigned sum = 0; _Pragma("unroll") for (int i = 0; i < ILP; i++) sum ^= v[i]; \
    if (sum == 0xDEADBEEF && N < 0) out[blockIdx.x] = sum; \
}

// CREDUX.MIN (different pipe)
#define KCREDUX_ILP(NAME, ILP) \
template <int SMSP_MASK> __launch_bounds__(1024, 1) __global__ void NAME(int *out, int N) { \
    int warp_id = threadIdx.x >> 5; \
    int smsp = warp_id & 3; \
    if (((SMSP_MASK >> smsp) & 1) == 0) return; \
    unsigned v[ILP]; \
    _Pragma("unroll") for (int i = 0; i < ILP; i++) v[i] = (threadIdx.x ^ i) + 0xCAFE; \
    for (int i = 0; i < N; i++) { \
        _Pragma("unroll") for (int j = 0; j < ILP; j++) { \
            unsigned r; asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v[j])); \
            v[j] = r ^ (j+1); \
        } \
    } \
    unsigned sum = 0; _Pragma("unroll") for (int i = 0; i < ILP; i++) sum ^= v[i]; \
    if (sum == 0xDEADBEEF && N < 0) out[blockIdx.x] = sum; \
}

// STS to ILP independent slots (volatile to force store)
#define KSTS_ILP(NAME, ILP) \
template <int SMSP_MASK> __launch_bounds__(1024, 1) __global__ void NAME(int *out, int N) { \
    __shared__ int smem[1024 * ILP]; \
    volatile int *vsmem = smem; \
    int warp_id = threadIdx.x >> 5; \
    int smsp = warp_id & 3; \
    if (((SMSP_MASK >> smsp) & 1) == 0) return; \
    int slot = warp_id * 32 + (threadIdx.x & 31); \
    int v = threadIdx.x ^ 0xCAFE; \
    for (int i = 0; i < N; i++) { \
        _Pragma("unroll") for (int j = 0; j < ILP; j++) { \
            vsmem[slot + j * 1024] = v + j; \
        } \
    } \
}

// LDS from ILP independent slots
#define KLDS_ILP(NAME, ILP) \
template <int SMSP_MASK> __launch_bounds__(1024, 1) __global__ void NAME(int *out, int N) { \
    __shared__ int smem[1024 * ILP]; \
    volatile int *vsmem = smem; \
    for (int s = threadIdx.x; s < 1024 * ILP; s += blockDim.x) smem[s] = s; \
    __syncthreads(); \
    int warp_id = threadIdx.x >> 5; \
    int smsp = warp_id & 3; \
    if (((SMSP_MASK >> smsp) & 1) == 0) return; \
    int slot = warp_id * 32 + (threadIdx.x & 31); \
    int acc[ILP]; \
    _Pragma("unroll") for (int j = 0; j < ILP; j++) acc[j] = 0; \
    for (int i = 0; i < N; i++) { \
        _Pragma("unroll") for (int j = 0; j < ILP; j++) acc[j] ^= vsmem[slot + j * 1024]; \
    } \
    int sum = 0; _Pragma("unroll") for (int i = 0; i < ILP; i++) sum ^= acc[i]; \
    if (sum == 0xDEADBEEF && N < 0) out[blockIdx.x] = sum; \
}

// ATOMS (smem atomicAdd) — use INC pattern (popc-merged)
#define KATOM_ILP(NAME, ILP) \
template <int SMSP_MASK> __launch_bounds__(1024, 1) __global__ void NAME(int *out, int N) { \
    __shared__ int smem[1024 * ILP]; \
    if (threadIdx.x < 1024 * ILP) smem[threadIdx.x] = 0; \
    __syncthreads(); \
    int warp_id = threadIdx.x >> 5; \
    int smsp = warp_id & 3; \
    if (((SMSP_MASK >> smsp) & 1) == 0) return; \
    int slot = warp_id * 32 + (threadIdx.x & 31); \
    for (int i = 0; i < N; i++) { \
        _Pragma("unroll") for (int j = 0; j < ILP; j++) { \
            atomicAdd(&smem[slot + j * 1024], 1); \
        } \
    } \
}

KSHFL_ILP(shfl_1, 1) KSHFL_ILP(shfl_2, 2) KSHFL_ILP(shfl_4, 4) KSHFL_ILP(shfl_8, 8)
KREDUX_ILP(redux_1, 1) KREDUX_ILP(redux_2, 2) KREDUX_ILP(redux_4, 4) KREDUX_ILP(redux_8, 8)
KCREDUX_ILP(credux_1, 1) KCREDUX_ILP(credux_2, 2) KCREDUX_ILP(credux_4, 4) KCREDUX_ILP(credux_8, 8)
KSTS_ILP(sts_1, 1) KSTS_ILP(sts_2, 2) KSTS_ILP(sts_4, 4) KSTS_ILP(sts_8, 8)
KLDS_ILP(lds_1, 1) KLDS_ILP(lds_2, 2) KLDS_ILP(lds_4, 4) KLDS_ILP(lds_8, 8)
KATOM_ILP(atom_1, 1) KATOM_ILP(atom_2, 2) KATOM_ILP(atom_4, 4) KATOM_ILP(atom_8, 8)

template <typename Fn>
double bench(Fn kfn, int *d_out, int N, int active_smsp, int ilp) {
    int blocks = 148, threads = 1024;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_out, N);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { return -1; }
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        kfn<<<blocks, threads>>>(d_out, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long active_threads = 8L * 32 * active_smsp * 148;
    long thr_ops = active_threads * (long)N * ilp;
    double clk_hz = 2032e6;
    return (double)thr_ops / (best/1000.0) / clk_hz / 148.0;  // thr-op/SM/cy
}

typedef void(*kfn_t)(int*, int);
void row(const char* op_name, kfn_t k1, kfn_t k2, kfn_t k4, kfn_t k8, int *d_out, int N, int smsp) {
    double r1 = bench(k1, d_out, N, smsp, 1);
    double r2 = bench(k2, d_out, N, smsp, 2);
    double r4 = bench(k4, d_out, N, smsp, 4);
    double r8 = bench(k8, d_out, N, smsp, 8);
    printf("  %-10s  ILP=1: %5.1f   ILP=2: %5.1f   ILP=4: %5.1f   ILP=8: %5.1f  thr-op/SM/cy\n",
           op_name, r1, r2, r4, r8);
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    int N = 5000;

    for (int n_smsp : {1, 2, 4}) {
        int mask;
        if (n_smsp == 1) mask = 0b0001;
        else if (n_smsp == 2) mask = 0b0011;
        else mask = 0b1111;
        printf("\n# === %d SMSP%s active (mask 0b%d%d%d%d) ===\n",
               n_smsp, n_smsp==1?"":"s", (mask>>3)&1, (mask>>2)&1, (mask>>1)&1, mask&1);
        if (n_smsp == 1) {
            row("SHFL.bfly", shfl_1<0b0001>, shfl_2<0b0001>, shfl_4<0b0001>, shfl_8<0b0001>, d_out, N, 1);
            row("REDUX.SUM", redux_1<0b0001>, redux_2<0b0001>, redux_4<0b0001>, redux_8<0b0001>, d_out, N, 1);
            row("CREDUX.MIN", credux_1<0b0001>, credux_2<0b0001>, credux_4<0b0001>, credux_8<0b0001>, d_out, N, 1);
            row("STS",       sts_1<0b0001>,   sts_2<0b0001>,   sts_4<0b0001>,   sts_8<0b0001>,   d_out, N, 1);
            row("LDS",       lds_1<0b0001>,   lds_2<0b0001>,   lds_4<0b0001>,   lds_8<0b0001>,   d_out, N, 1);
            row("ATOM.INC",  atom_1<0b0001>,  atom_2<0b0001>,  atom_4<0b0001>,  atom_8<0b0001>,  d_out, N, 1);
        } else if (n_smsp == 2) {
            row("SHFL.bfly", shfl_1<0b0011>, shfl_2<0b0011>, shfl_4<0b0011>, shfl_8<0b0011>, d_out, N, 2);
            row("REDUX.SUM", redux_1<0b0011>, redux_2<0b0011>, redux_4<0b0011>, redux_8<0b0011>, d_out, N, 2);
            row("CREDUX.MIN", credux_1<0b0011>, credux_2<0b0011>, credux_4<0b0011>, credux_8<0b0011>, d_out, N, 2);
            row("STS",       sts_1<0b0011>,   sts_2<0b0011>,   sts_4<0b0011>,   sts_8<0b0011>,   d_out, N, 2);
            row("LDS",       lds_1<0b0011>,   lds_2<0b0011>,   lds_4<0b0011>,   lds_8<0b0011>,   d_out, N, 2);
            row("ATOM.INC",  atom_1<0b0011>,  atom_2<0b0011>,  atom_4<0b0011>,  atom_8<0b0011>,  d_out, N, 2);
        } else {
            row("SHFL.bfly", shfl_1<0b1111>, shfl_2<0b1111>, shfl_4<0b1111>, shfl_8<0b1111>, d_out, N, 4);
            row("REDUX.SUM", redux_1<0b1111>, redux_2<0b1111>, redux_4<0b1111>, redux_8<0b1111>, d_out, N, 4);
            row("CREDUX.MIN", credux_1<0b1111>, credux_2<0b1111>, credux_4<0b1111>, credux_8<0b1111>, d_out, N, 4);
            row("STS",       sts_1<0b1111>,   sts_2<0b1111>,   sts_4<0b1111>,   sts_8<0b1111>,   d_out, N, 4);
            row("LDS",       lds_1<0b1111>,   lds_2<0b1111>,   lds_4<0b1111>,   lds_8<0b1111>,   d_out, N, 4);
            row("ATOM.INC",  atom_1<0b1111>,  atom_2<0b1111>,  atom_4<0b1111>,  atom_8<0b1111>,  d_out, N, 4);
        }
    }
    return 0;
}
