#include <cuda_fp16.h>

extern "C" __global__  void __cluster_dims__(8, 1, 1) kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N, int unused_1, int unused_2) {
    typedef half scalar_t;
    bool bounds_checking = false;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr size_t V = 16 / sizeof(scalar_t);
    
    if (!bounds_checking || idx*V + V-1 < N) {
        // this is the actual data in GPU registers (128-bit)
        int4 Ain = ((int4*)A)[idx];
        int4 Bin = ((int4*)B)[idx];
        int4 Cout;
        
        // scalar array (8 elements for 16-bit half)
        scalar_t* Ax = (scalar_t*)&Ain;
        scalar_t* Bx = (scalar_t*)&Bin;
        scalar_t* Cx = (scalar_t*)&Cout;

        // automatically unrolled by nvcc to avoid having to use local memory for array indexing
        // we let the compiler do its magic and end up with simple register reads & writes
        for (int k = 0; k < V; k++) {
            Cx[k] = Ax[k] + Bx[k];
        }
        
        // write back to global memory with a 128-bit store
        int4* C4 = (int4*)C;
        C4[idx] = Cout;
    } else {
        // out of bounds processing for test cases that aren't a multiple of V (8 for FP16)
        for (int k = 0; k < V; k++) {
            if (idx*V+k < N) {
                C[idx*V+k] = A[idx*V+k] + B[idx*V+k];
            }
        }
    }
}
