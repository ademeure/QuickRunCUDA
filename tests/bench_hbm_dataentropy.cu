// Test if HBM bandwidth depends on data entropy
extern "C" __global__ __launch_bounds__(256, 4)
void kernel(int* A, int* B, int* C, int iters, int mode, int verify) {
    int tid = blockIdx.x * 256 + threadIdx.x;
    int total = gridDim.x * 256;
    
    int sum = 0;
    for (int i = 0; i < iters; i++) {
        int idx = (tid + i * total) & 0x3FFFFFF;  // mask to 64M (256 MB int)
        int v = A[idx];
        sum += v;
    }
    if (verify || tid == 0) C[tid % 1024] = sum;
}
