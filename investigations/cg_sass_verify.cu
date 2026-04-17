#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;

extern "C" __global__ void cg_red(unsigned *out, unsigned in) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    unsigned r = cg::reduce(warp, in, cg::plus<unsigned>());
    if (threadIdx.x == 0) out[blockIdx.x] = r;
}

int main() {
    cudaSetDevice(0);
    unsigned *d; cudaMalloc(&d, sizeof(unsigned));
    cg_red<<<1, 32>>>(d, 1);
    cudaDeviceSynchronize();
    return 0;
}
