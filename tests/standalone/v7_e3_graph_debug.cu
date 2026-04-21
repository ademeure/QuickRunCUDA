// V7 E3: cudaGraphDebugDotPrint for visualizing graph structure
#include <cuda_runtime.h>
#include <cstdio>

__global__ void k1(unsigned int* buf) { if (threadIdx.x == 0) buf[0]++; }
__global__ void k2(unsigned int* buf) { if (threadIdx.x == 0) buf[1]++; }
__global__ void k3(unsigned int* buf) { if (threadIdx.x == 0) buf[2]++; }

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 16);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaGraphNode_t n1, n2, n3;
    cudaKernelNodeParams kp1 = {}; kp1.func = (void*)k1; kp1.gridDim = dim3(1); kp1.blockDim = dim3(32);
    void* args1[1] = {&buf}; kp1.kernelParams = args1;
    cudaGraphAddKernelNode(&n1, graph, nullptr, 0, &kp1);

    cudaKernelNodeParams kp2 = {}; kp2.func = (void*)k2; kp2.gridDim = dim3(1); kp2.blockDim = dim3(32);
    void* args2[1] = {&buf}; kp2.kernelParams = args2;
    cudaGraphAddKernelNode(&n2, graph, &n1, 1, &kp2);

    cudaKernelNodeParams kp3 = {}; kp3.func = (void*)k3; kp3.gridDim = dim3(1); kp3.blockDim = dim3(32);
    void* args3[1] = {&buf}; kp3.kernelParams = args3;
    cudaGraphAddKernelNode(&n3, graph, &n1, 1, &kp3);

    // n1 → n2; n1 → n3 (DAG with diamond pattern would need n2,n3 → n4)

    // DotPrint
    cudaError_t err = cudaGraphDebugDotPrint(graph, "/tmp/v7_e3.dot", 0);
    if (err != cudaSuccess) {
        printf("DotPrint failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("DotPrint OK → /tmp/v7_e3.dot\n");
    return 0;
}
