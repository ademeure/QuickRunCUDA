#include <cstdio>
__global__ void test_pdl(float *out) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (threadIdx.x == 0) out[blockIdx.x] = 1.0f;
}
int main() { printf("compiled\n"); return 0; }
