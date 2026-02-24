extern "C" __global__  void kernel(float *in, float* out) {
    printf("Hello, world! I'm thread %d of block %d.\n", threadIdx.x, blockIdx.x);
}
