extern "C" __global__  void kernel(float *in, float* out, int* integer_data, int dynamic_int, float dynamic_float, int unused_2) {
    printf("Hello, world! I'm thread %d of block %d.\n", threadIdx.x, blockIdx.x);
}
