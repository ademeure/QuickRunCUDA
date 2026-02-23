extern "C" __global__  void kernel(float *in, float* out, float* unused_C, int unused_0, int unused_1, int unused_2) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float input = in[id];
    float output = max(input, 0.0f); // RELU
    out[id] = output;
}