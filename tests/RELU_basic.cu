extern "C" __global__  void kernel(float *in, float* out) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float input = in[id];
    float output = max(input, 0.0f); // RELU
    out[id] = output;
}
