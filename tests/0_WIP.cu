extern "C" __global__  void kernel(float *in, float* out, int* integer_data, int dynamic_int, float dynamic_float, int unused_2) {
    // integer add
    int input = integer_data[0];
    int input2 = integer_data[1];
    int output;
    asm volatile("add.s32 %0, %1, %2;" : "=r"(output) : "r"(input), "r"(dynamic_int));
    asm volatile("add.s32 %0, %1, %2;" : "=r"(output) : "r"(output), "r"(input2));
    integer_data[0] = output;

    // floating point FMA
    /*
    float output = in[0] * 2.0f + 1.0f;
    out[0] = output;
    */

    // reduction for absmax
    /*
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float input = in[id];
    unsigned int absmax_uint_warp = *(unsigned int*)&input;
    asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint_warp));
    float output = *(float*)&absmax_uint_warp;
    out[id] = 0.0f;
    */
}

// Simple RELU
/*
extern "C" __global__  void relu(float *in, float* out, float* unused_C, int unused_0, int unused_1, int unused_2) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float input = in[id];
    float output = max(input, 0.0f); // RELU
    out[id] = output;
}
*/

/*
// warp absmax via reduction

extern "C" __global__  void kernel(float *in, float* out, float* unused_C, int unused_0, int unused_1, int unused_2) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float input = in[id];
    float output = max(input, 0.0f); // RELU
    out[id] = output;
}
*/
