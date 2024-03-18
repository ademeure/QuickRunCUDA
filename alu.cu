const size_t outerloop = 5000;
const size_t innerloop = 10;

extern "C" __global__ void kernel(const float *input, float *output, int sm0, int sm1, int unused) {
  float float0 = (float)input[threadIdx.x];
  float float1 = (float)input[(threadIdx.x) + 96];
  float float2 = (float)input[(threadIdx.x) + 160];
  float float3 = (float)input[(threadIdx.x) + 224];
  //float float4 = (float)input[(threadIdx.x) + 288];
  float float5 = (float)input[(threadIdx.x) + 352];
  float float6 = (float)input[(threadIdx.x) + 416];
  //float float7 = (float)input[(threadIdx.x) + 70];

  #pragma unroll 1
  for (int i = 0; i < outerloop; i++) {
    #pragma unroll innerloop
    for (int j = 0; j < innerloop; j++) {
      float0 = float0*float5+float6;
      float1 = float1*float5+float6;
      float2 = float2*float5+float6;
      float3 = float3*float5+float6;
    }
  }
  int out = (int)(float0+float1+float2+float3);
  if (out == 3756438) {
    output[threadIdx.x] = out;
  }
}

/*
extern "C" __global__ void kernel(const float *A, float *B, int arg0, int arg1, int arg2) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < arg0) {
    B[i] = A[i] + A[i+1];
  }
}
*/