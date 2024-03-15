
const size_t outerloop = 5000;

template <size_t innerloop>
__forceinline__ __device__ float inner(float f0, float f1, float f2, float f3) {
  #if EXTRA_FMA_ALIGN > 0
  #pragma unroll EXTRA_FMA_ALIGN
  for (int a = 0; a < EXTRA_FMA_ALIGN; a++) {
    f0 = f1*f0+f0;
  }
  #endif

  #pragma unroll 1
  for (int o = 0; o < outerloop; o++) {
    #pragma unroll innerloop
    for (int i = 0; i < innerloop; i++) {
      #pragma unroll FMA_LOOP_ALWAYS // FMA_LOOP_ALWAYS instructions => 16 * FMA_LOOP_ALWAYS bytes
      for (int r = 0; r < FMA_LOOP_ALWAYS; r++) {
        f0 = f0*f0+f0;
      }
      if (f0 < -9999.0f) { // 4 instructions (FSETP+BSSY+BRA+BSYNC) => 64 bytes
        #pragma unroll FMA_LOOP_SKIP // FMA_LOOP_SKIP instructions => 16 * FMA_LOOP_SKIP bytes
        for (int r = 0; r < FMA_LOOP_SKIP; r++) {
          f0 = f1*f0+f0;
        }
      }
    }
  }
  return f0;
}

extern "C" __global__ void kernel(const float *A, float *B, int sm0, int sm1, int unused) {
  unsigned int smid;
  asm("mov.u32 %0, %smid;" : "=r"(smid));

  if (smid != sm0 && smid != sm1) {
    return;
  }
  if (threadIdx.x >= 64) {
    return;
  }

  float base = (float)(threadIdx.x + sm0);
  float f0 = base+0.0f;
  float f1 = base+1.0f;
  float f2 = base+2.0f;
  float f3 = base+3.0f;

  clock_t start, end;
  clock_t first_start = clock64();
  start = clock64();
  if (smid == sm0) {
    if ((threadIdx.x & 63) < 32) {
      f0 += 9.8f;
      f0 = inner<innerloop>(f0, f1, f2, f3);
    } else {
      f0 += 1.1f;
      f0 = inner<innerloop>(f0, f1, f2, f3);
    }
  } else {
    if ((threadIdx.x & 63) < 32) {
      f0 += 2.2f;
      f0 = inner<innerloop>(f0, f1, f2, f3);
    } else {
      f0 += 3.3f;
      f0 = inner<innerloop>(f0, f1, f2, f3);
    }
  }
  end = clock64();

  __nanosleep(1000*1000);

  int duration = (int)(end-start);  
  if ((threadIdx.x & 31) == 0 && f0 != 77.0f) {
    int instructions = FMA_LOOP_ALWAYS + FMA_LOOP_SKIP + (FMA_LOOP_SKIP > 0 ? 4 : 0);
    int footprint = innerloop * instructions * 16;
    float IPC = (float)(instructions*innerloop*outerloop)/(float)(duration);
    printf("%d, %d, %d, %d, %d, %.5f\n", innerloop, footprint, smid, threadIdx.x/32, duration, IPC);
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