
extern "C" __global__ void kernel(unsigned int *A, unsigned int *B, int sm, int unused1, int unused2) {
  unsigned int smid;
  asm("mov.u32 %0, %smid;" : "=r"(smid));

  if (threadIdx.x >= 16 || smid != 0) {
    __nanosleep(10000);
    return;
  }

  constexpr int latency_offset = 100000;
  constexpr int num_measurements = 500;
  constexpr int num_baseline_measurements = 500;
  constexpr int last_idx = 131072;
  constexpr int last_idx_outer = 4096;

  float* latency_ptr = (float*)&B[0];
  float* l2_latency_ptr = (float*)&B[last_idx];
  float* dual_latency_ptr = (float*)&B[last_idx*2];
  unsigned int* group_ptr = &A[latency_offset];

  int4 data;
  data.x = 0; data.y = 0; data.z = 0; data.w = 0;

  int4 *A_base_ptr = (int4 *)&A[(threadIdx.x % 8) * 4];
  int4 *A_ptr;
  int4 tmp;

  for (int start_idx = 0; start_idx < last_idx - 2; start_idx++) {
    if ((start_idx & 2) != 0) {
      continue;
    }

    if (threadIdx.x < 8) {
      A_ptr = &A_base_ptr[start_idx * 16];
    } else {
      A_ptr = &A_base_ptr[(start_idx + 2) * 16 + 8];
    }






    asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l"(&A_ptr[data.x]));
    data.x += tmp.x; data.y += tmp.y; data.z += tmp.z; data.w += tmp.w;
    asm volatile ("discard.global.L2 [%0], 128;" : : "l"(A_ptr));
    asm volatile ("membar.cta;");
    __nanosleep(100);

    unsigned long long total_duration = 0;
    #pragma unroll 1
    for (int k = 0; k < num_baseline_measurements; k++) {
      unsigned long long start = clock64();
      asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l"(&A_ptr[data.x]));
      data.x += tmp.x; data.y += tmp.y; data.z += tmp.z; data.w += tmp.w;
      unsigned long long end = clock64();
      total_duration += end - start;
      asm volatile ("discard.global.L2 [%0], 128;" : : "l"(A_ptr));
      __nanosleep(50);
    }

    if (threadIdx.x == 0) {
      float baseline_time = (float)total_duration / (float)num_baseline_measurements;
      latency_ptr[start_idx] = baseline_time;
      latency_ptr[start_idx + 2] = baseline_time;
      if (start_idx % 100 == 0) {
        //printf("%d: %.3f\n", start_idx, baseline_time);
      }
    }





    asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l"(&A_ptr[data.x]));
    data.x += tmp.x; data.y += tmp.y; data.z += tmp.z; data.w += tmp.w;
    asm volatile ("membar.cta;");
    __nanosleep(100);

    total_duration = 0;
    #pragma unroll 1
    for (int k = 0; k < num_baseline_measurements; k++) {
      unsigned long long start = clock64();
      asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l"(&A_ptr[data.x]));
      data.x += tmp.x; data.y += tmp.y; data.z += tmp.z; data.w += tmp.w;
      unsigned long long end = clock64();
      total_duration += end - start;
      __nanosleep(100);
    }

    if (threadIdx.x == 0) {
      float baseline_time = (float)total_duration / (float)num_baseline_measurements;
      l2_latency_ptr[start_idx] = baseline_time;
      l2_latency_ptr[start_idx + 2] = baseline_time;
      if (start_idx % 100 == 0) {
        //printf("%d: %.3f\n", start_idx, baseline_time);
      }
    }
  }

  int current_group = 0;
  for (int start_idx = 0; start_idx < last_idx_outer; start_idx++) {
    if ((start_idx & 2) != 0 || group_ptr[start_idx] != 0) {
      for (int i = start_idx+4; i < last_idx; i++) {
        //dual_latency_ptr[start_idx * last_idx + i] = -1.0f;
      }
      continue;
    }
    if (threadIdx.x == 0) {
      //printf("============ %d: %.3f ============\n", start_idx, latency_ptr[start_idx]);
    }
    current_group++;
    group_ptr[start_idx] = current_group;

    if (threadIdx.x < 8) {
      A_ptr = &A_base_ptr[start_idx * 16];
    }
    float baseline_time = latency_ptr[start_idx];

    for (int i = start_idx+4; i < last_idx; i++) {
      if ((i & 2) != 0 || group_ptr[i] != 0) {
        //dual_latency_ptr[start_idx * last_idx + i] = -2.0f;
        continue;
      }
      float baseline_time_i = latency_ptr[i];

      if (fabsf(baseline_time - baseline_time_i) > 4.0f) {
        //dual_latency_ptr[start_idx * last_idx + i] = -1.0f;
        continue;
      }

      if (threadIdx.x >= 8) {
        A_ptr = &A_base_ptr[i * 16];
      }

      unsigned long long total_duration = 0;
      unsigned long long start_warmup = clock64();
      asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l"(&A_ptr[data.x]));
      data.x += tmp.x; data.y += tmp.y; data.z += tmp.z; data.w += tmp.w;
      unsigned long long end_warmup = clock64();
      asm volatile ("discard.global.L2 [%0], 128;" : : "l"(A_ptr));
      asm volatile ("membar.cta;");
      if (end_warmup - start_warmup < baseline_time + 4.0f || end_warmup - start_warmup < baseline_time_i + 4.0f) {
        continue;
      }
      __nanosleep(1000);

      #pragma unroll 1
      for (int k = 0; k < num_measurements; k++) {
        unsigned long long start = clock64();
        asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l"(&A_ptr[data.x]));
        data.x += tmp.x; data.y += tmp.y; data.z += tmp.z; data.w += tmp.w;
        unsigned long long end = clock64();
        total_duration += end - start;
        asm volatile ("discard.global.L2 [%0], 128;" : : "l"(A_ptr));
        __nanosleep(50);
      }
      if (threadIdx.x == 0) {
        float avg_time = (float)total_duration / (float)num_measurements;

        dual_latency_ptr[start_idx * last_idx + i] = avg_time;

        if (avg_time > baseline_time + 24.0f || avg_time > baseline_time_i + 24.0f) {
          //printf("%d/%d ==> %.3f (vs %.3f/%.3f) ==> %.3f (%d)\n", start_idx, i, avg_time, baseline_time, baseline_time_i, (avg_time - fmaxf(baseline_time, baseline_time_i)));
          //group_ptr[i] = current_group;
        }
      }
    }
  }

  if (data.x + data.y + data.z + data.w == 77) {
    B[threadIdx.x] = 3;
  }
}
