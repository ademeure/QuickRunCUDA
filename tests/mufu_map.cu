extern "C" __global__ void kernel(float* A, float* B, float* C, int a0, int a1, int a2) {
    float r = (float)threadIdx.x + 1.0f;
    float out1, out2, out3, out4, out5, out6, out7;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(out1) : "f"(r));
    asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(out2) : "f"(r));
    asm volatile("sin.approx.f32 %0, %1;" : "=f"(out3) : "f"(r));
    asm volatile("cos.approx.f32 %0, %1;" : "=f"(out4) : "f"(r));
    asm volatile("lg2.approx.f32 %0, %1;" : "=f"(out5) : "f"(r));
    asm volatile("rcp.approx.f32 %0, %1;" : "=f"(out6) : "f"(r));
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(out7) : "f"(r));
    if (a0 == 12345) ((float*)C)[threadIdx.x] = out1 + out2 + out3 + out4 + out5 + out6 + out7;
}
