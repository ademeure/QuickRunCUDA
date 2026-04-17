// Minimal kernel to measure launch latency (QuickRunCUDA times the kernel+event cost).
// For cleanest measurement, kernel does nothing but single write.

extern "C" __global__ void kernel(float* A, float* B, float* C, int u0, int u1, int u2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        ((unsigned long long*)C)[0] = t;
    }
}
