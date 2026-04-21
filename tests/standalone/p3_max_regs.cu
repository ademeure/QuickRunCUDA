// P3: max registers per thread = 255 hard cap
// At 256+, ptxas silently caps to 255 with warning
__global__ __launch_bounds__(64, 1)
void k(float* out, int u2) {
    float a[64];
    for (int i = 0; i < 64; i++) a[i] = (float)threadIdx.x + (float)i + (float)u2;
    float b = (float)u2;
    for (int j = 0; j < 100; j++)
        for (int i = 0; i < 64; i++) a[i] = a[i] * b + (float)j;
    float s = 0; for (int i = 0; i < 64; i++) s += a[i];
    if ((int)s == 12345) out[blockIdx.x * 64 + threadIdx.x] = s;
}
int main() { return 0; }
// Build: nvcc -maxrregcount=N
// N <= 255: compiles silently
// N >= 256: "ptxas warning : Too big maxrregcount value specified N, will be ignored"
