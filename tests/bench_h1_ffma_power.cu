// H1: FFMA pipe power
// Sustained FFMA loop on persistent grid; measure power via nvidia-smi sampling
// MODE 0: 1 FFMA per inner pos (low density)
// MODE 1: 4 FFMA per inner pos
// MODE 2: 8 FFMA per inner pos (high)
// MODE 3: 16 FFMA per inner pos (very high)
// MODE 4: pure NOP (idle baseline within SM)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float a = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float b = (float)(threadIdx.x ^ u2) * 0.002f + 1.0f;
    float c = 0.5f + (float)u2 * 1e-9f;
    float d = a + 0.1f, e = a + 0.2f, f = a + 0.3f;
    float g = a + 0.4f, h = a + 0.5f, ii = a + 0.6f, jj = a + 0.7f;
    float k = a + 0.8f, l = a + 0.9f, m = a + 1.0f, n = a + 1.1f;
    float o = a + 1.2f, p = a + 1.3f, q = a + 1.4f, r = a + 1.5f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            a = a*b + c;
#elif MODE == 1
            a = a*b + c; d = d*b + c; e = e*b + c; f = f*b + c;
#elif MODE == 2
            a = a*b + c; d = d*b + c; e = e*b + c; f = f*b + c;
            g = g*b + c; h = h*b + c; ii = ii*b + c; jj = jj*b + c;
#elif MODE == 3
            a = a*b + c; d = d*b + c; e = e*b + c; f = f*b + c;
            g = g*b + c; h = h*b + c; ii = ii*b + c; jj = jj*b + c;
            k = k*b + c; l = l*b + c; m = m*b + c; n = n*b + c;
            o = o*b + c; p = p*b + c; q = q*b + c; r = r*b + c;
#elif MODE == 4
            // baseline — empty volatile to keep loop alive
            asm volatile("");
#endif
        }
    }

    float sink = a+d+e+f+g+h+ii+jj+k+l+m+n+o+p+q+r;
    if ((int)sink == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = sink;
}
