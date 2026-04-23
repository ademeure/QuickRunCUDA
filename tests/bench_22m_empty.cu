// Truly empty kernel for §22m launch overhead audit (no SASS body work).
// QuickRunCUDA is going to invoke this many times via -T to measure
// per-launch overhead.
extern "C" __global__ void kernel(float* A, float* B, float* C, int u0, int u1, int u2) {
    // Empty body. Compiler will emit only the prologue + EXIT.
}
