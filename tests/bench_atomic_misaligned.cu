// atomicAdd misaligned: what happens with non-natural-alignment atomic?
// Mode 0: aligned (baseline) - atomicAdd on dword-aligned int*
// Mode 1: misaligned by 2 bytes (half-word offset)
// Mode 2: misaligned by 1 byte
// Mode 3: aligned int64 atomic on dword
// Mode 4: aligned half (16-bit) atomic

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    char* base = (char*)A;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int per_thread_offset = (tid * 32) & 0x3FFC;  // dword-aligned

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Aligned: dword pointer
        int* p = (int*)(base + per_thread_offset);
        atomicAdd(p, 1);
#elif MODE == 1
        // Misaligned by 2 bytes
        int* p = (int*)(base + per_thread_offset + 2);
        atomicAdd(p, 1);
#elif MODE == 2
        // Misaligned by 1 byte
        int* p = (int*)(base + per_thread_offset + 1);
        atomicAdd(p, 1);
#elif MODE == 3
        // 64-bit atomic on aligned addr
        unsigned long long* p = (unsigned long long*)(base + per_thread_offset);
        atomicAdd(p, 1ull);
#elif MODE == 4
        // 16-bit atomic via PTX
        unsigned short* p = (unsigned short*)(base + per_thread_offset + 2);
        atomicAdd(p, (unsigned short)1);
#endif
    }
}
