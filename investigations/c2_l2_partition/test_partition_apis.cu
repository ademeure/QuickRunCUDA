// Test: does any CUDA host API change which L2 partition an address lands in?
// Strategy: pin a kernel to one SM, measure atomic round-trip at each offset.
// Allocate buffer 4 ways:
//   1. cudaMalloc (baseline)
//   2. cudaMalloc + cudaMemAdvise (whatever advice we can pass on a discrete GPU)
//   3. cudaMallocManaged + cudaMemAdvise(SetPreferredLocation, device 0)
//   4. cudaMallocAsync (mempool)
// For each, sweep first 64 KB (16 chunks of 4 KB) and report side fingerprint.
// If side pattern changes between methods → API affects placement.
// If identical → it does NOT.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { cudaError_t e = x; if (e!=cudaSuccess) { printf("ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

__global__ void probe(unsigned* p, int iters, unsigned long long* out) {
    unsigned v = 1u;
    asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v) : "l"(p), "r"(v));
    unsigned long long t0,t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        unsigned r;
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v));
        v = r + 1;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    *out = t1 - t0;
}

void sweep(const char* label, unsigned* base, int n_offsets, unsigned long long* d_out, unsigned long long* h_out) {
    printf("--- %s ---\n", label);
    printf("offset_KB,cy_per_atomic,side\n");
    int iters = 500;
    std::vector<double> cyc;
    for (int i = 0; i < n_offsets; i++) {
        size_t off_bytes = (size_t)i * 4096;
        unsigned* p = (unsigned*)((char*)base + off_bytes);
        probe<<<1,1>>>(p, iters, d_out);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_out, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        double c = (double)*h_out / iters;
        cyc.push_back(c);
    }
    // determine threshold = avg of min&max
    double mn=cyc[0], mx=cyc[0];
    for (double c : cyc) { if(c<mn)mn=c; if(c>mx)mx=c; }
    double thr = (mn+mx)*0.5;
    for (int i = 0; i < n_offsets; i++) {
        char s = cyc[i] > thr ? 'F' : 'N';
        printf("%3d,%.1f,%c\n", i*4, cyc[i], s);
    }
    printf("Side fingerprint (4KB stride): ");
    for (int i = 0; i < n_offsets; i++) printf("%c", cyc[i] > thr ? 'F' : 'N');
    printf("\n\n");
}

int main(int argc, char** argv) {
    int n_off = 32; // 128 KB sweep
    unsigned long long *d_out, *h_out;
    CHECK(cudaMalloc(&d_out, sizeof(unsigned long long)));
    CHECK(cudaMallocHost(&h_out, sizeof(unsigned long long)));
    size_t sz = 8 * 1024 * 1024;

    // Method 1: plain cudaMalloc
    {
        unsigned* p; CHECK(cudaMalloc(&p, sz));
        CHECK(cudaMemset(p, 0, sz));
        sweep("M1 cudaMalloc", p, n_off, d_out, h_out);
        cudaFree(p);
    }
    cudaMemLocation loc; loc.type = cudaMemLocationTypeDevice; loc.id = 0;
    // Method 2: cudaMallocManaged + cudaMemAdvise(SetPreferredLocation = device 0)
    {
        unsigned* p; CHECK(cudaMallocManaged(&p, sz));
        CHECK(cudaMemAdvise(p, sz, cudaMemAdviseSetPreferredLocation, loc));
        CHECK(cudaMemPrefetchAsync(p, sz, loc, 0, 0));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(p, 0, sz));
        CHECK(cudaDeviceSynchronize());
        sweep("M2 Managed+SetPreferredLocation(dev0)", p, n_off, d_out, h_out);
        cudaFree(p);
    }
    // Method 3: cudaMallocManaged + cudaMemAdvise(SetAccessedBy = device 0)
    {
        unsigned* p; CHECK(cudaMallocManaged(&p, sz));
        CHECK(cudaMemAdvise(p, sz, cudaMemAdviseSetAccessedBy, loc));
        CHECK(cudaMemPrefetchAsync(p, sz, loc, 0, 0));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(p, 0, sz));
        CHECK(cudaDeviceSynchronize());
        sweep("M3 Managed+SetAccessedBy(dev0)", p, n_off, d_out, h_out);
        cudaFree(p);
    }
    // Method 4: cudaMallocAsync (mempool)
    {
        unsigned* p;
        cudaStream_t s; CHECK(cudaStreamCreate(&s));
        CHECK(cudaMallocAsync(&p, sz, s));
        CHECK(cudaMemsetAsync(p, 0, sz, s));
        CHECK(cudaStreamSynchronize(s));
        sweep("M4 cudaMallocAsync", p, n_off, d_out, h_out);
        CHECK(cudaFreeAsync(p, s));
        CHECK(cudaStreamSynchronize(s));
    }
    // Method 5: cudaMalloc + multiple kernel launches with launchAttribute MemSyncDomain=Remote
    // (test if MemSyncDomain affects placement)
    {
        unsigned* p; CHECK(cudaMalloc(&p, sz));
        CHECK(cudaMemset(p, 0, sz));
        cudaLaunchAttribute attr = {};
        attr.id = cudaLaunchAttributeMemSyncDomain;
        attr.val.memSyncDomain = cudaLaunchMemSyncDomainRemote;
        // Just verify property is settable; sweep with normal launch since we want side pattern
        printf("--- M5 cudaMalloc+MemSyncDomain=Remote (note: domain affects fences only) ---\n");
        sweep("M5 baseline (same alloc, normal launch)", p, n_off, d_out, h_out);
        cudaFree(p);
    }
    return 0;
}
