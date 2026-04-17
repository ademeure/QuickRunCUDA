// Probe all cudaLaunchAttribute capabilities on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void compute(float *out, int iters, int k) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + k * 0.00001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void memory_kernel(float *in, float *out, int N, int reps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0;
    for (int r = 0; r < reps; r++) {
        for (int i = tid; i < N; i += stride) {
            acc += in[i];
        }
    }
    if (acc == -42.0f) out[tid] = acc;
}

int main() {
    CK(cudaSetDevice(0));

    // ===== Device attribute probe =====
    int val;
    int probes[][2] = {
        {108, 108}, {109, 109}, {120, 120}, {126, 126},
        {133, 133}, {0, 0}
    };
    const char *names[] = {
        "MaxPersistingL2CacheSize",
        "MaxAccessPolicyWindowSize",
        "ClusterLaunch supported",
        "MemSyncDomainCount",
        "MpsEnabled",
        nullptr
    };

    printf("# B300 device attribute probe (relevant to launch attrs)\n");
    for (int i = 0; names[i]; i++) {
        cudaDeviceGetAttribute(&val, (cudaDeviceAttr)probes[i][0], 0);
        printf("  %-35s : %d (%s)\n", names[i], val,
               (probes[i][0]==108 || probes[i][0]==109) ? "bytes" : "");
    }

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    float *d_in, *d_out;
    int N = 64 << 20;  // 256MB
    CK(cudaMalloc(&d_in, N * sizeof(float)));
    CK(cudaMalloc(&d_out, N * sizeof(float)));
    CK(cudaMemset(d_in, 0x40, N * sizeof(float)));
    CK(cudaMemset(d_out, 0, N * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    // ===== Test 1: AccessPolicyWindow (L2 persisting) =====
    printf("\n## Test: cudaLaunchAttributeAccessPolicyWindow (L2 persisting cache)\n");
    {
        // Memory kernel that reads same window many times
        int N_window = 32 << 20;  // 128MB read region
        int reps = 4;

        // Baseline
        float t_base = bench([&]{
            memory_kernel<<<blocks, threads, 0, s>>>(d_in, d_out, N_window, reps);
        });

        // With AccessPolicyWindow (mark persisting)
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeAccessPolicyWindow;
        attr.val.accessPolicyWindow.base_ptr = d_in;
        attr.val.accessPolicyWindow.num_bytes = (size_t)N_window * sizeof(float);  // could be > max
        attr.val.accessPolicyWindow.hitRatio = 1.0f;
        attr.val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

        cudaLaunchConfig_t cfg = {dim3(blocks), dim3(threads), 0, s, &attr, 1};
        // Set L2 carveout for persisting
        size_t carveout_size = 0;
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxPersistingL2CacheSize, 0);
        cudaCtxResetPersistingL2Cache();

        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)val);
        printf("  Set L2 persisting carveout: %d bytes (%.1f MB)\n", val, val/(1024.f*1024.f));

        float t_persist = bench([&]{
            int args_N = N_window, args_reps = reps;
            void *args[] = {&d_in, &d_out, &args_N, &args_reps};
            cudaLaunchKernelExC(&cfg, (void*)memory_kernel, args);
        });
        printf("  baseline:        %.4f ms\n", t_base);
        printf("  persist policy:  %.4f ms (ratio %.2fx)\n", t_persist, t_persist / t_base);

        cudaCtxResetPersistingL2Cache();
    }

    // ===== Test 2: LaunchCompletionEvent =====
    printf("\n## Test: cudaLaunchAttributeLaunchCompletionEvent (event on completion)\n");
    {
        cudaEvent_t completion_event;
        CK(cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming));

        // Setup: kernel signals completion via event
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeLaunchCompletionEvent;
        attr.val.launchCompletionEvent.event = completion_event;
        attr.val.launchCompletionEvent.flags = 0;

        cudaLaunchConfig_t cfg = {dim3(blocks), dim3(threads), 0, s, &attr, 1};

        cudaStream_t s_cons;
        CK(cudaStreamCreateWithFlags(&s_cons, cudaStreamNonBlocking));

        float t_comp_event = bench([&]{
            int it = 5000, k = 0;
            void *args[] = {&d_out, &it, &k};
            cudaLaunchKernelExC(&cfg, (void*)compute, args);
            cudaStreamWaitEvent(s_cons, completion_event, cudaEventWaitDefault);
            compute<<<blocks,threads,0,s_cons>>>(d_out, 5000, 1);
        });

        // Compare: regular event
        cudaEvent_t reg_event;
        CK(cudaEventCreate(&reg_event));
        float t_reg_event = bench([&]{
            compute<<<blocks,threads,0,s>>>(d_out, 5000, 0);
            cudaEventRecord(reg_event, s);
            cudaStreamWaitEvent(s_cons, reg_event, 0);
            compute<<<blocks,threads,0,s_cons>>>(d_out, 5000, 1);
        });

        printf("  regular event:           %.4f ms\n", t_reg_event);
        printf("  LaunchCompletionEvent:   %.4f ms (diff %+.4f ms)\n",
               t_comp_event, t_comp_event - t_reg_event);

        cudaStreamDestroy(s_cons);
        cudaEventDestroy(completion_event);
        cudaEventDestroy(reg_event);
    }

    // ===== Test 3: cudaLaunchAttributePriority (per-launch priority) =====
    printf("\n## Test: cudaLaunchAttributePriority (per-launch prio)\n");
    {
        int prio_low, prio_high;
        cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high);
        printf("  Range: %d (high) to %d (low)\n", prio_high, prio_low);

        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributePriority;
        attr.val.priority = prio_high;

        cudaLaunchConfig_t cfg = {dim3(blocks), dim3(threads), 0, s, &attr, 1};

        float t_default = bench([&]{
            compute<<<blocks,threads,0,s>>>(d_out, 5000, 0);
        });
        float t_prio = bench([&]{
            int it = 5000, k = 0;
            void *args[] = {&d_out, &it, &k};
            cudaLaunchKernelExC(&cfg, (void*)compute, args);
        });
        printf("  default prio: %.4f ms\n", t_default);
        printf("  high prio:    %.4f ms\n", t_prio);
    }

    // ===== Test 4: PreferredSharedMemoryCarveout =====
    printf("\n## Test: cudaLaunchAttributePreferredSharedMemoryCarveout\n");
    {
        // 0 = max L1 cache, 100 = max shared memory
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributePreferredSharedMemoryCarveout;

        for (int carve : {0, 25, 50, 75, 100}) {
            attr.val.sharedMemCarveout = carve;
            cudaLaunchConfig_t cfg = {dim3(blocks), dim3(threads), 0, s, &attr, 1};

            float t = bench([&]{
                int it = 5000, k = 0;
                void *args[] = {&d_out, &it, &k};
                cudaLaunchKernelExC(&cfg, (void*)compute, args);
            });
            printf("  carveout=%-3d (0=max L1, 100=max shmem): %.4f ms\n", carve, t);
        }
    }

    cudaStreamDestroy(s);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
