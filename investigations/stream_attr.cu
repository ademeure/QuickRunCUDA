// cudaStreamSetAttribute / GetAttribute exploration
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    printf("# B300 stream attribute exploration\n\n");

    // CCR (compute-control region) - new in newer CUDA
    // SyncPolicy
    cudaStreamAttrValue attr;
    cudaError_t err;

    err = cudaStreamGetAttribute(s, cudaStreamAttributeSynchronizationPolicy, &attr);
    printf("  SynchronizationPolicy:        %s (val=%d)\n",
           err ? "ERR" : "OK", attr.syncPolicy);

    err = cudaStreamGetAttribute(s, cudaStreamAttributePriority, &attr);
    printf("  Priority:                     %s (val=%d)\n",
           err ? "ERR" : "OK", attr.priority);

    err = cudaStreamGetAttribute(s, cudaStreamAttributeMemSyncDomain, &attr);
    printf("  MemSyncDomain:                %s (val=%d)\n",
           err ? "ERR" : "OK", attr.memSyncDomain);

    err = cudaStreamGetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
    printf("  AccessPolicyWindow:           %s\n", err ? "ERR" : "OK");
    if (!err) {
        printf("    base_ptr: %p, num_bytes: %zu\n",
               attr.accessPolicyWindow.base_ptr, attr.accessPolicyWindow.num_bytes);
    }

    err = cudaStreamGetAttribute(s, cudaStreamAttributeMemSyncDomainMap, &attr);
    printf("  MemSyncDomainMap:             %s\n", err ? "ERR" : "OK");
    if (!err) {
        printf("    default: %u, remote: %u\n",
               attr.memSyncDomainMap.default_, attr.memSyncDomainMap.remote);
    }

    // Setting SyncPolicy options
    printf("\n## SyncPolicy options:\n");
    int policies[] = {cudaSyncPolicyAuto, cudaSyncPolicySpin, cudaSyncPolicyYield, cudaSyncPolicyBlockingSync};
    const char *policy_names[] = {"Auto", "Spin", "Yield", "BlockingSync"};
    for (int i = 0; i < 4; i++) {
        attr.syncPolicy = (cudaSynchronizationPolicy)policies[i];
        err = cudaStreamSetAttribute(s, cudaStreamAttributeSynchronizationPolicy, &attr);
        printf("  Set policy=%s: %s\n", policy_names[i], cudaGetErrorString(err));
    }

    return 0;
}
