// PCIe BW sweep via IPC: client reads varying sizes from server's buffer
#include <cuda_runtime.h>
#include <cstdio>
#include <unistd.h>
#define CHK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); exit(1); } } while(0)
int main(int argc, char**argv) {
    cudaSetDevice(0);
    bool srv = (argc > 1 && argv[1][0] == 's');
    if (srv) {
        size_t bytes = 256ull * 1024 * 1024;  // 256 MB
        void *d_buf;
        CHK(cudaMalloc(&d_buf, bytes));
        cudaMemset(d_buf, 0x42, bytes);
        cudaDeviceSynchronize();
        cudaIpcMemHandle_t h;
        CHK(cudaIpcGetMemHandle(&h, d_buf));
        FILE* f = fopen("/tmp/ipc_pcie_handle.bin", "wb");
        fwrite(&h, sizeof(h), 1, f); fclose(f);
        f = fopen("/tmp/ipc_pcie_ready", "w"); fputs("y", f); fclose(f);
        while (access("/tmp/ipc_pcie_done", F_OK) != 0) usleep(1000);
        cudaFree(d_buf);
        unlink("/tmp/ipc_pcie_handle.bin"); unlink("/tmp/ipc_pcie_ready"); unlink("/tmp/ipc_pcie_done");
    } else {
        while (access("/tmp/ipc_pcie_ready", F_OK) != 0) usleep(1000);
        cudaIpcMemHandle_t h;
        FILE* f = fopen("/tmp/ipc_pcie_handle.bin", "rb");
        fread(&h, sizeof(h), 1, f); fclose(f);
        void *d_buf;
        CHK(cudaIpcOpenMemHandle(&d_buf, h, cudaIpcMemLazyEnablePeerAccess));
        // Pinned host
        char *h_buf;
        CHK(cudaMallocHost((void**)&h_buf, 256 * 1024 * 1024));
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        // Sweep sizes
        for (size_t sz : {(size_t)4096, (size_t)16384, (size_t)65536, (size_t)262144, (size_t)1048576, (size_t)4194304, (size_t)16777216, (size_t)67108864, (size_t)268435456}) {
            int iters = (sz <= 65536) ? 1000 : (sz <= 1048576 ? 500 : (sz <= 16777216 ? 100 : 30));
            // Warmup
            for (int i = 0; i < 3; i++) cudaMemcpy(h_buf, d_buf, sz, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            cudaEventRecord(e0);
            for (int i = 0; i < iters; i++) cudaMemcpy(h_buf, d_buf, sz, cudaMemcpyDeviceToHost);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            double per_op_us = (double)ms * 1000.0 / iters;
            double gbps = (double)sz * iters / (ms / 1000.0) / 1e9;
            printf("[CLI] D2H %8zu B x%4d: %.2f us/op = %.1f GB/s\n", sz, iters, per_op_us, gbps);
        }
        cudaIpcCloseMemHandle(d_buf);
        cudaFreeHost(h_buf);
        f = fopen("/tmp/ipc_pcie_done", "w"); fputs("y", f); fclose(f);
    }
    return 0;
}
