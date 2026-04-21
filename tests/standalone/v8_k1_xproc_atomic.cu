// V8 K1: Cross-process atomicAdd via IPC
// Parent + child each launch kernel that atomicAdd N times to shared counter
// Verify final = parent_n + child_n (atomicity preserved across processes)
#include <cuda_runtime.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);} }while(0)

__global__ void atomic_kernel(unsigned int* counter, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        atomicAdd(counter, 1);
    }
}

int main() {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) < 0) { perror("socketpair"); return 1; }

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }

    if (pid == 0) {
        // CHILD
        close(sv[0]);
        cudaSetDevice(0);
        cudaIpcMemHandle_t handle;
        if (read(sv[1], &handle, sizeof(handle)) != sizeof(handle)) exit(1);

        unsigned int* counter;
        CK(cudaIpcOpenMemHandle((void**)&counter, handle, cudaIpcMemLazyEnablePeerAccess));

        int N_OPS = 100000;
        auto t0 = std::chrono::high_resolution_clock::now();
        atomic_kernel<<<128, 256>>>(counter, N_OPS);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("[CHILD] %d atomicAdds in %.2f ms\n", N_OPS, ms);

        CK(cudaIpcCloseMemHandle(counter));
        char ack = 'X'; write(sv[1], &ack, 1);
        close(sv[1]);
        return 0;
    } else {
        // PARENT
        close(sv[1]);
        cudaSetDevice(0);
        unsigned int* counter;
        CK(cudaMalloc(&counter, sizeof(unsigned int)));
        unsigned int zero = 0;
        cudaMemcpy(counter, &zero, 4, cudaMemcpyHostToDevice);

        cudaIpcMemHandle_t handle;
        CK(cudaIpcGetMemHandle(&handle, counter));
        write(sv[0], &handle, sizeof(handle));

        int N_OPS = 100000;
        auto t0 = std::chrono::high_resolution_clock::now();
        atomic_kernel<<<128, 256>>>(counter, N_OPS);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("[PARENT] %d atomicAdds in %.2f ms\n", N_OPS, ms);

        // Wait for child
        char ack; read(sv[0], &ack, 1);

        unsigned int final;
        cudaMemcpy(&final, counter, 4, cudaMemcpyDeviceToHost);
        printf("[PARENT] final counter = %u (expected %d)\n", final, N_OPS * 2);
        printf("[PARENT] match: %s\n", (final == (unsigned)(N_OPS * 2)) ? "YES — cross-process atomicity WORKS" : "NO");

        int status; waitpid(pid, &status, 0);
        cudaFree(counter);
        close(sv[0]);
        return 0;
    }
}
