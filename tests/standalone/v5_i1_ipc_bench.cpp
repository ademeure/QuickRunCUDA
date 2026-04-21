// V5 I1: IPC handle cross-process benchmark
// Parent allocates GPU mem, exports IPC handle.
// Forks child; child opens handle; both write/read; measure costs.
#include <cuda_runtime.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

int main() {
    // Need a UNIX socketpair to pass IPC handle bytes between parent and child
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) < 0) { perror("socketpair"); return 1; }

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }

    if (pid == 0) {
        // === CHILD ===
        close(sv[0]);
        cudaSetDevice(0);

        // Receive IPC handle from parent
        cudaIpcMemHandle_t handle;
        ssize_t n = read(sv[1], &handle, sizeof(handle));
        if (n != sizeof(handle)) { perror("read"); return 1; }

        // Open the handle
        auto t0 = std::chrono::high_resolution_clock::now();
        void* dev_ptr = nullptr;
        CK(cudaIpcOpenMemHandle(&dev_ptr, handle, cudaIpcMemLazyEnablePeerAccess));
        auto t1 = std::chrono::high_resolution_clock::now();

        // Read first int from parent's buffer to verify access
        int val;
        CK(cudaMemcpy(&val, dev_ptr, sizeof(int), cudaMemcpyDeviceToHost));
        auto t2 = std::chrono::high_resolution_clock::now();

        double open_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        double access_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
        printf("[CHILD] cudaIpcOpenMemHandle = %.1f us\n", open_us);
        printf("[CHILD] read (4B from parent's buf) = %.1f us, val=0x%X\n", access_us, val);

        // Write back from child
        int write_val = 0xCAFEBABE;
        CK(cudaMemcpy(dev_ptr, &write_val, sizeof(int), cudaMemcpyHostToDevice));

        // Notify parent we're done (1-byte ack)
        char ack = 'X';
        write(sv[1], &ack, 1);

        CK(cudaIpcCloseMemHandle(dev_ptr));
        close(sv[1]);
        return 0;
    } else {
        // === PARENT ===
        close(sv[1]);
        cudaSetDevice(0);

        // Allocate
        size_t bytes = 1 << 20;  // 1 MB
        int* dev_buf;
        CK(cudaMalloc((void**)&dev_buf, bytes));
        int sentinel = 0xDEADBEEF;
        CK(cudaMemcpy(dev_buf, &sentinel, sizeof(int), cudaMemcpyHostToDevice));

        // Get IPC handle
        auto t0 = std::chrono::high_resolution_clock::now();
        cudaIpcMemHandle_t handle;
        CK(cudaIpcGetMemHandle(&handle, dev_buf));
        auto t1 = std::chrono::high_resolution_clock::now();

        double get_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        printf("[PARENT] cudaIpcGetMemHandle = %.1f us\n", get_us);

        // Send handle to child
        write(sv[0], &handle, sizeof(handle));

        // Wait for child to ack
        char ack;
        read(sv[0], &ack, 1);

        // Read back what child wrote
        int child_val;
        CK(cudaMemcpy(&child_val, dev_buf, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[PARENT] read after child write: 0x%X\n", child_val);

        int status;
        waitpid(pid, &status, 0);
        CK(cudaFree(dev_buf));
        close(sv[0]);
        return 0;
    }
}
