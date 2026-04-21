// V7 J5: Cross-process persistent kernel via IPC handle
// Parent launches persistent kernel; child sends signals via shared IPC mailbox
#include <cuda_runtime.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <thread>

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void persistent_worker(volatile unsigned int* mailbox, volatile unsigned int* ack, int n_tasks) {
    if (threadIdx.x != 0) return;
    for (int task = 1; task <= n_tasks; task++) {
        while (*mailbox != (unsigned)task) {}
        *ack = task;
        __threadfence_system();
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

        cudaIpcMemHandle_t mb_handle, ack_handle;
        if (read(sv[1], &mb_handle, sizeof(mb_handle)) != sizeof(mb_handle)) exit(1);
        if (read(sv[1], &ack_handle, sizeof(ack_handle)) != sizeof(ack_handle)) exit(1);

        unsigned int *mb, *ack;
        CK(cudaIpcOpenMemHandle((void**)&mb, mb_handle, cudaIpcMemLazyEnablePeerAccess));
        CK(cudaIpcOpenMemHandle((void**)&ack, ack_handle, cudaIpcMemLazyEnablePeerAccess));

        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // wait for kernel start

        int N = 50;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int task = 1; task <= N; task++) {
            // Write to mailbox (CUDA memory accessible via IPC)
            unsigned int task_v = task;
            CK(cudaMemcpy(mb, &task_v, sizeof(task_v), cudaMemcpyHostToDevice));
            // Wait for ack
            unsigned int got = 0;
            while (got != task_v) {
                CK(cudaMemcpy(&got, ack, sizeof(got), cudaMemcpyDeviceToHost));
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double per_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

        printf("[CHILD] cross-process persistent RTT (via IPC + cudaMemcpy): %.2f us/task\n", per_us);

        cudaIpcCloseMemHandle(mb);
        cudaIpcCloseMemHandle(ack);
        char ackmsg = 'X';
        write(sv[1], &ackmsg, 1);
        close(sv[1]);
        return 0;
    } else {
        // PARENT
        close(sv[1]);
        cudaSetDevice(0);

        unsigned int *mailbox, *ack;
        CK(cudaMalloc(&mailbox, sizeof(unsigned int)));
        CK(cudaMalloc(&ack, sizeof(unsigned int)));
        unsigned int zero = 0;
        CK(cudaMemcpy(mailbox, &zero, sizeof(zero), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ack, &zero, sizeof(zero), cudaMemcpyHostToDevice));

        cudaIpcMemHandle_t mb_handle, ack_handle;
        CK(cudaIpcGetMemHandle(&mb_handle, mailbox));
        CK(cudaIpcGetMemHandle(&ack_handle, ack));

        write(sv[0], &mb_handle, sizeof(mb_handle));
        write(sv[0], &ack_handle, sizeof(ack_handle));

        // Launch persistent kernel
        persistent_worker<<<1, 32>>>((volatile unsigned int*)mailbox, (volatile unsigned int*)ack, 50);

        // Wait for child to finish
        char ackmsg;
        read(sv[0], &ackmsg, 1);

        cudaDeviceSynchronize();
        cudaFree(mailbox); cudaFree(ack);
        int status; waitpid(pid, &status, 0);
        close(sv[0]);
        return 0;
    }
}
