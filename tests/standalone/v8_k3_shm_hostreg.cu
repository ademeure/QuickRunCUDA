// V8 K3: POSIX shm + cudaHostRegister across processes
// Goal: determine whether two processes can both cudaHostRegister the same shm
// region to get x-process pinned/mapped host memory (complement to K1/K2 device IPC).
//
// Design:
//  - Parent shm_open + ftruncate + mmap (MAP_SHARED)
//  - Parent cudaHostRegister with cudaHostRegisterMapped
//  - Fork; child inherits mmap but has a DIFFERENT CUDA context
//  - Child cudaSetDevice + cudaHostRegister on same host pointer
//  - Both processes:
//     (a) cudaMemcpyAsync pinned timing
//     (b) kernel read via cudaHostGetDevicePointer
//     (c) child writes "hello" bytes; parent reads after ack
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CKSOFT(c, name) do { cudaError_t e=(c); if(e){ fprintf(stderr,"[%s] %s FAILED: %s\n", name, #c, cudaGetErrorString(e)); } else { fprintf(stderr,"[%s] %s OK\n", name, #c); } }while(0)

static const char* SHM_NAME = "/v8_k3_shm";
static const size_t SHM_SIZE = 16 * 1024 * 1024;   // 16 MB
static const int PINNED_ITERS = 20;

__global__ void sum_kernel(unsigned int* dev_host_ptr, unsigned int* out, int n) {
    unsigned int sum = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) {
        sum += dev_host_ptr[i];
    }
    __shared__ unsigned int part[32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if (lane == 0) part[warp_id] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int s = 0;
        for (int i = 0; i < blockDim.x/32; i++) s += part[i];
        atomicAdd(out, s);
    }
}

__global__ void write_kernel(unsigned int* dev_host_ptr, int n, unsigned int val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) dev_host_ptr[i] = val;
}

int main() {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) < 0) { perror("socketpair"); return 1; }

    // Parent creates shm first
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0600);
    if (fd < 0) { perror("shm_open"); return 1; }
    if (ftruncate(fd, SHM_SIZE) < 0) { perror("ftruncate"); return 1; }

    void* mem = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED) { perror("mmap"); return 1; }

    // Initialize: all zeros
    memset(mem, 0, SHM_SIZE);

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }

    if (pid == 0) {
        // ============ CHILD ============
        close(sv[0]);
        // Child has inherited mmap because MAP_SHARED + fork

        cudaSetDevice(0);
        // Try to register in child's context (use Mapped for device-pointer access)
        cudaError_t rr = cudaHostRegister(mem, SHM_SIZE, cudaHostRegisterMapped | cudaHostRegisterPortable);
        fprintf(stderr, "[CHILD] cudaHostRegister(Mapped|Portable) on inherited shm: %s\n", cudaGetErrorString(rr));
        bool child_registered = (rr == cudaSuccess);

        // If initial registration failed, try plain default
        if (!child_registered) {
            rr = cudaHostRegister(mem, SHM_SIZE, cudaHostRegisterDefault);
            fprintf(stderr, "[CHILD] cudaHostRegister(Default) on inherited shm: %s\n", cudaGetErrorString(rr));
            child_registered = (rr == cudaSuccess);
        }

        // Signal parent we're ready
        char ready = 'R';
        write(sv[1], &ready, 1);

        // Wait for parent's go
        char go;
        if (read(sv[1], &go, 1) != 1) exit(1);

        // Test (a): child time pinned cudaMemcpyAsync if registered
        if (child_registered) {
            float* d_buf; CK(cudaMalloc(&d_buf, SHM_SIZE));
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < PINNED_ITERS; i++) {
                CK(cudaMemcpyAsync(d_buf, mem, SHM_SIZE, cudaMemcpyHostToDevice, 0));
            }
            CK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / PINNED_ITERS;
            double gbs = (SHM_SIZE / 1e9) / (us / 1e6);
            fprintf(stderr, "[CHILD] pinned H2D: %.1f us/iter = %.2f GB/s\n", us, gbs);
            cudaFree(d_buf);
        }

        // Test (b): child writes marker pattern via kernel through device-mapped
        if (child_registered) {
            unsigned int* dev_host_ptr;
            cudaError_t r = cudaHostGetDevicePointer(&dev_host_ptr, mem, 0);
            fprintf(stderr, "[CHILD] cudaHostGetDevicePointer: %s\n", cudaGetErrorString(r));
            if (r == cudaSuccess) {
                int n_int = SHM_SIZE / sizeof(unsigned int);
                write_kernel<<<256, 256>>>(dev_host_ptr, n_int, 0xC4C4C4C4u);
                CK(cudaDeviceSynchronize());
                fprintf(stderr, "[CHILD] wrote pattern 0xC4C4C4C4 through host-mapped device pointer\n");
            }
        } else {
            // Alternative: child unpinned access fallback timing
            float* d_buf; CK(cudaMalloc(&d_buf, SHM_SIZE));
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 5; i++) {
                CK(cudaMemcpyAsync(d_buf, mem, SHM_SIZE, cudaMemcpyHostToDevice, 0));
            }
            CK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 5;
            double gbs = (SHM_SIZE / 1e9) / (us / 1e6);
            fprintf(stderr, "[CHILD] UNPINNED fallback H2D: %.1f us/iter = %.2f GB/s\n", us, gbs);
            cudaFree(d_buf);
            // Also write pattern directly via CPU (no device-mapped path)
            unsigned int* p = (unsigned int*)mem;
            for (int i = 0; i < 64; i++) p[i] = 0xC4C4C4C4u;
        }

        // Signal done
        char done = 'D';
        write(sv[1], &done, 1);

        if (child_registered) cudaHostUnregister(mem);
        munmap(mem, SHM_SIZE);
        close(sv[1]);
        return 0;
    } else {
        // ============ PARENT ============
        close(sv[1]);

        cudaSetDevice(0);
        cudaError_t pr = cudaHostRegister(mem, SHM_SIZE, cudaHostRegisterMapped | cudaHostRegisterPortable);
        fprintf(stderr, "[PARENT] cudaHostRegister(Mapped|Portable) on shm: %s\n", cudaGetErrorString(pr));
        bool parent_registered = (pr == cudaSuccess);

        // Pinned timing
        if (parent_registered) {
            float* d_buf; CK(cudaMalloc(&d_buf, SHM_SIZE));
            // Warmup
            for (int i = 0; i < 3; i++) cudaMemcpyAsync(d_buf, mem, SHM_SIZE, cudaMemcpyHostToDevice, 0);
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < PINNED_ITERS; i++) {
                CK(cudaMemcpyAsync(d_buf, mem, SHM_SIZE, cudaMemcpyHostToDevice, 0));
            }
            CK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / PINNED_ITERS;
            double gbs = (SHM_SIZE / 1e9) / (us / 1e6);
            fprintf(stderr, "[PARENT] pinned H2D (baseline before child marker): %.1f us/iter = %.2f GB/s\n", us, gbs);
            cudaFree(d_buf);
        }

        // Wait for child to signal ready
        char ready;
        if (read(sv[0], &ready, 1) != 1) exit(1);
        // Parent tells child GO
        char go = 'G';
        write(sv[0], &go, 1);

        // Wait for child done
        char done;
        if (read(sv[0], &done, 1) != 1) exit(1);

        // Verify child's marker arrived (read via CPU)
        unsigned int* p = (unsigned int*)mem;
        int n_int = SHM_SIZE / sizeof(unsigned int);
        int matches = 0;
        for (int i = 0; i < 64; i++) if (p[i] == 0xC4C4C4C4u) matches++;
        fprintf(stderr, "[PARENT] marker bytes from child (first 64 words): %d/64 match\n", matches);

        // Sum the whole thing via kernel through host device ptr (if mapped)
        if (parent_registered) {
            unsigned int* dev_host_ptr;
            cudaError_t r = cudaHostGetDevicePointer(&dev_host_ptr, mem, 0);
            if (r == cudaSuccess) {
                unsigned int* d_out; cudaMalloc(&d_out, sizeof(unsigned int));
                cudaMemset(d_out, 0, 4);
                sum_kernel<<<256, 256>>>(dev_host_ptr, d_out, n_int);
                CK(cudaDeviceSynchronize());
                unsigned int out;
                cudaMemcpy(&out, d_out, 4, cudaMemcpyDeviceToHost);
                unsigned int expected = 0xC4C4C4C4u * (unsigned)n_int;
                fprintf(stderr, "[PARENT] kernel-sum via mapped ptr = 0x%08x (expected 0x%08x, match=%s)\n",
                        out, expected, (out == expected) ? "YES" : "NO");
                cudaFree(d_out);
            }
        }

        int status; waitpid(pid, &status, 0);

        if (parent_registered) cudaHostUnregister(mem);
        munmap(mem, SHM_SIZE);
        shm_unlink(SHM_NAME);
        close(sv[0]);

        fprintf(stderr, "\n=== V8 K3 summary ===\n");
        fprintf(stderr, "  parent register: %s\n", parent_registered ? "OK" : "FAILED");
        fprintf(stderr, "  (child result printed above)\n");
        return 0;
    }
}
