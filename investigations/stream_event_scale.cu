// Test stream/event creation at scale - is there a limit or cost growth?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>

int main() {
    cudaSetDevice(0);

    printf("# B300 stream/event creation scale test\n\n");

    // Stream creation - test up to 10000
    printf("## Stream creation rate (with NonBlocking flag)\n");
    {
        std::vector<cudaStream_t> ss;
        ss.reserve(10000);

        for (int target : {100, 1000, 5000, 10000}) {
            // Reset
            for (auto &s : ss) cudaStreamDestroy(s);
            ss.clear();

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < target; i++) {
                cudaStream_t s;
                cudaError_t r = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
                if (r != cudaSuccess) {
                    printf("    FAIL at %d: %s\n", i, cudaGetErrorString(r));
                    break;
                }
                ss.push_back(s);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            printf("  Created %d streams: %.2f ms = %.2f us/stream\n",
                   (int)ss.size(), ms, ms*1000/ss.size());
        }

        // Cleanup
        for (auto &s : ss) cudaStreamDestroy(s);
    }

    // Event creation - test up to 100000
    printf("\n## Event creation rate (with DisableTiming flag)\n");
    {
        std::vector<cudaEvent_t> es;
        es.reserve(100000);

        for (int target : {1000, 10000, 50000, 100000}) {
            for (auto &e : es) cudaEventDestroy(e);
            es.clear();

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < target; i++) {
                cudaEvent_t e;
                cudaError_t r = cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
                if (r != cudaSuccess) {
                    printf("    FAIL at %d: %s\n", i, cudaGetErrorString(r));
                    break;
                }
                es.push_back(e);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            printf("  Created %d events: %.2f ms = %.3f us/event\n",
                   (int)es.size(), ms, ms*1000/es.size());
        }

        for (auto &e : es) cudaEventDestroy(e);
    }

    // Skip max-streams test (memory exhaustion at high counts)

    return 0;
}
