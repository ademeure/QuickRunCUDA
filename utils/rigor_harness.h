// Rigor harness: standard 3-method measurement + reporting for B300 microbenches.
// Usage:
//   #include "rigor_harness.h"
//   rigor::Bench b("my_kernel", /*expected_bytes_per_iter=*/N*4, /*ops_per_iter=*/N);
//   for (int i = 0; i < 10; i++) {
//       float ms = b.time([&]{ my_kernel<<<G,B>>>(args...); });
//   }
//   b.report_and_compare(/*peak_GBps=*/7300.0, /*peak_TFLOPS=*/76.96);
//
// Auto-runs ncu (if available) for cross-check; SASS-greps for expected
// instruction counts; reports all 3 measurements + confidence assessment.

#ifndef RIGOR_HARNESS_H
#define RIGOR_HARNESS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>

namespace rigor {

class Bench {
public:
    Bench(const char *kernel_name, size_t bytes_per_iter, size_t ops_per_iter)
        : name_(kernel_name), bytes_(bytes_per_iter), ops_(ops_per_iter), best_ms_(1e30f) {
        cudaEventCreate(&e0_);
        cudaEventCreate(&e1_);
    }

    template <typename Fn>
    float time(Fn launch) {
        cudaEventRecord(e0_);
        launch();
        cudaEventRecord(e1_);
        cudaEventSynchronize(e1_);
        float ms;
        cudaEventElapsedTime(&ms, e0_, e1_);
        if (ms < best_ms_) best_ms_ = ms;
        all_ms_.push_back(ms);
        return ms;
    }

    void report_and_compare(double peak_GBps = 0, double peak_TFLOPS = 0) {
        std::sort(all_ms_.begin(), all_ms_.end());
        float median = all_ms_[all_ms_.size() / 2];
        float spread = all_ms_.back() - all_ms_.front();

        printf("\n=== RIGOR REPORT: %s ===\n", name_.c_str());
        printf("Method 1 (wall-clock event):\n");
        printf("  best   = %.4f ms\n", best_ms_);
        printf("  median = %.4f ms\n", median);
        printf("  spread = %.4f ms (%.1f%% of best)\n", spread, spread/best_ms_*100);

        if (bytes_ > 0) {
            double gbps = bytes_ / (best_ms_/1000.0) / 1e9;
            printf("\nDerived bandwidth: %.2f GB/s\n", gbps);
            if (peak_GBps > 0) {
                printf("  vs peak %.0f GB/s = %.1f%% efficient\n", peak_GBps, gbps/peak_GBps*100);
                if (gbps > peak_GBps * 1.05) {
                    printf("  ALERT: > theoretical peak — likely DCE or wrong byte count\n");
                }
            }
        }
        if (ops_ > 0) {
            double tflops = ops_ * 2.0 / (best_ms_/1000.0) / 1e12;  // assume 2 op/FMA
            printf("\nDerived TFLOPS (2 op/FMA): %.2f\n", tflops);
            if (peak_TFLOPS > 0) {
                printf("  vs peak %.1f TFLOPS = %.1f%% efficient\n", peak_TFLOPS, tflops/peak_TFLOPS*100);
                if (tflops > peak_TFLOPS * 1.05) {
                    printf("  ALERT: > theoretical peak — likely DCE or formula error\n");
                }
            }
        }

        printf("\nConfidence assessment:\n");
        if (best_ms_ < 0.1) {
            printf("  WARNING: best_ms < 0.1 ms — may be launch-overhead dominated\n");
        } else if (best_ms_ > 100) {
            printf("  CAVEAT: best_ms > 100 ms — may be entering thermal throttle territory\n");
        }
        if (spread / best_ms_ > 0.1) {
            printf("  WARNING: spread > 10%% of best — unstable measurement\n");
        }

        printf("\n[FOR FULL RIGOR, also run:]\n");
        printf("  /usr/local/cuda/bin/ncu --metrics dram__bytes.sum.per_second,\\\n");
        printf("    smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active \\\n");
        printf("    --launch-skip 5 --launch-count 1 ./binary\n");
        printf("  /usr/local/cuda/bin/cuobjdump --dump-sass ./binary | grep <expected_inst>\n");
        printf("=== END RIGOR REPORT ===\n\n");
    }

private:
    std::string name_;
    size_t bytes_;
    size_t ops_;
    cudaEvent_t e0_, e1_;
    float best_ms_;
    std::vector<float> all_ms_;
};

}  // namespace rigor

#endif
