class benchmarkResults {
public:
  float peak_pct;
  float gigaops_per_second;

  std::vector<float> run_times;
  float total_cpu_timed = 0.0f;
  float min_time = 0.0f;
  float avg_time = 0.0f;
  uint threads = 0;

  uint avg_power = 0;
  uint avg_clock = 0;
  uint avg_temp = 0;

  uint max_power = 0;
  uint max_clock = 0;
  uint max_temp = 0;
  uint min_power = 0;
  uint min_clock = 0;

  uint size_multiplier = 1;
  std::vector<double> metrics;
  std::vector<nvmlStats> samples;
};

std::vector<std::string> csv_header = { "name",
                                        "peak_pct",
                                        "max_power",
                                        "min_clock",
                                        "gigaops_per_second",
                                        " ",
                                        "threads",
                                        "min_time",
                                        "avg_time",
                                        "avg_power",
                                        "avg_clock",
                                        "avg_temp",
                                        "max_clock",
                                        "max_temp",
                                        "min_power",
                                        "min_time_adjusted",
                                        "avg_time_adjusted",
                                        "total_cpu_timed",
                                        " ",
                                        "instruction_issue_pct",    // SM Instruction Issue %
                                        "dram_pct",                 // DRAM %
                                        "l2_pct",                   // L2 %
                                        "l1_pct",                   // L1 %
                                        "dram_bytes_read",          // DRAM Read Bytes
                                        "dram_bytes_write",         // DRAM Write Bytes
                                        "l2_bytes_read",            // L2 Read Bytes
                                        "l2_bytes_write",           // L2 Write Bytes
                                        "warps_occupancy_sm0",      // Warp Occupancy % (1st SM)
                                        "warps_eligible_all"        // Warps Eligible (All SMs)
};

class testRunner {
public:
    testRunner(CudaHelper &_cuda, nvmlClass &_nvml, std::string const &filename="results.csv") :
        CUDA(_cuda), nvml(_nvml) {
        outfile_.open( filename, std::ios::out );
        printHeader();
    }

    ~testRunner() {
        outfile_.close();
    }

    void printHeader() {
        for (int i = 0; i < csv_header.size(); i++) {
            outfile_ << csv_header[i];
            if (i < csv_header.size() - 1) {
                outfile_ << ", ";
            }
        }
        outfile_ << "\n";
    }

    void writeResults(const char* name, benchmarkResults const &results) {
        outfile_ << fixed << setprecision( 2 );

        float min_time_ajusted = results.min_time / results.size_multiplier;
        float avg_time_ajusted = results.avg_time / results.size_multiplier;
        
        outfile_ << std::setw(16) << name << std::setw(6) << ", " << results.peak_pct << ", "
                 << results.max_power / 1000.0f << ", " << results.min_clock << ", " << results.gigaops_per_second << ", "
                 << "     , " << results.threads << ", " << results.min_time << ", " << results.avg_time << ", "
                 << results.avg_power / 1000.0f << ", " << results.avg_clock << ", " << results.avg_temp << ", "
                 << results.max_clock << ", " << results.max_temp << ", " << results.min_power / 1000.0f << ", "
                 << min_time_ajusted << ", " << avg_time_ajusted << ", " << results.total_cpu_timed << ", "
                 << "     , ";
        
        for (int i = 0; i < results.metrics.size(); i++) {
            outfile_ << results.metrics[i];
            if (i < results.metrics.size() - 1) {
                outfile_ << ", ";
            }
        }
        outfile_ << "\n";
    }

    benchmarkResults launchBenchmarkClockSweep(std::vector<uint> clockSweep, int sleep_time,
                                           const char* name, CUfunction func, void** args,
                                           uint ops_per_thread=0, float ops_per_cycle=MAX_FLOPS_PER_CLOCK, float run_duration_ms=1000.0f,
                                           int warmup=1, int timedruns=4, bool cooperative=false, bool print=false) {
        benchmarkResults results;
        for (int i = 0; i < clockSweep.size(); i++) {
            nvml.lockClocks(clockSweep[i]);
            auto name_with_clock = std::string(name) + "_clock" + std::to_string(clockSweep[i]);
            results = launchSimpleBenchmark(name_with_clock.c_str(), func, args, ops_per_thread, ops_per_cycle, run_duration_ms,
                                            warmup, timedruns, cooperative, print);
            // Sleep for sleep_time ms
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        }
        return results;
    }
    
    benchmarkResults launchSimpleBenchmark(const char* name, CUfunction func, void** args,
                                           uint ops_per_thread=0, float ops_per_cycle=MAX_FLOPS_PER_CLOCK, float run_duration_ms=1000.0f,
                                           int warmup=1, int timedruns=4, bool cooperative=false, bool print=false) {
        benchmarkResults results;
        dim3 block_dims;
        dim3 thread_dims;
        CUDA.getKernelSize(block_dims, thread_dims);
        CUDA.deltaTimerCPU();

        cout << "Running benchmark: " << name << endl;

        for (int i = 0; i < warmup + timedruns; i++) {
            if (i == warmup) {
                // 1st timed run: get metrics (NVIDIA performance counters) during this run only
                measureMultiStart();
            }
            if (i == warmup+1 && timedruns >= 2) {
                nvml.clearSamples();
            }

            // Launch kernel
            float milliseconds = CUDA.launchSimpleTest(func, args, cooperative);

            if (i == warmup) {
                results.metrics = measureMultiStop();
                if (print) {
                    for (int i = 0; i < results.metrics.size(); i++) {
                        printf("Metric %d: %f (per second: %f)\n",
                            i, results.metrics[i], results.metrics[i] / (milliseconds / 1000.0));
                    }
                }
            }
            if (i >= warmup) {
                // Timed run
                results.run_times.push_back(milliseconds);
                results.avg_time += milliseconds / (float)timedruns;
                if (milliseconds < results.min_time || results.min_time == 0.0f) {
                    results.min_time = milliseconds;
                }
            } else {
                // Increase size if warmup was too fast
                dim3 new_block_dims = block_dims;
                if (milliseconds < run_duration_ms) {
                    int factor = max(1, (int)(run_duration_ms / milliseconds));
                    results.size_multiplier *= factor;
                    new_block_dims.x *= factor;
                    milliseconds *= (float)factor;
                    CUDA.setKernelSize(new_block_dims, thread_dims);
                }
            }
        }
        // Reset original kernel size in case we modified it
        CUDA.setKernelSize(block_dims, thread_dims);
        // Calculate total number of threads (after doublings, if any)
        results.threads = block_dims.x * thread_dims.x * block_dims.y * thread_dims.y * block_dims.z * thread_dims.z;
        results.threads *= results.size_multiplier;
        results.total_cpu_timed = CUDA.deltaTimerCPU();

        nvml.getSamples(results.samples);
        nvml.extractAvgPowerClockTemp(results.samples, results.avg_power, results.avg_clock, results.avg_temp);
        nvml.extractPeakPowerClockTemp(results.samples, results.max_power, results.min_clock, results.max_temp,
                                        results.min_power, results.min_clock);

        results.gigaops_per_second = (((float)ops_per_thread * (float)results.threads) / results.min_time) / 1e6f;
        results.peak_pct = 100.0f * (results.gigaops_per_second * 1e9f / (ops_per_cycle * results.min_clock));

        writeResults(name, results);
        return results;
    }

    benchmarkResults launchBenchmarkDeltaTime(const char* name, CUfunction func0, CUfunction func1, void** args0, void** args1,
                                              uint ops_per_thread=0, float ops_per_cycle=MAX_FLOPS_PER_CLOCK, float run_duration_ms=1000.0f,
                                              int warmup=1, int timedruns=3) {
        // This is typically used to run the same test twice with a different number of calculations (inner loop size)
        // This only works if the average and minimum clocks are the same though...
        std::string name0 = std::string(name) + "_0";
        std::string name1 = std::string(name) + "_1";
        std::string delta_name = std::string(name) + "_delta";
        auto results0 = launchSimpleBenchmark(name0.c_str(), func0, args0, ops_per_thread,     ops_per_cycle, run_duration_ms, warmup, timedruns);
        auto results1 = launchSimpleBenchmark(name1.c_str(), func1, args1, ops_per_thread * 2, ops_per_cycle, run_duration_ms, warmup, timedruns);

        if (results0.min_clock != results1.min_clock || results0.avg_clock != results1.avg_clock) {
            printf("Clocks are different, cannot compare results!\n");
            printf("Clock 0: %u, Clock 1: %u\n", results0.min_clock, results1.min_clock);
            printf("Clock 0: %u, Clock 1: %u\n", results0.avg_clock, results1.avg_clock);
            assert(false); // TODO
        }

        float time0 = results0.min_time / results0.size_multiplier;
        float time1 = results1.min_time / results1.size_multiplier;
        float time_delta = time1 - time0;

        float gigaops_per_second = (((float)ops_per_thread * (float)(results0.threads / results0.size_multiplier)) / time_delta) / 1e6f;
        float peak_pct = 100.0f * (gigaops_per_second * 1e9f / (ops_per_cycle * results0.max_clock));

        outfile_ << std::setw(16) << delta_name << std::setw(6) <<  ", " << peak_pct << ", , , " << gigaops_per_second << ", ";
        for (int i = 5; i < csv_header.size() - 1; i++) {
            outfile_ << ", ";
        }
        outfile_ << "\n";

        return results1;
    }

private:
    std::ofstream outfile_;

    CudaHelper &CUDA;
    nvmlClass &nvml;
};
