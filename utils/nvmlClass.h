/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* This is a header class that utilizes NVML library.
 */

#ifndef NVMLCLASS_H_
#define NVMLCLASS_H_

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include <nvml.h>

int constexpr size_of_vector { 100000 };
int constexpr nvml_device_name_buffer_size { 100 };

// *************** FOR ERROR CHECKING *******************
#ifndef NVML_RT_CALL
#define NVML_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<nvmlReturn_t>( call );                                                               \
        if ( status != NVML_SUCCESS )                                                                                  \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA NVML call \"%s\" in line %d of file %s failed "                                      \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     nvmlErrorString( status ),                                                                        \
                     status );                                                                                         \
    }
#endif  // NVML_RT_CALL
// *************** FOR ERROR CHECKING *******************

typedef struct _nvml_stats {
    std::time_t        timestamp;
    uint               temperature;
    uint               powerUsage;
    nvmlUtilization_t  utilization;
    nvmlMemory_t       memory;
    uint               clockSM;
    uint               clockMemory;
} nvmlStats;

class nvmlClass {
  public:
    nvmlClass( int const &deviceID, uint force_clock=0, bool gather_stats=true, bool force_fan_speed=true,
               bool write_csv=true, std::string const &filename="gpuStats.csv" ) :
        time_steps_ {}, filename_ { filename }, outfile_ {}, device_ {}, forced_fan_ { force_fan_speed },
        loop_ { false }, outer_loop_ { true }, inside_loop_ { false }, write_csv_ { write_csv } {

        // Initialize NVML library and get device handle
        NVML_RT_CALL( nvmlInit() );
        NVML_RT_CALL( nvmlDeviceGetHandleByIndex(deviceID, &device_) );
        time_steps_.reserve( size_of_vector );

        if (force_clock == 1) {
            unlockClocks();
        } else if (force_clock > 1) {
            lockClocks(force_clock);
        }
        if (forced_fan_) {
            nvmlDeviceSetFanSpeed_v2(device_, 0, 100); // maximum fan speed
        }

        if (write_csv) {
            outfile_.open( filename_, std::ios::out );
            printHeader();
        }
        if (gather_stats) {
            startThread();
        } else {
            thread_alive_ = false;
        }
    }

    ~nvmlClass() {
        killThread(write_csv_);
        if (forced_fan_) {
            nvmlDeviceSetFanSpeed_v2(device_, 0, 0);  // Set fan speed back to auto (default)
        }
        NVML_RT_CALL(nvmlShutdown());
    }

    void startThread() {
        thread_alive_ = true;
        loop_ = true, outer_loop_ = true, inside_loop_ = true;
        statsThread_ = std::thread(&nvmlClass::sampleStatsLoop, this);
    }

    void killThread(bool write_csv_now=false) {
        if (thread_alive_) {
            pauseLoop();
            outer_loop_ = false;
            statsThread_.join();
            thread_alive_ = false;
        }
        if (write_csv_now) {
            writeData();
        }
    }

    void lockClocks(uint clock) {
        NVML_RT_CALL ( nvmlDeviceSetGpuLockedClocks(device_, clock, clock) );
    }

    void unlockClocks() {
        NVML_RT_CALL ( nvmlDeviceResetGpuLockedClocks(device_) );
    }

    void lockMemoryClocks(uint clock) {
        NVML_RT_CALL ( nvmlDeviceSetMemoryLockedClocks(device_, clock, clock) );
    }

    void unlockMemoryClocks() {
        NVML_RT_CALL ( nvmlDeviceResetMemoryLockedClocks(device_) );
    }

    // Clone time_steps in a thread safe way (make sure we are not currently inside inner loop)
    void getSamples(std::vector<nvmlStats> &samples, bool clear_data=true) {
        bool current_loop = pauseLoop();
        // Copy time_steps_ to samples

        samples.insert(samples.end(), time_steps_.begin(), time_steps_.end());
        //samples = time_steps_;

        if (clear_data) {
            time_steps_.clear();
        }
        resumeLoop(current_loop);
    }

    void clearSamples() {
        bool current_loop = pauseLoop();
        time_steps_.clear();
        resumeLoop(current_loop);
    }

    void extractPeakPowerClockTemp(const std::vector<nvmlStats> &samples,
                                   uint &max_power, uint &max_clock, uint &max_temp, uint &min_power, uint &min_clock,
                                   int min_util=0, bool ignore_first_power_sample=true, bool print=false) {
        max_power = 0;
        max_clock = 0;
        max_temp = 0;
        min_power = 100000000;
        min_clock = 100000000;
        for (int i = 0; i < static_cast<int>(samples.size()); i++) {
            // Even with NVML_FI_DEV_POWER_INSTANT, we still only get 1 sample every 100ms...
            if (ignore_first_power_sample) {
                if (max_power == 0) {
                    max_power = samples[i].powerUsage;
                    continue;
                } else if (samples[i].powerUsage == max_power) {
                    continue;
                } else {
                    ignore_first_power_sample = false;
                }
            }

            if (samples[i].utilization.gpu >= min_util) {
                if (samples[i].powerUsage > max_power) {
                    max_power = samples[i].powerUsage;
                }
                if (samples[i].clockSM > max_clock) {
                    max_clock = samples[i].clockSM;
                }
                if (samples[i].temperature > max_temp) {
                    max_temp = samples[i].temperature;
                }
                if (samples[i].powerUsage < min_power && samples[i].powerUsage > 0) {
                    min_power = samples[i].powerUsage;
                }
                if (samples[i].clockSM < min_clock && samples[i].clockSM > 0) {
                    min_clock = samples[i].clockSM;
                }
            }
        }
        if (print) {
            std::cout << "Max Power: " << max_power << " mW" << std::endl;
            std::cout << "Max Clock: " << max_clock << " MHz" << std::endl;
            std::cout << "Max Temperature: " << max_temp << " C" << std::endl;
            std::cout << "Min Power: " << min_power << " mW" << std::endl;
            std::cout << "Min Clock: " << min_clock << " MHz" << std::endl;
        }
    }

    void extractAvgPowerClockTemp(const std::vector<nvmlStats> &samples,
                                  uint &avg_power, uint &avg_clock, uint &avg_temp,
                                  int min_util=10, bool ignore_first_power_sample=true, bool print=false) {
        uint num_samples = 0;
        avg_power = 0;
        avg_clock = 0;
        avg_temp = 0;
        for (int i = 0; i < static_cast<int>(samples.size()); i++) {
            // Even with NVML_FI_DEV_POWER_INSTANT, we still only get 1 sample every 100ms...
            if (ignore_first_power_sample) {
                if (avg_power == 0) {
                    avg_power = samples[i].powerUsage;
                    continue;
                } else if (samples[i].powerUsage == avg_power) {
                    continue;
                } else {
                    avg_power = 0;
                    ignore_first_power_sample = false;
                }
            }
            if (samples[i].utilization.gpu >= min_util) {
                avg_power += samples[i].powerUsage;
                avg_clock += samples[i].clockSM;
                avg_temp += samples[i].temperature;
                num_samples++;
            }
        }
        if (num_samples > 0) {
            avg_power /= num_samples;
            avg_clock /= num_samples;
            avg_temp /= num_samples;
        }
        if (print) {
            std::cout << "Avg Power: " << avg_power << " mW" << std::endl;
            std::cout << "Avg Clock: " << avg_clock << " MHz" << std::endl;
            std::cout << "Avg Temperature: " << avg_temp << " C" << std::endl;
        }
    }

    void getPeakPowerClockTemp(uint &max_power, uint &max_clock, uint &max_temp, uint &min_power, uint &min_clock, bool clear_data=true) {
        std::vector<nvmlStats> samples;
        getSamples(samples, clear_data);
        extractPeakPowerClockTemp(samples, max_power, max_clock, max_temp, min_power, min_clock);
    }

    void sampleStatsLoop() {
        nvmlStats device_stats {};
        while ( outer_loop_ ) {
            while ( loop_ ) {
                inside_loop_ = true;
                device_stats.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

                auto field_value_struct = nvmlFieldValue_t();
                field_value_struct.fieldId = NVML_FI_DEV_POWER_INSTANT;
                field_value_struct.value.uiVal = 0;
                nvmlDeviceGetFieldValues(device_, 1, &field_value_struct);
                device_stats.powerUsage = field_value_struct.value.uiVal;

                //NVML_RT_CALL( nvmlDeviceGetPowerUsage( device_, &device_stats.powerUsage ) );
                NVML_RT_CALL( nvmlDeviceGetTemperature( device_, NVML_TEMPERATURE_GPU, &device_stats.temperature ) );
                NVML_RT_CALL( nvmlDeviceGetUtilizationRates( device_, &device_stats.utilization ) );
                NVML_RT_CALL( nvmlDeviceGetMemoryInfo( device_, &device_stats.memory ) );
                NVML_RT_CALL( nvmlDeviceGetClock(device_, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &device_stats.clockSM ) );
                NVML_RT_CALL( nvmlDeviceGetClock( device_, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &device_stats.clockMemory ) );
                //NVML_RT_CALL( nvmlDeviceGetFanSpeed( device_, &device_stats.fanSpeed ) );

                time_steps_.push_back( device_stats );
                std::this_thread::sleep_for( std::chrono::milliseconds(1) );
            }
            inside_loop_ = false;
            std::this_thread::sleep_for( std::chrono::milliseconds(1) );
        }
    }

  private:
    std::vector<std::string> names_ = { "timestamp",
                                        "temperature_gpu",
                                        "power_draw_w",
                                        "utilization_gpu",
                                        "memory_used_mib",
                                        "clocks_gpu_mhz",
                                        "clocks_memory_mhz" };

    std::vector<nvmlStats> time_steps_;
    std::string        filename_;
    std::ofstream      outfile_;
    nvmlDevice_t       device_;

    bool               thread_alive_;
    bool               write_csv_;
    bool               forced_fan_;
    volatile bool      loop_;
    volatile bool      outer_loop_;
    volatile bool      inside_loop_;

    std::thread        statsThread_;

    void printHeader() {
        // Print header
        for (int i = 0; i < (static_cast<int>(names_.size()) - 1); i++ )
            outfile_ << names_[i] << ", ";
        // Leave off the last comma
        outfile_ << names_[static_cast<int>(names_.size()) - 1];
        outfile_ << "\n";
    }

    void writeData(bool close_file = true, bool clear_data = true) {
        if (!outfile_.is_open()) {
            return;
        }

        // Print data
        for ( int i = 0; i < static_cast<int>( time_steps_.size( ) ); i++ ) {
            outfile_ << time_steps_[i].timestamp << ", " << time_steps_[i].temperature << ", "
                     << (float)time_steps_[i].powerUsage / 1000.0f << ", "  // mW to W
                     << time_steps_[i].utilization.gpu << ", "
                     << time_steps_[i].memory.used / 1000000 << ", "  // B to MB
                     << time_steps_[i].clockSM << ", " << time_steps_[i].clockMemory << "\n";
        }
        if (close_file) {
            outfile_.close();
        }
        if (clear_data) {
            time_steps_.clear();
        }
    }

    bool pauseLoop() {
        bool current_loop = loop_;
        loop_ = false;
        while ( inside_loop_ ) {
            std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) ); // TODO: wait less than 1ms?
        }
        return current_loop;
    }

    void resumeLoop(bool value=true) {
        loop_ = value;
    }
};

#endif /* NVMLCLASS_H_ */
