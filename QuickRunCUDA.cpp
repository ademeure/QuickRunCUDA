/**
 * QuickRunCUDA - A fast iteration tool for CUDA kernel microbenchmarking
 *
 * This utility allows for rapid testing and benchmarking of CUDA kernels with minimal
 * overhead, making it ideal for quick performance measurements and experiments.
 *
 * Very, very loosely based on a NVIDIA sample (Ship of Theseus style).
 */

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <unistd.h>
#include <random>
#include <assert.h>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cctype>
#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "utils/cuda_helper.h"
#include "utils/ipc_helper.h"
#include "utils/nvmlClass.h"
#include "utils/CLI11.hpp"

// Global CUDA variables
CUdevice cuDeviceGlobal;
CUcontext cuContextGlobal;
CUdeviceptr d_flush = 0;  // L2 flush buffer

/**
 * Structure to hold all command line arguments
 */
struct CmdLineArgs {
	// Array sizes
	size_t arrayDwordsA = 64 * 1024 * 1024;  // Size of array A in DWORDs (32 bits)
	size_t arrayDwordsB = 64 * 1024 * 1024;  // Size of array B in DWORDs (32 bits)
	size_t arrayDwordsC = 64 * 1024 * 1024;  // Size of array C in DWORDs (32 bits)

	// Kernel configuration
	bool randomArrayA = false;               // Use random data in array A
	bool randomArrayB = false;               // Use random data in array B
	uint randomArraysBitMask = 0xFFFFFFFF;   // Bit mask (useful for power analysis)
	uint randomSeed = 1234;                  // Base seed for random number generation
	int kernel_int_args[3] = {0};            // Integer arguments to pass to kernel
	int threadsPerBlockX = 32;               // Number of threads per block
	int numBlocksX = 1;                      // Number of blocks
	bool persistentBlocks = false;           // Automatically set gridDim.x to number of SMs
	int sharedMemoryBlockBytes = 0;          // Shared memory size per block
	int sharedMemoryCarveoutBytes = 0;       // Shared memory carveout
	bool runInitKernel = false;              // Run initialization kernel

	// Benchmark settings
	int timedRuns = 0;                     // Number of timed runs
	float perfMultiplier = 0.0f;             // Performance multiplier
	float perfMultiplierPerThread = 0.0f;    // Per-thread performance multiplier
	float perfSpeedOfLight = 0.0f;           // Speed of light (in perf metric units)

	std::string perfMultiplier_unit = "ops/s"; // Unit for performance metric

	// Compilation and kernel settings
	std::string kernel_filename = "default_kernel.cu"; // Kernel source file
	std::string header = "";                 // Header to prepend to kernel
	bool reuse_cubin = false;                // Reuse existing compiled cubin

	// Operational modes
	bool server_mode = false;                // Run in server mode
	int clock_speed = 0;                     // GPU clock speed to set

	// Data I/O options
	std::string dump_c_array = "";           // File to dump C array to
	std::string dump_c_format = "raw";       // Format for C array dump
	std::string load_c_array = "";           // File to load C array from
	std::string reference_c_filename = "";   // Reference file to compare C array against
	float compare_tolerance = 0.0f;          // Floating point comparison tolerance

	// Positional arguments
	std::vector<std::string> positional_args; // For kernel filename as positional argument

	// Add to CmdLineArgs struct
	enum L2FlushMode {
		NO_FLUSH = 0,
		FLUSH_AT_START = 1,
		FLUSH_EVERY_RUN = 2
	} l2FlushMode = NO_FLUSH;

	bool listIndividualTimes = false;        // List individual run times
};

/**
 * Set up command line argument parser
 * @param app CLI11 app object
 * @param args Command line arguments struct to populate
 */
void setupCommandLineParser(CLI::App& app, CmdLineArgs& args) {
	// Operational modes
	auto modes_group = app.add_option_group("Operational Modes");
	modes_group->add_flag("--server", args.server_mode, "Run in server mode for subprocess control (advanced)");
	modes_group->add_option("--clock-speed", args.clock_speed, "GPU clock in MHz (0=no force, 1=unlocked) (does not reset after run)");

	// Kernel execution configuration
	auto kernel_exec_group = app.add_option_group("Kernel Execution Configuration");
	kernel_exec_group->add_option("-t,--threadsPerBlock", args.threadsPerBlockX, "Number of threads per block (blockDim.x)");
	kernel_exec_group->add_option("-b,--blocksPerGrid", args.numBlocksX, "Number of blocks (gridDim.x)");
	kernel_exec_group->add_flag("-p,--persistentBlocks", args.persistentBlocks, "Automatically set gridDim.x to number of SMs");
	kernel_exec_group->add_option("-s,--sharedMemoryBlockBytes", args.sharedMemoryBlockBytes, "Shared memory size per block in bytes");
	kernel_exec_group->add_option("-o,--sharedMemoryCarveoutBytes", args.sharedMemoryCarveoutBytes, "Shared memory carveout in bytes");
	kernel_exec_group->add_flag("-i,--runInitKernel", args.runInitKernel, "Run initialization kernel before main kernel");
	kernel_exec_group->add_option("--l2flush", args.l2FlushMode, "L2 flush mode: 0=none, 1=at start, 2=every run");

	// Performance measurement
	auto perf_group = app.add_option_group("Performance Measurement");
	perf_group->add_option("-T,--timedRuns", args.timedRuns, "Number of timed kernel executions");
	perf_group->add_option("-P,--perfMultiplier", args.perfMultiplier, "Performance multiplier to convert time to ops/s");
	perf_group->add_option("-N,--perfMultiplierPerThread", args.perfMultiplierPerThread, "Performance multiplier per thread");
	perf_group->add_option("-U,--perfMultiplier-unit", args.perfMultiplier_unit, "Performance multiplier unit (ops/s, ms, us, ns)");
	perf_group->add_option("-L,--perfSpeedOfLight", args.perfSpeedOfLight, "Speed of light (e.g. 2000 for GB/s on H100 PCIe)");
	perf_group->add_flag("--timesPerRun", args.listIndividualTimes, "List individual run times");

	// Array configuration
	auto array_group = app.add_option_group("Array Configuration");
	array_group->add_option("-A,--arrayDwordsA", args.arrayDwordsA, "Size of array A in DWORDs");
	array_group->add_option("-B,--arrayDwordsB", args.arrayDwordsB, "Size of array B in DWORDs");
	array_group->add_option("-C,--arrayDwordsC", args.arrayDwordsC, "Size of array C in DWORDs");
	array_group->add_flag("-r,--randomA", args.randomArrayA, "Initialize array A with random data");
	array_group->add_flag("--randomB", args.randomArrayB, "Initialize array B with random data");
	array_group->add_option("--randomMask", args.randomArraysBitMask, "Bit mask for random values (0x for hex, 0b for binary)")
		->transform([](std::string str) -> std::string {
			return str.substr(0,2) == "0x" ? std::to_string(std::stoull(str.substr(2), nullptr, 16)) :
				   str.substr(0,2) == "0b" ? std::to_string(std::stoull(str.substr(2), nullptr, 2)) :
				   str; });
	array_group->add_option("--randomSeed", args.randomSeed, "Base seed for random number generation");

	// Kernel arguments
	auto kernel_args_group = app.add_option_group("Kernel Arguments");
	kernel_args_group->add_option("-0,--kernel-int-arg0", args.kernel_int_args[0], "Kernel integer argument 0");
	kernel_args_group->add_option("-1,--kernel-int-arg1", args.kernel_int_args[1], "Kernel integer argument 1");
	kernel_args_group->add_option("-2,--kernel-int-arg2", args.kernel_int_args[2], "Kernel integer argument 2");

	// Kernel source and compilation
	auto kernel_source_group = app.add_option_group("Kernel Source and Compilation");
	kernel_source_group->add_option("-f,--kernel-filename", args.kernel_filename, "Kernel source filename (same as positional)")
		->check(CLI::ExistingFile);
	kernel_source_group->add_option("-H,--header", args.header, "Header string to prepend to kernel source");
	modes_group->add_flag("--reuse-cubin", args.reuse_cubin, "Reuse compiled cubin in output.cubin instead of recompiling");

	// Array I/O
	auto array_io_group = app.add_option_group("Array I/O");
	array_io_group->add_option("--dump-c", args.dump_c_array, "Dump C array to specified file");
	array_io_group->add_option("--dump-c-format", args.dump_c_format, "Format for C array dump (raw, int_csv, float_csv)");
	array_io_group->add_option("--load-c", args.load_c_array, "Load C array from specified file");
	array_io_group->add_option("--reference-c", args.reference_c_filename, "Compare C array to reference file (raw format)");
	array_io_group->add_option("--compare-tolerance", args.compare_tolerance, "Floating point tolerance for C vs reference C");

	// Add positional argument for kernel filename
	app.add_option("kernel", args.positional_args, "Kernel source filename")
		->check(CLI::ExistingFile);
}

/**
 * Parse a command string into CmdLineArgs (for server mode)
 * @param cmd Command string to parse
 * @return Populated CmdLineArgs structure
 */
CmdLineArgs parseCommandString(const std::string& cmd) {
    // Parse command into argc/argv
    std::vector<std::string> args_vec{"QuickRunCUDA"};
    std::istringstream iss(cmd);
    std::string current_arg;
    bool in_quotes = false;

    iss >> std::noskipws;
    char c;
    while (iss.get(c)) {
        if (c == '\'') {
            if (!in_quotes) {
                in_quotes = true;
                current_arg = "";
            } else {
                in_quotes = false;
                args_vec.push_back(current_arg);
            }
            continue;
        }

        if (in_quotes) {
            current_arg += c;
        } else if (!std::isspace(c)) {
            current_arg = c;
            while (iss.get(c) && !std::isspace(c)) {
                current_arg += c;
            }
            args_vec.push_back(current_arg);
        }
    }

    // Convert string vector to char* vector for CLI11
    std::vector<char*> argv_vec;
    for (const auto& s : args_vec) {
        argv_vec.push_back(const_cast<char*>(s.c_str()));
    }

    // Parse command line args
    CLI::App cmd_app{"QuickRunCUDA: Super fast iteration for CUDA microbenchmarking"};
    CmdLineArgs cmd_args;
    setupCommandLineParser(cmd_app, cmd_args);
    cmd_app.parse(args_vec.size(), argv_vec.data());

    // If positional arg provided, use it as kernel filename
    if (!cmd_args.positional_args.empty()) {
        cmd_args.kernel_filename = cmd_args.positional_args[0];
    }

    return cmd_args;
}

void flushL2Cache() {
    constexpr size_t L2_FLUSH_SIZE = 200 * 1024 * 1024;
    if (d_flush == 0) checkCudaErrors(cuMemAlloc(&d_flush, L2_FLUSH_SIZE));
    checkCudaErrors(cuMemsetD8(d_flush, 0, L2_FLUSH_SIZE));
}

/**
 * Run the CUDA test using the provided command line arguments
 * @param args Command line arguments struct
 * @return Exit code (0 for success)
 */
int run_cuda_test(CmdLineArgs& args);

/**
 * Main entry point
 */
int main(int argc, char **argv) {
	// Command line arguments
	CLI::App app{"QuickRunCUDA: Super fast iteration for CUDA microbenchmarking"};
	app.set_help_all_flag("--help-all", "Show all help options");
	CmdLineArgs args;
	setupCommandLineParser(app, args);

	try {
		// Parse command line
		CLI11_PARSE(app, argc, argv);

		// If positional arg provided, use it as kernel filename
		if (!args.positional_args.empty()) args.kernel_filename = args.positional_args[0];

		// Initialize CUDA (unfortunately a bit slow which is another reason why server mode is useful)
		checkCudaErrors(cuInit(0));
		checkCudaErrors(cuDeviceGet(&cuDeviceGlobal, 0));
		checkCudaErrors(cuCtxCreate(&cuContextGlobal, 0, cuDeviceGlobal));

		// Set GPU clock speed if requested
		if (args.clock_speed > 0) {
			nvmlClass nvml(0, args.clock_speed, false, false, false);
		}

		// Run in server mode (loop via IPC) or normal mode (single run based on provided arguments)
		if (!args.server_mode) {
			// Run the test directly
			int result = run_cuda_test(args);
		} else {
			// Server mode - loop waiting for new commands via IPC
			IPCHelper ipc;
			while (true) {
				std::string cmd;
				if (ipc.waitForCommand(cmd)) {
					if (cmd == "exit") {
						break;
					}
					// Redirect stdout for capturing output
					fflush(stdout);
					int stdout_fd = dup(STDOUT_FILENO);
					int pipe_fd[2];
					pipe(pipe_fd);
					dup2(pipe_fd[1], STDOUT_FILENO);
					close(pipe_fd[1]);
					// Parse command string into CmdLineArgs
					CmdLineArgs cmd_args = parseCommandString(cmd);

					// Run the test (!!!)
					int result = run_cuda_test(cmd_args);

					// Restore stdout and get captured output
					fflush(stdout);
					dup2(stdout_fd, STDOUT_FILENO);
					close(stdout_fd);
					// Read captured output
					std::stringstream buffer;
					char buf[4096];
					ssize_t n;
					while ((n = read(pipe_fd[0], buf, sizeof(buf)-1)) > 0) {
						buf[n] = '\0';
						buffer << buf;
					}
					close(pipe_fd[0]);
					// Send captured output
					ipc.sendResponse(buffer.str());
				}
			}

			// Write server mode's exit status to file
			std::ofstream file("returning.txt");
			file << "returning 0" << "\n";
			file.close();
			return 0;
		}
	} catch (const CLI::ParseError &e) {
		return app.exit(e);
	}
}

/**
 * Run the CUDA test using the provided command line arguments
 * @param args Command line arguments struct
 * @return Exit code (0 for success)
 */
int run_cuda_test(CmdLineArgs& args) {
	// Allocate memory
	CUdeviceptr d_A, d_B, d_C;
	size_t sizeA = args.arrayDwordsA * sizeof(uint);
	size_t sizeB = args.arrayDwordsB * sizeof(uint);
	size_t sizeC = args.arrayDwordsC * sizeof(uint);
	checkCudaErrors(cuMemAlloc(&d_A, sizeA));
	checkCudaErrors(cuMemAlloc(&d_B, sizeB));
	checkCudaErrors(cuMemAlloc(&d_C, sizeC));
	uint *h_C = reinterpret_cast<uint *>(malloc(sizeC));

	if (args.persistentBlocks) {
		checkCudaErrors(cuDeviceGetAttribute(&args.numBlocksX, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDeviceGlobal));
		printf("Using persistent blocks (%d = 1 per SM)\n", args.numBlocksX);
		args.persistentBlocks = false;
	}

	// Compile or load kernel
	char *cubin;
	CudaHelper CUDA(cuDeviceGlobal);
	CUfunction kernel_addr, init_addr;
	size_t cubin_size;
	CUmodule module;

	if (args.reuse_cubin) {
		// Read the cubin from the binary file
		std::ifstream cubin_file("output.cubin", std::ios::binary | std::ios::ate);
		if (cubin_file.is_open()) {
			cubin_size = cubin_file.tellg();
			cubin = new char[cubin_size];
			cubin_file.seekg(0, std::ios::beg);
			cubin_file.read(cubin, cubin_size);
			cubin_file.close();
		} else {
			fprintf(stderr, "Failed to open output.cubin for reading!\n");
			exit(EXIT_FAILURE);
		}
	} else {
		// Compile the kernel to CUBIN (!!!)
		CUDA.compileFileToCUBIN(cuDeviceGlobal, &cubin, args.kernel_filename.c_str(), args.header.c_str(), &cubin_size);

		// Write the cubin to a binary file for potential reuse (and disassembly)
		std::ofstream cubin_file("output.cubin", std::ios::binary);
		if (cubin_file.is_open()) {
			cubin_file.write(cubin, cubin_size);
			cubin_file.close();
		} else {
			fprintf(stderr, "Failed to open file to write cubin!\n");
			exit(EXIT_FAILURE);
		}
	}

	// Load the module and get function pointers
	module = CUDA.loadCUBIN(cubin, cuContextGlobal, cuDeviceGlobal);
	checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "kernel"));
	if (args.runInitKernel) {
		checkCudaErrors(cuModuleGetFunction(&init_addr, module, "init"));
	}

	// Load C array from file if specified, otherwise initialize with zeros
	if (!args.load_c_array.empty()) {
		std::ifstream infile(args.load_c_array, std::ios::binary);
		if (!infile) {
			fprintf(stderr, "Failed to open C array input file: %s\n", args.load_c_array.c_str());
			exit(EXIT_FAILURE);
		}
		infile.read(reinterpret_cast<char*>(h_C), sizeC);
		if (infile.gcount() != sizeC) {
			fprintf(stderr, "Input file size (%ld) does not match expected C array size (%ld)\n",
					infile.gcount(), sizeC);
			exit(EXIT_FAILURE);
		}
		checkCudaErrors(cuMemcpyHtoD(d_C, h_C, sizeC));
	} else {
		checkCudaErrors(cuMemsetD8(d_C, 0, sizeC));
	}

	// Initialize A/B arrays
	if (args.randomArrayA || args.randomArrayB) {
		uint *h_A = reinterpret_cast<uint *>(malloc(sizeA));
		uint *h_B = reinterpret_cast<uint *>(malloc(sizeB));
		const int chunk_size = 1024 * 1024;
		const int num_chunks = (std::max(args.arrayDwordsA, args.arrayDwordsB) + chunk_size - 1) / chunk_size;
		#pragma omp parallel
		{
			#pragma omp for schedule(static)
			for (size_t chunk = 0; chunk < num_chunks; chunk++) {
				std::mt19937_64 rng(args.randomSeed + chunk);
				std::uniform_int_distribution<uint> dist;
				size_t end = args.randomArrayA ? std::min((chunk + 1) * chunk_size, args.arrayDwordsA) : 0;
				for (size_t i = chunk * chunk_size; i < end; ++i) {
					h_A[i] = dist(rng) & args.randomArraysBitMask;
				}
				end = args.randomArrayB ? std::min((chunk + 1) * chunk_size, args.arrayDwordsB) : 0;
				for (size_t i = chunk * chunk_size; i < end; ++i) {
					h_B[i] = dist(rng) & args.randomArraysBitMask;
				}
			}
		}
		if (args.randomArrayA) checkCudaErrors(cuMemcpyHtoD(d_A, h_A, sizeA));
		if (args.randomArrayB) checkCudaErrors(cuMemcpyHtoD(d_B, h_B, sizeB));
		free(h_A);
		free(h_B);
	}
	if (!args.randomArrayA) {
		checkCudaErrors(cuMemsetD8(d_A, 0, sizeA));
	}
	if (!args.randomArrayB) {
		checkCudaErrors(cuMemsetD8(d_B, 0, sizeB));
	}

	// Prepare kernel arguments
	int kernel_int_args[3] = {args.kernel_int_args[0], args.kernel_int_args[1], args.kernel_int_args[2]};
	void *kernel_args[] = {
		reinterpret_cast<void *>(&d_A),
		reinterpret_cast<void *>(&d_B),
		reinterpret_cast<void *>(&d_C),
		reinterpret_cast<void *>(&kernel_int_args[0]),
		reinterpret_cast<void *>(&kernel_int_args[1]),
		reinterpret_cast<void *>(&kernel_int_args[2])
	};

	// Launch the init kernel if requested
	if (args.runInitKernel) {
		checkCudaErrors(cuFuncSetAttribute(init_addr, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
										   args.sharedMemoryCarveoutBytes));
		checkCudaErrors(cuFuncSetAttribute(init_addr, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
										  args.sharedMemoryBlockBytes));
		checkCudaErrors(cuLaunchKernel(init_addr,
									 args.numBlocksX, 1, 1,        /* grid dim */
									 args.threadsPerBlockX, 1, 1,  /* block dim */
									 args.sharedMemoryBlockBytes, 0, /* shared mem, stream */
									 kernel_args, 0));
	}
	if (args.l2FlushMode >= CmdLineArgs::FLUSH_AT_START) flushL2Cache();

	// Configure and launch the main kernel
	checkCudaErrors(cuFuncSetAttribute(kernel_addr, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
									  args.sharedMemoryCarveoutBytes));
	checkCudaErrors(cuFuncSetAttribute(kernel_addr, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
									  args.sharedMemoryBlockBytes));
	checkCudaErrors(cuLaunchKernel(kernel_addr,
								  args.numBlocksX, 1, 1,        /* grid dim */
								  args.threadsPerBlockX, 1, 1,  /* block dim */
								  args.sharedMemoryBlockBytes, 0, /* shared mem, stream */
								  kernel_args, 0));
	// Wait for kernel to complete
	if (args.l2FlushMode == CmdLineArgs::FLUSH_AT_START) flushL2Cache();
	checkCudaErrors(cuCtxSynchronize());

	// Perform timed runs if requested
	if (args.timedRuns > 0) {
		// Individual events have a slight overhead so avoid them unless we flush the L2 or list individual times
		bool individual_events = (args.l2FlushMode >= CmdLineArgs::FLUSH_EVERY_RUN || args.listIndividualTimes);
		// Create overall timing events to include L2 flushes
		CUevent overall_start, overall_stop;
		checkCudaErrors(cuEventCreate(&overall_start, CU_EVENT_DEFAULT));
		checkCudaErrors(cuEventCreate(&overall_stop, CU_EVENT_DEFAULT));

		// Create arrays of start/stop events and initialize them
		CUevent* start_events = new CUevent[args.timedRuns];
		CUevent* stop_events = new CUevent[args.timedRuns];
		float* run_times = new float[args.timedRuns];
		for (int i = 0; i < (individual_events ? args.timedRuns : 0); i++) {
			checkCudaErrors(cuEventCreate(&start_events[i], CU_EVENT_DEFAULT));
			checkCudaErrors(cuEventCreate(&stop_events[i], CU_EVENT_DEFAULT));
		}

		// Run the kernel N times, recording events without synchronizing
		checkCudaErrors(cuEventRecord(overall_start, nullptr));
		for (int i = 0; i < args.timedRuns; i++) {
			if (args.l2FlushMode == CmdLineArgs::FLUSH_EVERY_RUN) flushL2Cache();

			if (individual_events) { checkCudaErrors(cuEventRecord(start_events[i], nullptr)); }
			checkCudaErrors(cuLaunchKernel(kernel_addr,
										 args.numBlocksX, 1, 1,
										 args.threadsPerBlockX, 1, 1,
										 args.sharedMemoryBlockBytes, 0,
										 kernel_args, 0));
			if (individual_events) { checkCudaErrors(cuEventRecord(stop_events[i], nullptr)); }
		}
		checkCudaErrors(cuEventRecord(overall_stop, nullptr));
		checkCudaErrors(cuEventSynchronize(overall_stop));

		// Calculate elapsed time for each run
		float total_time = 0.f;
		for (int i = 0; i < (individual_events ? args.timedRuns : 0); i++) {
			checkCudaErrors(cuEventElapsedTime(&run_times[i], start_events[i], stop_events[i]));
			total_time += run_times[i];
		}

		// Calculate overall time including L2 flushes
		float overall_time = 0.f;
		checkCudaErrors(cuEventElapsedTime(&overall_time, overall_start, overall_stop));

		// Print individual times when requested
		if (args.listIndividualTimes) {
			printf("Individual runtimes: ");
			for (int i = 0; i < args.timedRuns; i++) {
				printf("%.5f %s", run_times[i], i < args.timedRuns - 1 ? " / " : "\n");
			}
		}

		// Print average time per run
		float avg_time = (individual_events ? total_time : overall_time) / args.timedRuns;
		printf("\n%.5f ms", avg_time);

		// Print overall time including L2 flushes for comparison
		if (individual_events) {
			printf(" (%.5f ms including L2 flushes)", overall_time / args.timedRuns);
		}

		// Print performance metric if requested
		float multiplier = args.perfMultiplier;
		if (args.perfMultiplierPerThread > 0.0f) {
			multiplier = args.perfMultiplierPerThread * args.threadsPerBlockX * args.numBlocksX;
		}
		if (multiplier > 0.0f) {
			float perf = multiplier / (avg_time / 1000.f);
			printf(" ==> %.4f %s", perf, args.perfMultiplier_unit.c_str());
			if (args.perfSpeedOfLight > 0.0f) printf(" ==> %.3f%%", 100.0f * perf / args.perfSpeedOfLight);
		}
		printf("\n\n");

		// Clean up events
		checkCudaErrors(cuEventDestroy(overall_start));
		checkCudaErrors(cuEventDestroy(overall_stop));
		for (int i = 0; i < (individual_events ? args.timedRuns : 0); i++) {
			checkCudaErrors(cuEventDestroy(start_events[i]));
			checkCudaErrors(cuEventDestroy(stop_events[i]));
		}
		delete[] start_events;
		delete[] stop_events;
		delete[] run_times;
	}

	// Ensure all kernels have completed
	checkCudaErrors(cuCtxSynchronize());

	// Dump C array if requested
	if (!args.dump_c_array.empty()) {
		checkCudaErrors(cuMemcpyDtoH(h_C, d_C, sizeC));

		if (args.dump_c_format == "raw") {
			// Dump raw binary data
			std::ofstream outfile(args.dump_c_array, std::ios::binary);
			outfile.write(reinterpret_cast<char*>(h_C), sizeC);
		} else {
			// Dump formatted data (CSV int or float)
			std::ofstream outfile(args.dump_c_array);
			for (size_t i = 0; i < args.arrayDwordsC; i++) {
				if (h_C[i] != 0) {
					if (args.dump_c_format == "int_csv") {
						outfile << h_C[i];
					} else if (args.dump_c_format == "float_csv") {
						outfile << std::fixed << std::setprecision(2) << *reinterpret_cast<float*>(&h_C[i]);
					}
				}
				if (i < args.arrayDwordsC - 1) outfile << ",";
			}
		}
	}

	// Optionally compare with reference file
	if (!args.reference_c_filename.empty()) {
		std::ifstream ref_file(args.reference_c_filename, std::ios::binary | std::ios::ate);
		if (!ref_file || ref_file.tellg() != sizeC) {
			fprintf(stderr, "Reference file missing or wrong size\n");
			exit(EXIT_FAILURE);
		}
		ref_file.seekg(0);

		uint *ref_C = reinterpret_cast<uint *>(malloc(sizeC));
		ref_file.read(reinterpret_cast<char*>(ref_C), sizeC);
		checkCudaErrors(cuMemcpyDtoH(h_C, d_C, sizeC));

		for (size_t i = 0; i < args.arrayDwordsC; i++) {
			if (args.compare_tolerance > 0.0f) {
				float diff = std::abs(*reinterpret_cast<float*>(&h_C[i]) - *reinterpret_cast<float*>(&ref_C[i]));
				if (diff > args.compare_tolerance) {
					printf("First difference at index %zu: %.8f vs %.8f (diff %.8f)\n",
						   i, *reinterpret_cast<float*>(&h_C[i]),
						   *reinterpret_cast<float*>(&ref_C[i]), diff);
					break;
				}
			} else if (h_C[i] != ref_C[i]) {
				printf("First difference at index %zu: %u vs %u\nHex: %08x vs %08x\nFP32: %.4f vs %.4f\n",
					   i, h_C[i], ref_C[i], h_C[i], ref_C[i],
					   *reinterpret_cast<float*>(&h_C[i]), *reinterpret_cast<float*>(&ref_C[i]));
				break;
			}
		}
		free(ref_C);
	}

	// Clean up resources
	free(h_C);
	checkCudaErrors(cuMemFree(d_A));
	checkCudaErrors(cuMemFree(d_B));
	checkCudaErrors(cuMemFree(d_C));
	if (d_flush) { checkCudaErrors(cuMemFree(d_flush)); d_flush = 0; }
	checkCudaErrors(cuModuleUnload(module));

	return 0;
}