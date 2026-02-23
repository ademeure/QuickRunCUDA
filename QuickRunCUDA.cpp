/**
 * QuickRunCUDA - A fast iteration tool for CUDA kernel microbenchmarking
 *
 * Compiles CUDA kernels via NVRTC (source -> PTX -> CUBIN), executes them
 * via the Driver API, and measures performance with configurable timing.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>

#include <omp.h>
#include <cuda.h>
#include <nvrtc.h>

#include "utils/cuda_helper.h"
#include "utils/CLI11.hpp"

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static CUdevice   g_device;
static CUcontext  g_context;
static CUdeviceptr g_flush_buf = 0;

// ---------------------------------------------------------------------------
// CmdLineArgs
// ---------------------------------------------------------------------------
struct CmdLineArgs {
	// Array sizes (in 32-bit dwords)
	size_t arrayDwordsA = 64 * 1024 * 1024;
	size_t arrayDwordsB = 64 * 1024 * 1024;
	size_t arrayDwordsC = 64 * 1024 * 1024;

	// Array initialization
	bool randomArrayA = false;
	bool randomArrayB = false;
	uint32_t randomArraysBitMask = 0xFFFFFFFF;
	uint32_t randomSeed = 1234;

	// Kernel config
	int kernel_int_args[3] = {0};
	int threadsPerBlockX = 32;
	int numBlocksX = 1;
	bool persistentBlocks = false;
	int sharedMemoryBlockBytes = 0;
	int sharedMemoryCarveoutBytes = 0;

	// Benchmark
	int timedRuns = 0;
	float perfMultiplier = 0.0f;
	float perfSpeedOfLight = 0.0f;
	bool listIndividualTimes = false;

	// L2 flush
	enum L2FlushMode { NO_FLUSH = 0, FLUSH_AT_START = 1, FLUSH_EVERY_RUN = 2 };
	L2FlushMode l2FlushMode = NO_FLUSH;

	// Compilation
	std::string kernel_filename = "default_kernel.cu";
	std::string header;
	std::string ptx_input;
	std::string cubin_input;

	// Array I/O
	std::string dump_c_array;
	std::string dump_c_format = "raw";
	std::string load_c_array;
	std::string reference_c_filename;
	float compare_tolerance = 0.0f;

	// Positional
	std::vector<std::string> positional_args;
};

// ---------------------------------------------------------------------------
// CLI setup
// ---------------------------------------------------------------------------
void setupCommandLineParser(CLI::App& app, CmdLineArgs& args) {
	// Kernel execution
	auto exec = app.add_option_group("Kernel Execution");
	exec->add_option("-t,--threadsPerBlock", args.threadsPerBlockX, "blockDim.x");
	exec->add_option("-b,--blocksPerGrid",   args.numBlocksX,       "gridDim.x");
	exec->add_flag  ("-p,--persistentBlocks", args.persistentBlocks, "Set gridDim.x = SM count");
	exec->add_option("-s,--sharedMemoryBlockBytes",    args.sharedMemoryBlockBytes,    "Dynamic shared memory per block (bytes)");
	exec->add_option("-o,--sharedMemoryCarveoutBytes", args.sharedMemoryCarveoutBytes, "Shared memory carveout (bytes)");
	exec->add_option("--l2flush", args.l2FlushMode, "L2 flush: 0=none, 1=at start, 2=every run");

	// Performance
	auto perf = app.add_option_group("Performance Measurement");
	perf->add_option("-T,--timedRuns",       args.timedRuns,       "Number of timed kernel runs");
	perf->add_option("-P,--perfMultiplier",  args.perfMultiplier,  "Convert time to perf metric (value / seconds)");
	perf->add_option("-L,--perfSpeedOfLight", args.perfSpeedOfLight, "Theoretical peak for %% utilization");
	perf->add_flag  ("--timesPerRun",        args.listIndividualTimes, "Print individual run times");

	// Arrays
	auto arrays = app.add_option_group("Array Configuration");
	arrays->add_option("-A,--arrayDwordsA", args.arrayDwordsA, "Size of array A in DWORDs");
	arrays->add_option("-B,--arrayDwordsB", args.arrayDwordsB, "Size of array B in DWORDs");
	arrays->add_option("-C,--arrayDwordsC", args.arrayDwordsC, "Size of array C in DWORDs");
	arrays->add_flag  ("-r,--randomA",      args.randomArrayA,  "Random data in array A");
	arrays->add_flag  ("--randomB",         args.randomArrayB,  "Random data in array B");
	arrays->add_option("--randomMask", args.randomArraysBitMask, "Bit mask for random values (0x hex, 0b binary)")
		->transform([](std::string str) -> std::string {
			if (str.size() > 2 && str.substr(0,2) == "0x")
				return std::to_string(std::stoull(str.substr(2), nullptr, 16));
			if (str.size() > 2 && str.substr(0,2) == "0b")
				return std::to_string(std::stoull(str.substr(2), nullptr, 2));
			return str;
		});
	arrays->add_option("--randomSeed", args.randomSeed, "Base seed for RNG");

	// Kernel integer args
	auto kargs = app.add_option_group("Kernel Arguments");
	kargs->add_option("-0,--kernel-int-arg0", args.kernel_int_args[0], "Integer arg 0");
	kargs->add_option("-1,--kernel-int-arg1", args.kernel_int_args[1], "Integer arg 1");
	kargs->add_option("-2,--kernel-int-arg2", args.kernel_int_args[2], "Integer arg 2");

	// Compilation
	auto compile = app.add_option_group("Kernel Source and Compilation");
	compile->add_option("-f,--kernel-filename", args.kernel_filename, "Kernel .cu source file")
		->check(CLI::ExistingFile);
	compile->add_option("-H,--header", args.header, "Header string to prepend to kernel source");
	compile->add_option("--ptx-input", args.ptx_input, "Load PTX directly (skip NVRTC)")
		->check(CLI::ExistingFile);
	compile->add_option("--cubin-input", args.cubin_input, "Load CUBIN directly (skip all compilation)")
		->check(CLI::ExistingFile);

	// Array I/O
	auto io = app.add_option_group("Array I/O");
	io->add_option("--dump-c",            args.dump_c_array,          "Dump C array to file");
	io->add_option("--dump-c-format",     args.dump_c_format,         "Format: raw, int_csv, float_csv");
	io->add_option("--load-c",            args.load_c_array,          "Load C array from file");
	io->add_option("--reference-c",       args.reference_c_filename,  "Compare C array to reference file");
	io->add_option("--compare-tolerance", args.compare_tolerance,     "FP32 tolerance for comparison");

	// Positional
	app.add_option("kernel", args.positional_args, "Kernel source filename")
		->check(CLI::ExistingFile);
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

std::vector<char> readFile(const std::string& path) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) {
		fprintf(stderr, "Failed to open %s\n", path.c_str());
		exit(EXIT_FAILURE);
	}
	size_t size = f.tellg();
	std::vector<char> data(size + 1);
	f.seekg(0);
	f.read(data.data(), size);
	data[size] = '\0';
	return data;
}

void ensureDir(const char* dir) {
	struct stat st;
	if (stat(dir, &st) != 0)
		mkdir(dir, 0755);
}

std::string makeTimestampedPath(const char* dir, const char* ext, size_t charCount) {
	auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	struct tm tm;
	localtime_r(&t, &tm);
	char buf[128];
	snprintf(buf, sizeof(buf), "%s/%04d%02d%02d_%02d%02d%02d_%zu.%s",
	         dir, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
	         tm.tm_hour, tm.tm_min, tm.tm_sec, charCount, ext);
	return buf;
}

void saveFile(const char* dir, const char* ext, const void* data, size_t size, size_t charCount) {
	ensureDir(dir);
	std::string path = makeTimestampedPath(dir, ext, charCount);
	std::ofstream f(path, std::ios::binary);
	f.write(static_cast<const char*>(data), size);
	printf("%s saved: %s\n", dir, path.c_str());
}

// ---------------------------------------------------------------------------
// Compilation pipeline
// ---------------------------------------------------------------------------

std::vector<char> compileSourceToPTX(CUdevice device, const char* filename,
                                      const char* header, size_t& sourceCharCount) {
	// Read source
	std::ifstream inputFile(filename, std::ios::binary | std::ios::ate);
	if (!inputFile) {
		fprintf(stderr, "Error: unable to open %s\n", filename);
		exit(EXIT_FAILURE);
	}
	size_t inputSize = inputFile.tellg();
	size_t headerSize = header ? strlen(header) : 0;
	sourceCharCount = inputSize + headerSize + 1;

	std::vector<char> source(sourceCharCount + 1);
	if (headerSize > 0)
		memcpy(source.data(), header, headerSize);
	source[headerSize] = '\n';
	inputFile.seekg(0);
	inputFile.read(source.data() + headerSize + 1, inputSize);
	source[sourceCharCount] = '\0';

	// Detect compute capability
	int major = 0, minor = 0;
	checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
	checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

	char archBuf[32];
	snprintf(archBuf, sizeof(archBuf), "--gpu-architecture=compute_%d%d%s",
	         major, minor,
	         ((major == 9 && minor == 0) || (major == 10 && minor == 0)) ? "a" : "");

	const char* opts[] = {
		"--generate-line-info",
		"-use_fast_math",
		"--std=c++17",
		archBuf,
		"-I/usr/local/cuda/include/"
	};

	// Compile
	nvrtcProgram prog;
	checkCudaErrors(nvrtcCreateProgram(&prog, source.data(), filename, 0, nullptr, nullptr));
	nvrtcResult res = nvrtcCompileProgram(prog, 5, opts);

	// Print log on error
	size_t logSize;
	checkCudaErrors(nvrtcGetProgramLogSize(prog, &logSize));
	if (logSize > 1) {
		std::vector<char> log(logSize + 1);
		checkCudaErrors(nvrtcGetProgramLog(prog, log.data()));
		fprintf(stderr, "\n------- COMPILATION LOG -------\n%s\n------- END LOG -------\n", log.data());
	}
	checkCudaErrors(res);

	// Extract PTX
	size_t ptxSize;
	checkCudaErrors(nvrtcGetPTXSize(prog, &ptxSize));
	std::vector<char> ptx(ptxSize);
	checkCudaErrors(nvrtcGetPTX(prog, ptx.data()));
	checkCudaErrors(nvrtcDestroyProgram(&prog));
	return ptx;
}

CUmodule compilePTXtoCUBIN(const char* ptx, size_t ptxSize, size_t charCount) {
	CUlinkState linkState;
	void* cubinData;
	size_t cubinSize;

	char infoLog[4096] = {0}, errorLog[4096] = {0};
	CUjit_option options[] = {
		CU_JIT_GENERATE_LINE_INFO,
		CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER,
		CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER
	};
	void* optionValues[] = {
		(void*)(uintptr_t)1,
		(void*)(uintptr_t)sizeof(infoLog), infoLog,
		(void*)(uintptr_t)sizeof(errorLog), errorLog
	};

	checkCudaErrors(cuLinkCreate(5, options, optionValues, &linkState));
	CUresult linkRes = cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
	                                  (void*)ptx, ptxSize, "kernel.ptx",
	                                  0, nullptr, nullptr);
	if (linkRes != CUDA_SUCCESS) {
		fprintf(stderr, "PTX link error:\n%s\n", errorLog);
		exit(EXIT_FAILURE);
	}
	checkCudaErrors(cuLinkComplete(linkState, &cubinData, &cubinSize));

	if (strlen(infoLog) > 0)
		printf("JIT info: %s\n", infoLog);

	// Save CUBIN and load module before destroying linker (cubinData is invalidated)
	saveFile("CUBIN", "cubin", cubinData, cubinSize, charCount);
	CUmodule module;
	checkCudaErrors(cuModuleLoadData(&module, cubinData));
	checkCudaErrors(cuLinkDestroy(linkState));
	return module;
}

// ---------------------------------------------------------------------------
// Kernel launch helper
// ---------------------------------------------------------------------------

void launchKernel(CUfunction func, const CmdLineArgs& args, void** kernel_args) {
	checkCudaErrors(cuFuncSetAttribute(func,
		CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, args.sharedMemoryBlockBytes));
	checkCudaErrors(cuFuncSetAttribute(func,
		CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, args.sharedMemoryCarveoutBytes));
	checkCudaErrors(cuLaunchKernel(func,
		args.numBlocksX, 1, 1, args.threadsPerBlockX, 1, 1,
		args.sharedMemoryBlockBytes, nullptr, kernel_args, nullptr));
}

// ---------------------------------------------------------------------------
// L2 cache flush
// ---------------------------------------------------------------------------

void flushL2Cache() {
	constexpr size_t L2_FLUSH_SIZE = 200 * 1024 * 1024;
	if (g_flush_buf == 0)
		checkCudaErrors(cuMemAlloc(&g_flush_buf, L2_FLUSH_SIZE));
	checkCudaErrors(cuMemsetD8(g_flush_buf, 0, L2_FLUSH_SIZE));
}

// ---------------------------------------------------------------------------
// Main test function
// ---------------------------------------------------------------------------

int run_cuda_test(CmdLineArgs& args) {
	// --- Persistent blocks ---
	if (args.persistentBlocks) {
		checkCudaErrors(cuDeviceGetAttribute(&args.numBlocksX,
			CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, g_device));
		printf("Using persistent blocks (%d = 1 per SM)\n", args.numBlocksX);
		args.persistentBlocks = false;
	}

	// --- Compilation pipeline ---
	CUmodule module;
	if (!args.cubin_input.empty()) {
		auto cubin = readFile(args.cubin_input);
		checkCudaErrors(cuModuleLoadData(&module, cubin.data()));
	} else if (!args.ptx_input.empty()) {
		auto ptx = readFile(args.ptx_input);
		module = compilePTXtoCUBIN(ptx.data(), ptx.size(), ptx.size());
	} else {
		size_t sourceCharCount;
		auto ptx = compileSourceToPTX(g_device, args.kernel_filename.c_str(),
		                               args.header.c_str(), sourceCharCount);
		saveFile("PTX", "ptx", ptx.data(), ptx.size(), sourceCharCount);
		module = compilePTXtoCUBIN(ptx.data(), ptx.size(), sourceCharCount);
	}

	// --- Get kernel functions ---
	CUfunction kernel_addr;
	checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "kernel"));

	CUfunction init_addr;
	bool has_init = (cuModuleGetFunction(&init_addr, module, "init") == CUDA_SUCCESS);

	// --- Allocate GPU memory ---
	CUdeviceptr d_A, d_B, d_C;
	size_t sizeA = args.arrayDwordsA * sizeof(uint32_t);
	size_t sizeB = args.arrayDwordsB * sizeof(uint32_t);
	size_t sizeC = args.arrayDwordsC * sizeof(uint32_t);
	checkCudaErrors(cuMemAlloc(&d_A, sizeA));
	checkCudaErrors(cuMemAlloc(&d_B, sizeB));
	checkCudaErrors(cuMemAlloc(&d_C, sizeC));

	// --- Initialize C array ---
	std::vector<uint32_t> h_C(args.arrayDwordsC, 0);
	if (!args.load_c_array.empty()) {
		auto data = readFile(args.load_c_array);
		if (data.size() - 1 != sizeC) {
			fprintf(stderr, "File size mismatch for --load-c (got %zu, expected %zu)\n",
			        data.size() - 1, sizeC);
			exit(EXIT_FAILURE);
		}
		memcpy(h_C.data(), data.data(), sizeC);
		checkCudaErrors(cuMemcpyHtoD(d_C, h_C.data(), sizeC));
	} else {
		checkCudaErrors(cuMemsetD8(d_C, 0, sizeC));
	}

	// --- Initialize A/B arrays ---
	if (args.randomArrayA || args.randomArrayB) {
		std::vector<uint32_t> h_A(args.arrayDwordsA), h_B(args.arrayDwordsB);
		constexpr size_t chunk_size = 1024 * 1024;
		size_t num_chunks = (std::max(args.arrayDwordsA, args.arrayDwordsB) + chunk_size - 1) / chunk_size;
		#pragma omp parallel for schedule(static)
		for (size_t chunk = 0; chunk < num_chunks; chunk++) {
			std::mt19937_64 rng(args.randomSeed + chunk);
			std::uniform_int_distribution<uint32_t> dist;
			if (args.randomArrayA) {
				size_t end = std::min((chunk + 1) * chunk_size, args.arrayDwordsA);
				for (size_t i = chunk * chunk_size; i < end; ++i)
					h_A[i] = dist(rng) & args.randomArraysBitMask;
			}
			if (args.randomArrayB) {
				size_t end = std::min((chunk + 1) * chunk_size, args.arrayDwordsB);
				for (size_t i = chunk * chunk_size; i < end; ++i)
					h_B[i] = dist(rng) & args.randomArraysBitMask;
			}
		}
		if (args.randomArrayA) checkCudaErrors(cuMemcpyHtoD(d_A, h_A.data(), sizeA));
		if (args.randomArrayB) checkCudaErrors(cuMemcpyHtoD(d_B, h_B.data(), sizeB));
	}
	if (!args.randomArrayA) checkCudaErrors(cuMemsetD8(d_A, 0, sizeA));
	if (!args.randomArrayB) checkCudaErrors(cuMemsetD8(d_B, 0, sizeB));

	// --- Prepare kernel arguments ---
	int kernel_int_args[3] = {args.kernel_int_args[0], args.kernel_int_args[1], args.kernel_int_args[2]};
	void* kernel_args[] = {
		&d_A, &d_B, &d_C,
		&kernel_int_args[0], &kernel_int_args[1], &kernel_int_args[2]
	};

	// --- Init kernel (auto-detected) ---
	if (has_init) {
		printf("Found init() kernel, running it first\n");
		launchKernel(init_addr, args, kernel_args);
		checkCudaErrors(cuCtxSynchronize());
	}
	if (args.l2FlushMode >= CmdLineArgs::FLUSH_AT_START) flushL2Cache();

	// --- Warm-up run ---
	launchKernel(kernel_addr, args, kernel_args);
	if (args.l2FlushMode == CmdLineArgs::FLUSH_AT_START) flushL2Cache();
	checkCudaErrors(cuCtxSynchronize());

	// --- Timed runs ---
	if (args.timedRuns > 0) {
		bool individual_events = (args.l2FlushMode >= CmdLineArgs::FLUSH_EVERY_RUN
		                          || args.listIndividualTimes);

		CUevent overall_start, overall_stop;
		checkCudaErrors(cuEventCreate(&overall_start, CU_EVENT_DEFAULT));
		checkCudaErrors(cuEventCreate(&overall_stop, CU_EVENT_DEFAULT));

		std::vector<CUevent> start_events, stop_events;
		std::vector<float> run_times(args.timedRuns);
		if (individual_events) {
			start_events.resize(args.timedRuns);
			stop_events.resize(args.timedRuns);
			for (int i = 0; i < args.timedRuns; i++) {
				checkCudaErrors(cuEventCreate(&start_events[i], CU_EVENT_DEFAULT));
				checkCudaErrors(cuEventCreate(&stop_events[i], CU_EVENT_DEFAULT));
			}
		}

		// Launch all timed runs
		checkCudaErrors(cuEventRecord(overall_start, nullptr));
		for (int i = 0; i < args.timedRuns; i++) {
			if (args.l2FlushMode == CmdLineArgs::FLUSH_EVERY_RUN) flushL2Cache();
			if (individual_events) checkCudaErrors(cuEventRecord(start_events[i], nullptr));
			launchKernel(kernel_addr, args, kernel_args);
			if (individual_events) checkCudaErrors(cuEventRecord(stop_events[i], nullptr));
		}
		checkCudaErrors(cuEventRecord(overall_stop, nullptr));
		checkCudaErrors(cuEventSynchronize(overall_stop));

		// Compute times
		float total_time = 0.f;
		if (individual_events) {
			for (int i = 0; i < args.timedRuns; i++) {
				checkCudaErrors(cuEventElapsedTime(&run_times[i], start_events[i], stop_events[i]));
				total_time += run_times[i];
			}
		}
		float overall_time = 0.f;
		checkCudaErrors(cuEventElapsedTime(&overall_time, overall_start, overall_stop));

		// Print individual times
		if (args.listIndividualTimes) {
			printf("Individual runtimes: ");
			for (int i = 0; i < args.timedRuns; i++)
				printf("%.5f%s", run_times[i], i < args.timedRuns - 1 ? " / " : "\n");
		}

		// Print average
		float avg_time = (individual_events ? total_time : overall_time) / args.timedRuns;
		printf("\n%.5f ms", avg_time);
		if (individual_events)
			printf(" (%.5f ms including L2 flushes)", overall_time / args.timedRuns);

		// Print perf metric
		if (args.perfMultiplier > 0.0f) {
			float perf = args.perfMultiplier / (avg_time / 1000.f);
			printf(" ==> %.4f", perf);
			if (args.perfSpeedOfLight > 0.0f)
				printf(" ==> %.3f%%", 100.0f * perf / args.perfSpeedOfLight);
		}
		printf("\n\n");

		// Cleanup events
		checkCudaErrors(cuEventDestroy(overall_start));
		checkCudaErrors(cuEventDestroy(overall_stop));
		for (auto& e : start_events) checkCudaErrors(cuEventDestroy(e));
		for (auto& e : stop_events)  checkCudaErrors(cuEventDestroy(e));
	}

	checkCudaErrors(cuCtxSynchronize());

	// --- Array dump ---
	if (!args.dump_c_array.empty()) {
		checkCudaErrors(cuMemcpyDtoH(h_C.data(), d_C, sizeC));
		if (args.dump_c_format == "raw") {
			std::ofstream f(args.dump_c_array, std::ios::binary);
			f.write(reinterpret_cast<char*>(h_C.data()), sizeC);
		} else {
			std::ofstream f(args.dump_c_array);
			for (size_t i = 0; i < args.arrayDwordsC; i++) {
				if (h_C[i] != 0) {
					if (args.dump_c_format == "int_csv")
						f << h_C[i];
					else if (args.dump_c_format == "float_csv")
						f << std::fixed << std::setprecision(2)
						  << *reinterpret_cast<float*>(&h_C[i]);
				}
				if (i < args.arrayDwordsC - 1) f << ",";
			}
		}
	}

	// --- Reference comparison ---
	if (!args.reference_c_filename.empty()) {
		auto ref_data = readFile(args.reference_c_filename);
		if (ref_data.size() - 1 != sizeC) {
			fprintf(stderr, "Reference file wrong size (got %zu, expected %zu)\n",
			        ref_data.size() - 1, sizeC);
			exit(EXIT_FAILURE);
		}
		auto* ref_C = reinterpret_cast<uint32_t*>(ref_data.data());
		checkCudaErrors(cuMemcpyDtoH(h_C.data(), d_C, sizeC));

		for (size_t i = 0; i < args.arrayDwordsC; i++) {
			if (args.compare_tolerance > 0.0f) {
				float a = *reinterpret_cast<float*>(&h_C[i]);
				float b = *reinterpret_cast<float*>(&ref_C[i]);
				if (std::abs(a - b) > args.compare_tolerance) {
					printf("First difference at %zu: %.8f vs %.8f (diff %.8f)\n",
					       i, a, b, std::abs(a - b));
					break;
				}
			} else if (h_C[i] != ref_C[i]) {
				printf("First difference at %zu: %u vs %u (hex: %08x vs %08x, fp32: %.4f vs %.4f)\n",
				       i, h_C[i], ref_C[i], h_C[i], ref_C[i],
				       *reinterpret_cast<float*>(&h_C[i]), *reinterpret_cast<float*>(&ref_C[i]));
				break;
			}
		}
	}

	// --- Cleanup ---
	checkCudaErrors(cuMemFree(d_A));
	checkCudaErrors(cuMemFree(d_B));
	checkCudaErrors(cuMemFree(d_C));
	if (g_flush_buf) { checkCudaErrors(cuMemFree(g_flush_buf)); g_flush_buf = 0; }
	checkCudaErrors(cuModuleUnload(module));
	return 0;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
	CLI::App app{"QuickRunCUDA: Fast iteration for CUDA microbenchmarking"};
	app.set_help_all_flag("--help-all", "Show all help options");
	CmdLineArgs args;
	setupCommandLineParser(app, args);

	try {
		CLI11_PARSE(app, argc, argv);
		if (!args.positional_args.empty())
			args.kernel_filename = args.positional_args[0];

		checkCudaErrors(cuInit(0));
		checkCudaErrors(cuDeviceGet(&g_device, 0));
		checkCudaErrors(cuCtxCreate(&g_context, nullptr, 0, g_device));

		return run_cuda_test(args);
	} catch (const CLI::ParseError& e) {
		return app.exit(e);
	}
}
