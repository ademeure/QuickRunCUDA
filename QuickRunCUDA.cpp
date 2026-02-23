// QuickRunCUDA: Fast CUDA kernel microbenchmarking via NVRTC + Driver API

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda.h>
#include "utils/cuda_helper.h"
#include "utils/cuda_compile.h"
#include "utils/cuda_arrays.h"
#include "utils/CLI11.hpp"

static CUdevice   g_device;
static CUcontext  g_context;
static CUdeviceptr g_flush_buf = 0;

struct CmdLineArgs {
	std::vector<size_t> arrayDwords = {64*1024*1024, 64*1024*1024, 64*1024*1024};
	std::vector<size_t> randomArrays;
	uint32_t randomArraysBitMask = 0xFFFFFFFF;
	uint32_t randomSeed = 1234;

	std::vector<int> intArgs = {0, 0, 0};
	std::vector<float> floatArgs;

	int threadsPerBlockX = 32;
	int numBlocksX = 1;
	bool persistentBlocks = false;
	int sharedMemoryBlockBytes = 0;
	int sharedMemoryCarveoutBytes = 0;

	int timedRuns = 0;
	float perfMultiplier = 0.0f;
	float perfSpeedOfLight = 0.0f;

	enum L2FlushMode { NO_FLUSH = 0, FLUSH_AT_START = 1, FLUSH_EVERY_RUN = 2 };
	L2FlushMode l2FlushMode = NO_FLUSH;

	std::string kernel_filename = "default_kernel.cu";
	std::string header;
	std::string ptx_input;
	std::string cubin_input;

	std::string dump_array;
	std::string dump_format = "raw";
	std::string load_array;
	std::string reference_filename;
	float compare_tolerance = 0.0f;

	std::pair<size_t, std::string> parseIndexedArg(const std::string& str) const {
		auto colon = str.find(':');
		if (colon != std::string::npos && colon > 0 &&
		    str.substr(0, colon).find_first_not_of("0123456789") == std::string::npos) {
			size_t idx = std::stoull(str.substr(0, colon));
			if (idx >= arrayDwords.size()) {
				fprintf(stderr, "Array index %zu out of range (%zu arrays)\n", idx, arrayDwords.size());
				exit(EXIT_FAILURE);
			}
			return {idx, str.substr(colon + 1)};
		}
		return {arrayDwords.size() - 1, str};
	}

	std::vector<std::string> positional_args;
};

void setupCLI(CLI::App& app, CmdLineArgs& args) {
	auto* exec = app.add_option_group("Kernel Execution");
	exec->add_option("-t,--threadsPerBlock", args.threadsPerBlockX, "blockDim.x");
	exec->add_option("-b,--blocksPerGrid", args.numBlocksX, "gridDim.x");
	exec->add_flag("-p,--persistentBlocks", args.persistentBlocks, "Set gridDim.x = SM count");
	exec->add_option("-s,--sharedMemoryBlockBytes", args.sharedMemoryBlockBytes, "Dynamic shared memory per block (bytes)");
	exec->add_option("-o,--sharedMemoryCarveoutBytes", args.sharedMemoryCarveoutBytes, "Shared memory carveout (bytes)");
	exec->add_option("--l2flush", args.l2FlushMode, "L2 flush: 0=none, 1=at start, 2=every run");

	auto* perf = app.add_option_group("Performance Measurement");
	perf->add_option("-T,--timedRuns", args.timedRuns, "Number of timed kernel runs");
	perf->add_option("-P,--perfMultiplier", args.perfMultiplier, "Convert time to perf metric (value / seconds)");
	perf->add_option("-L,--perfSpeedOfLight", args.perfSpeedOfLight, "Theoretical peak for %% utilization");


	auto* arr = app.add_option_group("Array Configuration");
	arr->add_option("-D,--dwords", args.arrayDwords, "Array sizes in DWORDs (comma-separated)")->delimiter(',');
	arr->add_option("-r,--random", args.randomArrays, "Which arrays to randomize (comma-separated indices)")->delimiter(',');
	arr->add_option("--randomMask", args.randomArraysBitMask, "Bit mask for random values (0x hex, 0b binary)")
		->transform([](std::string s) -> std::string {
			if (s.size() > 2 && s[0] == '0' && s[1] == 'x') return std::to_string(std::stoull(s.substr(2), nullptr, 16));
			if (s.size() > 2 && s[0] == '0' && s[1] == 'b') return std::to_string(std::stoull(s.substr(2), nullptr, 2));
			return s;
		});
	arr->add_option("--randomSeed", args.randomSeed, "Base seed for RNG");

	auto* kargs = app.add_option_group("Kernel Arguments");
	kargs->add_option("-I,--int-args", args.intArgs, "Int args after array pointers (comma-separated)")->delimiter(',');
	kargs->add_option("-F,--float-args", args.floatArgs, "Float args after int args (comma-separated)")->delimiter(',');

	auto* compile = app.add_option_group("Kernel Source and Compilation");
	compile->add_option("-f,--kernel-filename", args.kernel_filename, "Kernel .cu source file")->check(CLI::ExistingFile);
	compile->add_option("-H,--header", args.header, "Header string to prepend to kernel source");
	compile->add_option("--ptx-input", args.ptx_input, "Load PTX directly (skip NVRTC)")->check(CLI::ExistingFile);
	compile->add_option("--cubin-input", args.cubin_input, "Load CUBIN directly (skip all compilation)")->check(CLI::ExistingFile);

	auto* io = app.add_option_group("Array I/O");
	io->add_option("--dump", args.dump_array, "Dump array to file ([index:]filename)");
	io->add_option("--dump-format", args.dump_format, "Format: raw, int_csv, float_csv");
	io->add_option("--load", args.load_array, "Load array from file ([index:]filename)");
	io->add_option("--reference", args.reference_filename, "Compare array to reference ([index:]filename)");
	io->add_option("--compare-tolerance", args.compare_tolerance, "FP32 tolerance for comparison");

	app.add_option("kernel", args.positional_args, "Kernel source filename")->check(CLI::ExistingFile);
}

void launchKernel(CUfunction func, const CmdLineArgs& args, void** kernel_args) {
	checkCudaErrors(cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, args.sharedMemoryBlockBytes));
	checkCudaErrors(cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, args.sharedMemoryCarveoutBytes));
	checkCudaErrors(cuLaunchKernel(func,
		args.numBlocksX, 1, 1, args.threadsPerBlockX, 1, 1,
		args.sharedMemoryBlockBytes, nullptr, kernel_args, nullptr));
}

void flushL2Cache() {
	constexpr size_t L2_FLUSH_SIZE = 200 * 1024 * 1024;
	if (g_flush_buf == 0)
		checkCudaErrors(cuMemAlloc(&g_flush_buf, L2_FLUSH_SIZE));
	checkCudaErrors(cuMemsetD8(g_flush_buf, 0, L2_FLUSH_SIZE));
}

int run_cuda_test(CmdLineArgs& args) {
	if (args.persistentBlocks) {
		checkCudaErrors(cuDeviceGetAttribute(&args.numBlocksX,
			CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, g_device));
		printf("Using persistent blocks (%d = 1 per SM)\n", args.numBlocksX);
		args.persistentBlocks = false;
	}

	CUmodule module = loadModule(g_device, args.kernel_filename,
	                              args.header, args.ptx_input, args.cubin_input);
	CUfunction kernel_addr;
	checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "kernel"));
	CUfunction init_addr;
	bool has_init = (cuModuleGetFunction(&init_addr, module, "init") == CUDA_SUCCESS);

	GPUArrays arrays;
	arrays.allocate(args.arrayDwords);
	arrays.initRandom(args.randomArrays, args.randomSeed, args.randomArraysBitMask);
	if (!args.load_array.empty()) {
		auto [idx, file] = args.parseIndexedArg(args.load_array);
		arrays.load(idx, file);
	}

	std::vector<void*> kernel_args;
	for (auto& p : arrays.d)       kernel_args.push_back(&p);
	for (auto& v : args.intArgs)   kernel_args.push_back(&v);
	for (auto& v : args.floatArgs) kernel_args.push_back(&v);

	if (has_init) {
		printf("Found init() kernel, running it first\n");
		launchKernel(init_addr, args, kernel_args.data());
		checkCudaErrors(cuCtxSynchronize());
	}
	if (args.l2FlushMode >= CmdLineArgs::FLUSH_AT_START) flushL2Cache();

	launchKernel(kernel_addr, args, kernel_args.data());
	if (args.l2FlushMode == CmdLineArgs::FLUSH_AT_START) flushL2Cache();
	checkCudaErrors(cuCtxSynchronize());

	if (args.timedRuns > 0) {
		CUevent t0, t1;
		checkCudaErrors(cuEventCreate(&t0, CU_EVENT_DEFAULT));
		checkCudaErrors(cuEventCreate(&t1, CU_EVENT_DEFAULT));

		checkCudaErrors(cuEventRecord(t0, nullptr));
		for (int i = 0; i < args.timedRuns; i++) {
			if (args.l2FlushMode == CmdLineArgs::FLUSH_EVERY_RUN) flushL2Cache();
			launchKernel(kernel_addr, args, kernel_args.data());
		}
		checkCudaErrors(cuEventRecord(t1, nullptr));
		checkCudaErrors(cuEventSynchronize(t1));

		float elapsed = 0.f;
		checkCudaErrors(cuEventElapsedTime(&elapsed, t0, t1));
		float avg_time = elapsed / args.timedRuns;
		printf("\n%.5f ms", avg_time);
		if (args.perfMultiplier > 0.0f) {
			float perf = args.perfMultiplier / (avg_time / 1000.f);
			printf(" ==> %.4f", perf);
			if (args.perfSpeedOfLight > 0.0f)
				printf(" ==> %.3f%%", 100.0f * perf / args.perfSpeedOfLight);
		}
		printf("\n\n");

		checkCudaErrors(cuEventDestroy(t0));
		checkCudaErrors(cuEventDestroy(t1));
	}

	checkCudaErrors(cuCtxSynchronize());

	if (!args.dump_array.empty()) {
		auto [idx, file] = args.parseIndexedArg(args.dump_array);
		arrays.dump(idx, file, args.dump_format);
	}
	if (!args.reference_filename.empty()) {
		auto [idx, file] = args.parseIndexedArg(args.reference_filename);
		arrays.compare(idx, file, args.compare_tolerance);
	}

	arrays.free();
	if (g_flush_buf) { checkCudaErrors(cuMemFree(g_flush_buf)); g_flush_buf = 0; }
	checkCudaErrors(cuModuleUnload(module));
	return 0;
}

int main(int argc, char** argv) {
	CLI::App app{"QuickRunCUDA: Fast iteration for CUDA microbenchmarking"};
	app.set_help_all_flag("--help-all", "Show all help options");
	CmdLineArgs args;
	setupCLI(app, args);
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
