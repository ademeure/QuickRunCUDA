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
	std::vector<size_t> arrayDwords = {64*1024*1024, 64*1024*1024};
	std::vector<size_t> randomArrays;
	uint32_t randomArraysBitMask = 0xFFFFFFFF;
	uint32_t randomSeed = 1234;

	std::vector<int> intArgs;
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

	std::vector<std::string> dump_array;
	std::string dump_format = "raw";
	std::vector<std::string> load_array;
	std::string reference_filename;
	float compare_atol = 0.0f;
	float compare_rtol = 0.0f;

	std::pair<size_t, std::string> parseIndexedArg(const std::string& s) const {
		auto c = s.find(':');
		if (c > 0 && c != std::string::npos && s.substr(0, c).find_first_not_of("0123456789") == std::string::npos) {
			size_t i = std::stoull(s.substr(0, c));
			if (i >= arrayDwords.size()) { fprintf(stderr, "Array index %zu out of range\n", i); exit(1); }
			return {i, s.substr(c + 1)};
		}
		return {arrayDwords.size() - 1, s};
	}

	std::vector<std::string> positional_args;
};

void setupCLI(CLI::App& app, CmdLineArgs& args) {
	app.add_option("-t,--threadsPerBlock", args.threadsPerBlockX, "blockDim.x");
	app.add_option("-b,--blocksPerGrid", args.numBlocksX, "gridDim.x");
	app.add_flag("-p,--persistentBlocks", args.persistentBlocks, "Set gridDim.x = SM count");
	app.add_option("-s,--sharedMemoryBlockBytes", args.sharedMemoryBlockBytes, "Dynamic shared mem bytes");
	app.add_option("-o,--sharedMemoryCarveoutBytes", args.sharedMemoryCarveoutBytes, "Shared mem carveout bytes");
	app.add_option("--l2flush", args.l2FlushMode, "L2 flush: 0=none, 1=at start, 2=every run");
	app.add_option("-T,--timedRuns", args.timedRuns, "Number of timed kernel runs");
	app.add_option("-P,--perfMultiplier", args.perfMultiplier, "Perf metric = value / seconds");
	app.add_option("-L,--perfSpeedOfLight", args.perfSpeedOfLight, "Theoretical peak for %% utilization");
	app.add_option("-D,--dwords", args.arrayDwords, "Array sizes in DWORDs (comma-separated)")->delimiter(',');
	app.add_option("-r,--random", args.randomArrays, "Which arrays to randomize (indices)")->delimiter(',');
	app.add_option("--randomMask", args.randomArraysBitMask, "Bit mask for random values (supports 0x, 0b)")
		->transform([](std::string s) -> std::string {
			if (s.substr(0, 2) == "0b") return std::to_string(std::stoull(s.substr(2), nullptr, 2));
			return std::to_string(std::stoull(s, nullptr, 0));
		});
	app.add_option("--randomSeed", args.randomSeed, "Base seed for RNG");
	app.add_option("-I,--int-args", args.intArgs, "Int args after array pointers")->delimiter(',');
	app.add_option("-F,--float-args", args.floatArgs, "Float args after int args")->delimiter(',');
	app.add_option("-f,--kernel-filename", args.kernel_filename, "Kernel .cu source file")->check(CLI::ExistingFile);
	app.add_option("-H,--header", args.header, "Header string prepended to kernel source");
	app.add_option("--ptx-input", args.ptx_input, "Load PTX directly (skip NVRTC)")->check(CLI::ExistingFile);
	app.add_option("--cubin-input", args.cubin_input, "Load CUBIN directly (skip compilation)")->check(CLI::ExistingFile);
	app.add_option("--dump", args.dump_array, "Dump array to file ([idx:]file, repeatable)");
	app.add_option("--dump-format", args.dump_format, "Format: raw, int_csv, float_csv");
	app.add_option("--load", args.load_array, "Load array from file ([idx:]file, repeatable)");
	app.add_option("--reference", args.reference_filename, "Compare array to reference ([idx:]file)");
	app.add_option("--atol", args.compare_atol, "Absolute tolerance for --reference");
	app.add_option("--rtol", args.compare_rtol, "Relative tolerance (fallback if atol fails)");
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
	for (auto& la : args.load_array) {
		auto [idx, file] = args.parseIndexedArg(la);
		arrays.load(idx, file);
	}

	static int zero_pad[8] = {};
	std::vector<void*> kernel_args;
	for (auto& p : arrays.d)       kernel_args.push_back(&p);
	for (auto& v : args.intArgs)   kernel_args.push_back(&v);
	for (auto& v : args.floatArgs) kernel_args.push_back(&v);
	for (int i = 0; i < 8; i++)    kernel_args.push_back(&zero_pad[i]);

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

	for (auto& da : args.dump_array) {
		auto [idx, file] = args.parseIndexedArg(da);
		arrays.dump(idx, file, args.dump_format);
	}
	if (!args.reference_filename.empty()) {
		auto [idx, file] = args.parseIndexedArg(args.reference_filename);
		arrays.compare(idx, file, args.compare_atol, args.compare_rtol);
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
