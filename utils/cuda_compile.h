#pragma once

#include "cuda_helper.h"
#include <nvrtc.h>
#include <cuda.h>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <sys/stat.h>

inline std::vector<char> readFile(const std::string& path) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) { fprintf(stderr, "Failed to open %s\n", path.c_str()); exit(EXIT_FAILURE); }
	size_t size = f.tellg();
	std::vector<char> data(size + 1);
	f.seekg(0);
	f.read(data.data(), size);
	data[size] = '\0';
	return data;
}

inline void saveFile(const char* dir, const char* filename, const void* data, size_t size) {
	struct stat st;
	if (stat(dir, &st) != 0) mkdir(dir, 0755);
	std::string path = std::string(dir) + "/" + filename;
	std::ofstream f(path, std::ios::binary);
	f.write(static_cast<const char*>(data), size);
	printf("Saved %s (%zu bytes)\n", path.c_str(), size);
}

inline std::vector<char> compileSourceToPTX(CUdevice device, const char* filename, const char* header) {
	auto file = readFile(filename);
	std::string source;
	if (header && *header) source = std::string(header) + "\n";
	source.append(file.data());

	int major = 0, minor = 0;
	checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
	checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
	char archBuf[32];
	snprintf(archBuf, sizeof(archBuf), "--gpu-architecture=compute_%d%d%s",
	         major, minor, ((major == 9 && minor == 0) || (major == 10 && minor == 0)) ? "a" : "");

	const char* opts[] = { "--generate-line-info", "-use_fast_math", "--std=c++17", archBuf, "-I/usr/local/cuda/include/" };

	nvrtcProgram prog;
	checkCudaErrors(nvrtcCreateProgram(&prog, source.c_str(), filename, 0, nullptr, nullptr));
	nvrtcResult res = nvrtcCompileProgram(prog, 5, opts);

	size_t logSize;
	checkCudaErrors(nvrtcGetProgramLogSize(prog, &logSize));
	if (logSize > 1) {
		std::vector<char> log(logSize + 1);
		checkCudaErrors(nvrtcGetProgramLog(prog, log.data()));
		fprintf(stderr, "\n------- COMPILATION LOG -------\n%s\n------- END LOG -------\n", log.data());
	}
	checkCudaErrors(res);

	size_t ptxSize;
	checkCudaErrors(nvrtcGetPTXSize(prog, &ptxSize));
	std::vector<char> ptx(ptxSize);
	checkCudaErrors(nvrtcGetPTX(prog, ptx.data()));
	checkCudaErrors(nvrtcDestroyProgram(&prog));
	return ptx;
}

inline CUmodule compilePTXtoCUBIN(const char* ptx, size_t ptxSize) {
	char errorLog[4096] = {0};
	CUjit_option options[] = { CU_JIT_GENERATE_LINE_INFO, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER };
	void* values[] = { (void*)(uintptr_t)1, (void*)(uintptr_t)sizeof(errorLog), errorLog };

	CUlinkState linkState;
	checkCudaErrors(cuLinkCreate(3, options, values, &linkState));
	CUresult linkRes = cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
	                                  (void*)ptx, ptxSize, "kernel.ptx", 0, nullptr, nullptr);
	if (linkRes != CUDA_SUCCESS) { fprintf(stderr, "PTX link error:\n%s\n", errorLog); exit(EXIT_FAILURE); }

	void* cubinData; size_t cubinSize;
	checkCudaErrors(cuLinkComplete(linkState, &cubinData, &cubinSize));
	saveFile("CUBIN", "kernel.cubin", cubinData, cubinSize);

	CUmodule module;
	checkCudaErrors(cuModuleLoadData(&module, cubinData));
	checkCudaErrors(cuLinkDestroy(linkState));
	return module;
}

inline CUmodule loadModule(CUdevice device, const std::string& filename,
                           const std::string& header, const std::string& ptx_input,
                           const std::string& cubin_input) {
	if (!cubin_input.empty()) {
		auto cubin = readFile(cubin_input);
		CUmodule m; checkCudaErrors(cuModuleLoadData(&m, cubin.data())); return m;
	}
	if (!ptx_input.empty()) {
		auto ptx = readFile(ptx_input);
		return compilePTXtoCUBIN(ptx.data(), ptx.size());
	}
	auto ptx = compileSourceToPTX(device, filename.c_str(), header.c_str());
	saveFile("PTX", "kernel.ptx", ptx.data(), ptx.size());
	return compilePTXtoCUBIN(ptx.data(), ptx.size());
}
