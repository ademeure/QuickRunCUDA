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

// Core NVRTC compile: source string → PTX or CUBIN bytes
// When cubin=true, returns CUBIN (requires sm_ arch in opts). Otherwise returns PTX.
inline std::vector<char> nvrtcCompile(const char* source, const char* name,
                                       int nOpts = 0, const char** opts = nullptr,
                                       bool cubin = false) {
	nvrtcProgram prog;
	checkCudaErrors(nvrtcCreateProgram(&prog, source, name, 0, nullptr, nullptr));
	nvrtcResult res = nvrtcCompileProgram(prog, nOpts, opts);

	size_t logSize;
	checkCudaErrors(nvrtcGetProgramLogSize(prog, &logSize));
	if (logSize > 1) {
		std::vector<char> log(logSize + 1);
		checkCudaErrors(nvrtcGetProgramLog(prog, log.data()));
		fprintf(stderr, "\n------- COMPILATION LOG -------\n%s------- END LOG -------\n", log.data());
	}
	checkCudaErrors(res);

	size_t sz;
	if (cubin) { checkCudaErrors(nvrtcGetCUBINSize(prog, &sz)); }
	else       { checkCudaErrors(nvrtcGetPTXSize(prog, &sz)); }
	std::vector<char> out(sz);
	if (cubin) { checkCudaErrors(nvrtcGetCUBIN(prog, out.data())); }
	else       { checkCudaErrors(nvrtcGetPTX(prog, out.data())); }
	checkCudaErrors(nvrtcDestroyProgram(&prog));
	return out;
}

inline std::string getArchFlag() {
	CUdevice dev; checkCudaErrors(cuCtxGetDevice(&dev));
	int major = 0, minor = 0;
	checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
	checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
	char buf[32];
	snprintf(buf, sizeof(buf), "--gpu-architecture=sm_%d%d%s",
	         major, minor, (major >= 9 && minor == 0) ? "a" : "");
	return buf;
}

inline CUmodule compileSource(const char* source, const char* name, bool save = true) {
	auto arch = getArchFlag();
	const char* archStr = arch.c_str();
	const char* opts[] = { "--generate-line-info", "-use_fast_math", "--std=c++17", archStr, "-I/usr/local/cuda/include/" };
	auto cubin = nvrtcCompile(source, name, 5, opts, true);
	if (save) saveFile("CUBIN", "kernel.cubin", cubin.data(), cubin.size());
	CUmodule m;
	checkCudaErrors(cuModuleLoadData(&m, cubin.data()));
	return m;
}

inline CUmodule loadModule(CUdevice device, const std::string& filename,
                           const std::string& header, const std::string& ptx_input,
                           const std::string& cubin_input, bool save = true) {
	if (!cubin_input.empty()) {
		auto cubin = readFile(cubin_input);
		CUmodule m; checkCudaErrors(cuModuleLoadData(&m, cubin.data())); return m;
	}
	if (!ptx_input.empty()) {
		auto ptx = readFile(ptx_input);
		CUmodule m; checkCudaErrors(cuModuleLoadData(&m, ptx.data())); return m;
	}
	auto file = readFile(filename);
	std::string source;
	if (!header.empty()) source = header + "\n";
	source.append(file.data());
	return compileSource(source.c_str(), filename.c_str(), save);
}
