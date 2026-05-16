// Thin wrapper around NVRTC + a generic checkCudaErrors() macro.
// Used only by QuickRunCUDA.cpp at the moment.
//
// Public API:
//   checkCudaErrors(err)       — error-check macro (cudaError_t / CUresult / nvrtcResult)
//   compileFileToCUBIN(...)    — read .cu, prepend header, NVRTC-compile, return cubin
//   loadCUBIN(cubin, ...)      — cuModuleLoadData + delete[] cubin
//
// The cubin returned by compileFileToCUBIN is allocated with `new char[]`.
// loadCUBIN frees it with `delete[]`; if you don't pass it to loadCUBIN, you
// must `delete[]` it yourself.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

template <typename T>
inline void __checkCudaErrors(T err, const char *file, const int line) {
  if (err != 0) {
    const char *errorStr = "";
    if constexpr (std::is_same<T, cudaError_t>::value) {
      errorStr = cudaGetErrorString(err);
    } else if constexpr (std::is_same<T, CUresult>::value) {
      cuGetErrorString(err, &errorStr);
    } else if constexpr (std::is_same<T, nvrtcResult>::value) {
      errorStr = nvrtcGetErrorString(err);
    }
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
            (int)err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}
#endif

// Compile `filename` (with `header` prepended) to CUBIN for `cuDevice`'s SM arch.
// SM >= 90 gets the 'a' suffix (sm_90a, sm_100a, sm_103a) for arch-specific instructions.
// Sets `*cubinResult` to a `new char[]`-allocated buffer; caller must `delete[]` it
// (or hand it to loadCUBIN, which does).
//
// Include path: `cudaIncludePath` arg, else the CUDA_INCLUDE_PATH env var,
// else "/usr/local/cuda/include/".
inline void compileFileToCUBIN(
    CUdevice cuDevice,
    char **cubinResult,
    char const *filename,
    char const *header = nullptr,
    size_t *cubinResultSize = nullptr,
    char const *cudaIncludePath = nullptr)
{
  if (!filename || !*filename) {
    std::cerr << "compileFileToCUBIN: filename is empty\n";
    std::exit(EXIT_FAILURE);
  }
  std::ifstream inputFile(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if (!inputFile.is_open()) {
    std::cerr << "compileFileToCUBIN: cannot open " << filename << "\n";
    std::exit(EXIT_FAILURE);
  }
  size_t inputSize = (size_t)inputFile.tellg();
  size_t headerSize = header ? std::strlen(header) : 0;

  // Build source: <header>\n<file contents>\0
  std::string source(headerSize + 1 + inputSize, '\0');
  if (headerSize) std::memcpy(source.data(), header, headerSize);
  source[headerSize] = '\n';
  inputFile.seekg(0, std::ios::beg);
  inputFile.read(source.data() + headerSize + 1, inputSize);
  inputFile.close();

  int major = 0, minor = 0;
  checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  const char *archSuffix = (major >= 9) ? "a" : "";

  if (!cudaIncludePath) {
    if (const char *env = std::getenv("CUDA_INCLUDE_PATH")) cudaIncludePath = env;
    else cudaIncludePath = "/usr/local/cuda/include/";
  }

  std::string archOpt = "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor) + archSuffix;
  std::string includeOpt = std::string("-I") + cudaIncludePath;

  const char *const opts[] = {
      "--generate-line-info",
      "-use_fast_math",
      "--std=c++17",
      archOpt.c_str(),
      includeOpt.c_str(),
  };
  constexpr int numOpts = sizeof(opts) / sizeof(opts[0]);

  nvrtcProgram prog = nullptr;
  checkCudaErrors(nvrtcCreateProgram(&prog, source.c_str(), filename, 0, nullptr, nullptr));
  nvrtcResult res = nvrtcCompileProgram(prog, numOpts, opts);

  size_t logSize = 0;
  checkCudaErrors(nvrtcGetProgramLogSize(prog, &logSize));
  if (logSize > 1) {
    std::string log(logSize, '\0');
    checkCudaErrors(nvrtcGetProgramLog(prog, log.data()));
    std::cerr << "\n------- ERROR DURING KERNEL COMPILATION -------\n\n"
              << log
              << "\n------- END LOG -------\n";
  }
  checkCudaErrors(res);

  size_t codeSize = 0;
  checkCudaErrors(nvrtcGetCUBINSize(prog, &codeSize));
  char *code = new char[codeSize];
  checkCudaErrors(nvrtcGetCUBIN(prog, code));
  *cubinResult = code;
  if (cubinResultSize) *cubinResultSize = codeSize;

  nvrtcDestroyProgram(&prog);
}

// Load `cubin` (a `new char[]`-allocated buffer) into a module and `delete[]` it.
inline CUmodule loadCUBIN(char *cubin) {
  CUmodule module;
  checkCudaErrors(cuModuleLoadData(&module, cubin));
  delete[] cubin;
  return module;
}
