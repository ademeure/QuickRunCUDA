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

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

////////////////////////////////////////////////////////////////////////////////

char default_filename[] =  "kernel.cu";

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions
// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char *errorStr = NULL;
    cuGetErrorString(err, &errorStr);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}
#endif

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device) {
  checkCudaErrors(cuDeviceGetAttribute(attribute, device_attribute, device));
}

#define NVRTC_SAFE_CALL(Name, x)                                \
  do {                                                          \
    nvrtcResult result = x;                                     \
    if (result != NVRTC_SUCCESS) {                              \
      std::cerr << "\nerror: " << Name << " failed with error " \
                << nvrtcGetErrorString(result) << "\n";         \
      exit(1);                                                  \
    }                                                           \
  } while (0)

void compileFileToCUBIN(char *filename, char **cubinResult, char* header=NULL, size_t *cubinResultSize=NULL, int requiresCGheaders=0) {
  if (!filename) {
    std::cerr << "\nerror: filename is empty for compileFileToCUBIN()!\n";
    exit(1);
  }
  std::ifstream inputFile(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if (!inputFile.is_open()) {
    std::cerr << "\nerror: unable to open " << filename << " for reading!\n";
    exit(1);
  }

  std::streampos pos = inputFile.tellg();
  size_t inputSize = (size_t)pos;
  size_t headerSize = header ? strlen(header) : 0;
  size_t totalSize = inputSize + headerSize + 1;
  char *memBlock = new char[inputSize + headerSize + 2];

  // Copy header
  memcpy(memBlock, header, headerSize);
  memBlock[headerSize] = '\n';

  // Read file
  inputFile.seekg(0, std::ios::beg);
  inputFile.read(&memBlock[headerSize+1], inputSize);
  inputFile.close();
  memBlock[totalSize] = '\x0';

  int numCompileOptions = 0;
  char *compileParams[2];
  int major = 0, minor = 0;

  // Picks the 1st CUDA device
  CUdevice cuDevice;
  checkCudaErrors(cuDeviceGet(&cuDevice, 0));

  // get compute capabilities
  checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  
  // Compile cubin for the GPU arch on which are going to run cuda kernel.
  // HACK: Turn sm_90 into sm_90a
  std::string compileOptions;
  compileOptions = "--gpu-architecture=sm_";
  compileParams[numCompileOptions] = reinterpret_cast<char *>(malloc(sizeof(char) * (compileOptions.length() + 11)));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 10),
            "%s%d%d%s", compileOptions.c_str(), major, minor, (major == 9 && minor == 0) ? "a" : "");
#else
  snprintf(compileParams[numCompileOptions], compileOptions.size() + 10, "%s%d%d%s",
           compileOptions.c_str(), major, minor, (major == 9 && minor == 0) ? "a" : "");
#endif
  numCompileOptions++;

  /*
  if (requiresCGheaders) {
    std::string compileOptions;
    char HeaderNames[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(HeaderNames, sizeof(HeaderNames), "%s", "cooperative_groups.h");
#else
    snprintf(HeaderNames, sizeof(HeaderNames), "%s", "cooperative_groups.h");
#endif

    compileOptions = "--include-path=";

    char *strPath = sdkFindFilePath(HeaderNames, argv[0]);
    if (!strPath) {
      std::cerr << "\nerror: header file " << HeaderNames << " not found!\n";
      exit(1);
    }
    std::string path = strPath;
    if (!path.empty()) {
      std::size_t found = path.find(HeaderNames);
      path.erase(found);
    } else {
      printf(
          "\nCooperativeGroups headers not found, please install it in %s "
          "sample directory..\n Exiting..\n",
          argv[0]);
      exit(1);
    }
    compileOptions += path.c_str();
    compileParams[numCompileOptions] = reinterpret_cast<char *>(
        malloc(sizeof(char) * (compileOptions.length() + 1)));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 1),
              "%s", compileOptions.c_str());
#else
    snprintf(compileParams[numCompileOptions], compileOptions.size(), "%s",
             compileOptions.c_str());
#endif
    numCompileOptions++;
  }
  */

  // compile
  nvrtcProgram prog;
  NVRTC_SAFE_CALL("nvrtcCreateProgram",  nvrtcCreateProgram(&prog, memBlock, filename, 0, NULL, NULL));
  nvrtcResult res = nvrtcCompileProgram(prog, numCompileOptions, compileParams);

  // dump log
  size_t logSize;
  NVRTC_SAFE_CALL("nvrtcGetProgramLogSize", nvrtcGetProgramLogSize(prog, &logSize));
  char *log = reinterpret_cast<char *>(malloc(sizeof(char) * logSize + 1));
  NVRTC_SAFE_CALL("nvrtcGetProgramLog", nvrtcGetProgramLog(prog, log));
  log[logSize] = '\x0';

  if (strlen(log) >= 2) {
    std::cerr << "\n------- ERROR DURING KERNEL COMPILATION -------\n\n";
    std::cerr << log;
    std::cerr << "\n------- END LOG -------\n";
  }
  free(log);

  NVRTC_SAFE_CALL("nvrtcCompileProgram", res);

  size_t codeSize;
  NVRTC_SAFE_CALL("nvrtcGetCUBINSize", nvrtcGetCUBINSize(prog, &codeSize));
  char *code = new char[codeSize];
  NVRTC_SAFE_CALL("nvrtcGetCUBIN", nvrtcGetCUBIN(prog, code));
  *cubinResult = code;
  if (cubinResultSize) {
    *cubinResultSize = codeSize;
  }

  for (int i = 0; i < numCompileOptions; i++) {
    free(compileParams[i]);
  }
}

CUmodule loadCUBIN(char *cubin) {
  CUmodule module;
  CUcontext context;
  CUdevice cuDevice;
  cuDeviceGet(&cuDevice, 0);

  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuCtxCreate(&context, 0, cuDevice));
  checkCudaErrors(cuModuleLoadData(&module, cubin));
  free(cubin);

  return module;
}























/**
 * Host main routine
 */
int main(int argc, char **argv) {
  char *cubin;
  char *kernel_filename = default_filename;

  size_t arrayInputDwords = 1024*1024;
  size_t arrayOutputDwords = 1024*1024;
  int randomArrays = 0;
  int kernel_int_args[3] = {0};
  int blockDimX = 32;
  int numBlocks = 1;
  int sharedMemoryBlockBytes = 0;
  int sharedMemoryCarveoutBytes = 0;
  int runInitKernel = 0;
  int timedRuns = 0;
  char *header = NULL;

  // Using getopt was probably a mistake with this many options... oh well
  char c;
	static char usage[] = "usage: %s -a arrayInputDwords -o arrayOutputDwords -r randomArraysBool -b blockDimX -n numBlocksX -s sharedMemoryBlockBytes -c sharedMemoryCarveoutBytes ((((-c GPUClockMHz -m MemoryClockMHz -M metricRuns -T timedRuns)))) -i runInitKernelBool -H [header_string] -f [kernel_filename]\n";
  while ((c = getopt (argc, argv, "h0:1:2:a:r:b:n:s:c:i:H:f:T:")) != -1) {
    switch (c) {
      case 'h':
        printf(usage, argv[0]);
        exit(0);
        break;
      case '0':
      case '1':
      case '2':
        kernel_int_args[c-'0'] = atoi(optarg);
        break;
      case 'a':
        arrayInputDwords = strtoull(optarg, NULL, 0);
        break;
      case 'o':
        arrayOutputDwords = strtoull(optarg, NULL, 0);
        break;
      case 'r':
        randomArrays = atoi(optarg);
        break;
      case 'b':
        blockDimX = atoi(optarg);
        break;
      case 'n':
        numBlocks = atoi(optarg);
        break;
      case 's':
        sharedMemoryBlockBytes = atoi(optarg);
        break;
      case 'c':
        sharedMemoryCarveoutBytes = atoi(optarg);
        break;
      case 'i':
        runInitKernel = atoi(optarg);
        break;
      case 'f':
        kernel_filename = optarg;
        break;
      case 'H':
        header = optarg;
        break;
      case 'T':
        timedRuns = atoi(optarg);
        break;
      case '?':
        if (optopt == 'c')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
        return 1;
      default:
        abort();
      }
  }

  CUdevice cuDevice;
  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuDeviceGet(&cuDevice, 0));

  CUfunction kernel_addr, init_addr;
  compileFileToCUBIN(kernel_filename, &cubin, header);
  CUmodule module = loadCUBIN(cubin);
  checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "kernel"));
  if (runInitKernel) {
    checkCudaErrors(cuModuleGetFunction(&init_addr, module, "init"));
  }





  size_t inputSize = arrayInputDwords * sizeof(uint);
  size_t outputSize = arrayOutputDwords * sizeof(uint);

  // Allocate the host input vectors
  uint *h_A = reinterpret_cast<uint *>(malloc(inputSize));
  uint *h_O = reinterpret_cast<uint *>(malloc(outputSize));
  if (h_A == NULL || h_O == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors either randomly or to zero
  if (randomArrays) {
    std::mt19937_64 rng(1234ULL);
    std::uniform_int_distribution<uint> dist;
    for (int i = 0; i < arrayInputDwords; ++i) {
      h_A[i] = dist(rng);
    }
  } else {
    memset(h_A, 0, inputSize);
  }

  // Allocate the device vectors & copy inputs from host
  CUdeviceptr d_A, d_O;
  checkCudaErrors(cuMemAlloc(&d_A, inputSize));
  checkCudaErrors(cuMemAlloc(&d_O, outputSize));
  checkCudaErrors(cuMemcpyHtoD(d_A, h_A, inputSize));

  // Prepare arguments to kernel function
  void *arr[] = {reinterpret_cast<void *>(&d_A),
                 reinterpret_cast<void *>(&d_O),
                 reinterpret_cast<void *>(&kernel_int_args[0]),
                 reinterpret_cast<void *>(&kernel_int_args[1]),
                 reinterpret_cast<void *>(&kernel_int_args[2])};

  // Launch the init kernel if there is one
  if (runInitKernel) {
    checkCudaErrors(cuFuncSetAttribute(init_addr, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, sharedMemoryCarveoutBytes));
    checkCudaErrors(cuFuncSetAttribute(init_addr, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sharedMemoryBlockBytes));
    checkCudaErrors(cuLaunchKernel(init_addr,
                                  numBlocks, 1, 1, /* grid dim */
                                  blockDimX, 1, 1, /* block dim */
                                  sharedMemoryBlockBytes, 0, /* shared mem, stream */
                                  &arr[0],         /* arguments */
                                  0));
  }

  // Launch the main CUDA Kernel
  checkCudaErrors(cuFuncSetAttribute(kernel_addr, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, sharedMemoryCarveoutBytes));
  checkCudaErrors(cuFuncSetAttribute(kernel_addr, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sharedMemoryBlockBytes));
  checkCudaErrors(cuLaunchKernel(kernel_addr,
                                 numBlocks, 1, 1, /* grid dim */
                                 blockDimX, 1, 1, /* block dim */
                                 sharedMemoryBlockBytes, 0, /* shared mem, stream */
                                 &arr[0],         /* arguments */
                                 0));
  checkCudaErrors(cuCtxSynchronize());

  for(int i = 0; i < timedRuns; i++) {
    checkCudaErrors(cuLaunchKernel(kernel_addr,
                                  numBlocks, 1, 1, /* grid dim */
                                  blockDimX, 1, 1, /* block dim */
                                  sharedMemoryBlockBytes, 0, /* shared mem, stream */
                                  &arr[0],         /* arguments */
                                  0));
  }

  // Copy the device result vector in device memory to the host?
  //checkCudaErrors(cuMemcpyDtoH(h_O, d_O, size));
  // .....................

  checkCudaErrors(cuMemFree(d_A));
  checkCudaErrors(cuMemFree(d_O));
  free(h_A);
  free(h_O);
  return 0;
}
