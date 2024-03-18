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
#include "CLI11.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "cuda_helper.h"
#include "nvmlClass.h"
#include "cuda_metrics/measureMetricPW.hpp"
#include "testRunner.h"

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  char *cubin;

  CLI::App app{"QuickBenchCUDA: Super fast iteration for CUDA microbenchmarking"};

  size_t arrayInputDwords = 1024*1024;
  size_t arrayOutputDwords = 1024*1024;
  bool randomArrays = false;
  int kernel_int_args[3] = {0};
  int threadsPerBlockX = 32;
  int numBlocksX = 1;
  int sharedMemoryBlockBytes = 0;
  int sharedMemoryCarveoutBytes = 0;
  bool runInitKernel = false;
  bool timedRuns = false;
  std::string header = "";
  std::string kernel_filename = "kernel.cu";

  app.add_option("-a,--arrayInputDwords", arrayInputDwords, "Array input DWORDs");
  app.add_option("-o,--arrayOutputDwords", arrayOutputDwords, "Array output DWORDs");
  app.add_flag("-r,--randomArrays", randomArrays, "Random arrays");
  app.add_option("-t,--threadsPerBlockX", threadsPerBlockX, "Threads per block X");
  app.add_option("-b,--numBlocksX", numBlocksX, "Number of blocks X");
  app.add_option("-s,--sharedMemoryBlockBytes", sharedMemoryBlockBytes, "Shared memory block bytes");
  app.add_option("-c,--sharedMemoryCarveoutBytes", sharedMemoryCarveoutBytes, "Shared memory carveout bytes");
  app.add_flag("-i,--runInitKernel", runInitKernel, "Run init kernel");
  app.add_flag("-T,--timedRuns", timedRuns, "Timed runs");
  app.add_option("-H,--header", header, "Header string");
  app.add_option("-f,--kernel_filename", kernel_filename, "Kernel filename");
  app.add_option("--kernel-int-arg0", kernel_int_args[0], "Kernel integer argument 0");
  app.add_option("--kernel-int-arg1", kernel_int_args[1], "Kernel integer argument 1");
  app.add_option("--kernel-int-arg2", kernel_int_args[2], "Kernel integer argument 2");

  CLI11_PARSE(app, argc, argv);

  CUdevice cuDevice;
  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuDeviceGet(&cuDevice, 0));

  nvmlClass nvml(0, 1005);
  CudaHelper CUDA(cuDevice);

  CUfunction kernel_addr, init_addr;
  CUDA.compileFileToCUBIN(&cubin, kernel_filename.c_str(), header.c_str());
  CUmodule module = CUDA.loadCUBIN(cubin);
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
                                  numBlocksX, 1, 1, /* grid dim */
                                  threadsPerBlockX, 1, 1, /* block dim */
                                  sharedMemoryBlockBytes, 0, /* shared mem, stream */
                                  &arr[0],         /* arguments */
                                  0));
  }

  // Launch the main CUDA Kernel
  checkCudaErrors(cuFuncSetAttribute(kernel_addr, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, sharedMemoryCarveoutBytes));
  checkCudaErrors(cuFuncSetAttribute(kernel_addr, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sharedMemoryBlockBytes));
  checkCudaErrors(cuLaunchKernel(kernel_addr,
                                 numBlocksX, 1, 1, /* grid dim */
                                 threadsPerBlockX, 1, 1, /* block dim */
                                 sharedMemoryBlockBytes, 0, /* shared mem, stream */
                                 &arr[0],         /* arguments */
                                 0));
  checkCudaErrors(cuCtxSynchronize());

  for(int i = 0; i < timedRuns; i++) {
    checkCudaErrors(cuLaunchKernel(kernel_addr,
                                  numBlocksX, 1, 1, /* grid dim */
                                  threadsPerBlockX, 1, 1, /* block dim */
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
