// Very, very loosely based on a NVIDIA sample (Ship of Theseus style).

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

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "utils/cuda_helper.h"
#include "utils/ipc_helper.h"
#include "utils/nvmlClass.h"
#include "utils/CLI11.hpp"

// Structure to hold all command line arguments
struct CmdLineArgs {
  size_t arrayDwordsA = 256*1024*1024;
  size_t arrayDwordsB = 256*1024*1024;
  size_t arrayDwordsC = 256*1024*1024;
  bool randomArrays = false;
  int kernel_int_args[3] = {0};
  int threadsPerBlockX = 32;
  int numBlocksX = 1;
  int sharedMemoryBlockBytes = 0;
  int sharedMemoryCarveoutBytes = 0;
  bool runInitKernel = false;
  int timedRuns = 100;
  float perfMultiplier = 0.0f;
  float perfMultiplierPerThread = 0.0f;
  std::string perfMultiplier_unit = "ops/s";
  std::string header = "";
  std::string kernel_filename = "kernel.cu";
  bool reuse_cubin = false;
  bool dummy = false;
  bool server_mode = false;
  int clock_speed = 0;
  std::string dump_c_array = "";
  std::string dump_c_format = "raw"; // raw, int_csv, float_csv
  std::string load_c_array = ""; // New option to load C array from file
};

void setupCommandLineParser(CLI::App& app, CmdLineArgs& args) {
  app.add_flag("--server", args.server_mode, "Run in server mode");
  app.add_flag("--dummy", args.dummy, "Dummy flag");
  app.add_flag("--reuse-cubin", args.reuse_cubin, "Reuse cubin in output.cubin instead of compiling");
  app.add_option("-a,--arrayDwordsA", args.arrayDwordsA, "Array DWORDs for A");
  app.add_option("-b,--arrayDwordsB", args.arrayDwordsB, "Array DWORDs for B");
  app.add_option("-c,--arrayDwordsC", args.arrayDwordsC, "Array DWORDs for C");
  app.add_flag("-r,--randomA", args.randomArrays, "Random data for A");
  app.add_option("-t,--threadsPerBlockX", args.threadsPerBlockX, "Threads per block X");
  app.add_option("-n,--numBlocksX", args.numBlocksX, "Number of blocks X");
  app.add_option("-s,--sharedMemoryBlockBytes", args.sharedMemoryBlockBytes, "Shared memory block bytes");
  app.add_option("-o,--sharedMemoryCarveoutBytes", args.sharedMemoryCarveoutBytes, "Shared memory carveout bytes");
  app.add_flag("-i,--runInitKernel", args.runInitKernel, "Run init kernel");
  app.add_option("-T,--timedRuns", args.timedRuns, "Timed runs");
  app.add_option("-P,--perfMultiplier", args.perfMultiplier, "Performance multiplier to convert time to ops/s");
  app.add_option("-N,--perfMultiplierPerThread", args.perfMultiplierPerThread, "Performance multiplier per thread");
  app.add_option("-U,--perfMultiplier-unit", args.perfMultiplier_unit, "Performance multiplier unit (ops/s, ms, us, ns)");
  app.add_option("-H,--header", args.header, "Header string");
  app.add_option("-f,--kernel_filename", args.kernel_filename, "Kernel filename");
  app.add_option("-0,--kernel-int-arg0", args.kernel_int_args[0], "Kernel integer argument 0");
  app.add_option("-1,--kernel-int-arg1", args.kernel_int_args[1], "Kernel integer argument 1");
  app.add_option("-2,--kernel-int-arg2", args.kernel_int_args[2], "Kernel integer argument 2");
  app.add_option("--clock-speed", args.clock_speed, "Clock speed");
  app.add_option("--dump-c", args.dump_c_array, "Dump C array to specified file");
  app.add_option("--dump-c-format", args.dump_c_format, "Format for C array dump (raw, int_csv, float_csv)");
  app.add_option("--load-c", args.load_c_array, "Load C array from raw format file");
}

int run_cuda_test(const CmdLineArgs& args);

////////////////////////////////////////////////////////////////////////////////

CUdevice cuDeviceGlobal;
CUcontext cuContextGlobal;

int main(int argc, char **argv) {
  CLI::App app{"QuickBenchCUDA: Super fast iteration for CUDA microbenchmarking"};
  CmdLineArgs args;
  setupCommandLineParser(app, args);
  CLI11_PARSE(app, argc, argv);

  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuDeviceGet(&cuDeviceGlobal, 0));

  checkCudaErrors(cuCtxCreate(&cuContextGlobal, 0, cuDeviceGlobal));

  if (args.clock_speed > 0) {
    nvmlClass nvml(0, args.clock_speed);
  }

  if (args.server_mode) {
    IPCHelper ipc;

    while (true) {
        std::string cmd;
        if (ipc.waitForCommand(cmd)) {
            if (cmd == "exit") {
              break;
            }

            // Parse command into argc/argv
            std::vector<std::string> args_vec{"QuickRunCUDA"};
            std::istringstream iss(cmd);
            std::string arg;
            std::string current_arg;
            bool in_quotes = false;
            int i = 0;

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
            std::vector<char*> argv_vec;
            for (const auto& s : args_vec) {
                argv_vec.push_back(const_cast<char*>(s.c_str()));
            }

            // Redirect printf (stdout)
            fflush(stdout);
            int stdout_fd = dup(STDOUT_FILENO);
            int pipe_fd[2];
            pipe(pipe_fd);
            dup2(pipe_fd[1], STDOUT_FILENO);
            close(pipe_fd[1]);

            // Parse command line args
            CLI::App cmd_app{"QuickBenchCUDA: Super fast iteration for CUDA microbenchmarking"};
            CmdLineArgs cmd_args;
            setupCommandLineParser(cmd_app, cmd_args);
            cmd_app.parse(args_vec.size(), argv_vec.data());

            // Run main logic
            int result = run_cuda_test(cmd_args);

            // Restore stdout and get printf output
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
    // Write to file
    std::ofstream file("returning.txt");
    file << "returning 0" << "\n";
    file.close();
    return 0;
  } else {
    return run_cuda_test(args);
  }
}

int run_cuda_test(const CmdLineArgs& args) {
  // Allocate the device vectors
  CUdeviceptr d_A, d_B, d_C;
  size_t sizeA = args.arrayDwordsA * sizeof(uint);
  size_t sizeB = args.arrayDwordsB * sizeof(uint);
  size_t sizeC = args.arrayDwordsC * sizeof(uint);
  checkCudaErrors(cuMemAlloc(&d_A, sizeA));
  checkCudaErrors(cuMemAlloc(&d_B, sizeB));
  checkCudaErrors(cuMemAlloc(&d_C, sizeC));

  char *cubin;
  CudaHelper CUDA(cuDeviceGlobal);
  CUfunction kernel_addr, init_addr;
  size_t cubin_size;

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
      printf("cubin failed\n");
      fprintf(stderr, "Failed to open file to read cubin!\n");
      exit(EXIT_FAILURE);
    }
  } else {
    CUDA.compileFileToCUBIN(cuDeviceGlobal, &cubin, args.kernel_filename.c_str(), args.header.c_str(), &cubin_size);
    // Write the cubin to a binary file
    std::ofstream cubin_file("output.cubin", std::ios::binary);
    if (cubin_file.is_open()) {
      cubin_file.write(cubin, cubin_size);
      cubin_file.close();
    } else {
      fprintf(stderr, "Failed to open file to write cubin!\n");
      exit(EXIT_FAILURE);
    }
  }
  CUmodule module = CUDA.loadCUBIN(cubin, cuContextGlobal, cuDeviceGlobal);

  checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "kernel"));
  if (args.runInitKernel) {
    checkCudaErrors(cuModuleGetFunction(&init_addr, module, "init"));
  }

  // Allocate the host input vectors
  uint *h_A = reinterpret_cast<uint *>(malloc(sizeA));
  uint *h_B = reinterpret_cast<uint *>(malloc(sizeB));
  uint *h_C = reinterpret_cast<uint *>(malloc(sizeC));
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors either randomly or to zero
  if (args.randomArrays) {
    std::mt19937_64 rng(1234ULL);
    std::uniform_int_distribution<uint> dist;
    for (int i = 0; i < args.arrayDwordsA; ++i) {
      h_A[i] = dist(rng);
    }
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, sizeA));
  } else {
    memset(h_A, 0, sizeA);
    checkCudaErrors(cuMemsetD8(d_A, 0, sizeA));
  }
  memset(h_B, 0, sizeB);
  memset(h_C, 0, sizeC);
  checkCudaErrors(cuMemsetD8(d_B, 0, sizeB));

  // Load C array from file if specified
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

  // Prepare arguments to kernel function
  int kernel_int_args[3] = {args.kernel_int_args[0], args.kernel_int_args[1], args.kernel_int_args[2]};
  void *arr[] = {reinterpret_cast<void *>(&d_A),
                 reinterpret_cast<void *>(&d_B),
                 reinterpret_cast<void *>(&d_C),
                 reinterpret_cast<void *>(&kernel_int_args[0]),
                 reinterpret_cast<void *>(&kernel_int_args[1]),
                 reinterpret_cast<void *>(&kernel_int_args[2])};

  // Launch the init kernel if there is one
  if (args.runInitKernel) {
    checkCudaErrors(cuFuncSetAttribute(init_addr, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, args.sharedMemoryCarveoutBytes));
    checkCudaErrors(cuFuncSetAttribute(init_addr, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, args.sharedMemoryBlockBytes));
    checkCudaErrors(cuLaunchKernel(init_addr,
                                  args.numBlocksX, 1, 1, /* grid dim */
                                  args.threadsPerBlockX, 1, 1, /* block dim */
                                  args.sharedMemoryBlockBytes, 0, /* shared mem, stream */
                                  &arr[0],         /* arguments */
                                  0));
  }

  // Launch the main CUDA Kernel
  checkCudaErrors(cuFuncSetAttribute(kernel_addr, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, args.sharedMemoryCarveoutBytes));
  checkCudaErrors(cuFuncSetAttribute(kernel_addr, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, args.sharedMemoryBlockBytes));
  checkCudaErrors(cuLaunchKernel(kernel_addr,
                                 args.numBlocksX, 1, 1, /* grid dim */
                                 args.threadsPerBlockX, 1, 1, /* block dim */
                                 args.sharedMemoryBlockBytes, 0, /* shared mem, stream */
                                 &arr[0],         /* arguments */
                                 0));
  checkCudaErrors(cuCtxSynchronize());

  if (args.timedRuns > 0) {
    float total_time = 0.f;
    CUevent start, stop;
    checkCudaErrors(cuEventCreate(&start, CU_EVENT_DEFAULT));
    checkCudaErrors(cuEventCreate(&stop, CU_EVENT_DEFAULT));
    checkCudaErrors(cuEventRecord(start, nullptr));
    for(int i = 0; i < args.timedRuns; i++) {
      checkCudaErrors(cuLaunchKernel(kernel_addr,
                                  args.numBlocksX, 1, 1, /* grid dim */
                                  args.threadsPerBlockX, 1, 1, /* block dim */
                                  args.sharedMemoryBlockBytes, 0, /* shared mem, stream */
                                  &arr[0],         /* arguments */
                                  0));
    }
    checkCudaErrors(cuEventRecord(stop, nullptr));
    checkCudaErrors(cuEventSynchronize(start));
    checkCudaErrors(cuEventSynchronize(stop));
    checkCudaErrors(cuEventElapsedTime(&total_time, start, stop));
    printf("\n%.5f ms", total_time / args.timedRuns);

    float multiplier = args.perfMultiplier;
    if (args.perfMultiplierPerThread > 0.0f) {
      multiplier = args.perfMultiplierPerThread * args.threadsPerBlockX * args.numBlocksX;
    }
    if (multiplier > 0.0f) {
      printf(" ==> %.4f %s\n\n", args.timedRuns * multiplier / (total_time / 1000.f), args.perfMultiplier_unit.c_str());
    }
  }
  checkCudaErrors(cuCtxSynchronize());

  // Dump B array if requested
  if (!args.dump_c_array.empty()) {
    checkCudaErrors(cuMemcpyDtoH(h_C, d_C, sizeC));

    if (args.dump_c_format == "raw") {
      std::ofstream outfile(args.dump_c_array, std::ios::binary);
      outfile.write(reinterpret_cast<char*>(h_C), sizeC);
    } else {
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

  checkCudaErrors(cuMemFree(d_A));
  checkCudaErrors(cuMemFree(d_B));
  checkCudaErrors(cuMemFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);

  // free module
  checkCudaErrors(cuModuleUnload(module));
  return 0;
}
