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
#include <cctype>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "cuda_helper.h"
#include "nvmlClass.h"
#include "cuda_metrics/measureMetricPW.hpp"
#include "testRunner.h"
#include "ipc_helper.h"

// Structure to hold all command line arguments
struct CmdLineArgs {
  size_t arrayInputDwords = 16*1024*1024;
  size_t arrayOutputDwords = 16*1024*1024;
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
  bool reuse_cubin = false;
  bool dummy = false;
  bool server_mode = false;
  int clock_speed = 765;
  std::string dump_b_array = "";
  std::string dump_b_format = "raw"; // raw, int_csv, float_csv
};

void setupCommandLineParser(CLI::App& app, CmdLineArgs& args) {
  app.add_flag("--server", args.server_mode, "Run in server mode");
  app.add_flag("--dummy", args.dummy, "Dummy flag");
  app.add_flag("--reuse-cubin", args.reuse_cubin, "Reuse cubin in output.cubin instead of compiling");
  app.add_option("-a,--arrayInputDwords", args.arrayInputDwords, "Array input DWORDs");
  app.add_option("-o,--arrayOutputDwords", args.arrayOutputDwords, "Array output DWORDs");
  app.add_flag("-r,--randomArrays", args.randomArrays, "Random arrays");
  app.add_option("-t,--threadsPerBlockX", args.threadsPerBlockX, "Threads per block X");
  app.add_option("-b,--numBlocksX", args.numBlocksX, "Number of blocks X");
  app.add_option("-s,--sharedMemoryBlockBytes", args.sharedMemoryBlockBytes, "Shared memory block bytes");
  app.add_option("-c,--sharedMemoryCarveoutBytes", args.sharedMemoryCarveoutBytes, "Shared memory carveout bytes");
  app.add_flag("-i,--runInitKernel", args.runInitKernel, "Run init kernel");
  app.add_flag("-T,--timedRuns", args.timedRuns, "Timed runs");
  app.add_option("-H,--header", args.header, "Header string");
  app.add_option("-f,--kernel_filename", args.kernel_filename, "Kernel filename");
  app.add_option("--kernel-int-arg0", args.kernel_int_args[0], "Kernel integer argument 0");
  app.add_option("--kernel-int-arg1", args.kernel_int_args[1], "Kernel integer argument 1");
  app.add_option("--kernel-int-arg2", args.kernel_int_args[2], "Kernel integer argument 2");
  app.add_option("--clock-speed", args.clock_speed, "Clock speed");
  app.add_option("--dump-b", args.dump_b_array, "Dump B array to specified file");
  app.add_option("--dump-b-format", args.dump_b_format, "Format for B array dump (raw, int_csv, float_csv)");
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
  nvmlClass nvml(0, args.clock_speed);

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

  size_t inputSize = args.arrayInputDwords * sizeof(uint);
  size_t outputSize = args.arrayOutputDwords * sizeof(uint);

  // Allocate the host input vectors
  uint *h_A = reinterpret_cast<uint *>(malloc(inputSize));
  uint *h_O = reinterpret_cast<uint *>(malloc(outputSize));
  if (h_A == NULL || h_O == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors either randomly or to zero
  if (args.randomArrays) {
    std::mt19937_64 rng(1234ULL);
    std::uniform_int_distribution<uint> dist;
    for (int i = 0; i < args.arrayInputDwords; ++i) {
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
  int kernel_int_args[3] = {args.kernel_int_args[0], args.kernel_int_args[1], args.kernel_int_args[2]};
  void *arr[] = {reinterpret_cast<void *>(&d_A),
                 reinterpret_cast<void *>(&d_O),
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

  for(int i = 0; i < args.timedRuns; i++) {
    checkCudaErrors(cuLaunchKernel(kernel_addr,
                                  args.numBlocksX, 1, 1, /* grid dim */
                                  args.threadsPerBlockX, 1, 1, /* block dim */
                                  args.sharedMemoryBlockBytes, 0, /* shared mem, stream */
                                  &arr[0],         /* arguments */
                                  0));
  }

  // Dump B array if requested
  if (!args.dump_b_array.empty()) {
    checkCudaErrors(cuMemcpyDtoH(h_O, d_O, outputSize));

    if (args.dump_b_format == "raw") {
      std::ofstream outfile(args.dump_b_array, std::ios::binary);
      outfile.write(reinterpret_cast<char*>(h_O), outputSize);
    } else {
      std::ofstream outfile(args.dump_b_array);
      for (size_t i = 0; i < args.arrayOutputDwords; i++) {
        if (h_O[i] != 0) {
          if (args.dump_b_format == "int_csv") {
            outfile << h_O[i];
          } else if (args.dump_b_format == "float_csv") {
            outfile << std::fixed << std::setprecision(1) << *reinterpret_cast<float*>(&h_O[i]);
          }
        }
        if (i < args.arrayOutputDwords - 1) outfile << ",";
      }
    }
  }

  checkCudaErrors(cuMemFree(d_A));
  checkCudaErrors(cuMemFree(d_O));
  free(h_A);
  free(h_O);

  // free module
  checkCudaErrors(cuModuleUnload(module));
  return 0;
}
