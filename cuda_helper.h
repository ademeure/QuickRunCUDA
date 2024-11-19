#define GPU_SM_COUNT 128
#define FLOPS_PER_SM 256
#define DEFAULT_GPU_CLOCK 735
#define MAX_FLOPS_PER_CLOCK ((GPU_SM_COUNT * FLOPS_PER_SM) * 1e6f)

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions
// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
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








#define CUDA_MAX_STREAMS 4
class CudaHelper {
public:
  CudaHelper(int argc=0, char **argv=NULL, size_t shared_carveout_KB=64) :
  next_shared_carveout_KB_(shared_carveout_KB)
  {
    device_id_ = 0; //findCudaDevice(argc, (const char **)argv); // TODO: we want to support >1 devices eventually!
    checkCudaErrors(cudaGetDeviceProperties(&device_prop_, device_id_));

    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
    for(int i = 0; i < CUDA_MAX_STREAMS; i++) {
      // Negative numbers are higher priority (so stream 4 is highest priority)
      checkCudaErrors(cudaStreamCreateWithPriority(&streams_[i], cudaStreamNonBlocking, -i));
    }
  }

  ~CudaHelper() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
    for(int i = 0; i < CUDA_MAX_STREAMS; i++) {
      checkCudaErrors(cudaStreamDestroy(streams_[i]));
    }
  }

  void sync(cudaStream_t stream=0) {
    if (stream) {
      checkCudaErrors(cudaStreamSynchronize(stream));
    } else {
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }

  void setKernelSize(dim3 block_dims, dim3 thread_dims, int shared_memory_bytes=-1, int shared_carveout_KB=-1) {
    next_block_dims_ = block_dims;
    next_thread_dims_ = thread_dims;

    if (shared_memory_bytes >= 0) {
      next_shared_memory_bytes_ = shared_memory_bytes;
    }
    if (shared_carveout_KB >= 0) {
      next_shared_carveout_KB_ = shared_carveout_KB;
    }
  }

  void setKernelSize(unsigned int block_dim_x, unsigned int thread_dim_x, int shared_memory_bytes=-1, int shared_carveout_KB=-1) {
    setKernelSize(dim3(block_dim_x, 1, 1), dim3(thread_dim_x, 1, 1), shared_memory_bytes, shared_carveout_KB);
  }

  void getKernelSize(dim3 &block_dims, dim3 &thread_dims) {
    block_dims = next_block_dims_;
    thread_dims = next_thread_dims_;
  }

  void launch(CUfunction func, void** args, cudaStream_t stream=0, bool cooperative = false) {
    checkCudaErrors(cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, next_shared_carveout_KB_));
    checkCudaErrors(cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, next_shared_memory_bytes_));
    if (cooperative) {
      checkCudaErrors(cuLaunchCooperativeKernel(func, next_block_dims_.x, next_block_dims_.y, next_block_dims_.z, /* grid dim */
                                                next_thread_dims_.x, next_thread_dims_.y, next_thread_dims_.z, /* block dim */
                                                next_shared_memory_bytes_, stream, args));
    } else {
      checkCudaErrors(cuLaunchKernel(func, next_block_dims_.x, next_block_dims_.y, next_block_dims_.z, /* grid dim */
                                    next_thread_dims_.x, next_thread_dims_.y, next_thread_dims_.z, /* block dim */
                                    next_shared_memory_bytes_, stream, args, 0));
    }
  }

  void launchCooperative(CUfunction func, void** args, cudaStream_t stream=0) {
    launch(func, args, stream, true);
  }

  void startTimerGPU(bool sync=false, cudaStream_t stream=0) {
    assert(!active_timer_gpu_);
    active_timer_gpu_ = true;

    if (sync) { this->sync(stream); }
    cudaEventRecord(start_event_, stream);
  }

  float endTimerGPU(bool sync=false, cudaStream_t stream=0) {
    assert(active_timer_gpu_);
    active_timer_gpu_ = false;

    float ms;
    cudaEventRecord(stop_event_, stream);
    if (sync) { this->sync(stream); }
    cudaEventSynchronize(stop_event_); // TODO: Is this the right function?! If so, why do we seem to need sync() too?
    cudaEventElapsedTime(&ms, start_event_, stop_event_);
    return ms;
  }

  // Use active_count to prevent multiple timers being active at once since this is just a single delta timer
  // 0 == don't care, 1 == start, -1 == stop
  float deltaTimerCPU(int active_count=0) {
    assert(active_count == 0 || active_timer_cpu_ && active_count < 0 || !active_timer_cpu_ && active_count > 0);
    active_timer_cpu_ = (active_count != 0) ? (active_count > 0) : active_timer_cpu_;

    auto current_time = std::chrono::high_resolution_clock::now();
    float microseconds = std::chrono::duration_cast<std::chrono::microseconds>(current_time - timerCPU_).count();
    timerCPU_ = current_time;

    // Convert to milliseconds (using std::chrone::microseconds seems to increase accuracy? TBC...)
    return microseconds / 1000.0f;
  }

  float launchSimpleTest(CUfunction func, void** args, bool cooperative=false) {
    sync();
    startTimerGPU();
    launch(func, args, 0, cooperative);
    return endTimerGPU();
  }

  /*
  void rng_on_device(size_t size, unsigned int *d_data,
                    uint64_t seed = 1234735980ULL, cudaStream_t stream=0) {
    // TODO: Which generator to use?
    curandGenerator_t gen; // TODO: Allow keeping/reusing generator?
    assert(!curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    assert(!curandSetStream(gen, stream));
    //checkCudaErrors(curandSetGeneratorOffset(gen, offset));
    assert(!curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_DEFAULT));
    assert(!curandSetPseudoRandomGeneratorSeed(gen, seed));
    assert(!curandGenerate(gen, d_data, size));
    checkCudaErrors(cudaStreamSynchronize(stream));
    assert(!curandDestroyGenerator(gen));
  }
  */

  void compileFileToCUBIN(CUdevice cuDevice, char **cubinResult, char const *filename, char const *header=NULL, size_t *cubinResultSize=NULL, const char* cudaIncludePath = "/usr/local/cuda/include/") {
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

    // get compute capabilities
    int major = 0, minor = 0;
    checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

    int numCompileOptions = 0;
    char *compileParams[3];

    // Compile cubin for the GPU arch on which are going to run cuda kernel.
    // HACK: Turn sm_90 into sm_90a
    std::string compileOptions;
    compileOptions = "--gpu-architecture=sm_";
    compileParams[numCompileOptions] = reinterpret_cast<char *>(malloc(sizeof(char) * (compileOptions.length() + 11)));
  #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 10),
              "%s%d%d%s", compileOptions.c_str(), major, minor, (major == 9 && minor == 0) ? "a" : "");
  #else
    snprintf(compileParams[numCompileOptions], compileOptions.size() + 10, "%s%d%d%s", compileOptions.c_str(), major, minor, (major == 9 && minor == 0) ? "a" : "");
  #endif

    // Increment numCompileOptions to account for the added option
    numCompileOptions++;

    // Allocate memory and format the include directory option
    compileParams[numCompileOptions] = reinterpret_cast<char *>(malloc(sizeof(char) * (strlen(cudaIncludePath) + 3))); // +3 for -I and null terminator
    sprintf(compileParams[numCompileOptions], "-I%s", cudaIncludePath);

    // Increment numCompileOptions to account for the added option
    numCompileOptions++;

    // compile
    nvrtcProgram prog;
    checkCudaErrors(nvrtcCreateProgram(&prog, memBlock, filename, 0, NULL, NULL));
    nvrtcResult res = nvrtcCompileProgram(prog, numCompileOptions, compileParams);

    // dump log
    size_t logSize;
    checkCudaErrors(nvrtcGetProgramLogSize(prog, &logSize));
    char *log = reinterpret_cast<char *>(malloc(sizeof(char) * logSize + 1));
    checkCudaErrors(nvrtcGetProgramLog(prog, log));
    log[logSize] = '\x0';

    if (strlen(log) >= 2) {
      std::cerr << "\n------- ERROR DURING KERNEL COMPILATION -------\n\n";
      std::cerr << log;
      std::cerr << "\n------- END LOG -------\n";
    }
    free(log);

    checkCudaErrors(res);

    size_t codeSize;
    checkCudaErrors(nvrtcGetCUBINSize(prog, &codeSize));
    char *code = new char[codeSize];
    checkCudaErrors(nvrtcGetCUBIN(prog, code));
    *cubinResult = code;
    if (cubinResultSize) {
      *cubinResultSize = codeSize;
    }

    for (int i = 0; i < numCompileOptions; i++) {
      free(compileParams[i]);
    }
  }

  CUmodule loadCUBIN(char *cubin, CUcontext context, CUdevice cuDevice) {
    CUmodule module;
    checkCudaErrors(cuModuleLoadData(&module, cubin));
    free(cubin);
    return module;
  }

  cudaStream_t operator [] (int i) {
    return stream(i);
  }

  cudaStream_t stream(int i=0) { assert(i < CUDA_MAX_STREAMS); return streams_[i]; }
  int deviceID() { return device_id_; }
  cudaDeviceProp deviceProp() { return device_prop_; }

private:
  int device_id_;
  cudaStream_t streams_[CUDA_MAX_STREAMS];
  cudaDeviceProp device_prop_;

  bool active_timer_gpu_ = false;
  bool active_timer_cpu_ = false;
  cudaEvent_t start_event_, stop_event_;
  std::chrono::high_resolution_clock::time_point timerCPU_;

  dim3 next_block_dims_ = dim3(1,1,1);
  dim3 next_thread_dims_ = dim3(32,1,1);
  size_t next_shared_memory_bytes_ = 0;
  size_t next_shared_carveout_KB_ = 0;
};
