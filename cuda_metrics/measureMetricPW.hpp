#pragma once
#include "Eval.hpp"
#include "Metric.hpp"
#include <cuda.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <functional>
#include <nvperf_host.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

const CUpti_ProfilerReplayMode replayMode = CUPTI_KernelReplay;

#define DRIVER_API_CALL(apiFunctionCall)                                            \
do                                                                                  \
{                                                                                   \
    CUresult _status = apiFunctionCall;                                             \
    if (_status != CUDA_SUCCESS)                                                    \
    {                                                                               \
        const char *pErrorString;                                                   \
        cuGetErrorString(_status, &pErrorString);                                   \
                                                                                    \
        fprintf(stderr, "%s:%d: Error: Function %s failed with error: %s.\n",       \
                __FILE__, __LINE__, #apiFunctionCall, pErrorString);                \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define RUNTIME_API_CALL(apiFunctionCall)                                           \
do                                                                                  \
{                                                                                   \
    cudaError_t _status = apiFunctionCall;                                          \
    if (_status != cudaSuccess)                                                     \
    {                                                                               \
        fprintf(stderr, "%s:%d: Error: Function %s failed with error: %s.\n",       \
                __FILE__, __LINE__, #apiFunctionCall, cudaGetErrorString(_status)); \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define CUPTI_API_CALL(apiFunctionCall)                                             \
do                                                                                  \
{                                                                                   \
    CUptiResult _status = apiFunctionCall;                                          \
    if (_status != CUPTI_SUCCESS)                                                   \
    {                                                                               \
        const char *pErrorString;                                                   \
        cuptiGetResultString(_status, &pErrorString);                               \
                                                                                    \
        fprintf(stderr, "%s:%d: Error: Function %s failed with error: %s.\n",       \
                __FILE__, __LINE__, #apiFunctionCall, pErrorString);                \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define NVPW_API_CALL(apiFunctionCall)                                              \
do                                                                                  \
{                                                                                   \
    NVPA_Status _status = apiFunctionCall;                                          \
    if (_status != NVPA_STATUS_SUCCESS)                                             \
    {                                                                               \
        fprintf(stderr, "%s:%d: Error: Function %s failed with error: %d.\n",       \
                __FILE__, __LINE__, #apiFunctionCall, _status);                     \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

namespace {
CUcontext cuContext;

CUdevice cuDevice;
std::string chipName;
std::vector<std::string> metricNames;

std::vector<uint8_t> counterDataImage;
std::vector<uint8_t> counterDataImagePrefix;
std::vector<uint8_t> configImage;
std::vector<uint8_t> counterDataScratchBuffer;
std::vector<uint8_t> counterAvailabilityImage;
} // namespace

bool
CreateCounterDataImage(
    std::vector<uint8_t>& counterDataImage,
    std::vector<uint8_t>& counterDataScratchBuffer,
    std::vector<uint8_t>& counterDataImagePrefix)
{
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
    counterDataImageOptions.maxNumRanges = 1;
    counterDataImageOptions.maxNumRangeTreeNodes = 1;
    counterDataImageOptions.maxRangeNameLength = 16;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = { CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE };
    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    counterDataImage.resize(calculateSizeParams.counterDataImageSize);
    initializeParams.pCounterDataImage = &counterDataImage[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = { CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE };
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));
    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = { CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE };
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

    return true;
}


bool runTestStart(CUdevice cuDevice) {
  CUpti_Profiler_BeginSession_Params beginSessionParams = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };
  beginSessionParams.ctx = NULL;
  beginSessionParams.counterDataImageSize = counterDataImage.size();
  beginSessionParams.pCounterDataImage = &counterDataImage[0];
  beginSessionParams.counterDataScratchBufferSize = counterDataScratchBuffer.size();
  beginSessionParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
  beginSessionParams.range = CUPTI_AutoRange;
  beginSessionParams.replayMode = replayMode;
  beginSessionParams.maxRangesPerPass = 1;
  beginSessionParams.maxLaunchesPerPass = 1;
  CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

  CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
  setConfigParams.pConfig = &configImage[0];
  setConfigParams.configSize = configImage.size();
  setConfigParams.passIndex = 0;
  CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));

  CUpti_Profiler_EnableProfiling_Params enableProfilingParams   = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
  CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
  
  return true;
}

bool runTestEnd() {
  CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
  CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

  CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

  CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

  return true;
}

bool static initialized = false;

double measureMetricsStart(std::vector<std::string> newMetricNames) {
  if (!newMetricNames.size())
  {
      std::cout << "No metrics provided to profile" << std::endl;
      exit(EXIT_FAILURE);
  }

  if (!initialized) {
    cudaFree(0);

    int deviceNum = 0;
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    if (computeCapabilityMajor < 7) {
      printf("Unsupported on Device with compute capability < 7.0\n");
      return -2.0;
    }

    // TODO: Only need to do once?
    // CUPTI Profiler API initialization.
    CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
    CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

    /* Get chip name for the cuda  device */
    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = deviceNum;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    chipName = getChipNameParams.pChipName;

    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    getCounterAvailabilityParams.ctx = cuContext;
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    counterAvailabilityImage.clear();
    counterAvailabilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);
    getCounterAvailabilityParams.pCounterAvailabilityImage = counterAvailabilityImage.data();
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    // Generate configuration for metrics, this can also be done offline.
    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));
  }

  // check if new & old metric names are the exact same set of strings
  if (metricNames.size() != newMetricNames.size() || !std::equal(metricNames.begin(), metricNames.end(), newMetricNames.begin())) {
    metricNames = newMetricNames;
    
    counterDataImagePrefix = std::vector<uint8_t>();
    configImage = std::vector<uint8_t>();
    counterDataScratchBuffer = std::vector<uint8_t>();
    counterDataImage = std::vector<uint8_t>();

    if (!NV::Metric::Config::GetConfigImage(chipName, metricNames, configImage, counterAvailabilityImage.data()))
    {
        std::cout << "Failed to create configImage" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (!NV::Metric::Config::GetCounterDataPrefixImage(chipName, metricNames, counterDataImagePrefix))
    {
        std::cout << "Failed to create counterDataImagePrefix" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (!CreateCounterDataImage(counterDataImage, counterDataScratchBuffer, counterDataImagePrefix))
    {
        std::cout << "Failed to create counterDataImage" << std::endl;
        exit(EXIT_FAILURE);
    }
  }

  runTestStart(cuDevice);
  initialized = true;
  return 0.0;
}

std::vector<double> measureMetricsStop() {
  runTestEnd();

  //return std::vector<double>(10, 0.0);
  return NV::Metric::Eval::GetMetricValues(chipName, counterDataImage, metricNames);
}

extern "C" void measureMultiStart() {
  // 0: Instruction Issue % (All SMs?)
  // 1: DRAM %
  // 2: L2 %
  // 3: L1 % (All SMs?)
  // 4: DRAM Read Bytes
  // 5: DRAM Write Bytes
  // 6: L2 Read Bytes (after x32 in Stop())
  // 7: L2 Write Bytes (after x32 in Stop())
  // 8: Warp Occupancy % (1st SM only?)
  // 9: Number of Warps Eligible (All SMs?)
  measureMetricsStart({"smsp__issue_active.avg.pct_of_peak_sustained_active",
                       "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                       "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
                       "smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
                       "dram__bytes_read.sum",
                       "dram__bytes_write.sum",
                       "lts__t_sectors_srcunit_tex_op_read.sum",
                       "lts__t_sectors_srcunit_tex_op_write.sum",
                       "sm__warps_active.avg.pct_of_peak_sustained_active",
                       "smsp__warps_eligible.sum.per_cycle_active"});
}

extern "C" void measureDRAMBytesStart() {
  measureMetricsStart({"dram__bytes_read.sum", "dram__bytes_write.sum"});
}

extern "C" void measureL2BytesStart() {
  measureMetricsStart({"lts__t_sectors_srcunit_tex_op_read.sum",
                      "lts__t_sectors_srcunit_tex_op_write.sum"});
}

std::vector<double> measureMultiStop() {  
  auto values = measureMetricsStop();
  values[6] *= 32;
  values[7] *= 32;
  return values;
}

std::vector<double> measureDRAMBytesStop() {
  auto values = measureMetricsStop();
  return values;
}

std::vector<double> measureL2BytesStop() {  
  auto values = measureMetricsStop();
  values[0] *= 32;
  values[1] *= 32;
  return values;
}