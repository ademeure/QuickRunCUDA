################################################################################
# Makefile for QuickRunCUDA
################################################################################

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

# Host compiler
HOST_COMPILER ?= g++

# NVCC compiler
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# Detect host architecture
HOST_ARCH := $(shell uname -m)

# Set target architecture to host architecture
TARGET_ARCH := $(HOST_ARCH)

# Debug build flags
ifeq ($(dbg),1)
    NVCCFLAGS += -g -G
    BUILD_TYPE := debug
else
    BUILD_TYPE := release
endif

# Common NVCC flags
NVCCFLAGS := -m64
CCFLAGS   :=
LDFLAGS   :=

# Add C++17 support and compiler flags
ALL_CCFLAGS := --threads 0 --std=c++17
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

# GPU Compute capability
# Check for GPU compute capability if not explicitly set
ifndef GPU_COMPUTE_CAPABILITY
    # Remove decimal points, sort numerically in ascending order, and select the first (lowest) value
    GPU_COMPUTE_CAPABILITY := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/\.//g' | sort -n | head -n 1)
    GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))
endif

# If we found a GPU, add the appropriate architecture flags
ifneq ($(GPU_COMPUTE_CAPABILITY),)
    # Add 'a' suffix to sm_xx for compute capabilities 90 and 100
    SM_COMPUTE_CAPABILITY := $(GPU_COMPUTE_CAPABILITY)$(if $(filter 90 100,$(GPU_COMPUTE_CAPABILITY)),a,)
    GENCODE_FLAGS := --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(SM_COMPUTE_CAPABILITY)]
    NVCCFLAGS += $(GENCODE_FLAGS)
endif

# Linker flags
ALL_LDFLAGS := $(addprefix -Xlinker ,$(LDFLAGS))

# Libraries
LIBRARIES :=

# Find libcuda.so on Linux
CUDA_SEARCH_PATH := $(CUDA_PATH)/lib64/stubs $(CUDA_PATH)/lib/stubs
CUDALIB := $(shell find -L $(CUDA_SEARCH_PATH) -maxdepth 1 -name libcuda.so 2> /dev/null | head -1)

ifeq ("$(CUDALIB)","")
    $(info >>> WARNING - libcuda.so not found, CUDA Driver is not installed. Please re-install the driver. <<<)
else
    CUDALIB := $(shell echo $(CUDALIB) | sed "s/ .*//" | sed "s/\/libcuda.so//" )
    LIBRARIES += -L$(CUDALIB) -lcuda
    LIBRARIES += -L$(CUDA_PATH)/lib64 -lcupti -lnvidia-ml -lnvperf_host -lnvperf_target -lcurand
endif

# Always add NVRTC
LIBRARIES += -lnvrtc

# Include paths
INCLUDES := -I$(CUDA_PATH)/include

# Source files
SOURCES := QuickRunCUDA.cpp

# Target rules
all: build

build: QuickRunCUDA

QuickRunCUDA: $(SOURCES)
	$(NVCC) $(NVCCFLAGS) $(ALL_CCFLAGS) $(ALL_LDFLAGS) $(INCLUDES) -o $@ $+ $(LIBRARIES)

run: build
	./QuickRunCUDA

clean:
	rm -f QuickRunCUDA QuickRunCUDA.o output.cubin

.PHONY: all build run clean