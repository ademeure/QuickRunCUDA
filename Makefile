################################################################################
# Makefile for QuickRunCUDA
################################################################################

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

# Host compiler
HOST_COMPILER ?= g++

# NVCC compiler
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# Common flags
NVCCFLAGS := -m64
ALL_CCFLAGS := --threads 0 --std=c++17
ALL_LDFLAGS :=

# GPU Compute capability (auto-detect)
ifndef GPU_COMPUTE_CAPABILITY
    GPU_COMPUTE_CAPABILITY := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/\.//g' | sort -n | head -n 1)
    GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))
endif

ifneq ($(GPU_COMPUTE_CAPABILITY),)
    SM_COMPUTE_CAPABILITY := $(GPU_COMPUTE_CAPABILITY)$(if $(filter 90 100,$(GPU_COMPUTE_CAPABILITY)),a,)
    GENCODE_FLAGS := --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(SM_COMPUTE_CAPABILITY)]
    NVCCFLAGS += $(GENCODE_FLAGS)
endif

# Libraries: only CUDA Driver API + NVRTC
LIBRARIES :=
CUDA_SEARCH_PATH := $(CUDA_PATH)/lib64/stubs $(CUDA_PATH)/lib/stubs
CUDALIB := $(shell find -L $(CUDA_SEARCH_PATH) -maxdepth 1 -name libcuda.so 2> /dev/null | head -1)

ifeq ("$(CUDALIB)","")
    $(info >>> WARNING - libcuda.so not found, CUDA Driver is not installed. <<<)
else
    CUDALIB := $(shell echo $(CUDALIB) | sed "s/ .*//" | sed "s/\/libcuda.so//" )
    LIBRARIES += -L$(CUDALIB) -lcuda
endif

LIBRARIES += -lnvrtc

# Include paths
INCLUDES := -I$(CUDA_PATH)/include

# Source files
SOURCES := QuickRunCUDA.cpp

# Targets
all: build

build: QuickRunCUDA

QuickRunCUDA: $(SOURCES)
	$(NVCC) $(NVCCFLAGS) $(ALL_CCFLAGS) $(ALL_LDFLAGS) $(INCLUDES) -o $@ $+ $(LIBRARIES)

run: build
	./QuickRunCUDA

clean:
	rm -f QuickRunCUDA QuickRunCUDA.o

.PHONY: all build run clean
