#pragma once

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <cuda.h>
#include <nvrtc.h>

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

template <typename T>
inline void __checkCudaErrors(T err, const char *file, const int line) {
    if (err != 0) {
        const char *errorStr = "";
        if constexpr (std::is_same<T, CUresult>::value)
            cuGetErrorString(err, &errorStr);
        else if constexpr (std::is_same<T, nvrtcResult>::value)
            errorStr = nvrtcGetErrorString(err);
        fprintf(stderr, "CUDA error = %04d \"%s\" at %s:%d\n",
                (int)err, errorStr, file, line);
        exit(EXIT_FAILURE);
    }
}
#endif
