// LD_PRELOAD hook: intercept all CUDA launch APIs, log function name
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>

typedef int (*cuLaunchKernel_t)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**);
typedef int (*cuLaunchKernelEx_t)(const void*, void*, void**, void**);
typedef int (*cuFuncGetName_t)(const char**, void*);

static cuLaunchKernel_t real_launch = NULL;
static cuLaunchKernelEx_t real_launch_ex = NULL;
static cuFuncGetName_t real_get_name = NULL;
static int counter = 0;

static void init() {
    if (!real_launch) {
        void *handle = dlopen("libcuda.so.1", RTLD_LAZY);
        real_launch = (cuLaunchKernel_t)dlsym(handle, "cuLaunchKernel");
        real_launch_ex = (cuLaunchKernelEx_t)dlsym(handle, "cuLaunchKernelEx");
        real_get_name = (cuFuncGetName_t)dlsym(handle, "cuFuncGetName");
    }
}

int cuLaunchKernel(void *func, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, unsigned by, unsigned bz,
                   unsigned shmem, void *stream, void **args, void **extra) {
    init();
    const char *name = "<unknown>";
    if (real_get_name) real_get_name(&name, func);
    fprintf(stderr, "HOOK[%d] cuLaunchKernel: %s grid=(%u,%u,%u) blk=(%u,%u,%u)\n",
            counter++, name, gx, gy, gz, bx, by, bz);
    return real_launch(func, gx, gy, gz, bx, by, bz, shmem, stream, args, extra);
}

int cuLaunchKernelEx(const void *config, void *func, void **args, void **extra) {
    init();
    const char *name = "<unknown>";
    if (real_get_name) real_get_name(&name, func);
    fprintf(stderr, "HOOK[%d] cuLaunchKernelEx: %s\n", counter++, name);
    return real_launch_ex(config, func, args, extra);
}
