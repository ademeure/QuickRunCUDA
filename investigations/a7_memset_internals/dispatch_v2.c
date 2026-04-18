// Hook the _v2 / _ptds / _ptsz variants the runtime actually uses
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>

static int counter = 0;

#define HOOK(NAME, ARGS, CALL) \
    typedef int (*NAME##_t) ARGS; \
    static NAME##_t real_##NAME = NULL; \
    int NAME ARGS { \
        if (!real_##NAME) real_##NAME = (NAME##_t)dlsym(RTLD_NEXT, #NAME); \
        fprintf(stderr, "HOOK[%d] %s\n", counter++, #NAME); \
        return real_##NAME CALL; \
    }

HOOK(cuMemsetD8_v2_ptds, (void* p, unsigned char v, size_t n), (p, v, n))
HOOK(cuMemsetD16_v2_ptds, (void* p, unsigned short v, size_t n), (p, v, n))
HOOK(cuMemsetD32_v2_ptds, (void* p, unsigned int v, size_t n), (p, v, n))
HOOK(cuMemsetD8Async_ptsz, (void* p, unsigned char v, size_t n, void* s), (p, v, n, s))
HOOK(cuMemsetD32Async_ptsz, (void* p, unsigned int v, size_t n, void* s), (p, v, n, s))
HOOK(cuMemsetD2D8_v2_ptds, (void* p, size_t pitch, unsigned char v, size_t w, size_t h), (p, pitch, v, w, h))
HOOK(cuMemsetD2D32_v2_ptds, (void* p, size_t pitch, unsigned int v, size_t w, size_t h), (p, pitch, v, w, h))
