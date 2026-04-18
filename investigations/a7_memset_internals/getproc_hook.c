// Hook cuGetProcAddress to see what runtime asks for
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>

typedef int (*cuGetProcAddress_t)(const char* sym, void** ptr, int ver, unsigned long long flags);
typedef int (*cuGetProcAddress_v2_t)(const char* sym, void** ptr, int ver, unsigned long long flags, void* status);

static cuGetProcAddress_t real_v1 = NULL;
static cuGetProcAddress_v2_t real_v2 = NULL;
static int counter = 0;

int cuGetProcAddress(const char* sym, void** ptr, int ver, unsigned long long flags) {
    if (!real_v1) real_v1 = (cuGetProcAddress_t)dlsym(RTLD_NEXT, "cuGetProcAddress");
    int r = real_v1(sym, ptr, ver, flags);
    if (strstr(sym, "Memset") || strstr(sym, "memset"))
        fprintf(stderr, "GETPROC[%d] %s ver=%d -> ret=%d ptr=%p\n", counter++, sym, ver, r, ptr ? *ptr : NULL);
    return r;
}

int cuGetProcAddress_v2(const char* sym, void** ptr, int ver, unsigned long long flags, void* status) {
    if (!real_v2) real_v2 = (cuGetProcAddress_v2_t)dlsym(RTLD_NEXT, "cuGetProcAddress_v2");
    int r = real_v2(sym, ptr, ver, flags, status);
    if (strstr(sym, "Memset") || strstr(sym, "memset") || strstr(sym, "Launch"))
        fprintf(stderr, "GETPROC2[%d] %s ver=%d -> ret=%d ptr=%p\n", counter++, sym, ver, r, ptr ? *ptr : NULL);
    return r;
}
