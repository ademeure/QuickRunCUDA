#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "bench_22l_atomic_counter.fatbin.c"
extern __attribute__((visibility("hidden"))) void __device_stub__Z18kernel_atomic_syncPyS_PjPii(unsigned long long *, unsigned long long *, unsigned *, int *, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
__attribute__((visibility("hidden"))) void __device_stub__Z18kernel_atomic_syncPyS_PjPii(unsigned long long *__par0, unsigned long long *__par1, unsigned *__par2, int *__par3, int __par4){__cudaLaunchPrologue(5);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 32UL);__cudaLaunch(((char *)((void ( *)(unsigned long long *, unsigned long long *, unsigned *, int *, int))kernel_atomic_sync)));}
# 36 "bench_22l_atomic_counter.cu"
void kernel_atomic_sync( unsigned long long *__cuda_0,unsigned long long *__cuda_1,unsigned *__cuda_2,int *__cuda_3,int __cuda_4)
# 40 "bench_22l_atomic_counter.cu"
{__device_stub__Z18kernel_atomic_syncPyS_PjPii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 75 "bench_22l_atomic_counter.cu"
}
# 1 "kdir_atomic/bench_22l_atomic_counter.compute_103.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T19) {  __nv_dummy_param_ref(__T19); __nv_save_fatbinhandle_for_managed_rt(__T19); __cudaRegisterEntry(__T19, ((void ( *)(unsigned long long *, unsigned long long *, unsigned *, int *, int))kernel_atomic_sync), kernel_atomic_sync, 128); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
