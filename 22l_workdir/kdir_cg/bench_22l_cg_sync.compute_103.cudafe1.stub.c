#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "bench_22l_cg_sync.fatbin.c"
extern __attribute__((visibility("hidden"))) void __device_stub__Z14kernel_cg_syncPyS_iPi(unsigned long long *, unsigned long long *, int, int *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
__attribute__((visibility("hidden"))) void __device_stub__Z14kernel_cg_syncPyS_iPi(unsigned long long *__par0, unsigned long long *__par1, int __par2, int *__par3){__cudaLaunchPrologue(4);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaLaunch(((char *)((void ( *)(unsigned long long *, unsigned long long *, int, int *))kernel_cg_sync)));}
# 18 "bench_22l_cg_sync.cu"
void kernel_cg_sync( unsigned long long *__cuda_0,unsigned long long *__cuda_1,int __cuda_2,int *__cuda_3)
# 18 "bench_22l_cg_sync.cu"
{__device_stub__Z14kernel_cg_syncPyS_iPi( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 62 "bench_22l_cg_sync.cu"
}
# 1 "kdir_cg/bench_22l_cg_sync.compute_103.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T50) {  __nv_dummy_param_ref(__T50); __nv_save_fatbinhandle_for_managed_rt(__T50); __cudaRegisterEntry(__T50, ((void ( *)(unsigned long long *, unsigned long long *, int, int *))kernel_cg_sync), kernel_cg_sync, 128); __cudaRegisterVariable(__T50, __shadow_var(_ZN51_INTERNAL_b513959c_20_bench_22l_cg_sync_cu_d24d41134cuda3std3__45__cpo9iter_swapE,::cuda::std::__4::__cpo::iter_swap), 0, 1UL, 0, 0); __cudaRegisterVariable(__T50, __shadow_var(_ZN51_INTERNAL_b513959c_20_bench_22l_cg_sync_cu_d24d41134cuda3std9execution3__43seqE,::cuda::std::execution::__4::seq), 0, 1UL, 0, 0); __cudaRegisterVariable(__T50, __shadow_var(_ZN51_INTERNAL_b513959c_20_bench_22l_cg_sync_cu_d24d41134cuda3std9execution3__43parE,::cuda::std::execution::__4::par), 0, 1UL, 0, 0); __cudaRegisterVariable(__T50, __shadow_var(_ZN51_INTERNAL_b513959c_20_bench_22l_cg_sync_cu_d24d41134cuda3std9execution3__49par_unseqE,::cuda::std::execution::__4::par_unseq), 0, 1UL, 0, 0); __cudaRegisterVariable(__T50, __shadow_var(_ZN51_INTERNAL_b513959c_20_bench_22l_cg_sync_cu_d24d41134cuda3std9execution3__45unseqE,::cuda::std::execution::__4::unseq), 0, 1UL, 0, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
