#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "cg_sass_check.fatbin.c"
extern __attribute__((visibility("hidden"))) void __device_stub__Z6cg_redPjj(unsigned *, unsigned);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
__attribute__((visibility("hidden"))) void __device_stub__Z6cg_redPjj(unsigned *__par0, unsigned __par1){__cudaLaunchPrologue(2);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaLaunch(((char *)((void ( *)(unsigned *, unsigned))cg_red)));}
# 5 "/tmp/cg_sass_check.cu"
void cg_red( unsigned *__cuda_0,unsigned __cuda_1)
# 5 "/tmp/cg_sass_check.cu"
{__device_stub__Z6cg_redPjj( __cuda_0,__cuda_1);



}
# 1 "cg_sass_check.compute_103a.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T47) {  __nv_dummy_param_ref(__T47); __nv_save_fatbinhandle_for_managed_rt(__T47); __cudaRegisterEntry(__T47, ((void ( *)(unsigned *, unsigned))cg_red), cg_red, (-1)); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std3__45__cpo9iter_swapE,::cuda::std::__4::__cpo::iter_swap), 0, 1UL, 0, 0); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std3__45__cpo5beginE,::cuda::std::__4::__cpo::begin), 0, 1UL, 0, 0); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std3__45__cpo3endE,::cuda::std::__4::__cpo::end), 0, 1UL, 0, 0); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std3__45__cpo6cbeginE,::cuda::std::__4::__cpo::cbegin), 0, 1UL, 0, 0); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std3__45__cpo4cendE,::cuda::std::__4::__cpo::cend), 0, 1UL, 0, 0); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std3__447_GLOBAL__N__6d7068ed_16_cg_sass_check_cu_cg_red6ignoreE,::cuda::std::__4::_NV_ANON_NAMESPACE::ignore), 0, 1UL, 0, 0); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std3__419piecewise_constructE,::cuda::std::__4::piecewise_construct), 0, 1UL, 0, 0); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std9execution3__43seqE,::cuda::std::execution::__4::seq), 0, 1UL, 0, 0); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std9execution3__43parE,::cuda::std::execution::__4::par), 0, 1UL, 0, 0); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std9execution3__49par_unseqE,::cuda::std::execution::__4::par_unseq), 0, 1UL, 0, 0); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std9execution3__45unseqE,::cuda::std::execution::__4::unseq), 0, 1UL, 0, 0); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std6ranges3__45__cpo4swapE,::cuda::std::ranges::__4::__cpo::swap), 0, 1UL, 0, 0); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std6ranges3__45__cpo9iter_moveE,::cuda::std::ranges::__4::__cpo::iter_move), 0, 1UL, 0, 0); __cudaRegisterVariable(__T47, __shadow_var(_ZN45_INTERNAL_6d7068ed_16_cg_sass_check_cu_cg_red4cuda3std6ranges3__45__cpo7advanceE,::cuda::std::ranges::__4::__cpo::advance), 0, 1UL, 0, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
