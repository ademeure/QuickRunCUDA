#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "mma_sync_peak_proper.fatbin.c"
extern __attribute__((visibility("hidden"))) void __device_stub__Z13mma_bf16_ilp1Pfii(float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z13mma_bf16_ilp2Pfii(float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z13mma_bf16_ilp4Pfii(float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z13mma_bf16_ilp8Pfii(float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z14mma_bf16_ilp12Pfii(float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z14mma_bf16_ilp16Pfii(float *, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
__attribute__((visibility("hidden"))) void __device_stub__Z13mma_bf16_ilp1Pfii(float *__par0, int __par1, int __par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 12UL);__cudaLaunch(((char *)((void ( *)(float *, int, int))mma_bf16_ilp1)));}
# 35 "mma_sync_peak_proper.cu"
void mma_bf16_ilp1( float *__cuda_0,int __cuda_1,int __cuda_2)
# 35 "mma_sync_peak_proper.cu"
{__device_stub__Z13mma_bf16_ilp1Pfii( __cuda_0,__cuda_1,__cuda_2);
# 52 "mma_sync_peak_proper.cu"
}
# 1 "mma_sync_peak_proper.compute_103.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z13mma_bf16_ilp2Pfii( float *__par0,  int __par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 12UL); __cudaLaunch(((char *)((void ( *)(float *, int, int))mma_bf16_ilp2))); }
# 56 "mma_sync_peak_proper.cu"
void mma_bf16_ilp2( float *__cuda_0,int __cuda_1,int __cuda_2)
# 56 "mma_sync_peak_proper.cu"
{__device_stub__Z13mma_bf16_ilp2Pfii( __cuda_0,__cuda_1,__cuda_2);
# 78 "mma_sync_peak_proper.cu"
}
# 1 "mma_sync_peak_proper.compute_103.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z13mma_bf16_ilp4Pfii( float *__par0,  int __par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 12UL); __cudaLaunch(((char *)((void ( *)(float *, int, int))mma_bf16_ilp4))); }
# 82 "mma_sync_peak_proper.cu"
void mma_bf16_ilp4( float *__cuda_0,int __cuda_1,int __cuda_2)
# 82 "mma_sync_peak_proper.cu"
{__device_stub__Z13mma_bf16_ilp4Pfii( __cuda_0,__cuda_1,__cuda_2);
# 109 "mma_sync_peak_proper.cu"
}
# 1 "mma_sync_peak_proper.compute_103.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z13mma_bf16_ilp8Pfii( float *__par0,  int __par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 12UL); __cudaLaunch(((char *)((void ( *)(float *, int, int))mma_bf16_ilp8))); }
# 113 "mma_sync_peak_proper.cu"
void mma_bf16_ilp8( float *__cuda_0,int __cuda_1,int __cuda_2)
# 113 "mma_sync_peak_proper.cu"
{__device_stub__Z13mma_bf16_ilp8Pfii( __cuda_0,__cuda_1,__cuda_2);
# 140 "mma_sync_peak_proper.cu"
}
# 1 "mma_sync_peak_proper.compute_103.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z14mma_bf16_ilp12Pfii( float *__par0,  int __par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 12UL); __cudaLaunch(((char *)((void ( *)(float *, int, int))mma_bf16_ilp12))); }
# 144 "mma_sync_peak_proper.cu"
void mma_bf16_ilp12( float *__cuda_0,int __cuda_1,int __cuda_2)
# 144 "mma_sync_peak_proper.cu"
{__device_stub__Z14mma_bf16_ilp12Pfii( __cuda_0,__cuda_1,__cuda_2);
# 171 "mma_sync_peak_proper.cu"
}
# 1 "mma_sync_peak_proper.compute_103.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z14mma_bf16_ilp16Pfii( float *__par0,  int __par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 12UL); __cudaLaunch(((char *)((void ( *)(float *, int, int))mma_bf16_ilp16))); }
# 175 "mma_sync_peak_proper.cu"
void mma_bf16_ilp16( float *__cuda_0,int __cuda_1,int __cuda_2)
# 175 "mma_sync_peak_proper.cu"
{__device_stub__Z14mma_bf16_ilp16Pfii( __cuda_0,__cuda_1,__cuda_2);
# 202 "mma_sync_peak_proper.cu"
}
# 1 "mma_sync_peak_proper.compute_103.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T9) {  __nv_dummy_param_ref(__T9); __nv_save_fatbinhandle_for_managed_rt(__T9); __cudaRegisterEntry(__T9, ((void ( *)(float *, int, int))mma_bf16_ilp16), _Z14mma_bf16_ilp16Pfii, 1024); __cudaRegisterEntry(__T9, ((void ( *)(float *, int, int))mma_bf16_ilp12), _Z14mma_bf16_ilp12Pfii, 1024); __cudaRegisterEntry(__T9, ((void ( *)(float *, int, int))mma_bf16_ilp8), _Z13mma_bf16_ilp8Pfii, 1024); __cudaRegisterEntry(__T9, ((void ( *)(float *, int, int))mma_bf16_ilp4), _Z13mma_bf16_ilp4Pfii, 1024); __cudaRegisterEntry(__T9, ((void ( *)(float *, int, int))mma_bf16_ilp2), _Z13mma_bf16_ilp2Pfii, 1024); __cudaRegisterEntry(__T9, ((void ( *)(float *, int, int))mma_bf16_ilp1), _Z13mma_bf16_ilp1Pfii, 1024); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
