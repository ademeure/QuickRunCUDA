import sys
sys.path.insert(0, '/root/cutlass/examples/python/CuTeDSL/blackwell')
import cutlass
import sm103_dense_blockscaled_gemm_persistent as mod
print("# Full mma_tiler × cluster sweep (NVF4, sf_dtype=e8m0)")
configs = [
    # (mma_M, mma_N, cluster_M, cluster_N)
    (128, 128, 1, 1), (128, 128, 1, 2), (128, 128, 2, 2), (128, 128, 2, 4),
    (128, 256, 1, 1), (128, 256, 1, 2), (128, 256, 2, 2), (128, 256, 2, 4),
    (256, 128, 2, 1), (256, 128, 2, 2), (256, 128, 2, 4),
    (256, 256, 2, 1), (256, 256, 2, 2), (256, 256, 2, 4), (256, 256, 4, 4),
]
M, N, K = 16384, 16384, 16384
for (mm, mn, cm, cn) in configs:
    try:
        exec_us = mod.run(
            mnkl=(M, N, K, 1),
            ab_dtype=cutlass.Float4E2M1FN, sf_dtype=cutlass.Float8E8M0FNU, sf_vec_size=16,
            c_dtype=cutlass.Float16, a_major="k", b_major="k", c_major="n",
            mma_tiler_mn=(mm, mn), cluster_shape_mn=(cm, cn),
            use_tma_store=False, tolerance=0.1,
            warmup_iterations=3, iterations=10, skip_ref_check=True, use_cold_l2=False,
        )
        ops = 2 * M * N * K
        tflops = ops / (exec_us * 1e-6) / 1e12
        print(f"  mma=({mm},{mn}) cl=({cm},{cn})  {exec_us:7.1f}us  {tflops:5.0f} TF  ({tflops/15000*100:4.1f}% of 15PF)")
    except Exception as e:
        print(f"  mma=({mm},{mn}) cl=({cm},{cn})  FAIL: {str(e)[:60]}")
