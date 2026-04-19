import sys
sys.path.insert(0, '/root/cutlass/examples/python/CuTeDSL/blackwell')
import cutlass
import sm103_dense_blockscaled_gemm_persistent as mod

# Test multiple sizes
shapes = [
    (4096, 4096, 6144, 1),
    (8192, 8192, 8192, 1),
    (8192, 8192, 16384, 1),
    (16384, 16384, 8192, 1),
    (16384, 16384, 16384, 1),
]

for mnkl in shapes:
    M, N, K, L = mnkl
    try:
        exec_us = mod.run(
            mnkl=mnkl,
            ab_dtype=cutlass.Float4E2M1FN,
            sf_dtype=cutlass.Float8E8M0FNU,
            sf_vec_size=16,  # NVF4
            c_dtype=cutlass.Float16,
            a_major="k", b_major="k", c_major="n",
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 4),
            use_tma_store=False,
            tolerance=0.1,
            warmup_iterations=5,
            iterations=20,
            skip_ref_check=True,
            use_cold_l2=False,
        )
        # exec_us is microseconds per iteration
        ops = 2 * M * N * K * L
        tflops = ops / (exec_us * 1e-6) / 1e12
        print(f"M={M:5d} N={N:5d} K={K:5d}  {exec_us:8.2f} us  {tflops:7.0f} TFLOPS  ({tflops/15000*100:.1f}% of 15 PF spec)")
    except Exception as e:
        print(f"M={M} N={N} K={K}: FAILED — {e}")
