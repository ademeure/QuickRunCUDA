import sys
sys.path.insert(0, '/root/cutlass/examples/python/CuTeDSL/blackwell')
import cutlass
import sm103_dense_blockscaled_gemm_persistent as mod

# Sweep variants — sf_dtype back to default Float8E8M0FNU per script default
print("# NVF4 (sf_vec=16, sf_dtype=e8m0) — testing for K=96-multiple ULTRA path peak")
configs = [
    # (M, N, K, mma, cluster, use_tma_store)
    (16384, 16384, 16384, (256, 256), (2, 4), False),
    (16384, 16384, 16384, (256, 256), (2, 4), True),
    (16384, 16384, 16384, (256, 256), (2, 2), False),
    (16384, 16384, 16384, (128, 256), (2, 4), False),
    (8192, 65536, 16384, (256, 256), (2, 4), False),
    (32768, 32768, 16384, (256, 256), (2, 4), False),
]
for (M, N, K, mma, cluster, tma_st) in configs:
    try:
        exec_us = mod.run(
            mnkl=(M, N, K, 1),
            ab_dtype=cutlass.Float4E2M1FN,
            sf_dtype=cutlass.Float8E8M0FNU,  # default NVF4 scale
            sf_vec_size=16,
            c_dtype=cutlass.Float16,
            a_major="k", b_major="k", c_major="n",
            mma_tiler_mn=mma,
            cluster_shape_mn=cluster,
            use_tma_store=tma_st,
            tolerance=0.1,
            warmup_iterations=5,
            iterations=20,
            skip_ref_check=True,
            use_cold_l2=False,
        )
        ops = 2 * M * N * K
        tflops = ops / (exec_us * 1e-6) / 1e12
        print(f"M={M:5d} N={N:5d} K={K:5d}  tile={mma} cl={cluster} tma_st={tma_st}  {exec_us:7.1f}us  {tflops:5.0f} TF  ({tflops/15000*100:.1f}% of 15PF)")
    except Exception as e:
        print(f"M={M} N={N} K={K} tile={mma} cl={cluster} tma_st={tma_st}: FAIL — {str(e)[:60]}")
