# 17 — NVRTC + Module Load + JIT Cache — CORRECTED

Source: `b300_clean/17_nvrtc_module.md` (original retained verbatim).
Corrections derived 2026-04-22 from cross-file audit of CLAUDE.md, V41-V48,
06_tensor_cores, B300_TRUE_REFERENCE, MEMORY (`feedback_nvrtc_fast_math_ftz`).

Confidence markers unchanged from original (HIGH / MED / LOW).

---

## A. Inherited HIGH-confidence facts (still valid)

These survive the cross-check:

1. NVRTC compile cost ladder (5.4 / 5.8 / 23 ms tiny / medium / 5000-FMA).
2. Cold-process +11 ms framework init; +240 ms `cuCtxCreate` (per
   B300_TRUE_REFERENCE §4).
3. NVRTC rejects bare `-O0..-O3`. Use `--ptxas-options=` for ptxas opts.
4. `cuModuleLoadData(cubin)` ≈ 10 µs flat across 5-80 KB tested range.
5. PTX-JIT load scales with PTX size, up to 155× cubin-load at 5000 FMA.
6. `cuModuleGetFunction` ≈ 39 ns; `cuLibraryGetKernel` ≈ 13 ns (per CUDA 12+
   path).
7. VMM: 2 MB granularity floor, `cuMemCreate` dominates (~0.5 µs/MB beyond
   floor), other steps <1 µs.
8. NVTX (no profiler) and `cudaGetLastError` both ≈ 19-20 ns.

---

## B. CONFIRMED — `-use_fast_math` is on by default in QuickRunCUDA

Original sec 2 line 54 mentions this in a parenthetical only. The MEMORY note
`feedback_nvrtc_fast_math_ftz` and `b300_clean/14_math_intrinsics.md` line 87
both confirm it has CONCRETE numerical consequences:

- All FFMA emit as **`FFMA.FTZ`** under the harness; subnormals are flushed
  to zero on the FFMA pipe.
- All `__fdividef`, `1.0f/x`, `sqrtf` get the approximate path (vs the
  ~243 cy `div.rn.f32` of standalone `nvcc`).
- This is ALSO why `tests/` kernels measure rsqrt at MUFU rates — they get
  the approx path automatically.

Implication for any subnormal-handling test you might run via QuickRunCUDA:
**you cannot measure non-FTZ behaviour via QuickRunCUDA without first patching
out `--use_fast_math`** in `utils/cuda_helper.h:227`. (Standalone `nvcc`
builds in `b300_clean/M7_V5_SYNTHESIS.md` D1 confirmed B300 supports full-speed
subnormal FFMA at 4.11 cy when `-ftz=false` is used.)

ACTION: 17 sec 2 should call this out as a NUMERICAL gotcha, not just a
side-note. Add a HIGH-confidence row in the option-flags table and elevate
the note above the table.

---

## C. CUDA 13.2 PTX-acceptance: NVRTC is more permissive than static ptxas

Original sec 8 retirement row "tcgen05 PTX rejected by NVRTC" is correct in
direction (NVRTC accepts; static ptxas rejects on sm_103). Adding more
catalog data:

| PTX form | NVRTC sm_103a | Static ptxas (CUDA 13.2) | Source |
|---|:---:|:---:|---|
| `tcgen05.mma` | accepts | rejects | 06_tensor_cores §6 line 93 |
| `tcgen05.alloc` | accepts | rejects | 06_tensor_cores §6 line 93 |
| `cvt.rn.satfinite.e2m1x4.f32` | **REJECTS** | rejects | V41_V48_FINDINGS l.61-62 |
| `cvt.scalefactor` variants | (not yet tested in catalog) | (not yet tested) | needs V8/V9 follow-up |
| `cvt.rz/.rm/.rp.e4m3x2.f32` | rejects | rejects | CURIOSITY_LIST_V7 H1 |

Pattern: NVRTC inherits the same PTX 8.7 rejections as static ptxas for the
narrow x4 forms — so the NVRTC-vs-ptxas gap is NOT universal. Specifically:

- **`tcgen05.*`** — NVRTC > ptxas (NVRTC accepts).
- **`cvt.rn.satfinite.e2m1x4.f32`** — NVRTC = ptxas (BOTH reject).
- **Narrow cvt non-`.rn` rounding modes** — NVRTC = ptxas (BOTH reject).

ACTION: 17 sec 8 row 8 should be REWORDED to remove the implication that
NVRTC always accepts more than ptxas. Specify it as "NVRTC accepts
`tcgen05.*` PTX even where static ptxas in CUDA 13.2 rejects — but does NOT
help on narrow-cvt PTX bugs."

---

## D. Init/main kernel arg conflict — QuickRunCUDA harness quirk

MEMORY note `feedback_compute_pipe_methodology` documents a real bug class:
QuickRunCUDA passes the same `-0/-1/-2` ints to BOTH the optional `init`
kernel AND the timed `kernel`. If you reuse arg slot 0 as `iters` for the
main kernel, the init kernel's "use" of `iters` may be invalid or destructive.

This is NOT in the 17 catalog at all — it is a documented user-tripping pit
that affects every benchmark using `-i`. Recommend adding a SHORT subsection
under sec 2 / sec 7:

```
### NVRTC harness quirk: shared arg0/arg1/arg2 between init and main

QuickRunCUDA passes the same -0/-1/-2 to both the init and timed kernel
(see QuickRunCUDA.cpp). Workaround: pack init parameters into a single
int and bit-shift extract inside the init kernel; reserve -0 (typically
'iters') for the main kernel.
```

---

## E. RETRACTIONS

1. **Sec 8 row 8 ("tcgen05 PTX rejected by NVRTC — Wrong direction")** is
   directionally correct but OVER-STATED. It implies NVRTC always > ptxas;
   actually NVRTC is more permissive ONLY for `tcgen05.*` PTX. Rewrite as
   per §C above.

2. **Sec 2 parenthetical on `--use_fast_math`** is too easy to miss. The
   numerical implication (.FTZ on every FFMA, fast-path on every reciprocal)
   deserves prominence — currently a reader who didn't open
   `cuda_helper.h:227` themselves would not know. PROMOTE.

3. No retractions for compile-cost numbers (sec 1), VMM costs (sec 4),
   NVTX/error-check costs (sec 5/6), or end-to-end JIT pipeline summary
   (sec 7). All independently confirmed.

---

## F. UNRESOLVED

1. **`cvt.scalefactor.*` PTX forms** — V6 H3 hypothesized these are the
   replacement for the rejected `cvt.rn.satfinite.e2m1x4.f32`. NOT tested
   end-to-end via NVRTC; sample compile would close this loop in <30 min.

2. **NVRTC vs nvcc cubin-equivalence** — sec 9 open Q #3. Worth a
   `cuobjdump --dump-sass` diff for one production kernel. We trust both
   paths run the same ALU recipes today (because peak FFMA tests via
   QuickRunCUDA hit the same 74.6 TF as standalone `nvcc` builds in
   `b300_clean/04_fp32_peak.md`), but no formal sass diff exists in the
   catalog.

3. **`cuLibrary*` direct measurement** — sec 9 open Q #1 unchanged. Catalog
   relies on a single line of older catalog as source for the 6.5× speedup
   claim. Re-measure with current driver.

4. **`-G` runtime impact** — sec 9 open Q #4 unchanged. Only compile-time
   and cubin-size impact were measured; runtime slowdown not quantified.

5. **Whether `--use_fast_math` affects narrow-cvt rounding** — does
   `cvt.rn.satfinite.e4m3x2.f32` get re-routed under fast_math? Probably no
   (rounding mode is explicit in PTX), but unverified.

6. **MEMORY note on QuickRunCUDA init/main arg sharing** is a well-known
   harness quirk; NEVER documented in 17. Add a "Harness quirks" subsection.

7. **Whether `cuLibraryLoadData` sees the same 6.5× when the library
   contains a tcgen05 PTX module** — interaction of CUDA-12-API path with
   13.2-period PTX bug not tested.
