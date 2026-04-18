# B300 Integer + Bit-Ops Throughput

**GPU**: NVIDIA B300 SXM6, sm_103a, 148 SMs, 2.032 GHz boost (1.92 GHz when `-lgc`-pinned)
**Confidence**: HIGH for ALU-pipe ops (cross-checked against `pipe_alu = 2.00 w-inst/SM/cy` ncu cap and against catalog lines 13104-13125 + 196-200). MEDIUM for shfl/ballot/match (clock64-bracketed only). LOW for the EXTENDED_FINDINGS Gops/s rollups (see retired list).

---

## TL;DR ops/SM/cycle table

All numbers are SASS warp-instructions/SM/cycle. Multiply by `32 lanes × 148 SMs × 2.032 GHz` = **9.628** to get chip Gops/s; multiply by `2.032` to get Gops/s/SM.

| Op | SASS | Pipe | w-inst/SM/cy | Gops/s/SM | Chip TIOPS | Latency (cy) |
|---|---|---|---:|---:|---:|---:|
| **IADD3** (3-input add) | `IADD3` | alu (+ split fmaH) | **2.46** | 158 | **23.4** | 4 |
| **IMAD / IMUL / IMAD.X** (32-bit) | `IMAD` | fmaH | **2.00** | 128 | 19.0 | 4 |
| **IMAD.HI.U32** (high half) | `IMAD.HI.U32` | fmaH | **1.00** | 64 | 9.5 | 13 |
| `mul.wide.u32` (u32×u32→u64) | `IMAD.WIDE` | fmaH | 2.00 | 128 | 19.0 | 4 |
| **LOP3 / LOP / LOP32I** (and/or/xor/3-LUT) | `LOP3.LUT` | alu | **2.00** | 128 | 19.0 | 4 |
| **PRMT** (byte permute) | `PRMT` | alu | 2.00 | 128 | 19.0 | 4 |
| **SHF / SHL / SHR** (incl. funnel) | `SHF.{L,R}.{W,U,S,HI}` | alu | 2.00 | 128 | 19.0 | 4 |
| `BFE.u32` | `SHF.R + SGXT` (2 SASS) | alu | 1.00 | 64 | 9.5 | 8 |
| `BFI.b32` | folds to `LOP3.LUT` | alu | 2.00 | 128 | 19.0 | 4 |
| `bfind` / `clz` / `ffs` | `FLO.U32` (+IADD3 for offset) | **xu** | 0.50 | 32 | 4.7 | 29 (clz) / 48 (ffs) |
| `popc.b32` | `POPC` | **xu** | 0.50 | 32 | 4.7 | 24 |
| `brev.b32` | `BREV` | **xu** | 0.50 | 32 | 4.7 | 24 |
| **ISETP / FSETP / setp.lt.{u32,s32,f32}** | `ISETP` / `FSETP` | alu | 2.00 | 128 | 19.0 | 4 |
| `setp + selp` chain | `ISETP + SEL` (2 SASS) | alu | 1.00 | 64 | 9.5 | 8 |
| `min.u32` / `max.u32` | `IMNMX` (1 SASS) | alu | 2.00 | 128 | 19.0 | 4 |
| `dp4a.s32.s32` (4×i8 MAC) | `IDP4A` | fmaH | ~2.00 | 41 (×4 MAC = 164 op) | **~49 TOPS** | — |
| `vabsdiff4.u32` | `VABSDIFF4` | alu | ~2.00 | — | 6.15 Gops chip | — |
| `usad` / `vabsdiff.s32` | `PRMT + SHF.R.S32.HI` (2 SASS) | alu | 1.00 | 64 | 9.5 | — |
| **udiv (32-bit, runtime divisor)** | NR reciprocal chain (~12 SASS) | mixed | ~0.16 | ~10 | **~1.5** | — |
| **u64 add** | `IADD3 + IMAD.X` (2 SASS) | alu+fmaH | 1.00 | 64 | 9.5 | 6 |
| u64 and/or/xor | 2× `LOP3.LUT` | alu | 1.00 | 32 | 4.7 | — |
| u64 mul.lo | `IMAD + IMAD.WIDE + IADD3` ×3 | mostly fmaH | ~0.19 | ~12 | 1.8 | — |
| u64 shl/shr | `SHF.L.U64.HI + SHF.L.U32 + …` (3 SASS) | alu | 0.5 | ~16 | 2.4 | — |
| **shfl.bfly / .idx / .up / .down** (32-bit) | `SHFL.{BFLY,IDX,UP,DOWN}` | **lsu** | 1.00 | 32 | 4.7 | 24 |
| `shfl.idx` with const-0 src (broadcast) | UIMOV / R2UR (uniform path) | uniform | — | — | ~free | 2 |
| **`vote.sync.ballot.b32`** | `VOTE.ANY` (1 SASS, b32 mask) | alu | 2.00 | 128 | **7.32 Gops chip** | 22 |
| `vote.{all,any,uni}.pred` | `VOTE + SELP` (2 SASS) | alu | 1.00 | 64 | 3.32 Gops chip | 28 |
| `__match_all_sync` | single-pred check | adu | — | — | — | 35 |
| **`__match_any_sync`** | N×N intra-warp compare | **adu** (slow) | ~0.05 | — | — | **375** |
| `redux.sync.add.u32` | `REDUX.SUM` (+ minor IMAD) | adu | 0.50 | — | 3.11 Gops chip | 45 |
| `redux.sync.min.u32` | `CREDUX.MIN + IMAD.U32` (2 SASS) | alu+fmaH | 1.92 | — | 6.92 Gops chip | 19 |

Pipe identification cross-checked against catalog §2.3, §2.4, §3 (lines 196-201, 246-280, 339-413), §13100-13125, and §26 (line 2117).

---

## Key facts (validated against catalog + investigations)

1. **All "fast" integer ops cap at 2 warp-inst/SM/cy on pipe_alu = 128 thread-ops/SM/cy = ~19 TIOPS chip.** This includes LOP3, PRMT, SHF, IMAD, IMUL, IADD3, ISETP, FSETP, IMNMX, BFI. Every one of them lives behind the same `pipe_alu = 2.00` (or `pipe_fmaheavy = 2.00` for the multiplier-using ones) ceiling. The "everything converges at 17.8 TOps/s" claim from EXTENDED_FINDINGS is correct in shape (all hit ~18-19 TIOPS) and consistent with catalog line 13105-13110.
2. **IADD3 wins by ~25%** because the compiler dual-issues IADD3 across alu AND fmaheavy when there is dispatch headroom (catalog line 13105: 79 SASS/cy/SM = 2.46 w-inst/cy/SM, vs the 64 SASS/cy/SM = 2.00 w-inst/cy/SM for everything else). One IADD3 is worth two adds, so logical adds/SM/cy ≈ 128.
3. **IMUL ≈ IMAD ≈ IMAD.WIDE.** Same SASS opcode internally; same 2.00/SM/cy rate. The "imul = 57 K Gops/s" line in some bench output is wrong (would imply 6 SASS/SM/cy, exceeds dispatch cap of 4). It is DCE — see retirement list.
4. **`mul.hi.u32` is half-rate** (32 SASS/SM/cy = 1.00 w-inst, 13-cy latency). Not the same as `mul.lo`.
5. **POPC, BREV, CLZ all live on the XU pipe at 0.5/SM/cy (4×–6× slower than other bit ops).** The 4-cy "POPC = 24 cy" / "BREV = 24 cy" / "CLZ = 29 cy" latencies in the latency table are XU pipeline depth, not throughput. Throughput of all three is ~4.7 TIOPS chip. UBREV/UFLO on the uniform pipe are 4-12 cy and avoid the XU bottleneck if the compiler emits them.
6. **`udiv` (runtime divisor) is ~10× slower than IMAD** because nvcc lowers it to a Newton-Raphson reciprocal chain (~12 SASS/op including IMAD.HI). 1.5–1.8 TIOPS chip is consistent. Compile-time divisors are folded to a multiply.
7. **Shuffle is on pipe_lsu, not pipe_alu.** Capped at 32 SASS/SM/cy = 4.7 GOps chip (catalog line 405, 2123-2124). Distance/index/lane changes don't move the rate. The "5.8 TOps/s" EXTENDED finding is in line with this once you account for ILP.
8. **Ballot beats other vote/match** (one SASS, b32 result). 7.32 Gops chip (catalog line 2121) ≈ 2.3× faster than `vote.{all,any,uni}.pred` (which need a SELP). The "ballot 3.2 TOps/s" in EXTENDED is the 2-SASS fallback path; real `__ballot_sync` at full ILP hits 7.3 Gops.
9. **`__match_any_sync` is 20× slower than every other warp primitive** (375 cy latency, ~0.15 TOps/s throughput). Catalog line 1367 + EXTENDED line 120 agree. Avoid in hot loops.
10. **`shfl.idx` with literal `0` src DCEs to a uniform-pipe broadcast** (1.9 cy, R2UR). The "85 K Gops/s" figure is the uniform-broadcast fast-path, NOT the SHFL pipe — useful but not a real shuffle. Other constant indices still emit SHFL.IDX at the normal rate.
11. **`setp.lt.s32` ≈ `setp.lt.u32` ≈ `setp.lt.f32` at SASS level** — all one ISETP/FSETP, both on pipe_alu, 2.00/SM/cy. The "setp.lt.f32 is 2× faster" reading from `setp_compare.cu` is loop-overhead artifact: the FP loop's update (`a*1.0001f+0.0001f`) maps to a 1-cycle FFMA, while the int loop's update (`a*7+1`) is an IMAD on the same fmaH pipe as ISETP — so the int loop has artificial pipe contention. Same setp speed, different loop body cost.
12. **`setp.lt.s64` is ~2× setp.s32** because s64 compare lowers to `ISETP.EX + ISETP` (2 SASS chained on alu). Matches the catalog §2.4 u64 pattern.

---

## Co-issue map (which ops can run concurrently)

| Pair | Concurrency | Notes |
|---|---|---|
| LOP3 / PRMT / SHF + FFMA | **near-perfect** (alu + fmaH+L disjoint) | Bit-ops are free alongside FP32 FFMA. Catalog line 14536. |
| LOP3 + IMAD | partial (alu + fmaH) | u ≈ 1.61 (line 1797). |
| IADD3 + FFMA scalar | excellent at 2:1 ratio (1 IADD3 per 2 FFMA2) | catalog §17. |
| IMUL + FFMA | bad (same fmaH pipe) | each IMAD blocks ~6-8 cy of FFMA. |
| LOP3 + LOP3 | none (both alu) | u ≈ 1.0 → no speedup, just queue. |
| SHFL + ALU | yes (lsu separate from alu/fma) | shfl reductions interleave well with arithmetic. |

---

## Pitfall summary — DCE-prone tests in this category

The compiler eliminates the following loop bodies unless explicit anti-DCE is used:

- `iadd / and / xor` with constant operand: collapses to a single SASS or zero. CONSOLIDATED_FINDINGS.md line 324 explicitly notes `__brev`, `__clz`, `__ffs` were DCE'd. Defense: runtime kernel-arg operands + final XOR-fold + `if(impossible) store`.
- `popc`, `brev`, `clz` chained: same. Catalog line 4187 documents that "sequences of XORs with constant masks fold to zero or a single XOR".
- `shfl.idx src=0`: NOT a SHFL — it's a uniform broadcast. Always check SASS for `SHFL.IDX` before reporting throughput.
- Self-dep chains (`a = a op a`) with constant: folded to `MOV` or removed. The `bench_iadd3_test.cu` `MODE 0` form `a + b + (a^b)` is the right pattern (catalog line 2767).
- `imul = 57 K Gops/s` from a sub-agent: this is faster than dispatch cap allows (would be ~6 SASS/SM/cy). Almost certainly DCE'd to `MOV` or const. Real IMUL is 2.00/SM/cy = 19 TIOPS.

---

## RETIRED claims (do not cite)

| Claim | Source | Why retired |
|---|---|---|
| **"imul: 57 K Gops/s"** | sub-agent on `bench_imul_*` | Exceeds dispatch cap (4.00 w-inst/cy/SM total). Almost certainly DCE'd self-op. Real IMUL = 2.00/SM/cy = ~19 TIOPS chip. |
| **"shfl.idx with constant src: 85 K Gops/s"** | shfl_bw.cu test | Compiler recognizes broadcast (src=0) and emits `R2UR/UIMOV` uniform-pipe path (1.9 cy). NOT through the SHFL pipe. Real SHFL at 32/SM/cy = 4.7 Gops chip. |
| **"setp.lt.f32 is 2× faster than setp.lt.s32"** | setp_compare.cu raw timings | Loop-update body differs (FFMA vs IMAD) — same setp speed, different anti-loop costs. ISETP and FSETP are both pipe_alu @ 2.00. |
| **"vmad u32: 11.7 K Gops/s"** | exotic_int_ops.cu | The PTX `vmad.u32.u32.u32 d, a, b, 0` lowers to a single IMAD on B300 (no native vector multiply path). Number is just the IMAD rate; the "vmad" framing implies a separate pipe that doesn't exist. |
| **"umulhi: 6.3 K Gops/s"** | exotic_int_ops.cu | Misreported as a "specialized HW path." It is `IMAD.HI.U32` at 32 SASS/SM/cy = 1.00 w-inst (catalog line 252) — half the rate of regular `mul.lo`. Real chip rate ≈ 9.5 TIOPS, not 6.3 K. The 6.3 K figure may be Gops chip (consistent within ~30%) but mislabeled as Gops/s/SM. |
| **"LOP3 / PRMT / USAD / SHF all converge at 17.8 TOps/s"** | EXTENDED_FINDINGS line — partly correct | Direction is right (all live on pipe_alu @ 2.00) but USAD = `vabsdiff` is **2 SASS** so it runs at half rate (9.5 TIOPS, not 19). Don't lump USAD with LOP3/PRMT/SHF. |
| **"IMAD/IMUL takes 6-8 cycles"** | catalog line 4551 | This is wrong — IMAD latency is **4 cy** (catalog line 2021), throughput is 2.00/SM/cy. The "6-8 cy" was inferred from a contended FFMA+IMAD test where IMAD waits for FFMA on the shared fmaH pipe. Solo IMAD = 4 cy. |
| **"imad: 18 Gops/s"** as a per-SM rate | EXTENDED line 91 | Number is **chip-wide TIOPS** (= 18.2), not Gops/s/SM. EXTENDED's column header is misleading. Per-SM rate is 128 logical IMAD/SM/cy = 260 GOps/s/SM at 2032 MHz. |

---

## SASS-verified anchors

- `bench_iadd3_test.cu` MODE 1 (8 chains, 1024 unrolled): `IADD3 + LOP3` per logical iter, 4.2 cy/SASS (catalog 2773).
- `bench_imad_peak.cu` (8 chains, 128 INNER × 100 OUTER): emits pure IMAD chain, hits 2.00 w-inst/SM/cy.
- `bench_bitops_audit.cu`: clock64-bracketed; OP=4 (BFE) and OP=5 (BFI) compile to 2 SASS each per the catalog mapping; OP=8 (SHF) and OP=6 (PRMT) to 1 SASS.
- `int_ops.cu` and `exotic_int_ops.cu` use runtime kernel args + a final XOR-fold and impossible store — correct anti-DCE pattern. Their per-op Gops/s numbers are sanity-checked: divide by `(blocks × threads × 4)` to get inst/thread/sec, then divide by clock to get inst/cy/thread.

---

## Practical guidance

- Use **IADD3 + LOP3** as the primary "do bit math fast" pair — they fully co-issue with FFMA and reach 128 logical ops/SM/cy each.
- **Replace `popc/brev/clz` in inner loops** with PRMT/LOP3 tricks if possible — they are 4× slower (XU pipe).
- **Avoid `__match_any_sync`** entirely in hot paths (375 cy = 80× a regular shuffle).
- **`__ballot_sync` is the cheapest collective** at 22 cy / 7.3 Gops chip; prefer over `__any_sync`/`__all_sync`.
- **`shfl_idx` broadcast (src=0)** truly is free (UIMOV path); use it for warp-uniform broadcasts.
- **Avoid runtime `udiv`**; replace with multiply-by-reciprocal at compile time, or use `__umulhi` with a precomputed magic constant.
- **u64 ops cost ~2-3× their u32 cousins** — use 32-bit indexing where the data permits.
- **All setp variants (s32/u32/f32) are equally fast.** Pick whichever is semantically right.

---

## Files referenced

- `/root/github/QuickRunCUDA/investigations/int_ops.cu`
- `/root/github/QuickRunCUDA/investigations/exotic_int_ops.cu`
- `/root/github/QuickRunCUDA/investigations/setp_compare.cu`
- `/root/github/QuickRunCUDA/investigations/shfl_bw.cu`
- `/root/github/QuickRunCUDA/investigations/shfl_cost.cu`
- `/root/github/QuickRunCUDA/investigations/CONSOLIDATED_FINDINGS.md` (lines 219-233, 318-324)
- `/root/github/QuickRunCUDA/investigations/EXTENDED_FINDINGS.md` (lines 90-92, 117-120)
- `/root/github/QuickRunCUDA/B300_PIPE_CATALOG.md` lines 196-201, 246-280, 294-413, 596-630, 1340-1376, 2017-2034, 2117-2131, 4540-4570, 8851, 11040-11055, 13100-13127
- `/root/github/QuickRunCUDA/tests/bench_bitops_audit.cu`
- `/root/github/QuickRunCUDA/tests/bench_iadd3_test.cu`
- `/root/github/QuickRunCUDA/tests/bench_imad_peak.cu`
- `/root/github/QuickRunCUDA/tests/bench_lop3_pure.cu`
- `/root/github/QuickRunCUDA/tests/bench_imul_hi.cu`
- `/root/github/QuickRunCUDA/tests/bench_shfl_audit.cu`
- `/root/github/QuickRunCUDA/tests/bench_shfl_mask.cu`
- `/root/github/QuickRunCUDA/tests/bench_shfl_vs_redux.cu`
- `/root/github/QuickRunCUDA/tests/bench_triple_prmt.cu`
