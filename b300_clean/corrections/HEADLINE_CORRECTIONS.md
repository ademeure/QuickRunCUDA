# HEADLINE CORRECTIONS — 1-page TL;DR

**The 10 things that changed since `B300_TRUE_REFERENCE.md` (2026-04-18) was
written.** Everything below is sourced to a corrections file in this directory;
see `MASTER_INDEX.md` for the full mapping and `B300_TRUE_REFERENCE_v2_DRAFT.md`
for the proposed replacement.

---

## If you remember nothing else…

| # | What changed | Old claim | New claim | Source |
|---|---|---|---|---|
| 1 | **NVLink generation** | "NVLink v7" (CLAUDE.md memory, `13_pcie_system.md`) | **NVLink-5** (5th gen, Blackwell) | `NVLINK_PCIE_INCONSISTENCY_LOG.md` #1 |
| 2 | **NVLink spec denominator** | TRUE_REF used 757 GB/s/dir → "1.04× spec" | **900 GB/s/dir** spec → measured 86% read / 80% write | NVLINK_PCIE log #2 |
| 3 | **HBM read SoL** | V46 announced "98.5% NEW BEST" at 7.20 TB/s | 7.20 < 7.30 (NINJA) < 7.344 (TMA bulk) < 7.365 (LDG.E.128). V46 used 7.31 (empirical) as denominator instead of 7.672 (post-ECC spec). **Real read SoL: ~7.30 TB/s = 95% of 7672**. V46 is the new TMA-pipelined high (a real improvement over V33's 6.72) but **not** an architectural new ceiling. | HBM log #5 |
| 4 | **MUFU saturated peak** | M14/M16: "XU peak 47.8 G MUFU/s @ 99.5%" | 47.8 G is **1-chain LATENCY-bound rsqrt**, not the saturated pipe. **Saturated MUFU = 4.74 G/chip** (V41, EX2 outlier 9.22). M14/M16 mislabeled by ~10×. | MATH log #3 |
| 5 | **IADD3 pipe placement** | V9_INT_OPS_PIPES: "separate ALU pipe at 38 TOPS" | **IADD3 lives on the FMA pipe** (V40, 25-26 Glane/s = 67% of FMA SoL, same tier as FFMA). The "separate ALU" framing is wrong; LOP3/PRMT/IMUL are the real INT-bit/permute pipes (half rate). | INT log #A, COMPUTE log #E |
| 6 | **Dual-issue cap** | Catalog `a0bde33`/`f578755`: "FFMA + IADD3 free / 100% / 114 TOPS combined" | **Same-warp FFMA+ALU = 55%; warp-specialized = 74%** (V49/V50). Same-pipe-cluster ops contend; "free" only when the 2nd op slot fills FFMA bubbles (e.g. MUFU at 1/(4cy)). | COMPUTE log #F, M-SYNTH log #1 |
| 7 | **DSMEM aggregate BW** | V8/V10: "37 TB/s read / 11.8 TB/s write / writes 4-5× SLOWER than reads" | **All V8/V10 DSMEM TB/s numbers were DCE artifacts.** Real: read aggregate ≈ 40 GB/s/cluster; **writes ≈ 560 GB/s/cluster (writes are FASTER, not 4× slower).** Local/DSMEM ratio = 7.5×, not 0.8% and not 4.7×. Cluster=2 is 21% slower than cluster≥3 (single-GPC routing). | DSMEM log A/B/C |
| 8 | **Operand A vs B power impact** | CLAUDE memory: "A:B impact ~1:3, B dominant" / "A is FREE" | **3 different ratios depending on test geometry.** cuBLAS path: A>B 3:1 (because TMA multicasts B). Pure tcgen05: B>>A 15-30×. K=96 single-kernel: B>A 2.6×. The "A is FREE" rule holds only when B is constant; when B varies, A varying adds ~60 W marginal. | NVFP4 log "A vs B"; DEDUP log §2 |
| 9 | **3-source FFMA cap** | Headlines all use 2-source FFMA (75 TFLOPS = 97% peak) | **Realistic GEMM (3-distinct-source FFMA) caps at ~50 TFLOPS = 65% of peak** due to RF port pressure. 2-source recipe is not representative. | COMPUTE log #G; V8/V10 misc log #C |
| 10 | **NVFP4 cuBLAS ceiling** | Memory: "10.8 PF (72%)" | **11.07 PF plain Lt (73.7%) → 11.42 PF with cudaGraph BPG=16 (76.2%)** at K=38400. The 10.8 PF was a smaller-K shape. Memory note is stale. | NVFP4 log "MAJOR cuBLAS ceiling" |

---

## Honourable mentions (impact below the top 10)

- **Random data is up to 43% slower than zero data** for FP8 cuBLAS under power cap (entire data-dep table now in TRUE_REF row 68). Always cite zero AND random.
- **HBM3E spec is 7672 GB/s POST-ECC**, not 8 TB/s nominal. Three different denominators (7672/7.31/8.0) appeared across the corpus.
- **L2 BW must be labelled** with one of {kernel-effective ≈24 / wire-lts ≈13 / L1-amplified ≈30} TB/s. Bare numbers are meaningless.
- **L2 = 126 MB**, not 50 / 192 / 256 MB. Older numbers were unit/scope confusions.
- **TMEM = 60 TB/s read**, not 295 or 830 (those were DCE).
- **GPC layout: 8 GPCs (2×20 + 6×18 = 148)**, not 9 or 10; "spare SM" framing is wrong; "GPC-rows" should be "stride-16 columns".
- **`pipe_tensor.cycles_active` does NOT measure tcgen05 ops** — use `smsp__inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma_scope_2cta.sum`.
- **Branch divergence: 2-way is 2.57× (TRUE), not 1.09× (predicated)**. Same `V9_BRANCH_DIVERGENCE.md` has both tables; readers picked the wrong one.
- **REDUX is NOT 4× SHFL** — algorithm-level 2.34×, raw rate equal. The 4× was unsourced.
- **`nvidia-smi -lgc 2032` paradoxically pins to 1920** — well documented, repeated for emphasis.
- **B300 can stick at 1005 MHz under load with NO explicit lock** — silent failure mode; sample clock during long runs and use `-rgc` to recover.
- **Single-chain "1543 TFLOPS BF16" was over-counted; RETRACTED** (real ~570). Likewise FP8 mma.sync "7500-8200 TFLOPS" RETRACTED (SASS showed HMMA.16816 not 16832). "BF16 90.5% of 2500" RETRACTED (mislabeled — 23% of tcgen05 spec or 93.7% of legacy 616).
- **K-uniform-per-N "28% NVFP4 power saving" RETRACTED** (was background-process contamination; real 1-3%).
- **"4-slot pattern cache" model RETRACTED** for tcgen05 sub-tile dedup. Real model: sticky activation + (BF16 m128n128 only) two-half processing.
- **SASS-level FADD = FMUL = FFMA** (1 inst/SMSP/cy, 4.22 cy latency, same FMA pipe). FFMA only "wins" in FLOPS counted per instruction.

---

## Still under investigation (top 10 from MASTER_INDEX)

1. **Whether NINJA STG (`e75c7e1`) really hit 7.57 TB/s, or that was V8's TMA bulk store (`28211ce`) miscredited.** Two files attribute 7.57 to different paths.
2. **`__threadfence_system` true cost** — 1750 (08) vs 2870 (DSMEM) vs 3042 (V9) cy. 1.74× spread.
3. **Whether 1259 W transient peak is real or NVML aliasing.** Need kHz-rate external power probe.
4. **L2 atomic unit count** — TRUE_REF says 32 plateau; ATOMIC_REVERIFY_DEEP says ceiling could be much higher.
5. **Single-MMA cache depth** — 1 slot, 2 slots, or 1+alternation predictor? Two methodologies disagree.
6. **Why 2-pattern (ABAB) sub-tile is WORSE than 3-pattern (ABCABC)** for tcgen05 dedup.
7. **TMA pipeline-depth optimum** — V46 used 8, knee unknown. Sweep depth 2..16.
8. **`cuStreamWriteValue32` cost** — 0.45 µs (memory) or 2.47 µs (catalog)? Decompose host-call vs full pair.
9. **LDS 32-way bank-conflict cost** in 4 different regimes (1×, 2×, 5.7×, 8.2×, 8.81×) — single-warp vs multi-warp matrix never measured uniformly.
10. **Cooperative-grid SM mapping** — never measured.

(Full list of 20 in `MASTER_INDEX.md` §4.)
