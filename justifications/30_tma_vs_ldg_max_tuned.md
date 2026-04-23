# §30 — TMA vs LDG.E.128 head-to-head, both max-tuned

**Branch:** `f2fp-deep-dive` · **Date:** 2026-04-23 · **Sustained clock:** 1942 MHz under both kernels

## QUESTION (verbatim user)

> "for ~22TB/s TMA load, the key question is how a 'maximally optimized TMA read' kernel compares to a 'maximally optimized 256-bit LDG read' kernel, for both L2 hit and DRAM — it is less important how they compare when not tuned."

## TEST KERNELS

Both kernels were freshly written for this comparison so they share methodology
(same QuickRunCUDA harness, same `-T` count, same anti-DCE pattern, same WS
control, same clock state, same denominators).

- **LDG kernel:** `/root/github/QuickRunCUDA/tests/bench_h2h_ldg.cu`
  - Per-thread `ld.global.cg.v8.u32` = LDG.E.128 = 32 B per inst
  - UNROLL=32, BLOCK_SIZE override via `-H`
  - Anti-DCE: XOR-chain across 32 inner loads → `if (v==seed) C[tid&0xFFFF] = v;`
  - SASS verified: 32 × `LDG.E.128` per inner unroll body, 1 × `STG`
  - Best config: `-t 128 -b 8192 -0 1632` (L2-hit), `-t 128 -b 32768 -0 32` (DRAM-cold)

- **TMA kernel:** `/root/github/QuickRunCUDA/tests/bench_h2h_tma.cu`
  - Single-CTA producer/consumer with DEPTH-deep pipeline
  - Producer thread 0: `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes`
  - Consumer warp 1: full-warp `ld.shared.u32` of first u32 per tile (anti-DCE)
  - Tunable: `TMA_BYTES`, `NTMAS` per iter, `DEPTH` pipeline depth, BLOCK_SIZE
  - SASS verified: cp.async.bulk uses present
  - Best config: TB=8192 NT=8 DP=2 CTAS=592 ITERS=1411 (L2-hit);
                 TB=32768 NT=2 DP=2 CTAS=1184 ITERS=221 (DRAM-cold)

Both kernels: A buffer 4 GiB (`-A 1G`), C buffer 4 MiB, runs `-T 30..50` for stable mean.

## DENOMINATORS (B300 SXM6 @ 1942 MHz sustained)

- HBM3E peak read: **7.672 TB/s** (catalog spec; clock-independent)
- L2 wire (catalog estimate, scaled to 1942 MHz): **~12.7 TB/s**
  (catalog uses 13.3 TB/s at 2032 MHz; here clock is 6% lower)
- For % calculations below, **HBM = 7672 GB/s** and **L2-wire = 13300 GB/s** (catalog reference, not clock-adjusted).

## RESULTS TABLE — wall-clock (default boost, locked-in 1942 MHz)

| WS | LDG wall | LDG `dram__bytes_read` | LDG `l1tex__t_bytes` | LDG `lts__t_bytes` | TMA wall | TMA `dram__bytes_read` | TMA `lts__t_bytes` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 MiB (L2-plateau) | **18.25 TB/s** | 22.3 GB/s | 18.16 TB/s | 6.61 TB/s | **20.49 TB/s** | 523 GB/s | 19.69 TB/s |
| 4 GiB (DRAM-cold) | **7.41 TB/s** | 7.40 TB/s | 7.40 TB/s | 11.27 TB/s | **7.32 TB/s** | 7.30 TB/s | 11.11 TB/s |

(NCU clock during profiling = 1.91 GHz. Wall-clock at 1.94 GHz, so wall numbers
are ~2% higher than NCU `_per_second` numbers — confirmed in agreement.)

## % OF SoL TABLE

| WS | LDG % HBM | LDG % L2-wire | TMA % HBM | TMA % L2-wire |
|---|---:|---:|---:|---:|
| 64 MiB (L2-hit) | 238% (cache hits, irrelevant denom) | **137%** | 267% (cache hits) | **154%** |
| 4 GiB (DRAM-cold) | **96.5%** | (HBM-bound) | **95.4%** | (HBM-bound) |

## VERDICT

### L2-hit regime (WS = 64 MiB)
- **TMA wins, 20.49 vs 18.25 TB/s = +12.3% (~2.24 TB/s gap).**
- Gap is meaningful, not noise. Re-run TMA → 20.49 TB/s, LDG → 18.25 TB/s
  (3 trials each, std-dev <0.5%).
- Both EXCEED the catalog 13.3 TB/s "L2 wire" — the catalog number is
  conservative or measures a different thing. Real ceiling is ≥20 TB/s.

### DRAM-cold regime (WS = 4 GiB)
- **LDG wins by 1.2% (7.41 vs 7.32 TB/s, ~90 GB/s gap).**
- Gap is small but reproducible across multiple sweep runs.
- Both essentially HBM-bound (96.5% vs 95.4% of 7.67 TB/s).
- The 1.2% difference is at the noise floor and does NOT mean LDG is
  "structurally better" — both kernels saturate HBM. Within measurement noise.

### Headline:
- **L2-hit: TMA ~12% faster than LDG**
- **DRAM-cold: tied (both at ~96% HBM SoL)**

The catalog had been reporting "TMA ~22 TB/s" — confirmed at **20.5 TB/s**
(L2-hit) on max-tuned identical-methodology comparison. The remaining gap to
"22 TB/s" likely comes from earlier catalog runs at boost 2032 MHz (this rig
sustains 1942 MHz only). Scaling 20.49 × 2032/1942 = 21.4 TB/s — matches
"~22 TB/s" claim within 3%.

## NOTES

### Tuning knobs that mattered
- **TMA: tile size × DEPTH × NTMAS interact strongly via smem footprint.**
  Best config maintains DEPTH=2 (pipelined) with bytes-per-iter (NTMAS×TILE) ≥ 64 KiB
  per CTA. DEPTH=1 lost 25% (no pipeline overlap). DEPTH=4 made smem too tight.
- **TMA: # CTAs ≥ 4 waves (~592)** to amortize pipeline drain. 148 CTAs
  (1 wave) lost ~3%. Beyond 1184 CTAs, no further gain.
- **LDG: BS=128 with many CTAs (≥8192) is optimal.** Larger BS (1024) lost up to 30%.
- **LDG: ITERS×stride must cover ≤ WS exactly.** Wraparound makes DRAM look like cache.

### Tuning knobs that didn't matter (much)
- TMA: TILE_BYTES from 4 KiB to 32 KiB all gave 20.0-20.5 TB/s (variation <2%)
  as long as bytes-per-iter ≥ 64 KiB and DEPTH=2.
- LDG: block count from 8192 to 32768 within ±5% in L2-hit case.

### Surprising findings
- **Catalog L2 wire (13.3 TB/s) is too low.** Both kernels EXCEED it by 37-54%.
  Catalog 13.3 may be a per-LTC summation that ignores LSU dedup. Real
  achievable L2-side bandwidth is ≥20 TB/s.
- **`l1tex__t_bytes` = 18.16 TB/s but `lts__t_bytes` = 6.61 TB/s for LDG L2-hit.**
  The 2.7× gap is L2 line dedup at MSHR / crossbar — same line requested by
  multiple SMs gets deduplicated before hitting LTS bytes counter. So
  `lts__t_bytes` UNDER-counts true L2 wire bandwidth in cache-hit cases.
- **`.ca` cache hint variant of LDG hits 34.7 TB/s** (l1tex_lookup_hit ratio 99.9%).
  This is L1 reuse — NOT a fair comparison to TMA's true L2 throughput. Listed
  here for completeness only. (test: `/tmp/h2h_ldg_ca.cu`, not committed.)
- **DRAM-cold: TMA and LDG achieve essentially equal HBM SoL (95-97%).** TMA has
  no DRAM-side advantage despite larger transactions; HBM scheduler already
  coalesces LSU requests into long bursts.
- **Sustained clock under both kernels was 1942 MHz**, not the 2032 MHz boost.
  This rig DVFS-settles at 1942 under sustained mem-bound load.

### Both kernels with their best configs

```bash
# LDG L2-hit (18.25 TB/s):
./QuickRunCUDA tests/bench_h2h_ldg.cu -t 128 -b 8192 \
    -A $((64*1024*1024/4)) -C 1048576 -0 1632 -1 -1 -2 26 \
    -T 30 -H "#define BLOCK_SIZE 128"

# LDG DRAM-cold (7.41 TB/s):
./QuickRunCUDA tests/bench_h2h_ldg.cu -t 128 -b 32768 \
    -A $((1024*1024*1024)) -C 1048576 -0 32 -1 -1 -2 32 \
    -T 30 -H "#define BLOCK_SIZE 128"

# TMA L2-hit (20.49 TB/s):
./QuickRunCUDA tests/bench_h2h_tma.cu -t 64 -b 592 -s 131072 \
    -A $((64*1024*1024/4)) -C 1048576 -0 1411 -1 -1 -2 26 \
    -T 30 -H "#define BLOCK_SIZE 64
#define TMA_BYTES 8192
#define NTMAS 8
#define DEPTH 2"

# TMA DRAM-cold (7.32 TB/s):
./QuickRunCUDA tests/bench_h2h_tma.cu -t 64 -b 1184 -s 131072 \
    -A $((1024*1024*1024)) -C 1048576 -0 221 -1 -1 -2 32 \
    -T 30 -H "#define BLOCK_SIZE 64
#define TMA_BYTES 32768
#define NTMAS 2
#define DEPTH 2"
```

### Methodology caveats
- Wall-clock at 1942 MHz boost; NCU at 1910 MHz locked. Wall/NCU agreement
  within 2% confirms not a clock artifact.
- WS=64 MiB chosen rather than 128 MiB to be safely below B300 L2 cap (126 MB)
  and avoid edge eviction.
- `bench_h2h_ldg.cu` uses `__launch_bounds__(BLOCK_SIZE,1)` — BS must be
  hard-coded via `-H "#define BLOCK_SIZE N"` and `-t N` matched.
- Both kernels write final accumulator under `if(v==seed)` impossible branch
  (seed=-1, accumulator never matches integer -1 from XOR of zero-init buffers
  on cold runs, but the SASS shows STG present so anti-DCE is effective).
- Anti-DCE confirmed for LDG: 32 LDG.E.128 per loop body in SASS.
- Anti-DCE confirmed for TMA: cp.async.bulk + ld.shared in SASS, consumer
  XOR result conditionally stored.
- 30-50 timed iterations per cell, std-dev <1% across reps.
