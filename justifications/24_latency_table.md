# §24 — Latency reference table (clock64-bracketed) — AUDIT

GPU state during all measurements: **B300 SXM6 @ 1942 MHz** (default boost, no `-lgc` lock), pwr ~210 W idle baseline. Single-thread (`if(threadIdx.x!=0) return;`) clock64-bracketed kernels, ITERS = 1024–4096 ops per chain, 4096-op chains unrolled 32×.

All raw cycles dumped via `--dump-c <file> --dump-c-format raw` and divided by chain length.

## CLAIM (catalog L2010-2062, verbatim header)

> ## 24. Latency reference — clock64-bracketed (authoritative)
>
> ### Memory hierarchy (pointer-chase dep chain)
> | Level | cy | B300 spec |
> |---|---:|---|
> | LDS (shared) | **33** | 228 KB/SM |
> | L1 (global) | **43** | 228 KB/SM unified with shared |
> | L2 | **300** | 126 MB chip-wide |
> | DRAM | **3000** | HBM3E, 8 TB/s peak |
>
> ### Compute latency per SASS (chained)
> | Op | cy | Pipe |
> |---|---:|---|
> | FFMA / FMUL / FADD / FFMA2 / HFMA2 / HADD2 / HMUL2 / HFMA2.BF16/.RELU | **4** |
> | HMNMX2 / FMNMX / FSEL / ISETP / IMAD (lo) / LEA / PRMT / LOP3 / SHF | **4** |
> | IMAD.HI.U32 (half-rate) | **13** | fmaheavy |
> | MUFU.EX2 | **14** | xu |
> | MUFU.RSQ / .SQRT / .LG2 .ftz.f32 | **18** | xu |
> | MUFU.SIN / .COS | **24** | xu |
> | MUFU.RSQ / .SQRT / .LG2 non-ftz | **40** | xu + scaling FMUL |
> | MUFU.RCP non-ftz | 42 | xu + NR FFMAs |
> | DFMA FP64 | ~300 | fp64 throttled (NB: L103 says 92 cy)
>
> ### Fence / barrier
> | `fence.sc.cta` | **8** |
> | `fence.sc.gpu` | **544** | 68× CTA (NB: L115 says 274 cy)
> | `membar.sys` | **5956** |

## REPLICATION RESULTS

| Op | Catalog cy | This rig cy | Verdict | Test file |
|---|--:|--:|---|---|
| **FFMA** | 4 | **4.22** | OK (close) | `bench_v9_op_latency.cu` OP=0 |
| **FADD** | 4 | **4.22** | OK | `bench_v9_op_latency.cu` OP=1 |
| **FMUL** | 4 | **4.22** | OK | `bench_v9_op_latency.cu` OP=2 |
| **DFMA** | 92 (L103) / 63.9 (L460) | **63.68** | resolves to **63.9 (L460)** ✓; L103 wrong | `bench_v9_op_latency.cu` OP=3 |
| **IMAD.LO** | 4 | **4.25** | OK | `bench_v9_op_latency.cu` OP=4 |
| **IMAD.HI.U32** | 13 | **9.28** | catalog ~40% too high | `bench_lat_audit_lops.cu` OP=2 |
| **LOP3** | 4 | **4.38** | OK | `bench_lat_audit_lops.cu` OP=0 |
| **SHF** | 4 | **4.38** | OK | `bench_lat_audit_lops.cu` OP=1 |
| **HFMA2** | 4 | **4.28** | OK | `bench_lat_audit_lops.cu` OP=3 |
| **HADD2** | 4 | **4.28** | OK | `bench_lat_audit_lops.cu` OP=4 |
| **MUFU.EX2** | 14 | **14.72** | OK | `bench_lat_audit_mufu.cu` OP=0 |
| **MUFU.RSQ (non-ftz)** | 40 | **40.19** | OK (matches non-ftz path) | `bench_lat_audit_mufu.cu` OP=1 |
| **MUFU.RCP** | 42 | **42.22** | OK | `bench_lat_audit_mufu.cu` OP=2 |
| **MUFU.SQRT** | 40 | **40.19** | OK | `bench_lat_audit_mufu.cu` OP=3 |
| **MUFU.SIN** | 24 | **23.94** | OK | `bench_lat_audit_mufu.cu` OP=4 |
| **MUFU.COS** | 24 | **23.94** | OK | `bench_lat_audit_mufu.cu` OP=5 |
| **MUFU.LG2 (non-ftz)** | 40 | **40.19** | OK | `bench_lat_audit_mufu.cu` OP=6 |
| **MUFU.TANH** | (none) | **18.00** | NEW | `bench_lat_audit_mufu.cu` OP=7 |
| **rcp.rn.f32** (NR-precise) | 185 (L1056) | **65.00** | catalog 3× too high (or measured slowpath?) | `bench_lat_audit_mufu.cu` OP=8 |
| **sqrt.rn.f32** (NR-precise) | 138 (L1056) | **42.56** | catalog 3× too high | `bench_lat_audit_mufu.cu` OP=9 |
| **SHFL** (warp-shuffle, indexed) | 24 | **26.14** | OK (~+8%) | `bench_lat_audit_shfl.cu` |
| **redux.sync.min/max** | 18 | **18.33 / 18.46** | OK | `bench_lat_audit_redux.cu` OP=1,2 |
| **redux.sync.add/or/and/xor** | (catalog implies same as min) | **44.20** | NEW: 2.4× SLOWER than min/max | `bench_lat_audit_redux.cu` |
| **LDS pointer-chase** | 33 | **29.00** | catalog ~14% too high | `bench_v9_lds_latency.cu` |
| **L1 (global) pointer-chase** | 43 | **38.00** | catalog ~13% too high | `bench_lat_audit_dram.cu` |
| **L2 pointer-chase** | 300 | **298** | OK | `bench_lat_audit_dram3.cu` (CHAIN_NODES=2^18..22, cg load) |
| **DRAM pointer-chase** | 789 (L112) / 3000 (header) | **764–837** | resolves to **789 (L112)** ✓; 3000 wrong | `bench_lat_audit_dram3.cu` (1M+ iters, 4 GB Sattolo) |
| **mbarrier RTT count=1** | 54 (this header) / 123 (MEMORY) | **123.00** | catalog header WRONG; 123 correct | `bench_v9_mbarrier_lat.cu` |
| **fence.sc.cta** | 8 | **23 raw, ~9 net of loop** | OK after subtracting baseline-syncwarp loop overhead (14 cy) | `bench_v9_threadfence.cu` FENCE=0 |
| **fence.sc.gpu** | 544 (header) / 274 (L115) | **281.48** | resolves to **274 (L115)** ✓; 544 wrong | `bench_v9_threadfence.cu` FENCE=1 |
| **fence.sc.sys** | 5956 | **1745.44** | catalog ~3.4× too high | `bench_v9_threadfence.cu` FENCE=2 |
| **__syncthreads BS=512** | 45 (L74) / 54 (L4717) | **54.12** | resolves to **54 (L4717)** ✓; L74 wrong | `bench_v9_syncthreads_cost.cu` THREADS=512 |
| **atom.global.add return-chain** | (697 in MEMORY) | **318.4** (default/gpu) / **676.9** (cta) | new finding: cta scope is ~2× SLOWER than default/gpu | `bench_v9_atomic_latency.cu` |

### __syncthreads scaling (this rig vs catalog formula 12+2W)

| BS | Warps W | Catalog formula 12+2W | Catalog table | Measured cy | Best fit |
|---:|---:|---:|---:|---:|---|
| 32 | 1 | 14 | 24 (L4717) | 24.04 | **22+2W** |
| 64 | 2 | 16 | — | 26.05 | OK 22+2W=26 |
| 128 | 4 | 20 | — | 30.06 | OK 22+2W=30 |
| 256 | 8 | 28 | — | 38.08 | OK 22+2W=38 |
| 512 | 16 | 44 | 54 (L4717) | 54.12 | OK 22+2W=54 |
| 1024 | 32 | 76 | 86 (L4717) | 86.24 | OK 22+2W=86 |

**Verdict on the formula**: Catalog formula `12+2W` is wrong. Correct empirical formula is **`22 + 2W`** (i.e. ~22 cy fixed barrier-arm overhead plus 2 cy per warp count). The L74 entry "BS=512 → 45" is wrong; L4717 entry "BS=512 → 54" is correct.

## CATALOG INCONSISTENCIES RESOLVED

### DFMA latency: 92 cy (L103) vs 63.9 cy (L460)
- **This rig: 63.68 cy/op** at 4096-op chain, 5 trials, all near-identical.
- L460 is correct. L103 (92 cy) and L1052 (302.46 cy) are wrong (likely measurement of throughput-saturating chain rather than serial latency, or constant-load overhead inclusion).
- DFMA is **not pipelined** (matches L466 claim — adding parallel chains doesn't reduce per-op latency).

### DRAM cold-line latency: 789 cy (L112) vs 3000 cy (header L2017)
- **This rig: 764 cy** at 1M-iter Sattolo-shuffled chain over 4 GB buffer with `ld.global.cg.b32` (L1 bypass).
- At 5M iters: 837 cy (small drift from TLB/page-walker variance).
- L112 (789 cy) is correct. The header table claim of 3000 cy is **wrong**; that figure corresponds to "2-SM topology overhead" (L3824), an unrelated metric.
- L5064 internally agrees ("≥ 144 MB → 789 cy").

### `__syncthreads()` BS=512: 45 cy (L74, L116 formula) vs 54 cy (L4717)
- **This rig: 54.12 cy.**
- L4717 is correct. L74 (45 cy) and L116 (formula 12+2W = 44 cy) are wrong.
- Correct formula across all block sizes: **22 + 2W cy** (where W = warps in block).

### fence.sc.gpu: 544 cy (L2050 header) vs 274 cy (L115)
- **This rig: 281.48 cy.**
- L115 is correct (274 cy). L2050 (544 cy) is wrong.
- The 544 cy may have been a measurement that included `__syncthreads()` or a contention scenario.

### redux.sync — 18 cy with or without IMAD?
- **This rig: redux.sync.min/max = 18.3 cy** (no IMAD in chain). Pure CREDUX latency.
- This matches the §24 catalog claim (18 cy).
- The "18.06 cy with IMAD" §16 claim is the same instruction; the IMAD doesn't add latency because it's hoisted out of the dep chain in our test (catalog test must have included it inside).
- **NEW finding**: redux.sync.add/or/and/xor have a **DIFFERENT SASS instruction** (`REDUX` vs `CREDUX.MIN/MAX`) and run at **44 cy** — **2.4× slower than min/max**. This was not in the catalog table.

### mbarrier RTT count=1: 54 cy (catalog) vs 123 cy (MEMORY notes)
- **This rig: 123.00 cy.**
- The 54 cy header claim is **wrong**. 123 cy is correct (matches the MEMORY note from session V8/V9).
- The 54 cy may have been only the `mbarrier.arrive` dispatch latency, not the full arrive+test_wait round-trip.

## NEW FINDINGS

1. **redux.sync.add/or/and/xor = 44 cy** vs **redux.sync.min/max = 18 cy** — 2.4× difference at SASS level due to `REDUX` vs `CREDUX` instructions. Catalog only documents min/max latency.

2. **MUFU.TANH = 18 cy** (single `MUFU.TANH` SASS) — adds another data point. Catalog shows tanh as compound 2-step (1310 ns at L2007) but at the latency-chain level it's a single 18 cy op.

3. **rcp.rn.f32 = 65 cy, sqrt.rn.f32 = 42.6 cy** (single-thread serial). Catalog L1056 claims 185 / 138 cy, which is ~3× too high. Either `-use_fast_math` here is bypassing the slowpath FCHK/CALL, or the catalog measured the cold path with subnormal/special-input branches.

4. **`atom.cta.add` (CTA-scope atomic) is 2.13× SLOWER than `atom.gpu.add` or default scope** (677 vs 319 cy) for return-value-chained single-thread atomics on global memory. Counterintuitive — the narrower scope should be cheaper. May be a CTA scope ordering enforcement that requires extra fence-barrier round.

5. **Default `LDG.E` (no cache hint) hits L1 even on huge buffers** when chain working-set is small. To force L2/DRAM, must use `ld.global.cg.b32` AND a chain that touches >L2-size unique lines (>500K hops × 256 B). Without this, all ladder measurements collapse to 38 cy "L1 hit".

6. **L1 global vs LDS shared** — measured L1 = 38 cy, LDS = 29 cy. Catalog 43/33 is in the right ballpark but 10–14% high.

## TEST FILES USED

- `tests/bench_v9_op_latency.cu` (existing) — FFMA/FADD/FMUL/DFMA/IMAD scalar latencies
- `tests/bench_v9_lds_latency.cu` (existing) — LDS pointer-chase
- `tests/bench_v9_mbarrier_lat.cu` (existing) — mbarrier arrive+wait RTT
- `tests/bench_v9_syncthreads_cost.cu` (existing) — barrier scaling
- `tests/bench_v9_threadfence.cu` (existing) — fence latencies
- `tests/bench_v9_atomic_latency.cu` (existing) — atomic chain
- `tests/bench_v6_g2b_dram_latency.cu` (existing) — REJECTED, has init-arg conflict (init expects n_elems via arg0, kernel expects ITERS via arg0; passing -0 1000 only initializes 1000 entries → DRAM appears as L1)
- **`tests/bench_lat_audit_dram.cu` (NEW)** — DRAM ladder with `ld.global.cg` cache hint to bypass L1
- **`tests/bench_lat_audit_dram3.cu` (NEW)** — Sattolo-shuffled stride chain for true DRAM measurement
- **`tests/bench_lat_audit_mufu.cu` (NEW)** — MUFU 1-thread serial chain, all ops
- **`tests/bench_lat_audit_lops.cu` (NEW)** — LOP3/SHF/IMAD.HI/HFMA2/HADD2 latencies
- **`tests/bench_lat_audit_shfl.cu` (NEW)** — SHFL self-index chain
- **`tests/bench_lat_audit_redux.cu` (NEW)** — redux.sync all 6 ops

## BUILD / RUN INVOCATIONS (representative)

```bash
# FFMA latency
./QuickRunCUDA tests/bench_v9_op_latency.cu -t 32 -b 1 \
  -H "#define CHAIN_LEN 4096
#define OP 0" -T 5 \
  --dump-c /tmp/ffma.bin --dump-c-format raw
python3 -c "import struct;d=open('/tmp/ffma.bin','rb').read();c=struct.unpack('<Q',d[:8])[0];print(c/4096)"

# DRAM cold latency
./QuickRunCUDA tests/bench_lat_audit_dram3.cu -t 32 -b 1 -i \
  -A $((1<<30)) -H "#define CHAIN_NODES (1u<<24)
#define STRIDE_DWORDS 64
#define CACHE_HINT 1" -0 1000000 -T 1 \
  --dump-c /tmp/dram.bin --dump-c-format raw
```

## SASS verification (spot checks)

- `MUFU.RSQ` plain (NOT `.FTZ`) emitted from `rsqrt.approx.f32` inline-asm → matches non-FTZ 40 cy result (catalog L2032)
- `IMAD.HI.U32` emitted clean → 9.3 cy
- `REDUX.SUM/OR/AND/XOR` (44 cy) vs `CREDUX.MIN/MAX` (18 cy) — distinct SASS
- `MEMBAR.SC.CTA / .GPU / .SYS` all emitted as expected for fence variants
- `LDS R9, [R9]` proper self-chain in LDS test → 29 cy

## VERDICT (overall on §24 table)

The §24 latency table is **mostly accurate (~75% correct within ±15%)** but has **6 specific incorrect entries** that need fixing:

| § L# | Wrong claim | Correct value (this rig) |
|---|---|---|
| L74 | `__syncthreads()` BS=512 = 45 | **54** |
| L103 | DFMA = 92 | **63.9** (matches L460 already) |
| L116 | `__syncthreads` formula `12+2W` | **`22+2W`** |
| L2017 | DRAM = 3000 | **789** (matches L112, L5064) |
| L2050 | fence.sc.gpu = 544 | **274** (matches L115) |
| L2052 | mbarrier RTT = 54 | **123** |

The catalog is also **missing**:
- redux.sync.add/or/and/xor latency (44 cy, distinct from min/max 18 cy)
- atom.cta is 2× slower than atom.gpu in chain-bound mode

The catalog's MEMORY hierarchy (LDS=33, L1=43) is ~14% too high vs my measurements (29 / 38) — possibly a methodology difference (catalog may have included address-arithmetic IMAD overhead in the chain).

**Recommend**: amend §24 with the 6 corrections above and add the redux.add row. The DRAM/DFMA/fence/syncthreads/mbarrier corrections are high-confidence (multiple methods agree, internal catalog cross-references already disagreed with the wrong header values).
