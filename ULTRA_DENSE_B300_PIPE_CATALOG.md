# ULTRA_DENSE — B300 / Blackwell sm_103a Reference

Reference card for **NVIDIA B300 SXM6 AC** (sm_103a). Only fully-validated numbers are in the body — every entry was independently re-run on this rig with SASS dump + ncu metrics + wall-clock evidence. For per-section evidence see `JUSTIFIED_B300_PIPE_CATALOG.md`; for items still under doubt see the **TODO** list at the bottom.

**Conventions**

- Clock state always tagged: `@2032` (boost ceiling), `@1942` (rig DVFS settling under sustained compute), `@1920` (`-lgc 2032` paradox), `@1800` (recommended lock for repeatability), `@1500`, `@1005`. If unstated, assume `@1942`.
- Bandwidth denominator: spec **7,680 GB/s**; this device (post-ECC) **7,672 GB/s**.
- TFLOPS counts FLOPS, not FMA-ops (so FFMA = 2 FLOPS).
- `r` = warp-instructions issued per SM per cycle; total dispatch ceiling **4.00**.
- `cy` always SM clocks unless prefixed `ns`.

---

## 1. Hardware

| Property | Value |
|---|---|
| GPU | NVIDIA B300 SXM6 AC |
| Compute capability | sm_103a (PTX 10.3) |
| SMs | **148** (IDs 1–147; SM 0 never scheduled) |
| GPCs | **9 × 16 SMs + 1 partial × 4 SMs = 148** |
| Warps per SM | 64 (4 SMSPs × 16 warps each) |
| Max CTAs per SM | 32 |
| Max threads per CTA | 1024 |
| FP32 cores per SM | **128** (4 SMSPs × 32 lanes) |
| FP32 cores total | 18,944 |
| Registers per SM | 65,536 (256 KB) |
| Max regs per thread | 232 (`setmaxnreg`); spill cliff at ~32 live floats with `MIN_BLOCKS=16` |
| Shared memory per SM | 228 KB pool (227 KB usable per CTA opt-in; 1 KB reserved) |
| `sharedMemPerBlock` (default, no opt-in) | **48 KB** |
| `sharedMemPerBlockOptin` (max per CTA) | 227 KB |
| `sharedMemPerMultiprocessor` (per-SM HW max) | 228 KB |
| L1 cache | 256 KB (shared with smem pool) |
| L2 cache | **126.5 MiB** (`cudaDeviceProp.l2CacheSize`) |
| HBM | **268 GiB HBM3e** post-ECC |
| HBM bus | **7,680 bits** (1/16 controllers fused on AC SKU) |
| HBM stacks | **8 × 12-Hi** (3 GB/die) |
| HBM I/O clock | 3,996 MHz → 7.992 Gbps/pin → **7,672 GB/s post-ECC** |
| PCIe | Gen 6 x16 card; realized Gen 5 on this host (~57 GB/s/dir) |
| NVLink | 18 lanes × 53.125 GB/s = 956 GB/s bidirectional per peer |
| ECC | always on |
| Async copy engines | 4 |
| Media | 7 NVDEC + 1 NVENC + 1 OFA + 7 JPEG |
| MIG | up to 7 × 1g.34gb |
| TDP | 1,100 W |
| Idle baseline | 144–198 W |

---

## 2. Clock states

| State | MHz observed | When |
|---|--:|---|
| `nvidia-smi -q` boost ceiling | 2032 | datasheet boost (rarely sustained under heavy load) |
| Sustained FFMA, no lock, long kernel | ~1990 | DVFS settling under one specific FFMA-peak workload |
| `-lgc 1800` lock (recommended for repeatability) | 1801 (98.3% honored) | sustained |
| `-lgc 1500` lock | 1468–1495 (98%) | sustained |
| `-lgc 1005` lock | 987 | long kernel only — short kernels firmware-override upward |
| `-lgc 510` lock | 501 | sustained |
| `-lgc N` for N ≥ 1920 | silently clamps to ~1920 | known firmware paradox |
| Stuck-low silent floor | 1005 | leftover process / driver state can park here without warning |

**Default rig settling under sustained compute is ≈1942 MHz**, not 2032 boost and not 1920 base — most entries below assume this. For across-workload comparable measurements use `-lgc 1800`.

---

## 3. Pipe topology

| Pipe | Cap (warp-inst/SM/cy) | Hosts |
|---|--:|---|
| pipe_fma (scalar) | **4.00** | FFMA, FMUL, FADD |
| pipe_fma (packed = both fmaH+fmaL per inst) | 2.00 | FFMA2, HFMA2, HFMA2.BF16 |
| pipe_alu | **2.00** | LOP3, PRMT, SHF, F2FP, FMNMX, ISETP, FSETP, I2FP, SEL, VOTE |
| pipe_fmaheavy | 2.00 | IMAD, IMAD.X, IMAD.WIDE, IDP.4A/2A, HADD2.F32 |
| pipe_fmalite | 2.00 | "lite" half of FFMA path |
| pipe_xu (simple) | **1.00** | MUFU.EX2, RCP, RSQ, SQRT, LG2, TANH, POPC, BREV, FLO/CLZ |
| pipe_xu (compound) | 0.50 | MUFU.SIN, COS (range-reduction) |
| pipe_lsu | 1.00 | LDG, STG, LDS, STS, LDSM, SHFL.SYNC |
| pipe_adu | 0.50 | LDC.32, BAR.SYNC, MATCH.ANY, REDUX.SUM/OR/AND/XOR, S2R |
| pipe_uniform | ~1.0 | UFFMA family, UPRMT, UIADD3, UIMAD, UMOV, UISETP, ULOP3 |
| pipe_fp64 | 0.06 | DFMA, DADD, DMUL |
| pipe_tensor | per-kind | HMMA, IMMA (legacy mma.sync only — does NOT measure tcgen05) |
| pipe_cbu | — | BRA, EXIT (invisible in steady state) |

**Hard ceiling**: total warp-inst/SM/cy ≤ 4.00. Cross-pipe sums (alu + fma) can exceed 100% per-pipe accounting because the pipes are independent; this is normal, not a violation.

**FFMA dispatches to ONE sub-pipe per cycle** (alternating fmaH/fmaL); the catalog phrasing "uniquely uses both sub-pipes simultaneously" is wrong.

---

## 4. SASS opcode → pipe map

| SASS | Pipe | r |
|---|---|--:|
| FFMA, FMUL, FADD | fma | 4.00 |
| FFMA2, HFMA2, HFMA2.BF16, HADD2, HMUL2 | fmaH+fmaL fused | 2.00 |
| IMAD, IMAD.X, IMAD.WIDE | fmaH | 2.00 |
| IMAD.HI.U32 | fmaH | 1.00 (half rate) |
| IDP.4A.S8.S8, IDP.2A | fmaH | 2.00 |
| IADD3 | alu | 2.00 (= 4.00 logical adds via 3-source fusion) |
| LOP3.LUT, SHF.{L,R}, PRMT, SEL | alu | 2.00 |
| F2FP.{F16,BF16}.{E4M3,E5M2,E2M1,E2M3,E3M2,E8}.UNPACK_B | alu | 2.00 |
| F2FP.{F16,BF16}.F32.PACK + PRMT | alu | 1.00 (PRMT tax) |
| FMNMX, FMNMX.NAN, HMNMX2, HMNMX2.BF16 | alu | 2.00 |
| FMNMX3 (Blackwell 3-input fused FP min/max) | alu | 2.00 |
| VIMNMX3 (3-input int min/max) | alu | 2.00 |
| ISETP, FSETP, VOTE.ANY | alu | 2.00 |
| F2IP.U8.F32.NTZ | alu | 2.00 |
| F2I.S8.NTZ | xu | 0.5 (4× slower than F2IP.U8) |
| I2FP.F32.{U32,S32} | alu | 2.00 |
| I2F.S64 | xu | 0.04 (super slow — avoid) |
| MUFU.EX2 | xu | 1.00 |
| MUFU.{TANH,SIN,COS,SQRT,RSQ,LG2,RCP} | xu | 0.5 |
| MUFU.EX2.BF16x2 | xu | 0.5 inst (= 1.0 op via 2-element pack) |
| POPC, BREV, FLO/CLZ, BFE | xu | 0.5 |
| BFI | alu (folds to LOP3) | 2.00 |
| LDG.E, STG.E | lsu | DRAM-bound |
| LDS, STS | lsu | 1.00, bank-conflict-sensitive |
| **LDC.32** (`ld.const`) | **adu** | 0.5 — broadcast amplification 31.7× |
| LD.E (`ld.shared::cluster` → DSMEM) | lsu (global path) | latency 207–223 cy |
| SHFL.SYNC.* | lsu | 1.00 |
| MATCH.ANY | adu | 0.5 (and slow — ~375 cy) |
| BAR.SYNC.DEFER, BAR.ARV, BAR.RED.POPC | adu | 0.36–0.47 |
| REDUX.SUM/OR/AND/XOR | adu | 0.50 (4× slower than CREDUX.MIN/MAX which is 1.92 alu+fmaH) |
| CREDUX.MIN, CREDUX.MAX | alu+fmaH | 1.92 (almost saturates both pipes) |
| MEMBAR.SC.CTA | lsu | 0.83 |
| MEMBAR.SC.GPU + ERRBAR | adu+lsu | dominates fence cost |
| ATOM.E.CAS.STRONG.{GPU,SYS} | lsu | bandwidth-bound |
| REDG.E.* (no-return atomic) | lsu | 25× faster than ATOMG with return |
| ATOMS.POPC.INC.32 | lsu | 0.84 (only emitted for `atomicAdd(addr, 1u)` no-return on shared) |
| LDSM.x1 / LDSM.x4 | uniform+lsu | 1.0 / 0.25 |
| UTCQMMA, UTCHMMA, UTCOMMA, UTCOMMA.BLOCK16 | tcgen05 (one pipe per SM) | per-MMA cycle costs in §14 |
| UTCATOMSWS, UTCBAR | tcgen05 control | — |
| DFMA, DADD, DMUL | fp64 | 0.06 |

---

## 5. Compute throughput — chip-wide peaks

| Path | Peak measured | % of theoretical | Conditions |
|---|--:|--:|---|
| FP32 FFMA scalar | **75.5 TF** | 98.1% of 76.96 TF | unlocked, single launch ≥ 40 ms, ~1990 MHz observed |
| FP32 FFMA scalar | **67.0 TF** | 98.3% of 68.20 TF | `-lgc 1800` (recommended for repeatability) |
| FP32 FFMA scalar | **71.8 TF** | 96.6% of 74.32 TF | `@1942` short-kernel (12,800 iters) |
| FFMA2 + LOP3 1:1 (total useful) | **314 ops/SM/cy** | — | dual-issue sweet spot — see §21 |
| FP16 / BF16 HFMA2 (non-tensor) | 35.2 TF | 99% pipe_fma | `@1942`; scalar `fma.rn.f16` also emits HFMA2 with one half wasted → ½ rate |
| FP16 / BF16 mma.sync m16n8k16 | **571 TF** | 99.5% pipe_tensor | `@1942` |
| TF32 mma.sync m16n8k8 | **285.7 TF** | — | `@1942` |
| FP8 mma.sync m16n8k32 (emulated F2FP+HMMA) | **309 TF** | — | `@1942`; catalog 276 was 12% low |
| INT8 mma.sync IMMA | **142.4 TOPS** | — | `@1942` |
| FP4 tcgen05.mma `kind::mxf4nvf4.block_scale.block16` K=64 | **9.26 PF** | 92.6% of 10 PF spec; 98.4% of theoretical at observed clock | `@1942`, SASS `UTCOMMA.BLOCK16` |
| FP64 DFMA | **1.06 TF** | 88% of 1.20 TF; 99.95% pipe peak | `@1942`; ratio 1:64 vs FP32 spec |
| MUFU EX2 | 8,850 G ops/s | 97% of pipe_xu peak | `@1942` |
| MUFU SIN/COS/RSQ/SQRT/LG2/TANH | ~4,500 G ops/s | 50% of pipe_xu peak | `@1942` |

For tcgen05.mma other formats (BF16/FP16/TF32/FP8/sparse), see §14 — those rows still need a per-format re-run on this rig.

---

## 6. Per-instruction issue rates (warp-inst/SM/cy)

### FP32 / FP16 / BF16

| PTX | SASS | r | Logical work |
|---|---|--:|---|
| `fma.rn.f32` | FFMA | 4.00 | 128 FMAs/SM/cy = 256 FLOPS |
| `mul.rn.f32` / `add.rn.f32` | FMUL / FADD | 4.00 | 128 |
| `fma.rn.f32x2` | FFMA2 | 2.00 | 128 FMAs (same FLOPs as scalar; uses 2 dispatch slots) |
| `fma.rn.f16x2` | HFMA2 | 2.00 | 128 FP16 FMAs |
| `fma.rn.bf16x2` | HFMA2.BF16 | 2.00 | 128 BF16 FMAs |
| `fma.rn.f16` (scalar) | HFMA2 | 2.00 | 64 useful (other half wasted) |

### Integer

| PTX | SASS | r | Notes |
|---|---|--:|---|
| `mad.lo.u32` | IMAD | 2.00 | 64 IMAD |
| `mul.hi.u32` | IMAD.HI.U32 | 1.00 | half rate |
| `dp4a.s32.s32` | IDP.4A.S8.S8 | 2.00 | 512 INT8 dot ops |
| `add.u32` (single) | IADD3 / IMAD.IADD (compiler splits 2:1) | 4.00 total | exploits independent alu + fmaH pipes |
| `add.u32 a,b,c,d` (3-input) | IADD3 fused | 2.00 | = 128 logical adds |
| `add.u64` | IADD3 + IMAD.X (2 SASS) | 1.00 each | 64 u64 adds/SM/cy |
| `mul.lo.u64` | IMAD + IMAD.WIDE + 3× IADD3 | — | ~12 u64 muls/SM/cy |
| `and/or/xor.b64` | 2× LOP3 | — | 32 |

### Narrow-format CVT (UNPACK)

All variants identical at **r = 2.00 (≥99.98% pipe_alu)**:

| Type | SASS |
|---|---|
| FP8 e4m3 / e5m2 | F2FP.F16.E4M3.UNPACK_B / .E5M2 |
| FP4 e2m1 | F2FP.F16.E2M1.UNPACK_B |
| FP6 e2m3 / e3m2 | F2FP.F16.E2M3.UNPACK_B / .E3M2 |
| ue8m0 → bf16 | F2FP.BF16.E8.UNPACK_B |

Friction: with a 1-per-iter LOP3 zero-extension, effective rate halves to 1.00.

### Other CVTs

| PTX | SASS | r |
|---|---|--:|
| `cvt.rn.f16.f32` (pack) | F2FP.F16.F32.PACK + PRMT | 1.00 |
| `cvt.f32.f16` | HADD2.F32 | 2.00 |
| `cvt.rn.f32.{u32,s32}` | I2FP.F32.* | 2.00 |
| `cvt.rni.sat.u8.f32` | F2IP.U8.F32.NTZ | 2.00 |
| `cvt.rni.sat.s8.f32` | F2I.S8.NTZ | 0.5 (4× slower than the U8 path) |
| `cvt.rn.f32.s64` | I2F.S64 | 0.04 |

### Bitwise / shift / permute / compare

| PTX | SASS | r |
|---|---|--:|
| `xor/and/or/not/lop3.b32` | LOP3.LUT | 2.00 |
| `shf.{l,r}.wrap.b32` | SHF.{L,R}.W.U32 | 2.00 |
| `prmt.b32` | PRMT | 2.00 |
| `bfi.b32` | LOP3.LUT | 2.00 |
| `bfe.u32` | SHF.R.U32.HI + SGXT.U32 (2 SASS) | 1.00 |
| `popc.b32` | POPC | 0.5 |
| `brev.b32` | BREV | 0.5 |
| `clz.b32` / `bfind` | FLO.U32 (+ IADD3) | 0.5 |
| `setp.*.{u32,s32,f32}` | ISETP / FSETP | 2.00 |
| `selp.b32` | SEL | 2.00 |
| `min/max.f32` | FMNMX | 2.00 |
| 2× `min.f32` | **FMNMX3** (Blackwell fused 3-input) | 2.00 inst = 128 logical mins |
| `min/max.f16x2` / `min/max.bf16x2` | HMNMX2 / HMNMX2.BF16 | 2.00 |
| `min/max.s32` | VIMNMX3 (compiler folds 2 mins) | 2.00 |
| `copysign.f32` | LOP3.LUT | 2.00 |

### Const memory

| PTX | SASS | Pipe | Throughput |
|---|---|---|--:|
| `ld.const.u32` (broadcast) | LDC.32 | adu | **17.99 TB/s effective / 0.562 TB/s actual** at BS=512 (31.7× broadcast amplification) |

BS=256 only reaches 90% of cap; BS=512 saturates at 99.5% pipe_adu.

### Warp / barrier

| PTX | SASS | Pipe | r |
|---|---|---|--:|
| `shfl.sync.{bfly,idx,up,down}` | SHFL.* | lsu | 1.00 |
| `vote.ballot` | VOTE.ANY + ISETP | alu | 2.00 combined |
| `vote.{any,all,uni}` | VOTE.* | alu | 2.00 |
| `match.any.sync.b32` | MATCH.ANY | adu | 0.5 (and 375 cy — slow) |
| `bar.sync 0` | BAR.SYNC.DEFER | adu | 0.36 |
| `bar.arrive` | BAR.ARV | adu | 0.47 |
| `redux.sync.min/max.u32` | CREDUX.MIN/MAX | alu+fmaH | 1.92 (almost saturates both pipes — fastest reduction) |
| `redux.sync.{add,or,and,xor}.b32` | REDUX.SUM/OR/AND/XOR | adu | 0.50 (**4× slower than min/max**) |
| `membar.cta` | MEMBAR.SC.CTA | lsu | 0.83 |
| `ldmatrix.sync.x1.b16` | LDSM (1 quad) | uniform+lsu | 1.0 |
| `ldmatrix.sync.x4.b16` | LDSM (4 quads) | uniform+lsu | 0.25 |

### FP64

| PTX | SASS | Pipe | r | Chip TFLOPS |
|---|---|---|--:|--:|
| `fma.rn.f64` | DFMA | fp64 | 0.06 (99.95% pipe peak) | 1.06 |
| `add.rn.f64` / `mul.rn.f64` | DADD / DMUL | fp64 | 0.06 | 1.06 |

DFMA is **not pipelined** — 4 independent chains give zero ILP benefit (latency = throughput = 63.9 cy/op). FFMA + ALU can co-issue freely during the 64 cy window.

---

## 7. Memory hierarchy bandwidth

Single-source-of-truth for max-tuned throughput, `@1942`:

| Tier | Read TB/s | Write TB/s | Conditions |
|---|--:|--:|---|
| Shared memory `ld.shared.v4.u32` | **35.88** (97.5% of 36.79 theoretical) | ~34 | per-SM 128 B/clk × 148 SMs |
| L2 hit, WS = 64 MiB, LDG max-tuned | **18.25** | — | BS=128, ≥8192 CTAs, LDG.E.128 |
| L2 hit, WS = 64 MiB, TMA max-tuned | **20.49** | — | DEPTH=2, bytes/iter ≥ 64 KiB/CTA, ≥4 waves |
| L2 → DRAM cliff | — | — | at WS = 126.5 MB (`l2CacheSize`) |
| HBM3E read, WS = 4 GiB, LDG max-tuned | **7.41** (96.5% of 7672) | — | LDG.E.128, BS=128 |
| HBM3E read, TMA max-tuned | 7.32 (95.4%) | — | DRAM-cold |
| L1 cached `.ca` (WS ≤ 228 KB) | **13.13** | — | 1.88× faster than `.cg` |
| L2-bypass `.cg` | 6.97 | — | (NOT 1.25× of `.ca` as some sources claim) |
| `.L2::256B` cache hint at stride-256B 4B reads | **7.06** (92% HBM SoL) | — | LDG.E.LTC256B; +40% over baseline |
| PCIe H2D / D2H (Gen 5 realized) | 0.057 / 0.057 | — | 6.5 µs / 9.0 µs first-byte |
| PCIe full-duplex | 0.099 combined | — | — |
| NVLink peer (2-GPU rig) | 0.820 | 0.718 | ~2,700 cy P2P latency |
| Pinned (zero-copy) | 0.054 | 0.053 | ~1 µs/hop |
| DSMEM read, single-chain | — | — | 207–223 cy/load (latency-bound) |
| DSMEM read, ILP=32 chains | — | — | 9 cy/load (LDS-equivalent) |
| DSMEM aggregate write, max-tuned | — | **~2.4 TB/s chip** | v4 width × cluster=8 × 18 clusters |
| DSMEM aggregate read, max-tuned | **~1.9 TB/s chip** | — | same |

Footnotes:
- DSMEM bypasses L2 in both directions (~0.04% of byte volume hits L2).
- `lts__t_bytes` undercounts LDG L2-hit by 2.7× (MSHR dedup) — for LDG use `l1tex__t_bytes`; for TMA use `lts__t_bytes`.
- DRAM denominator: post-ECC 7,672 GB/s (NOT 8,000 GB/s spec — AC SKU has 1/16 controller fused).

---

## 8. Latency table (cycles, single dependent chain via clock64)

| Op | cy |
|---|--:|
| FFMA / FMUL / FADD | 4.2–4.4 |
| HFMA2 / LOP3 / SHF | 4.2–4.4 |
| IMAD | 4.15 |
| **DFMA** | **63.9** (NOT 92 — that catalog value was wrong) |
| HMMA tensor | ~20 |
| MUFU.EX2 simple | 14 |
| MUFU.SIN / COS compound | 24 |
| MUFU.RSQ / SQRT / LG2 ftz | 18 |
| MUFU.RSQ / SQRT / LG2 non-ftz | ~40 |
| MUFU.RCP | 44 |
| `redux.sync.min/max.u32` (CREDUX.MIN/MAX) | 18 |
| `redux.sync.{add,or,and,xor}.b32` (REDUX.SUM/...) | **44** (2.4× slower than min/max — different SASS family) |
| SHFL | 24 |
| LDS hit | 29 |
| L1 hit `.ca` | 38 |
| L2 | 301 |
| DRAM cold (`.cg`, Sattolo-shuffled, WS > L2) | 789 |
| DSMEM read (single-chain) | 207–223 |
| DSMEM read (ILP=32) | 9 |
| `__syncthreads` | **`22 + 2W` cy** (W = warps/CTA) → 54 at BS=512, 86 at BS=1024 |
| `__threadfence_block` | 8 |
| `__threadfence` (gpu) | **267** sustained (+ ~280 first-fence-after-write) |
| `__threadfence_system` (single-GPU) | **1,727** (= 850 ns @ 2032) |
| `__threadfence_system` (2-GPU NVLink rig) | 2,806 (extra coherence round-trip) |
| `mbarrier.arrive` | **27** for default `.shared.b64`; 8.1 for `.relaxed.cta` |
| `mbarrier` arrive→test_wait round-trip | 123 |
| Coalesced LDG L1-hit, stride 4 B | 56 |
| LDG L1-hit, stride 32 B | 160 (monotonic rise from stride 4) |
| LDG L2-hit, stride > L1 sector footprint | 158–328 (regime-dependent) |

**Methodology notes that matter for L2/DRAM latency**: default `LDG.E` hits L1 even for a "DRAM" test unless you BOTH (a) use `ld.global.cg` (or `.cs`/`.lu`) to bypass L1 AND (b) chain over a Sattolo-shuffled WS exceeding 126.5 MB.

---

## 9. Fence costs

Single-CTA, single-warp, no busy chip:

| Fence | cy | ns @ 2032 |
|---|--:|--:|
| `__threadfence_block` (cta) | **8** | 3.9 |
| `__threadfence` (gpu) | **267** | 131.5 |
| `__threadfence_system` (single-GPU) | **1,727** | 850 |
| `__threadfence_system` (NVLink-attached, 2-GPU rig) | 2,806 | 1,381 |

**First-fence-after-write tax is FIXED, not linear**: the first store after a fence triggers L2 drain (+ ~280 cy); subsequent stores ride the open drain (no per-write cost). The "+60 cy/write" linear claim from older sources is wrong.

`release.gpu` drain is **SM-wide** — drains all co-resident CTAs' loads + stores. Compounds to 11,000+ cy at high occupancy.

`cp.async` (LDGSTS) **bypasses the acquire-fence drain** unless `commit_group` is issued first.

`CCTL.IVALL` itself is **~3 cy on idle pipeline** — fence cost is dominated by drain-wait for in-flight loads (acquire) or `MEMBAR.ALL.GPU` (release).

---

## 10. Atomics

### Latency (single-thread chain)

| Op | cy |
|---|--:|
| `atom.global.add.u32` chain | **45** (matches LDS chain at 45 cy) |
| `atom.global.add.u64` | 156 |
| `atom.global.cas.b64` | 731 |
| `atom.global.add.f32` | ~24% **faster** than u32 chip-wide |
| `atomicAdd(__half / __nv_bfloat16)` | emits `ATOM.E.CAS.STRONG.GPU` loop, **6.3× slower than u32** (NOT 45×) |
| `atom.global.add.{f16x2, bf16x2}` PTX | NATIVE `REDG.E.ADD.F16x2`, within 12% of u32 |
| `atomicAdd(addr, 1u)` no-return | emits `ATOMS.POPC.INC.32` — 2.5× speedup at warp-broadcast |
| `red.add.global` (no-return REDG) vs ATOMG with return | **25× faster** (32 cy vs 790 cy) |

### Contention (148 CTAs × 128 threads)

| Pattern | Throughput Gops/s | vs 1-hotspot |
|---|--:|--:|
| 1 hotspot (single addr, all 18,944 threads) | 49.1 | 1× baseline |
| **N=2 addresses** | 1.69 | **29× SLOWER** (real anomaly: L2 atomic-unit merging is lost, addresses serialize on same L2 slice) |
| N=4 | 4.4 | 11× slower |
| N=8 | 9.9 | 5× slower |
| N=16 | 9.7 | 5× slower |
| N=64 | 14.9 | 3.3× slower |
| N=256+ | 22+ | 2.0× slower (asymptote) |
| Per-warp clean (`addr_idx = warpId`) | 53.7 | 1.09× **FASTER** |
| Per-CTA clean | 609 | 12.4× **FASTER** |
| Coalesced unique-per-lane | 221 | 0.023 atom/cy/lane |

### Scope penalty (apples-to-apples)

| Comparison | Penalty |
|---|--:|
| `.relaxed` vs `.acq_rel`, warp-contend | 2.03× (738→1,501 cy) |
| `.relaxed` vs `.acq_rel`, chip-wide | 2.22× |
| `.cta` vs `.gpu` vs `.sys` for L2-hit data | FREE (no per-scope tax when L2-resident) |

### SASS attribution (compiler picks one of three)

- `REDG.E.ADD` — return value discarded
- `ATOMG.E.ADD` — return value used, default scope
- `ATOM.E.ADD` — some scoped variants

When pulling ncu metrics, capture both `lts__t_sectors_op_red` AND `lts__t_sectors_op_atom`.

`atom.global.cas` SASS emit is `STRONG.SYS` (system scope, NVLink-visible) — not `STRONG.GPU` as some references claim.

### Histogram / reduction design rules

1. Single global counter is surprisingly optimal if you need ONE number (L2 same-address merging dominates).
2. **Small counter arrays (N=2–32) are the worst case** — pick N=1 or N≥256.
3. Per-warp / per-SM privatization (unique cacheline per warp) beats sharing.

---

## 11. Warp / cluster cooperative primitives

| Op | GOps/s chip | SASS | Notes |
|---|--:|---|---|
| `vote.sync.ballot.b32` | 7,320 | VOTE.ANY → R | 1 SASS |
| `vote.sync.{all,any,uni}.pred` | 3,315 | ISETP+VOTE.ANY+SELP | 2.2× slower than ballot |
| `shfl.sync.bfly.b32` | 5,576 | SHFL.BFLY | pipe_lsu 1.00 |
| `redux.sync.min.u32` | 6,923 | CREDUX.MIN | matches alu+fmaH at 1.92 |
| `redux.sync.add.u32` | 3,107 | REDUX.SUM | 2.2× slower than min/max |
| `__syncwarp` | ~free | (no SASS in some cases) | |
| `__ballot_sync` | — | — | 29 cy |

Warp-tree min/add via shfl is **3–7× slower** than the corresponding `redux.sync` — always use the hardware reduction.

---

## 12. MUFU transcendentals

Per-warp throughput (single warp, ILP=16, other SMSPs idle):

| PTX | SASS | cy/op/warp | rate (op/cy/SM) |
|---|---|--:|--:|
| `ex2.approx.f32` | MUFU.EX2 | **4.28** | **0.93** |
| `tanh.approx.f32` | MUFU.TANH | 8.07 | 0.50 |
| `sin.approx.f32` | MUFU.SIN | 8.56 | 0.47 |
| `cos.approx.f32` | MUFU.COS | 8.56 | 0.47 |
| `rsqrt.approx.f32` | MUFU.RSQ | 8.74 | 0.46 |
| `sqrt.approx.f32` | MUFU.SQRT | 8.74 | 0.46 |
| `lg2.approx.f32` | MUFU.LG2 | 8.74 | 0.46 |
| `rcp.approx.f32` | MUFU.RCP | 9.09 | 0.44 |
| `ex2.approx.ftz.bf16x2` | MUFU.EX2.BF16x2 | — | same op-throughput at half dispatch pressure |

**EX2 is uniquely 2× faster than every other MUFU op** — designed for activation functions. Latency at low ILP: RCP=44, SIN=24, EX2=18; need ILP ≥ ~5 (EX2) or ~10 (SIN) to saturate.

Practical:
- Softmax: prefer `ex2` (`exp = ex2 × ln(2)`).
- Normalization: `rsqrt × x` over `sqrt → rcp`.
- `tanh.approx` is 8 cy native (cheaper than synthetic `(ex2(2x)-1)/(ex2(2x)+1)` at ~22 cy).
- For division: `__fdividef(a, b)` (= `div.approx`, 5.5 cy) is **3× faster than `rcp(b) × a`**.

---

## 13. mma.sync legacy tensor

| Path | TFLOPS | Conditions |
|---|--:|---|
| `mma.sync.m16n8k16` BF16/FP16 | **571** | 99.5% pipe_tensor, `@1942` |
| `mma.sync.m16n8k8` TF32 | **285.7** | `@1942` |
| `mma.sync.m16n8k32` FP8 e4m3 (emulated F2FP+HMMA) | **309** | `@1942`; SASS = HMMA preceded by F2FP (no native QMMA) |
| `mma.sync.m16n8k32` INT8 IMMA | **142.4 TOPS** | `@1942` |

`pipe_tensor` ncu metric is the right counter for mma.sync (HMMA family). It does **not** measure tcgen05.mma.

The naïve `bench_mma_all_precisions.cu` OP=3 collapses FP8 to 2 HMMA + 1056 FADD per iter and reports false 2,163 TFLOPS — always anti-DCE chain inputs for FP8.

---

## 14. tcgen05.mma — NVFP4 working path

Verified path on this rig:

```
tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16
    [d_tmem], [a_tmem], b_desc, idesc, [scale_A_tmem], [scale_B_tmem], pred;
```

SASS: **`UTCOMMA.BLOCK16`**.

| Field | Value |
|---|--:|
| K | 64 |
| cy/MMA | 128.001 |
| TFLOPS/SM | 66.58 |
| Chip throughput | **9.26 PFLOPS @1942** = 92.6% of 10 PF spec, 98.4% of theoretical at observed clock |
| Correctness | 15/15 (A, B, sA, sB) tuples bit-exact (D[0] ∈ {0, 16, 288, 576, 1152, 2304, 18432, 36864}) |

### What works in PTX 13.2.78

| Syntax | Status |
|---|---|
| `kind::mxf4nvf4.block_scale.block16` | ✅ compiles + runs (UTCOMMA.BLOCK16) |
| `kind::mxf4.block_scale.block32` | compiles, **crashes at runtime** with illegal-instruction |
| `kind::mxf4` (default = `.block32`) | rejected by ptxas |
| `kind::mxf8f6f4` | rejected by ptxas |
| `kind::f8f6f4.block_scale` | rejected (cannot combine) |
| `kind::i8` | rejected on sm_103a (B300 has no IMMA in tcgen05 — use FP8) |

### tcgen05.cp setup (smem → TMEM for A matrix)

| Shape | Status |
|---|---|
| `128x128b` | ✅ works (128 rows × 16 B = 2,048 B) |
| `128x256b` | ✅ runs cleanly with 8 KB smem |
| `4x256b` | ✅ runs |
| `64x128b` / `32x128b` | needs `.warptype` modifier |
| `64x256b` / `32x256b` / `32x32b` | invalid syntax |

For K=64 FP4 (32 B/row), do two `128x128b` copies at column offsets +0 and +4. Requires ≥ 32 KB smem.

### K=96 ULTRA via idesc bit 31

**Does NOT add MACs.** K=64 D[0] = 288 and K=96 D[0] = 288 (bit-identical with same inputs). If K=96 had worked, D[0] would have been 432. The `kind::mxf4` form (proper K=96 path) is rejected by ptxas 13.2.78 — wait for newer NVCC.

### Methodology notes for tcgen05.mma

1. `idesc` must use UMMA::InstrDescriptor bit layout (sparse_id2_, c_format_, a_format_, b_format_, n_dim_, m_dim_). `idesc=0` is invalid.
2. Smem matrix descriptor: layout_type=0 (no swizzle): LBO=16, SBO=128.
3. `tcgen05.alloc/dealloc/relinquish` are `.sync.aligned` — must be called by ALL threads in the warp; behind `if (tid==0)` deadlocks.
4. Real `mbarrier.init`'d 64-bit slot required for `tcgen05.commit.mbarrier::arrive::one.b64`.
5. M=256 requires `cta_group::2`.

---

## 15. TMA (cp.async.bulk)

Issue cost is regime-dependent:

| Mode | cy/TMA |
|---|--:|
| Pure single-issue, size-independent 16 B–8 KB | ~65 |
| Amortized in N-batch (1 mbarrier × N TMAs) | 48–50 |

Per-SM peak engine throughput **240–260 GB/s/SM** (independent of L2 vs HBM source). Reached by:
- 64 KB tile × DEPTH=3 → 252 GB/s/SM
- 32 KB × DEPTH=4 → 248
- 8 KB × NT=12 × DEPTH=2 → 255

### TMA vs LDG max-tuned head-to-head

| Regime | LDG.E.128 | TMA cp.async.bulk | Winner |
|---|--:|--:|---|
| L2-hit (WS = 64 MiB) | 18.25 TB/s | **20.49 TB/s** | TMA (+12%) |
| DRAM-cold (WS = 4 GiB) | **7.41 TB/s** (96.5% SoL) | 7.32 TB/s (95.4%) | tied |

### Tuning recipes

**TMA winning recipe**: DEPTH=2, bytes/iter ≥ 64 KiB/CTA, broad ridge (any 4–32 KB tile works), ≥ 4 waves (~592 CTAs).

**LDG winning recipe**: BS=128, ≥ 8,192 CTAs, LDG.E.128 (256-bit per inst), per-warp 1-KB bursts. BS=256/512/1024 all 30% worse.

### Mechanism

TMA's L2-hit advantage comes from its wider effective burst (descriptor encodes 64–128 B at-a-time vs LDG's 32 B per inst); HBM scheduler already coalesces LSU bursts so neither path beats the other on DRAM.

---

## 16. DSMEM (cluster shared memory)

`ld.shared::cluster.u32` compiles to **`LD.E`** (global LSU path), not LDS. This is the mechanical reason for the 9× latency penalty.

### Latency vs cluster size and ILP

| Cluster | Single-chain cy/load |
|---:|--:|
| 2 | 222 |
| 4 | 207 |
| 8 | 207 |
| 16 (non-portable, opt-in) | 231 |

| ILP (chains/warp) | cy/load (cluster=8) |
|---:|--:|
| 1 | 207 |
| 4 | 52 |
| 8 | 26 |
| 16 | 13 |
| **32** | **9** (LDS-equivalent) |

### Width

| Width | SASS | cy/load (cluster=2) | cy/byte |
|---|---|--:|--:|
| u32 | LD.E | 222 | 55.5 |
| .v2.u32 | LD.E.64 | 257 | 32.1 |
| .v4.u32 | LD.E.128 | 261 | 16.3 |

v4 is **3.5× more efficient per byte** — always prefer v4.

### Throughput (sustained, fenced)

| Cluster | Per-cluster GB/s | 18-cluster aggregate |
|---:|--:|--:|
| 2 | 69 | — |
| 4 | 139 | 105/cluster (sub-linear) |
| **8** | **278** | **117/cluster (2.12 TB/s)** |
| 16 | 25 | similar |

Chip aggregate ceiling at v4 × cluster=8 × max-tuned: **~2.4 TB/s write, ~1.9 TB/s read**.

### Other facts

- DSMEM **bypasses L2** in both directions (~0.04% of byte volume hits L2).
- Per-GPC silicon variation **20%** (GPC2 fastest at 189 cy, GPC1 slowest at 229 cy).
- R+W concurrent fabric is shared — no separate R/W channels (R+W=124 GB/s/cluster vs W-only 126).
- Fence cost **~1,500 cy fixed**, independent of N_STORES — amortize by accumulating stores per fence.
- Cluster=16 IS achievable via `cudaFuncAttributeNonPortableClusterSizeAllowed` (only 11% slower per access than cluster=8).

---

## 17. Predication / divergence

Pipe rate is **independent of active-lane count within 1%**:

| Active lanes | inst/cy/SM | % pipe_fma peak |
|---:|--:|--:|
| 32 of 32 | 2.91 | 72.6% |
| 16 of 32 | 2.94 | 73.4% |
| **1 of 32** | **2.94** | **73.5%** |

Implications:
- Predication does NOT save throughput.
- Warp specialization (`elect.sync` → 1 lane works) does NOT free pipe slots.
- What predication DOES save: register-read traffic, write-back to masked-off lanes, semantic correctness.

A simple 2-way `if` is **faster than no-branch** (compiler emits `selp`, no real branch). True divergence appears only when the compiler can't predicate (table lookup, function pointer, switch).

---

## 18. Cache control / prefetch hints

| PTX | SASS | cy | Effect |
|---|---|--:|---|
| `prefetch.global.L1` | CCTL.E.PF1 | 2 | async prefetch |
| `prefetch.global.L2` | CCTL.E.PF2 | 2 | async prefetch |
| `applypriority.global` | CCTL.E.DML2 | 2 | demote hint |
| `discard.global.L2` | CCTL.E.RML2 | 2 | evict hint |
| `ld.global.L1::evict_last/first` | LDG.E.{EL,EF} | 2 | LDG with hint |
| Idle CCTL.IVALL (no prior loads) | CCTL.IVALL | 2.83 | invalidate L1 |
| LD L1-hit + CCTL | — | 25 | drain wait |
| LD L2-hit + CCTL | — | 83 | drain wait |
| LD DRAM + CCTL | — | 902 | drain wait |
| 1 ST + CCTL (acquire ignores stores) | — | 9 | — |

**`.L2::256B` cache hint** gives **40% DRAM BW boost** for sparse-but-spatially-local LDG: stride-256B 4B-load reaches **7.06 TB/s = 92% HBM SoL** (vs 5.07 baseline). Compiler emits `LDG.E.LTC256B`. Use whenever consecutive threads in a warp span a full 256B+ DRAM sector.

**`.ca` beats `.cg` by 1.88×** for L1-fitting workloads (`.ca` = 13.13 TB/s L1TEX vs `.cg` = 6.97). Always prefer `.ca` over `.cg` when WS ≤ 228 KB.

---

## 19. Host API costs

| Operation | µs |
|---|--:|
| `cudaGetLastError` | 0.011 |
| `cudaEventElapsedTime` | 0.037 |
| `cudaStreamWaitEvent` (host enqueue) | 0.13 |
| NVTX push+pop (no profiler) | ~0 |
| `cudaMallocAsync` + `FreeAsync` cycle | 0.4–1.2 |
| `cudaMalloc` | 18 |
| `cudaFree` | 20 |
| Kernel launch `<<<>>>` (pipelined) | **2.05** (flat across cluster sizes 1/2/4/8) |
| `cudaLaunchKernelEx` + PSS | 1.47 |
| cudaGraph launch (1,000 kernels) | 0.56 / kernel |
| `cudaGraphExecUpdate` | 0.15 |
| `cudaStreamSynchronize` (after tiny kernel) | 6.3 |
| `cudaDeviceSynchronize` (idle) | 1.3 |
| CUDA cold start (cuInit → first kernel) | 326 ms |
| NVRTC compile | 6 ms warm |
| `cuLibraryLoadData` | 14 (6.5× faster than cuModule) |

Kernel-size table (per-iter event mode, BS=256, 1 CTA):

| Kernel inst | µs |
|--:|--:|
| 10 | 2.05 |
| 100 | 2.05 |
| 1,000 | 4.10 |
| 4,000 | 10.25 |

---

## 20. Roofline ridges (operational intensity)

| Compute path | Ridge OI (FLOP/byte) |
|---|--:|
| Scalar FFMA | 18 |
| FP16 tensor (tcgen05) | 314 |
| FP8 tensor | 628 |

Most ML inference ops sit below OI = 1 → memory-bound → fusion is king.

---

## 21. Dual-issue patterns

**FFMA2 + LOP3 1:1 is the dual-issue sweet spot.** It saturates **all three pipes** (fmaH 98% / fmaL 97% / alu 97%) and yields **314 useful ops/SM/cy** vs scalar FFMA + LOP3's 187:

| Path | Dispatch use | FP32 FLOPS/SM/cy | LOP3 ops/SM/cy | Total useful |
|---|--:|--:|--:|--:|
| Scalar FFMA solo | 4.00 | 256 | 0 | 256 |
| Scalar FFMA + LOP3 | 3.95 | 128 (halved!) | 30 | 187 |
| FFMA2 solo | 2.04 | 256 | 0 | 256 (alu idle) |
| **FFMA2 + LOP3 1:1** | **3.94** | **252** | **31** | **314 ← winner** |
| FFMA2 + LOP3 2:1 | ~3.0 | 256 (full) | 16 | 272 (LOP3 free) |
| LOP3 solo | 2.05 | 0 | 64 | 64 |

Mechanism: FFMA2 occupies fmaH AND fmaL per inst but uses only 1 dispatch slot — leaving 2.0 slots free for ALU. Scalar FFMA needs all 4.0 slots for the same FLOPS.

The 4.00 dispatch ceiling is unchanged. FP32 FLOPS still capped at 256/SM/cy. The win is "ALU work as a free side dish".

`.reuse` operand-cache annotation: scalar FFMA carries it on **99.9%** of instructions (1023/1024 in §0 peak); FFMA2 across 5 configs ranges **82.8%–99.2%**. To approach FFMA2 peak the compiler MUST find operand-reuse opportunities.

**Other clean co-issue patterns**:
- u64 ADD: pipe_alu (IADD3) + pipe_fmaheavy (IMAD.X) saturate together → 64 u64-adds/SM/cy.

**Compute-mem overlap**: cold DRAM load is 882 cy → ~225 free FFMAs can overlap it (warm L2 = 335 cy → ~85 FFMAs).

---

## 22. Register spilling

Cy/FFMA in MIN_BLOCKS sweep (lower is better):

| MIN_CTAS/SM | Avail regs/thread | N_LIVE=16 | =32 | =64 | =128 |
|---:|--:|--:|--:|--:|--:|
| 1 | ~232 | 2.25 | 1.79 | 1.68 | 1.61 |
| 2 | ~116 | 2.25 | 1.79 | 1.68 | 1.61 |
| 4 | ~58 | 2.25 | 1.79 | 1.68 | 1.61 |
| 8 | ~29 | 2.25 | 1.79 | 1.68 | 1.61 |
| **16** | **~14** | 2.25 | 1.79 | 1.68 | **2.44 (SPILL)** |

Spill penalty: ~50% slowdown when the compiler can't fit live values in registers. Avoid `MIN_CTAS_PER_SM > 8` unless verified low-pressure.

---

## 23. B300 vs H100 vs A100

| Spec | A100 | H100 | **B300 SXM6** | B300/A100 | B300/H100 |
|---|--:|--:|--:|--:|--:|
| SMs | 108 | 132 | **148** | 1.37× | 1.12× |
| FP32 TFLOPS scalar | 19.5 | 67 | **75** | 3.8× | 1.12× |
| FP16 tensor TFLOPS | 312 | 990 | **2,325 spec / 2,034 cuBLAS** | 6.5× | 2.1× |
| FP8 tensor TFLOPS | — | 1,979 | **4,651** | — | 2.3× |
| FP4 tensor PFLOPS | — | — | **9.26 measured / 10 spec** | — | — |
| HBM capacity | 80 GB | 80 GB | **268 GB** | 3.4× | 3.4× |
| HBM bandwidth | 2.0 TB/s | 3.35 TB/s | **7.4 TB/s** | 3.7× | 2.2× |
| L2 cache | 40 MB | 50 MB | **126.5 MB** | 3.2× | 2.5× |
| NVLink BW (bidi) | 600 GB/s | 900 GB/s | **956 GB/s** | 1.6× | 1.06× |
| SM clock (boost) | 1,410 | 1,830 | **2,032** | 1.44× | 1.11× |
| TDP | 400 W | 700 W | **~490 W tensor / 1,100 W max** | 1.23× | 0.70× |
| Compute capability | sm_80 | sm_90 | **sm_103a** | — | — |

---

## 24. Architectural facts (non-obvious)

These are load-bearing surprises that aren't on most spec sheets — every one is verified on this rig.

1. **`cvt.rni.sat.u8.f32` (F2IP.U8) is 4× faster than `cvt.rni.sat.s8.f32` (F2I.S8)** — alu vs xu pipe. Use the U8 form when the format permits.
2. **`atomicAdd(addr, 1u)` no-return → `ATOMS.POPC.INC.32`** (compiler counts active lanes via popcount). 2.5× speedup at warp-broadcast contention. Only fires for the literal constant `1`.
3. **`red.add.global` (no-return REDG) is 25× faster than `atom.global.add` with return.** 32 cy vs 790 cy.
4. **MUFU.EX2 is 2× faster than every other MUFU op** (4 cy vs 8 cy at saturation). Use for activation functions.
5. **`MUFU.EX2.BF16x2` packs 2 EX2 results per instruction** — same op-throughput as f32 EX2 at half dispatch pressure.
6. **`min.f32` × 2 chained → ONE `FMNMX3` SASS** (Blackwell 3-input fused FP min/max) → 128 logical mins/SM/cy.
7. **u64 ADD demonstrates clean alu+fmaH co-issue** (IADD3 + IMAD.X both saturate together) → 64 u64-adds/SM/cy.
8. **CCTL.IVALL is essentially FREE on idle pipeline (~3 cy)** — observed fence cost is the drain-wait for in-flight loads.
9. **`release.gpu` drain is SM-WIDE** — drains all co-resident CTAs' loads + stores.
10. **`cp.async` (LDGSTS) bypasses acquire-fence drain** unless `commit_group` is issued first.
11. **`.L2::256B` cache hint = 40% DRAM BW boost = 92% HBM SoL** at stride-256B 4B reads.
12. **`.ca` beats `.cg` by 1.88×** at L1-fitting WS (NOT 1.25× as some references say).
13. **`ld.const.u32` (LDC.32) dispatches via the ADU pipe**, not LSU. ADU peak 0.5 inst/SM/cy. Use BS=512 to saturate. 17.99 TB/s effective via 31.7× broadcast amplification.
14. **`ld.shared::cluster` compiles to `LD.E` (global LSU path)** — DSMEM is 9× slower than local SMEM single-chain. Hide it with ILP=32 to recover LDS-equivalent 9 cy/load.
15. **B300 SXM6 AC GPC topology = 9 × 16 SMs + 1 partial × 4 SMs = 148.** "8 GPCs" is wrong.
16. **Per-GPC silicon variation = 20%** in DSMEM latency (GPC2 fastest at 189 cy, GPC1 slowest at 229).
17. **FFMA2 + LOP3 1:1 dual-issue saturates 3 pipes** for 314 useful ops/SM/cy (vs 187 for scalar FFMA + LOP3).
18. **`__syncthreads` empirical formula = `22 + 2W` cy** (NOT `12 + 2W` as catalog says) — fixed barrier-instantiation overhead missed.
19. **DFMA latency is 63.9 cy, not 92 cy.** And DFMA is NOT pipelined — 4 chains give zero ILP benefit; FFMA + ALU can co-issue freely during the window.
20. **`redux.sync.{add,or,and,xor}.b32` is 2.4× SLOWER than `min/max`** — different SASS family (REDUX.SUM/OR/AND/XOR via ADU vs CREDUX.MIN/MAX via alu+fmaH).
21. **First-fence-after-write tax is FIXED, not linear** — drain happens once per fence, subsequent writes ride the open drain.
22. **mma.sync FP8 on sm_103a is EMULATED via F2FP+HMMA** (no native QMMA opcode). For native FP8, use tcgen05.mma.
23. **`tcgen05.alloc/dealloc/relinquish` are `.sync.aligned`** — must be called by ALL threads; behind `if (tid==0)` deadlocks.
24. **`tcgen05.mma kind::i8` is rejected on sm_103a** — B300 has no IMMA in tcgen05.mma. Use FP8 for INT8 inference.
25. **Cluster launch overhead = 2.05 µs flat across cluster sizes 1/2/4/8** in pipelined mode.
26. **`__fdividef(a, b)` is 3× faster than `rcp(b) × a`** — use for f32 division.
27. **2-way `if` branches are FASTER than no-branch** because the compiler emits `selp` (no real branch).

---

## 25. Footguns

1. **Default `LDG.E` hits L1 even for "DRAM" tests.** To actually measure DRAM latency, use `ld.global.cg` (or `.cs`/`.lu`) AND chain over a Sattolo-shuffled WS exceeding 126.5 MB.
2. **`sm__sass_data_bytes_mem_shared_op_ld.sum` reports WARP-aggregated bytes.** Naive 16 B/inst accounting undercounts SMEM bandwidth by 32×.
3. **`lts__t_bytes` undercounts LDG L2-hit by 2.7×** (MSHR/crossbar dedup). Use `l1tex__t_bytes` for LDG, `lts__t_bytes` for TMA. Mixing them across paths gives wildly inconsistent "% of L2 SoL" numbers.
4. **`-lgc 2032` silently clamps to ~1920 MHz** (firmware paradox) — your "boost lock" is actually base lock.
5. **Stuck-at-1005-MHz silent floor**: leftover process or driver state can park clocks at 1005 MHz with no warning. Sample with `nvidia-smi -q` during your run, and `-rgc` if needed.
6. **`pipe_tensor` ncu metric does NOT measure tcgen05.mma** — only legacy mma.sync (HMMA family). Use wall-clock × cy/MMA × ops/MMA + SASS UTC*MMA counts for tcgen05.
7. **`scalar fma.rn.f16/.bf16` emits HFMA2 with one half wasted** — gives ½ the per-useful-FLOP throughput of `fma.rn.f16x2`. To hit 70.4 TF, use the packed form (or `__half2` overloads).
8. **Smem capacity confusion**: `sharedMemPerBlock` (default) = 48 KB; `sharedMemPerBlockOptin` (max per CTA) = 227 KB; `sharedMemPerMultiprocessor` (per-SM HW max) = 228 KB. Three different caps; don't conflate.
9. **Atomic SASS attribution**: the compiler picks `REDG.E` (no return), `ATOMG.E` (return + default scope), or `ATOM.E` (some scoped variants). For ncu, capture both `lts__t_sectors_op_red` AND `..._op_atom`.
10. **DCE in microbenchmarks**: chain-feedback patterns let the compiler eliminate 32× of an LDS loop body even with anti-DCE store. Use independent loads with loop-counter-derived addresses + unconditional store.
11. **`bench_tma_acquire_v2.cu` silently zero-corrupts C[0]** when `data_xor == seed`. Always pass `-1 12345`.
12. **`init` and `main` kernel arg conflict in QuickRunCUDA**: same `-0 -1 -2` slots are passed to both kernels. If `iters` (main) collides with an `init` param, the working set silently shrinks. Pack `init` params into one slot via bit-shift.
13. **`-use_fast_math` is on by default in QuickRunCUDA's NVRTC**: all FFMA become `.FTZ`. Subnormal handling can't be measured without removing this flag.
14. **`.kind::i8` not supported on sm_103a** for tcgen05.mma — fall back to FP8 (4,651 TF) or dp4a SIMD (54.5 TOPS, 85× slower).
15. **Concurrent processes on a shared GPU silently inflate cy/MMA up to 8.5×** even at locked clock. Always `pkill -9` and verify GPU 0% utilization before microbenches.
16. **`"5.57 cy/iter for pure FFMA2"` is wrong** — real issue-bound peak is **2.14 cy/inst** (NC=2, 1 warp); latency-bound is 4.03 cy/inst (NC=1 RAW); chip-level 0.5 cy/inst per SMSP.
17. **The `tcgen05.mma 100K-iter throttling cliff DOES NOT REPRODUCE** under clean conditions — flat 64–67 cy/MMA across 5K → 100K iters at any unroll factor. The original "cliff" was likely contamination from concurrent processes.
18. **NVRTC vs offline `nvcc` SASS can differ** — verify with the actual NVRTC-emitted cubin (`output.cubin` written by QuickRunCUDA), not a separate nvcc invocation.

---

## TODO — needs validation before quoting

Each item: **what** the claim is — **why uncertain** — **how to validate**. Mark `[x]` with a note when resolved.

### High-value to close

- [ ] **tcgen05.mma per-format throughput** (BF16/FP16/TF32/FP8 dense/FP8 sparse) — only NVFP4 K=64 path has been re-run on this rig. Catalog quotes 2,325 / 1,163 / 4,651 / 7,439 TF respectively from a separate self-consistent linear-scaling table, but no fresh measurement here. **Validate**: build `bench_tcgen05_multiformat.cu` exercising all 5 kinds at M=128 N=256, 10K iters, single warp/SM; cross-check chip-wide via wall-clock × 148.
- [ ] **TMEM read/write throughput** (catalog claims 55.92 TB/s read / 97.93 TB/s write) — likely includes a broadcast amplification artifact analogous to LDC.32. **Validate**: write a TMEM-only kernel (`tcgen05.alloc` + `tcgen05.{ld,st}` + mbarrier setup); compare per-warp vs chip-wide; check ncu `lts__t_bytes` to detect L2 absorption.
- [ ] **Power efficiency table** (TF32 = 3,531 TF/kW, FP16 = 6,597, FP4 = 15,000) — catalog claim from one set of GEMMs, no rigor-clean re-measurement. M11 vs `16_power_clock` show a 2× discrepancy in some pipe-power claims. **Validate**: NVML power sampling at 100 ms intervals across a 10 s sustained kernel per format; report mean with idle baseline subtracted.
- [ ] **L2 wire bandwidth** (catalog claims 13.30 / 23.85 / 30 TB/s split by access pattern) — TMA max-tuned hits 20.49 TB/s and LDG max-tuned 18.25, both exceeding the "13.3 wire" claim by 37–54%. **Validate**: a kernel that pins the L2 wire (no DRAM reads) at every access pattern variant; reconcile against TMA/LDG max-tuned.
- [ ] **Cluster launch overhead deep dive** (catalog §57/§58 separate from the 2.05 µs pipelined number we have) — `cudaLaunchKernelEx` + PSS at 1.47 µs is an aggregated claim; want a clean per-cluster-size sweep at fixed work-per-CTA. **Validate**: existing `bench_launch_overhead` test, sweep cluster ∈ {1,2,4,8} × CTA-count.
- [ ] **Multi-GPU NVLink-attached `__threadfence_system`** — single-GPU is 1,727 cy / 850 ns; 2-GPU rig adds one NVLink coherence round-trip → 2,806 cy. Need re-run when both GPUs visible (`nvidia-fabricmanager` failed since 2026-04-17 → only GPU 0 visible to CUDA). **Validate**: restore fabric manager; re-run `bench_fence_costs.cu` with 2 GPUs.
- [ ] **B6 DRAM write peak under clean isolation** — catalog claims 7.09 TB/s standard / 7.57 TB/s contested. Last attempt used wrong launch config (BS=512 instead of catalog recipe BS=256, causing 2-wave queueing + L2 pollution between waves). **Validate**: `bench_dram_peak.cu` OP=5 (256-bit STG `st.global.v8.u32`), `-t 256 -b 1184` (BS=256 = 8 CTAs/SM × 1 wave), WS ≥ 4 GB, isolated GPU; ncu `dram__bytes_write.sum.per_second` cross-check.

### Catalog-preserved but lower-priority

- [ ] **Branch divergence patterns table** (no-divergence 28 cy, 32-way LUT 153 cy = 5.5× slowdown) — single source, not re-tested. **Validate**: `bench_divergence.cu` sweeping divergence width 1–32.
- [ ] **INT8 dp4a TOPS rates** (54.5 TOPS dp4a; 25.4 dp2a; 18.1 IMAD/IMAD.WIDE) — derived from issue rate × clock, not directly measured. **Validate**: anti-DCE chain INT8 dot-product loop, count IDP.4A SASS, divide measured time.
- [ ] **`acq_rel.sys` 17–37% overhead vs `sc.sys`** — catalog claim, no sweep. **Validate**: `bench_fence_scope.cu` sweeping (cta/gpu/sys) × (relaxed/acquire/release/acq_rel/sc).
- [ ] **TMA multicast on sm_103a** — catalog claims `cp.async.bulk.multicast::cluster` works despite cccl gating it to SM_90a/100a/110a, and quotes 14.9 TB/s aggregate (L2-resident-source amplification). **Validate**: build a multicast TMA kernel; verify SASS opcode emission; measure aggregate vs single-cluster baseline.
- [ ] **Atomic contention at large CTA counts** (catalog quotes 132 cy at 148 CTAs single-address — 2.6× slower than 32 CTAs) — likely correct trend but not re-run on this rig. **Validate**: `bench_atom_chip_scope.cu` sweep CTA ∈ {1, 2, 8, 32, 148}.
- [ ] **Cluster-launched contended atomic scope ladder** (atom.acq_rel.cluster = 1,646 cy = 48× relaxed) — different regime from §10 single-thread chain. **Validate**: cluster-launched atomic kernel sweeping op ∈ {add/release.cta/acquire.cta/acquire.gpu/release.cluster/acq_rel.cta/acq_rel.cluster}.
- [ ] **Iteration cliff at 100K MMAs** (cy/iter jumps 128 → 394 between 30K and 100K iters) — has been disproven for tcgen05.mma at single-warp under clean conditions (flat 64–67 cy/MMA), but catalog also reports it for some other regimes. **Validate**: was the cliff real in any specific kernel layout? Or pure contamination?
- [ ] **`MUFU.EX2.BF16x2` actual element-rate gain** — listed at "same op-throughput at half dispatch pressure" but the practical FLOP gain depends on whether co-issued work uses the freed dispatch slot. **Validate**: measure EX2.BF16x2 + LOP3 vs EX2.f32 solo at full occupancy.
- [ ] **`mbarrier.arrive` modifier latency table** — only `.shared.b64` (27 cy) and `.relaxed.cta` (8.1 cy) are nailed down; full modifier sweep (`.relaxed.gpu`, `.acquire.cta`, `.release.cta`, etc.) not done. **Validate**: `bench_mbarrier_costs.cu` sweep all modifier × scope combos.
- [ ] **L1 hit bandwidth at WS ≤ 1 MB** — catalog quotes 36.1 TB/s; current benches mix L1/L2. **Validate**: pure L1-resident kernel at WS = 256 KB with `.ca`, ncu `l1tex__t_bytes.sum.per_second`.
- [ ] **SHFL broadcast peak** — catalog L83 mentions "1.9 cy free" in some context vs general 7.46 cy. Need explicit broadcast vs general benchmark. **Validate**: `bench_shfl.cu` with broadcast variant (all lanes read same source) vs general bfly.
- [ ] **`atom.shared.add.f32` 97 cy emulation** (BSYNC + CAS loop) — catalog claim, no SASS confirmation here. **Validate**: dump SASS of `atomicAdd((float*)smem, val)`; count cy via clock64.
- [ ] **`MATCH.ANY` 375 cy claim** — single source, suspicious. **Validate**: isolated `match.any.sync.b32` chain.

### Methodology / infrastructure

- [ ] **NVRTC `-use_fast_math` impact on FP results** — currently forced on; can't measure subnormal handling. **Validate**: rebuild QuickRunCUDA without the flag, re-run F2I/FP CVT correctness with subnormals.
- [ ] **`cudaGraphExecUpdate` 0.15 µs vs `cudaLaunchKernelEx` 1.47 µs reconciliation** — both quoted, different setups. **Validate**: end-to-end "10K kernels via graph + ExecUpdate" vs "10K kernels via cudaLaunchKernelEx" benchmark with strict timing.
- [ ] **L2 partition/hash mapping** (catalog mentions "2 partitions with address hash" but no measured stride → partition function) — would unlock smarter L2-aware scheduling. **Validate**: synthesize address pattern that hits one partition vs alternating; observe L2 partition counters.
- [ ] **DSMEM at cluster=12** (catalog has 218 cy interpolated; data sparse 4 ↔ 16). **Validate**: extend `bench_dsmem.cu` cluster sweep to include 12.
- [ ] **NVLink directional asymmetry** (P2P W=718 GB/s vs R=820 — 14% gap) — needs reproduction now and explanation. **Validate**: when fabric manager works, re-run `multigpu/MGFenceBench.cpp` write/read directional sweep.
- [ ] **PCIe full-duplex 0.099 GB/s combined** is implausibly low if H2D and D2H individual are 0.057 each (sum should be ~0.114). **Validate**: simultaneous H2D + D2H stream pair, measure each direction's effective BW; reconcile.

### Architectural conjectures (unverified mechanism, observation correct)

- [ ] **Per-GPC silicon variation 20%** — observation is solid; mechanism (yield binning vs systematic per-GPC topology distance) unconfirmed. **Validate**: run the latency probe on a different B300 SXM6 AC chip; if pattern is the SAME GPC ordering, it's structural; if it's different, it's binning.
- [ ] **CTA scheduler fills smallest GPC first** — observed for one launch; not confirmed across launch geometries / persistent kernels / cluster launches. **Validate**: log `%smid` for first N CTAs across launch counts {16, 64, 148, 296, 1184}.
- [ ] **`release.gpu` SM-wide drain mechanism** — cost confirmed (compounds to 11,000+ cy at high occupancy); whether it's drain-of-all-CTA's-loads or just per-warp is fuzzy. **Validate**: vary co-resident CTAs while holding per-CTA load count fixed; isolate the drain-target.

---

**End.** For per-claim raw evidence (run command, SASS, ncu output, wall-clock data) see `JUSTIFIED_B300_PIPE_CATALOG.md` and `justifications/<id>.md`. For yes/no review of remaining uncertain catalog items see `REVIEW_CHECKLIST_B300.md`.
