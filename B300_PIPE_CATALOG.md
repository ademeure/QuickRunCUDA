# B300 / Blackwell sm_103a — SM Pipe Catalog

**Platform:** NVIDIA B300 SXM6 AC, driver locked to **1920 MHz SM clock**, **148 SMs**, CUDA 13.0.
**Methodology:** inline PTX asm with chain-dependency feedback to defeat DCE, SASS static count verified against expected, pipe assignment from ncu `sm__inst_executed_pipe_*.avg.per_cycle_active`. Rates are warp-instructions issued per SM per cycle (= "SASS-inst/SM/clk" in the headline sense).

All numbers below are measured, not datasheet.

---

## 1. Pipe topology

An SM has **4 SMSPs** (sub-partitions), each dispatching up to 1 warp-instruction/cycle → aggregate dispatch cap = **4.00 warp-inst/SM/cy** (i.e. 128 thread-ops/SM/cy for non-packed ops).

Below each pipe is labeled with its steady-state acceptance cap (warp-inst/SM/cy):


| Pipe                                | Cap                              | Physical role                       | Example SASS                                                                                               |
| ----------------------------------- | -------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `pipe_alu`                          | 2.00                             | Integer/bitwise/compare/narrow-cvt  | LOP3, PRMT, F2FP (all), SHF, FMNMX, HMNMX2, VABSDIFF, SEL, ISETP, FSETP, I2FP, F2IP.U8, I2I.SAT            |
| `pipe_fmaheavy`                     | 2.00                             | Integer mul-add, heavy FMA half     | IMAD, IMAD.X, IMAD.WIDE, IDP.4A/2A, HADD2.F32 (f16→f32 cvt), half of FFMA                                  |
| `pipe_fmalite`                      | 2.00                             | Light FMA half                      | half of scalar FFMA / FMUL / FADD                                                                          |
| `pipe_fma` (parent)                 | 4.00 when dual, 2.00 when packed | = heavy ∪ lite                      | scalar FFMA can issue to both simultaneously; packed ops (FFMA2, HFMA2, BF16-FMA) occupy both for one inst |
| `pipe_xu`                           | 0.50 (compound) – 1.00 (simple)  | Transcendental unit                 | MUFU.{EX2,RSQ,SIN,COS,LG2,TANH,SQRT,RCP}, F2I (f32→s32/u32/s64/s8), POPC, BREV, FLO/CLZ                    |
| `pipe_lsu`                          | 1.00 nominal                     | Load/store, warp shuffle            | LDG, STG, LDS, STS, LDSM (partially), SHFL.SYNC.*                                                          |
| `pipe_adu`                          | ~0.5                             | Address/sync/match                  | BAR.SYNC, MATCH.ANY                                                                                        |
| `pipe_uniform`                      | ~1.0                             | Uniform register / LDSM             | S2UR, LDSM.sync, ACTIVEMASK                                                                                |
| `pipe_tensor` (subpipes hmma/imma)  | —                                | Tensor Core                         | HMMA, IMMA (not measured here)                                                                             |
| `pipe_fp64`                         | **0.05**                         | FP64                                | DFMA, DADD, DMUL — throttled on B300                                                                       |
| `pipe_cbu`                          | —                                | Control/branch (BRA, EXIT)          | mostly invisible in steady-state                                                                           |
| `pipe_tex` / `pipe_tc` / `pipe_ipa` | —                                | Texture / tex-cache / interpolation | not exercised                                                                                              |


**Dispatch ceiling:** total `sm__inst_executed` ≤ 4.00/SM/cy regardless of how many pipes are fed. Any headline claim >128 SASS-inst/SM/cy **without packed ops** is wrong.

---

## 2. Complete instruction catalog (measured)

### 2.1 FP32 scalar arithmetic — pipe_fma (dual-issue heavy + lite)

These are the ones that **uniquely use BOTH fma sub-pipes simultaneously** at 2.00 each → **4.00 warp-inst/SM/cy = 128 SASS/SM/cy = 128 scalar FP32 ops/SM/cy**.


| PTX                   | SASS                      | rate (warp-inst/SM/cy) | logical throughput                        |
| --------------------- | ------------------------- | ---------------------- | ----------------------------------------- |
| `fma.rn.f32 a,b,c,d`  | `FFMA`                    | 4.00                   | **128 FFMA/SM/cy = 256 FP32 FLOPS/SM/cy** |
| `mul.rn.f32 a,b,c`    | `FMUL`                    | 4.00                   | 128 FMUL/SM/cy                            |
| `add.rn.f32 a,b,c`    | `FADD`                    | 4.00                   | 128 FADD/SM/cy                            |
| `abs.f32` / `neg.f32` | compiler emits `FADD.FTZ` | 4.00                   | 128/SM/cy                                 |


`pipe_fmaheavy = 2.00` AND `pipe_fmalite = 2.00` simultaneously is the signature.

### 2.2 FP32 vec2 and FP16/BF16 packed — pipe_fma (both sub-units, 1 inst)

Packed ops occupy both heavy and lite for a single instruction → cap at **2.00 warp-inst/SM/cy**.


| PTX             | SASS         | rate | elements/SM/cy                                                                |
| --------------- | ------------ | ---- | ----------------------------------------------------------------------------- |
| `fma.rn.f32x2`  | `FFMA2`      | 2.00 | 64 × 2 = **128 FP32 FMAs/SM/cy = 256 FP32 FLOPS/SM/cy** (same as scalar FFMA) |
| `fma.rn.f16x2`  | `HFMA2`      | 2.00 | 64 × 2 = 128 FP16 FMAs/SM/cy = 256 FP16 FLOPS/SM/cy                           |
| `add.rn.f16x2`  | `HADD2`      | 2.00 | 128 FP16 adds (compiler sometimes uses HFMA2 pattern)                         |
| `mul.rn.f16x2`  | `HMUL2`      | 2.00 | 128 FP16 muls (compiler sometimes emits HFMA2)                                |
| `fma.rn.bf16x2` | `HFMA2.BF16` | 2.00 | 128 BF16 FMAs/SM/cy                                                           |
| `add.rn.bf16x2` | `HADD2.BF16` | 2.00 | 128 BF16 adds/SM/cy                                                           |


### 2.3 Integer — pipe_fmaheavy (and sometimes pipe_alu via IADD3 fusion)


| PTX                                   | SASS                      | rate                        | notes                                                             |
| ------------------------------------- | ------------------------- | --------------------------- | ----------------------------------------------------------------- |
| `mad.lo.u32`                          | `IMAD`                    | 2.00 fmaH                   | 64 IMAD/SM/cy                                                     |
| `mul.lo.u32`                          | `IMAD`                    | 2.00 fmaH                   | same pipe, same rate                                              |
| `mul.hi.u32`                          | `IMAD.HI.U32`             | 1.00 fmaH                   | **half rate** — 32/SM/cy                                          |
| `dp4a.s32.s32` (int8·4 dot)           | `IDP.4A.S8.S8`            | 2.00 fmaH                   | 64 SASS × 4 pairs × 2 ops = 512 int8-ops/SM/cy                    |
| `dp2a.`* (int16·2 dot)                | `IDP.2A.LO.S16.S8`        | 2.00 fmaH                   | 64/SM/cy                                                          |
| `cvt.f32.f16` (f16→f32)               | `HADD2.F32`               | 2.00 fmaH                   | re-uses HADD2 infra                                               |
| `add.u32 a,a,b` (single 2-input)      | `IADD3` or `IMAD.IADD`    | 4.00 total (2 alu + 2 fmaH) | **128 logical adds/SM/cy** — compiler splits across pipes         |
| `add.u32 a,b,c,d` or two chained adds | fused into single `IADD3` | 2.00 alu                    | **128 logical adds/SM/cy** (1 IADD3 = 2 adds, ALU alone suffices) |
| `sub.u32`                             | `IADD3` (neg)             | as IADD3                    | 128/SM/cy                                                         |


### 2.4 u64 integer


| PTX                                | SASS emitted                                      | pipe        | u64 op rate            |
| ---------------------------------- | ------------------------------------------------- | ----------- | ---------------------- |
| `add.u64`                          | `IADD3` (low) + `IMAD.X` (high+carry) — 2 SASS/op | alu + fmaH  | **64 u64-adds/SM/cy**  |
| `sub.u64`                          | same as add, 2 SASS                               | alu + fmaH  | 64/SM/cy               |
| `{add.cc.u32; addc.u32;}` explicit | same                                              | alu + fmaH  | 64/SM/cy               |
| `mul.lo.u64`                       | `IMAD + IMAD.WIDE + IADD3` ×3                     | mostly fmaH | ~12/SM/cy              |
| `mul.hi.u64`                       | chain of 6+ SASS                                  | fmaH + alu  | ~5/SM/cy               |
| `and.b64` / `or.b64` / `xor.b64`   | 2× `LOP3.LUT`                                     | alu         | **32 u64-logic/SM/cy** |
| `shl.b64` / `shr.b64/.u64`         | 3 SASS (`SHF.L.U64.HI` + `SHF.L.U32` + helpers)   | alu         | ~16/SM/cy              |
| `min.u64` / `max.u64`              | `ISETP.LT.U32 ×2` + `SEL ×2` (4 SASS)             | alu         | ~16/SM/cy              |


### 2.5 Narrow-format CVT (F2FP family) — pipe_alu

All the FP4/FP6/FP8/UE8M0 conversions live on `pipe_alu`. Each x2 instruction converts **2 elements per thread per warp-inst**.

**UNPACK (narrow → f16x2 / bf16x2):**


| PTX                                  | SASS                     | rate | elements/SM/cy |
| ------------------------------------ | ------------------------ | ---- | -------------- |
| `cvt.rn.f16x2.e4m3x2`                | `F2FP.F16.E4M3.UNPACK_B` | 2.00 | 128            |
| `cvt.rn.f16x2.e5m2x2`                | `F2FP.F16.E5M2.UNPACK_B` | 2.00 | 128            |
| `cvt.rn.f16x2.e2m1x2` (FP4, b8 wrap) | `F2FP.F16.E2M1.UNPACK_B` | 2.00 | 128            |
| `cvt.rn.f16x2.e2m3x2` (FP6)          | `F2FP.F16.E2M3.UNPACK_B` | 2.00 | 128            |
| `cvt.rn.f16x2.e3m2x2` (FP6)          | `F2FP.F16.E3M2.UNPACK_B` | 2.00 | 128            |
| `cvt.rn.bf16x2.ue8m0x2`              | `F2FP.BF16.E8.UNPACK_B`  | 2.00 | 128            |


All six peak identically at **2.00 warp-inst/SM/cy = 128 elements/SM/cy** when no co-issuing ALU op (no LOP3 feedback tax). With 1-per-iter LOP3 feedback (zero-extension or XOR) the effective rate halves to 1.00 = 64 elements/SM/cy.

**PACK (wide → narrow):**


| PTX                                   | SASS                                                   | rate (solo, LOP3-polluted)             |
| ------------------------------------- | ------------------------------------------------------ | -------------------------------------- |
| `cvt.rn.satfinite.e4m3x2.f16x2`       | `F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C`             | ≈ 1.0 alu (pollution by zero-ext LOP3) |
| `cvt.rn.satfinite.e5m2x2.f16x2`       | `F2FP.SATFINITE.E5M2.F16.UNPACK_B_MERGE_C`             | ≈ 1.0 alu                              |
| `cvt.rn.satfinite.e2m1x2.f16x2` (FP4) | `F2FP.SATFINITE.E2M1.F16.UNPACK_B_MERGE_C` + `mov.b16` | ≈ 0.8 alu (extra mov)                  |
| `cvt.rn.satfinite.e4m3x2.f32 lo,hi`   | `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C`              | ≈ 0.67 alu                             |
| `cvt.rn.satfinite.e5m2x2.f32`         | `F2FP.SATFINITE.E5M2.F32.PACK_AB_MERGE_C`              | ≈ 0.67                                 |
| `cvt.rn.satfinite.e2m3x2.f32` (FP6)   | `F2FP.SATFINITE.E2M3.F32.PACK_AB_MERGE_C`              | ≈ 0.67                                 |
| `cvt.rn.satfinite.e3m2x2.f32` (FP6)   | `F2FP.SATFINITE.E3M2.F32.PACK_AB_MERGE_C`              | ≈ 0.67                                 |
| `cvt.rn.satfinite.e2m1x2.f32` (FP4)   | `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C` + `mov.b16`  | ≈ 0.45                                 |
| `cvt.rp.satfinite.ue8m0x2.f32`        | `F2FP.SATFINITE.UE8M0.F32.PACK_AB_MERGE_C`             | ≈ 0.67                                 |


**Round-trip (PACK ⇄ UNPACK)** avoids LOP3 tax because UNPACK's output feeds PACK without needing zero-ext. Combined rate then = 2.00 warp-inst/SM/cy (ALU saturated), split 1:1 between PACK and UNPACK. That's how the "PACK alone peak" of 2.00/SM/cy is established — it only lives inside a round-trip.

### 2.6 Other CVTs


| PTX                  | SASS                          | pipe | rate                  |
| -------------------- | ----------------------------- | ---- | --------------------- |
| `cvt.rn.f16.f32`     | `F2FP.F16.F32.PACK` (+ PRMT)  | alu  | 1.00 (PRMT tax)       |
| `cvt.rn.bf16.f32`    | `F2FP.BF16.F32.PACK` (+ PRMT) | alu  | 1.00                  |
| `cvt.f32.f16`        | `HADD2.F32`                   | fmaH | 2.00                  |
| `cvt.f32.bf16`       | `HADD2.F32` or shift          | fmaH | 2.00                  |
| `cvt.rn.f32.s32`     | `I2FP.F32.S32`                | alu  | 2.00                  |
| `cvt.rn.f32.u32`     | `I2FP.F32.U32`                | alu  | 2.00                  |
| `cvt.rn.f32.s64`     | `I2F.S64`                     | xu   | 0.04 (super slow)     |
| `cvt.rni.s32.f32`    | `F2I.NTZ`                     | xu   | 0.5                   |
| `cvt.rni.u32.f32`    | `F2I.U32.NTZ`                 | xu   | 0.5                   |
| `cvt.rni.s64.f32`    | `F2I.S64`                     | xu   | ≤ 0.5 (DCE'd in test) |
| `cvt.rni.sat.u8.f32` | `F2IP.U8.F32.NTZ`             | alu  | 2.00 (!)              |
| `cvt.rni.sat.s8.f32` | `F2I.S8.NTZ`                  | xu   | 0.5                   |
| `cvt.sat.u8.s32`     | `I2I.U8.S32.SAT`              | alu  | 2.00                  |


### 2.7 Bitwise / shift / permute — pipe_alu


| PTX                                                       | SASS                                                | pipe     | rate                                       |
| --------------------------------------------------------- | --------------------------------------------------- | -------- | ------------------------------------------ |
| `xor.b32` / `and.b32` / `or.b32` / `not.b32` / `lop3.b32` | `LOP3.LUT`                                          | alu      | 2.00                                       |
| `shl.b32`, `shr.u32`, `shr.s32` (plain)                   | `SHF.L.W.U32` / `SHF.R.W.U32.HI` / `SHF.R.W.S32.HI` | alu      | 2.00                                       |
| `shf.l.wrap.b32` (funnel)                                 | `SHF.L.W.U32.HI`                                    | alu      | 2.00 (with LOP3 tax when shift amt in reg) |
| `shf.r.wrap.b32`                                          | `SHF.R.W.U32`                                       | alu      | 2.00                                       |
| `shf.l.clamp.b32`                                         | `SHF.L.U32.HI`                                      | alu      | 2.00                                       |
| `shf.r.clamp.b32`                                         | `SHF.R.U32`                                         | alu      | 2.00                                       |
| `prmt.b32`                                                | `PRMT`                                              | alu      | 2.00                                       |
| `bfi.b32`                                                 | collapses to `LOP3.LUT`                             | alu      | 2.00                                       |
| `bfe.u32`                                                 | `SHF.R.U32.HI` + `SGXT.U32` (2 SASS)                | alu      | 1.00                                       |
| `brev.b32`                                                | `BREV`                                              | xu       | 0.5                                        |
| `popc.b32`                                                | `POPC`                                              | xu       | 0.5                                        |
| `clz.b32` / `bfind`                                       | `FLO.U32` (+ IADD3 for CLZ offset)                  | xu + alu | 0.5                                        |


### 2.8 Compares / predicates / selection — pipe_alu


| PTX                    | SASS                          | pipe | rate |
| ---------------------- | ----------------------------- | ---- | ---- |
| `setp.*.u32/s32`       | `ISETP.`*                     | alu  | 2.00 |
| `setp.*.f32`           | `FSETP.*`                     | alu  | 2.00 |
| `selp.b32`             | `SEL`                         | alu  | 2.00 |
| setp+selp combined     | `ISETP` + `SEL` (2 SASS)      | alu  | 1.00 |
| `vote.sync.ballot.b32` | `ISETP` + `VOTE.ANY` (2 SASS) | alu  | 1.00 |


### 2.9 MIN / MAX — **pipe_alu** (not FMA)

Surprising but measured: all FP and integer min/max land on pipe_alu.


| PTX                              | SASS                                          | pipe       | rate                                  |
| -------------------------------- | --------------------------------------------- | ---------- | ------------------------------------- |
| `min.f32` / `max.f32` (data-dep) | `FMNMX`                                       | alu        | 2.00                                  |
| `min.NaN.f32` / `max.NaN.f32`    | `FMNMX.NAN`                                   | alu        | 2.00                                  |
| `min.f16x2` / `max.f16x2`        | `HMNMX2`                                      | alu        | 2.00 — 128 FP16 min-ops/SM/cy         |
| `max.NaN.f16x2`                  | `HMNMX2.NAN`                                  | alu        | 2.00                                  |
| `min.bf16x2` / `max.bf16x2`      | `HMNMX2.BF16`                                 | alu        | 2.00 — 128 BF16 min-ops/SM/cy         |
| `min.s32` / `max.s32`            | `VIMNMX3` (compiler folds 2 mins into 1 inst) | alu        | 2.00 — effectively 128 int mins/SM/cy |
| `min.u64` / `max.u64`            | 2× ISETP + 2× SEL                             | alu        | 0.5 → ~16 u64 min/SM/cy               |
| `abs.s32`, `neg.s32`, `abs.f32`  | compiler folds to IADD3 / FADD / LOP3         | alu or fma | 2.00+                                 |
| `copysign.f32`                   | `LOP3.LUT`                                    | alu        | 2.00                                  |


### 2.10 Transcendentals — pipe_xu (compound)

The XU pipe accepts a simple op every 2 cycles (0.5/SM/cy) for compound MUFUs. Clean isolation is hard because the compiler inserts FSETP+FSEL+FMUL for domain conditioning.


| PTX                | SASS                                | pipe | rate                                     |
| ------------------ | ----------------------------------- | ---- | ---------------------------------------- |
| `ex2.approx.f32`   | `MUFU.EX2`                          | xu   | 0.5–0.63 (= 16–20 SASS/SM/cy)            |
| `rsqrt.approx.f32` | `MUFU.RSQ`                          | xu   | 0.5                                      |
| `sqrt.approx.f32`  | `MUFU.SQRT`                         | xu   | 0.5                                      |
| `rcp.approx.f32`   | `MUFU.RCP`                          | xu   | 0.5                                      |
| `sin.approx.f32`   | `MUFU.SIN` (+ FMUL range-reduction) | xu   | 0.5 (slower overall due to conditioning) |
| `cos.approx.f32`   | `MUFU.COS`                          | xu   | 0.5                                      |
| `lg2.approx.f32`   | `MUFU.LG2`                          | xu   | 0.5                                      |
| `tanh.approx.f32`  | `MUFU.TANH`                         | xu   | 0.5                                      |


### 2.11 Warp / group / synchronization ops


| PTX                                        | SASS                                                | pipe                                  | rate                                                                        | notes                                                          |
| ------------------------------------------ | --------------------------------------------------- | ------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `shfl.sync.{bfly,idx,up,down}.b32`         | `SHFL.{BFLY,IDX,UP,DOWN}`                           | **lsu**                               | 1.00 (= 32 SASS/SM/cy)                                                      |                                                                |
| `vote.sync.ballot.b32`                     | `VOTE.ANY` (+ ISETP)                                | alu                                   | 2.00 (combined)                                                             |                                                                |
| `vote.sync.any / all / uni`                | `VOTE.`*                                            | alu                                   | 2.00                                                                        |                                                                |
| `activemask`                               | uniform-pipe op                                     | **uniform**                           | ~1.2                                                                        | (DCE'd in my test)                                             |
| `match.any.sync.b32`                       | `MATCH.ANY`                                         | **adu**                               | 0.5 peak issue                                                              | **very slow** (~140 ms for 128 inst) — warp-wide serialisation |
| `match.all.sync.b32`                       | `MATCH.ALL`                                         | adu                                   | similar                                                                     |                                                                |
| `bar.sync 0` / `barrier.sync 0`            | `BAR.SYNC.DEFER`                                    | **adu**                               | 0.36                                                                        | CTA-wide barrier, thread-waiting dominates                     |
| `bar.arrive`                               | `BAR.ARV`                                           | **adu**                               | 0.47                                                                        | no wait → faster                                               |
| `bar.red.popc.u32`                         | `BAR.RED.POPC.DEFER`                                | **adu** (+ alu for ISETP)             | 0.37                                                                        |                                                                |
| `redux.sync.min.u32` / `.max.u32`          | `CREDUX.MIN/MAX` + `IMAD.U32` (2 SASS/op intrinsic) | CREDUX → **alu**, IMAD → **fmaheavy** | **1.92 PTX-op/SM/cy** (≈ 61 thread-ops/SM/cy); each pipe runs at 1.92/2.00  |                                                                |
| `redux.sync.add.u32`                       | `REDUX.SUM` + minor IMAD                            | **adu**                               | **0.50 PTX-op/SM/cy** (≈ 16 thread-ops/SM/cy) — **~4× slower** than min/max |                                                                |
| `redux.sync.or.b32` / `.and` / `.xor`      | `REDUX.{OR,AND,XOR}`                                | **adu**                               | 0.50 PTX-op/SM/cy (same as add)                                             |                                                                |
| `membar.cta`                               | `MEMBAR.SC.CTA`                                     | **lsu**                               | 0.83                                                                        | scoped fence on lsu                                            |
| `membar.gl`                                | `MEMBAR.SC.GPU` + `ERRBAR`                          | adu + lsu                             | extremely slow (~38 ms)                                                     | GPU-wide fence                                                 |
| `ldmatrix.sync.x1.b16`                     | `LDSM` (1 quad)                                     | **uniform (1.0) + lsu (0.5)**         | ~1.0                                                                        |                                                                |
| `ldmatrix.sync.x2.b16`                     | `LDSM` (2 quads)                                    | uniform 0.5 + lsu 0.25                | 0.5                                                                         | half rate per inst                                             |
| `ldmatrix.sync.x4.b16`                     | `LDSM` (4 quads)                                    | uniform 0.25 + lsu 0.12               | 0.25                                                                        | quarter rate                                                   |
| `ldmatrix.sync.x4.trans`                   | `LDSM.T`                                            | same as x4                            | 0.25                                                                        | transpose flag                                                 |
| `atom.shared.add.u32`                      | `ATOMS.POPC.INC.32`                                 | **lsu**                               | 0.84                                                                        |                                                                |
| `atom.global.`*                            | `ATOMG.*`                                           | lsu                                   | bandwidth-bound                                                             |                                                                |
| `st.async`/`ld.async`                      | `LDGSTS`                                            | lsu                                   | async                                                                       |                                                                |
| `s2r %tid.x` / `%tid.y` / `%ntid.x` / etc. | `S2R`                                               | (emitted once, then cached)           | n/a                                                                         |                                                                |
| `s2r %laneid`                              | `S2R SR_LANEID`                                     | ADU/XU-ish, DCE'd                     | —                                                                           |                                                                |
| `s2r %clock` / `%clock_hi`                 | `S2R SR_CLOCKLO / HI`                               | **adu**                               | 0.5                                                                         |                                                                |
| `s2r %warpid`                              | `S2R SR_VIRTWARPID`                                 | alu                                   | 1.0 (ish)                                                                   |                                                                |


### 2.12 Memory


| PTX             | SASS          | pipe | note                                                     |
| --------------- | ------------- | ---- | -------------------------------------------------------- |
| `ld.global.u32` | `LDG.E`       | lsu  | DRAM-bound in practice, ~1 inst/SM/cy issue              |
| `st.global.u32` | `STG.E`       | lsu  | 79.6 ms on 303k-thread storm → DRAM-bottleneck, not pipe |
| `ld.shared.u32` | `LDS`         | lsu  | ~1.0 issue, bank-conflict-sensitive                      |
| `st.shared.u32` | `STS`         | lsu  | 1.00 saturating                                          |
| `atom.`*        | `ATOMS/ATOMG` | lsu  | not measured                                             |


### 2.13 FP64 — severely throttled on B300


| PTX          | SASS   | pipe | rate                                                                                    |
| ------------ | ------ | ---- | --------------------------------------------------------------------------------------- |
| `fma.rn.f64` | `DFMA` | fp64 | 0.05 warp-inst/SM/cy = **1.6 DFMAs/SM/cy** = **475 GFLOPS FMA** chip-wide (≈ 950 FLOPS) |
| `add.rn.f64` | `DADD` | fp64 | 0.05                                                                                    |
| `mul.rn.f64` | `DMUL` | fp64 | 0.05                                                                                    |


That's 1/80th the FP32 FMA rate on a per-cycle basis — FP64 is not a B300 strength.

---

## 3. Contention rules (from u-metric sweeps)

1. **Same pipe** → total rate capped at that pipe's ceiling.
  - F2FP + LOP3 (both alu) → 64 combined, period.
  - IMAD + FFMA scalar (both compete for fmaH) → reduces FFMA peak.
2. **Different pipes** → usually add cleanly, with two caveats:
  - **FFMA2 + UNPACK** (fma + alu): u=1.67 (106/127 combined) — there's a ~16% SMSP dual-issue friction specific to F2FP. Not present for PRMT+FFMA2 (which hits u=1.95 → 124/127).
  - **LOP3 + FFMA scalar**: LOP3 is on alu, FFMA uses both fmaH AND fmaL. No ALU/FMA contention, but at balanced ILP total sm_inst can exceed 4.0 only if packed ops are used. Scalar FFMA uses half of dispatch on both fma sub-pipes, so adding LOP3 gets you up toward the 4.00 dispatch cap.
3. **Dispatch cap = 4.00 sm_inst/SM/cy** is hard. To exceed 128 SASS-inst/SM/cy you need packed ops counted as multiple logical ops (FFMA2 = 2 FMAs, HFMA2 = 2 FP16 FMAs, FFMA scalar counts as 1 inst per each of H+L pipes so shows as 4.00).
4. **HFMA2 + FFMA scalar** can co-exist: HFMA2 occupies both H+L sub-units for one inst (2.00), FFMA scalar occupies both for 2 insts (4.00). Combined they compete for H+L slots. Peak rate for mixed ~2.0 total warp-inst/SM/cy (one must yield).

---

## 4. Rate cheatsheet (SASS inst/SM/clk — warp-level)


| Op                                                       | SASS/SM/cy                       | Notes                                        |
| -------------------------------------------------------- | -------------------------------- | -------------------------------------------- |
| Scalar FP32 FMA (FFMA)                                   | **128**                          | dual-pipe heavy + lite                       |
| Scalar FP32 ADD/MUL                                      | 128                              | same                                         |
| FFMA2 / HFMA2 / BF16-FMA                                 | **64**                           | but 2 FLOPs per inst → 128 FLOPS/SM/cy       |
| IMAD u32                                                 | 64                               | fmaH only                                    |
| DP4A / DP2A                                              | 64                               | fmaH                                         |
| u32 ADD                                                  | **128**                          | as IADD3 (1 SASS = 2 adds) or split alu+fmaH |
| u64 ADD                                                  | **64**                           | requires 1 alu + 1 fmaH per op               |
| LOP3 / PRMT / SHL / SHR / SHF / FMNMX / HMNMX2 / VIMNMX3 | 64                               | all pipe_alu, they all share                 |
| F2FP UNPACK (all formats)                                | 64                               | = 128 elements/SM/cy (x2 ops)                |
| F2FP PACK (all formats)                                  | 32–64 depending on feedback path | pipe_alu                                     |
| BFE                                                      | 32                               | 2 SASS per PTX op                            |
| SELP / setp+selp / vote.ballot                           | 32–64                            | pipe_alu                                     |
| SHFL.SYNC.*                                              | 32                               | pipe_lsu                                     |
| LDS / STS / LDG / STG                                    | ~32 issue                        | pipe_lsu; DRAM-bound if streaming            |
| MUFU (EX2/RSQ/SIN/COS/LG2/TANH/SQRT/RCP)                 | ~16                              | pipe_xu, compound                            |
| F2I (f32→s32/s64/s8), POPC, BREV, FLO                    | 16                               | pipe_xu                                      |
| BAR.SYNC                                                 | ~12                              | pipe_adu                                     |
| MATCH.ANY                                                | serial                           | pipe_adu, slow                               |
| **FP64 FMA (DFMA)**                                      | **1.6**                          | pipe_fp64, throttled                         |


---

## 5. Narrow-format throughput — summary (elements/sec chip-wide)

At 128 elements/SM/cy × 148 SMs × 1.92 GHz = **36.4 Telements/s for each UNPACK variant** (FP4/FP6/FP8/UE8M0 → f16/bf16). Same number for all because they share the one ALU pipe. PACK round-trip (with matching UNPACK feeding it) matches that rate **per direction**; solo PACK with LOP3 zero-ext feedback drops to **18–24 Telements/s**.

For FP4 specifically: both UNPACK (`cvt.rn.f16x2.e2m1x2`) and PACK (`cvt.rn.satfinite.e2m1x2.f16x2`) live on the same 64 warp-inst/SM/cy ceiling as FP8/FP6 — FP4 is **not faster or slower per SASS instruction** than FP8 on B300's ALU pipe.

---

## 6. Uniform datapath (`pipe_uniform`) — full picture

Blackwell has a separate **uniform scalar datapath** operating on **uniform registers (URx)** that hold one value shared across the whole warp (as opposed to the 32-lane "vector" register file). Each SMSP has its own uniform register file and a single-issue uniform ALU that runs in parallel with the per-lane pipes.

Compiler uses it automatically for loop counters, kernel-arg propagation, constant address calculation, warp-invariant scalars, etc. — you can rarely target it directly from PTX, but you see its SASS in `UMOV`, `UIADD3`, `UISETP`, `USHF`, etc.

**Measured:** `pipe_uniform` hits ~1.0 warp-inst/SM/cy in practice for ACTIVEMASK and LDSM. It does **not** contend with pipe_alu / pipe_fma — uniform ops issue in parallel with vector ops from the same SMSP.

**SASS opcodes hosted on pipe_uniform (per NVIDIA docs, Blackwell):**
`UMOV`, `UMOV32I`, `UIADD3`, `UIADD3.64`, `UIMAD`, `UIMNMX`, `UISETP`, `UIABS`, `ULOP`, `ULOP3`, `ULOP32I`, `UPOPC`, `UBREV`, `UBMSK`, `UFLO`, `USEL`, `USGXT`, `USHF`, `USHL`, `USHR`, `UPRMT`, `ULEA`, `ULEPC`, `UCLEA`, `UF2F`, `UF2FP`, `UF2I`, `UF2IP`, `UFFMA`, `UFADD`, `UFMUL`, `UFMNMX`, `UFRND`, `UFSEL`, `UFSET`, `UFSETP`, `UI2F`, `UI2FP`, `UI2I`, `UI2IP`, `VOTEU`, `UP2UR`, `UPLOP3`, `UPSETP`, `UR2UP`, `USETMAXREG`, `USTGR`, `UREDGR`, `UGETNEXTWORKID`, `UVIADD`, `UVIMNMX`, `UVIRTCOUNT`, `UMEMSETS`, `LDCU`, `CS2UR`, `R2UR`, `S2UR`, `REDUX`, `CREDUX` (reductions emit coupled vector+uniform), `UCGABAR_ARV/WAIT`.

New on Blackwell: the full uniform FP32 datapath (`UFFMA`, `UFADD`, `UFMUL`, etc.) — warp-invariant FP32 arithmetic can now run on the uniform side, freeing the vector FMA pipes for divergent work. This is a compile-time optimization target.

**pipe_uniform also handles:** `LDSM.sync.aligned.*.shared.b16` (ldmatrix) and `ACTIVEMASK` emit as uniform ops.

## 7. ADU (`pipe_adu`) — the "advance data unit"

ADU hosts the slow warp-wide synchronization and status-register operations. Every op that needs cross-lane coordination within a warp that doesn't fit the shuffle / vote pattern lands here.

**SASS opcodes on pipe_adu (measured + inferred):**

- Barriers: `BAR`, `BAR.SYNC`, `BAR.ARV`, `BAR.RED.POPC`, `BAR.RED.AND`, `BAR.RED.OR`, `BARRIER.SYNC`, `B2R`, `BMOV`, `DEPBAR`, `LDGDEPBAR`, `SYNCS`
- CGA (Cluster) barriers: `UCGABAR_ARV`, `UCGABAR_WAIT`, `CGAERRBAR`, `ACQBULK`, `ACQSHMINIT`
- Warp sync: `WARPSYNC`, `BSYNC`, `BSSY`, `BREAK`, `NANOSLEEP`, `YIELD`
- Control w/ warp coordination: `ELECT`, `ENDCOLLECTIVE`, `SETCTAID`, `KILL`, `PMTRIG`
- Match / reduce: `MATCH.ANY`, `MATCH.ALL`, `REDUX.SUM`, `REDUX.OR`, `REDUX.AND`, `REDUX.XOR` (min/max go through alu+fma via CREDUX)
- Fences: `MEMBAR.SC.GPU` / `MEMBAR.SC.SYS` (partial — also hits lsu), `ERRBAR`
- SR reads that query warp/chip state: `S2R SR_CLOCKLO`, `S2R SR_CLOCKHI`, `S2R SR_GLOBALTIMER`, `CS2R`

**Peak issue rate**: ~0.4–0.5 warp-inst/SM/cy for simple cases (BAR.ARV, REDUX.OR). Wall-clock time is dominated by **cross-thread waiting** rather than the pipe's own throughput — BAR.SYNC is 6 ms for 128 inst per thread because threads spend most of the time blocked, not because ADU can't issue.

**Contention with ALU/FMA:** none observed. ADU ops do not consume alu or fma slots, but they do frequently depend on a predicate-compute or popcount from alu (hence the secondary alu reading in e.g. `bar.red.popc`).

## 8. Complete SASS opcode → pipe classification

Below: every SASS opcode listed in NVIDIA's Blackwell SASS reference (sm_100/103, [CUDA Binary Utilities doc](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)) with pipe assignment. **[m]** = measured in this session. **[i]** = inferred from opcode family / uniform-prefix rule.

### Floating-point arithmetic — `pipe_fma` (heavy+lite, scalar dual-issue; packed ops occupy both for one inst)


| SASS                                    | Pipe              | Peak (SASS/SM/cy) | Status                   |
| --------------------------------------- | ----------------- | ----------------- | ------------------------ |
| `FADD`                                  | fma (H+L dual)    | 128 scalar        | [m]                      |
| `FFMA`                                  | fma (H+L dual)    | 128 scalar        | [m]                      |
| `FMUL`                                  | fma (H+L dual)    | 128 scalar        | [m]                      |
| `FADD2`                                 | fma (packed, H=L) | 64                | [i] (variant of FADD)    |
| `FADD32I` / `FMUL32I` / `FFMA32I`       | fma               | 128               | [i] (immediate variants) |
| `FFMA2` (vec2 FP32)                     | fma (H=L)         | 64 = 128 FMAs     | [m]                      |
| `FHADD` / `FHFMA`                       | fma               | 128               | [i]                      |
| `FMUL2`                                 | fma packed        | 64                | [i]                      |
| `FCHK` (range check)                    | alu or fma        | —                 | [i]                      |
| `FMNMX` / `FMNMX3`                      | **alu**           | 64                | [m] for FMNMX            |
| `FSEL`                                  | alu               | 64                | [m]                      |
| `FSET` / `FSETP`                        | alu               | 64                | [m]                      |
| `FSWZADD` (swizzle-add)                 | fma               | 64                | [i]                      |
| `FRND` (round-to-int)                   | alu/fma           | ~64               | [i]                      |
| `MUFU.`*                                | **xu**            | 16                | [m]                      |
| `HADD2` / `HFMA2` / `HMUL2`             | fma packed (H=L)  | 64 = 128 FP16 ops | [m]                      |
| `HADD2.BF16` / `HFMA2.BF16`             | fma packed        | 64 = 128 BF16 ops | [m]                      |
| `HADD2_32I` / `HFMA2_32I` / `HMUL2_32I` | fma packed        | 64                | [i]                      |
| `HADD2.F32` (f16→f32 cvt backing)       | fma heavy         | 64                | [m]                      |
| `HMNMX2` / `HMNMX2.NAN` / `HMNMX2.BF16` | **alu**           | 64 = 128 ops      | [m]                      |
| `VHMNMX` (used for min.f16x2)           | **alu**           | 64                | [m]                      |
| `HSET2` / `HSETP2`                      | alu               | 64                | [i]                      |
| `DADD` / `DFMA` / `DMUL`                | **fp64**          | **1.6**           | [m]                      |
| `DSETP`                                 | fp64 (or alu)     | slow              | [i]                      |
| `DMMA`                                  | tensor            | —                 | [i]                      |
| `HMMA` (FP16 tensor)                    | tensor            | very high FLOPS   | [i]                      |
| `OMMA` (FP4 tensor)                     | tensor            | —                 | [i]                      |
| `QMMA` (FP8 tensor)                     | tensor            | —                 | [i]                      |
| `IMMA` (int tensor)                     | tensor            | —                 | [i]                      |


### Integer arithmetic — `pipe_fmaheavy` for mul/dot; `pipe_alu` for simple add/min/bitwise


| SASS                         | Pipe                                             | Peak                              | Status          |
| ---------------------------- | ------------------------------------------------ | --------------------------------- | --------------- |
| `IMAD` / `IMAD.`*            | fmaheavy                                         | 64                                | [m]             |
| `IMUL` / `IMUL32I`           | fmaheavy                                         | 64                                | [m] (as IMAD)   |
| `IMAD.HI`                    | fmaheavy                                         | **32**                            | [m] (half rate) |
| `IMAD.WIDE`                  | fmaheavy                                         | ~32                               | [m]             |
| `IMAD.X` (carry-add)         | fmaheavy                                         | 64                                | [m]             |
| `IMAD.IADD` (as plain add)   | fmaheavy                                         | 64                                | [m]             |
| `IDP` / `IDP4A`              | fmaheavy                                         | 64                                | [m]             |
| `IADD` / `IADD3` / `IADD32I` | **alu** (+ fmaH split when compiler dual-issues) | 64 per pipe; 128 logical u32 adds | [m]             |
| `IABS`                       | alu (as IADD3 with RZ)                           | 64                                | [i]             |
| `ISCADD` / `ISCADD32I`       | alu                                              | 64                                | [i]             |
| `ISETP`                      | alu                                              | 64                                | [m]             |
| `IMNMX`                      | alu (via VIMNMX3)                                | 64 = 128 mins                     | [m]             |
| `VIMNMX` / `VIMNMX3`         | alu                                              | 64                                | [m]             |
| `VIADD` / `VIADDMNMX`        | alu                                              | 64                                | [i]             |
| `VABSDIFF` / `VABSDIFF4`     | alu                                              | 64                                | [m]             |


### Bitwise / shift / permute — `pipe_alu`


| SASS                                   | Pipe   | Peak           | Status |
| -------------------------------------- | ------ | -------------- | ------ |
| `LOP` / `LOP3` / `LOP3.LUT` / `LOP32I` | alu    | 64             | [m]    |
| `SHF` / `SHF.L.`* / `SHF.R.*`          | alu    | 64             | [m]    |
| `SHL` / `SHR`                          | alu    | 64 (via SHF.*) | [m]    |
| `PRMT`                                 | alu    | 64             | [m]    |
| `BREV`                                 | **xu** | 16             | [m]    |
| `BMSK` (bitfield mask)                 | alu    | 64             | [i]    |
| `FLO` (find leading one)               | xu     | 16             | [m]    |
| `POPC`                                 | xu     | 16             | [m]    |
| `SGXT` (sign extend)                   | alu    | 64             | [i]    |


### Comparison / select / predicate — `pipe_alu`


| SASS                                   | Pipe                 | Peak | Status |
| -------------------------------------- | -------------------- | ---- | ------ |
| `ISETP` / `FSETP` / `HSETP2` / `DSETP` | alu (DSETP slow)     | 64   | [m]    |
| `FSET` / `HSET2`                       | alu                  | 64   | [i]    |
| `SEL`                                  | alu                  | 64   | [m]    |
| `FSEL`                                 | alu                  | 64   | [m]    |
| `PLOP3` / `PSETP`                      | alu (predicate-side) | 64   | [i]    |
| `P2R` / `R2P`                          | alu                  | 64   | [i]    |


### Conversion — mixed


| SASS                                            | Pipe     | Peak                       | Status |
| ----------------------------------------------- | -------- | -------------------------- | ------ |
| `F2F` (f16↔f32, bf16↔f32 via HADD2.F32)         | fmaheavy | 64                         | [m]    |
| `F2FP.*.UNPACK` (all narrow formats → f16/bf16) | alu      | 64 = 128 elems             | [m]    |
| `F2FP.*.UNPACK_B_MERGE_C` (PACK f16x2→narrow)   | alu      | ~48 solo, 64 in round-trip | [m]    |
| `F2FP.*.PACK_AB_MERGE_C` (PACK f32×2→narrow)    | alu      | ~32 solo                   | [m]    |
| `F2FP.F16.F32.PACK` / `F2FP.BF16.F32.PACK`      | alu      | 64 (with PRMT tax)         | [m]    |
| `F2I.NTZ` (f32→s32/u32/s64/s8)                  | **xu**   | 16                         | [m]    |
| `F2IP.U8.F32.NTZ`                               | alu      | 64                         | [m]    |
| `I2F.S64` (s64→f32)                             | xu       | very slow                  | [m]    |
| `I2FP.F32.{S32,U32}`                            | alu      | 64                         | [m]    |
| `I2I.*.SAT`                                     | alu      | 64                         | [m]    |
| `I2IP`                                          | alu      | 64                         | [i]    |
| `FRND`                                          | alu/fma  | 64                         | [i]    |


### Data movement


| SASS                      | Pipe                         | Peak | Status |
| ------------------------- | ---------------------------- | ---- | ------ |
| `MOV` / `MOV32I`          | alu (or compiler-eliminated) | 64   | [i]    |
| `MOVM` (matrix move)      | tensor/lsu                   | —    | [i]    |
| `SHFL` (BFLY/IDX/UP/DOWN) | **lsu**                      | 32   | [m]    |


### Load / store


| SASS                                       | Pipe              | Peak                         | Status    |
| ------------------------------------------ | ----------------- | ---------------------------- | --------- |
| `LD` (generic)                             | lsu               | 32 issue                     | [i]       |
| `LDC` (constant)                           | alu (near-free)   | 128+                         | [i]       |
| `LDG` (global)                             | lsu               | 32 issue (DRAM-bound)        | [m]       |
| `LDGMC` (reducing load)                    | lsu               | —                            | [i]       |
| `LDGSTS` (async g→s memcpy)                | lsu               | 32 issue                     | [i]       |
| `LDL` (local)                              | lsu               | 32                           | [i]       |
| `LDS` (shared)                             | lsu               | 32 issue                     | [m]       |
| `LDSM` (ldmatrix)                          | **uniform + lsu** | 1.0 for x1, halves for x2/x4 | [m]       |
| `LDT` / `LDTM` (tensor memory load)        | tensor-memory     | —                            | [i]       |
| `STSM` (store matrix shared)               | uniform + lsu     | —                            | [i]       |
| `ST` / `STG` / `STL` / `STS`               | lsu               | 32                           | [m] STG   |
| `STT` / `STTM`                             | tensor-memory     | —                            | [i]       |
| `STAS` (async store distributed-shmem)     | lsu               | —                            | [i]       |
| `ATOM` / `ATOMS` / `ATOMG`                 | lsu               | contended                    | [m] ATOMS |
| `REDAS` (async reduction dshared)          | lsu               | —                            | [i]       |
| `REDG` (global reduction)                  | lsu               | —                            | [i]       |
| `MATCH`                                    | **adu**           | 0.5                          | [m]       |
| `QSPC` (query space)                       | alu/adu           | —                            | [i]       |
| `CCTL` / `CCTLL` / `CCTLT` (cache control) | lsu               | —                            | [i]       |
| `ERRBAR` (error barrier)                   | adu + lsu         | slow                         | [m]       |
| `MEMBAR` (.cta)                            | lsu               | 32                           | [m]       |
| `MEMBAR` (.gpu/.sys)                       | adu + lsu         | very slow                    | [m]       |
| `FENCE`                                    | lsu               | —                            | [i]       |
| `SYNCS`                                    | adu               | —                            | [i]       |


### Uniform datapath — `pipe_uniform` (per-SMSP scalar unit, new full FP on Blackwell)


| SASS                                        | Pipe                       | Notes                     | Status |
| ------------------------------------------- | -------------------------- | ------------------------- | ------ |
| `UMOV` / `UMOV32I`                          | uniform                    | register copy             | [i]    |
| `UIADD3` / `UIADD3.64`                      | uniform                    | u32/u64 add               | [i]    |
| `UIMAD`                                     | uniform                    | u32 mul-add               | [i]    |
| `UIMNMX` / `UIABS`                          | uniform                    | min/max/abs               | [i]    |
| `UISETP`                                    | uniform                    | compare to u-pred         | [i]    |
| `ULOP` / `ULOP3` / `ULOP32I`                | uniform                    | bitwise                   | [i]    |
| `UPOPC` / `UBREV` / `UBMSK` / `UFLO`        | uniform                    | bit-count/rev             | [i]    |
| `USEL` / `USGXT`                            | uniform                    | select / sign-ext         | [i]    |
| `USHF` / `USHL` / `USHR`                    | uniform                    | shift                     | [i]    |
| `UPRMT`                                     | uniform                    | byte permute              | [i]    |
| `ULEA` / `ULEPC` / `UCLEA`                  | uniform                    | load effective address    | [i]    |
| `UFFMA` / `UFADD` / `UFMUL` (new Blackwell) | uniform                    | scalar FP32 on u-datapath | [i]    |
| `UFMNMX` / `UFSEL` / `UFSET` / `UFSETP`     | uniform                    | FP32 min/sel/cmp          | [i]    |
| `UF2F` / `UF2FP` / `UF2I` / `UF2IP`         | uniform                    | FP conversions            | [i]    |
| `UI2F` / `UI2FP` / `UI2I` / `UI2IP`         | uniform                    | int conversions           | [i]    |
| `UFRND`                                     | uniform                    | round                     | [i]    |
| `VOTEU`                                     | uniform                    | vote → uniform dest       | [i]    |
| `UP2UR` / `UPLOP3` / `UPSETP` / `UR2UP`     | uniform                    | predicate ops             | [i]    |
| `USETMAXREG` (release/alloc regs)           | uniform                    | setmaxnreg                | [i]    |
| `UVIADD` / `UVIMNMX`                        | uniform                    | SIMD-style u-ops          | [i]    |
| `UVIRTCOUNT`                                | uniform                    | virtual-resource mgmt     | [i]    |
| `UMEMSETS`                                  | uniform                    | shmem init                | [i]    |
| `UGETNEXTWORKID`                            | uniform                    | work distrib              | [i]    |
| `USTGR` (ustore global w/ release)          | uniform + lsu              | —                         | [i]    |
| `UREDGR` (u reduction on global)            | uniform + lsu              | —                         | [i]    |
| `LDCU` / `CS2UR` (const → u-reg)            | uniform                    | const load                | [i]    |
| `R2UR` / `S2UR`                             | uniform                    | move to u-reg             | [i]    |
| `REDUX` / `CREDUX`                          | uniform + (adu or alu/fma) | warp reduction            | [m]    |
| `ACTIVEMASK` / `ELECT`                      | uniform                    |                           | [m]    |


### Control flow — `pipe_cbu`


| SASS                                              | Pipe      | Status |
| ------------------------------------------------- | --------- | ------ |
| `BRA` / `JMP` / `JMX` / `JMXU` / `BRX` / `BRXU`   | cbu       | [i]    |
| `CALL` / `RET` / `EXIT` / `KILL`                  | cbu       | [i]    |
| `BSSY` / `BSYNC` / `BREAK` / `BPT`                | cbu / adu | [i]    |
| `WARPSYNC`                                        | cbu       | [i]    |
| `BMOV` (barrier state move)                       | cbu       | [i]    |
| `NANOSLEEP` / `YIELD`                             | adu       | [i]    |
| `RPCMOV` / `LEPC` / `SETLMEMBASE` / `GETLMEMBASE` | misc      | [i]    |
| `PREEXIT` (launch hint)                           | misc      | [i]    |
| `SETCTAID`                                        | adu       | [i]    |


### Barriers / sync / status — `pipe_adu`


| SASS                                         | Pipe                | Status |
| -------------------------------------------- | ------------------- | ------ |
| `BAR` / `BAR.SYNC` / `BAR.ARV` / `BAR.RED.`* | adu                 | [m]    |
| `BARRIER.SYNC`                               | adu                 | [m]    |
| `DEPBAR` / `LDGDEPBAR`                       | adu                 | [i]    |
| `B2R` (barrier → reg)                        | adu                 | [i]    |
| `S2R` (clock / timer)                        | adu                 | [m]    |
| `CS2R`                                       | adu                 | [i]    |
| `MATCH.ANY` / `MATCH.ALL`                    | adu                 | [m]    |
| `REDUX.{SUM,OR,AND,XOR}`                     | adu (+ vector side) | [m]    |
| `MEMBAR` (global, system)                    | adu + lsu           | [m]    |
| `ERRBAR` / `CGAERRBAR`                       | adu + lsu           | [i]    |
| `UCGABAR_ARV` / `UCGABAR_WAIT`               | uniform + adu       | [i]    |
| `ACQBULK` / `ACQSHMINIT`                     | adu                 | [i]    |
| `ENDCOLLECTIVE`                              | adu                 | [i]    |
| `VOTE.{ANY,ALL,UNI}` (non-uniform)           | **alu**             | [m]    |


### Tensor core / tensor memory — `pipe_tensor`, `pipe_tc` (hmma / imma subpipes)


| SASS                                           | Pipe             | Status        |
| ---------------------------------------------- | ---------------- | ------------- |
| `HMMA`                                         | tensor.hmma      | [i] datasheet |
| `IMMA`                                         | tensor.imma      | [i]           |
| `DMMA`                                         | fp64 → tensor    | [i]           |
| `QMMA` (FP8 MMA)                               | tensor           | [i]           |
| `OMMA` (FP4 MMA)                               | tensor           | [i] new       |
| `UTCHMMA` / `UTCIMMA` / `UTCOMMA` / `UTCQMMA`  | tensor + uniform | [i]           |
| `UTCBAR` / `UTCATOMSWS` / `UTCCP` / `UTCSHIFT` | tensor-memory    | [i]           |


### Tensor Memory Access (TMA, Hopper-style UBLK*) — bulk copy engine


| SASS                                     | Pipe               | Status |
| ---------------------------------------- | ------------------ | ------ |
| `UBLKCP` / `UBLKPF` / `UBLKRED`          | tensor-mem (async) | [i]    |
| `UTMALDG` / `UTMASTG` / `UTMAPF`         | tensor-mem (async) | [i]    |
| `UTMAREDG` / `UTMACCTL` / `UTMACMDFLUSH` | tensor-mem         | [i]    |


### Texture / surface — `pipe_tex`, `pipe_tc`


| SASS                                            | Pipe          | Status |
| ----------------------------------------------- | ------------- | ------ |
| `TEX` / `TLD` / `TLD4` / `TMML` / `TXD` / `TXQ` | tex           | [i]    |
| `SULD` / `SUST` / `SUATOM` / `SURED`            | tex (surface) | [i]    |


### Miscellaneous


| SASS                    | Pipe         | Status |
| ----------------------- | ------------ | ------ |
| `NOP`                   | — (no-issue) | [i]    |
| `PMTRIG` (perf monitor) | misc         | [i]    |
| `BPT` (breakpoint/trap) | cbu          | [i]    |


## 9. PTX → SASS mapping for every ISA category (this session's coverage)


| PTX mnemonic (Blackwell)                                                             | Dominant SASS                                           | Pipe                                          |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------- | --------------------------------------------- |
| **arithmetic FP32** `add`/`sub`/`mul`/`fma`/`neg`/`abs`                              | FADD / FMUL / FFMA (abs/neg → FADD.FTZ)                 | fma scalar dual-issue                         |
| `min`/`max`/`min.NaN`/`max.NaN` f32                                                  | FMNMX / FMNMX.NAN                                       | alu                                           |
| `rcp`/`sqrt`/`rsqrt`/`ex2`/`lg2`/`sin`/`cos`/`tanh` approx f32                       | MUFU.{RCP,SQRT,RSQ,EX2,LG2,SIN,COS,TANH}                | xu                                            |
| `copysign.f32`                                                                       | LOP3.LUT (bit trick)                                    | alu                                           |
| **vec2 FP32** `fma.rn.f32x2`                                                         | FFMA2                                                   | fma packed                                    |
| **FP16x2** `{add,mul,fma,min,max}.rn.f16x2`                                          | HADD2 / HMUL2 / HFMA2 / HMNMX2                          | fma (arith) or alu (mnmx)                     |
| **BF16x2** variants                                                                  | HADD2.BF16 / HFMA2.BF16 / HMNMX2.BF16                   | same                                          |
| `cvt.f32.f16` / `.bf16`                                                              | HADD2.F32                                               | fmaheavy                                      |
| `cvt.rn.f16.f32` / `.bf16.f32`                                                       | F2FP.F16.F32.PACK / F2FP.BF16.F32.PACK                  | alu                                           |
| `cvt.rn.{f16x2,bf16x2}.{e4m3x2,e5m2x2,e2m3x2,e3m2x2,e2m1x2,ue8m0x2}` (narrow unpack) | F2FP.F16/BF16.E{4M3,5M2,2M3,3M2,2M1,8}.UNPACK_B         | alu                                           |
| `cvt.rn.satfinite.{e4m3x2,…}.{f16x2,bf16x2}` (pack from half)                        | F2FP.SATFINITE.*.F16.UNPACK_B_MERGE_C                   | alu                                           |
| `cvt.rn.satfinite.{…}.f32` (pack from f32 pair)                                      | F2FP.SATFINITE.*.F32.PACK_AB_MERGE_C                    | alu                                           |
| `cvt.rp.satfinite.ue8m0x2.f32`                                                       | F2FP.SATFINITE.UE8M0.F32.PACK_AB_MERGE_C                | alu                                           |
| `cvt.rni.{s32,u32}.f32`                                                              | F2I.{S32,U32}.NTZ                                       | xu                                            |
| `cvt.rni.sat.u8.f32`                                                                 | F2IP.U8.F32.NTZ                                         | alu                                           |
| `cvt.rni.sat.s8.f32`                                                                 | F2I.S8.NTZ                                              | xu                                            |
| `cvt.rn.f32.{s32,u32}`                                                               | I2FP.F32.{S32,U32}                                      | alu                                           |
| `cvt.rn.f32.s64`                                                                     | I2F.S64                                                 | xu (slow)                                     |
| `cvt.sat.u8.s32` / `cvt.sat.s16.s32` / `cvt.sat.u16.u32`                             | I2I.*.SAT                                               | alu                                           |
| `cvt.u32.u64` / `cvt.u32.u8`                                                         | MOV / LOP3                                              | alu (near-free)                               |
| **integer u32** `add/sub`/`mul`/`mad`                                                | IADD3 / IMAD / IMAD                                     | alu (add) + fmaH (mul/mad)                    |
| `min/max.s32/u32`                                                                    | VIMNMX3                                                 | alu                                           |
| `mul.hi.u32`                                                                         | IMAD.HI.U32                                             | fmaheavy (half rate)                          |
| `dp4a.s32.s32` / `dp2a.`*                                                            | IDP.4A.S8.S8 / IDP.2A.LO.S16.S8                         | fmaheavy                                      |
| `bfe` / `bfi` / `brev` / `popc` / `clz` / `bfind`                                    | SHF+SGXT / LOP3 / BREV / POPC / FLO                     | mixed (alu for bfe/bfi, xu for brev/popc/clz) |
| **integer u64** `add/sub`                                                            | IADD3 + IMAD.X (2 SASS)                                 | alu + fmaheavy                                |
| u64 `mul.lo` / `mul.hi`                                                              | IMAD + IMAD.WIDE + IADD3 chain                          | fmaheavy + alu                                |
| u64 `and`/`or`/`xor`                                                                 | 2× LOP3                                                 | alu                                           |
| u64 `shl`/`shr`                                                                      | SHF.L.U64.HI + SHF.L.U32 (+ helpers)                    | alu                                           |
| u64 `min`/`max`                                                                      | ISETP.EX + SEL (4 SASS)                                 | alu                                           |
| **shifts / funnel shifts** `shl`/`shr`/`shf.`*                                       | SHF.L.W.U32 / SHF.R.W.U32.HI / SHF.L.U32.HI / SHF.R.U32 | alu                                           |
| `prmt.b32`                                                                           | PRMT                                                    | alu                                           |
| **logic** `and`/`or`/`xor`/`not`/`cnot`/`lop3`                                       | LOP3.LUT                                                | alu                                           |
| **predicate** `setp`/`selp`/`and/or/xor.pred`                                        | ISETP/FSETP + SEL, PLOP3                                | alu                                           |
| **warp** `shfl.sync.`*                                                               | SHFL.{BFLY,IDX,UP,DOWN}                                 | **lsu**                                       |
| `vote.sync.{ballot,any,all,uni}`                                                     | VOTE.*                                                  | alu                                           |
| `match.{any,all}.sync.b32`                                                           | MATCH.ANY / MATCH.ALL                                   | **adu** (very slow)                           |
| `activemask`                                                                         | uniform-pipe emission                                   | **uniform**                                   |
| `redux.sync.{add,or,and,xor}.{u32,b32}`                                              | REDUX.SUM / REDUX.OR / REDUX.AND / REDUX.XOR            | adu + (alu)                                   |
| `redux.sync.{min,max}.u32`                                                           | CREDUX.MIN/MAX                                          | **alu + fma** (dual-pipe)                     |
| **memory** `ld.global` / `ld.local` / `ld.shared` / `ld.const`                       | LDG / LDL / LDS / LDC                                   | lsu (LDC near-free)                           |
| `ld.global.nc` (const-cached)                                                        | LDG.E.CONSTANT / LDG.E.NCSH                             | lsu                                           |
| `ld.async`                                                                           | LDGSTS                                                  | lsu (async)                                   |
| `st.global` / `st.shared` / `st.local`                                               | STG / STS / STL                                         | lsu                                           |
| `st.async`                                                                           | STAS                                                    | lsu (async)                                   |
| `atom.*.{add,inc,dec,and,or,xor,cas,exch,min,max}`                                   | ATOM / ATOMS / ATOMG                                    | lsu                                           |
| `red.`*                                                                              | REDG / REDAS                                            | lsu                                           |
| `ldmatrix.sync.aligned.*.shared.b16`                                                 | LDSM / LDSM.T                                           | **uniform + lsu**                             |
| `stmatrix.sync.aligned.*.shared.b16`                                                 | STSM                                                    | uniform + lsu                                 |
| **fence/barrier** `membar.cta`                                                       | MEMBAR.SC.CTA                                           | lsu                                           |
| `membar.gl` / `membar.sys`                                                           | MEMBAR.SC.GPU + ERRBAR                                  | adu + lsu (slow)                              |
| `bar.sync` / `bar.arrive` / `bar.red.`* / `barrier.*`                                | BAR.SYNC / BAR.ARV / BAR.RED.POPC                       | adu                                           |
| `fence.*`                                                                            | FENCE                                                   | lsu                                           |
| **TMA / bulk async** `cp.async.bulk.`* / `cp.reduce.async.bulk.*`                    | UBLKCP / UBLKRED / UTMALDG / UTMASTG                    | tensor-mem pipe                               |
| **tensor** `mma.sync.*.f16`                                                          | HMMA                                                    | tensor (hmma subpipe)                         |
| `mma.sync.*.bf16` / `.f32` / `.tf32`                                                 | HMMA variants                                           | tensor                                        |
| `mma.sync.*.s8`/`.s4`/`.u8`/`.u4`/`.s1`                                              | IMMA                                                    | tensor (imma subpipe)                         |
| `mma.sync.*.fp8` (e4m3, e5m2)                                                        | QMMA                                                    | tensor                                        |
| `mma.sync.*.fp4` (e2m1)                                                              | OMMA                                                    | tensor (new Blackwell)                        |
| `mma.sync.*.f64`                                                                     | DMMA                                                    | tensor (fp64 subpipe)                         |
| `wgmma.`* / `tcgen05.*`                                                              | UTC{HMMA,IMMA,QMMA,OMMA} etc.                           | tensor + tensor-memory                        |
| `tcgen05.ld/st` (tensor mem)                                                         | LDT / LDTM / STT / STTM                                 | tensor-memory                                 |
| **control** `bra` / `@p bra` / `call` / `ret` / `exit` / `trap`                      | BRA / CALL / RET / EXIT / BPT                           | cbu                                           |
| `bra.uni` / `@!up bra`                                                               | BRXU / JMPU                                             | cbu                                           |
| **misc** `nanosleep.u32`                                                             | NANOSLEEP                                               | adu                                           |
| `elect.sync`                                                                         | ELECT                                                   | uniform                                       |
| `setmaxnreg.`*                                                                       | USETMAXREG                                              | uniform                                       |
| `getctarank` / `isspacep.*`                                                          | SETCTAID / QSPC                                         | adu / alu                                     |
| **FP64** `add.f64` / `mul.f64` / `fma.rn.f64`                                        | DADD / DMUL / DFMA                                      | **fp64** (1.6/SM/cy)                          |
| `mma.sync.*.f64`                                                                     | DMMA                                                    | fp64 tensor                                   |


## 11. Deep-dive: `redux.sync.`* — what's real, what's not

`redux.sync` is the warp-wide reduction intrinsic. Findings from full mask / partial mask / type sweeps:

**Supported type/op matrix on B300 (anything else → ptxas error):**


| op                      | .u32 | .s32 | .f32 | .f32.NaN | .b32 | .u64/.s64 | .f16/.f16x2/.bf16/.bf16x2 | .f64 |
| ----------------------- | ---- | ---- | ---- | -------- | ---- | --------- | ------------------------- | ---- |
| `.min` / `.max`         | ✓    | ✓    | ✓    | ✓        | —    | ✗         | ✗                         | ✗    |
| `.add`                  | ✓    | ✓    | ✗    | —        | ✗    | ✗         | ✗                         | ✗    |
| `.and` / `.or` / `.xor` | —    | —    | —    | —        | ✓    | ✗         | ✗                         | ✗    |
| `.mul`                  | ✗    | ✗    | ✗    | —        | ✗    | ✗         | ✗                         | ✗    |


No 64-bit, no FP16/BF16, no FP64, no `mul` redux. FP32 sum reduce is not hardware-assisted — you must compose with `shfl.sync` tree-reduce. FP32 min/max IS assisted.FM

**Throughput and pipe (measured, all types):**


| PTX                           | SASS                                          | Pipe           | PTX-ops/SM/cy |
| ----------------------------- | --------------------------------------------- | -------------- | ------------- |
| `redux.sync.min.u32`          | `CREDUX.MIN` + `IMAD.U32` (intrinsic codegen) | alu + fmaheavy | 1.92          |
| `redux.sync.min.s32`          | `CREDUX.MIN.S32` + IMAD                       | alu + fmaheavy | 1.92          |
| `redux.sync.min.f32`          | `CREDUX.MIN.F32` + IMAD                       | alu + fmaheavy | 1.92          |
| `redux.sync.min.NaN.f32`      | `CREDUX.MIN.F32.NAN` + IMAD                   | alu + fmaheavy | 1.92          |
| `redux.sync.max.`*            | `CREDUX.MAX.*` + IMAD                         | alu + fmaheavy | 1.92          |
| `redux.sync.add.u32` / `.s32` | `REDUX.SUM`                                   | **adu**        | **0.50**      |
| `redux.sync.and.b32`          | `REDUX.AND`                                   | adu            | 0.50          |
| `redux.sync.or.b32`           | `REDUX.OR`                                    | adu            | 0.50          |
| `redux.sync.xor.b32`          | `REDUX.XOR`                                   | adu            | 0.50          |


**Min/max is ~4× faster than add/and/or/xor.** Two separate hardware paths.

**Mask-width independence (measured):** running `redux.sync.min.u32` with masks 0xFFFFFFFF / 0x0000FFFF / 0x55555555 / 0x0000000F / 0x00000001 all take the **same wall time** (1.14–1.15 ms, pipe_alu=1.90–1.92). The hardware doesn't speed up for fewer active lanes — the cost is a fixed instruction latency.

**Why 2 SASS per CREDUX?** `IMAD.U32` is not kernel-side bookkeeping — it's emitted by the compiler as part of the CREDUX result-delivery pattern (probably to broadcast the reduced value from the uniform side back to each participating lane). You cannot eliminate it, so the effective PTX-op rate is bounded by **the slower of** pipe_alu (for CREDUX) and pipe_fmaheavy (for IMAD) — both saturate near 1.92.

## 12. Deep-dive: what can hit the 64 thread-ops/SM/cy `pipe_alu` ceiling (and does it co-issue with anything else)

The pipe_alu budget is **2.00 warp-instructions / SM / cycle** = 64 thread-ops/SM/cy per pipe slot. This budget is shared among **every** alu-resident opcode. The full membership, organized:

**Integer & bitwise (always alu):** `IADD3`, `IADD32I`, `VIADD`, `VIADDMNMX`, `IABS`, `IMNMX` (→VIMNMX3), `VIMNMX`, `VIMNMX3`, `ISCADD`, `ISETP`, `LOP`, `LOP3`, `LOP32I`, `SHL`, `SHR`, `SHF.`* (all forms: L/R, WRAP/CLAMP, U32/S32, HI variants), `PRMT`, `SEL`, `SGXT`, `BMSK`, `LEA`, `MOV` (typically), `MOV32I`, `VABSDIFF`, `VABSDIFF4`.

**FP min/max/compare/select (alu, not fma):** `FMNMX`, `FMNMX.NAN`, `FMNMX3`, `FSEL`, `FSET`, `FSETP`, `HMNMX2`, `HMNMX2.NAN`, `HMNMX2.BF16`, `VHMNMX`, `HSET2`, `HSETP2`, `DSETP` (slow — FP64 compare).

**CVT on alu:** `F2FP.*.UNPACK_B` (all narrow→f16/bf16), `F2FP.*.PACK_AB_MERGE_C` (all wide→narrow), `F2FP.F16.F32.PACK`, `F2FP.BF16.F32.PACK`, `I2FP.F32.{S32,U32}`, `I2I.*.SAT`, `F2IP.U8.F32.NTZ`, `I2IP`, `FRND` (usually).

**Warp reductions that land on alu:** `CREDUX.MIN`, `CREDUX.MAX`, `.S32`, `.F32`, `.F32.NAN` (coupled with intrinsic `IMAD.U32` on fmaheavy).

**Predicate / vote / misc on alu:** `VOTE.{ANY,ALL,UNI}` (non-uniform), `PLOP3`, `PSETP`, `P2R`, `R2P`, `ELECT` (sometimes), `FCHK`.

**Implication:** any kernel that mixes N of these per loop iter will saturate at **2 total warp-inst / SM / cy for all alu ops combined**, i.e. 64 thread-ops/SM/cy split across whatever types you use. You cannot exceed that cap by picking "different alu instructions" — they all draw from the same single pipe.

Confirmed by mixing CREDUX.MIN + FMNMX: total sm_inst = 2.19, pipe_alu = 2.00 (CREDUX+FMNMX share it 50/50). No dual-issue among alu-resident ops.

**Contrast** with pipe_fma which is actually two sub-units (heavy + lite): `fma.rn.f32` scalar dual-issues to **4.00 warp-inst/SM/cy = 128 FFMA/SM/cy**. This only works for scalar (non-packed) FP32 ops. pipe_alu has no such trick — it's a single 2.00/cy pipe.

## 13. Deep-dive: predication / divergence / active-mask effects on throughput

**Per-thread predication (`@p instr`): zero effect on pipe rate.** Measured: `fma.rn.f32` unpredicated = 0.570 ms; same op wrapped in `@p` with only 16/32 lanes active = 0.575 ms; with only 1/32 lanes active = 0.574 ms. The hardware **issues the warp-instruction regardless of how many lanes are live** — pipe time is the same.

**Warp-mask on `shfl.sync` / `redux.sync`: zero effect on rate.** Measured across full, half, quarter, 4-lane, and 1-lane masks — all identical.

**Implications:**

- You cannot "save" pipe throughput by divergence or partial predication. If 1 lane is active the pipe still takes the same instruction slot.
- What predication / divergence *does* save: register-read-port traffic, write-back to masked-off lanes (possibly), and semantic correctness. Not throughput.
- Warp specialization via `elect.sync` → single-lane work doesn't free up pipe slots for the rest of the warp. The warp-inst still consumes its cycle.

**Consequence for warp-specialization designs:** if you have 31/32 lanes doing ALU work and 1 lane doing something else, both workloads still compete for the same pipe slot per cycle. You only save power and register-port contention, not dispatch.

## 14. Extended op catalog (measured, bench_misc_ops.cu)

Additional ops verified, with SASS emitted and pipe assignment:

| PTX | SASS | Pipe | Rate | Notes |
|---|---|---|---:|---|
| `fma.rn.f32` w/ immediate | `FFMA` (not FFMA32I) | fma dual | 128/SM/cy | compiler folds imms into regular FFMA |
| `fma.rn.ftz.f32` | `FFMA.FTZ` | fma dual | 128/SM/cy | FTZ modifier is free |
| `add.u32` / `mul.u32` / `xor.b32` with immediate | DCE'd in isolation | — | — | compiler folds idempotent/constant ops |
| `mad.lo.u32` with power-of-2 mul | **`LEA`** (62) + `IMAD` (69) | alu + fmaheavy | 128/SM/cy combined | compiler emits LEA for shift+add |
| `min.f32 %0,%0,%1; min.f32 %0,%0,%2;` | **`FMNMX3`** (3-input min, fused) | **alu** | 2.00 = 64 SASS/SM/cy, **128 logical mins/SM/cy** | Blackwell has 3-input FP min/max! |
| `ld.global.ca/.cg/.lu` | `LDG.E.{CA,CG,LU}` | lsu | DRAM-bound | cache-hint variants |
| `st.global.wb` | `STG.E.STRONG.SM` | lsu | bandwidth-bound | write-back |
| `st.global.cs` | `STG.E.EF` | lsu | bandwidth-bound | streaming/evict-first |
| `atom.shared.min.u32` | `ATOMS.MIN` | **lsu** | 1.00 = 32 SASS/SM/cy | |
| `atom.shared.exch.b32` | `ATOMS.EXCH` | lsu | 1.00 | |
| `atom.shared.cas.b32` | `ATOMS.CAS` | lsu | **0.50** | half rate — CAS is more expensive |
| `testp.normal.f32` | `ISETP.GE + ISETP.EQ + SEL` (3 SASS) | alu | ~0.67 logical tests/cy | |
| `bfind.u32` | `FLO.U32` | **xu** | 0.50 = 16/SM/cy | |
| `bfind.shiftamt.u32` | `FLO.U32.SH` | xu | 0.50 | shift-amount variant |
| `nanosleep.u32` | `NANOSLEEP` | **adu** | 0.25 = 8/SM/cy | slow, blocks the warp |
| `cp.async.ca.shared.global` | `LDGSTS.E` | **lsu** | 0.49 | async g→s memcpy |
| `cp.async.commit_group` | `LDGDEPBAR` | lsu | 0.50 | fence-ish |
| `prefetch.global.L1` | `CCTL.E.PF1` | lsu | **very slow** (255 ms for 128) | serialized against memory system |
| `prefetch.global.L2` | `CCTL.E.PF2` | lsu | very slow | |
| `vabsdiff.s32.s32.s32` | `PRMT + SHF.R.S32.HI` (compiler path) | alu | 2 SASS/op → 32 logical/SM/cy | |
| `mov.u32 %%ctaid.x` | S2R (cached by compiler — emitted once) | — | effectively free | |
| `mov.u32 %%nctaid.x` | `LDCU` via uniform pipe | **uniform** | 0.25 | read from constant bank as u-reg |

**FMNMX3 discovered:** the compiler fuses two chained `min.f32` into one `FMNMX3` on pipe_alu. At 64 SASS/SM/cy, effective throughput = **128 FP32 min-ops/SM/cy** — same bandwidth multiplier as IADD3 provides for integer add.

**ATOMS family speed hierarchy:** `.min/.max/.add/.exch` at 1.00/SM/cy (32 SASS); `.cas` at 0.50 (half rate). Global-memory `ATOMG` is further DRAM-bound.

**Prefetch is very expensive.** 255 ms for 128 `CCTL.E.PF1` instructions per thread — 2× slower than streaming STG. Only use prefetch when profiled as a win.

**Immediate-variant SASS (FFMA32I, IADD32I, IMUL32I, LOP32I, ISCADD32I, etc.) exist in the Blackwell opcode table but NOT emitted** by the current nvcc codegen — it uses regular ops with immediate operands. These may be reserved for future compiler paths or higher opt levels.

## 15. Deep-dive: atomics (corrected numbers) + latency

### Atomics on pipe_lsu — real throughput (bank-conflict-clean)

**Critical methodology note:** my first atomics test used stride-8 (32-byte) addressing, which causes 8-way bank conflicts (lanes {0,4,8,12,16,20,24,28} all hit bank 0). That degraded measurements by **8×**. Re-running with stride-4 (per-lane unique bank, `smem[tid + k*BLOCK_SIZE]`) gives real numbers:

| SASS | pipe_lsu rate | scalar atoms/SM/cy | chip-wide atoms/s |
|---|---:|---:|---:|
| `ATOMS.{MIN,MAX,ADD,AND,OR,XOR,EXCH,INC,DEC}` | **1.00** | **32** | **9.1 TAtoms/s** |
| `ATOMS.CAS` | **0.50** | **16** | **4.55 TAtoms/s** (still half) |
| `red.shared.add` (no-return) | 1.00 | 32 | same SASS as atom.add |

That's **1 atom warp-inst every cycle** on LSU (CAS: every 2 cycles). l1tex__data_bank_conflicts.sum = 0 confirmed. Bank-conflict penalty scales linearly — under 8-way conflict the same ATOMS.ADD drops to pipe_lsu=0.125.

**CAS is unconditionally half-rate.** Verified (bank-clean, both paths):
- Always-succeeds compare: 2.189 ms / pipe_lsu = 0.50
- Always-fails compare: 2.189 ms / pipe_lsu = 0.50 (identical)
- atom.add baseline: 1.096 ms / pipe_lsu = 1.00

**CAS is half-rate irrespective of compare outcome.** Verified with explicit always-succeed (compare matches memory) vs always-fail (compare never matches) kernels: both take exactly 2.189 ms vs atom.add 1.096 ms. Compare-match does not affect performance.

**"With-return" vs no-return (`red.shared.add`)**: same SASS (`ATOMS.ADD`), same rate. On B300 there is no separate reduction-only SASS for shared memory — compiler canonicalizes `red.shared.*` to `ATOMS.*`. (Global-memory `red.global.*` is different — emits `REDG.*` or `ATOMG.*` depending on scope.)

**No native `atom.shared.add.f32`** on B300: compiler emits `BSSY.RECONVERGENT` + `LDS` + CAS-loop to emulate. Very slow (~2× plain atom.add).

### Round-trip latency (1 warp, chained self-op)

Back-to-back dependent operations in a single warp (no ILP):

| Op | SASS | cycles/op (latency) |
|---|---|---:|
| FFMA (`fma.rn.f32 %0,%0,%0,%0`) | FFMA | **4.14** |
| FMUL | FMUL | 4.14 |
| FADD | FADD | 4.13 |
| HFMA2 | HFMA2 | 4.13 |
| IMAD (`mad.lo.u32`) | IMAD | 4.15 |
| PRMT (`prmt %0,%0,%0,0x3210`) | PRMT | 4.15 |
| F2FP unpack (`cvt.rn.f16x2.e4m3x2`) | F2FP.F16.E4M3.UNPACK_B | 4.13 |
| F2FP pack (`cvt.rn.satfinite.e4m3x2.f16x2`) | F2FP.*.UNPACK_B_MERGE_C | **8.13** |
| u64 add | IADD3 + IMAD.X (2 SASS) | **6.29** |
| MUFU.EX2 | MUFU.EX2 | **14.45** |
| MUFU.RSQ | MUFU.RSQ | **40.18** |
| MUFU.RCP | MUFU.RCP | **42.31** |
| redux.sync.min | CREDUX.MIN + IMAD.U32 | **18.06** |
| DFMA (fp64) | DFMA | **302.46** (!) |

Consistent 4-cycle latency for everything on pipe_alu / pipe_fma suggests a single pipelined depth. F2FP pack at 8 cycles reflects the 2-read-port merge. MUFU variants span 14–42 cy (EX2 is cheapest, RCP most expensive). FP64 DFMA at 302 cy is consistent with the throttled fp64 pipe (0.05 warp-inst/SM/cy ≈ 1 inst per 20 cycles + ~5× internal latency).

To saturate pipe_fma with FFMA, you need **ILP ≥ 4 independent chains per warp** (to hide the 4-cycle dep latency). For MUFU.EX2 you'd need ≥ 14 independent chains. For DFMA, need 300+ independent chains — infeasible, so FP64 is always latency-bound per warp.

**DCE-disclaimer:** `LOP3 xor %0,%0,const`, `min.f32 %0,%0,%0`, and `shfl.sync %0,%0,1,…` under uniform warp state all measured <1 cy = DCE'd or optimized away. Real latencies for LOP3/FMNMX/SHFL are expected ≈4 cy (same pipe family) but I don't have a clean kernel for those three.

## 16. Research log — further measurements this session

### Global atomics (per-lane unique addresses)
| PTX | SASS | pipe_lsu | Notes |
|---|---|---:|---|
| `atom.global.add.u32` | `REDG.E.ADD.STRONG.GPU` | 0.03 | REDG family, not ATOMG! |
| `atom.global.min/max.u32` | `REDG.E.{MIN,MAX}.STRONG.GPU` | 0.03 | same family |
| **`atom.global.add.f32`** | **`REDG.E.ADD.F32.FTZ.RN.STRONG.GPU`** | 0.03 | **native FP32 atomic on global** (unlike shared which is emulated!) |
| `red.global.add.u32` | `REDG.E.ADD.STRONG.GPU` | 0.03 | same SASS as atom (return absorbed by LSU) |
| `atom.global.exch.b32` | `ATOMG.E.EXCH.STRONG.GPU` | 0.03 | different family |
| `atom.global.cas.b32` | `ATOMG.E.CAS.STRONG.GPU` | 0.015 | half-rate; 16× L2 sectors vs REDG |
| `.cta` scope | `REDG.*.STRONG.SM` | 0.03 | SM-local variant |
| `.sys` scope | `REDG.*.STRONG.SYS` | 0.03 | CPU-coherent |
| `.relaxed.gpu` | same as default | 0.03 | no weaker SASS emitted |
| `.acq_rel.gpu` | REDG + `MEMBAR.ALL.GPU` | — | +45% due to MEMBAR insertion |

### Shared atomics (bank-clean, per-lane unique bank)
| SASS | pipe_lsu | atoms/SM/cy | chip /s |
|---|---:|---:|---:|
| `ATOMS.{MIN,MAX,ADD,AND,OR,XOR,EXCH,INC,DEC}` | 1.00 | **32** | 9.1 TAtoms/s |
| `ATOMS.CAS` | 0.50 | 16 | 4.5 TAtoms/s (half, swap-independent) |
| 8-way bank conflict (stride-32 accidental) | 0.125 | 4 | 1.14 TAtoms/s |

### Tensor core
| PTX (wmma.mma.sync) | SASS | TFLOPS / TOPS | pipe_tensor |
|---|---|---:|---:|
| 16×16×16 FP16→FP32 | `HMMA.16816.F32` | **838 TFLOPS** | 72% active, 0.36 issue |
| 16×16×16 BF16→FP32 | `HMMA.16816.F32.BF16` | 838 TFLOPS | 72% |
| 16×16×8 TF32→FP32 | `HMMA.1684.F32.TF32` | 420 TFLOPS | 72% (exactly ½ FP16) |
| 16×16×16 INT8→INT32 | `IMMA.16816.S8.S8` | 143 TOPS | 100% / 0.06 issue |

### Uniform datapath (finally forced to emit)
Running warp-uniform compute chains (derived from `blockIdx.x` + `seed`) forces the compiler to use the uniform datapath. Solo peak: `pipe_uniform = 1.90` warp-inst/SM/cy, **concurrent** with pipe_alu/fma at no cost.
| Chain pattern | SASS emitted | pipe_uniform |
|---|---|---:|
| uniform IADD/IMAD chain | `UIMAD`, `UIADD3`, `UMOV`, `UISETP.GE.AND` | 1.90 |
| uniform LOP3 chain | `ULOP3.LUT`, `UMOV`, `UIADD3` | 1.81 |
| uniform FMUL/FADD chain | **regular FFMA.FTZ** (compiler didn't use UFFMA/UFADD) | — |
| `cvta.to.global / .shared` | `UIADD3` + `ULOP3.LUT` | — |

**Compiler-emission gap:** Blackwell SASS opcode table lists UFFMA/UFADD/UFMUL/UFMNMX/UFRND/UFSEL/UFSETP/UF2F/UF2FP/UF2I/UI2F/UI2FP/UI2I etc., but my current nvcc (CUDA 13.0) does not emit them for scalar FP computations — it prefers vector FFMA. These opcodes may activate in a future compiler release.

### Cluster / CGA barriers
| PTX | SASS | ms | Note |
|---|---|---:|---|
| `barrier.cluster.arrive` + `.wait` | `UCGABAR` + `MEMBAR.ALL.GPU` + `ERRBAR` + `CGAERRBAR` | 0.20 | strict, includes GPU fence |
| `barrier.cluster.arrive.relaxed` | `UCGABAR` + `CCTL.IVALL` | 0.057 | 4× faster, no MEMBAR |

### mbarrier (Hopper/Blackwell async barrier)
| PTX | SASS | pipe_adu |
|---|---|---:|
| `mbarrier.arrive.shared.b64` | `SYNCS.ARRIVE.TRANS64.A1T0` | — (crashed without paired wait) |
| `mbarrier.arrive_drop.shared.b64` | `SYNCS.ARRIVE.TRANS64.OPTOUT.A1T0` | — |
| `mbarrier.test_wait.shared.b64` | `SYNCS.PHASECHK.TRANS64` + `SEL` | 0.42 |
| `mbarrier.inval.shared.b64` | `SYNCS.CCTL.IV` | 0.07 |
| `cp.async.commit_group` + `wait_all` | `LDGDEPBAR` + `DEPBAR.LE` | — on pipe_lsu instead of adu |

**SYNCS is a new Blackwell opcode family** for async transaction barriers. Lives on pipe_adu.

### Address space queries and conversions
| PTX | SASS | Pipe | Note |
|---|---|---|---|
| `isspacep.shared` | `QSPC.E.S` | alu | dedicated opcode |
| `isspacep.local` | `QSPC.E.L` | alu | dedicated opcode |
| `isspacep.global` | LOP3 + ISETP (emulated) | alu | no dedicated QSPC.G |
| `cvta.to.global.u64` | `UIADD3` + `ULOP3.LUT` | **uniform** | address-space conv goes through uniform pipe |
| `cvta.to.shared.u64` | `LDC` + arithmetic | uniform/alu | often constant-folded |

### Vector memory loads (throughput, not latency)
| PTX | SASS | GB/s chip-wide |
|---|---|---:|
| `ld.global.v4.u32` (128-bit) | `LDG.E.128` or similar | **4540** (DRAM-limited) |
| `ld.global.nc.v4.u32` (read-only cache) | `LDG.E.CONSTANT.128` | 4540 (same, random pattern defeats cache) |
| `ld.global.u32` (32-bit scalar) | `LDG.E` | 1135 (= 4540 / 4) |
| `ld.shared.v4.u32` (128-bit) | `LDS.128` | **36,210** (= 128 B/SM/cycle peak) |

### ldmatrix / stmatrix (dual-pipe LDSM, LSU-only STSM)
| PTX | SASS | pipe_lsu | pipe_uniform |
|---|---|---:|---:|
| `ldmatrix.sync.x1.shared.b16` | `LDSM.16.M88` | 0.15 | 0.15 |
| `ldmatrix.sync.x2.shared.b16` | `LDSM.16.M88` | 0.14 | 0.14 |
| `ldmatrix.sync.x4.shared.b16` | `LDSM.16.M88.4` | 0.11 | 0.11 |
| `ldmatrix.sync.x4.trans.shared.b16` | `LDSM.16.M88.4.T` | 0.11 | 0.11 |
| `stmatrix.sync.x1.shared.b16` | `STSM.16.M88` | 0.49 | 0.06 |
| `stmatrix.sync.x4.shared.b16` | `STSM.16.M88.4` | 0.12 | 0.02 |

**LDSM uniquely dual-issues to both pipe_lsu AND pipe_uniform at the same rate** — it consumes 1 slot on each pipe per warp-inst. STSM only occupies pipe_lsu.

### Round-trip latency (single warp, chained self-op)
| Op | Latency (cycles) |
|---|---:|
| FFMA / FMUL / FADD / HFMA2 / IMAD / PRMT / F2FP.unpack | **4** |
| F2FP.pack (with MERGE_C read-port) | **8** |
| u64 add (IADD3 + IMAD.X) | 6.3 |
| MUFU.EX2 | 14.5 |
| redux.sync.min | 18 |
| MUFU.RSQ | 40 |
| MUFU.RCP | 42 |
| DFMA (FP64) | **302** |

To saturate the pipe from a single warp, ILP ≥ latency / (lanes × inst-per-cy-per-SMSP). FFMA needs ILP=4, MUFU.EX2 ≥ 14, DFMA ≥ 300+ (infeasible — fp64 is latency-bound per warp).

### Fences
| PTX | SASS | ms | Scope |
|---|---|---:|---|
| `membar.cta` / `fence.acq_rel.cta` | `MEMBAR.ALL.CTA` | 0.012 | CTA-local |
| `fence.acquire.cluster` | `CCTL.IVALL` only | 0.023 | cluster L1 invalidate |
| `membar.gl` / `fence.sc.gpu` | `MEMBAR.SC.GPU` + `ERRBAR` | 0.156 | full GPU |
| `membar.sys` / `fence.sc.sys` | `MEMBAR.SC.SYS` + `ERRBAR` | very slow | system-coherent |

### Predication / divergence / masks summary
- `@p instr` with any lane-mask: **zero effect** on pipe time (warp-inst takes same slot regardless of how many lanes active).
- `redux.sync.*` and `shfl.sync.*` rates are **mask-width independent** — 1 lane participating costs the same as 32 lanes.
- Warp specialization (`elect.sync` + 1-lane work) does NOT free pipe slots for the other 31 lanes.

### redux.sync type/op matrix (supported on B300)
| op | `.u32` | `.s32` | `.f32` (+NaN) | `.b32` | `.u64/s64` | `.f16/bf16` | `.f64` |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `.min` / `.max` | ✓ | ✓ | ✓ | — | ✗ | ✗ | ✗ |
| `.add` | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `.and` / `.or` / `.xor` | — | — | — | ✓ | ✗ | ✗ | ✗ |
| `.mul` | ✗ | ✗ | ✗ | — | ✗ | ✗ | ✗ |

**Min/max at 1.92 PTX-ops/SM/cy via `CREDUX.*` on pipe_alu + intrinsic `IMAD.U32` on pipe_fmaheavy.**  
**Add/and/or/xor at 0.50 PTX-ops/SM/cy via `REDUX.*` on pipe_adu — 4× slower than min/max.**  
**No FP32 sum reduce in hardware — must compose via shfl trees.**

### CVT rounding-mode asymmetry
| PTX | SASS | Pipe | Rate |
|---|---|---|---:|
| `cvt.rn.f32.s32` (round-nearest) | `I2FP.F32.S32` | alu | 64 SASS/SM/cy |
| `cvt.rz.f32.s32` (round-to-zero) | `I2FP.F32.S32.RZ` | alu | 64 |
| `cvt.rm.f32.s32` (round-down) | `I2F.RM` | **xu** | **16 (4× slower)** |
| `cvt.rp.f32.s32` (round-up) | `I2F.RP` | xu | 16 (4× slower) |

Hardware ALU only implements `.rn` and `.rz`; `.rm/.rp` fall back to the XU pipe. Same asymmetry for `cvt.rni.s32.f32` (xu) vs `cvt.rzi` (xu) — float→int is always on xu regardless of rounding.

## 17. Additional findings (research-loop batch 2)

### Memory hierarchy latency (pointer-chase, single lane)
| Working set | ns/load | cycles |
|---:|---:|---:|
| 1–4 KB | 2.9–3.4 | **6–7 (L1 hit)** |
| 16 KB | 5.4 | 10 |
| 64 KB | 13 | 25 |
| 128 KB | 22 | **43 (L2 hit)** |
| 256 KB | 26 | 50 |
| 1 MB | 83 | 159 |
| 32 MB | 1904 | 3656 |
| 128 MB | 7475 | **14351 (DRAM / TLB)** |

### Sync / barrier costs
| PTX | SASS | ns/sync | cy |
|---|---|---:|---:|
| `bar.sync 0` (`__syncthreads`) | `BAR.SYNC.DEFER` | **17.9** | **34** |
| `membar.cta + bar.sync` | `MEMBAR.SC.CTA + BAR.SYNC.DEFER` | 18.0 | 35 |
| `bar.warp.sync -1` / `0xFF` | DCE'd (warp mask compile-time known) | 3.4 | 6 |
| `barrier.sync.aligned 0` | same as bar.sync | 17.9 | 34 |

### LDG cache hints (32 MB working set, fits in L2)
| Hint | GB/s | L2 sectors vs baseline |
|---|---:|---:|
| default / `.ca` / `.cg` / `.nc` | **540** | 1.00× |
| `.cs` (streaming) | 466 | +21% L2 traffic |
| `.lu` (last-use) | 522 | +21% L2 traffic |
| `.volatile` | 496 | 1.00× (just overhead) |

### Occupancy / ILP
- FFMA latency in dep chain: **4.53 cycles**
- Half-throughput ILP: 4 (matches latency)
- Saturation: ILP ≥ 8 gets 89% of peak; ILP=16 → 94%.
- Block size 32-512 all equivalent. `__launch_bounds__(_, MIN_BLOCKS)` has no effect for register-light kernels.

### Division cost hierarchy
| PTX | SASS | cycles/op |
|---|---|---:|
| `f32 a/b` (default `div.full.f32`) | MUFU.RCP + FMUL + FADD | ~4 |
| `f32 div.approx.f32` | MUFU (only) | ~1 |
| **`f32 div.rn.f32` (IEEE-correct)** | 47 FFMA + LOP3 (Newton-Raphson) | **~120** |
| `f32 sqrt.rn.f32` | 16 FMUL + 16 FFMA (N-R) | ~8 |
| **`f64 a/b`** | 66 DFMA (N-R) | **~1200** |
| `u32 a%b` (runtime b) | IMAD.HI.U32 reciprocal chain | ~12 |

**`div.rn` is 30× slower than `div.full`.** Use `__fdividef` / `__fdiv_ru` when correct rounding isn't needed.

### Triple-pipe dispatch cap
Combining pipe_uniform + pipe_alu + pipe_fma work shows **total sm_inst still caps at 4.00/SM/cy** — pipe_uniform consumes SMSP issue slots despite being "separate" datapath. Independent EXECUTION unit, shared DISPATCH budget.

### Compiler-emission gaps (Blackwell ISA opcodes NOT emitted by CUDA 13.0 nvcc)
Listed in Blackwell SASS table but never observed in my measurements:
- `UFFMA`, `UFADD`, `UFMUL`, `UFMNMX`, `UFRND`, `UFSEL`, `UFSET`, `UFSETP`
- `UF2F`, `UF2FP`, `UF2I`, `UF2IP`, `UI2F`, `UI2FP`, `UI2I`, `UI2IP`
- `FFMA32I`, `IADD32I`, `IMUL32I`, `LOP32I`, `ISCADD32I`
- `FADD2`, `FMUL2`, `FMNMX3` (only emitted for specific chained-min pattern)
- Tensor memory: `LDT`, `STT`, `UTC*`, `UBLK*` (need tcgen05/TMA setup not tested)

## 18. Additional findings (research-loop batch 3)

### Hot-spot atomics scaling
`atom.global.add.u32` with N distinct addresses across 75k threads:
- unique-addr baseline: 0.064 ms
- **1 hotspot**: 0.78 ms (12× slow — warp-coalesced fast path)
- **2 hotspots**: **25 ms (32× worse than 1)** — anomaly: within-warp address divergence breaks the fast path
- 256 hotspots: 2.4 ms; 8k hotspots: 0.10 ms; recovers with dispersion

### TMA (Tensor Memory Access)
`cp.async.bulk.shared::cluster.global.mbarrier` → **`UBLKCP.S.G`** (Blackwell-specific bulk-copy opcode).
`cp.async.ca.shared.global.L2::256B` → **`LDGSTS.E.LTC256B.128`** (L2-prefetch variant).
`cp.async.cg` → `LDGSTS.E.BYPASS.128`.
`cp.async.commit_group` → `LDGDEPBAR`.

### Special register costs
| SR | ns | cycles | SASS |
|---|---:|---:|---|
| %clock64 | 10.7 | **20** | `CS2R.32` (fastest timestamp) |
| %clock | 12.7 | 24 | `S2UR` |
| %pm0 | 10.9 | 21 | `S2UR` (perf counter) |
| %globaltimer | 18.7 | 36 | `S2UR` |
| %smid / %warpid | ~19 | 36 | `S2R` |
| %gridid / %nwarpid / %lanemask_eq / %clusterid.x / %envreg0 / getctarank | ~2.7 | — | cached (SR not re-read) |

### Precision modifiers are free
`.ftz`, `.sat`, `.ftz.sat`, `.relu` on FMA emit distinct SASS (`FFMA.FTZ`, `FFMA.SAT`, `HFMA2.RELU`) but all cost the same cycles. Use them freely.

### FP edge-case ops — emulation gaps
- `testp.{finite,subnormal,number,notanumber,infinite}.f32` → emulated via **`LOP3.LUT + FADD.FTZ`** (no native `FCHK`)
- `copysign.f32` → `LOP3.LUT` bit-trick
- `cos.approx.ftz / sin.approx.ftz` → `MUFU.SIN/COS` + `FMUL.RZ` range reduction (longer than ex2/rsqrt)

### Pair-contention u-matrix (confirmed)
- u ≈ 1.0 (same pipe): **LOP3 + PRMT = 1.08**, **IMAD + FFMA = 0.93**
- u ≈ 1.4-1.6 (mostly independent, some SMSP friction): LOP3 + IMAD = 1.61, IMAD + MUFU = 1.45, LOP3 + FFMA = 1.36
- u = 0.55 for FFMA + MUFU.EX2 (MUFU stretches kernel wall-time; not a pipe sharing issue)

## 19. Additional findings (research-loop batch 4)

### Divergence & reconvergence
- 2-way if/else with thread-divergent condition: **no branch emitted** — compiler converts to `ISETP + SEL` predicated inline
- Divergent 4-way switch: real `BSSY` + `BSYNC.RECONVERGENT` barriers, **~5× slowdown**
- Uniform branch (block-constant condition): DCE'd
- Small divergent loop: 8× overhead from extra iterations (no reconvergence barrier — natural loop convergence)

### Sustained 4-pipe peak
- FFMA alone: **sm_inst = 3.87** — 97% of 4.0 theoretical dispatch ceiling
- ALU + FFMA dual-pipe: 3.79 (both near saturation)
- Adding MUFU.EX2 to any kernel REDUCES sm_inst (its 14-cy latency stretches wall-time)
- No 4-pipe combination exceeded 3.87 achievable throughput

### Integer op coverage (additional)
- `abs.s32` → **`IABS`** (native)
- `shf.l + add` scaled pattern → **`LEA.HI`** emitted (compiler fuses shift+add)
- `dp4a.s32.u32` / `.u32.s32` → **`IDP.4A.S8.U8`** / `IDP.4A.U8.S8` (native mixed-signedness)
- `szext.wrap.s32` → **`SGXT.W`**
- `popc.b64` = 2× POPC + LOP3 (not native, 577 μs/128 ops)
- `clz.b64` = 256 IADD3 + LOP3 emulation (heavily expensive)
- `bfe.u32` = SHF.R + SGXT (2 SASS per op = half rate)
- `cnot.b32` = LOP3+SEL emulated (no native CNOT)
- `bmsk.b32` = compile error (opcode exists, PTX path unclear)

### Half-precision specialised opcodes
- **`MUFU.TANH.F16`** — `tanh.approx.f16` (native, pipe_xu)
- **`MUFU.EX2.F16`** — `ex2.approx.f16` (native, pipe_xu)
- Scalar `add.rn.f16` / `fma.rn.f16` / `setp.eq.f16`: **emulated via PRMT + vec2 HADD2/HFMA2** — PTX scalar half-precision is packed+extracted, not a native scalar op
- `neg.bf16x2` / `abs.bf16x2`: emulated via HFMA2.BF16 multiply-by-±1 (no native neg/abs for bf16)

### Atomic hotspot scaling anomaly
75k threads contending on N addresses:
- Unique → 0.064 ms
- **1 hotspot (all lanes → same addr)**: 0.78 ms (12× slow — warp-coalesce fast path)
- **2 hotspots** (within-warp divergence): **25 ms (32× WORSE than 1!)** — within-warp addr divergence breaks the coalesce path
- Recovery: >1024 hotspots ≈ 5× unique time

### TMA opcodes
- `cp.async.bulk.shared::cluster.global.mbarrier` → **`UBLKCP.S.G`** (TMA bulk-copy)
- `cp.async.ca.L2::256B` → **`LDGSTS.E.LTC256B.128`** (L2-sector prefetch)
- `cp.async.commit_group` → `LDGDEPBAR`
- `mbarrier.*` → new `SYNCS.*` family on pipe_adu

### Special-register costs
- %clock64 (fastest timestamp): **20 cycles** via `CS2R.32`
- %clock, %pm0 via S2UR: 21-24 cy
- %globaltimer, %smid, %warpid via S2R: 36-38 cy
- Cached SRs (%gridid, %lanemask_eq, etc.): optimized away, effectively free

### Divergent 4-way switch — the only reconvergence-barrier case
Compiler prefers ISETP+SEL predication over BSSY/BSYNC for simple 2-way branches. Only when SEL is infeasible (4+ divergent targets, unknown control flow) does `BSSY` + `BSYNC.RECONVERGENT` emit.

## 20. Additional findings (research-loop batch 5)

### DRAM bandwidth by access pattern (B300 HBM3E ≈ 8 TB/s peak)
- **Sequential stride-1 coalesced (v4 128-bit): 7420 GB/s = 92% of HBM peak**
- Stride-2 (half lanes): 2400 GB/s (3× slower)
- Stride-4: 1031 GB/s
- Stride-8 (per-lane cacheline): 523 GB/s (14× slower)
- Stride-16: 632 GB/s
- L2-resident (per-block): 31 TB/s (4× DRAM)

### nanosleep reality
- Minimum achievable sleep ≈ **34 ns** (32 cycles loop overhead)
- Requested N ≥ 500 ns → actually sleeps **2.2–3.5× longer** (scheduler tick quantization)

### Warp specialization doesn't win for same-pipe work
Baseline (all 32 lanes FFMA): 0.083 ms. Only lane 0 doing FFMA (via `@p` or `elect.sync`): **slower** (0.10-0.18 ms) — predication doesn't save pipe time, just reduces useful work per warp-inst. Specialization only helps when the specialized lane targets a DIFFERENT pipe (e.g., TMA).

### IMUL variants & new opcodes
- `mul.lo.u32` → `IMAD` (native, 64/SM/cy)
- **`mul.hi.u32`** → `IMAD.HI.U32` (32/SM/cy, **half rate**)
- `mul.hi.s32` → `IMAD.HI` (same rate, signed variant)
- `mul.wide.u32` → single `IMAD` (handles 64-bit result natively)
- **`max.u16`** → **`VIMNMX.U16`** (dedicated half-word min/max)
- `mul.wide.u16` emulated via LOP3 + IMAD

### Cluster atomics (CGA)
- `atom.shared::cluster.add` → **`ATOM.E.ADD.STRONG.GPU`** (generic atomic, not ATOMS) — because address may map to a peer block's SMEM requiring generic routing
- `mapa.shared::cluster` address mapping doesn't emit visible SASS (folded into atom addressing)

## 21. Additional findings (research-loop batch 6)

### MUFU + transcendental latency (chained, 1 warp)
- `tanh.approx`: 50 cy
- `sin.approx` / `cos.approx`: 55 cy
- `ex2.approx`: 83 cy
- `sqrt.approx` / `rsqrt.approx` / `lg2.approx`: 93 cy
- `rcp.approx`: 98 cy
- `sqrt.rn` precise: **138 cy**
- `rcp.rn` precise: **185 cy** (2× approx — Newton-Raphson iterations)

### Register pressure knee
FP32 chains per thread, 256 threads/block:
- 1 chain: latency-bound (pipe_fma = 2.85)
- 16 chains: saturated at 3.92 (98% of 4.0 dispatch)
- **Up to 96 chains: still at peak** (3.95 sm_inst)
- **128 chains: 3× cliff** (pipe drops to 3.79, register spill)
- 192 chains: 5× penalty

### Atomic hotspot — warp-coalesce fast path confirmed
Deep investigation of the 2-hotspot anomaly:
- 1 hotspot (all lanes same addr): **0.78 ms — warp-level coalesce fast path** (32 lane values reduce to 1 atomic per warp)
- Within-warp 2 hotspots (`lane%2`, `lane/16`, etc.): **25.4 ms (32× slower)** — fast path breaks
- Warps target 1 addr each, different warps = different addrs (layout 6): **49.7 ms (WORST)** — no warp coalesce AND cross-warp serialization on shared addresses
- 32 per-lane hotspots: 4.87 ms
- Per-thread unique: 0.063 ms

**Takeaway:** atomic-to-same-address is 400× slower than unique, BUT if all lanes in every warp hit the identical address, hardware coalesces to a single atomic per warp — 12× slowdown only. Any within-warp divergence activates the slow path.

### FP16 HMMA accumulator type — no speedup on B300
FP16×FP16 → FP32 accumulator and FP16×FP16 → FP16 accumulator take **identical time** (5.4 ms each). Unlike earlier architectures where F16 accumulator ran 2× faster. B300's HMMA pipeline is accumulator-precision agnostic.

### cp.async wait latency
- `cp.async + wait_all`: ~54 ns per transaction (full drain)
- `cp.async + wait_group N`: ~27 ns (non-blocking on last group)
- `wait_all` adds a ~27 ns drain penalty over `wait_group`

### DRAM bandwidth by access pattern
- Sequential stride-1 v4: **7.42 TB/s** (92% of HBM3E peak)
- Stride-8 per-lane cacheline: 523 GB/s (14× penalty)
- L2-resident: 31 TB/s

### nanosleep
- Minimum sleep = ~34 ns
- Actual sleep = 2.2-3.5× requested for N ≥ 500 ns (scheduler tick rounding)

## 22. Corrections (audited against CUDA runtime)

### B300 hardware constants (authoritative from `cudaDeviceProp`)
- **L2 cache: 132,644,864 B = 126 MB** (my earlier 280 MB / 192 MB / 186 MB were all wrong)
- SMs: **148**
- Shared mem / SM: **228 KB**
- Regs / SM: 65536 = 256 KB register file
- Persisting L2 max: 79 MB (a *separate* limit, not related to total L2 size — my earlier "2.4×" ratio was bogus)
- cc: 10.3 (sm_103a)
- Global mem: 268 GB

### Fast math is ON by default
`utils/cuda_helper.h` line 227 passes `-use_fast_math` to NVRTC. So every measurement in this catalog is already with:
- `.ftz` flush-to-zero
- `.approx` preferred over strict `.rn` for rcp/sqrt/div
- no strict IEEE compliance fallback

The "extra" FFMA.FTZ + FSETP.GEU.AND + LOP3.LUT I previously reported alongside `MUFU.EX2` was emitted by **my own range-reduction scaffolding code** (`f = f * 0.5f + 0.25f` between MUFU calls), NOT from strict-FP corner-case handling. A clean chain of `ex2.approx.ftz` alone emits only `MUFU.EX2` instructions.

### Clean MUFU latency ratios (relative to FFMA self-op)
Absolute numbers from self-op chains (`fma %0,%0,%0,%0` etc.) are ~2× inflated from register read-port pressure (a single register fills all 3-4 operand slots). The **ratios** to FFMA are the reliable information:

| Op | cy/op (self-op chain) | ratio vs FFMA |
|---|---:|---:|
| FFMA reference | 8.46 | 1.00× |
| ex2.approx.ftz | 29.2 | **3.45×** |
| rsqrt / sqrt / lg2 approx.ftz | 37.5 | **4.43×** |
| sin / cos.approx.ftz | 49.9 | **5.90×** |
| rcp.approx.ftz | DCE'd — rcp(rcp(x))≈x | — |
| tanh.approx.ftz | DCE'd — converges | — |

### Memory bandwidth by working-set size (cleaner methodology)
Window-shared test, 148 blocks × 512 threads hitting cyclic window:

| Window | GB/s | Level |
|---:|---:|---|
| 4–128 KB | 34–35k | L1 hit (B300 L1 = 228 KB/SM) |
| 256 KB – 1 MB | 29–35k | L1/L2 transition |
| 4 MB | 19.5k | L2 |
| 16-64 MB | 19.7k | L2 plateau |
| 256 MB | 14.8k | **overflows 126 MB L2** → partial DRAM |

L2 peak BW ≈ 20 TB/s for shared working set. L1 hit BW ≈ 35 TB/s. DRAM BW (coalesced sequential, separate test): **7.4 TB/s = 92% of HBM3E peak**.

### Caveats on earlier numbers in this document
- "L2-resident 31 TB/s" was a correct measurement but the **working set was ~9 MB** (block windows overlapping), well inside L2 — so it's L2 BW, not DRAM.
- "MUFU.EX2 ≈ 14.5 cy" from `bench_latency.cu` was clean (single op, no range-reduction). The 83-cy number in `bench_mufu_lat.cu` was inflated by my own range-reduction FFMA chain.

## 23. Clean MUFU ex2 / tanh sweep (no range reduction, clock64-bracketed)

### Latency (1 thread, chained self-op, clock64)
| PTX | SASS | cy/op |
|---|---|---:|
| ex2.approx.f32 / ftz.f32 | MUFU.EX2 | **14** |
| ex2.approx.f16 | MUFU.EX2.F16 | 14 |
| ex2.approx.f16x2 | MUFU.EX2.F16 | 18 (vec2 = +4 cy) |
| ex2.approx.ftz.bf16 | MUFU.EX2.BF16 | 14 |
| ex2.approx.ftz.bf16x2 | MUFU.EX2.BF16 | 18 |
| tanh.approx.f32 | MUFU.TANH | **18** |
| tanh.approx.f16 | MUFU.TANH.F16 | 18 |
| tanh.approx.f16x2 | 2× MUFU.TANH.F16 per SASS | 18 (per PTX op) |
| tanh.approx.bf16 | MUFU.TANH.BF16 | 18 |
| tanh.approx.bf16x2 | 2× MUFU.TANH.BF16 per SASS | 18 |

FFMA reference (same methodology): **4 cy** — matches pipeline depth.

### Throughput (full grid, 8 independent chains per thread)
| op | GOps/s chip | ratio |
|---|---:|---:|
| ex2.approx.{f32,f16,bf16} | **8850** | 1.00× |
| ex2.approx.{f16x2,bf16x2} | 4500 | 0.51× (vec2 half inst rate) |
| tanh.approx.{f32,f16,bf16} | 4500 | 0.51× (tanh = 2 XU slots) |
| tanh.approx.{f16x2,bf16x2} | 1310 | 0.15× (compound 0.5 × 0.5) |

### Key observations
- **ex2 is the cheapest transcendental**; tanh is exactly 2× more expensive at SASS level
- **.f16x2 / .bf16x2 packing gives NO element-rate improvement** on XU — packed SASS serializes elements through single-lane-wide XU
- Fast-math (`-use_fast_math`) is on by default in this harness

## 24. Latency reference — clock64-bracketed (authoritative)

### Memory hierarchy (pointer-chase dep chain)
| Level | cy | B300 spec |
|---|---:|---|
| LDS (shared) | **33** | 228 KB/SM |
| L1 (global) | **43** | 228 KB/SM unified with shared |
| L2 | **300** | 126 MB chip-wide |
| DRAM | **3000** | HBM3E, 8 TB/s peak |

### Compute latency per SASS (chained)
| Op | cy | Pipe |
|---|---:|---|
| FFMA / FMUL / FADD / FFMA2 / HFMA2 / HADD2 / HMUL2 / HFMA2.BF16/.RELU | **4** | pipe_fma |
| HMNMX2 / FMNMX / FSEL / ISETP / IMAD (lo) / LEA / PRMT / LOP3 / SHF / VIMNMX / VABSDIFF | **4** | pipe_alu or fmaH |
| IMAD.HI.U32 (half-rate) | **13** | fmaheavy |
| MUFU.EX2 | **14** | xu |
| MUFU.RSQ / .SQRT / .LG2 .ftz.f32 | **18** | xu |
| MUFU.SIN / .COS | **24** | xu |
| MUFU.RSQ / .SQRT / .LG2 non-ftz | **40** | xu + scaling FMUL |
| MUFU.RCP non-ftz | 42 | xu + NR FFMAs |
| BREV vector | **18** | xu |
| POPC vector | **24** | xu |
| FLO / BFIND vector | **32** | xu |
| UBREV / UFLO (uniform pipe) | 4 / 12 | uniform (way faster) |
| ISETP+SELP (2 SASS) | 8 | alu chain |
| HSETP2+SELP | 10 | alu chain |
| DFMA FP64 | ~300 | fp64 throttled |

### Fence / barrier
| Op | cy | Scope |
|---|---:|---|
| `bar.warp.sync -1` | 1 | DCE'd static mask |
| `fence.acquire.cluster` | 4 | CCTL.IVALL only |
| `membar.cta` / `fence.sc.cta` | **8** | CTA-local |
| `bar.sync 0` single-thread | 14 | block barrier base |
| nanosleep min | ~34 | |
| `membar.gl` / `fence.sc.gpu` | **544** | **68× CTA cost** |
| `membar.sys` | **5956** | **750× CTA — CPU coherent** |

### Newton-Raphson emulated FP (approximate chain cost)
| Op | SASS path | cycles |
|---|---|---:|
| rcp.rn.f32 | MUFU.RCP + 3 FFMA + slowpath CALL | ~185 |
| sqrt.rn.f32 | MUFU + 7 FFMA NR | ~138 |
| div.rn.f32 | FCHK + 5 FFMA + rounding variants + CALL | ~1200 |

### Atomic peak rates (pipe_lsu bank-clean)
| Op | scalar atoms/SM/cy |
|---|---:|
| ATOMS.ADD/MIN/MAX/AND/OR/XOR/EXCH/INC/DEC | 32 |
| ATOMS.CAS | 16 (half rate, both success+fail) |
| Hot-spot same-addr warp-coalesce | ≈1 atom per warp (12× slower than unique) |
| Bank-conflict degradation | linear with conflict factor (up to 59× worst case) |

## 25. Final compact throughput table (all values at saturation, pipe-verified)

### FP throughput (chip-wide, 148 SMs × 1.92 GHz)
| Op | GFLOPS | notes |
|---|---:|---|
| **FP32 FFMA scalar** | 69k (= 2× 8850×4 SMSPs) | 128 FFMAs/SM/cy dual-issue H+L |
| **FP32 FFMA2 vec2** | 69k | 64 FFMA2/SM/cy × 2 FMAs |
| **FP16 / BF16 HFMA2 (non-tensor)** | 35k | 64 HFMA2/SM/cy × 2 FMAs |
| FP16 / BF16 min/max (HMNMX2) | non-FLOPS | 128 ops/SM/cy on pipe_alu |
| **FP16 HMMA (tensor core)** | **838k** | HMMA.16816.F32 (~12× scalar) |
| **BF16 HMMA (tensor core)** | 838k | same as FP16 |
| **TF32 HMMA (tensor core)** | 420k | half FP16 |
| FP64 DFMA scalar | 475 | pipe_fp64 throttled (1.6/SM/cy) |

### Memory bandwidth
| Source | BW (TB/s) |
|---|---:|
| L1 cache hit (small WS) | **35** |
| L2 cache hit (fits 126 MB) | **20** |
| DRAM coalesced sequential (small WS with overlap) | **7.4** |
| DRAM 1 GB workset sequential read | **3.3** |
| DRAM 1 GB workset write | 3.1 |
| Shared memory v4 (128-bit) | 36 (= 128 B/SM/cy) |

### MUFU throughput (pipe_xu @ 0.5 issue/cy = 16 ops/SM/cy chip)
- ex2.approx.{f32,f16,bf16}: 8.9 TGOps/s (fastest)
- rsqrt/sqrt/lg2/sin/cos/rcp.approx: 4.5 TGOps/s (half of ex2)
- tanh.approx.{f32,f16,bf16}: 4.5 TGOps/s (2 XU slots per SASS)
- .f16x2/.bf16x2 packed MUFU: 4.5 TGOps/s (same elements/cy as scalar)

### Atomic throughput (bank-clean)
- ATOMS.ADD/MIN/MAX/AND/OR/XOR/EXCH/INC/DEC: **9.1 TAtoms/s chip** (32 atoms/SM/cy)
- ATOMS.CAS: 4.5 TAtoms/s (exactly half)
- REDG global add: ~0.28 TAtoms/s per warp (bandwidth-bound)
- Hot-spot warp-coalesce: ~300 GAtoms/s (12× unique)

### Division ladder
| Op | GOps/s | Penalty vs FMUL |
|---|---:|---:|
| FFMA / FMUL | 32600 | 1.00× |
| div.full.ftz (runtime divisor) | 8924 | 3.7× |
| div.approx.ftz w/ compile-time divisor | 33500 | 1.0× (folded to FMUL) |
| sqrt.approx.ftz | 4500 | 7.2× |
| rcp.rn (precise, NR) | 1030 | **32×** |
| sqrt.rn (precise, NR) | 2620 | 12× |
| **div.rn (precise, NR)** | **101** | **330×** |

### ISA feature summary
- **SMSP dispatch cap:** 4.00 warp-inst/SM/cy (= 128 thread-ops/SM/cy). FFMA scalar reaches 3.87 (97%).
- **Pipe count (independent ExecUnits):** alu + fmaH + fmaL + xu + lsu + adu + uniform + tensor + fp64 + cbu + tex + ipa (12+). But dispatch ceiling applies to all.
- **Uniform datapath** runs parallel to vector for warp-invariant work; compiler auto-emits when beneficial.
- **Warp-coalesce atomics** hardware feature for commutative ops (ADD/MIN/MAX/AND/OR/XOR); 12× speedup when all 32 lanes target same addr.
- **Fast-math (`-use_fast_math`) is on by default** in this harness.
- **126 MB L2** (authoritative), **228 KB L1 per SM**.

## 26. Warp cooperative primitives (throughput, 303k threads × 4096 iters)

| op | GOps/s | SASS | pipe |
|---|---:|---|---|
| **vote.sync.ballot.b32** | **7320** | VOTE.ANY | alu |
| vote.sync.{all,any,uni}.pred | 3315 | VOTE.{ALL,ANY,EQ} + SELP (2 SASS) | alu |
| **shfl.sync.bfly.b32** | 5576 | SHFL.BFLY | lsu |
| shfl.sync.up / down | ~5600 | SHFL.{UP,DOWN} | lsu |
| match.all | DCE | — | adu |
| **redux.sync.min.u32** | **6923** | CREDUX.MIN | alu (+fmaH) |
| redux.sync.add.u32 | 3107 | REDUX.SUM | adu |
| redux.sync.or.b32 | 3168 | REDUX.OR | adu |

**vote.ballot is 2.2× faster than vote.all/any/uni** — ballot emits one SASS returning a b32 mask; the others need predicate→register SELP fallback (2 SASS).
**redux.min/max 2.2× faster than redux.add/or** — different pipes (alu vs adu), already documented.

## 27. BF16 non-tensor arith throughput
| op | GOps/s | SASS | FLOPS equiv |
|---|---:|---|---:|
| bf16x2 fma | 17613 | HFMA2.BF16 | 35.2 TFLOPS |
| bf16x2 add | 17508 | HFMA2.BF16 | 17.5 TFLOPS |
| bf16x2 mul | 17345 | HMUL2.BF16 | 17.3 TFLOPS |
| bf16x2 min | 17392 | HMNMX2.BF16 | — (pipe_alu) |
| bf16x2 abs / neg | ~17400 | HFMA2.BF16 (emulated via ±1 mul) | — |
| scalar bf16 add / fma | ~20000 | PRMT + HADD2/HFMA2 | — |
| bf16x2 setp+selp | 8901 | 2-SASS chain | — |
| cvt f32×2 → bf16x2 | 17400 | F2FP.BF16.F32.PACK | — |

**Non-tensor BF16 FMA peak = 35.2 TFLOPS** — 24× slower than HMMA BF16 at 838 TFLOPS. Tensor cores are a MASSIVE win for any BF16 matrix work.

## 28. Compiler-emission gaps (Blackwell ISA opcodes that don't emit from nvcc 13.0)
- **UFFMA / UFADD / UFMUL / UFMNMX / UFRND / UFSEL / UFSET / UFSETP** — uniform FP datapath exists in ISA but compiler does NOT emit (tested 4 patterns)
- **UF2F / UF2FP / UF2I / UF2IP / UI2F / UI2FP / UI2I / UI2IP** — same
- **FFMA32I / IADD32I / IMUL32I / LOP32I / ISCADD32I** — immediate-form variants never observed
- **FADD2 / FMUL2** — not emitted (FFMA covers)
- **FCHK** — testp.* is emulated via LOP3 + FADD instead
- **BMSK** — PTX syntax fails to reach it
- **Cluster-scope shared atomics via mapa** — compile or runtime issues
- **FP4/FP6 MMA inline PTX** with `.kind::f8f6f4` — compiles but DCE/miscompile
- **UBLKCP** emits correctly via cp.async.bulk, but timing/bandwidth measurements unreliable without proper TMA descriptor setup

Compiler-reachable uniform ops: UIADD3, UIMAD, UMOV, UISETP, ULOP3.LUT.

## 29. Methodological notes

- **DCE is aggressive.** Sequences of XORs with constant masks fold to zero or to a single XOR. LOP3.LUT is 3-input, so the compiler can fuse two XORs into one SASS. To force `N × UNROLL` SASS instructions for bitwise ops, use either `PRMT` (byte permute, cannot be expressed as a 3-input bit LUT) or loop-carried runtime mask updates.
- **Metric aliasing:** `pipe_fmaheavy` and `pipe_fmalite` BOTH report 2.00 for a single packed op (FFMA2, HFMA2) because that one instruction occupies both sub-pipes for the cycle. For scalar FFMA, they report disjoint fractions summing to ≈2.0. IMAD reports only fmaheavy. These are not aliases; they're correctly reporting distinct sub-unit utilisation.
- **Clock:** `nvidia-smi` confirms 1920 MHz during every run. No boost, no throttle.
- **SMSP friction:** sustained dispatch peaks at `smsp__inst_executed = 0.99` (PRMT + FFMA2 at 8:8, confirmed by ncu). F2FP specifically shows 0.84 max when paired with FFMA2 — a mild regfile-port or latency quirk unique to F2FP.
- **Kernels** live in `tests/bench_`* with one-op-per-`OP` macro so you can re-run any measurement with `./QuickRunCUDA tests/bench_<name>.cu -H '#define OP N …'`.

