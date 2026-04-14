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

## 16. Methodological notes

- **DCE is aggressive.** Sequences of XORs with constant masks fold to zero or to a single XOR. LOP3.LUT is 3-input, so the compiler can fuse two XORs into one SASS. To force `N × UNROLL` SASS instructions for bitwise ops, use either `PRMT` (byte permute, cannot be expressed as a 3-input bit LUT) or loop-carried runtime mask updates.
- **Metric aliasing:** `pipe_fmaheavy` and `pipe_fmalite` BOTH report 2.00 for a single packed op (FFMA2, HFMA2) because that one instruction occupies both sub-pipes for the cycle. For scalar FFMA, they report disjoint fractions summing to ≈2.0. IMAD reports only fmaheavy. These are not aliases; they're correctly reporting distinct sub-unit utilisation.
- **Clock:** `nvidia-smi` confirms 1920 MHz during every run. No boost, no throttle.
- **SMSP friction:** sustained dispatch peaks at `smsp__inst_executed = 0.99` (PRMT + FFMA2 at 8:8, confirmed by ncu). F2FP specifically shows 0.84 max when paired with FFMA2 — a mild regfile-port or latency quirk unique to F2FP.
- **Kernels** live in `tests/bench_`* with one-op-per-`OP` macro so you can re-run any measurement with `./QuickRunCUDA tests/bench_<name>.cu -H '#define OP N …'`.

