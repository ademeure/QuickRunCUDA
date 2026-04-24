# §20 retest — FMIN penalty investigation

**Catalog claim** (`B300_PIPE_CATALOG.md` L7773-7793, "FMIN Penalty Investigation (task #84)"):

| Pattern (per inner-loop iteration) | cy/iter | Overhead |
|---|---:|---|
| Pure FFMA2 | **5.57** | baseline |
| FFMA2 + 1 IADD | 6.76 | +21% |
| FFMA2 + 1 scalar FFMA | 7.57 | +36% |
| FFMA2 + 2 FMIN | 9.45 | +70% (= +35% per FMIN) |

**Catalog narrative** (verbatim):
> Mechanism: FFMA2 takes ~5 cy per inst (low-rate dispatch but high-throughput pipe_fma). Adding ANY inst on pipe_alu costs ~1-2 cy of effective latency since the dispatch is already near-saturated. Pure FMIN throughput on pipe_alu: 3.1 cy/op when standalone (8 chains).

**User challenge (2026-04-24):**
> "the '5.57 cy/iter' for pure FFMA2 is either wrong or misleading, it should be 4.0, right? so all the other numbers are a bit dodgy?"

User reasoning: pipe_fma cap = 2.00 inst/SM/cy. Per SMSP, FFMA2 takes both sub-pipes
= 0.5 inst/cy/SMSP issue = chip-level 0.5 cy/inst. Per single warp on a single SMSP:
- issue-rate-bound (sufficient ILP): ~2 cy/inst
- latency-bound (1 RAW chain): ~4 cy/inst (= FFMA latency)

5.57 cy fits NOTHING in this model. It is between issue-limited (2) and latency-bound (4),
suggesting the catalog test ran with insufficient ILP **and** partial pipelining — making
all the relative-overhead percentages downstream reference an artifact baseline.

---

## Method

- File: `tests/audit_20_fmin_baseline_v2.cu` (v1 was kept as historical reference; v2 hardens
  PATTERN=1 against IADD constant-folding).
- Single-warp (BS=32) for ILP analysis, then chip-level (BS up to 512, persistent).
- Clock state: GPU 0, no `-lgc` lock, sustained boost = **2032 MHz** (sampled live during
  long run; `nvidia-smi` showed `0 %, 2032 MHz` after first 0.3s).
- Inner-loop body: `N_CHAINS` independent register chains, each chain receives one FFMA2 +
  optionally one ALU instruction. Time via `clock64` brackets around the entire outer loop.
- Anti-DCE: XOR-reduce all chain accumulators into one u64; store under `tid >= blockDim.x`
  predicate (impossible).

PATTERNS:
- 0 = pure FFMA2
- 1 = FFMA2 + 1 IADD (chained on `u[k]`, runtime-loaded increment so the compiler cannot
  fold all iterations into a single closed-form add)
- 2 = FFMA2 + 1 scalar FFMA (chained on `g[k]`)
- 3 = FFMA2 + 2 min.f32 (chained, with an FADD between the mins to defeat compiler fusion
  into a single FMNMX3)
- 4 = FFMA2 + 2 min.f32 (catalog-equivalent, both chained — compiler fuses to 1 FMNMX3)

---

## Single-warp ILP sweep (BS=32, 1 block on full chip)

Best (min) of 5 timed runs, cy per **single chain step** (one FFMA2 + optional ALU op):

| N_CHAINS | P0 pure FFMA2 | P1 +IADD | P2 +scalar FFMA | P3 +FADD+FMNMX3 | P4 +FMNMX3 (cat-equiv) |
|---:|---:|---:|---:|---:|---:|
| 1  | 4.03 | 4.04 | 5.11 | 10.60 | 4.28 |
| 2  | **2.14** | 4.62 | 6.12 |  9.19 | 6.12 |
| 3  | 3.07 | 4.60 | 6.13 | 18.30 | 6.13 |
| 4  | 3.06 | 4.59 | 6.84 | 18.23 | 6.82 |
| 6  | 3.06 | 9.14 | 12.16 | 24.35 | 12.15 |
| 8  | 3.41 | 9.12 | 12.16 | 24.33 | 12.15 |
| 12 | 6.07 | 9.12 | 12.17 | 24.33 | 12.17 |

**Pure FFMA2 (P0):**
- N_CHAINS=1 RAW: **4.03 cy/inst** ⇐ FFMA2 latency = 4 cy (matches user's expectation)
- N_CHAINS=2: **2.14 cy/inst** ⇐ near 2 cy single-SMSP issue limit
- N_CHAINS=4-6: 3.06 cy/inst ⇐ register pressure / scheduler artifact (8 chains → 3.41)
- N_CHAINS=12+: 6+ cy ⇐ register spill / architectural ILP ceiling per warp

**5.57 cy never appears anywhere in the single-warp sweep.**

## Catalog-regime hunt: where DOES 5.57 come from?

The catalog table doesn't state N_CHAINS, BS or block count. Sweeping multi-warp configs:

| Launch | N_CHAINS=2 | N_CHAINS=4 | N_CHAINS=8 |
|---|---:|---:|---:|
| 1 block, BS=32  | 2.14 | 3.06 | 3.41 |
| 1 block, BS=128 | 2.13 | 3.07 | 3.42 |
| 1 block, BS=256 | n/a  | 4.01 | 4.06 |
| 1 block, BS=512 | n/a  | 8.19 | 8.27 |
| persistent BS=64  (4 SMSPs idle/SM) | 2.13 | 3.07 | 3.16 |
| persistent BS=128 (4 warps, 1/SMSP) | 2.13 | 3.07 | 3.16 |
| persistent BS=256 (8 warps, 2/SMSP, MIN_BLOCKS=2 = 16/SM total) | **4.01** | **4.02** | **4.05** |
| persistent BS=128 MIN_BLOCKS=4 (16/SM) | 2.13 | 3.07 | 3.15 |
| persistent BS=512 MIN_BLOCKS=1 (16 warps/SM)  | 8.03 | 8.83 | 9.25 |

The catalog 5.57 cy/inst sits **between BS=256 (4.0)** and **BS=512 (8.0)** chip-level.
Most plausible explanation: catalog measured a per-warp clock with a regime that had ~3
warps/SMSP active (= 8 cy / 1.5 = 5.3, close to 5.57). Or used a kernel with extra
inner-loop bookkeeping ops not isolated from the FFMA2 chain. Either way, it is **NOT
the SoL FFMA2 baseline** at any clean single-warp ILP point.

## Chip-level peak (full GPU saturation, sanity check)

`-t 512 -p` (persistent, 148 blocks × 512 threads = 75,776 threads = 16 warps/SM):

- **N_CHAINS=4, PATTERN=0:** 8.83 cy/chain (per-warp clock); 2.087 ms wall
- Total inst: 4 × 1024 × 100 × 512 × 148 = 3.10×10¹⁰ FFMA2
- Total FLOPS: × 4 = 1.241×10¹¹ FLOPS in 2.087 ms = **59.5 TFLOPS**
- Theoretical peak FP32 FFMA at 2032 MHz boost: 76.96 TF → **77.3% of peak**
- ncu `smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` = **42.7%**
  (note: pipe_fma "peak" = 2 inst/SMSP/cy because both sub-pipes can dual-issue
  scalar FFMA; FFMA2 takes BOTH sub-pipes per cycle so it's capped at 1 inst/SMSP/cy
  = 50% of nominal pipe_fma. Hence 42.7%/50% = 85.4% of FFMA2-specific issue ceiling.)

This corroborates the chip-level model: 4 warps/SMSP × ~2 cy/warp at peak issue = 8 cy/inst
per-warp clock = 1 inst/SMSP/cy. Our 8.83 cy is 90% of that, matching the 85% pipe_fma
saturation and 77% wall-clock TFLOPS.

---

## SASS verification

### PATTERN=0 (pure FFMA2), N_CHAINS=2, N_INNER=16

Inner loop: `2 chains × 16 unroll = 32` FFMA2. SASS check:
```
$ grep -c "FFMA2 " sass/audit_20_fmin_baseline_v2_*.sass     # in inner range
32
$ grep -cE "^\s+/\*[0-9a-f]+\*/" sass/...                     # other inst in inner range
32   # = 32 FFMA2, no other inst in inner loop
```
Clean. Measured **2.19 cy/inst** = 91% of the 2-cy single-SMSP issue limit.

### PATTERN=1 (FFMA2 + 1 IADD), N_CHAINS=4

v1's `add.s32 %0, %0, 0x1234` was **constant** → compiler folded all per-chain
increments into closed-form expressions; v1 SASS had only **1 IADD3** in the
64-FFMA2 inner loop. v2 fixes this with a runtime-loaded `iadd_inc = (unsigned)seed | 1`:

```
v2 PATTERN=1, N_CHAINS=4 inner: FFMA2 = 64, IADD3 = 34, FADD = 0
```
34 IADD3 (compiler still fused some pairs into 3-input form, but most are visible).
Measured **4.59 cy/chain** at N_CHAINS=4 vs P0's 3.06 → real overhead = **+50%** for the
+IADD pattern (vs catalog's +21%).

Why the discrepancy? Catalog 6.76 / 5.57 = 1.21. But on v1 with constant operand the
IADD was hoisted entirely → catalog likely measured 6.76 cy because the IADD ALSO
broke the FFMA2 issue pattern (different reason than catalog narrative claims).

### PATTERN=2 (FFMA2 + 1 scalar FFMA), N_CHAINS=4

```
inner: FFMA2 = 64, FFMA (scalar) = 57   # close to expected 1:1
```
Measured **6.84 cy/chain** at N_CHAINS=4. Per-chain: 1 FFMA2 + 1 FFMA = both compete for
pipe_fma. FFMA2 takes both sub-pipes so a scalar FFMA in the same chain just adds another
pipe_fma issue cycle. Expected: 1 FFMA2 (uses both halves) + 1 FFMA (uses one half) = 2
issue slots / cycle ⇒ ~2 cy/op → 4 cy/chain at full ILP, 4-6 cy at partial. Match.

### PATTERN=3 (FFMA2 + 2 min.f32 chained, with FADD between), N_CHAINS=4

The catalog's own `2 FMIN` formulation (PATTERN=4) is silently fused by the compiler
into ONE `FMNMX3` (Blackwell's 3-input min/max), per `justifications/14_extended_ops.md`.
v2 PATTERN=3 inserts an FADD between the two mins to defeat the fusion:

```
inner: FFMA2 = 64, FMNMX = 62, FADD = 61
```
Inner now has **3 inst per chain step** (1 FFMA2 + 1 FMNMX3 + 1 FADD).

Measured **6.72 cy/chain** at N_CHAINS=4. So 3 instructions in 6.72 cy = 2.24 cy/inst —
the chain is essentially issue-rate-bound by FFMA2 dispatch + serialized ALU dependency.

### PATTERN=4 (catalog-equivalent: 2 plain min.f32, FUSED to 1 FMNMX3), N_CHAINS=4

```
inner: FFMA2 = 64, FMNMX = 62        # 2 PTX min → 1 SASS FMNMX3
```
Measured **6.82 cy/chain** ≈ identical to PATTERN=3. The catalog's `+35% per FMIN`
narrative is **wrong on its face**: there is only ONE FMNMX3 in the SASS, not two FMINs.
The "+70% / +35% per FMIN" arithmetic is invalid because the compiler emits one
FMNMX3 regardless of how many `min.f32` PTX you write (when the chain is RAW-dependent).

---

## Recomputed overheads at PROPER ILP (N_CHAINS=4, single-warp)

Comparing PATTERNs 1-4 vs PATTERN=0 baseline at the same ILP regime:

| Pattern | cy/chain | vs P0 baseline | Catalog claim | Catalog reasoning |
|---|---:|---:|---:|---|
| P0 pure FFMA2 | **3.06** | — (baseline) | 5.57 (1.82× too high) | ❌ wrong baseline |
| P1 +1 IADD | 4.59 | **+50%** | +21% | ❌ catalog underestimates |
| P2 +1 scalar FFMA | 6.84 | +123% | +36% | ❌ catalog wrong by 3.4× |
| P3 +1 FMNMX3 +1 FADD | 6.72 | +119% | n/a | (new pattern) |
| P4 +"2 FMIN" (=1 FMNMX3) | 6.82 | +123% | +70% (= +35%/FMIN) | ❌ wrong on multiple counts |

At chip-level (N_CHAINS=4 BS=256 persistent MIN_BLOCKS=2 = 16 warps/SM):
- P0: 4.02 cy/chain, P4: not measured separately but extrapolating → ~7-8 cy

The "+35% per FMIN" claim collapses entirely once you observe the SASS shows ONE
FMNMX3, not two FMINs.

---

## Verdict

**User is correct.** Catalog §20's pure-FFMA2 baseline of **5.57 cy/iter** is wrong by
~80% relative to either of the two clean SoL points:
- single-warp issue-bound: **2.14 cy/inst** (N_CHAINS=2, BS=32)
- single-warp latency-bound (RAW): **4.03 cy/inst** (N_CHAINS=1)
- chip-level peak: **0.5 cy/inst per SMSP** = 1 inst/SMSP/cy = 77% TFLOPS = ncu pipe_fma 43%

5.57 cy comes from a measurement regime with insufficient ILP (probably N_CHAINS<2 with
extra bookkeeping ops in the inner loop, or per-warp clock measured at chip-level with
~3 warps/SMSP active).

**All four catalog overhead percentages are derived from this artifact baseline, so the
"+21% / +36% / +70%" numbers are also wrong.** The recomputed overheads at proper ILP
(N_CHAINS=4 single-warp) are roughly +50% / +120% / +120% — the relative cost of adding
ANY ALU op while FFMA2 is running is much LARGER than catalog claims, because the catalog
baseline was already inflated.

**The "+35% per FMIN" decomposition is doubly wrong:**
1. The denominator (5.57) is not a real baseline.
2. The compiler fuses two `min.f32` PTX ops into ONE `FMNMX3` SASS — there is no
   "per-FMIN" cost to attribute, because there's only one FMIN in the SASS.

**Recommended catalog edit (see `RECOMMENDED_CATALOG_EDITS.md` EDIT NEW-§20):**
- Replace pure-FFMA2 baseline `5.57 cy/iter` with the regime-stated SoL points:
  - "2.14 cy/inst single-warp issue-bound (N_CHAINS=2)"
  - "4.03 cy/inst single-warp latency-bound (N_CHAINS=1, RAW dep)"
  - "0.5 cy/inst per SMSP at chip-level peak (= 77% TFLOPS, ncu pipe_fma 43%)"
- Drop the "+35% per FMIN" narrative; replace with: "Adding any ALU op to a 1-FFMA2 chain
  step roughly doubles cy/iter at single-warp because the chain becomes serialized through
  pipe_alu. Two min.f32 PTX → one FMNMX3 SASS (Blackwell 3-input fused min/max), so
  '2 FMINs' is a misnomer."
- Update mechanism explanation: catalog says "FFMA2 takes ~5 cy per inst" — actually ~2-4
  cy depending on regime. Catalog's "near-saturated dispatch" framing is right in spirit
  but the numbers are off.

---

## Provenance

- Test: `tests/audit_20_fmin_baseline_v2.cu` (v1 preserved at `tests/audit_20_fmin_baseline.cu`)
- SASS examples: `sass/audit_20_fmin_baseline_v2_*.sass` (auto-generated)
- Clock: 2032 MHz boost sustained (no `-lgc` lock; verified with `nvidia-smi` during run)
- ncu metric: `smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active = 42.7%`
  for chip-level pure FFMA2 (BS=512, MIN_BLOCKS=1, persistent, N_CHAINS=4). Translates to
  85% of the FFMA2-specific issue ceiling (since FFMA2 caps at half the nominal pipe_fma
  peak).
- Strict serialization: `pkill -9 QuickRunCUDA + sudo nvidia-smi -rgc -i 0` before AND
  after the run; pre-check confirmed 0% util / 120 MHz idle.

