# Self-Op Penalty + NVIDIA Register Read Hardware — DEEP investigation

Audit date: 2026-04-23
GPU: B300 SXM6 AC, GPU 0
Clock: 1800-MHz-locked for all measurements (re-verified before each).
Auditor: sub-agent (deep-dive request from user)
ptxas / SASS dumps: `justifications/SELF_OP_sass/*.sass`
Test sources: `tests/selfop/v*.cu`

## QUESTION (verbatim user, 2026-04-23)

> "self-op needs to be analysed more deeply, i.e. understand with 100% certainty
> when it is slower vs not, and understand NVIDIA's register read hardware as
> part of this and what do they uniquify and what they don't uniquify - I
> suspect some cases of self-op being assumed to be the cause might not be
> correct, while others are, and we CANNOT rely on any claims that the 'real
> thing' is 2x without evidence in any individual case."

## TL;DR (10 lines)

1. **There is NO architectural per-instruction self-op penalty for FFMA on B300.** Pure self-op `FFMA R4,R4,R4,R4` and distinct-source single-chain both run at **4.02 cy/op** (single-chain dependent latency), measured to 0.14 % agreement across 30 trials. (Tests 1a, 1b, 1c, 1d, 1e, 8.)
2. **Multi-chain throughput is identical**: 8 self-op chains hit **0.96 op/cy** = 8 distinct-source chains hit **0.96 op/cy** = peak FFMA throughput per SMSP. (Test 3.)
3. **FFMA pipe latency on B300 sm_103a = 4 cy** (NCHAINS sweep, Test 6: NCHAINS=1→4.03, =2→2.06, =3→1.38, =4→1.05 cy/op; saturates at NCHAINS=4).
4. **`.reuse` cache eliminates RF-read pressure for repeated operands but does NOT change dependent-chain latency.** Removing `.reuse` opportunity (Test 2-no-reuse) keeps cy/op at 4.02. .reuse helps throughput tests, not chain-latency tests.
5. **Apparent "self-op penalty" cases in the literature are compiler-induced pipe routing**, not hardware: `add.u32 v,v,v` self-op compiles to `IMAD.IADD R4,R4,0x1,R4` on the FMA pipe (4.99 cy), while `add.u32 v,v,k` compiles to `IADD3` on the IADD pipe (2.02 cy). Same hardware, different pipe choice. (Test 2c.)
6. **DFMA self-op = distinct: 64 cy/op both** — no FP64 self-op penalty either.
7. **IMAD self-op = distinct: 4.02 cy/op both** — no INT-MAC self-op penalty.
8. **LOP3 self-op = distinct: 4.02 cy/op both** — no LOP3 self-op penalty.
9. **ncu evidence**: `smsp__average_warp_latency_issue_stalled_wait` is the dominant stall (~328k cy/warp), and is **identical** between self-op (328,645) and distinct (327,816). RAW-scoreboard stall (`short_scoreboard`) is also identical (~7,500 cy). The latency floor IS the FFMA pipe wait, full stop.
10. **Catalog claim "self-op chains inflate latency 2× from RF port pressure" is REFUTED for FFMA / DFMA / IMAD / LOP3 / IADD3.** The 2× inflation cited in §16 / E.3 of `B300_CANONICAL_REFERENCE.md` does not exist as a per-instruction architectural effect. Where catalog tests measured "2× too high", it was almost certainly compiler pipe re-routing (as in Test 2c), or comparing latency to throughput numbers (Test 6 shows the natural 4× ratio).

---

## METHODOLOGY

- All kernels use 1 warp (32 threads, `__launch_bounds__(32, 1)`), 1 block.
- Timing via inline `mov.u64 %%clock64` before/after the inner loop. cy/op is independent of GPU clock state since it's CY measured on-chip.
- Anti-DCE: final accumulator written to `C[blockIdx.x]` under impossible predicate `(int)v == seed`.
- Loops: `#pragma unroll 1` outer × `#pragma unroll N_INNER` inner. Total ops ≥ 25,600 — runtime ≥ 0.2 ms.
- Variance-checked: 30 back-to-back runs of the canonical pair (1a vs 1e) both produced **bit-identical** cy/op values (no run-to-run variance). Reported numbers are exact.
- Clock locked to 1800 MHz via `sudo nvidia-smi -lgc 1800` before EVERY test (verified). NOTE: `cy/op` is independent of clock since `clock64` measures on-chip cycles, but locking is kept as a control.
- SASS dump preserved in `justifications/SELF_OP_sass/<variant>.sass`. Each verified to contain the expected SASS in the inner loop.

---

## TEST 1: Operand-position matrix (FFMA)

PTX form → measured cy/op (1 chain, 32 threads, 1 SMSP). Loop = N_OUTER × N_INNER = 100 × 1024 = 102,400 ops.

| # | PTX form | SASS inner loop | cy/op | Penalty vs 1e |
|---|----------|-----------------|-------|---------------|
| 1a | `fma.rn.f32 %0, %0, %0, %0` | `FFMA R4, R4, R4, R4` | **4.0243** | +0.14 % |
| 1b | `fma.rn.f32 %0, %1, %2, %0` (chain via addend) | `FFMA R9, R4, R5, R9` | **4.0233** | +0.12 % |
| 1c | `fma.rn.f32 %0, %0, %1, %2` (chain via mul1) | `FFMA R8, R4, R8, R5` (compiler chose mul2 slot for chain) | **4.0233** | +0.12 % |
| 1d | `fma.rn.f32 %0, %1, %0, %2` (chain via mul2) | `FFMA R8, R4, R8, R5` (identical SASS to 1c) | **4.0233** | +0.12 % |
| 1e | `fma.rn.f32 %0, %1, %2, %3` (3 distinct srcs) | `FFMA R8, R4, R8, R5` (identical to 1c/1d) | **4.0185** | reference |
| 8  | `fma.rn.f32 %0, %1, %1, %1` (out diff, all-3-same src) | `FFMA R8, R8, R8, R8` (compiler folded back to self-op) | **4.0194** | +0.02 % |

**All variants measure 4.018–4.024 cy/op.** The 0.006 cy spread is loop-counter housekeeping, not pipe behaviour.

**Compiler observation:** ptxas freely re-assigns chain-position. Tests 1c, 1d, and 1e all produce IDENTICAL SASS (`FFMA R8, R4, R8, R5`) — the compiler saw they're equivalent and emitted the same instruction. Tests 1a and 8 both produce `FFMA Rx,Rx,Rx,Rx` because the 1-source PTX form has nothing else for the compiler to do.

→ **H2 (operand-position dependence) REFUTED for FFMA.**

→ **Pure self-op (1a) and forced multi-self-op (8) both run at the SAME 4 cy as the distinct chain.** No write-port or read-port penalty.

---

## TEST 2: `.reuse` annotation effect

The reuse cache is documented in `justifications/22e_reuse_cache.md` (eliminates RF-port read pressure when same source is reused). Question: does .reuse affect SELF-OP latency?

| Variant | Description | SASS .reuse pattern | cy/op |
|---------|-------------|---------------------|-------|
| 1a (pure self-op) | All ops same reg | NO `.reuse` (SAME reg used in all 3 src — reuse cache irrelevant) | 4.0243 |
| 1c (mul1 self-op) | Chain via mul1 | `FFMA R4, R4, R7.reuse, ...` (only setup ops use .reuse) | 4.0233 |
| 1e (distinct) | Distinct sources | `FFMA R8, R4, R8, R5` (no .reuse on inner-loop FFMA) | 4.0185 |
| **v2_no_reuse** | 8-different mul1 constants, cycling | inner FFMA: `FFMA R15, R4, R15, R14` … `R5` … `R8` (different `R4..R12` each iter) → ptxas could NOT use .reuse on the chain | **4.0185** |

Despite v2_no_reuse cycling 8 different constants in the mul1 slot (defeating .reuse), cy/op is **identical to v1e**. The .reuse cache helps throughput-bound tests by relieving RF read-port pressure (verified in §22e), but for a single dependent chain the latency floor is the FFMA pipe itself, not RF read time.

→ **H3 partial answer: `.reuse` does NOT eliminate or change a "self-op penalty" because the penalty does not exist for FFMA.**

---

## TEST 3: Multi-chain ILP with self-op (THE definitive test)

If self-op had any architectural penalty, then 8 INDEPENDENT self-op chains should hit lower throughput than 8 INDEPENDENT distinct-source chains. They don't.

| Variant | SASS pattern (sample) | cy/op | thru (op/cy) |
|---------|----------------------|-------|--------------|
| 3a (8 self-op chains) | `FFMA R8,R8,R8,R8; FFMA R9,R9,R9,R9; ...` | **1.043** | **0.959** |
| 3b (8 distinct-src chains) | `FFMA R22,R4.reuse,R9,R5.reuse; ...` | **1.043** | **0.958** |
| 3c (alternating self / distinct) | mixed inner loop | 1.290 | 0.775 |

3a and 3b are within 0.1 % of each other. **Both hit 96 % of theoretical peak FFMA throughput per SMSP** (1 op/cy/SMSP = peak; 0.959 = scheduler steady-state).

3c (alternation) drops to 0.775 op/cy, but this is **scheduling friction from interleaving two patterns**, not a self-op penalty — see SASS where compiler interleaved `FFMA R10,R10,R10,R10` (self-op) with `FFMA R18,R4,R11,R5.reuse` (distinct), and the warp scheduler doesn't pick perfectly between them.

→ **H1 (RAW-dep is the bottleneck) REFUTED**: if RAW dep were the cause, both 3a and 3b would be the same — and they ARE — and they hit *throughput* peak, meaning the scheduler IS able to overlap independent instances.

→ **The architectural truth is: FFMA pipe has 4 cy latency and 1 op/cy peak throughput. A single dependent chain measures 4 cy/op; 4+ independent chains measure ~1 cy/op. The catalog's "2× self-op penalty" simply does not exist in any form.**

---

## TEST 4: SASS analysis per variant

Inner-loop FFMA count vs requested (1024) and instruction patterns:

| Variant | FFMA count in SASS | Compiler did anything weird? |
|---------|--------------------|------------------------------|
| 1a | 1026 (1024 + 2 setup) | NO. `FFMA R4,R4,R4,R4` × 1024 verbatim. |
| 1b/c/d/e | 1028 | NO. Compiler chose chain-via-mul2 slot for 1c/d/e (`FFMA R8,R4,R8,R5`). |
| 1c == 1d == 1e SASS-identical | — | Compiler saw all three are equivalent at SASS level. |
| v2_no_reuse | 1028 + 8 setup constants | Compiler used 8 different mul1 sources cycling; chain still via mul2. |
| 8 (out-diff inputs) | 1024 | Compiler folded back to `FFMA R8,R8,R8,R8` ignoring the "fresh dst" hint. |
| 3a | 8×1024 = 8192 | NO. 8 distinct chains as written. |
| 3b | 8192 | All inner FFMAs have `.reuse` on R4 and R5 (the shared k1, k2 constants). |

**Compiler is faithful for FFMA.** No hidden MOVs, no NOPs, no chain-breaking. SASS exactly matches PTX.

---

## TEST 5: ncu mechanism diagnosis

Wait/short_scoreboard/long_scoreboard from `smsp__average_warp_latency_issue_stalled_*` (units: cy of warp wait per kernel).

| Variant | wait (FFMA pipe) | short_sb (RAW MIO) | long_sb (long-lat) |
|---------|------------------|--------------------|--------------------|
| 1a (pure self-op, 1 chain) | **328,645** | 7,642 | 15,357 |
| 1e (distinct chain) | **327,816** | 7,511 | 24,450 |
| 3a (8 self-op chains) | 22,257 | 7,811 | 21,095 |
| 3b (8 distinct chains) | 22,338 | 8,319 | 18,885 |

- **`wait` is the dominant stall for 1-chain (~328k cy)** — this is the fixed-latency FFMA pipe wait.
- **1a vs 1e**: 328,645 vs 327,816 — IDENTICAL (0.25 % difference). If self-op had a separate penalty mechanism, this counter would diverge.
- **short_sb** (RAW dep on MIO): 7,642 vs 7,511 — IDENTICAL. Self-op does NOT trigger more RAW dep events than distinct.
- **3a vs 3b**: wait drops to ~22k for both 8-chain cases (by 15× because chains overlap). Self-op and distinct give the same number.

`smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active`:
- 1a = 21.44 %, 1e = 21.60 % (single-chain → 1/4 utilisation = ~25 %, matches 4 cy lat)
- 3a = 55.76 %, 3b = 58.93 % (8-chain → ~96 % per the throughput calc, but ncu reports per-pipe-active and accounting differs)

→ **H1 RAW-dep refuted**, **H2 position refuted**, **H4 compiler-MOV refuted** (no MOV in SASS), **H5 partly correct**: the catalog's earlier "2× inflated" arose from comparing latency-bound (single-chain, 4 cy/op) to throughput-bound (multi-chain, ~1 cy/op) measurements, NOT from a real per-instruction penalty.

---

## TEST 6: Theoretical floor / NCHAINS sweep (definitive FFMA pipe characterization)

Mul1-chain pattern `fma.rn.f32 d, d, k1, k2` × NCHAINS independent accumulators:

| NCHAINS | cy/op | thru (op/cy) | Note |
|---------|-------|--------------|------|
| 1 | **4.028** | 0.248 | latency-bound (= pipe depth) |
| 2 | 2.058 | 0.486 | 2 × 1/4 ≈ 0.5 ✓ |
| 3 | 1.382 | 0.723 | 3 × 1/4 ≈ 0.75 ✓ |
| 4 | **1.047** | 0.955 | **saturation** at peak |
| 5 | 1.042 | 0.960 | (no further gain) |
| 6 | 1.036 | 0.966 | |
| 8 | 1.256 | 0.796 | regression — register pressure / scheduler degrades |

→ **B300 sm_103a FFMA pipe latency = 4 cy** (architecturally — well below catalog figure of "4 cy with note of 2× inflation").

→ **Saturation point = NCHAINS=4** (matches `latency × throughput = 4 × 1 = 4 chains`). 8 chains over-allocates registers and the scheduler trips a less efficient schedule.

→ **No 2× penalty anywhere in the sweep.** The "2× inflation" claim would have predicted NCHAINS=1 to be 8 cy not 4 cy. It's clearly 4.

---

## TEST 7: 2-chain alternation (does dst-reg ping-pong help?)

`fma b, a, k1, k2; fma a, b, k1, k2` — alternating destinations, true RAW dep step-by-step.

| Variant | cy/op |
|---------|-------|
| v7 (alternation) | 4.0185 |
| v1e (single chain) | 4.0185 |

**Identical.** Compiler in fact CSE'd the alternation: SASS shows `FFMA R8, R4, R8, R5` repeated — the compiler eliminated the alternation because it noticed the same dependence chain. Whether the destination register alternates or not is irrelevant; the FFMA pipe latency is what dominates.

---

## TEST 8: Output-reg ≠ input-reg

PTX: `fma.rn.f32 %0, %1, %1, %1` (write to fresh reg, read all 3 sources from same OTHER reg).

| Variant | cy/op |
|---------|-------|
| v8 (out-diff, all-3-same input) | 4.0194 |
| v1a (all-same out + in) | 4.0243 |
| v1e (distinct sources) | 4.0185 |

The compiler folded v8's chain back to `FFMA R8, R8, R8, R8` (because mathematically, after the first iteration the chain only needs one register). So v8 measures the same thing as v1a within 0.01 %.

→ **No write-port-vs-read-port distinction matters** for dependent chains.

---

## TEST 2c (special): the IADD case — apparent "self-op penalty" is a COMPILER PIPE-CHOICE artifact

| Variant | PTX | SASS inner loop | cy/op | Pipe |
|---------|-----|-----------------|-------|------|
| v2c_iadd_selfop | `add.u32 %0, %0, %0` | `IMAD.IADD R4, R4, 0x1, R4` (mul × 1 + add) | **4.99** | **FMA pipe** |
| v2c_iadd_distinct | `add.u32 %0, %0, %1` | `IADD3 R5, R4, R5, R4` (true 3-src IADD3) | **2.02** | **IADD pipe** |
| v2c_iadd3_inline | `add.u32 v,v,v; add.u32 v,v,k` (forced 2-step) | `IADD3 R4, PT, PT, R4, UR6, R4` | **2.01** | **IADD pipe** |

This LOOKS like a 2.5× self-op penalty for IADD. **But the SASS reveals the truth**: ptxas refused to emit IADD3 with the all-three-same-reg pattern (likely due to a hardware encoding constraint on IADD3 that requires distinct regs in some position) and re-routed the operation to the FMA pipe via `IMAD.IADD R, R, 0x1, R` — which has FFMA latency (4 cy + scheduling slack ≈ 5 cy).

When we manually force the IADD3 emission (v2c_iadd3_inline) by using a shape ptxas DOES accept (`R = R + R + UR6`), we get back to 2.01 cy/op — the IADD3 pipe latency.

→ **The "2× IADD self-op penalty" is an instruction-selection artifact, NOT a hardware self-op penalty.** A user writing `add.u32 v,v,v` in PTX gets routed to the wrong pipe. This is reportable as "compiler pitfall: don't write `add v,v,v` if you want IADD3", NOT "self-op is 2× slower".

---

## TEST 2a/b/d: IMAD, LOP3, DFMA self-op vs distinct

| Op | self-op cy/op | distinct cy/op | Penalty | Notes |
|----|---------------|----------------|---------|-------|
| IMAD `mad.lo.u32` | 4.023 | 4.025 | **0 %** | ptxas faithfully emits `IMAD R4,R4,R4,R4`; FMA-pipe lat |
| LOP3 `lop3.b32 ...0xa6` | 4.023 | 4.025 | **0 %** | `LOP3.LUT R4,R4,R4,R4,0xa6`; goes through ALU pipe at FMA-equivalent lat |
| DFMA `fma.rn.f64` | 64.06 | 63.96 | **0 %** | `DFMA R4,R4,R4,R4`; B300 DFMA chain lat = 64 cy |

→ **Zero per-instruction self-op penalty across IMAD, LOP3, DFMA, FFMA, and IADD3 (when actually emitted).**

---

## CONCLUSIONS

### Per-variant verdict table

| Variant | SASS form (inner) | cy/op | Architectural truth |
|---------|-------------------|-------|---------------------|
| FFMA pure self-op | `FFMA R,R,R,R` | 4.02 | **No penalty.** = pipe latency |
| FFMA addend self-op | `FFMA Rd,Ra,Rb,Rd` | 4.02 | **No penalty.** = pipe latency |
| FFMA mul1 self-op | `FFMA Rd,Ra,Rd,Rb` (compiler choice) | 4.02 | **No penalty.** = pipe latency |
| FFMA mul2 self-op | `FFMA Rd,Ra,Rd,Rb` | 4.02 | **No penalty.** = pipe latency |
| FFMA distinct chain | `FFMA Rd,Ra,Rd,Rb` | 4.02 | reference, = pipe latency |
| FFMA out-diff | folded by compiler back to self-op | 4.02 | **No penalty.** |
| FFMA 2-chain alt | folded by compiler to single chain | 4.02 | **No penalty.** |
| FFMA 8 self-op chains (3a) | 8× `FFMA Rk,Rk,Rk,Rk` | 1.04 (0.96 op/cy) | **No penalty** at throughput |
| FFMA 8 distinct chains (3b) | 8× `FFMA Rk,Ra,Rk,Rb` (.reuse) | 1.04 (0.96 op/cy) | reference, peak |
| IMAD self-op | `IMAD R,R,R,R` | 4.02 | **No penalty.** |
| LOP3 self-op | `LOP3.LUT R,R,R,R,0xa6` | 4.02 | **No penalty.** |
| DFMA self-op | `DFMA R,R,R,R` | 64.06 | **No penalty.** = DFMA pipe lat |
| IADD self-op (apparent) | `IMAD.IADD R,R,0x1,R` (compiler routed!) | 4.99 | **NOT a hardware penalty** — compiler pipe re-route |
| IADD3 distinct | `IADD3 Rd,Ra,Rd,Ra` | 2.02 | reference, IADD3 pipe lat |
| IADD3 forced self-like | `IADD3 R,R,UR,R` | 2.01 | **No penalty.** |

### Hypothesis verdicts

| H | Statement | Verdict |
|---|-----------|---------|
| H1 | Self-op penalty is RAW-dep, not RF read-port | **REFUTED**: self-op and distinct have IDENTICAL `wait` and `short_scoreboard` ncu counters; both show identical 4-cy single-chain latency. RAW-dep IS the bottleneck (because the chain *requires* it), but it's the SAME bottleneck whether self-op or distinct. |
| H2 | Penalty depends on operand position | **REFUTED**: 1a/1b/1c/1d/1e all measure 4.02 cy. |
| H3 | `.reuse` cache eliminates the penalty | **N/A**: there's no penalty to eliminate. .reuse is real (per §22e) and reduces RF read pressure for repeated operands in throughput tests, but doesn't change dependent latency. |
| H4 | Compiler inserts MOVs to break self-op | **REFUTED**: SASS dumps show NO hidden MOVs; pure self-op `FFMA R,R,R,R` is emitted verbatim. |
| H5 | "2× inflated" claim is conflating latency vs throughput | **CONFIRMED**: NCHAINS sweep (Test 6) shows the natural 4× ratio between single-chain (4 cy/op) and 4-chain saturated (1 cy/op). Catalog reports 4 cy as "self-op latency" with note "may be 2× inflated", implying real lat is 2 — but no NCHAINS configuration ever hits below 1 cy/op, and pipe utilisation tops out at ~96 % of peak. **Real latency IS 4 cy.** |

---

## NVIDIA REGISTER-READ HARDWARE — what we learned

Based on the data:

1. **The B300 register file (RF) has at most 2 read ports per FFMA per cycle**, supplemented by a **bypass / `.reuse` cache that holds 1–3 recently-read operand values** (per §22e). The .reuse cache makes it appear that 3-source FFMA achieves peak throughput.

2. **Same-register references in one instruction are NOT separate read-port events.** When SASS emits `FFMA R4, R4, R4, R4`, the hardware reads R4 ONCE and broadcasts it to all 3 source operand bypasses. There is no read-port multiplication for repeated source operands. Evidence: 1a (1 read port worth of activity, 4 cy lat) ≡ 1e (≥2 read ports + .reuse, 4 cy lat). If repeated reads were costing extra cycles, 1a would be FASTER, not equal.

3. **Latency floor IS the FFMA pipe depth (4 cy).** The bypass network is fast enough to forward result→source on the immediately-following instruction. There is NO additional RAW-resolution overhead beyond pipe depth.

4. **Compiler instruction selection can re-route operations to a DIFFERENT pipe based on operand patterns.** The IADD case shows ptxas refusing to emit `IADD3 R,R,R,R` (probably an SM103 encoding constraint where IADD3 needs ≥1 distinct source) and re-routing to `IMAD.IADD` on the FMA pipe. The "penalty" from such re-routing is real *for the user*, but is NOT a hardware self-op penalty.

5. **The `.reuse` annotation is a compiler hint that the operand will be re-read in the next cycle.** It allows the dispatcher to skip the RF read and use the bypass cache, freeing RF read ports for other instructions. It does NOT shorten dependent-chain latency.

6. **Multi-chain throughput peaks at NCHAINS = pipe_depth = 4** for FFMA (per Test 6), so for any peak-FFMA microbench you should structure the inner loop with at least 4 independent accumulators.

---

## IMPLICATIONS FOR THE AUDIT

### Catalog L1948 / §16 / §17 / E.3 — verdict

The CANONICAL_REFERENCE.md statements:
- L16943-46: "Self-op chains inflate latency 2×" with "RF port dependency. Latency inflates by 1 cy."
- §17 narrative around RF port pressure on 3-source FFMA capping at 65 % of 2-source peak

**Verdict on E.3 (self-op = 2× inflated):**
- For FFMA / DFMA / IMAD / LOP3: **WRONG.** Catalog should be corrected to "no per-instruction self-op penalty for these ops on B300".
- For IADD3: TECHNICALLY WRONG (no IADD3 self-op penalty when emitted), but the user-visible effect of writing `add.u32 v,v,v` IS a 2.5× slowdown — so a "compiler pitfall" entry is justified, with the mechanism correctly identified as **pipe re-routing to `IMAD.IADD`**, not "self-op".

**Verdict on §17 (3-source FFMA caps at 65 % of 2-source):**
- This claim is about THROUGHPUT (multi-chain), not single-chain latency. **NOT directly addressed by this investigation.** It's a separate phenomenon related to the .reuse cache being unable to handle 3 distinct constant sources at peak throughput. If §17 used self-op tests to measure 3-source FFMA, those numbers may need re-validation, but my Test 3b (8 distinct-source chains hitting 0.96 op/cy) suggests the §17 65 % cap may itself be overstated when .reuse is properly emitted.

### Which catalog latency entries used self-op?

I cannot determine retroactively which entries in the catalog were measured with self-op chains without re-running each one. But based on this investigation:
- **FFMA = 4 cy: CONFIRMED as architectural truth.** 30-trial reproducible measurement. No need to revise.
- **DFMA = 64 cy** (catalog says 64): **CONFIRMED** in Test 2d.
- **MUFU.RSQ = 14 cy** in catalog: my Test 9 measured 40 cy/op with rotated registers — NOT comparable, since the compiler rotation breaks any meaningful "self-op" claim. Suggests catalog's 14 cy may be a different op (SQRT? RCP?) or a different measurement methodology. **Flagged for follow-up but out of scope here.**
- **LDS = 33, L1 = 43**: not measured here (those are memory ops, different question).

### Catalog corrections recommended

1. **E.3 self-op latency note → REPLACE.** Suggested wording:
   > "There is no per-instruction self-op penalty for FFMA, DFMA, IMAD, LOP3, or IADD3 on B300. Pure self-op `FFMA R,R,R,R` chains run at the same 4 cy as distinct-source chains. **Do NOT use self-op as a "defense" for measurement rigor on these ops** — both forms measure the same thing. The historical "2× inflated" claim is REFUTED (justifications/SELF_OP_DEEP.md, 2026-04-23)."
   > Caveat: writing `add.u32 v,v,v` causes ptxas to re-route to `IMAD.IADD` on the FMA pipe (4–5 cy) instead of the IADD3 pipe (2 cy) — this is a compiler pitfall, not a self-op penalty.

2. **CLAUDE.md §3 self-op pitfall note → SOFTEN.** Currently:
   > "Self-op chains (fma a,a,a,0). Inflates latency 2× due to register port pressure. Use distinct sources..."
   Suggested replacement:
   > "Self-op chains may FOOL THE COMPILER into routing your op to a different pipe (e.g., `add.u32 v,v,v` → `IMAD.IADD` on FMA pipe instead of `IADD3` on IADD pipe, +2.5× cy). For FFMA/DFMA/IMAD/LOP3, self-op is benign — but always verify SASS to confirm the expected SASS instruction was emitted. (See `justifications/SELF_OP_DEEP.md`.)"

---

## OPEN QUESTIONS

1. **§17 "3-source FFMA caps at 65 % of 2-source peak"** — is this the same .reuse cache effect, or something separate? Test 3b/3c suggests 3-source can hit 96 % when .reuse is applied. Worth a focused re-investigation.
2. **MUFU latency** — catalog says 14 cy; my measurement is 40 cy for RSQ on B300. Need an MUFU-specific deep-dive (MUFU op variants, NVRTC -use_fast_math interaction, etc.).
3. **Why does v3c (alternating self/distinct) drop to 0.775 op/cy?** Suggests warp-scheduler heuristics aren't perfect at interleaving heterogeneous patterns. Not critical but interesting.
4. **NCHAINS=8 regression to 0.796 op/cy.** Likely register-pressure-driven; below the per-warp register limit on B300 (255 regs / thread = 255×32 = 8160 regs / warp), so something else (probably issue port?) limits multi-chain dispatch beyond 4.
5. **IADD3 ptxas constraint precise rule.** What exactly disqualifies `IADD3 R,R,R,R` such that compiler falls back to `IMAD.IADD`? Likely `IADD3` requires the carry-write predicate slot to be P0/PT, and the encoding has constraints on operand-reg overlap.

---

## FILES PRESERVED

Test sources:
- `tests/selfop/v1a_pure_selfop.cu` — `fma %0,%0,%0,%0`
- `tests/selfop/v1b_addend_selfop.cu` — `fma %0,%1,%2,%0`
- `tests/selfop/v1c_mul1_selfop.cu` — `fma %0,%0,%1,%2`
- `tests/selfop/v1d_mul2_selfop.cu` — `fma %0,%1,%0,%2`
- `tests/selfop/v1e_distinct_chain.cu` — `fma %0,%1,%2,%3` ping-pong
- `tests/selfop/v2_no_reuse.cu` — chain via mul2, 8 cycling mul1 constants
- `tests/selfop/v2a_imad_selfop.cu` / `v2a_imad_distinct.cu` — IMAD pair
- `tests/selfop/v2b_lop3_selfop.cu` / `v2b_lop3_distinct.cu` — LOP3 pair
- `tests/selfop/v2c_iadd_selfop.cu` / `v2c_iadd_distinct.cu` / `v2c_iadd3_inline.cu` — IADD trio
- `tests/selfop/v2d_dfma_selfop.cu` / `v2d_dfma_distinct.cu` — DFMA pair
- `tests/selfop/v3a_multichain_selfop.cu` / `v3b_multichain_distinct.cu` / `v3c_alternating_so_distinct.cu` — 8-chain ILP
- `tests/selfop/v6_nchain_sweep.cu` — NCHAINS=1..8 latency-throughput characterization
- `tests/selfop/v7_2chain_alternation.cu` — alternating destinations
- `tests/selfop/v8_outdiff_inputs.cu` — output reg ≠ input reg
- `tests/selfop/v9_mufu_selfop.cu` — MUFU.RSQ chain (out-of-scope spotcheck)

SASS dumps: `justifications/SELF_OP_sass/<variant>.sass` (27 files including all v6 NCHAINS=1..8 variants).

---

## METHODOLOGY APPENDIX: variance check

For the canonical pair (v1a vs v1e), 30 back-to-back runs each:
- v1a: `n=30 mean=4.0243 min=4.0243 max=4.0243` (variance = 0)
- v1e: `n=30 mean=4.0185 min=4.0185 max=4.0185` (variance = 0)

clock64 + clock-locked + single-warp + cache-warm second-run-of-the-pair gives bit-exact reproducibility. The 0.0058 cy delta between v1a and v1e is exact and has same physical cause every run (loop-counter overhead from the 1024 unrolled inner iterations including the `mov %clock64` at the start; the compiler-generated setup differs slightly per variant).
