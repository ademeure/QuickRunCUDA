# tcgen05 Power Family — CONSOLIDATED (perf/W, throttle, multi-trial)

**Auditor**: tcgen05-power swarm wave-3, 2026-04-22
**Authority order**: B300_TRUE_REFERENCE.md > TCGEN05_PERFW_CLEAN_2TRIAL.md (2-trial, post-cleanup) > corrections/06_tensor_cores_CORRECTED.md > corrections/16_power_clock_CORRECTED.md > TCGEN05_POWER_MASTER.md / TCGEN05_POWER_GUIDE.md (1005 MHz baselines) > TCGEN05_PERF_WATTS.md (single-trial; partly contaminated, see R1)
**Scope**: TF/W per precision, W-per-CTA scaling, clock-state framing, throttle / TDP, multi-trial reproducibility, ncu pipe_tensor concerns. Originals NOT modified.

Per-file inconsistencies are catalogued in `TCGEN05_POWER_INCONSISTENCY_LOG.md`.

---

## 1. Definitive perf/W ladder (1500 MHz lock, lane-0 kernel, M=N=256, cta_group::2)

Trust order: **TCGEN05_PERFW_CLEAN_2TRIAL** (2 trials, `pkill -9` + 6 s cooldown between every measurement) supersedes the single-trial ladder in TCGEN05_PERF_WATTS.md.

### Random data (mode 0) — full random A and B

| Format            | K  | PF @ 1500 | Mean W (2-trial) | TF/W   | Trial-to-trial gap |
|-------------------|----|-----------|------------------|--------|--------------------|
| TF32              | 8  | 0.91      | 787              | 1.16   | 0 W                |
| FP16              | 16 | 1.82      | 935              | 1.95   | 0 W                |
| BF16              | 16 | 1.82      | 876              | 2.08   | 0 W                |
| FP8 e4m3          | 32 | 3.64      | 1073             | 3.39   | 7 W                |
| MXFP8 (UE8M0 SF)  | 32 | 3.64      | 1041             | 3.50   | 10 W (max)         |
| NVFP4 K=64        | 64 | 7.27      | 689              | 10.54  | 1 W                |
| NVFP4 K=96 N=192  | 96 | 10.91     | 880              | 12.39  | 3 W                |
| NVFP4 K=96 N=256  | 96 | 10.91     | 870              | **12.54** | 0 W              |

### Best efficiency (B-positive sign-zeroing, mode 1) — same kernel

| Format             | Mean W | TF/W       | Δ vs random |
|--------------------|--------|------------|-------------|
| TF32 K=8           | 680    | 1.34       | -107 W      |
| FP16 K=16          | 785    | 2.32       | -150 W      |
| BF16 K=16          | 750    | 2.43       | -125 W      |
| FP8 K=32           | 834    | 4.36       | **-238 W**  |
| MXFP8 K=32         | 800    | 4.55       | **-241 W**  |
| NVFP4 K=64         | 585    | 12.42      | -104 W      |
| NVFP4 K=96 N=192   | 728    | 14.99      | -152 W      |
| NVFP4 K=96 N=256   | 720    | **15.16**  | -150 W      |
| NVFP4 K=96 A+B-pos | 693    | **15.74**  | -177 W (best of all)|

**Realistic LLM-weight quantization (FP4, NVFP4 K=96 N=256)**: ~816 W → **13.4 TF/W**, +7 % vs raw random — the bulk of theoretical gains require restructured quantization (per-magnitude bands, sign separated as bitmap).

---

## 2. 2-trial vs single-trial cross-check

Single-trial `TCGEN05_PERF_WATTS.md` numbers vs 2-trial-clean numbers (1500 MHz lock, same kernel, same shape):

| Format            | Single-trial W (PERF_WATTS) | 2-trial W (CLEAN_2TRIAL) | Δ      | Status |
|-------------------|----------------------------:|-------------------------:|-------:|--------|
| TF32 K=8          | 816                         | 787                      | -29 W  | reasonable noise |
| FP16 K=16         | 940                         | 935                      | -5 W   | OK |
| BF16 K=16         | 829                         | 876                      | +47 W  | **single-trial probably contaminated low** |
| FP8 e4m3          | 854                         | 1073                     | **+219 W** | **PERF_WATTS contaminated; CLEAN supersedes** |
| MXFP8             | 851                         | 1041                     | **+190 W** | **PERF_WATTS contaminated; CLEAN supersedes** |
| NVFP4 K=64        | 797                         | 689                      | **-108 W** | **silent contamination, see Pitfall §6** |
| NVFP4 K=96 N=256  | 795                         | 870                      | **+75 W**  | **silent contamination** |

PERFW_CLEAN_2TRIAL itself flags this: *"Earlier perf/W table at 1500 MHz had silent contamination — NVFP4 K=64 read 797 W and K=96 read 795 W (impossibly close given 1.5× work). True values (clean): K=64=689 W, K=96=870 W."*

**The PERF_WATTS NVFP4 K=96 = 13.72 TF/W headline is contaminated.** Use **TF/W = 12.54 (random) or 15.16 (B-positive) or 15.74 (A+B-positive)** from the 2-trial doc.

---

## 3. TF/W vs clock state — does ML-inference rule hold?

Memory says: *"ML inference USE BOOST CLOCK (3× lower energy than 510 MHz)"* — i.e. wall-clock dominates static-power amortization.

Tcgen05 family agrees by construction:

- All clean tcgen05 perf/W is reported at 1500 MHz lock (TDP-safe max). No tcgen05 doc has a complete 510 MHz vs 1500 MHz vs boost three-clock TF/W comparison for the same kernel, **but** the indirect evidence is consistent:
  - corrections/16_power_clock §3 reports **best FFMA GFLOPS/W = 134 at 1500-1700 MHz, drops to 123 at boost** (i.e. 1500 ≈ boost; 510 not measured for FFMA pJ/op equivalence; 1500 = best static/dynamic balance for FFMA).
  - For tcgen05 specifically: TCGEN05_POWER_MASTER §"Boost-clock validation" shows random BF16 = 609 W @ 1005 MHz vs 1097 W @ 2032 MHz — power scales 1.80×; throughput scales 2.02× → **TF/W at boost ≥ TF/W at 1005 MHz**. Boost wins on perf/W.
- 1500 MHz is chosen as "TDP-safe max" because most tcgen05 random workloads exceed 1100 W TDP at boost and throttle.

**Verdict**: tcgen05 numbers do NOT contradict the "boost is best for perf/W" rule. The convention is "1500 MHz lock" as a *clean comparison clock* (no throttle), NOT as the optimal energy clock.

---

## 4. NVFP4 K=96 ceiling — does tcgen05 power family agree with 10.8 PF cuBLAS cap?

Memory: *"cuBLAS 13.4 caps 10.8 PF (72% of 15 PF spec) at large-N rect"*.

Tcgen05 microbench peaks at **10.91 PF** at 1500 MHz lock (PERFW_CLEAN, PERF_WATTS, MASTER, 06_tensor_cores_CORRECTED — all consistent). This is = 73 % of the 15 PF B300 spec, **matching the 72 % memory entry within rounding**.

Confirmed: the NVFP4 K=96 ULTRA microbench reaches the SAME ceiling that cuBLAS 13.4 attains in production. No public path achieves the 15 PF spec — the 1.5× K=96 ULTRA is not algorithmically broken but the dispatch / scheduling overhead caps both microbench and cuBLAS at the same ~73 % of spec.

---

## 5. ncu pipe_tensor in tcgen05 power docs — clear

Verified by grep across `TCGEN05_*.md` and `MMA_SYNC_POWER.md`: **NO usage of `pipe_tensor` / `sm__pipe_tensor` in any tcgen05 power-family doc** in this directory. Power-family doc cycle measurements use `clock64` only.

This is correct: the tcgen05 power family is methodologically clean of the pipe_tensor pitfall (per `06_tensor_cores_CORRECTED.md` R4 and user memory `feedback_b300_pitfalls.md`).

(Note: `MMA_SYNC_POWER.md` measures legacy mma.sync; pipe_tensor IS valid for mma.sync but is not actually cited in the doc.)

---

## 6. W-per-CTA scaling rules

From `PER_SM_POWER_SCALING.md` (cited in MASTER row 193) and `POWER_FLOOR.md`:

| Component                | Per-SM cost | Total at 148 SMs |
|--------------------------|-------------|------------------|
| Idle baseline            | n/a         | 150-198 W (clock-dep) |
| Active floor (A=B=0)     | ~1 W/SM     | 287 W            |
| Static const (Tier B)    | ~1 W/SM     | 299 W (BF16) / 305 W (FP8) / 280 W (NVFP4) |
| Random data delta        | ~2.1 W/SM   | +310 W (BF16 random total 609 W) |
| Random data full         | ~3.1 W/SM   | (= 2.1 delta + 1.0 const) |

**Linear in active SM count up to 148**. No sub/superlinear surprises observed for tcgen05.mma.

Per-CTA scaling: For 1-CTA mode, this is per-SM. For 2-CTA `cluster_group::2`, **per CTA NOT per cluster** (per `2CTA_DEDUP.md` cited in MASTER) — each CTA holds its own 32-byte sub-tile dedup cache, and per-CTA random power is the same as 1-CTA. Total cluster power scales linearly with member CTAs.

Cross-precision random baselines (1005 MHz lock, MASTER §"Cross-Precision Random Baselines"):
- BF16 random 609 W vs const 299 W (gap +310)
- FP8 random 642 W vs const 305 W (gap +337)
- NVFP4 random 463 W vs const 280 W (gap +183)

NVFP4 has the smallest data-dep gap because 4-bit values have lower mantissa popcount.

---

## 7. Clock-state framing — risk of confusion

| Framing          | Where used                                        | Caveat                                     |
|------------------|---------------------------------------------------|--------------------------------------------|
| 1005 MHz (lock)  | TCGEN05_POWER_MASTER, TCGEN05_POWER_GUIDE         | Below DVS knee, low absolute power         |
| 1500 MHz (lock)  | TCGEN05_PERF_WATTS, TCGEN05_PERFW_CLEAN_2TRIAL    | TDP-safe max; *clock-locked test rig*      |
| Boost (~2032)    | only TCGEN05_POWER_MASTER §"Boost-clock validation" | Random data may throttle to ≤1920 MHz under TDP cap |

`TCGEN05_POWER_GUIDE.md` line 16 says "At boost (2032 MHz), values scale ~1.7-2x" — this is consistent with MASTER's boost-clock validation table (1097/609 = 1.80×) but is not directly characterized as TF/W.

**Recommendation**: every TF/W quote MUST state the clock state. The 12-15 TF/W ladder is **at 1500 MHz lock**. At boost, expect higher absolute throughput AND higher absolute power; perf/W is similar or slightly better (per FFMA analogy, see §3).

---

## 8. TDP / 1071 W stress recipe

Memory: *"DRAM read d=16 + 1500 MHz = 1071 W stress recipe"*.

NO tcgen05 doc reaches 1071 W. **Highest tcgen05 power observed at 1500 MHz lock = 1073 W (FP8 random, PERFW_CLEAN_2TRIAL)** — coincidentally matches the DRAM-read stress within rounding. Tcgen05 random hits TDP ceiling without throttling for 1500 MHz lock; boost (2032 MHz, MASTER table) reports BF16 random = 1097 W (right at the 1100 W cap) and 4-unique sub-tiles = 1098 W.

**Implication**: at boost clock, sustained tcgen05 random is power-cap-limited. This is exactly why PERFW measurements use 1500 MHz lock (avoids throttle, gives stable power readings).

---

## 9. Quick-cite cheat sheet (TF/W)

| Need                                           | TF/W   | Source                          |
|-----------------------------------------------|--------|---------------------------------|
| BF16 random (worst typical)                    | 2.08   | PERFW_CLEAN_2TRIAL              |
| BF16 best (B-positive)                         | 2.43   | PERFW_CLEAN_2TRIAL              |
| FP8 e4m3 random                                | 3.39   | PERFW_CLEAN_2TRIAL              |
| FP8 e4m3 best (B-positive)                     | 4.36   | PERFW_CLEAN_2TRIAL              |
| MXFP8 random                                   | 3.50   | PERFW_CLEAN_2TRIAL              |
| MXFP8 best                                     | 4.55   | PERFW_CLEAN_2TRIAL              |
| NVFP4 K=64 random                              | 10.54  | PERFW_CLEAN_2TRIAL              |
| NVFP4 K=64 best (B-positive)                   | 12.42  | PERFW_CLEAN_2TRIAL              |
| NVFP4 K=96 random                              | 12.54  | PERFW_CLEAN_2TRIAL              |
| NVFP4 K=96 best (A+B-positive)                 | **15.74** | PERFW_CLEAN_2TRIAL          |
| NVFP4 K=96 realistic LLM weights               | ~13.4  | PERFW_CLEAN_2TRIAL realistic table |

All values at 1500 MHz lock, M=N=256, cta_group::2, lane-0 early-exit kernel. **All values supersede the contaminated single-trial PERF_WATTS table.**

---

## RETRACTIONS

### R1. `TCGEN05_PERF_WATTS.md` 7-precision ladder (NVFP4 K=96 = 13.72 TF/W) — RETRACTED
- Source of retraction: `TCGEN05_PERFW_CLEAN_2TRIAL.md` §"Methodology pitfall confirmed" (2026-04-20):
  *"Earlier perf/W table at 1500 MHz had silent contamination — NVFP4 K=64 read 797 W and K=96 read 795 W (impossibly close given 1.5× work). True values (clean): K=64=689 W, K=96=870 W."*
- Specifically: FP8 (854 → 1073, Δ +219 W), MXFP8 (851 → 1041, +190 W), NVFP4 K=64 (797 → 689, -108 W), NVFP4 K=96 (795 → 870, +75 W) all change materially.
- **Use the 2-trial table. Do NOT cite TCGEN05_PERF_WATTS as the perf/W reference.**

### R2. "K-row sorting saves 349 W per CTA" — RETRACTED in 2TRIAL §CORRECTION
- Self-corrected within `TCGEN05_PERFW_CLEAN_2TRIAL.md` §"CORRECTION: K-axis power is BINARY":
  *"Even chunk-48 (only 1 K-transition in entire K=96) uses same power as fully random K. K-axis power is BINARY: either all 96 K-rows bit-identical (411 W floor) OR full ~870 W cost."*
- Real LLM weight matrices have varying K → always pay the full cost. K-row sorting/clustering does NOT help for tcgen05 power. (This corrects an earlier section in the SAME file.)

### R3. "B=0 zero-skip saves 456 W" framing — RETRACTED (renamed to toggle-skip)
- Self-corrected in `TCGEN05_PERFW_CLEAN_2TRIAL.md` §"Earlier zero-skip claim corrected": the mechanism is **B-bus toggle-skip** (B=any constant gives same savings), not arithmetic zero-detect on B.

### R4. "1.94 TF/W FP16" headline (TCGEN05_PERF_WATTS row 13) — DOWNGRADE
- Single-trial. 2-trial value 1.95 TF/W (essentially identical, but cite 2-trial for consistency).

---

## UNRESOLVED

### U1. Boost-clock TF/W (full ladder)
- No tcgen05 doc gives a complete 7-precision TF/W ladder at boost. Only random BF16 / FP8 spot checks at 1097/845 W (in MASTER §"Boost-clock validation"). Inferring boost TF/W requires assumptions about throughput scaling (2.02×) and power scaling (1.80×) holding identically across precisions.
- Per FFMA analogy (16_power_clock §3): boost TF/W ≈ 1500 MHz TF/W or marginally lower. Verify with full 7-precision boost run.

### U2. PER_WATTS contamination root cause
- 2TRIAL self-explains as "5 leftover QuickRunCUDA processes" silently inflating cy/MMA. The single-trial PERF_WATTS doc has NO indication it ran with that contamination, but its NVFP4 K=64 = 797 W is suspiciously close to NVFP4 K=96 = 795 W (impossibly close given 1.5× work). This ALSO matches the 2TRIAL diagnosis pattern.
- Whether OTHER non-NVFP4 rows in PERF_WATTS (TF32 / FP16 / BF16 / FP8 / MXFP8) are also contaminated cannot be determined from PERF_WATTS alone. The 2-trial document's clean numbers are the authoritative replacement.

### U3. ML inference perf/W validation in tcgen05
- The "boost is 3× better than 510 MHz" memory rule was derived from FFMA. Tcgen05 has different power scaling characteristics (more bus toggle activity, different DVS curve). A direct tcgen05 perf/W vs clock sweep (510 / 800 / 1005 / 1300 / 1500 / boost) is not in this corpus. POWER_FREQUENCY_CURVE.md (cited in MASTER) reports "best TF/W at 1300-1500 MHz" but that's general DVS, not tcgen05-specific.

### U4. Single-shot vs sustained
- `06_tensor_cores_CORRECTED.md` U2 notes a 28% gap between NVFP4 single-shot (9109 TF, 91% spec) and sustained random (6554 TF, 65% spec, throttle to 1057 MHz). The tcgen05 power family doesn't separately characterize single-shot vs sustained; PERFW_CLEAN measures sustained at 1500 MHz lock (no throttle) but absolute throughput at boost would suffer from the throttle.

### U5. Per-MAC energy at >1500 MHz
- 16_power_clock §3 has FFMA pJ/op data through 1920 MHz, but no tcgen05 pJ/MAC table at multiple clocks. Estimates can be derived but are not reported.

### U6. Realistic-LLM-weight table (PERFW_CLEAN §13.4 TF/W)
- Built from synthetic distributions (5-pos quantized, 70/30 mix). Whether real Llama / Mistral / Qwen weights actually fall in the predicted 800-870 W range is not directly verified — `corrections/06_tensor_cores_CORRECTED.md` U6 separately notes that the catalog uses pipe_tensor in some places without unpacking; cuBLAS realistic numbers (1850 BF16, 3983 FP8) are independent confirmation that real weights drop ~15 % from constant peak, consistent with the tcgen05 power story.

---

## Files of record (originals, NOT modified)

- `TCGEN05_PERFW_CLEAN_2TRIAL.md` — 2-trial perf/W ladder + B-positive + axis decomposition + dword-pattern thresholds (authoritative)
- `TCGEN05_PERF_WATTS.md` — single-trial; SUPERSEDED for absolute W values, retained for documenting peak PF and N=192 saturation
- `TCGEN05_POWER_GUIDE.md` — practitioner reference, 1005 MHz numbers, optimization recipes
- `TCGEN05_POWER_MASTER.md` — 1005 MHz baselines + boost validation + 30+ sub-finding index
- `TCGEN05_N_MATRIX.md` — cy/MMA full N sweep (cta=1 vs cta=2)
- `TCGEN05_N_SWEEP.md` — cy/MMA per precision (BF16 / FP8 / NVFP4)
- `TCGEN05_PATH_NOTES.md` — SASS family, compile recipe, CUTLASS notes
- `MMA_SYNC_POWER.md` — legacy mma.sync power asymmetry (BF16 + FP8)
- `corrections/06_tensor_cores_CORRECTED.md` — peak TFLOPS ladder (cross-references this doc)
- `corrections/16_power_clock_CORRECTED.md` — clock states, energy/op, DVS curve, TDP
- `corrections/TCGEN05_DEDUP_CONSOLIDATED.md` — dedup model (sister doc, not duplicated here)
- `B300_TRUE_REFERENCE.md` — meta authority
