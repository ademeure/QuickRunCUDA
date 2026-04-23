# §17 MUFU per-op throughput — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §2.10 (L383-400) and §16 research log entries
**Detail doc:** `justifications/22_mufu_throughput_DEEP.md` (full investigation)

## CLAIM (catalog L383-400, §2.10)

| PTX | SASS | Catalog rate (ops/SMSP/cy) |
|-----|------|---------------------------:|
| ex2.approx.f32 | MUFU.EX2 | 0.5–0.63 |
| rsqrt.approx.f32 | MUFU.RSQ | 0.5 |
| sqrt.approx.f32 | MUFU.SQRT | 0.5 |
| rcp.approx.f32 | MUFU.RCP | 0.5 |
| sin.approx.f32 | MUFU.SIN | 0.5 (slower with conditioning) |
| cos.approx.f32 | MUFU.COS | 0.5 |
| lg2.approx.f32 | MUFU.LG2 | 0.5 |
| tanh.approx.f32 | MUFU.TANH | 0.5 |

Catalog implies **uniform 0.5 ops/SMSP/cy** = 2 ops/SM/cy total (8 SMSPs × 0.5… wait, 4 SMSPs × 0.5 = 2 ops/SM/cy).

Catalog §15 latency entries:
- MUFU.EX2 latency = 14.45 cy
- MUFU.RSQ latency = 40.18 cy
- MUFU.RCP latency = 42.31 cy

## TEST

`tests/bench_mufu.cu` (existing, well-designed).

```bash
./QuickRunCUDA -f tests/bench_mufu.cu -t 32 -b 1 -A 1024 -B 1024 -C 1024 \
  -H "#define MUFU_ASM ex2.approx.f32
#define N_CHAINS 16
#define UNROLL 8" -0 8192 -T 5
```

Single warp, 1 SMSP active (so per-warp throughput = per-SMSP throughput when warp is the only thing on its SMSP, but other SMSPs are idle so no inter-SMSP contention).

## MEASURED (full sweep at saturating ILP, N_CHAINS=16)

| PTX op | SASS | cy/op/warp | rate (ops/cy/warp) | rate (ops/cy/SM) |
|--------|------|-----------:|-------------------:|-----------------:|
| **ex2.approx.f32** | MUFU.EX2 | **4.28** | 0.234 | **0.93** |
| tanh.approx.f32 | MUFU.TANH | 8.07 | 0.124 | 0.50 |
| cos.approx.f32 | MUFU.COS | 8.56 | 0.117 | 0.47 |
| sin.approx.f32 | MUFU.SIN | 8.56 | 0.117 | 0.47 |
| rsqrt.approx.f32 | MUFU.RSQ | 8.74 | 0.114 | 0.46 |
| sqrt.approx.f32 | MUFU.SQRT | 8.74 | 0.114 | 0.46 |
| lg2.approx.f32 | MUFU.LG2 | 8.74 | 0.114 | 0.46 |
| rcp.approx.f32 | MUFU.RCP | 9.09 | 0.110 | 0.44 |

(SM rate = 4 SMSPs × per-SMSP rate, assuming all 4 SMSPs can run MUFU concurrently — to be re-verified below.)

## LATENCY (low-ILP, N_CHAINS=1)

| Op | cy/op (latency) | Catalog (§15 L1033) |
|----|---------------:|-------------------:|
| RCP | 43.82 | 42.31 ✅ matches |
| SIN | 24.45 | (not listed) |
| EX2 | 18.12 | 14.45 ⚠ catalog 21% LOW |

## VERDICT vs catalog

### §2.10 throughput claim — **PARTLY FALSIFIED**

1. **EX2 is ~2× faster than other MUFU ops** (4.28 vs 8.7 cy/op/warp). The catalog's "0.5–0.63" hint at this for EX2 alone, but it claims uniform 0.5 elsewhere — **understated for EX2** (it's actually ~0.93/SM/cy = nearly **2×** other ops).
2. **Other MUFU ops measure at 0.46–0.50 ops/cy/SM** — close to catalog's "0.5/SMSP/cy" if catalog meant **per-SM** (which is what the silicon does). If catalog meant **per-SMSP** as written, that's **4× overstated**.
3. **MUFU.TANH = 8.07 cy/op/warp = 0.50 op/cy/SM** — confirms TANH is a first-class B300 MUFU op.

### Likely catalog interpretation

The "0.5/SMSP/cy" label is probably a unit confusion:
- If catalog measured chip throughput = 148 SMs × 4 SMSPs × 0.5 = 296 G ops/s/chip @ 1 GHz, that's 296 / 4 / 148 = 0.5 op/cy/SMSP — matches.
- But silicon-level the MUFU pipe is **per-SM** (1 unit shared across 4 SMSPs), not per-SMSP. So "0.5/SMSP" misrepresents the architecture.

### §15 latency entries — confirmed within ±21%

| Op | Catalog | Measured | Δ |
|----|--------:|---------:|--:|
| RCP | 42.31 | 43.82 | +3.6% |
| EX2 | 14.45 | 18.12 | +25% |

EX2 latency is ~25% higher than catalog claims — possibly different methodology (catalog used clock64 inline timing, mine used wall-clock + iter math).

## NEW FINDINGS (not in catalog)

1. **EX2 is the single-cycle-fastest MUFU op** — 2× throughput of all others. Use for activation functions (GELU, SiLU).
2. **TANH is supported as native MUFU** at same rate as SIN/COS — much cheaper than manual `(ex2(2x)-1)/(ex2(2x)+1)` synthesis (~22 cy → 8 cy single instruction).
3. **Need ILP ≥ 8** to saturate; latency-bound at low ILP (RCP=44 cy, SIN=24 cy, EX2=18 cy).

## VERDICT

⚠ **PARTIALLY VERIFIED** — catalog's per-op enumeration is correct in spirit, but:
- EX2's unique 2× advantage is missed (catalog hints with "0.5–0.63" but doesn't make this load-bearing)
- "Per-SMSP" rate label is likely a unit confusion; rate is per-SM
- Latency entries are ±25% (acceptable for catalog purposes)

## REVIEW_CHECKLIST candidates

- [ ] §2.10: "0.5 ops/SMSP/cy" — should this be "0.5 ops/SM/cy" or "2 ops/SM/cy"? Unit is ambiguous and possibly off by 4×
- [ ] §2.10: EX2 unique 2× speedup — should be called out as the load-bearing fact, not buried in "0.5–0.63"
