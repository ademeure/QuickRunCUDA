# Legacy mma.sync Power Asymmetry

Date: 2026-04-20. Tests if data-dependent power asymmetry exists in legacy
mma.sync (Hopper-style tensor path), comparing to tcgen05 findings.

## Setup
- `tests/bench_mma_sync_power.cu`
- mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
- 1184 blocks × 128 threads = 148 SMs × 8 blocks × 4 warps each
- @ -lgc 1005 MHz, 5M iters

## Results

| Mode | A pattern | B pattern | Power (W) | cy/inst |
|------|-----------|-----------|----------:|--------:|
| 0 | const +1.0 | const +1.0 | 175 | 24.4 |
| 1 | const | random | 201 | 24.4 |
| 2 | random | const | 184 | 24.4 |
| 3 | random | random | 206 | 24.4 |
| 4 | const | zero | 175 | 24.4 |
| 5 | zero | zero | 170 | 24.4 |

## Findings

1. **A vs B asymmetry EXISTS in mma.sync** (mode 1 vs mode 2):
   - B random + A const: 201W (+26W from baseline)
   - A random + B const: 184W (+9W from baseline)
   - **17W gap** (vs tcgen05's 250W gap)

2. **Random vs constant gap is SMALLER in mma.sync**:
   - mma.sync: 206W vs 175W = +31W
   - tcgen05: 609W vs 299W = +310W
   - 10× smaller in mma.sync

3. **Zero data**: 170W (-5W vs const) — much smaller Tier A vs Tier B than tcgen05

## Per-MAC analysis

mma.sync m16n8k16 = 128 outputs × 16 K = 2048 MACs/inst.
Per warp throughput: 2048 / 24.4 = 84 MACs/cy.
Per SM (4 SMSPs * 4 warps each, contended): ~336 MACs/cy.
148 SMs total: ~50 GMACs/cy = 50 TFLOPS at 1005 MHz.

vs theoretical mma.sync peak ~285 TFLOPS @ 1005 MHz: ~17% utilization.
(Heavy warp contention on tensor pipes; many warps per pipe).

Per-MAC random penalty:
- mma.sync: 31W / 50 GMACs = ~0.6 nW/MAC
- tcgen05: 310W / ~600 GMACs = ~0.5 nW/MAC
- COMPARABLE per-MAC cost

## Conclusion: dedup mechanism is multiplier-fundamental

The data-dependent power profile is a **fundamental property of the BF16
multiplier hardware**, present in both legacy mma.sync and modern tcgen05.

The MAGNITUDE differs because:
- tcgen05 m128n128k16 issues ~128× more MACs per instruction
- Higher concurrent multiplier activity → higher absolute power
- Same per-MAC physics

## Implications

1. **Power optimization recipes apply to BOTH paths**:
   - Sort B by similarity (saves more on tcgen05 because higher absolute)
   - A operand is preferentially the high-entropy operand
2. **mma.sync is LESS power-sensitive in practice**:
   - 31W gap is small relative to baseline (17%)
   - tcgen05's 310W gap is 51% of baseline
3. **The dedup mechanism is multiplier hardware, NOT tcgen05 software**

## Confidence

- HIGH on A vs B asymmetry being real in mma.sync (5+ measurements, monotonic)
- HIGH on per-MAC equivalence (algebra checks out)
- MEDIUM on the absolute mma.sync utilization (depends on occupancy)
- LOW on whether other mma.sync shapes (m16n16k16, m8n8k4) behave same

---

## mma.sync FP8 e4m3 m16n8k32 — same dedup mechanism

| Mode | A | B | Power (W) |
|------|---|---|----------:|
| 0 | const +1.0 | const +1.0 | 197 |
| 1 | const | random | 240 (+43) |
| 2 | random | const | 218 (+21) |
| 3 | random | random | 251 (+54) |
| 4 | const | zero | 195 |
| 5 | zero | zero | 190 |

A vs B asymmetry: 22W (B random +43, A random +21).

vs BF16 mma.sync (m16n8k16):
- BF16 random gap: +31W
- FP8 random gap: +54W (~75% more)
- Reason: FP8 has K=32 (2x more multiplier work per inst)

**Dedup mechanism is precision-INDEPENDENT** — present in both BF16 and FP8
mma.sync. The fundamental BF16/FP8 multiplier hardware shares the same
dedup capability.

cy/inst:
- BF16 mma.sync: 24.4 cy
- FP8 mma.sync: 31.79 cy (FP8 has K=32 vs BF16 K=16)
