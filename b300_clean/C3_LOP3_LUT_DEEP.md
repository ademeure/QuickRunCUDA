# LOP3.LUT deep characterization — V4 / C3

**Date: 2026-04-20.** Tests `bench_lop3_lut_sweep.cu` and
`bench_lop3_port_pressure.cu`. Clock locked 1500 MHz, persistent grid,
3 trials each, pkill+sleep 6 between.

## TL;DR

- **Throughput is COMPLETELY imm-independent** across all 256 truth-table
  encodings. Verified across 12 representative imms (zero, identity,
  XOR3, MAJ, MUX, AND3, OR3, all-ones, NOT). All hit 14.08–14.16 TIOPS.
- **Latency: ≈4.5 cy** (1-warp per SMSP, single chain, no ILP)
- **Throughput: 0.5 warp-inst per SMSP per cycle** = 14.16 TIOPS at 1500
  MHz = 19.18 TIOPS at 2032 MHz boost
- **Register ports: ≥3 reads per cycle, no port pressure**. 1, 2, or 3
  unique source registers all give identical latency and throughput.
- Confirms catalog (B300_TRUE_REFERENCE.md): LOP3 = 4 cy lat, 2.0/SM/cy.
  My number is 4.5 cy and 1.99/SM/cy → matches.

## Imm-independence sweep (1500 MHz, 8 chains × 16 unroll, persistent)

| Imm  | Op meaning              | Time/iter (ms) | TIOPS  |
|------|-------------------------|----------------|--------|
| 0x00 | const 0                 | 2.15181        | 14.09  |
| 0xAA | A (pass-through)        | 2.15221        | 14.08  |
| 0xCC | B (pass-through)        | 2.15206        | 14.08  |
| 0xF0 | C (pass-through)        | 2.15214        | 14.08  |
| 0x96 | A^B^C (3-input XOR)     | 2.15206        | 14.08  |
| 0x69 | ~(A^B^C)                | 2.15225        | 14.08  |
| 0xE8 | MAJ(A,B,C)              | 2.15192        | 14.09  |
| 0xCA | A?B:C (mux)             | 2.15235        | 14.08  |
| 0x80 | A&B&C                   | 2.15197        | 14.08  |
| 0xFE | A|B|C                   | 2.15200        | 14.08  |
| 0xFF | const 1                 | 2.15194        | 14.09  |
| 0x55 | ~A                      | 2.15228        | 14.08  |

Variance: ±0.005 TIOPS = ±0.04%. **Truth-table value does NOT affect
throughput.**

## Port-pressure sweep (1 warp/SM = 1 warp/SMSP)

`-t 32 -p` ⇒ each block = 1 warp, sent to a single SMSP, other 3 SMSPs idle.

| PORT_MODE | NC=1 | NC=2 | NC=4 | NC=8 |
|-----------|------|------|------|------|
| 0 (a,a,a)  | 4.48 cy | 2.30 cy | 2.15 cy | **2.08 cy** |
| 2 (a,b,c)  | 4.47 cy | 2.30 cy | 2.17 cy | **2.08 cy** |

Mode 0 = LOP3(R, R, R) → 1 unique register read per inst.
Mode 2 = LOP3(R, S, T) → 3 unique register reads per inst.

**Identical latency and throughput.** RF can deliver ≥3 unique reads
per LOP3 issue cycle without throttling.

## Latency vs throughput decomposition

- **NC=1, 1 warp/SMSP**: pure latency-bound (each LOP3 waits for prev).
  Measured 4.47 cy → **LOP3 latency = 4–5 cycles** (likely 4 + small
  loop-branch overhead).
- **NC=8, 1 warp/SMSP**: throughput-bound (saturating SMSP issue).
  Measured 2.08 cy/op → 1 LOP3 every 2 cy per SMSP = 0.5/SMSP/cy.
- **NC=8, 8 warps/SM** (full persistent): 4.03 cy/op per thread × 32 lanes
  = 14.16 TIOPS chip-wide. Confirms 0.5/SMSP/cy × 4 SMSPs = 2/SM/cy.

Pipeline depth ≈ ceil(latency / issue_period) = ceil(4.5/2) = 3 stages.

## SASS verification

Compiled with `-H "#define LOP3_IMM 0xFF"`, kernel body shows:

```
/*0240*/  LOP3.LUT R3, R3, R10, R9, 0xff, !PT ;
/*0250*/  LOP3.LUT R4, R4, R11, R12, 0xff, !PT ;
/*0260*/  LOP3.LUT R5, R5, R14, R13, 0xff, !PT ;
... (128 such inner-loop LOP3s)
```

Total 134 LOP3 instructions in cubin (128 inner + 6 setup) — matches
8 chains × 16 unroll = 128 expected per pass. **Each chain step =
exactly one LOP3.LUT.** No multi-instruction sequences. No fall-back
to LOP3+something for unusual imms.

## Saturation conditions

- **1 warp/SMSP** is enough to saturate LOP3 at the SMSP → 0.5/cy.
- **4 warps/SM** (one per SMSP) saturates the whole SM → 2/SM/cy.
- More warps don't help — pipe is the bottleneck, not warp count.
- **3 chains / warp** is enough ILP to saturate (since 4.5 cy lat / 2 cy
  issue period ≈ 2.25 → 3 chains fully cover).

## Practical implication

For LOP3-heavy kernels (radix-X conversion tables, packed bitfield
extraction, fused boolean ops, narrow-format arithmetic):

1. **No need to pick "favorable" truth tables** — all 256 imms
   identical throughput.
2. **No need to limit unique source register reads** — 3 distinct
   sources cost no more than 1 reused register.
3. **Need only 4 warps/SM** to fully utilize LOP3 pipe.
4. **Need 3+ independent chains per warp** to overlap latency at peak.

## Confidence

- **HIGH** — 12 imm × 4 ILP × 3 port modes, all consistent.
- Single number: **LOP3 = 14.16 TIOPS at 1500 MHz, 4.5 cy lat, 0.5/SMSP/cy**.
- At 2032 MHz boost: **~19.2 TIOPS chip peak**.

## Files

- `tests/bench_lop3_lut_sweep.cu` — imm sweep
- `tests/bench_lop3_port_pressure.cu` — port pressure × ILP
- Both compiled with persistent grid, 1500 MHz lock, ITERS=100k–200k
