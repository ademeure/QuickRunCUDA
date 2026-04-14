# Report #3 — CORRECTION: the "+1 STG halves F2FP" was an addressing artifact

Initial +1 STG → −50% F2FP result (Report #2) was because 592 blocks × 16 warps
were all writing to the same 1 KiB region of `C`. Inter-block cache-line
collisions caused an L1/L2 store-queue pile-up that serialized against F2FP
issue, NOT a direct F2FP↔STG pipe/port conflict.

## With proper per-block distinct addresses + warp-coalesced STG:

### 8 F2FPs/iter base (unpack round-trip with 1 pair)

| +STG/iter | F2FP /SM/clk | Δ |
|---:|---:|---:|
| 0 | 63.70 | — |
| 1 | 63.64 | ~0% |
| 2 | 55.74 | −12% |
| **3** | **12.60** | **cliff** |
| 4 | 9.64 | further drop |
| 8 | 5.35 | |
| 32 | 1.65 | |

### 32 F2FPs/iter base (4 pairs)

| +STG/iter | F2FP /SM/clk | Δ |
|---:|---:|---:|
| 0 | 62.13 | — |
| 1 | 63.71 | +2.5% |
| 2 | 63.77 | +2.6% |
| 3 | 63.64 | ~0 |
| **4** | **55.48** | **−11% starts** |
| 6 | 27.81 | |
| 8 | 21.24 | |
| 32 | 6.58 | |

## Interpretation

Two separate effects stacked in the original measurement:

1. **Inter-block address collision** — 592 blocks writing to same 1 KiB region
   caused massive L2-write-queue pressure. This serialized stores across
   the whole GPU. With per-block distinct addresses (own 32KB region), this
   goes away — the first 1–3 STGs/iter are essentially free.

2. **Store-queue saturation cliff** — past some STG-per-iter threshold (≈3 for
   8 F2FPs; ≈4 for 32 F2FPs), per-SM store queue fills and warps stall on
   it. This is proportional to F2FP count × (store cost per STG), not a pipe
   sharing with F2FP execution.

## Store cache policy doesn't matter

All `st.global` cache hints (`.cs`, `.cg`, `.wt`, default) behave identically:

| Hint | +STG=4 F2FP /SM/clk |
|---|---:|
| default | 31.84 |
| `.cs` | 31.85 |
| `.cg` | 31.88 |
| `.wt` | 31.88 |

## Other store types

| Store type | +STG=4 F2FP /SM/clk | +STG=16 |
|---|---:|---:|
| st.global.f32 | 31.84 | 7.98 |
| st.shared.f32 | 63.00 | 14.39 |
| st.local.f32 (stack) | 63.58 | **63.59** (completely free) |
| st.global.v4.f32 (wider) | 10.85 | 2.72 |
| atom.shared.or.b32 | 50.34 | 14.96 |

**st.local is completely free** (likely compiled to register-file ops).
**st.shared is ~4× cheaper** than st.global.
**st.global.v4** costs 4× more than st.global.f32 (proportional to bytes stored).

## Lesson for the session

The saturating-config measurements are very sensitive to address patterns.
Every future contention test needs to verify: (a) warp-coalesced access
and (b) block-distinct addresses, to avoid measuring L2-queue contention
instead of SM-local effects.
