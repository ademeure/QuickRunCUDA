# Report #2 — F2FP contention with LD/ST (2026-04-14, continuing session)

Same saturating F2FP config as REPORT_01 baseline (pure round-trip, 32 F2FPs/iter
= 63.83 thread-F2FPs/SM/clk). Added N_LDG loads and/or N_STG stores per iter on
independent register chains.

## Headline

**STG contends catastrophically with F2FP.** Even a single store per iter halves
F2FP rate from 63.8 → 32.3 /SM/clk. LDG contends mildly (−33% at +32 LDG) like
FFMA/IMAD.

## Data (F2FP /SM/clk at saturating config, N_CHAINS=4 × CHAIN_PAIRS=4)

| Companion | +0 | +1 | +2 | +4 | +8 | +16 | +32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LDG (L1 hit) | 63.83 | 63.75 | 63.77 | 60.06 | 56.73 | 51.03 | 42.43 |
| LDG (wide address) | 62.26 | 63.76 | 63.76 | 59.98 | 56.64 | 50.73 | 31.70 |
| **STG (L1 hit)** | 63.75 | **32.33** | **31.92** | **31.76** | **12.55** | **4.78** | **1.25** |
| LDG+STG mix (both N) | 63.83 | 32.41 | 31.92 | 29.98 | 12.55 | 5.02 | — |

## Interpretation

**The +1 STG → 32/SM/clk drop is surgical.** 32/SM/clk is exactly the pack-alone
rate. Interpretation: the store instruction blocks the "heavy" SFU slot that the
pack F2FP (with MERGE_C) was using, leaving only unpack (the light slot) still
firing. With more stores, even unpack is starved.

Possible mechanisms:
1. **STG.E's register-read port conflicts with F2FP's destination-read port
   (MERGE_C).** Both want to read a register in the same cycle; scheduler serializes.
2. **STG uses a warp-scheduler dispatch slot that serves as the SM's "writeback"
   queue, which MERGE_C also uses.** Pack needs to read-then-write the destination
   in sequence; STG reads a register to buffer the write — shared resource.
3. **STG commits through the same LSU (Load-Store Unit) that on Blackwell is
   coupled to the SFU dispatcher.** The 1→32 drop pattern suggests slot-level
   contention, not pipe sharing (which would be linear).

**LDG (both cached and uncached) contends mildly**, similar magnitude to FFMA:
 −0.1%/op at low counts, −33% at +32. Consistent with LDG on the memory pipe
not conflicting with SFU execution — only fighting for scheduler bandwidth.

## Next experiments

- Try `st.global.wt` (write-through) vs `st.global.cg` (cache-global) vs
  `st.global.cs` (streaming) to see if cache policy changes the pattern.
- Try `ld.shared` / `st.shared` — does SMEM have the same issue?
- Check SASS of the +1 STG kernel to verify it's a single STG per iter.
- Rerun with pack-only F2FP (32/clk) vs unpack-only (64/clk) — does STG kill both
  equally or only pack?

