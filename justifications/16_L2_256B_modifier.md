# §16 / user L925 — L2::256B cache modifier (Blackwell-specific) — JUSTIFIED record

**Date:** 2026-04-23
**User concern:** `reviewed_errors_b300.md` L925: *"at one point we definitely had a microbenchmark loading 256 byte per thread by using a stride of 256B for 4B reads and the .256B L2 cache modifier, I think this was the actual highest DRAM BW % we saw in any microbenchmark"*

## CLAIM (catalog L1773 + L1840)

- `cp.async.ca.shared.global.L2::256B` → `LDGSTS.E.LTC256B.128`
- `ld.global.L2::256B.u32` → `LDG.E.LTC256B`

This is a Blackwell-specific 256-byte L2 sector prefetch hint.

## TEST (this audit)

`tests/bench_ldgmc.cu` OP=0 contains exactly this pattern:
```c
asm volatile("ld.global.L2::256B.u32 %0, [%1];" : "=r"(r) : "l"(gbase));
```

### SASS confirmed

```
LDG.E.LTC256B
```

The `.LTC256B` modifier IS emitted as expected — confirms the catalog mapping.

### Wall-clock + ncu (this test config — small WS, NOT the user's recipe)

- OP=0 (.L2::256B): 0.84 ms / DRAM BW 706-732 GB/s
- OP=1 (standard): 0.87 ms / (similar)

**Only 3% wall-clock difference** at this WS — but this test uses small enough WS that DRAM isn't saturated (~700 GB/s vs 7000 GB/s HBM peak).

## NOT REPRODUCED

User's specific recipe ("stride 256B, 4B reads, .256B modifier giving highest DRAM BW") was NOT reconstructed in this audit:
- The pattern requires careful per-thread address striding (256B stride between consecutive threads)
- 4-byte loads with .L2::256B modifier
- Aiming for HW to prefetch the full 256B sector per thread access
- Goal: exploit the L2 sector size to fetch 64× the requested data per LDG

The IDEA: if the L2 sector is 256B but you only request 4B, the .L2::256B hint tells the cache "yes, fetch all 256B". With consecutive threads at 256B stride, all 64 lanes of a warp would fetch contiguous 256B blocks = 64 × 256B = 16 KB per warp instruction. Could plausibly hit very high DRAM BW.

This is high-value but **deferred** — would require careful reconstruction of the user's exact test geometry.

## VERDICT

✅ **SASS emit confirmed**: `ld.global.L2::256B.u32` → `LDG.E.LTC256B`
🟡 **Performance claim ("highest DRAM BW% ever measured") preserved but not reproduced** — needs careful test geometry per user's L925 recipe; small-WS test in this audit doesn't saturate DRAM.

## REVIEW_CHECKLIST candidates

- [x] §16 .L2::256B modifier emits LDG.E.LTC256B — ✅ confirmed via SASS
- [ ] User L925 recipe (stride 256B + 4B + .L2::256B) for highest DRAM BW% — NOT reproduced; valuable test to reconstruct in a future iteration
