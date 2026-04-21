# A2: Warp scheduler policy — fair, but sibling SMSP warps share issue port

## Theoretical
B300 SMSP scheduler choices (Volta+ heritage):
- Round-robin (RR): each warp gets equal cycles
- LRU (least-recently-used): rotates across ready warps
- GTO (greedy then oldest): one warp runs until stalled, then next
- Loose Round-Robin (LRR): RR with stall-skip

If RR/LRU/LRR: per-warp times should be equal under uniform load.
If GTO: one warp finishes much earlier, others queued.

Cluster A/B mapping check: if MUFU heavy warp slows its sibling FFMA warp on same SMSP, the issue port is fully shared.

## Methodology
- Each warp records its own `clock64` start/end (lane 0 only)
- Per-warp times saved to C buffer, dumped via `--dump-c`
- 4 modes × 3 warp counts = 12 measurements
- Range = max/min per scenario quantifies fairness

## Measured (per-warp cy/iter, single-block, 1000 iters × 16 unroll)
| NWARPS | MODE | min cy | max cy | range | description |
|--------|------|--------|--------|-------|-------------|
| 4 | 0 (all FFMA) | 71061 | 71064 | 1.00004× | extremely fair |
| 8 | 0 | 72067 | 72073 | 1.00008× | extremely fair |
| 16 | 0 | 80958 | 80988 | 1.00037× | extremely fair |
| 4 | 1 (all LDS) | 619077 | 619083 | 1.00001× | extremely fair |
| 8 | 1 | 634078 | 634086 | 1.00001× | extremely fair |
| 16 | 1 | 648109 | 650136 | 1.00313× | very fair |
| 4 | 2 (mix FFMA/LDS) | 638014 | 638057 | 1.00007× | fair |
| 8 | 2 | 641763 | 653935 | 1.019× | bimodal: LDS warps +1.9% |
| 16 | 2 | 698523 | 717595 | 1.027× | LDS warps consistently +2.7% |
| **4 | 3 (warp 0 = MUFU, rest FFMA)** | **264056** | **292054** | **1.106×** | warp 0 MUFU = 292K |
| **8 | 3** | **267066** | **292062** | **1.094×** | warps 0 AND 4 = 292K (same SMSP!) |
| 16 | 3 | 512047 | 512065 | 1.00004× | all 4 warps/SMSP saturate issue port |

## Critical observation: NWARPS=8 MODE=3
Warps 0 and 4 BOTH took 292K cy; warps 1-3, 5-7 took 267K cy.

CUDA assigns warps to SMSPs round-robin: warp_id % 4 → SMSP_id. So:
- Warp 0 (MUFU) → SMSP 0
- Warp 4 (FFMA) → SMSP 0
- Other warps (FFMA only) → SMSPs 1-3

Result: **the FFMA warp on SMSP 0 (warp 4) was slowed to MUFU's pace**, even though it doesn't use MUFU. This proves the SMSP issue port is shared between Cluster A (FFMA) and the MUFU pipe.

## Conclusion
1. **B300 warp scheduler is highly fair** (≤0.04% variance for uniform work, ≤0.3% across warp counts).
2. **Mixed-pipe loads show ~2-3% bias** between pipe types — minor, not starvation.
3. **No GTO behavior observed** — all warps progress uniformly.
4. **SMSP issue port is shared across pipes**: a slow MUFU warp slows its same-SMSP FFMA siblings by ~10%.
5. At 4 warps/SMSP: issue port is fully saturated; per-warp throughput drops 2× vs 1 warp/SMSP.

## Practical implications
- Don't worry about scheduler fairness — it's fine.
- For latency-critical fast warp + slow warp mix, place them on DIFFERENT SMSPs (use explicit warp ID logic) rather than relying on automatic assignment.
- Saturate at 2-4 warps per SMSP for steady throughput; more = same aggregate issue port shared.

## Confidence: HIGH
Verified by:
- Reproducible across 3 warp counts
- Per-warp clock measurements (no scheduler-introspection ambiguity)
- Bimodal pattern in MODE 2/3 matches SMSP assignment (warp_id % 4)
- Self-consistent fairness numbers (uniform work → uniform times)

What would change it: if a future test shows large per-warp variance (>10%) under uniform load, scheduler model would update.
