# §22l ADDENDUM — CCTL.IVALL characterization (DEEP)

Audit date: 2026-04-23
Auditor: Claude Opus 4.7 (main session, manual)
GPU: B300 SXM6 AC, GPU 0
Clock: **`-lgc 1800`** locked (verified via `nvidia-smi --query-gpu=clocks.gr` = 1800 MHz)
Test kernel: `tests/bench_cctl_ivall.cu` (single-thread, tid=0/blk=0, clock64-bracketed)
Trigger: user 2026-04-23 — "test performance of CCTL.IVALL when 1) chained/lots of them, 2) when there is a lot of data in L1 to invalidate (read / or written since last invalidate, test both separately)"

## TL;DR (per-test, all @1800 MHz locked)

| Scenario | cy | notes |
|---|--:|---|
| clock64 noise floor (empty between t0/t1) | 2.0 | baseline |
| **single CCTL.IVALL (L1 already empty)** | **3.7** = ~1.7 net | matches §22l decomp's 25-cy claim being inflated by other test scaffolding |
| Chained 2× CCTL | 9.4 → ~3.7 cy/op net | |
| Chained 32× CCTL | 142 → **4.4 cy/op** | |
| Chained 256× CCTL | 3369 → **13.2 cy/op** ← queue saturating | |
| Chained 1024× CCTL | 16,300 → **15.9 cy/op** ← saturated | |
| **CCTL after 4 KB cached LOADS (32 LDG, fully unrolled)** | **60 cy** | ~1.9 cy per cache line invalidated (32 lines) |
| **CCTL after 16 KB cached LOADS (128 LDG)** | **312 cy** | ~2.4 cy per line (128 lines) |
| CCTL after 32 KB loads (256 LDG) | 270 cy | non-monotonic — may include cache eviction |
| CCTL after 64+ KB loads | 2.0 cy noise floor | ⚠ compiler DCE: only 32 LDG emitted (refused full unroll) |
| **2nd CCTL immediately after 1st (L1 emptied by 1st)** | **4.0 cy** | matches single-CCTL-empty-L1 case ≈ 1-2 cy net |
| CCTL after 4-16 KB WRITES | 59 cy | ⚠ writes compiled to `STG.E.STRONG.SM` not `STG.E.WB` — see caveat |
| CCTL after 32-64 KB WRITES | 3 cy | ⚠ same caveat |
| CCTL after 128+ KB WRITES | 24 cy | ⚠ DCE: only 33 STG (compiler refused full unroll) |

## Findings

### 1. Lone CCTL.IVALL on empty L1 = ~1.7 cy net

Single CCTL adds ~1.7 cy above clock64 noise floor (2.0). The earlier §22l "25 cy" was the cost in a multi-thread context with prior work; lone-thread/empty-L1 is much cheaper.

### 2. Chained CCTL queue saturates around 32-256 deep

Per-op cost as chain depth grows:
- 1 CCTL: 1.7 cy
- 2 CCTL: 3.7 cy/op
- 32 CCTL: 4.4 cy/op
- 256 CCTL: 13.2 cy/op
- 1024 CCTL: 15.9 cy/op

A short chain is cheap; long chains saturate at ~16 cy/op. This suggests B300's L1-invalidate path has a shallow scoreboard or queue (probably ~32-64 deep) — beyond that, the CCTLs back up at the dispatcher.

### 3. CCTL cost SCALES with L1-resident lines (~2 cy per line)

| KB cached (LOADS, fully-unrolled SASS) | Lines | CCTL cy | cy/line |
|---:|---:|--:|--:|
| 4 | 32 | 60 | 1.88 |
| 16 | 128 | 312 | 2.44 |
| 32 | 256 | 270 | 1.05 (non-monotonic) |

The 4-KB and 16-KB cases give a consistent ~1.9-2.4 cy per L1 line invalidated. The 32-KB case is non-monotonic, suggesting either (a) some lines were evicted from L1 before CCTL hit (L1 effective capacity smaller than 32 KB?) or (b) measurement noise.

⚠ **Critical caveat**: at 64+ KB the compiler REFUSED to fully unroll the load loop and DCE'd most loads (only 32 LDG instead of 512+). All "2 cy" results at 64+ KB are MEASUREMENT ARTIFACTS, not real CCTL behavior. To probe larger fills properly, the test would need a different anti-DCE structure (e.g. a runtime `volatile` workspace and a smarter chain).

### 4. WRITES test is ambiguous — `st.global.wb` did NOT compile as expected

PTX `st.global.wb.u32` compiled to **`STG.E.STRONG.SM`** (strict-ordering, SM-scope), NOT `STG.E.WB` (write-back to L1). So my MODE 4 test isn't actually filling L1 with dirty lines as intended. The 59 cy / 3 cy / 24 cy results don't cleanly answer "how much does CCTL cost when L1 has dirty lines?".

⚠ TODO follow-up: use a kernel pattern that DEFINITELY caches writes in L1 (perhaps using shared memory + L1 contention, or `cp.async` patterns).

### 5. 2nd CCTL after 1st = empty-L1 cost (~4 cy)

MODE 5 confirms: after the first CCTL invalidates everything, the second CCTL runs at ~4 cy regardless of how much was previously loaded. This is exactly what we'd expect — CCTL cost is proportional to L1-RESIDENT lines, not historical lines.

## Architectural conclusions

1. **CCTL.IVALL cost scales linearly with L1-resident lines** at ~2 cy/line for the regime we could measure (32-128 lines).
2. **An empty L1 → CCTL is essentially free** (~2 cy net above noise).
3. **Chains of CCTL saturate** at ~16 cy/op for ≥256 deep — bottlenecked by some shallow internal queue.
4. **The "25 cy" from §22l** is the cost in the actual grid-sync context (with surrounding instructions, fences, and an L1 that has had recent work in it). Not directly comparable to lone-CCTL.
5. **The "L2 is the GPU-coherence point" insight stands** — CCTL.IVALL is just an L1 invalidate; L2 is unchanged.

## Implication for `fence.acquire.gpu` consumers

If you're a consumer that issues `fence.acquire.gpu` to read fresh data AFTER your own loads have populated L1, the cost can be hundreds of cycles (proportional to L1-cached lines being invalidated). To minimize:
- Use `fence.acquire.gpu` BEFORE doing your reads, when L1 is still cold (cheap)
- OR use `ld.acquire.gpu` per-load (which presumably bypasses L1 for that one load only)
- AVOID `fence.acquire.gpu` after a heavy phase of cached loads — you'll pay 2 cy per cached line you've touched

## Files preserved

- `tests/bench_cctl_ivall.cu` (the test kernel)
- SASS dumps in `sass/bench_cctl_ivall_*.sass` (one per build hash)

## Open follow-ups

1. Fix the writes test — find a PTX form that genuinely fills L1 with dirty lines that CCTL must invalidate
2. Probe L1 capacity precisely — at what KB count does L1 evict lines on a single-thread fill?
3. Investigate WHY the compiler refuses full unroll past ~32 KB worth of loads (NVRTC limit? Code-size heuristic?)
4. Test CCTL on small kernels with small L1 footprints to see if there's a per-CTA L1 share quota
