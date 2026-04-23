# §22l ADDENDUM — CCTL.IVALL characterization (DEEP)

## ⚠⚠ MAJOR CORRECTION 2026-04-23 — earlier "2 cy/line" claim is WRONG

User skepticism (2026-04-23): "Are you 100% confident about 2 cycle per cache line? Did you try the 'max L1, minimum shmem' with fully loading the L1 (try both read and write), nanosleep for many cycles, then try it - and do this at different L1 vs shmem configs which sets how much L1 there can be"

**The user was right.** The earlier measurement methodology (load → immediately CCTL → measure) was NOT measuring CCTL invalidation cost — it was measuring **in-flight cached-load drain wait**. The CCTL.IVALL itself is essentially free regardless of L1 contents.

**New rigorous test (`tests/bench_cctl_rigor.cu`)** adds `nanosleep.u32 10000;` between fill and timed CCTL, which forces all loads to complete before timing starts.

| Test | Earlier (no drain) | NEW (nanosleep drain) | Verdict |
|---|--:|--:|---|
| CCTL on empty L1 | ~3.7 cy (1.7 net) | **2.0 cy noise floor** | matches |
| CCTL after 4 KB cached loads | 60 cy (claimed ~2 cy/line) | **2.0 cy** | WAS DRAIN WAIT, not invalidation |
| CCTL after 16 KB cached loads | 312 cy (claimed ~2 cy/line) | **2.0 cy** | WAS DRAIN WAIT, not invalidation |
| CCTL after 32 KB cached loads | 270 cy | **10 cy** (single outlier) | mostly free, possibly L1 capacity boundary |
| CCTL after 64-256 KB cached loads | 2.0 cy | **2.0 cy** | both noise floor |
| Second CCTL after first | 4.0 cy | 2.0 cy | second CCTL has nothing to wait for |

**The "2 cy per cache line" claim is RETRACTED.** CCTL.IVALL on B300 sm_103a is essentially free regardless of L1 fill state. The previous "60-312 cy" measurements reflected the cost of completing in-flight cached LDG operations before the timed window starts, not invalidation cost.

### Corrected understanding

`fence.acquire.gpu` (=CCTL.IVALL) cost depends on:
- **In-flight load drain** at the moment the fence is encountered (waits for them to complete)
- NOT on number of L1-resident lines (invalidation itself is fast, possibly hardware-tagged)

For producer-consumer protocols: the consumer's `fence.acquire.gpu` will wait for any of ITS OWN in-flight loads to complete, but does NOT pay per-line invalidation cost.

### Open questions still to investigate

1. **Carveout effect** — QuickRunCUDA doesn't expose `cudaFuncSetAttribute(cudaFuncAttributePreferredSharedMemoryCarveout, ...)`. With the default carveout, the actual L1 size may be smaller than 256 KB. Worth testing with explicit max-L1 / min-shmem, but given CCTL is constant ~2 cy across all current fills, unlikely to change qualitative result.
2. ~~**Write fill behavior** with proper drain~~ — **DONE**, see addendum below
3. **Why the 32 KB outlier (10 cy)?** — non-monotonic at 32/36/40 KB shows 10 cy, while 24/28/30/31/33/34/48 KB show 2 cy. Even with 50 µs sleep the 32 KB still shows 10 cy. Not drain-wait. Suspect SASS-emission boundary at specific unroll levels — but only an 8 cy difference. Negligible.

### MODE 5 verification — CCTL after WRITES with drain = 2 cy ALL sizes

| Fill KB | cy with drain |
|---:|--:|
| 4 | 2.00 |
| 16 | 2.00 |
| 32 | 2.00 (no outlier here, unlike loads!) |
| 64 | 2.00 |
| 128 | 2.00 |
| 256 | 2.00 |

The earlier "59 cy after writes" was ALSO drain wait — same root cause as the load case. With proper nanosleep, writes + CCTL is the same noise floor as loads + CCTL.

### FINAL CORRECTED CCTL.IVALL MODEL

**CCTL.IVALL on B300 sm_103a is essentially FREE (~2 cy)** under ALL tested conditions when prior memory ops have drained. The earlier non-zero costs were ALL drain wait.

`fence.acquire.gpu` cost = drain time for in-flight memory ops at the moment of execution. Once drained, the CCTL itself is essentially instantaneous.

This is hardware-tagged invalidation (probably a single-cycle "bump valid bit" operation across all L1 lines) rather than per-line work. Makes sense architecturally — it's exactly what you'd want for an acquire fence to be cheap.



This is a textbook example of the user's "MECHANISM is a hypothesis to test" rule. The earlier conclusion looked plausible and matched the catalog narrative ("L1 invalidate is per-line"), but a more rigorous test reveals the cost was elsewhere. Lesson reinforced.

---


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

1. ~~Fix the writes test — find a PTX form that genuinely fills L1 with dirty lines that CCTL must invalidate~~ — **PARTIALLY ADDRESSED** (see addendum below)
2. Probe L1 capacity precisely — at what KB count does L1 evict lines on a single-thread fill?
3. Investigate WHY the compiler refuses full unroll past ~32 KB worth of loads (NVRTC limit? Code-size heuristic?)
4. Test CCTL on small kernels with small L1 footprints to see if there's a per-CTA L1 share quota

---

## ADDENDUM 2 — Store cache hints don't control L1 (2026-04-23)

Tested 3 different store-fill modes:

| Mode | PTX | SASS emitted | CCTL @4 KB | @16 KB | @32 KB |
|---|---|---|--:|--:|--:|
| 4 | `st.global.wb.u32` (L1+L2 documented) | `STG.E.STRONG.SM` | 59 | 59 | 3 |
| 6 | `st.global.cg.u32` (L2-only documented) | `STG.E.STRONG.GPU` | 59 | 59 | 3 |
| 8 | RMW: `ld.global.ca` + `st.global.wb` | `LDG.E.STRONG.SM` + `STG.E.STRONG.SM` | 36 | 3 | 3 |

**Key observation**: `.wb` (L1+L2) and `.cg` (L2-only) give **identical CCTL costs**. If the cache modifiers actually controlled L1 caching as PTX docs imply, `.wb` should fill L1 with dirty lines (high CCTL cost) while `.cg` should bypass L1 entirely (low CCTL cost). **They behave the same.**

This means EITHER:
- Both `.wb` and `.cg` bypass L1 caching for stores on B300, OR
- All store paths inherently cache in L1 (regardless of hint), OR
- The cache hints affect SOMETHING ELSE (e.g. memory ordering, write-combining) but not L1 residency

Note the SASS difference: `.wb` → `.STRONG.SM` (SM-scope ordering), `.cg` → `.STRONG.GPU` (GPU-scope ordering). So the hints DO affect ordering scope, but not L1 caching.

**Re-interpretation of the 59 cy CCTL cost**: not "invalidating dirty L1 lines" but more likely **waiting for in-flight stores to drain to L2**. At 4-16 KB worth of stores, some are still in flight when t0/CCTL/t1 happens; the CCTL waits. At 32+ KB the writes have had more time to drain naturally, so by the time CCTL hits there's nothing to wait for.

**MODE 8 (RMW)** is interesting: 36 cy at 4 KB but 3 cy at 16+ KB. The load-then-store pattern serializes naturally (load must complete before store), so by the time CCTL hits at 16+ KB the work is fully drained.

### Implication

The original hypothesis "CCTL after dirty L1 = expensive" is **NOT verified** by these tests. The 59 cy plateau at small write counts is more plausibly **drain wait** than dirty-line invalidation.

The earlier load-fill results (60 cy at 4 KB / 312 cy at 16 KB cached LOADS, scaling with line count) DO show genuine L1-resident-line scaling — that part stands.

### Updated TL;DR

| Scenario | cy | mechanism |
|---|--:|---|
| Lone CCTL on truly-empty L1 | ~1.7 net | SASS-cycle floor |
| Chained CCTL (1024-deep) | ~16/op | shallow internal queue saturates |
| CCTL after CACHED LOADS (4-16 KB filled) | 60-312 | ~2 cy per L1-resident line — **REAL invalidation cost** |
| CCTL after STORES (any KB count) | 3-59 | NOT dirty-L1 invalidation; store-drain wait dominates. Stores apparently bypass L1 caching on B300 regardless of `.wb`/`.cg` hint |

### Action

The "consumer of cached data pays per-line invalidation cost" finding (~2 cy/line for cached loads) is robust.

The "producer of cached data pays similar cost on next CCTL" claim is **NOT supported** — store paths on B300 don't appear to fill L1 in the way that would make subsequent CCTLs more expensive.

This is an important refinement: `fence.acquire.gpu` cost depends on what your kernel CACHED-LOADED prior, not on what it WROTE. If your kernel is write-only before the acquire, CCTL is cheap. If it has cached reads, CCTL pays per-line.
