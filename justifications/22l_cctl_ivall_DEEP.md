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

---

## ADDENDUM 3 (2026-04-23): The DEFINITIVE mechanism — CCTL drains in-flight loads

A further deep-dive in `tests/bench_cctl_rigor.cu` MODES 30-76 reveals the
mechanism even more clearly. **CCTL.IVALL itself is essentially free.** What we
were measuring as "fence cost" is the **drain-wait for in-flight global loads**
that the acquire fence forces to retire before invalidation.

### Key measurements (B300 sm_103a, `-lgc 1800`, 1 thread, single-block)

| MODE | Pattern (always followed by `fence.acquire.gpu`) | cy/iter |
|------|---------------------------------------------------|--------:|
| 0  | clock-clock baseline                                  | 2.00 |
| 2  | nanosleep → CCTL on truly idle pipeline               | 2.00 |
| 44 | CCTL with no prior op, no nanosleep                   | **2.83** |
| 52 | 1× ld.ca L1-hit (warmed) + CCTL                       | **24.93** |
| 50 | 1× ld.volatile L2-hit + CCTL                          | **82.70** |
| 53 | 1× ld.global default L2-hit + CCTL                    | 82.63 |
| 70 | 1× ld.relaxed.gpu L2-hit + CCTL                       | 83.13 |
| 72 | 1× ld.relaxed.cta L2-hit + CCTL                       | 81.97 |
| 73 | 1× ld.weak L2-hit + CCTL                              | 95.13 |
| 51 | 1× ld.volatile DRAM-cold + CCTL                       | **1042** |
| 71 | 1× ld.relaxed.gpu DRAM-cold + CCTL                    | 976 |
| 41 | 8× ld.ca pipelined (CSE→8 unique L2 lines) + CCTL     | 184 |
| 40 | 8× ld.cg pipelined DRAM-cold + CCTL                   | 361 |
| 42 | 32× ld.cg pipelined DRAM-cold + CCTL                  | 338 |
| 74 | 8× ld.relaxed.gpu DRAM-cold (4KB stride) + CCTL       | 1325 |
| 75 | 1× ld.weak (NO fence, reg-dep clock t1)               | 762 |
| 76 | 1× ld.weak + 8 FFMA chain + CCTL                      | **2.00** |

### What this proves

1. **CCTL.IVALL on idle pipeline = 2-3 cy.** The "186 cy MEMBAR.ALL.GPU" we
   measured earlier is for the *release* fence — that one DOES have intrinsic
   cost. Acquire is essentially free.

2. **`fence.acquire.gpu` waits for all in-flight global loads to retire.**
   The cost = completion latency of the slowest in-flight load. This is the
   acquire-fence ordering guarantee enforced at the hardware level.

3. **The PTX load modifier doesn't matter.** `ld.volatile`, `ld.relaxed.gpu`,
   `ld.relaxed.cta`, `ld.weak`, default `ld.global` — all give roughly the
   same drain cost when followed by `fence.acquire.gpu`. The compiler
   uniformly upgrades them all to `LDG.E.STRONG.{SYS|GPU|SM}` (scope inherited
   from PTX modifier) precisely so that the acquire fence can correctly drain
   them.

4. **The load tier dominates the cost** (when load is followed by fence):
   - L1-hit drain: ~22 cy
   - L2-hit drain: ~80 cy
   - DRAM-cold drain: ~975-1042 cy

5. **Multiple loads pipeline.** 32 DRAM loads cost only ~340 cy of CCTL drain,
   not 32×975. The drain-wait is bounded by the slowest in-flight load, not
   the sum.

6. **Shadow-the-load with work to make CCTL free.** MODE 76 (1× weak L2 load,
   then 8 FFMA chain, then CCTL) = 2 cy. The FFMA chain absorbs the load
   latency, so by the time CCTL fires the load has retired and there's
   nothing to drain. The compiler emits plain `LDG.E` (no `.STRONG`) when no
   fence follows, allowing the load to pipeline with downstream work.

### SASS evidence

When `fence.acquire.gpu` follows a load:
```
LDG.E.STRONG.GPU R5, [...]    ← compiler-promoted from PTX `ld.weak`/`relaxed`/`volatile`
CS2R R8, SR_CLOCKLO            ← t0 (load may still be in flight)
CCTL.IVALL                     ← waits for LDG to retire, then invalidates
CS2R R10, SR_CLOCKLO           ← t1
```

When NO fence follows the load (MODE 76):
```
LDG.E R5, [...]                ← plain LDG, allowed to pipeline
FFMA × 8                       ← absorbs load latency
CS2R R10, SR_CLOCKLO           ← t0 (load done by now)
CCTL.IVALL                     ← nothing to drain → 2 cy
CS2R R12, SR_CLOCKLO           ← t1
```

### Practical implication

**Cost-model rule for `fence.acquire.gpu` on Blackwell:**

`cost ≈ max(2 cy, completion_latency_of_slowest_in_flight_global_load)`

If you want a cheap acquire fence, ensure all prior loads have retired (e.g., by
chaining their results into compute and giving compute enough time to drain).
The compiler will help by emitting plain `LDG.E` when no fence is present, but
once you place an acquire fence, all preceding loads get promoted to STRONG and
must drain before the fence completes.

### Catalog correction

`B300_PIPE_CATALOG.md` claims around CCTL.IVALL / fence.acquire.gpu cost should
be re-framed as "drain-wait for in-flight loads, not intrinsic CCTL cost."
The original "fence cost" measurements that didn't isolate the load drain were
conflating two effects.

Methodology takeaway: **always measure fences with both an empty-pipeline
baseline AND with controlled in-flight memory ops** to separate intrinsic
fence cost from drain wait.

---

## ADDENDUM 4 (2026-04-23): The acquire/release asymmetry — direct HW evidence

**Fence type dictates which in-flight memory ops are drained.** This is the
textbook C++ memory model mapped directly onto B300 silicon.

### Measurements (B300 sm_103a, `-lgc 1800`)

| Pattern                               | Fence type          | cy/iter |
|---------------------------------------|---------------------|--------:|
| CCTL alone (idle)                     | acquire             | 2.83    |
| 1× ld.volatile L2-hit + fence         | acquire             | 73.53   |
| 1× ld.volatile DRAM-cold + fence      | acquire             | 902.50  |
| **1× st.wb + fence**                  | **acquire**         | **9.27**|
| **8× st.wb + fence**                  | **acquire**         | **13.70**|
| 1× atom.add + fence                   | acquire             | 20.07   |
| 1× st.wb + fence                      | release             | 541.77  |
| 8× st.wb + fence                      | release             | 560.03  |

### Conclusion

- **Acquire fence (CCTL.IVALL)** drains prior **LOADS**, not stores.
  Stores before an acquire fence pass through at ~10-14 cy regardless of count.
- **Release fence (MEMBAR.ALL.GPU)** drains prior **STORES**, not loads.
  Stores cost ~540 cy at release time (drain to L2 + invalidate+visibility).
- **Load scope doesn't matter** (volatile/relaxed/weak all behave the same
  because the compiler promotes them all to `LDG.E.STRONG.X` before an
  acquire fence).
- **Store scope/hint doesn't matter** at this granularity (all `.wb` stores
  require release-time drain).

### C++/PTX memory-model mapping

| Semantic | Hardware op | Drains |
|----------|-------------|--------|
| `memory_order_acquire`  | CCTL.IVALL       | prior loads |
| `memory_order_release`  | MEMBAR.ALL.GPU   | prior stores |
| `memory_order_acq_rel`  | CCTL + MEMBAR    | prior loads AND stores |
| `memory_order_seq_cst`  | full fence       | both + global serialization |

This is consistent with the observed costs:
- `fence.acq_rel.gpu` = 272 cy (CCTL ~0 + MEMBAR ~186 + serialization)
- `fence.sc.gpu` = 272 cy (~same as acq_rel on idle)

### Practical rule of thumb for B300 kernels

1. **Reader-only kernels**: acquire fence costs ≈ L2 round-trip per outstanding
   load. Issue loads, do unrelated compute long enough to amortize, then fence.
2. **Writer-only kernels**: release fence costs ≈ L2 write-through time (~540
   cy for 1 store, ~560 for 8 — stores batch well). Batch stores before fence.
3. **Mixed**: use `fence.acq_rel.gpu` — it's only ~80 cy more than either
   individual fence on idle pipeline (272 vs 186 for release, 272 vs 2 for
   acquire).
4. **Cross-CTA coordination**: the fence cost in production is **dominated by
   memory drain**, not by the fence instruction itself. To minimize
   coordination latency, design your kernel to have loads/stores complete
   BEFORE the fence — e.g., insert compute that depends on the load result.

---

## ADDENDUM 5 (2026-04-23): CORRECTION to Addendum 4 — release IS a full fence

ADDENDUM 4 above claimed "release fence drains stores only". That is WRONG.
Further testing (MODES 90-103) shows:

### Updated cost matrix (B300 sm_103a, `-lgc 1800`, single-thread)

| Pattern                                  | cy/iter | Decomposition                |
|------------------------------------------|--------:|------------------------------|
| **Acquire fence (CCTL.IVALL)**           |         |                              |
| Idle pipeline                            |   ~3    | intrinsic                    |
| 1 LD volatile L2-hit + acquire.gpu       |   83    | drains load                  |
| 1 LD volatile DRAM + acquire.gpu         |  900+   | drains load (DRAM tier)      |
| **1 ST.wb + acquire.gpu**                | **9**   | **does NOT drain stores**    |
| 8 ST.wb + acquire.gpu                    |   14    | does NOT drain stores        |
| 1 atom.cas + acquire.gpu                 |  802    | drains atomic                |
| 1 LD volatile L2 + acquire.cta           |   ~85   | acquire still drains load    |
| 1 LD volatile L2 + acquire.sys           |   83    | acquire still drains load    |
| **Release fence (MEMBAR.ALL.GPU)**       |         |                              |
| Idle pipeline                            |  186    | intrinsic                    |
| FFMA chain (no memory) + release.gpu     |  186    | intrinsic only               |
| **1 LD volatile L1-hit + release.gpu**   | **202** | **DRAINS LD too** (~16 + 186)|
| **1 LD volatile L2-hit + release.gpu**   | **557** | **DRAINS LD too** (~370+186) |
| **1 LD volatile DRAM + release.gpu**     | **1060**| **DRAINS LD too** (~875+186) |
| 1 ST L2 + release.gpu                    |  542    | drains ST + 186              |
| 1 ST DRAM + release.gpu                  |  759    | drains ST DRAM + 186         |
| 8 ST L2 + release.gpu                    |  560    | stores batch                 |
| 1 LD DRAM + 1 ST DRAM + acq_rel.gpu      |  1167   | both drain                   |
| **CTA-scope fences**                     |         |                              |
| 1 LD vol L2 + release.cta                |   9     | LD pipelined past fence      |
| 1 ST + release.cta                       |   12    | cheap CTA-local              |
| 1 LD vol L2 + acquire.cta                |   85    | drain like acquire.gpu       |
| 1 ST + release.sys                       |  1663   | NVLink visibility            |

### Refined cost model

The fence type — **acquire vs release vs full** — and its **scope** —
**CTA / GPU / SYS** — determine which in-flight memory ops must drain:

```
┌────────────────────┬─────────────────────────────────────────────┐
│ Fence              │ Drains                                      │
├────────────────────┼─────────────────────────────────────────────┤
│ acquire.cta        │ Loads (compiler-orders via STRONG)          │
│ acquire.gpu        │ Loads (CCTL.IVALL invalidates L1 + drain)   │
│ acquire.sys        │ Loads + system-scope ordering (~same drain) │
│ release.cta        │ Stores (cheap, no GPU/SYS visibility)       │
│ release.gpu        │ ALL: loads AND stores (full GPU-scope fence)│
│ release.sys        │ ALL + system-scope visibility (~1663 cy)    │
│ acq_rel.gpu        │ ALL (same as release.gpu, +~80 cy)          │
│ sc.gpu             │ ALL + sequential-consistency serialization  │
└────────────────────┴─────────────────────────────────────────────┘
```

### Why release.gpu is a full fence

`fence.release.gpu` compiles to `MEMBAR.ALL.GPU` — a **full GPU-scope barrier**.
For GPU-scope visibility, all prior memory ops (loads AND stores) must reach a
consistent state in L2 before subsequent ops can proceed. So the fence drains
both. The intrinsic ~186 cy is the L2-coordination cost for the barrier; on top
of that, you pay drain wait for any in-flight load/store.

`fence.acquire.gpu` is asymmetric because **CCTL.IVALL** is just an L1
invalidation primitive — it doesn't broadcast to L2 or coordinate stores. The
~3 cy intrinsic is just the L1 invalidation; the only drain is for in-flight
loads (so subsequent loads see post-invalidation state correctly).

### Practical implications

1. **For lock-free queue dequeue (acquire-only)**: use `fence.acquire.gpu`
   = ~3 cy + load drain. This is genuinely the cheapest cross-CTA acquire.

2. **For lock-free queue enqueue (release-only)**: use `fence.release.gpu`
   = 186 cy + store drain. The 186 cy is unavoidable.

3. **For RMW (acq_rel)**: 272 cy on idle, plus drains for both. Can't beat this
   with separate acquire+release because they don't compose well.

4. **For CTA-local coordination**: use `.cta` scope. Release.cta = 9-12 cy is
   essentially free; acquire.cta still drains loads (~83 cy at L2-hit) so it
   isn't free, but it skips the L2-broadcast.

5. **For inter-GPU (NVLink)**: release.sys = 1663 cy — this is the dominant
   cost for cross-GPU coordination, far more than the actual data transfer.

### Catalog impact

`B300_PIPE_CATALOG.md` "fence cost on idle pipeline" measurements need to be
recalibrated against this matrix. The earlier "186 cy MEMBAR.ALL.GPU on idle"
measurement IS correct for the intrinsic. The earlier "315 cy" measurements
that mixed loads with fences should be re-attributed to drain-wait, not fence
overhead.

---

## ADDENDUM 6 (2026-04-23): Warp-width scaling of drain cost

| MODE | 1 thread | 32 threads (warp) | Notes |
|------|---------:|------------------:|-------|
| CCTL idle (no load)             |   2.83 |   2.83 | per-warp constant |
| LD L2-hit + acquire.gpu         |     86 |  **873** | drain scales with #loads |
| LD DRAM-cold + acquire.gpu      |    913 |    764 | (rand addr, similar tail latency) |

**Key finding**: `fence.acquire.gpu` drain cost depends on the **number of
in-flight loads**, not just on the slowest one. A full warp issuing 32 distinct
L2-hit loads pays ~10× the cost of a single thread doing 1 load (873 cy vs 86
cy), because the L2 port serializes the 32 line requests.

For DRAM-cold the per-load latency dominates and warp-wide tail latency
roughly matches single-thread tail (variance only). 32 cold-DRAM loads pay
~764 cy, similar to 1 cold load — the misses pipeline through HBM.

**Practical implication**: a coordination kernel that issues one acquire fence
per warp pays for 32 load drains, not 1. To minimize drain cost:
1. Have only one thread per warp do the load
2. Use SHFL/cluster-shared to broadcast the result
3. Then issue the acquire fence (1-thread cost ≈ 86 cy at L2-hit)

This explains why some classic lock-free patterns (CAS-based queue) can be
surprisingly slow at warp granularity — the implicit drain cost is real.

---

## ADDENDUM 7 (2026-04-23): Verifying the wait is IN CCTL (not in LDG scoreboard)

Valid concern raised: **maybe LDG.E.STRONG.SYS blocks CS2R t0 on its own via a
hardware scoreboard, so the 900 cy "CCTL cost" is actually just the LDG drain
naturally backpressuring the clock read?**

### Decisive tests (MODES 110-113)

| MODE | Sequence                                              | cy/iter |
|------|-------------------------------------------------------|--------:|
| 44   | (idle) CCTL + clocks                                  |   2.83  |
| 51   | LDG.volatile DRAM + `t0` + CCTL + `t1`                |  983    |
| 110  | LDG.volatile DRAM + `t0` + `t1` (NO CCTL)             | **2.27**|
| 111  | LDG.volatile DRAM + `t0` + 8×IADD + `t1` (NO CCTL)    |   2.43  |
| 112  | LDG + 1-deep SHFL(loaded) + `t0` + CCTL + `t1`        | 958     |
| 113  | LDG + 32-deep IMUL chain on `loaded` + `t0` + CCTL + `t1` | **2.00** |

### SASS verification

MODE 110 (no fence) and MODE 51 (with fence) BOTH emit **identical** `LDG.E.STRONG.SYS`
(because `ld.volatile.global` forces STRONG.SYS regardless of what follows).

```
MODE 110:  LDG.E.STRONG.SYS R4 → CS2R t0 → CS2R t1              (2.27 cy)
MODE 51:   LDG.E.STRONG.SYS R4 → CS2R t0 → CCTL.IVALL → CS2R t1 (983 cy)
```

The ONLY difference between the two SASS bodies is the presence of CCTL.IVALL
in the timed window. The LDG's scoreboard barrier attaches to R4 (the
destination register); CS2R reads SR_CLOCKLO and writes R2 — no dependency on
R4, so CS2R can issue immediately.

### Why CCTL specifically waits

CCTL.IVALL is NOT a scoreboard-dependent instruction — it's a semantic
memory-model op that must drain in-flight STRONG loads before invalidating L1.
This drain requirement is enforced at the MMU/LSU level, not via standard
register-scoreboard wait-masks.

**This is the fence's acquire ordering enforced in HW**: if a STRONG load
is in flight when you invalidate L1, the load could end up writing stale
data into the post-invalidation L1. The HW prevents this by blocking CCTL
until the STRONG load retires (either writes-back into L1 OR gets canceled).

### Conclusion

The 900 cy cost is **definitively from CCTL**, not from CS2R stalling on
LDG's scoreboard. Three independent proofs:

1. **Strip the fence** (MODE 110): the exact same LDG.E.STRONG.SYS drops cost
   to 2.27 cy — the load doesn't naturally block following CS2R.
2. **Drain load before the fence** (MODE 113): 32-deep IMUL chain on `loaded`
   forces the load to retire BEFORE t0 → CCTL sees no in-flight load → 2 cy.
3. **Keep the fence, don't drain** (MODE 51/112): 900+ cy — CCTL waits for
   the STRONG load's retirement before invalidating.

The "shadow load with compute" trick from ADDENDUM 3 is validated again: if
you arrange for the load to retire BEFORE reaching the fence, the fence is
free. If the load is still in flight at fence time, you pay the full drain.

---

## ADDENDUM 8 (2026-04-23): CCTL drain is PER-WARP (not per-SM/CTA)

User-suggested test: have warp 0 issue many in-flight DRAM loads, then have
warp 1 time a fence. If fence drain is SM-scope, warp 1's fence cost should
depend on warp 0's load state. If per-warp, warp 1 fences cheaply.

### Test setup (`tests/bench_cctl_crosswarp.cu`)

- 1 CTA, 64 threads (2 warps)
- Warp 0: issues 32 ld.volatile.global (DRAM-cold, 4 KB-strided) — guaranteed
  many in-flight loads
- Warp 1: `nanosleep(N_ns)` + timed `fence.acquire.gpu`
- No __syncthreads between (to avoid memory barrier confound)
- N_ns swept from 0 to 50,000 ns

### Results

| SLEEP_NS | Warp 1 fence cost (MODE 0, cross-warp) | MODE 2 (warp 1 own load) |
|---------:|---------------------------------------:|-------------------------:|
|       0  |                                 2.0 cy | 1001.6 cy |
|      10  |                                 2.0 cy | — |
|     100  |                                 2.0 cy | 982.7 cy |
|    1000  |                                 2.0 cy | 1000.2 cy |
|   10000  |                                 2.0 cy | 824.8 cy |
|   50000  |                                 2.0 cy | 830.9 cy |

**Warp 1's `fence.acquire.gpu` is 2 cy regardless of warp 0's load state.**
But when warp 1 has ITS OWN in-flight load (MODE 2), it pays full drain (~1000 cy).

### Conclusion

**CCTL.IVALL drain is SCOPED TO THE EXECUTING WARP.** The fence waits only for
the issuing warp's in-flight strong-ordered loads, not for other warps' loads
on the same SM.

### Implications

1. **Cross-warp coordination via acquire fence is cheap (2 cy per warp)** —
   each warp independently observes its own ordering.

2. **L1 invalidation still happens SM-wide** — CCTL.IVALL flushes L1 for all
   warps on the SM. But the DRAIN-WAIT is per-warp.

3. **Cooperative producer-consumer within a CTA**: if producer warp writes and
   consumer warp acquire-fences, the consumer sees the producer's updates
   (because L1 was invalidated) but doesn't pay drain cost for the producer's
   in-flight stores.

4. **This design makes sense from a HW perspective**: the acquire-drain is a
   property of the warp's scoreboard / in-flight-load tracker, which is
   per-SMSP. L1 is shared across all SMSPs on the SM, so invalidation is
   broadcast, but the "wait for in-flight" guarantee is local.

### Relation to C++ memory model

C++ `memory_order_acquire` is defined PER-THREAD: thread T's acquire synchronizes
with other threads' releases through atomics. It doesn't promise to wait for
other threads' in-flight operations to complete.

The B300 hardware realization matches: CCTL.IVALL invalidates L1 (enabling
subsequent loads to see other warps' writes, which already went to L2) and
drains the executing warp's in-flight loads (so the warp's own ordering is
preserved). Other warps' in-flight ops are irrelevant to this warp's acquire
ordering.

### Related question worth testing

If we had **cluster-scope** fences (`fence.acquire.cluster`), would drain
reach across SMs in the same cluster? Not tested here.

---

## ADDENDUM 9 (2026-04-23): Complete fence scope hierarchy on B300

**Test setup**: cross-warp drain probe (`tests/bench_cctl_crosswarp.cu`) and
cross-CTA drain probe (`tests/bench_cctl_crossCTA.cu`). Warp/CTA 0 issues many
in-flight DRAM ops; warp/CTA 1 nanosleeps then times a fence.

### Drain scope matrix

| Fence | Cross-warp same CTA | Cross-CTA | Drain scope |
|-------|--------------------:|----------:|-------------|
| acquire.cta | 2 cy | 2 cy | **PER-WARP** |
| acquire.gpu | 2 cy | 2 cy | **PER-WARP** |
| acquire.sys | 2 cy (presumed) | 2 cy (presumed) | per-warp |
| **release.cta** | **9 cy** | 9 cy | **PER-WARP** (CTA visibility only) |
| **release.gpu** | **179-1546 cy** (drains) | 179 cy (intrinsic only) | **CTA-WIDE** drain |
| sc.gpu | 265-1050 cy (drains) | 265 cy (intrinsic only) | CTA-WIDE drain |
| release.sys | 1663-2249 cy (drains) | 1663 cy intrinsic | CTA-wide + NVLink |

### Decoded interpretation

**The PTX scope (`.cta` / `.gpu` / `.sys`) controls TWO things separately:**

1. **VISIBILITY scope**: which threads will observe ordering after the fence
   (cta / cluster / gpu / sys).
2. **DRAIN scope**: which in-flight memory ops the fence must wait for.

The two are NOT the same!

| Fence form | Visibility | Drain scope | Intrinsic cost |
|------------|------------|-------------|---------------:|
| acquire.cta | within CTA  | per-warp     | 2 cy |
| acquire.gpu | GPU-wide    | per-warp     | 2 cy |
| release.cta | within CTA  | per-warp     | 9 cy |
| release.gpu | GPU-wide    | **CTA-wide** | 179 cy |
| sc.gpu      | GPU-wide    | CTA-wide     | 265 cy |
| release.sys | system      | CTA-wide + NVLink | 1663 cy |

### Why this asymmetry exists

**Acquire** synchronizes against PRIOR releases. The HW only needs to ensure
THIS warp's view is consistent (invalidate L1 + drain THIS warp's loads). It
doesn't care about other warps' state because acquire is a per-thread guarantee.

**Release.cta**: makes THIS warp's prior writes visible to other warps in the
CTA. Since CTA-scope visibility doesn't require crossing the L2 boundary, no
wait for other ops. Other warps will see this warp's writes as soon as they
reach the SM-shared cache.

**Release.gpu**: makes prior writes visible to ALL GPU threads. This requires
**flushing the SM's write-combining buffer to L2**. To preserve store ordering
across the boundary, the HW must drain ALL in-flight stores from the CTA (not
just this warp), because L2 must observe them in program order. Hence
CTA-wide drain.

**Release.sys**: same as gpu + must hit L2 in a way visible to NVLink peers,
adding ~1500 cy.

### Practical consequences

1. **Ultra-cheap intra-warp synchronization** with `release.cta` + `acquire.cta`
   = 9 + 2 = 11 cy total, fully isolated per-warp.
2. **Cheap cross-warp synchronization within a CTA** with `release.gpu` (179 cy)
   + `acquire.gpu` (2 cy) = 181 cy. The release pays CTA-wide drain only the
   first time after writes; subsequent fences on already-drained state are 179 cy
   (intrinsic).
3. **Cross-CTA synchronization** also uses `release.gpu` + `acquire.gpu` but
   the release.gpu in CTA X does NOT block on CTA Y's in-flight ops. Each CTA
   pays only for its own drain.
4. **NVLink cross-GPU sync** (`release.sys`) = 1663 cy intrinsic; an order of
   magnitude more expensive. Hide behind compute.

### Updated cost-model formula

```
fence cost = intrinsic + drain_wait

intrinsic:
  acquire.{cta,gpu}     ≈ 2 cy
  release.cta           ≈ 9 cy
  release.gpu           ≈ 179 cy
  sc.gpu                ≈ 265 cy
  release.sys           ≈ 1663 cy
  acq_rel.gpu           ≈ 272 cy

drain_wait (depends on scope):
  acquire.*             ≈ max(0, completion_latency_of_THIS_WARP's slowest in-flight load)
  release.cta           ≈ max(0, completion_latency of THIS WARP's stores)
  release.gpu           ≈ max(0, completion_latency of ALL CTA's stores+loads)
  release.sys           ≈ same as release.gpu + NVLink
```

---

## ADDENDUM 10 (2026-04-23): release.gpu drain is SM-WIDE (not CTA-local)

ADDENDUM 9's cross-CTA test (`bench_cctl_crossCTA.cu`) showed release.gpu = 179
cy regardless of other CTA's activity. I tentatively concluded "drain scope is
CTA-LOCAL". WRONG.

The reason: with only 2 CTAs and 148 SMs, the scheduler placed CTA 0 on SM 142
and CTA 1 on SM 143 (different SMs). So CTA 1's release.gpu had no other-CTA
in-flight ops to drain on its own SM.

### Correct test: force same-SM via high occupancy

`tests/bench_cctl_sameSM.cu`: launch 296 CTAs (2× SM count) with
`__launch_bounds__(64, 8)`. Use atomicCAS races to elect 2 CTAs per SM into
"loader" and "fencer" roles. With 8 CTAs/SM and 296 CTAs, every SM gets at
least 2 CTAs participating, guaranteeing a same-SM pair on every SM.

### Same-SM cross-CTA results (148 SMs measured per condition)

| Fence | SLEEP=0 | SLEEP=10000 | SLEEP=50000 |
|-------|--------:|------------:|------------:|
| acquire.gpu | 2 | 2 | 2 |
| **release.gpu** | **avg 952 (805-1088)** | **avg 273 (255-295)** | **avg 214 (208-227)** |
| sc.gpu | avg 1049 (923-1167) | avg 360 (341-391) | avg 300 (294-313) |
| acquire.cta | 2 | 2 | 2 |
| release.cta | 9 | 9 | 9 |

### Refined drain scope hierarchy (CORRECTED)

| Fence | Drain scope | Intrinsic |
|-------|-------------|----------:|
| acquire.{cta,gpu,sys} | **PER-WARP** | 2-3 cy |
| release.cta | **PER-WARP** | 9 cy |
| release.gpu | **SM-WIDE** (all CTAs + warps on SM) | 179 cy |
| sc.gpu | SM-WIDE | 265 cy |
| release.sys | SM-WIDE + NVLink | 1663 cy |

### Why SM-wide (not CTA-local)

The SM's L2-bound write-combining buffer is shared across all warps and CTAs
co-resident on the SM. `release.gpu` must flush this buffer to L2 (so the
writes become globally visible). To preserve store ordering, the buffer must
drain in program order — and that buffer holds writes from ALL CTAs/warps on
the SM. Hence the drain is SM-wide.

`acquire.gpu` only invalidates L1 (per-SM operation, broadcast to all warps)
and drains the executing warp's load tracker. Other warps/CTAs' load trackers
are untouched, so drain is per-warp.

`release.cta` only requires CTA-internal visibility. Since same-CTA threads
share L1 and SM-shared registers/smem, no L2-flush is needed. The HW just
ensures local ordering for THIS warp's writes — drain is per-warp.

### Updated cost-model formula (SUPERSEDES ADDENDUM 9)

```
fence cost = intrinsic + drain_wait

intrinsic:
  acquire.{cta,gpu,sys} ≈ 2-3 cy
  release.cta           ≈ 9 cy
  release.gpu           ≈ 179 cy
  sc.gpu                ≈ 265 cy
  release.sys           ≈ 1663 cy

drain_wait (depends on the fence's drain scope):
  acquire.*             ≈ max(0, completion_latency_of_THIS_WARP's slowest in-flight load)
  release.cta           ≈ max(0, completion_latency of THIS WARP's stores)
  release.gpu           ≈ max(0, completion_latency of ALL SM's in-flight stores+loads)
  release.sys           ≈ as release.gpu + NVLink visibility
```

### Practical impact

This is a CRITICAL distinction for designing kernels:

1. **A single CTA on an SM** has minimal release.gpu drain (only its own ops).
2. **Multiple CTAs co-resident on an SM** see release.gpu drain ALL of them,
   not just the issuing CTA. This is a hidden coordination cost.
3. **High-occupancy kernels with frequent release.gpu** will pay the
   compounded drain cost across all co-resident CTAs.
4. **To avoid this**: use release.cta when only CTA-internal visibility is
   needed. Or design the kernel so writes are batched and release.gpu happens
   infrequently.
5. **Cross-CTA on different SMs**: release.gpu drain stays local (SM-wide
   means within each SM's own buffer). Different SMs don't drain each other.

### Verification: SM IDs in original cross-CTA test

In `bench_cctl_crossCTA.cu` with -b 2 -t 64, the scheduler placed CTAs on:
- run 1-5 (consistent): CTA0_smid=142, CTA1_smid=143

Different SMs every time, hence the misleading "179 cy CTA-local" result.
The same-SM test definitively shows it's SM-wide drain.

---

## ADDENDUM 11 (2026-04-23): Drain cost scales with co-resident CTAs (and STORES are 4-5× worse than LOADS)

`tests/bench_cctl_sameSM_scale.cu` and `bench_cctl_sameSM_stores.cu` measure
how `release.gpu` cost scales with the number of co-resident CTAs that have
in-flight memory ops on the same SM.

### Setup

- 1184 CTAs launched, `__launch_bounds__(64, 8)` → 8 CTAs/SM (148 SMs × 8 = 1184)
- atomicCAS-based pair election: per SM, claim N_LOADERS as loader CTAs and 1 as fencer
- Each loader continuously issues 32 DRAM-cold ops per outer iter
- Fencer measures `fence.release.gpu` cost in 50 outer iters (averaged across 148 SMs)

### Results: release.gpu drain cost vs co-resident loaders

#### Loaders are LOAD CTAs (volatile DRAM loads)

| N_LOAD_CTAs | SLEEP=0 cy | SLEEP=50µs cy | × intrinsic |
|-------------|-----------:|--------------:|------------:|
| 0 (intrinsic) | 185 | 185 | 1× |
| 1 | 958 | 215 | 5.2× |
| 2 | 1413 | 220 | 7.6× |
| 4 | 2180 | 269 | 11.8× |
| 7 | 2580 | 325 | 13.9× |

#### Loaders are STORE CTAs (writeback stores)

| N_STORE_CTAs | SLEEP=0 cy | SLEEP=50µs cy | × intrinsic |
|--------------|-----------:|--------------:|------------:|
| 0 | 185 | 185 | 1× |
| 1 | 1778 | 253 | 9.6× |
| 2 | 3242 | 354 | 17.5× |
| 4 | 3974 | 491 | 21.5× |
| 7 | **11,470** | 1627 | **62×** |

### Key findings

1. **Drain scales with co-resident activity, sub-linearly for LOADS but
   super-linearly for STORES at high CTA counts.**
2. **Stores cost ~4-5× more to drain than loads at high CTA counts**
   (7 store-CTAs → 11,470 cy; 7 load-CTAs → 2,580 cy).
3. **At high SM occupancy, release.gpu can cost 10,000+ cycles** (~6 µs at 1.8
   GHz). The catalog's "186 cy intrinsic" massively understates real-world
   coordination cost in high-occupancy kernels.
4. **SLEEP between issue and fence helps** (215-1627 cy at 50µs sleep), but
   fundamentally the drain cost is bounded by what's still in flight when the
   fence fires.

### Why stores are worse than loads

- **Loads** return one cache line per request; the L2 returns data quickly
  through parallel banks; the fence waits for the slowest in-flight read.
  L2 has many parallel serving paths (~20+ partitions on B300).
- **Stores** must COMMIT to L2 in program order. Each store goes through:
  1. SM's write-combining buffer (per-SM, shared across CTAs)
  2. L2-bound network
  3. L2 cache update + coherence broadcast (if any subscribers)
- Each step is sequential per address; the buffer must drain in order. With
  7 CTAs × 32 threads × 32 stores = 7168 stores in flight, the buffer must
  process all of them in order before release.gpu completes.

### Practical implications

For high-throughput kernels with frequent release.gpu fences:

1. **Reduce occupancy if release.gpu is in hot loop**: lower CTAs/SM = less
   drain compounding. Counter to usual "more occupancy is better" wisdom.
2. **Batch writes between fences**: 100 stores + 1 fence costs roughly the
   same as 1 store + 1 fence (because batching maximizes parallelism in the
   write buffer). Per-write fence cost approaches intrinsic 186 cy.
3. **Use release.cta where possible**: 9 cy intrinsic, no SM-wide drain.
   Only use release.gpu when cross-CTA visibility is required.
4. **Producer-consumer queues**: producer's release.gpu cost depends on what
   ALL co-resident CTAs are doing. For predictable cost, use dedicated
   producer warps/CTAs that aren't doing other stores.
5. **Avoid release.gpu in inner loops** of high-occupancy kernels.

### Updated cost-model formula (FINAL)

```
fence_cost ≈ intrinsic + drain_wait

intrinsic (idle pipeline):
  acquire.{cta,gpu,sys}: 2-3 cy
  release.cta:           9 cy
  release.gpu:           186 cy
  sc.gpu:                265 cy
  release.sys:           1663 cy

drain_wait scope:
  acquire.*:    THIS WARP's in-flight loads only
  release.cta:  THIS WARP's in-flight stores only
  release.gpu:  ALL co-resident CTAs' in-flight loads AND stores on the SM

drain_wait magnitude (rule of thumb at high occupancy):
  release.gpu + N CTAs each with K in-flight loads ≈ 200 + N × 350 cy
  release.gpu + N CTAs each with K in-flight stores ≈ 200 + N × 1500 cy
                                                       (super-linear at N>2)
```

---

## ADDENDUM 12 (2026-04-23): cp.async semantically bypasses acquire fence drain

`cp.async` (asynchronous global→shared copy, Hopper+) has its own memory
ordering namespace via `commit_group` / `wait_group`. We tested whether fences
respect this separation.

### Test (`tests/bench_cctl_cpasync.cu`, single thread, `-lgc 1800`)

| MODE | Pattern (16 cp.async + fence) | SLEEP=0 cy | SLEEP=10µs cy |
|------|-------------------------------|-----------:|--------------:|
| 0 | cp.async (no commit) + acquire.gpu  | **2.1** | 2.0 |
| 1 | cp.async + commit_group + acquire.gpu | **878** | 2.0 |
| 2 | cp.async + commit_group + wait_group 0 + acquire.gpu | 2.6 | 2.0 |
| 3 | cp.async BIG WS DRAM (no commit) + acquire.gpu | 4.1 | 2.0 |
| 4 | cp.async (no commit) + release.gpu | **928** | 633 |
| 5 | cp.async (no commit) + acquire.cta | 2.1 | 2.0 |

### Key findings

1. **Acquire fence does NOT drain uncommitted cp.async** (MODE 0, 3, 5: ~2 cy).
   The compiler/hardware tracks cp.async as a separate ordering namespace until
   it's committed via `cp.async.commit_group`.
2. **Once committed, acquire fence DOES drain it** (MODE 1: 878 cy = same
   ballpark as a regular ld.volatile DRAM + acquire).
3. **Release fence ALWAYS drains cp.async** even without commit (MODE 4: 928 cy).
   Release.gpu is a full-pipeline barrier — it doesn't care about cp.async
   commit semantics.
4. **Explicit drain via wait_group resets the state** (MODE 2: 2.6 cy after
   wait_group 0 because the cp.async is fully drained before fence fires).

### Why this matters

This is a powerful tool for coordination protocols. The classic acquire/release
pair has expensive drain costs (see ADDENDUMs 3-11). With cp.async:

- **Producer**: `cp.async.cg.shared.global ...` (fire-and-forget) → `cp.async.commit_group` → `fence.release.gpu` (drains).
  - Consumer can see the writes after the release fence (because release is
    full-pipeline). Cost = release.gpu drain (~900 cy).
- **Consumer**: `fence.acquire.gpu` (cheap if no other in-flight loads) →
  `cp.async.wait_group N` (waits for producer's data to land) → use data.
  - The acquire fence here is essentially free (2 cy) because cp.async is
    handled by wait_group, not the fence.

This is fundamentally cheaper than the equivalent with synchronous loads:
- Old way: producer release.gpu (~900 cy drain) + consumer acquire.gpu + load
  (~900 cy drain) = 1800+ cy.
- New way: producer release.gpu (~900 cy drain) + consumer acquire.gpu (2 cy) +
  cp.async.wait (whatever it takes to land) = 902+ cy.

The savings come from: cp.async's wait_group is cheaper than fence drain
because it's tracked at the cp.async unit level, not at the global memory
ordering level.

### SASS check (likely)

cp.async maps to `LDGSTS` instruction (load global, store shared) with various
qualifiers. `cp.async.commit_group` maps to `CCTL` or similar commit primitive.
The "in-flight" set for fences is the LDGSTS scoreboard, which only includes
committed groups.

### Practical recipes

1. **Bulk-copy + dependent compute**: use cp.async + wait_group; skip fences entirely.
2. **Cross-CTA producer/consumer**: producer uses release.gpu (drains everything);
   consumer uses acquire.gpu (cheap) + wait_group (waits only for cp.async).
3. **Avoid `cp.async.commit_group` if you don't need acquire-visibility**: keep
   the cp.async out-of-band for cheaper acquire fences.
4. **TMA (`cp.async.bulk.tensor`)**: not tested here, but likely follows same
   pattern with mbarrier instead of commit_group.

### Caveat

The "free acquire fence" trick only works if you ACTUALLY don't care about
acquire-visibility for the cp.async. If you need other threads to see the
cp.async result via acquire fence semantics, you must commit the group first,
and pay the drain cost.

---

## ADDENDUM 13 (2026-04-23): Cluster fences are syntactic sugar for GPU fences on B300

PTX provides `.cluster` scope for fences (e.g., `fence.acquire.cluster`,
`fence.release.cluster`). On Hopper/Blackwell, clusters are groups of
co-resident CTAs on the same GPC with cluster-shared memory.

### Test (`tests/bench_cctl_cluster.cu`)

| Fence | Idle cy | SASS |
|-------|--------:|------|
| acquire.cluster | 2 | (NOP / scoreboard-only) |
| release.cluster | **179** | **MEMBAR.ALL.GPU** |

### Conclusion

`fence.release.cluster` compiles to the EXACT SAME `MEMBAR.ALL.GPU` SASS
instruction as `fence.release.gpu`. There's no cluster-specific opcode on B300.

**This is consistent with B300's memory architecture**: there's no
cluster-private cache or write buffer. The L2 is GPU-shared. So any store
that needs visibility outside its CTA must go through L2 — at which point it's
visible to the entire GPU. Cluster scope vs GPU scope is a distinction without
a difference for store-visibility.

The `.cluster` PTX form exists for forward compatibility (in case future
architectures add cluster-private caches) and for cluster-shared memory
coherence which is handled by the L1.cluster cache extension.

For practical purposes on B300, treat `fence.{acquire,release}.cluster` as
identical to `fence.{acquire,release}.gpu` in cost and behavior.

---

## ADDENDUM 14 (2026-04-23): L1 invalidation scope is SM-WIDE (CCTL.IVALL invalidates ALL CTAs' L1)

User question: when CTA X executes `fence.acquire.gpu` (CCTL.IVALL), does it
invalidate L1 only for CTA X, or for ALL co-resident CTAs on the SM?

### Test (`tests/bench_cctl_l1_scope.cu`)

- 1184 CTAs, `__launch_bounds__(64, 8)` → 8 CTAs/SM
- atomicCAS pair-election per SM: warmer + invalidator
- Warmer fills L1 with 32 cache lines (4 KB) via `ld.global.ca`
- Handshake via global volatile flag
- Invalidator runs `fence.acquire.gpu` → signals back
- Warmer re-reads same lines, measures latency
- L1-hit ≈ 6 cy/load; L2-hit ≈ 14-15 cy/load

### Results

| MODE | Description | cy/load | Conclusion |
|------|-------------|--------:|------------|
| 0 | baseline (no CCTL between fill & re-read) | **6.38** | warmer's L1 fully primed |
| 1 | OTHER CTA on SAME SM does CCTL | **10.47** (range 7.3-27.9) | bimodal — L1 partially evicted |
| 2 | warmer ITSELF does CCTL | **14.30** | full L1 invalidation, L2-hit refill |

### Interpretation

CCTL.IVALL DOES affect other CTAs' L1 on the same SM (10.47 cy avg vs 6.38
baseline) but the effect is timing-dependent (range 7-28 cy). The bimodal
pattern suggests:

- **CCTL.IVALL is SM-wide L1 invalidation** at the moment of execution
- BUT: the warmer's NEXT load may refill some lines BEFORE the warmer
  re-reads, so by the time the warmer's measured re-read fires, the L1
  state is partially recovered.

The ~14.3 cy for self-CCTL is the maximum effect (full L1 miss). The 10.47
cross-CTA average is in between because:
- Some lines got evicted and missed (~14 cy)
- Some lines were refilled by other CTA activity in the interim (~6 cy)
- Average ≈ 10 cy

### Key takeaway

**L1 cache is shared SM-wide. CCTL.IVALL invalidates the whole L1.**
This means co-resident CTAs DO interfere with each other's L1 state via
acquire fences. A frequent acquire-fencer in one CTA degrades L1 hit rate
for other CTAs on the same SM.

This is consistent with B300 architecture: L1 is a single cache per SM,
shared by all warps and CTAs on that SM. There's no per-CTA tagging that
would allow selective invalidation.

---

## ADDENDUM 15 (2026-04-23): L1 carveout effects (response to user question)

User question: how does the L1/SMEM carveout configuration affect all the
prior CCTL/fence findings?

### L1 effective size measured (`tests/bench_l1_size_probe.cu`)

Sweep working-set size and find latency cliff (L1-hit ~7 cy → L2-hit ~14.7 cy):

| Carveout setting | L1 cliff at WS | Effective L1 size |
|------------------|---------------:|------------------:|
| smem=0 KB (max L1)   | between 196 and 256 KB | **~200-228 KB** |
| smem=200 KB (min L1) | between 32 and 48 KB   | **~48 KB** |

This confirms B300 has ~256 KB total L1+SMEM combined (matches Hopper-class
spec). Carveout works as documented.

### Effect on prior fence/CCTL findings

For our earlier microbenchmarks (4 KB working set), the carveout has NO
significant effect because the workload fits in L1 at any carveout:

| Test | smem=0KB | smem=100KB | smem=200KB |
|------|---------:|-----------:|-----------:|
| L1-hit baseline (cy/load) | 6.38 | 6.39 | 6.38 |
| smem stride-2 bank-conflict | 6.88 | 6.88 | 6.88 |
| CCTL drain (1 DRAM load + acquire) | 1041 | 1055 | 964 (noise) |
| Cross-CTA CCTL effect | 10.37 | 10.47 | (test hung at this carveout due to occupancy collapse) |
| Self-CCTL re-read | 14.30 | 14.31 | — |

### When carveout DOES matter

1. **Workloads with WS > 32 KB**: small-L1 carveout starts forcing L2 hits.
2. **Workloads with WS > 196 KB**: even max-L1 carveout starts forcing L2 hits.
3. **High-occupancy kernels with 200 KB smem**: only ~1 CTA fits per SM. This
   is why the 200KB-smem test hung (CAS pair-election couldn't find 2 CTAs
   on same SM). Carveout is implicitly tied to occupancy via `__launch_bounds__`.

### Practical implications

1. For fence-cost characterization (small WS), the carveout is irrelevant.
2. For bank-conflict measurements (32 banks regardless), carveout is irrelevant.
3. For workloads that benefit from L1 caching at >48 KB, prefer carveout=0%
   (max L1, ~200 KB available).
4. For TMA/cp.async-heavy workloads that need large smem, accept the L1
   reduction.
5. **High-smem allocations reduce CTAs/SM** — so release.gpu drain (which
   scales with co-resident CTAs per ADDENDUM 11) is REDUCED at high smem
   carveout. Tradeoff: more L2 hits for compute, but cheaper inter-CTA fences.

---

## ADDENDUM 16 (2026-04-23): The bimodal cross-CTA L1 invalidation has propagation latency

ADDENDUM 14 found that cross-CTA CCTL gives 10.47 cy avg (vs 14.30 for self-CCTL).
This is "partial" invalidation. Probing further with `tests/bench_cctl_l1_scope_v2.cu`:

### Findings

| Variable swept | Result | Interpretation |
|----------------|--------|----------------|
| **N_CCTL** (back-to-back CCTLs by invalidator) | 10.65 (N=1) → 12.42 (N=32) cy | Repeating doesn't help much — single CCTL ≈ effective |
| **PRE_DELAY** (warmer nanosleep AFTER handshake, BEFORE re-read) | **10.71 (0ns) → 14.31 (5µs)** | **Saturates at FULL self-CCTL level** |
| **POST_DELAY** (invalidator nanosleep AFTER CCTL, BEFORE signaling) | 10.85 (0ns) → 9.39 (20µs) | Counterintuitive: more delay = fewer L1 misses |

### Critical insight: CCTL has propagation latency

The PRE_DELAY result is decisive: after a 5µs delay between the invalidator's
CCTL and the warmer's measurement, the cost reaches **14.31 cy = full self-CCTL
level**. This means CCTL.IVALL **DOES** fully invalidate L1 SM-wide — but the
invalidation takes a few microseconds to fully propagate.

Interpretation:
- At zero PRE_DELAY (10.71 cy avg), the warmer's measurement races the CCTL's
  propagation. Some lines are observed as still-cached (race-window L1-hit);
  others are invalidated (L2-hit). Bimodal average.
- At PRE_DELAY ≥ 5µs, CCTL fully completed → uniform 14.31 cy.

### Conclusion (updates ADDENDUM 14)

**`fence.acquire.gpu` (CCTL.IVALL) is SM-wide full L1 invalidation, with
propagation latency of a few microseconds before all lines are observably
invalid.**

This is consistent with:
- L1 invalidation is broadcast across all sets/ways via an internal queue
- The queue takes a few µs to drain
- A racy observer right after CCTL may see partial state

For practical purposes:
- **Acquire fence costs ~3 cy intrinsic + drain wait** (per ADDENDUMs 3-9)
- **The L1 invalidation effect on co-resident CTAs is full but with µs-scale propagation**
- A cross-CTA reader within the propagation window might still see cached data
  (race), but stable observation after the propagation completes shows all
  lines are invalidated.

### POST_DELAY counterintuition

The POST_DELAY trend (more delay → fewer L1 misses) is harder to explain. One
hypothesis: during POST_DELAY, the invalidator's spin-wait reads refill some
L1 lines, but only those that happen to map to the same L1 sets as warmer's
addresses. In any case the effect is small (10.85 → 9.39 cy) compared to
PRE_DELAY's saturation effect.

Worth noting: the invalidator's activity IS visible in L1 even though it's a
different CTA. SM-wide L1 sharing is symmetric.
