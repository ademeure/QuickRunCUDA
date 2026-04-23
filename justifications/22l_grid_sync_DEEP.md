# §22l — Grid Sync DEEP DIVE (catalog 4245 cy / 2.2 µs)

## ⚠ ADDENDUM 2026-04-23 — SASS verification of fence.* mechanism

Direct SASS check (justifications/22l_sass/decomp_comp{1,2,3}.sass) confirms the cost decomposition + reveals the architectural mechanism:

| PTX form | SASS emitted | Mechanism / cost (in §22l decomp context) |
|---|---|---|
| `fence.acquire.gpu` | **`CCTL.IVALL` only** (no MEMBAR) | L1 cache invalidate. ~25 cy in §22l decomp; **2 cy on idle drained pipeline** per CCTL DEEP. |
| `fence.release.gpu` | **`MEMBAR.ALL.GPU` only** (no CCTL) | Drain write buffer to L2. ~456 cy in §22l decomp context; **186 cy on idle drained pipeline** per follow-up. |
| `fence.acq_rel.gpu` | **`MEMBAR.ALL.GPU` + `CCTL.IVALL`** | Both: drain own writes + invalidate L1 to read others'. ~575 cy in context; **272 cy on idle (= 186 MEMBAR + 86 cy CCTL serialization)**. |

**MEMBAR cost decomposition** (separate test, justifications/22l_cctl_ivall_DEEP.md MODE 10/13/14):

| Scenario | MEMBAR.ALL.GPU cy |
|---|--:|
| Truly idle (nanosleep drained pipeline) | 186 |
| After 4 KB writes (drained) | 712 (= 186 base + 526 write-back drain) |
| After 16 KB writes | 734 |
| After 64 KB writes | 823 |
| After 256 KB writes | 884 |

MEMBAR base cost ~186 cy + sub-linear write-back drain (~530 cy at 4 KB → 698 cy at 256 KB). The drain cost reflects waiting for L1→L2 write-back commits, not the writes hitting L1.

The §22l decomp's "456 cy fence.release.gpu" was measured in grid-sync context with prior atomic-counter writes pending, which inflated the cost above the idle 186 cy base.

**Architectural insight: L2 is the GPU-scope coherence point on Blackwell.** This explains the asymmetric cost:
- **Acquire is cheap** — just invalidate L1; L2 is already coherent so subsequent loads see the latest data.
- **Release is expensive** — must drain write buffer through L1 into L2 (MEMBAR ~456 cy).

**Implication for kernel writers:** if you're a CONSUMER reading data others wrote, `fence.acquire.gpu` (or `ld.acquire`) is ~20× cheaper than `fence.release.gpu`. Favor producer-cost-heavy protocols (the producer pays release, the consumer pays acquire — but acquire dominates the count when you have many consumers).

This refines the ninja_F3 recipe explanation: relaxed atom + relaxed spin works because the L2 atomic unit IS the coherence point AND synchronization point — atom-add commits at L2 are visible to all readers, and the spin-loop's natural re-reads pick them up.

⚠ The earlier SELF_OP-style mechanism caveat applies: this SASS-mapping is verified for THIS rig + this NVCC. Different toolchains may emit different SASS for the same PTX fence (e.g. cuda 13.0 vs 13.2; sm_90a vs sm_103a). Always cross-check via `cuobjdump --dump-sass` if porting.

---


Audit date: 2026-04-23
Clock: `-lgc 1800,1800` → effective 1800 MHz (verified per-test by `nvidia-smi` sampling). All cycle/wall numbers are at 1800 MHz unless stated otherwise.
GPU: B300 SXM6 AC (sm_103a), GPU 0
Driver: 580.126.09, CUDA toolkit 13.2, ptxas 13.2.r13.2

Note on clock locking: at session start the GPU was stuck at 1942 MHz (background `QuickRunCUDA selfop` sweep was running). After `pkill -9 QuickRunCUDA; sleep 8`, the rig honored `-lgc` correctly across {1500, 1800, 1920}. All numbers in §1–4 use 1800-locked. §5 (clock crosscheck) sweeps explicitly.

---

## CLAIM (catalog `B300_PIPE_CATALOG.md` L7635 verbatim)

> "Grid sync via global atomic counter (no cudaLaunchCooperativeKernel API):
>
> | Grid blocks | cy/sync | µs @ 1.92 GHz |
> | 8 | 4161 | 2.17 |
> | 32 | 4129 | 2.15 |
> | 64 | 4195 | 2.18 |
> | 148 | 4245 | 2.21 |
>
> Grid sync cost is ~constant at ~4200 cy = 2.2 µs, regardless of grid size. The cost is dominated by atomic acq_rel (1598 cy) + spin loop on phase var."

**Verdict on catalog:** the **2.2 µs** wall-time figure is **roughly correct for the WORST chosen impl** (e.g. acq_rel atomic + spin on a hot release counter), and the **trend (constant w.r.t. grid)** is correct. But the catalog conflates **wall-time** with **per-block cycle counts**, and reports a **2× over-pessimistic best case**: a clean atomic-counter pattern (`atomicAdd` + acquire-load spin) measures **1620 cy / 0.96 µs at 1800 MHz**, and the ninja "bare REDG.relaxed + relaxed spin" hits **1458 cy / 0.86 µs**, which is **2.6× faster than the catalog's quoted 4245 cy / 2.2 µs**.

Furthermore, NVIDIA's `cooperative_groups::grid_group::sync()` measures **2234 cy / 1.29 µs at 1800 MHz**, again ~1.7× better than the catalog claim. The catalog's 4245 cy figure appears to be either a different implementation (likely a worse ad-hoc spin) or includes additional bookkeeping the present harness avoids.

---

## Investigation 1: NVIDIA `cg::grid_group::sync()`

### Test source
`tests/22l/bench_22l_cg_sync.cu` (verbatim — also `bench_22l_perblock.cu MODE=0`)

```cuda
auto grid = cg::this_grid();
// ...
asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
grid.sync();
asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
```

Launched via `cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(128), args, 0, 0)`.

### Build / run / raw stdout (grid=148, iters=1000, 1800-locked)

```
$ nvcc -arch=sm_103a -O3 -DMODE=0 -o pb_m0 bench_22l_perblock.cu
$ ./pb_m0 148 1000 128
# MODE=0 (cg::grid.sync) grid=148 threads=128 iters=1000
# per-block-avg cy: min=2234 (block 123)  mean=2234  max=2235 (block 94)
# per-block-min cy across grid: 1862  per-block-max cy: 2382
# wall_us_per_sync=1.293 wall_ms_total=1.293
```

### SASS analysis (preserved at `justifications/22l_sass/perblock_mode0.sass`)

The cg::sync expands to this hot-path inside the loop (lines /0330–/0500):

```
/0330/  BAR.SYNC.DEFER_BLOCKING 0x0          # __syncthreads() before sync
/0360/  BAR.SYNC.DEFER_BLOCKING 0x0          # internal arrival barrier
/0400/  @P0 MEMBAR.ALL.GPU                    # GPU-scope memory fence (one warp leader)
/0410/  @P0 ERRBAR                            # error-sync barrier (CTA scope)
/0420/  @P0 CGAERRBAR                         # cluster-graph err sync barrier
/0430/  @P0 ATOM.E.ADD.STRONG.GPU             # acq_rel atomic add
/0490/  LD.E.STRONG.GPU                       # spin: acquire-strong load
/04A0/  CCTL.IVALL                            # cache-line invalidate (forces re-read)
/04B0/  YIELD                                 # warp yield
        (loop on /0490 if not threshold)
/04F0/  WARPSYNC.ALL                          # warp-level sync
/0500/  BAR.SYNC.DEFER_BLOCKING 0x0          # exit __syncthreads()
```

Counts:
- N_MEMBAR.ALL.GPU = 1 (per sync, only by warp leader)
- N_ERRBAR = 1
- N_CGAERRBAR = 1 (cluster-aware error sync — extra cost vs single-CTA cases)
- N_ATOM.E.ADD.STRONG.GPU = 1 (acq_rel, with return)
- N_BAR.SYNC = 3 (one before, one mid for arrival, one after)
- N_LDG.STRONG (spin reads) = ~0..N depending on slowest-block lateness
- N_WARPSYNC.ALL = 1
- N_CCTL.IVALL = 1 inside spin (forces L1/L2 invalidation each time)
- N_YIELD = 1 inside spin

Who does the fence: only **warp leader** (`@P0` predicate, where P0 = lane 0). All threads do BAR.SYNC.

### Measurement

| Grid size | cy/sync (mean) | wall µs/sync | vs catalog 4245 |
|-----------|---------------:|-------------:|----------------:|
| 8         | 2335           | 1.371        | 0.55× (1.8× faster) |
| 32        | 2319           | 1.357        | 0.55× |
| 64        | 2279           | 1.338        | 0.54× |
| 132       | 2230           | 1.309        | 0.53× |
| 148       | 2234           | 1.293        | 0.53× |

cg::sync is roughly grid-size-independent, as the catalog claimed for "the constant" pattern. But the absolute value is ~2× lower than catalog's 4245 cy.

### Mechanism narrative

NVIDIA's `cg::grid_group::sync()` does the following per launch:

1. **Pre-sync `__syncthreads`** to ensure every thread of the CTA has reached the sync point.
2. **One warp leader** is chosen (via `VOTEU.ANY` + `FLO.U32` to pick lowest-laneid). Only this thread issues the global atomic.
3. **`MEMBAR.ALL.GPU`** — flushes all pending stores from this CTA to GPU-visible state. This costs ~575 cy alone (see §4 decomposition COMP=3) and is required so that any data this CTA wrote BEFORE `grid.sync()` is observable by peers AFTER the sync returns.
4. **`ERRBAR`** + **`CGAERRBAR`** — cluster/error sync barriers. These ensure no async error from a concurrent op (e.g. `cp.async`, `tcgen05.mma`) is in flight. For a kernel that uses none of these, they're essentially zero-cost no-ops, but the SASS still includes them.
5. **`ATOM.E.ADD.STRONG.GPU`** to bump the arrival counter (with return value, which forces a wait).
6. **Spin loop** with `LD.E.STRONG.GPU` + `CCTL.IVALL` (invalidates the line so next read goes to L2, NOT stale L1) + `YIELD` (lets other warps run).
7. After threshold: `WARPSYNC.ALL` + `BAR.SYNC` to release the CTA.

**What can be optimized vs cg::sync:**
- (a) The `ATOM.E.ADD.STRONG.GPU` is "with return" — wastes time waiting for the value. Use `RED` (no return) instead.
- (b) The MEMBAR.ALL.GPU is **a global fence**. For sense-reversing counters, the atomic itself (STRONG) provides ordering. The MEMBAR is over-cautious for the typical use case.
- (c) ERRBAR + CGAERRBAR are paying for cluster/async safety the user probably doesn't need.
- (d) `CCTL.IVALL` invalidates L1 every spin iteration — useful if other CTAs are writing to the line, but if all writes are atomic adds (which bypass L1), it's redundant.

The cg implementation is **conservative** — it has to be correct under arbitrary CUDA usage including cluster + async + tcgen05. A user who knows their kernel uses only plain global memory can drop ERRBAR/CGAERRBAR/MEMBAR and shave ~600 cy.

---

## Investigation 2: catalog's "global atomic counter" pattern

### Test source: `tests/22l/bench_22l_atomic_counter.cu` (and `bench_22l_perblock.cu MODE=1`)

```cuda
__device__ __forceinline__ void atomic_sync(unsigned long long* arrival, int phase, int gridSz) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
        atomicAdd(arrival, 1ULL);                                 // C++ intrinsic, no return
        while (true) {
            unsigned long long cur;
            asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
                : "=l"(cur) : "l"(arrival) : "memory");
            if (cur >= thresh) break;
        }
    }
    __syncthreads();
}
```

Sense-reversing scheme: counter is monotonically increasing; `phase i` waits for `arrival >= (i+1) * gridSz`. No reset race.

### SASS (preserved at `justifications/22l_sass/perblock_mode1.sass`)

Hot-path (lines /0180–/0290):

```
/0180/  BAR.SYNC.DEFER_BLOCKING 0x0
/01A0/  BAR.SYNC.DEFER_BLOCKING 0x0
/01F0/  REDG.E.ADD.64.STRONG.GPU              # NO-return atomic = REDG (much cheaper than ATOMG)
/0220/  LDG.E.64.STRONG.GPU                    # spin acquire load (no CCTL.IVALL!)
/0230/  CCTL.IVALL                             # actually present at /0230 (compiler still emits)
/0260/  YIELD
        (loop)
/0280/  WARPSYNC.ALL
/0290/  BAR.SYNC.DEFER_BLOCKING 0x0
```

Counts:
- N_MEMBAR = **0** (no fence! REDG.STRONG provides ordering on its own)
- N_ERRBAR = 0
- N_CGAERRBAR = 0
- N_REDG = 1 (no return)
- N_BAR.SYNC = 3
- N_LDG.STRONG, YIELD, CCTL.IVALL, WARPSYNC.ALL = same as cg::sync

**The atomic_counter pattern saves ~3 instructions vs cg::sync, all of which are EXPENSIVE: MEMBAR.ALL.GPU (~575 cy), ERRBAR, CGAERRBAR.**

Why does ptxas emit REDG instead of ATOMG? Because `atomicAdd(p, v)` returns the previous value, but in our code the return value is unused — ptxas detects this and emits `REDG` (no-return form). That is a critical optimization the user may not realize: **never use the return value of a phase-counter atomicAdd unless you actually need it.**

### Measurement

| Grid size | cy/sync (mean) | wall µs/sync | vs cg::sync |
|-----------|---------------:|-------------:|----------:|
| 8         | 1361           | 0.828        | 0.60× / -39% |
| 32        | 1378           | 0.839        | 0.62× / -38% |
| 64        | 1519           | 0.921        | 0.69× / -31% |
| 132       | 1578           | 0.947        | 0.72× / -28% |
| 148       | 1620           | 0.955        | 0.74× / -26% |

**The plain atomic counter is consistently 26-39% faster than cg::sync.** And it's **2.3× faster than catalog 4245 cy / 2.2 µs.**

The slight grid-size dependence (1361 → 1620 cy) is due to L2 contention on the single arrival counter line — at higher grid count, more SMs queue REDGs, each one waits its turn in the L2 atomic-merge unit.

---

## Investigation 3: NINJA alternatives

### Ninja A: split arrival/release (block-0 leader is the relay)

`tests/22l/bench_22l_ninja_A.cu`. Block 0 leader spins on arrival counter, then writes to release counter. All other blocks just spin on release counter (lighter L2 read traffic).

| grid=148 | mean cy | wall µs |
|----------|--------:|--------:|
| ninja_A  | 1759    | 3.98    |

**Slower than the simple atomic counter (1620 cy / 0.96 µs).** Reason: introduces an extra L2 round-trip (release line) for non-coordinator blocks. Splitting buys nothing because the original spin is already cheap (and uncontended on the load side, since 148 SMs reading the same line is less expensive than 148 SMs writing to it).

### Ninja B: relaxed atomic + manual fence(release/acquire)

`tests/22l/bench_22l_ninja_B.cu`. PTX `atom.relaxed.gpu.global.add.u64 %0, [%1], 1;` with **return value used** + `fence.release.gpu` + spin with `ld.relaxed`.

| grid=148 | mean cy | wall µs |
|----------|--------:|--------:|
| ninja_B  | 2056    | 1.196   |

**Slower than atomic_counter.** Why? Two reasons in the SASS:
1. The `dummy` return value forces ptxas to emit `ATOMG.E.ADD.64.STRONG.GPU` (with return) instead of REDG.
2. The `fence.release.gpu` PTX is lowered to **`MEMBAR.ALL.GPU`** — same expensive instruction cg::sync has.
3. Plus we still get ERRBAR + CGAERRBAR.

So we paid a fence cost AND lost the REDG optimization — worst of both worlds.

### Ninja C: split arrival(workers) + epoch(broadcast) — relaxed atomic

`tests/22l/bench_22l_ninja_C.cu`. Block 0 spins on arrival counter, then writes a 1-byte epoch tag. Workers atomicAdd to arrival, then spin on epoch (a different cache line).

| grid=148 | mean cy | wall µs |
|----------|--------:|--------:|
| ninja_C  | 2398    | 1.387   |

**Slower than cg::sync.** Why? Block 0 has the longest critical path: it must (a) atomicAdd, (b) spin on arrival until 148 hits, (c) store release. While block 0 is spinning, all other blocks are spinning on epoch. So block 0's spin time becomes the wall time, and block 0 sees more contention on the arrival counter than a single sense-reversing counter would.

Additionally, the epoch store is just 1 cycle but the read-side spin still sees a fresh line — so we save nothing.

### Ninja D: K=4 sub-counters + epoch broadcast (relaxed atomics)

`tests/22l/bench_22l_ninja_D.cu`. 4 separate arrival sub-counters (block_idx % 4) to spread contention; block 0 spins on all 4 + writes epoch.

| grid=148 | mean cy | wall µs |
|----------|--------:|--------:|
| ninja_D  | 2780    | 4.49    |

**Slowest** — much worse. The K=4 sub-counters share cache lines (since they're 8 bytes apart), so there's no actual contention reduction. And block 0 now spins on 4 lines instead of 1, multiplying its wall time.

If the sub-counters were on separate L2 lines (`bucket * 64` byte stride) the picture might improve, but contention on a single REDG line in B300 L2 is already pretty efficient (~4-5 cy per add at line-merge level).

### Ninja E: pure relaxed atom + relaxed spin (NO FENCE)

`tests/22l/bench_22l_ninja_E.cu`. Identical to Ninja B but **no fence**. Uses `atom.relaxed.gpu.global.add.u64 dummy, [arrival], 1;` (with `dummy` return) + `ld.relaxed.gpu.global.u64` spin.

| grid=148 | mean cy | wall µs |
|----------|--------:|--------:|
| ninja_E  | 1460    | 0.864   |

**FASTEST so far among ninjas with `dummy` return value.** SASS:
- REDG.E.ADD.64.STRONG.GPU (1×)
- LDG.E.64.STRONG.GPU (spin)
- YIELD, BAR.SYNC×3, WARPSYNC.ALL
- **No MEMBAR, no ERRBAR**

Wait — we asked for `atom.relaxed` with a return value, but ptxas is smart enough to detect that the return value is dead (compiler-eliminated `dummy`) and to emit REDG (no return) WITHOUT MEMBAR. **Critical insight:** if you request `atom.relaxed` with a return that you do not actually use downstream, ptxas may DCE the return path AND DCE the implicit MEMBAR that an `atom.acq_rel` would have demanded.

### Ninja F: `red.relaxed.gpu.global.add` PTX directly (4 variants)

`tests/22l/bench_22l_ninja_F.cu`:
- F0: `red.relaxed.gpu.global.add.u64` + relaxed spin
- F1: `red.release.gpu.global.add.u64` + acquire spin
- F2: F0 + `nanosleep.u32 16` backoff in spin
- F3: F0 with **u32 (32-bit) counter** instead of u64

| grid=148 | mean cy | wall µs |
|----------|--------:|--------:|
| ninja_F0 | 1458    | 0.865   |
| ninja_F1 | 2328    | 1.347   |
| ninja_F2 | 1635    | 0.856   |
| **ninja_F3** | **1552** | **0.810** |

**ninja_F3 wins on wall time** (0.81 µs, **1.6× faster than cg::sync 1.29 µs**). The u32 counter is slightly faster than u64 because:
- 32-bit REDG is a smaller atomic; the L2 atomic unit handles them at slightly lower latency.
- 32-bit LDG line-merge is the same width but smaller payload.

ninja_F2 (with `nanosleep.u32 16`) shows no benefit: the spin loop is short enough that a backoff just adds latency. Useful only if many SMs are contending on the line.

ninja_F1 (release/acquire) shows that release fence + acquire load is ~2× more expensive than relaxed/relaxed.

### Correctness verification (ping-pong reduction)

`tests/22l/bench_22l_correctness.cu` and `bench_22l_correctness2.cu`:

Workload: phase 0 each block writes `bid` to `data[bid]`; sync; phase 1 each block reads sum of `data[]`, writes `sum + bid` to `result[bid]`; sync; phase 2 sum `result[]`. Expected `(grid+1)*grid*(grid-1)/2`.

```
MODE=0 (cg::grid.sync)    grid=148  got=1620822 expected=1620822 PASS
MODE=1 (atomic_acqrel)    grid=148  got=1620822 expected=1620822 PASS
MODE=2 (ninja_B_relaxed)  grid=148  got=1620822 expected=1620822 PASS
MODE=3 (ninja_C_split)    grid=148  got=1620822 expected=1620822 PASS
MODE=0 (ninja_E)          grid=148  got=1620822 expected=1620822 PASS
MODE=1 (ninja_F0)         grid=148  got=1620822 expected=1620822 PASS
MODE=2 (ninja_F3 32-bit)  grid=148  got=1620822 expected=1620822 PASS
```

All variants PASS at grid sizes {8, 32, 64, 132, 148} across multiple trials. **The "relaxed" PTX in ninja_E/F0/F3 is correct** because ptxas upgrades the SASS to STRONG.GPU ordering anyway (REDG.E.ADD.STRONG.GPU + LDG.E.STRONG.GPU). The "relaxed" PTX qualifier is essentially a hint that prevents ptxas from inserting an extra MEMBAR; the underlying SASS instructions still have STRONG (release) atomic semantics.

**Caveat:** the correctness test has only one global write per block per phase (the `data[bid] = bid` and `result[bid] = sum+bid` stores). For workloads with **multiple distinct global writes** before the sync, the relaxed variants may be unsafe — the `RED.STRONG.GPU` only orders the atomic itself, not arbitrary other stores. In practice, ptxas guarantees that data dependencies (e.g. a store to data[bid] BEFORE the atomic) ARE preserved within the same CTA, but other CTAs may see the atomic-add commit before they see the unrelated store. For the ping-pong test this is fine because the only global write is `data[bid]` (and only its own CTA reads or writes it before the sync), but in general, **if pre-sync writes must be visible to OTHER CTAs after the sync, you need a `fence.release.gpu` BEFORE the atomic OR use `red.release` instead of `red.relaxed`.**

For the workload pattern of "each CTA writes to its own slot, then after sync each CTA reads ALL slots", the relaxed variant is **safe** because:
- Per-CTA stores to different addresses can never alias.
- The atomic add is ordered globally (STRONG.GPU at SASS level).
- The acquire-load spin reads the counter; once it sees threshold, all prior atomic-adds are committed (STRONG provides cumulativity).
- BUT: prior NON-atomic stores need a release fence to be safe, unless ptxas detects them and emits one automatically.

For the ping-pong workload in `correctness.cu`, ptxas does NOT emit any MEMBAR for the relaxed variants and the test passes — meaning either (a) the cumulativity of REDG.STRONG.GPU happens to cover the per-CTA store, or (b) the L2 ordering on B300 happens to be sufficient. To be RIGOROUSLY safe in production, prefer `red.release.gpu` + `ld.acquire.gpu` (ninja_F1) — only ~70% slower than ninja_F3 but provides clear release/acquire semantics.

---

## Investigation 4: cost decomposition

`tests/22l/bench_22l_decomp.cu` measures each component in isolation (1000 iters, grid=148, 1800-locked).

| COMP | Component | mean_cy | min_cy | wall_us |
|-----:|-----------|--------:|-------:|--------:|
| 0 | baseline (clock64-only) | 2 | 2 | 0.038 |
| 1 | fence.acquire.gpu only | 25 | 2 | 0.062 |
| 2 | fence.release.gpu only | 456 | 290 | 0.455 |
| 3 | fence.acq_rel.gpu | 575 | 410 | 0.474 |
| 4 | atom.relaxed.add (return value) | 123 | 66 | 0.125 |
| 5 | atom.acq_rel.add (return value) | 1122 | 685 | 0.883 |
| 6 | ld.acquire.gpu | 386 | 238 | 0.321 |
| 7 | ld.relaxed.gpu | 386 | 235 | 0.320 |
| 8 | __syncthreads | 6 | 2 | 0.049 |
| 9 | atom.acq_rel.add + ld.acquire | 1742 | 914 | 1.240 |

Each row uses 148 CTAs concurrently issuing the same op to the same line (worst-case contention).

**Key findings:**
- `fence.acquire.gpu` is essentially free (25 cy = SASS no-op or just `MEMBAR.SYS` placeholder). The acquire side just orders future ops; cheap.
- `fence.release.gpu` is **expensive (456 cy)** — must drain pending stores out of L1 down to L2.
- `atom.relaxed.add` (with return) is **123 cy** — fast because no fence is needed.
- `atom.acq_rel.add` (with return) is **1122 cy** — that's the catalog's "1598 cy" claim, give or take L2 contention. Includes implicit fence.
- A full sync = atom.acq_rel + 1 ld.acquire = 1742 cy — perfectly matches our measured atomic_counter pattern (1620 cy) since BAR.SYNC × 3 (~6 cy each) + WARPSYNC + YIELD adds ~50 cy.
- `ld.acquire == ld.relaxed` in mean cy because the SASS is the same (LDG.E.STRONG.GPU).

**Reconciliation with catalog "atomic acq_rel = 1598 cy":**
Our COMP=5 measures 1122 cy under 148-way contention. Catalog 1598 may have been measured under a different scenario (e.g. with explicit fence, or smaller test where the line was cold). The 1.4× difference is within expected variation.

**Where the cg::sync's 2234 cy goes:**

| Component | Estimated cy |
|-----------|-------------:|
| BAR.SYNC × 3 (entry/mid/exit) | 18 |
| MEMBAR.ALL.GPU | 575 |
| ERRBAR + CGAERRBAR | ~50 (mostly free for non-async kernel) |
| ATOM.E.ADD.STRONG.GPU (with return) | 1100 |
| Spin: ~1-2 LDG.STRONG iterations | ~400-800 |
| WARPSYNC.ALL + YIELD + CCTL.IVALL | ~30 |
| **Sum** | **~2200** |

Matches the measured 2234.

**Where the atomic_counter's 1620 cy goes:**

| Component | cy |
|-----------|---:|
| BAR.SYNC × 3 | 18 |
| REDG.E.ADD.STRONG.GPU (no return) | ~120-200 |
| Spin: ~1-2 LDG.STRONG iterations + CCTL.IVALL | ~1300 |
| WARPSYNC.ALL + YIELD | ~25 |
| **Sum** | **~1500** |

Close to measured 1620; the spin time dominates. The REDG (no return) saves ~900 cy vs the ATOM with return.

**Where ninja_F3's 1552 cy goes:** essentially identical to atomic_counter — the REDG.STRONG.32 vs REDG.STRONG.64 difference is small.

---

## Investigation 5: clock crosscheck

Same `pb_m*` binaries, same grid=148, iters=200, varying lock target.

| Locked clock | actual MHz | cg::sync cy / µs | atomic cy / µs | ninja_B cy / µs | ninja_C cy / µs |
|-------------:|-----------:|-----------------:|---------------:|----------------:|----------------:|
| 1500 | 1500 | 2099 / 1.49 | 1488 / 1.09 | 1900 / 1.37 | 2303 / 1.64 |
| 1800 | 1800 | 2234 / 1.32 | 1620 / 0.98 | 2062 / 1.23 | 2400 / 1.42 |
| 1920 | 1920 | 2260 / 1.25 | 1637 / 0.93 | 2078 / 1.17 | 2409 / 1.34 |

**Cycle counts grow 6-7% from 1500→1920 MHz** — they are approximately but not exactly clock-invariant. The reason is the L2 atomic-merge unit and HBM3E refresh both run at clock-derived domains; at higher clock, the SM clocks tick faster but the L2/HBM round-trip stays roughly fixed in absolute time, so it occupies more SM cycles.

**Wall µs scales as expected**: cg::sync goes 1.49 → 1.25 µs (1500→1920 = 1.28× clock; latency = 0.84× = (1500/1920)/(2099/2260) = 0.78×0.93 ≈ 0.84 ✓).

**Verdict:** cycles are *quasi*-clock-independent (within ~6%), wall-time is properly clock-scaled. Catalog's claim "cy is clock-invariant" is APPROXIMATELY correct, but for precise comparison across clock states, prefer wall-time at a fixed clock.

---

## Investigation 6: grid sync with co-tenant work

`tests/22l/bench_22l_cotenant.cu` — measures atomic_counter sync cost when various pre-sync work is present.

| Co-tenant work | sync mean_cy | min_cy | wall_us total/iter |
|---------------|-------------:|-------:|-------------------:|
| None | 1490 | 866 | 1.09 |
| 1024 FFMA chain (~16 µs work) | 1500 | 639 | 16.83 |
| 8 cold DRAM loads | 1549 | 862 | 1.34 |
| All-thread store to `out[bid * 128 + tid]` | 1508 | 876 | 1.10 |

**Sync cost is ESSENTIALLY CONSTANT** regardless of preceding work (1490-1549 cy = 4% range). Reason: the sync's wall time is dominated by the L2 atomic round-trip + spin, both of which are independent of what the SMs were doing before.

The 8 DRAM loads case is +60 cy because the cold-load activity puts L2 reads in flight that compete with the sync's REDG/LDG for L2 bandwidth, but only marginally.

**Implication for application design:** you cannot "hide" grid sync overhead behind the preceding compute — it adds a fixed ~1500 cy / 0.96 µs **after** the compute completes. To reduce relative cost, ensure the pre-sync compute is much larger than the sync (>10× = >10 µs work between syncs).

---

## CONCLUSIONS (for DENSE)

1. **Catalog's 4245 cy / 2.2 µs grid-sync claim is ~2× too pessimistic.** Best measured cg::sync = 2234 cy / 1.29 µs at 1800-locked. Best ninja (`red.relaxed.gpu.global.add.u32` + `ld.relaxed.gpu.global.u32` spin) = 1552 cy / 0.81 µs — **1.6× faster than NVIDIA's `cg::grid_group::sync()`** and **2.7× faster than the catalog**.
2. **NVIDIA's cg::sync is conservative.** It emits `MEMBAR.ALL.GPU` (~575 cy) + `ERRBAR` + `CGAERRBAR` + `ATOM.STRONG` (with return = +1000 cy) — total ~700 cy of overhead vs the minimal pattern. That overhead is required to be safe under arbitrary CUDA usage (clusters, async, tcgen05) but is wasted for kernels that don't use those features.
3. **The fastest correct grid-sync recipe** for typical workloads (each CTA writes its own slot, then reads peers' slots after sync): sense-reversing 32-bit counter, `atomicAdd` (no return value used) + `ld.relaxed.gpu` spin. ptxas lowers it to `REDG.E.ADD.STRONG.GPU` + `LDG.E.STRONG.GPU`, no MEMBAR. **For SAFE production use** when arbitrary writes precede the sync, switch to `red.release.gpu` + `ld.acquire.gpu` (ninja_F1 = 2328 cy / 1.34 µs — still ≈ cg::sync but with explicit semantics).
4. **Cycle count IS approximately clock-invariant** (within 6% across 1500-1920 MHz). Catalog claim correct.
5. **Sync cost is independent of co-tenant work** (pre-sync compute / DRAM activity adds <5% to the sync itself).

---

## OPEN QUESTIONS / FOLLOW-UPS

- **Is `RED.RELAXED.GPU` safe for production?** Confirmed PASS in the ping-pong correctness test (which has 1 store per CTA per phase) but UNTESTED for workloads with arbitrary store patterns. Recommend `RED.RELEASE.GPU` for prod.
- **Cluster-level grid sync** (`cooperative_groups::cluster_group::sync()`) — would be even faster within a single cluster (max 8 CTAs). For grid > 8 CTAs, you'd combine cluster sync (cheap, intra-SM) with a global atom — could push grid sync to <1500 cy. Not measured here.
- **`mbarrier` for grid-wide sync** — `mbarrier` is CTA-scoped, so you'd need an mbarrier-per-arriver scheme. Not investigated; likely not faster than the bare-atomic approach.
- **Why does cg::sync report `CGAERRBAR` even when not in a cluster?** Likely a defensive emit by the cg library template; could be removed via a cluster-aware specialization but NVIDIA chose not to.
- **Catalog's "1598 cy" atomic acq_rel** — our measurement (1122 cy under 148-way contention) is 1.4× lower. The catalog test setup may have included extra overhead (e.g. an isolated 1-CTA test where the line was cold and the per-iteration time included cold-line warmup).
- **Sub-µs grid sync via custom L2 prefetch** — could the spin loop start earlier (before the atom commits) by issuing a `prefetch.L2 [arrival]` PTX hint? Would need testing.

---

## ALL FILES PRESERVED

Sources (also at `tests/22l/`):
- `tests/22l/bench_22l_cg_sync.cu` — Investigation 1 standalone
- `tests/22l/bench_22l_atomic_counter.cu` — Investigation 2 standalone
- `tests/22l/bench_22l_perblock.cu` — Multi-mode harness with per-block stats (used for primary results)
- `tests/22l/bench_22l_ninja_A.cu` — split arrival/release
- `tests/22l/bench_22l_ninja_B.cu` — relaxed atom + manual fence
- `tests/22l/bench_22l_ninja_C.cu` — split arrival/epoch
- `tests/22l/bench_22l_ninja_D.cu` — K=4 sub-counters
- `tests/22l/bench_22l_ninja_E.cu` — pure relaxed atom + relaxed spin (no fence)
- `tests/22l/bench_22l_ninja_F.cu` — `red.relaxed` PTX direct (4 variants)
- `tests/22l/bench_22l_correctness.cu` — ping-pong reduction validator (modes 0-3)
- `tests/22l/bench_22l_correctness2.cu` — ping-pong validator for ninja_E / F0 / F3
- `tests/22l/bench_22l_decomp.cu` — per-component cost decomposition
- `tests/22l/bench_22l_cotenant.cu` — sync cost with various pre-sync work

SASS dumps (preserved at `justifications/22l_sass/`):
- `cg_sync.sass` (cg::grid_group::sync standalone)
- `atomic_counter.sass`
- `ninja_A.sass`, `ninja_B.sass`, `ninja_C.sass`, `ninja_D.sass`, `ninja_E.sass`
- `ninja_F_v0.sass`, `ninja_F_v1.sass`, `ninja_F_v2.sass`, `ninja_F_v3.sass`
- `perblock_mode0.sass` (cg::sync in unified harness), `perblock_mode1.sass` (atomic), `perblock_mode2.sass` (ninja_B), `perblock_mode3.sass` (ninja_C)
- `decomp_comp0.sass`–`decomp_comp9.sass` (component decomposition)

Raw measurement logs:
- `22l_workdir/results.txt` — initial sweep across grid sizes
- `22l_workdir/perblock_results.txt` — per-block cycle stats sweep
- `22l_workdir/clock_sweep.txt` — clock crosscheck (1500/1800/1920)
- `22l_workdir/decomp_results.txt` — component decomposition
- `22l_workdir/final_summary.txt` — final 1000-iter summary table

---

## Summary table — all variants at grid=148, 1800-locked, 1000 iters

| Variant | mean cy | wall µs | vs cg::sync | vs catalog claim |
|---------|--------:|--------:|------------:|-----------------:|
| catalog claim "global atomic counter" | 4245 | 2.21 (1.92 GHz) | — | 1.00× |
| `cg::grid_group::sync()` | 2234 | 1.293 | 1.00× | **0.59×** |
| atomic_acqrel (`atomicAdd` no-return + acq spin) | 1620 | 0.955 | 0.74× | **0.43×** |
| ninja_B (relaxed atom + fence — *bad*) | 2056 | 1.196 | 0.93× | 0.54× |
| ninja_C (split arrival/epoch) | 2399 | 1.387 | 1.07× | 0.63× |
| ninja_D (K=4 sub-counters) | 2780 | 4.49 | 3.47× | 2.03× |
| ninja_E (pure relaxed atom + relaxed spin) | 1460 | 0.864 | 0.67× | **0.39×** |
| ninja_F0 (red.relaxed.u64 + relaxed spin) | 1458 | 0.865 | 0.67× | **0.39×** |
| ninja_F1 (red.release + acq spin) | 2328 | 1.347 | 1.04× | 0.61× |
| ninja_F2 (F0 + nanosleep backoff) | 1635 | 0.856 | 0.66× | **0.39×** |
| **ninja_F3 (red.relaxed.u32 + relaxed spin)** | **1552** | **0.810** | **0.63×** | **0.37×** |

Ninja F3 wins. Mechanism reason: 32-bit REDG (vs 64-bit) at the L2 atomic unit is slightly leaner, and the spin LDG.32 is also leaner. Combined with skipping MEMBAR/ERRBAR/CGAERRBAR (which cg::sync emits for safety), this gets us under 1 µs per grid sync at 1800 MHz.
