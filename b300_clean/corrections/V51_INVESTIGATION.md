# V51 Multi-Stream HBM Investigation

**File:** `tests/standalone/v51_multistream_hbm.cu` (untracked, never committed).
**Question it tries to settle:** Can N concurrent streams aggregate exceed
single-stream HBM peak (~7.30 TB/s)? Hypothesis: NO — HBM is shared physical bus.

## Current state

Allocates `d_src[8]` as **8 separate 4 GB device buffers**, but **never passes
a per-stream buffer to the kernel**. Both warmup and timed launches do:

```cpp
tma_8deep<...><<<..., streams[s]>>>((const float*)d_src, d_out[s], ...);
```

`d_src` is `float*[8]` (a host array of pointers). `(const float*)d_src`
casts the *host stack address of the pointer-array* to `float*`, then offsets
into device global memory using that bogus pointer. **The per-stream-buffer
fix was attempted but never wired through** — exactly the symptom the task
description warns about. Should be `d_src[s]`.

## Bugs found

1. **CRITICAL — wrong-pointer UB.** `(const float*)d_src` is the host
   pointer-array address, not a device buffer. Either the kernel will fault,
   or (worse) by coincidence read from some valid device address, fabricating
   a result. This is the real reason the previous run was unreliable.
2. **Resource leak hides the bug.** `cudaFree(d_src)` at end frees the host
   array address, leaks all 8 device buffers (32 GB). Driver will not error.
3. **mbarrier double-init is harmless but wasteful.** Re-initing in the loop
   resets the phase; combined with `arrive.expect_tx` + `try_wait …, 0` it
   works (phase 0 each iter), but the outer `mbarrier.init` before the loop
   is dead. Not a correctness bug.
4. **`cap` math is fine** (`words=1G dwords` = 4 GB matches `cap=1 GB *
   sizeof... ` actually `1ull*1024^3` = 1 G dwords = 4 GB; modulo
   `(cap - TILE/4)` keeps offset in-buffer). Per-stream `BLOCKS/n_streams`
   means smaller working set per stream — risk that `n_streams=8` becomes
   L2-resident (4 GB ÷ 8 = 512 MB; still > L2's 126 MB, OK).
5. **`unsigned` offset arg to `cp.async.bulk` is `r`** but `off` is `size_t`;
   `src + off` resolves to `l`-class pointer so PTX is fine.
6. **Even if fixed, `clock64`-based timing inside the kernel is wrong for
   this question** — the kernels in different streams clock independently;
   you cannot sum cy/cta across streams. Use the existing `cudaEvent` pair
   only (which is what the print does — good).

## Relevance to HBM denominator question

`01_hbm_bandwidth_CORRECTED.md` UNRESOLVED §E ("Multi-GPU contention on
shared HBM") asks about cross-GPU. **Single-GPU multi-stream aggregate is
NOT in any UNRESOLVED list** — it is implicitly settled (HBM is one bus,
streams cannot aggregate beyond it). V51 would only re-confirm a known
physical fact and is not on the V8/V9 curiosity list.

## Recommendation: (c) REMOVE

- Repo convention: broken/abandoned standalone tests are not kept (no
  `BROKEN_*` or `wip_*` files in `tests/standalone/`; v10\_* tests that
  shipped are all working). Untracked + broken = delete.
- The question (multi-stream aggregate ≤ single-stream peak) is
  architecturally trivial; not worth a fixed test.
- If anyone wants the answer empirically, replace with a 20-line
  `cudaMemcpyAsync` two-stream race that reuses the d2d harness already
  proven in `01_hbm_bandwidth_CORRECTED.md` §8.

**Action:** `rm tests/standalone/v51_multistream_hbm.cu`. No replacement
needed; if curiosity returns, file an entry in `CURIOSITY_LIST_V4.md`
under HBM and let it be picked up properly.