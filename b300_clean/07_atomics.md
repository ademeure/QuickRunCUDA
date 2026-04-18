# B300 Atomic Operations — Clean Reference

**Hardware**: NVIDIA B300 SXM6 (sm_103a), 148 SMs, 2.032 GHz boost, 126 MB L2.
**Confidence key**: H = SASS+ncu verified, M = single test or formula-confirmed, L = inferred / single source.

---

## 1. Single-thread chained latency (no contention, hot location)

Pipelined cost (loop where return value is *not* a dependency):

| Op (u32) | cy/op | ns | Confidence |
|---|---:|---:|---|
| atomicInc | 7.9 | 3.9 | H |
| atomicDec | 7.0 | 3.4 | H |
| atomicAdd | 15.2 | 7.5 | H |
| atomicSub | 15.2 | 7.5 | H |
| atomicMin / Max | 15.7 | 7.7 | H |
| atomicAnd / Or / Xor | 23.5 | 11.6 | H |
| atomicExch | 49.5 | 24.4 | H |
| atomicCAS | 52.5 | 25.9 | H |

True round-trip latency (when the next op consumes the return) is ~97 ns (~197 cy) for any global atomic — see scope table. The ~7-26 ns numbers above are **pipelined throughput per chained op**, not latency.

**True per-op latency (1 warp, dependent chain, L2-hit, per-thread addresses)**: ~310 cy near-L2 / ~680 cy far-L2 / ~1169 cy in the most contended single-chain measurement. **B300 has 2 L2 partitions, hash flips ~every 4 KB → 2.19× near/far ratio** (H).

---

## 2. Scope ladder — global atomicAdd (single thread, hot location)

| Variant | cy | ns | Notes |
|---|---:|---:|---|
| atomicAdd_block (global) | 198 | 97 | **No benefit** vs default — _block only matters on shared (H) |
| atomicAdd (default = .gpu) | 198 | 97 | Baseline |
| atomicAdd_system (global) | 198 | 97 | Same as default — scope affects ordering, not latency (H) |
| atomicAdd_block (**shared**) | 28 | 14 | **7× faster** — uses HW ATOMS path on smem |

**Rule**: `_block` only buys you anything on shared memory. On global, the L2 round-trip dominates and scope qualifier is free. (H)

---

## 3. PTX scope × ordering matrix (1-warp, per-thread addresses, true latency)

**Shared memory (atom.shared.add.u32):**

| Ordering | .cta | .cluster/.gpu | .sys |
|---|---:|---:|---:|
| relaxed | 44 cy / 22 ns | 44 | 44 |
| acquire | 50 cy | 50 | 50 |
| release | 52 cy | **304 cy / 150 ns** | 4000-7000 cy (variable) |
| acq_rel | 58 cy | **312 cy / 153 ns** | 4000-16500 cy (variable) |
| seq_cst | rejected by ptxas | — | — |

**Global memory (atom.global.add.u32, L2-hit):**

| Ordering | .cta | .cluster/.gpu | .sys |
|---|---:|---:|---:|
| relaxed | 413 cy / 203 ns | 413 | 413 |
| acquire | 419 cy | 421 | 421 |
| release | 421 cy | **1455 cy / 716 ns** | ~5800 cy (variable) |
| acq_rel | 427 cy | **1463 cy / 720 ns** | ~5800 cy (variable) |

**Rules (H)**:
- Scope is **free at relaxed**. Default = .gpu = .relaxed.gpu.
- Ordering penalty appears at **release/acq_rel × cluster/gpu**: +1040 cy global / +260 cy shared (MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR triple).
- `acquire` adds only +6-8 cy (CCTL.IVALL only).
- `seq_cst` not supported by ptxas on sm_103a.
- `.sys` release/acq_rel: 4000-20000 cy, highly variable (NVLink coherence).
- **Use atom.relaxed + separate fence at batch boundaries** — pay fence once, not per atomic.

---

## 4. FP atomics — scalar half/bfloat16 falls back to CAS loop

| Type | cy/op | ns/op | Path | Confidence |
|---|---:|---:|---|---|
| atomicAdd float (FP32) | 6.8 | 3.3 | HW REDG.E.ADD.F32 | H |
| atomicAdd double (FP64) | 9.2 | 4.5 | HW | H |
| atomicAdd __half2 packed | 64 | 31.6 | HW (per pair = 16 ns/elt) | H |
| atomicAdd __nv_bfloat162 packed | 64 | 31.7 | HW (per pair = 16 ns/elt) | H |
| atomicAdd __half (scalar) | 1422 | **700** | **CAS loop** — 200× slower than FP32 | H |
| atomicAdd __nv_bfloat16 (scalar) | 1389 | **683** | CAS loop | H |
| red.global.add.noftz.f16 (PTX direct) | 1379 | 679 | Also CAS loop — no native HW path | H |

**Rule (H)**: NEVER use scalar `__half` / `__nv_bfloat16` atomicAdd. Either pack to half2/bf162 (5× per-element cost vs FP32, still 40× faster than scalar) or accumulate in FP32 and convert at the end.

---

## 5. Op-type half-rate: CAS

| SASS | pipe_lsu rate | atoms/SM/cy |
|---|---:|---:|
| ATOMS.{ADD,MIN,MAX,AND,OR,XOR,EXCH,INC,DEC} | **1.00** | 32 |
| ATOMS.CAS | **0.50** | 16 |
| 8-way bank conflict (any ATOMS) | 0.125 | 4 |

CAS is unconditionally half-rate (always-succeed = always-fail = 2.189 ms vs 1.096 ms for ADD). Verified bank-clean. (H)

Global `atom.global.cas.b32` is also ~half-rate vs `atom.global.add.u32`: 287 vs 504 Gops/s at stride=4B (compiler can't coalesce 8 distinct CAS values into one 32B L2 packet).

---

## 6. Contention scaling — per-warp is the WORST pattern

148 × 128 threads = 18.94 M ops, atom.global.add.u32 (H):

| Pattern | Gops/s | Notes |
|---|---:|---|
| All threads → A[0] | 27-49 | Warp coalesces to 1 HW op/warp; L2 has fast-path serializer for single CL |
| Per-CTA address (148 hotspots) | 38-89 | Same as all-same — L2 serializer is bottleneck |
| **Per-warp address (592 hotspots, 32-way intra)** | **7** | **5-12× SLOWER than per-CTA** — anti-pattern |
| Per-thread (151,552 unique) | 402-504 | Peak |

**Why per-warp is pathological (H)**: HW cannot intra-warp-coalesce when each lane needs a distinct return value. 592 addresses × 32-way contention scatter across L2 partitions without deduplication. Single-hotspot wins because L2 has a fast-path single-CL serializer + warp-uniform coalesces 32 → 1 HW op.

**Rule**: avoid one-atomic-per-warp on distinct addresses (naive histograms). Either go fully coalesced (per-thread) OR fully concentrated (per-CTA into smem, flush to global once).

---

## 7. Cache-line spread sweep (148 × 128 thr × 1000 iter)

| Stride (bytes) | Gatomic/s | Notes |
|---:|---:|---|
| 4 | 7.2 | 32 atomics in same CL |
| 16 | 25.0 | |
| 64 (1 CL) | 48 | First plateau |
| 128 | 86 | |
| 256 | 111 | **15× faster than stride=4** |

Practical (H): pad atomic targets ≥128 B apart for one-per-CL minimum; ≥256 B gives another 1.7×; ≥1024 B saturates.

---

## 8. Cache-residency cliff: L2 vs DRAM (UNROLL=16, full chip)

| Stride | Footprint | Cache | Gatomic/s |
|---:|---:|---|---:|
| 4 B | 9.7 MB | L2 | **504** |
| 32 B | 78 MB | L2 | 76 |
| 64 B | 155 MB | **DRAM** | 38 |
| 256 B | 621 MB | DRAM | 12 |

**The 43× gap between stride=4B (504 Gops/s) and stride=256B (12 Gops/s) is L2 vs DRAM, NOT coalescing.** Earlier "2.7× speedup from coalescing" claim is wrong attribution. (H, ncu-validated).

**Peak L2 atomic rate (UNROLL=32, stride=4B)**: 1005 Gops/s = 125 G HW packets/s. Per L2 slice: ~1 packet per 2.4 cy at 2032 MHz. Catalog "137/372 Gops/s" peaks were L2-resident at lower ILP — same hardware, less in-flight.

**Keep counter arrays ≤126 MB** for L2-resident; DRAM-bound atomics drop ~40-100×.

---

## 9. red.global vs atom.global — DO NOT use red.global

`red.global.add.u32` PTX should be "fire-and-forget" but the compiler inserts `CCTL.IVALL` (cache invalidate-all) **between every instruction**, completely serializing throughput. Measured ~5 Gops/s vs 504 Gops/s for atom.global. (H)

Use `atom.global.add.u32` (REDG.E.ADD.STRONG.GPU) even if you discard the return value.

Note: `red.shared.add.u32` is fine — same SASS as `atom.shared.add` (compiler canonicalizes to ATOMS).

---

## 10. SASS family map

| PTX | SASS | Notes |
|---|---|---|
| atom.shared.* (ADD/MIN/MAX/AND/OR/XOR/EXCH/INC/DEC) | ATOMS.* | Native HW |
| atom.shared.cas | ATOMS.CAS | Half-rate |
| atom.shared.add.f32 | BSSY+LDS+CAS loop | **Emulated, no native f32 ATOMS** |
| atom.global.add.u32 (ADD/MIN/MAX) | REDG.E.*.STRONG.GPU | Native, both with-return and no-return |
| atom.global.add.f32 | REDG.E.ADD.F32.FTZ.RN.STRONG.GPU | **Native FP32 atomic on global** (unlike shared) |
| atom.global.exch | ATOMG.E.EXCH.STRONG.GPU | Different family |
| atom.global.cas | ATOMG.E.CAS.STRONG.GPU | Half-rate (16× L2 sectors vs REDG) |
| atom.acq_rel.gpu.global | REDG + MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR | +1040 cy |
| acquire-side ordering | + CCTL.IVALL | +6-8 cy |

---

## 11. Cross-GPU (NVLink) — flag for NVLink agent

LOCAL vs REMOTE (148 × 32 × 32 thread atom.global.add.u32):
- Unique addresses: 137 G LOCAL vs 9 G REMOTE → **15× slower remote** (NVLink packet-rate bound)
- Contended (warp-uniform): 49 G LOCAL vs 16 G REMOTE → 3× slower (coalescing saves NVLink)
- Single-thread cross-GPU latency: ~1.8 µs vs 354 ns local round-trip
- All op types (Add/Min/Max/Xor/Or/And/Exch/CAS) within 1% of each other on remote chain — round-trip dominates

Detailed NVLink atomic numbers belong to the NVLink agent.

---

## 12. RETIREMENT — supersedes earlier catalog entries

| Old claim | Status |
|---|---|
| "atomic peak 137 Gops/s" | Correct at UNROLL≈3 — see updated peak below |
| "atomic peak 273 Gops/s" | Correct at UNROLL≈6 |
| "atomic peak 372 Gops/s stride=4" | Correct at UNROLL≈8 |
| "2.7× speedup from coalescing" | **WRONG attribution** — actually 43× from L2-vs-DRAM |
| "acquire.gpu = 780 cy / 17× relaxed" | Was warp-serialization on contended single-address — under no contention it's only +6 cy |
| "All atomic scopes are FREE" | True only at relaxed ordering; release.gpu adds +1040 cy on global |
| "red.global is faster than atom.global" | **WRONG** — compiler inserts CCTL.IVALL → 100× SLOWER. Use atom.global. |
| "atom round-trip 137 cy" | Was L2 *cache* latency, not atomic. Real atom round-trip 310-1169 cy. |

**True peak**: 1005 Gops/s (UNROLL=32, stride=4B, L2-resident, full chip).

## Cross-references

- L2 / cache: `b300_clean/` memory hierarchy doc
- Fences (membar): `b300_clean/` fences doc — fence costs separate from atomic costs
- Cross-GPU: NVLink agent (P2P atomic latency/throughput)
- SASS opcode table: B300_PIPE_CATALOG.md §8
