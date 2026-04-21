# M5 — B300 Memory Hierarchy Cheatsheet (2026-04-21)

**Memory subsystem reference: capacities, latencies, throughputs, surprises.**
Synthesizes D1-D10, N2-N4, E5, E7 findings.

---

## 1. Capacity ladder

| Level | Per SM | Total | Latency | Lines / Sectors |
|-------|-------:|------:|--------:|------------------|
| Registers | 65536 (32-bit) | 9.7 MiB | 1 cy | 255 max/thread (P3) |
| TMEM | 256 KB | 38 MB | (varies) | 0 idle power (H7) |
| L1 / SMEM unified | 256 KB | 38 MB | 38 cy hit | 128 KB L1 effective (D2) |
| L2 cache | (per SM ~850 KB share) | **126 MB** | 152 ns / 230 cy | 32 B sectors, 128 B lines (D3) |
| HBM3E | (per stack ~8 GB) | **96 GB** | 705 cy / 470 ns | (varies) |

---

## 2. Bandwidth peaks (sustained, B300 SXM6)

| Path | Peak | % theoretical | Source / commit |
|------|-----:|-------------:|------------------|
| L1 / SMEM | ~46 TB/s | (per cat) | catalog |
| L2 (single-SM) | 23 TB/s | (per cat) | catalog |
| **HBM3E read** | **7.30 TB/s** | 95% of 7672 spec | a04d9c8 (B300_TRUE_REFERENCE) |
| HBM3E write (uncoalesced) | ~885 GB/s × 8 sector amp | 7080 GB/s actual DRAM | D10 (#4eefc38) |
| NVLink (peer) | 740 GB/s | 77% of 956 | J2 (#e16901f) |
| PCIe Gen 6 ×16 | 58 GB/s effective | (catalog) | catalog |

---

## 3. L1 properties (D2 #538339e, N2 #5e452ea)

- **Per-SM effective L1 ≈ 128 KB = 1024 cache lines** (sharp boundary at 1024 lines)
- **Hit latency: 39 cy chained** (= 26 ns)
- **No power-of-2 stride aliasing** observed → likely hashed indexing
- L1 hit rate measured via `l1tex__t_sector_hit_rate.pct`:
  - 64 lines (256 KB): 97.56%
  - 1024 lines (4 MB at 4KB stride): 83.81%
  - 2048 lines: 23.33% (sharp drop)
- **Default LDG = `LDG.E` (STRONG.SM, L1 cached)** — only `.cg` bypasses L1
- Cache config preferences (`cudaFuncCachePreferShared`/L1) are NO-OP on Hopper+ (N4)

---

## 4. L2 properties (D1 #e1bc97a, D3 #8c6e5b5)

- **Total L2: 126 MB shared across 148 SMs**
- **Cache line = 128 B = 4 sectors of 32 B each**
- **L2 sector size = 32 bytes** — confirmed via DRAM RMW ratio
  - 4 B writes (sub-sector): **7× DRAM read amplification** (RMW)
  - 16 B writes (half sector): 1.7× amp
  - 32 B+ aligned: 0× amp (clean)
- **Replacement policy = pseudo-LRU / hash-based, NOT strict LRU**
  - Hot-after-cold-sweep ratio = 1.69× (64 MB cold) to 2.06× (256 MB+)
  - Even sub-L2 cold sweep partially evicts hot data
- **L2 hit latency: 152 ns / 228 cy chained**
- **L2 partition asymmetry: 1.27-1.85×** in latency depending on address (E5 #d958607)
  - Stride 128 B = local minimum in BW (D10 #4eefc38)
  - 256+ B strides hit DRAM uniformly at peak

---

## 5. HBM properties

- **96 GB total / 12 stacks**
- Read latency: 705 cy / 470 ns chained
- Per-channel mapping NOT observable via ncu (D4 deferred)
- **128 B stride = local minimum** (804 GB/s) likely partition concentration (D10)
- **256-128K B strides plateau at 885 GB/s effective payload** = 7080 GB/s DRAM (HBM peak after 8× sector amplification)

---

## 6. Atomics ladder

| Atomic target | Throughput | Source |
|---------------|-----------:|--------|
| SMEM atomic | 9603 Gops/s = 99.78% of arch ceiling | catalog |
| L2 atomic (uncombined) | 191 Gops/s = 766 GB/s | catalog |
| HBM atomic (uncombined) | 52 Gops/s = 208 GB/s (3.7× slower than L2) | catalog |
| HBM atomic (combine 32) | 769 Gops/s = 3.08 TB/s payload | catalog |
| **HBM atomic DRAM ceiling** | **5.52 TB/s** | catalog |
| **Sysmem atomic (host-mapped)** | **0.03 Gatomic/s = 50× SLOWER than device** | E7 (#77bfaf3) |
| Managed-mem atomic (after migrate) | = device atomic | E7 |

Per-block atomic latency variance: **1.79-1.85×** across L2 partitions (E5).

---

## 7. Async copy / TMA

- **cp.async pipe depth: ~16 in-flight per warp** (10.4× gain at N=16, 32.8 cy/chunk) (F5)
- **TMA bulk read = LDG read** within 0.3% at HBM SoL (catalog c40c016)
- **TMA + cluster multicast = 8× BW savings** for shared inputs across cluster CTAs (Q1 / f890323)
- Required: `__align__(16)` on smem destination, address aligned to 16 B

---

## 8. Cache hints + special PTX

| PTX | What it does | Available on B300? |
|-----|--------------|---------------------|
| `ld.global.cg` | Bypass L1 (cache global only) | YES (G6 #9eb988c) |
| `ld.global.nc` | Read-only / constant cache | YES (= LDG.E.CONSTANT) |
| `__ldg(p)` | Same as .nc + read-only marker | YES |
| `cctl.ivall.L1` | Invalidate L1 | **NO** (compile error, N3 #512b470) |
| `cctl.wb.L1` | Writeback L1 | **NO** |
| `cctl.iv.L1 [addr]` | Invalidate single line | **NO** |
| `discard.global.L2 [addr], <bytes>` | Hint to evict | YES (~+45 cy/call) |

---

## 9. Address generation pipeline (D8 #f18899b)

- **AGEN pipelined at 6-7 cy** for address-independent loads
- **Address-dependent chain (a[b[i]]) = 33 cy/op** = LDS latency exposed
- **IADD3 work BEFORE a load is FREE** (pipeline absorbs it)

**Practical:** pre-compute scatter/gather indices to avoid AGEN stalls.
Indirect-addressed loads cost **5× direct loads**.

---

## 10. Practical recipes

### For HBM-bound kernels
- Coalesced 16 B writes (float4 / uint4) — 1.7× RMW amp acceptable
- 32 B aligned writes (uint4 + uint4 = 32 B together) — 0× amp
- Avoid 4 B scattered writes (7× amp)

### For L1/L2-bound kernels
- Keep working set ≤ 128 KB per SM (L1 hit zone)
- Use `__restrict__` to enable LDG.E.CONSTANT routing
- Use `.cg` for streaming loads that pollute L1

### For atomic-heavy kernels
- Prefer SMEM atomic (9.6 Gops/s, near peak)
- Combine atomics by 32× when possible (3.08 TB/s payload)
- NEVER use host-mapped memory for atomics (50× slower)

### For cross-CPU↔GPU coordination
- Managed memory + CPU spin: 4.4 µs RT (L1)
- cuStreamWriteValue32: 460 ns (L2)
- NVLink ping-pong: 1.55 µs one-way (J1)

---

For per-task rigor docs and complete hash list, see `M1_V4_DEEP_DIVE_INDEX.md`.
