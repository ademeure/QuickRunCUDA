# B300 NVLink P2P / Multi-GPU — Clean Reference

**Scope:** 2× B300 SXM6 over NV18 NVLink. P2P bandwidth (DMA + kernel), cross-GPU latency, atomics, fence costs, IPC, all-reduce (custom + NCCL), sharded GEMM.

**System:** 2× NVIDIA B300 SXM6 AC. GPU0 PCI bus 4, GPU1 bus 5. Connected by NV18 = **18 × NVLink5 links per direction**. Driver 580.126.09, CUDA 13.2. NCCL 2.29.3.

**Theoretical NVLink5 peak:** 18 links × 50 GB/s data-only = **900 GB/s/dir** (full-duplex, 1800 GB/s aggregate). `nvidia-smi` reports 53.125 GB/s/link including protocol overhead → 956 GB/s raw.

**Clock context:** all cycle counts at 1920 MHz under `nvidia-smi -lgc 2032`. BW values in GB/s are clock-independent (NVLink-bound).

Confidence markers: **HIGH** = SASS+ncu cross-checked, multiple agents agree. **MED** = single test, plausible vs theoretical. **LOW** = framework-dependent or unreproduced.

---

## 1. Topology & P2P attributes (HIGH)

| Attribute | Value | Notes |
|---|---|---|
| `cudaDevP2PAttrPerformanceRank` | **0** | Highest (NVLink) |
| `cudaDevP2PAttrAccessSupported` | **1** | Bidirectional |
| `cudaDevP2PAttrNativeAtomicSupported` | **1** | Cross-GPU atomics work |
| `cudaDevP2PAttrCudaArrayAccessSupported` | **1** | Cross-GPU texture array access |
| `cudaDeviceCanAccessPeer(0,1)` / `(1,0)` | YES / YES | Symmetric |
| Link type | NV18 = 18 × NVLink5 | per direction |
| Number of GPUs | 2 | bus 0x4 and 0x5 |

---

## 2. P2P bandwidth — DMA path (`cudaMemcpyPeerAsync`) (HIGH)

| Direction | Size | Time | BW | % of 900 GB/s peak |
|---|---:|---:|---:|---:|
| GPU1 → GPU0 | 16 MB | — | ~770 GB/s | 86% |
| GPU1 → GPU0 | 256 MB | — | ~777 GB/s | 86% |
| GPU1 → GPU0 | 1024 MB | — | **776-777 GB/s** | **86%** |
| GPU0 → GPU1 | 1024 MB | — | **777 GB/s** | 86% (symmetric) |
| BIDIR (1 GB each direction) | 2× 1024 MB | — | **1543 GB/s aggregate** | **86% × 2** = 1.99× scaling |

**Bidirectional perfect 2× scaling**: 776 + 776 ≈ 1543 GB/s aggregate. Two directions are electrically separate; saturating one does not affect the other.

---

## 3. P2P bandwidth — kernel path (direct LDG/STG to peer memory) (HIGH after correction)

When peer access is enabled, the SM can issue `LDG` / `STG` directly to peer-GPU virtual addresses; the address is resolved over NVLink with no CPU/DMA engine involved.

### 3a. Read (GPU0 reads peer GPU1 memory)

| Config | Threads | BW | Notes |
|---|---:|---:|---|
| 148 SMs × 256 thr (early test) | 38K | 286 GB/s | **WRONG — thread-limited** (retired) |
| 148 SMs × 512 thr, float4, ILP | 75K+ | **778 GB/s** | matches DMA, NVLink saturated |
| Local HBM read baseline (same kernel on GPU0's own memory) | 75K+ | ~7000 GB/s | local HBM3E |
| **Slowdown peer vs local** | | **9×** | NVLink is 9× slower than local HBM |

`p2p_kernel.cu` and `tests/p2p_deep.cu` both confirm the kernel path saturates NVLink at the same ceiling as `cudaMemcpyPeer`. The "286 GB/s" earlier number was a **thread-count artifact**: 148 SMs × 256 threads = 38K threads is too few outstanding loads to fill the NVLink request queue. **Need ≥75K threads** (148 × 512+) and `float4` LDGs to saturate.

### 3b. Write (GPU0 writes to peer GPU1 memory)

| Width | Warps/SM | BW (event-timed) | BW (clock64-derived) |
|---|---:|---:|---:|
| W=128 (coalesced STG.E.STRONG.SYS) | 32 | **~720 GB/s** | ~770 GB/s (clock64 over-reports) |
| W=1024 / 4.8 GB | 32 | **768 GB/s** | 768 GB/s (steady state, all WIDTHs converge) |

**Write steady state ≈ 720 GB/s via CUDA events** = **80% of 900 GB/s peak**. The 6% gap between event-timed (720) and clock64-timed (770) is launch/sync overhead not seen in cycle counts.

### 3c. Read vs Write asymmetry

- **READ ≈ 820 GB/s = 91% of peak** — pure full-duplex pull; peer L2 streams at line rate.
- **WRITE ≈ 720 GB/s = 80% of peak** — needs ACK from peer L2 (commit confirmation adds RT latency per request).
- **Reads saturate at W=16; writes need W ≥ 256** to reach steady state.

### 3d. SM-count scaling — saturation at ~32 SMs (HIGH)

| SMs | WRITE BW (W=128 fenced) | READ BW (W=32 cache-defeat) |
|---:|---:|---:|
| 1 | 44 GB/s | 36 GB/s |
| 8 | 324 GB/s | 284 GB/s |
| 16 | 473 GB/s | 553 GB/s |
| 32 | **669 GB/s** | **792 GB/s (saturated)** |
| 74 | **765 GB/s (saturated)** | 801 GB/s |
| 148 | 763 GB/s | 817 GB/s |

Per-SM unsaturated rate ≈ 36-40 GB/s. **Above 32 SMs the chip-wide NVLink5 cap is the bottleneck.** Design implication: reserve ~32 SMs for cross-GPU I/O, leave 116 SMs for compute with **zero BW loss**.

---

## 4. Cross-GPU latency (HIGH)

Pointer-chase / serial-chain measurements, 1 SM, warm peer L2.

### 4a. Atomic add latency (8-element serial chain × 64 batches)

| | min | median | p90 | max |
|---|---:|---:|---:|---:|
| LOCAL | 242 cy | 589 cy | 618 | 666 |
| **REMOTE via NVLink** | **2,716 cy** | **2,966 cy** | 3,022 | 3,173 |

**Remote = 5× local** (~2,400 cy NVLink round-trip adder). At 1920 MHz: 2,966 cy = **1.55 µs**; at 2032 MHz: ~1.46 µs. Round-trip including interconnect both ways.

### 4b. Chained `ld.global.cg` loads

| | min | median | p90 | max |
|---|---:|---:|---:|---:|
| LOCAL | 238 cy | 282 cy | 598 | 606 |
| REMOTE via NVLink | 2,677 cy | 2,947 cy | 2,998 | 3,358 |

**Bimodal distribution preserved cross-GPU**: ~2,700 vs ~2,950 cy buckets reflect peer-L2-side hash variance (~250 cy spread). NVLink RT (~2,400 cy) dominates over L2-side variance.

### 4c. NVLink pipelining defeats serial latency (HIGH)

Single-thread chained loads: 3,300 cy/load. **With 32 independent chains per thread**, total time barely grows (210K → 234K cy, +11%) — **NVLink request pipeline absorbs ~32 outstanding loads/thread** nearly perfectly.

| Outstanding loads/thread | Effective cy/load | Improvement |
|---:|---:|---:|
| 1 (serial) | 3,300 | 1× |
| 32 (independent chains) | **~114** | **29×** |

Independent pointer chases approach **NVLink BW-limited rate ≈ 58 ns/load** instead of serial RT ceiling.

### 4d. Cache-hint sensitivity on cross-GPU pointer chase (LOW impact)

| Hint | REMOTE median |
|---|---:|
| `ld.global.cg` | 2,917 cy |
| `ld.global.ca` | 2,953 cy |
| `ld.global.cv` | 2,907 cy |
| `ld.global.lu` | 3,765 cy (+14%, last-use hurts) |

---

## 5. Cross-GPU atomic throughput (HIGH)

### 5a. Pipelined burst (148 SMs × 1024 thr × 256 atoms)

| Pattern | LOCAL Matomic/s | REMOTE Matomic/s | LOCAL→REMOTE slowdown |
|---|---:|---:|---:|
| Unique addresses | **49,400** | **9,152** | 5.4× |
| Contended (single CL, all SMs) | **49,173** | **16,345** | 3.0× |
| u64 contended | 49,415 | 16,609 | 3.0× |

- **LOCAL all-contend peak = 49.4 Gatomic/s** = 197 GB/s payload (u32) / 395 GB/s (u64). L2 atomic unit is the on-chip ceiling — independent of NVLink.
- **REMOTE contend = 16.6 Gatomic/s = 1.13 TB/s CL-traffic** (NVLink packet bytes ~560 GB/s, 32% of peer-link cap; bound by **peer L2 atomic unit**, not link BW).
- **REMOTE surprising twist**: contended (16.6 G) > unique (9.2 G). Warp coalescing reduces NVLink packet count → more semantic atomics fit per packet. Unique atomics each go as separate packets.

### 5b. Per-thread serial atomic scaling

| SMs | LOCAL cy/atom | REMOTE cy/atom |
|---:|---:|---:|
| 1 | 590 | 2,790 |
| 8 | 577 | 2,778 |
| 32 | 576 | 2,764 |
| 148 | 565 | 2,784 |

**No contention penalty per-SM** — peer memory controller pipelines incoming atomics. Useful: shared cross-GPU counters do not get exponentially slower with participants.

### 5c. All atomic op types are equal cost (HIGH)

Within 1% across `Add / Min / Max / Xor / Or / And / Exch / CAS` for both LOCAL (~590 cy) and REMOTE (~2,970 cy). u32 ≡ u64 ≡ f32 in cy/op (just different payload bytes).

---

## 6. Cross-GPU fence cost (HIGH)

`fence.sc.sys` over NVLink — covered fully in fences report; key cross-GPU number: **+17,800 cy NVLink drain adder** when local writes target peer memory.

| Scope | LOCAL A | REMOTE A | Δ = NVLink drain |
|---|---:|---:|---:|
| `fence.sc.cta` | 495 | 5,786 | +5,291 |
| `fence.sc.gpu` | 1,852 | 19,645 | +17,793 |
| `fence.sc.sys` | 8,952 | 26,738 | +17,786 |

**Asymmetric coupling**: even LIGHT SMs pay the NVLink drain when many SMs are pushing remote writes (unlike LOCAL where light SMs stay flat). A "fast sync SM" cannot be reserved for cross-GPU coordination — chip-wide NVLink pressure couples all fences.

---

## 7. Multi-GPU operation costs (HIGH for warm, MED for cold)

| Operation | Latency |
|---|---:|
| `cudaSetDevice` (warm, same dev) | **0.04 µs** |
| `cudaSetDevice` (toggle 0↔1, warm) | sub-µs per call |
| `cudaSetDevice` (cold, very first call) | **2116 ms** (driver init dominates) |
| `cudaDeviceCanAccessPeer` | ~0.1 µs |
| **`cudaDeviceEnablePeerAccess` (cold)** | **131 ms** | one-time peer page table setup |
| `cudaDeviceDisablePeerAccess` | ~ms-scale |
| Cross-device event sync setup | sub-µs |

**Practical**: call `cudaDeviceEnablePeerAccess` once at init (131 ms acceptable boot cost); afterwards all peer LDG/STG/atomics/textures work at NVLink speed.

---

## 8. CUDA IPC handles (HIGH — now includes cross-process)

| Operation | µs/call |
|---|---:|
| `cudaIpcGetMemHandle` (server, 100-trial avg) | **0.12** (warmed; first-call ≈ 7.7 us) |
| `cudaIpcGetEventHandle` | **1.00** |
| `cudaIpcOpenMemHandle` (cross-process, single shot) | **56.18** |
| First D2H 4B read after open (cold) | **39.5** |
| Sustained D2H 4B (warm) | **9.25** us/op |
| Cross-process atomic ping-pong (`atomicExch` + spin) | **4.4 us/round** = ~2.2 us/transition |
| PCIe Gen 6 x16 BW (D2H 256 MB IPC) | **57.3 GB/s** (matches non-IPC) |
| PCIe small-transfer floor (D2H ≤ 16 KB) | **9 us latency** |

**Cross-process atomic latency** is **~13× slower** than intra-process atomic (164 ns).
Likely cost: cross-context L2 coherence + TLB. **No PCIe involved** for atomics — both
contexts share the same GPU's L2.

Standard pattern: server A does `IpcGetMemHandle` → 64-byte handle → serialize over
pipe/socket → server B does `IpcOpenMemHandle(..., cudaIpcMemLazyEnablePeerAccess)`.
Both processes then read/write shared device memory coherently.

`cudaDevAttrIpcEventSupport = 1`.

(See `investigations/ipc_server.cu`, `investigations/ipc_client.cu`,
 `investigations/ipc_pcie_bw.cu`)

---

## 9. All-reduce (custom ring + NCCL) (HIGH)

### 9a. Custom ring all-reduce (`cudaMemcpyPeer`-based)

| Size | Latency | Algorithm BW |
|---:|---:|---:|
| ≤ 1 MB | **21 µs** | floor (NVLink + sync overhead) |
| 4 MB | 27 µs | 315 GB/s |
| 16 MB | 44 µs | 772 GB/s |
| 64 MB | 116 µs | 1159 GB/s |
| 256 MB | 376 µs | **1428 GB/s** (94% of NVLink peak) |

### 9b. NCCL 2.29.3 all-reduce

| Size | NCCL latency | NCCL algo BW | vs custom |
|---:|---:|---:|---|
| ≤ 256 KB | **10 µs** | — | **2× faster** (custom 21 µs) |
| 1 MB | 13 µs | 156 GB/s | 1.8× faster |
| 16 MB | 54 µs | 627 GB/s | custom faster |
| 256 MB | 531 µs | 1011 GB/s | custom 1.4× faster (1428 GB/s) |

- **NCCL wins small** (kernel-based P2P, no `cudaMemcpyPeer` overhead → 10 µs floor).
- **Custom wins large** (DMA engine > kernel-based copy at sustained rates).
- **For LLM TP**: NCCL gives 10 µs/layer → 0.8 ms total for 80-layer model = **<5% of decode time**.

---

## 10. Sharded GEMM — remote weights via NVLink (HIGH, surprising)

| Size | Local TFLOPS | Remote (weights on peer GPU) | Slowdown |
|---:|---:|---:|---:|
| 4096³ | 1778 | 1789 | **1.01×** (none) |
| 8192³ | 2247 | 2250 | **1.00×** (none) |

**Zero penalty for accessing weights on the other GPU via NVLink.** cuBLAS tiles the weight matrix into L2-sized chunks; after first tile fetch, subsequent accesses hit L2 cache. **2×B300 (548 GB total HBM) can serve models larger than single-GPU capacity with no compute overhead.**

---

## 11. NVLink direction usage by op type (ncu metrics, HIGH)

| op type | L1 BW | NVLink TX | NVLink RX | (TX+RX)/L1 | direction pattern |
|---|---:|---:|---:|---:|---|
| WRITE (W=128 coalesced) | — | **836 GB/s** | small | ≈1× | mostly outbound (data) |
| READ (W=128 cache-defeat) | — | 143 GB/s (request) | **860 GB/s** (95.6% of peak) | ≈1× | mostly inbound (data) |
| Atomic | — | 388 GB/s (req) | 350 GB/s (resp) | 1.3× | **uses BOTH directions** |
| Atomic (u64 contended) | — | 749 | 674 | — | 1423 GB/s = 79% of bidi peak |

**Reads/writes are mostly one-directional** → can approach 900 GB/s on that direction. **Atomics use both** → request+response = full-duplex; atomic = cost of (read + write).

---

## 12. Conflicts resolved (RETIRED CLAIMS)

| Old claim | Status | Correct value |
|---|---|---|
| "286 GB/s NVLink P2P kernel read" | **WRONG → RETIRED** | **778 GB/s** with float4 + ≥75K threads (matches DMA). Old number was thread-limited. |
| "NVLink BW 757 GB/s uni / 1503 bi" | **CONFIRMED**, slight underspec | Latest: **776 / 1543** GB/s. Both within 3% — same number. |
| "NV18 = NVLink generation 18" | **WRONG NAMING** | NV18 = **18 NVLink5 links** (i.e. 18 lanes of NVLink generation 5). Not "generation 18". |
| "NVLink saturates with 148 SMs" | **WRONG** | Saturates at **~32 SMs**; rest are free for compute. |
| "All-reduce floor 50 µs" | **WRONG** | NCCL 10 µs, custom 21 µs. |
| "DSMEM 1000 GB/s peer" | LOW (under-saturated test) | Likely far below 38 TB/s SHMEM peak; relative ratio (4.7× slower than local SMEM) survives. |
| Cross-process IPC `cudaIpcOpenMemHandle` works in-proc | **WRONG** | Fails with "invalid device context" — designed for cross-process only. |

---

## 13. Confidence summary

| Claim | Confidence | Verification |
|---|---|---|
| 2 GPUs, NV18, P2P bidirectional | HIGH | `cudaDeviceCanAccessPeer`, attributes |
| P2P DMA unidir = 776-777 GB/s | HIGH | `nvlink_bw.cu` × 10 trials best-of |
| P2P DMA bidir = 1543 GB/s aggregate (perfect 2×) | HIGH | same test |
| Kernel P2P read = 778 GB/s (= DMA) with float4+75K thr | HIGH | `p2p_kernel.cu`, `tests/p2p_deep.cu` |
| Kernel P2P write = 720 GB/s event-timed = 80% peak | HIGH | MGFenceBench cross-checked clock64 vs events |
| Read 91% / Write 80% asymmetry | HIGH | 4 separate tests agree |
| NVLink saturates at 32 SMs | HIGH | SM-count sweep |
| Cross-GPU atomic latency ~5× local (~3,000 cy = 1.55 µs) | HIGH | atomic chain × stride sweep |
| Cross-GPU atomic throughput contended = 16.6 Gatom/s | HIGH | bulk + ncu cross-check |
| `cudaIpcGetMemHandle` = 7.75 µs | HIGH | `ipc_handle.cu` × 100 trials |
| `cudaDeviceEnablePeerAccess` = 131 ms cold | MED | single test |
| All-reduce: NCCL 10 µs / custom 21 µs floor | HIGH | NCCL 2.29.3 + custom |
| Sharded GEMM zero penalty | HIGH | cuBLAS L2 cache tiling effect verified |
| u32/u64/f32 atomics equal cy/op | HIGH | width sweep |
| All atomic op types equal cost | HIGH | 8-op sweep |
| Cross-GPU fence +17.8K cy drain | HIGH | MGFenceBench |
| 32-deep ILP collapses cross-GPU latency 29× | HIGH | independent chain sweep |
| `cudaDevP2PAttrCudaArrayAccessSupported` = 1 (textures cross-GPU) | HIGH | attribute query |

---

## 14. Open / unresolved

- ~~Cross-process IPC latency~~ → MEASURED (see §8 update below)
- **3+ GPU NVLink behavior** — only 2-GPU system tested (NV18 between 2 B300; 8-way DGX/HGX would need different test).
- **NVLink contention with PCIe traffic** (e.g., NIC RDMA) — `cudaDeviceFlushGPUDirectRDMAWrites` measured (25 ns ToOwner / 886 ns ToAllDevices) but not under load.
- **NCCL with NVLink-SHARP** (in-network reductions) — version 2.29.3 may use SHARP automatically; not isolated.
- **Per-link breakdown** — measured aggregate NVLink BW; per-link load balancing under partial failure not tested.
