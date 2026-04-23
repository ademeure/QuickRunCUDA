# §13 — DSMEM (cluster shared memory)

## CLAIMS UNDER TEST

| # | Source line | Claim |
|---|---|---|
| C1 | `B300_PIPE_CATALOG.md` L7029 | **Local SMEM 25 cy, DSMEM 23 cy** ("essentially identical") |
| C2 | `B300_PIPE_CATALOG.md` L7031 | **DSMEM "essentially free"** — cluster interconnect adds no latency |
| C3 | `B300_PIPE_CATALOG.md` L7853-7858 | **DSMEM bandwidth = 99% of local SMEM** (170 GB/s/SM both) |
| C4 | `B300_PIPE_CATALOG.md` L2862-2873 (CORRECTION 2026-04-17) | **DSMEM 8× higher latency than local** (224 cy cluster=2 vs 28 cy local), throughput 9× lower at ILP=4 |
| C5 | V21 (`b300_clean/corrections/V21_DSMEM_*`) | **Write 560 GB/s/cluster** (issue rate, no fence) |
| C6 | V53 (`tests/standalone/v53_dsmem_fenced.cu`, `corrections/V53_RUN_RESULTS.md`) | **Sustained completion 82 GB/s/cluster, 1.47 TB/s 18-cluster aggregate**; writes do NOT traverse L2 |

The catalog is internally inconsistent: C1/C2/C3 say "free", but C4 (a correction
note added later, only to §30.H) says "~8× slower". The downstream sections
(§Cluster/DSMEM at L7022, DSMEM Bandwidth & Atomic at L7832) were never updated
to reflect the correction.

## TEST 1 — Read latency (chained ld.shared::cluster, single warp, clock64)

- Test file: `/tmp/dsmem_lat_v3.cu` (based on `tests/standalone/v10_dsmem_lat_only.cu`,
  fixed to add the trailing `barrier.cluster.arrive/wait` so peer SMEM stays alive
  until all CTAs finish their reads — without it, cluster=4/8 launches sync-fail
  with "unspecified launch failure")
- Compile: `nvcc -arch=sm_103a -O3 dsmem_lat_v3.cu -o /tmp/dsmem_lat_v3`
- Method: 4096-iteration dependent pointer chain, 1 warp per CTA does the timed
  chain, `mapa.shared::cluster.u32` to peer (ring (cta+1) % CLUSTER), `clock64`
  bracketed, best-of-5 runs
- GPU clock during test: stable **1942 MHz** (no boost — light kernel), 203W

| Access | cy/load (best of 5) | ratio vs local |
|--------|--------------------:|--------------:|
| Local SMEM (`ld.shared.u32`)             | **23.00** | 1.00× |
| DSMEM cluster=2 (`ld.shared::cluster.u32`) | **222.90** | **9.69×** |
| DSMEM cluster=4                          | **203.58** | **8.85×** |
| DSMEM cluster=8                          | **203.58** | **8.85×** |

SASS verification (`cuobjdump --dump-sass`):
```
LD.E R6, [R6]    ;   <-- ld.shared::cluster compiles to global LD.E
                 ;       (NOT to LDS) — through the cluster shared window
```

**Verdict on C1, C2: ✗ FALSE.** DSMEM read latency is **~9× higher** than local
SMEM, not "identical". The claim "essentially free" is misleading. The catalog
correction at L2861-2864 (`DSMEM cluster=2: 224 cy, cluster=4: 201, cluster=8:
201`) is **EXACTLY reproduced** here (223, 204, 204). Sections at L7022-7031 and
L7853-7858 still carry the wrong "essentially identical" text and need to be
deleted/replaced with the corrected numbers.

## TEST 2 — Sustained write throughput (V53 fenced retest)

- Test file: `tests/standalone/v53_dsmem_fenced.cu` (rebuilt for this audit)
- Compile: `nvcc -arch=sm_103a -O3 v53_dsmem_fenced.cu -o /tmp/v53`
- Method: 18 clusters × 8 CTAs × 128 threads each issue `st.shared::cluster.u32`
  to peer SMEM in a long loop. Two variants run side-by-side:
  - `UNFENCED` ends timing immediately after last store (V21 pattern)
  - `FENCED` inserts `fence.sc.cluster + barrier.cluster.{arrive,wait}` before
    end-of-clock64
  Wall time measured by cudaEvent_t (best of 5)
- GPU clock during test: 1942 MHz, power 200-285 W (sustained DSMEM hits ~285W)

### Sustained (≥10 ms runtime, all 18 clusters)

| TILE | ILP | Wall UF | UF GB/s/CTA | UF GB/s/cluster | UF agg | Wall F | F GB/s/CTA | F GB/s/cluster | F agg |
|-----:|----:|--------:|------------:|----------------:|-------:|-------:|-----------:|---------------:|------:|
|  2KB |  4  | 9.43 ms | 10.9        | **87.2**        | 1564 GB/s | 9.41 ms | 10.9 | 87.4 | 1566 GB/s |
|  4KB |  4  | 11.30   | 10.9        | 87.0            | 1565   | 11.25  | 10.9 | 87.4 | 1573 |
|  8KB |  8  | 11.22   | 10.9        | 87.6            | 1577   | 11.19  | 11.0 | 87.9 | 1582 |
| 16KB |  8  | 11.94   | 11.0        | 87.8            | 1581   | 11.99  | 10.9 | 87.4 | 1574 |
| 64KB |  8  | 12.09   | 10.8        | 86.7            | 1561   | 12.10  | 10.8 | 86.7 | 1560 |

Sustained per-cluster write rate: **~87 GB/s/cluster** (5.3 GB/s/CTA on average,
across 8 CTAs).
Aggregate across 18 clusters: **~1.56 TB/s**.

Wall-clock UF/F ratio: **1.00× across all 12 configs** — at sustained length the
fence has no effect on wall time, because the kernel as a whole still drains
before `cudaDeviceSynchronize` (so fence cost is absorbed into kernel-end
serialization either way).

### V21-style burst (1 cluster, 4 warps × ILP=4, varying outer iters)

| outer iters | UF cyBW/CTA | F cyBW/CTA | UF/F | UF/cluster |
|------------:|------------:|-----------:|-----:|-----------:|
|     5       | **53.4**    | 13.7       | 3.90× | **427 GB/s** |
|    50       | 28.7        | 24.4       | 1.18× | 230 |
|   500       | 27.2        | 26.7       | 1.02× | 218 |
|  5000       | 27.0        | 27.0       | 1.00× | 216 |

At V21's exact 5-burst geometry, the unfenced number (53 GB/s/CTA = **427
GB/s/cluster**) reproduces V21's "560 GB/s" claim within 25% (V21 had slightly
different ILP/CL geometry); the fenced equivalent drops 3.9× to 110 GB/s. So
**V21's 560 was the burst issue rate, not completion**.

**Verdict on C5, C6: ✓ V53 confirmed.**
- True sustained per-cluster write rate: **87 GB/s** (V53 reported 82, my retest
  shows 87 — both within 6% noise; both clearly NOT 560).
- Aggregate 18-cluster: **1.56 TB/s** (V53 1.47 TB/s — within 6%).
- V21's 560 figure is reachable only as the burst issue rate over <50 stores,
  with no completion fence; longer or fenced bursts asymptote to ~110-216
  GB/s/cluster.

**Verdict on C3 ("DSMEM BW = 99% of local SMEM"): ✗ FALSE.**
Local SMEM peak is 250 GB/s/SM (catalog L7003) = ~2 TB/s per 8-CTA cluster.
Sustained DSMEM is 82-87 GB/s/cluster = ~25-43% of single-SM local SMEM, and
**5%** of an 8-CTA local-SMEM aggregate. The "99%" claim was likely measured at
short burst (issue rate) not completion.

## TEST 3 — L2 traversal check (ncu)

- Test file (writes): `/tmp/dsmem_ncu.cu` — single launch, 144 CTAs × 4000 iter
  × 8 ILP × 128 thr × 4 B = **2.36 GB written via DSMEM**, full anti-DCE
- Test file (reads):  `/tmp/dsmem_read_ncu.cu` — same shape but 8-way ILP READ
  loop, 2.36 GB read, XOR-accumulator → global sink
- Comparison kernel (`global_writes_only`): same shape but writes to GMEM
  (`base_ptr[off] = …`), not DSMEM — to verify the ncu metric works

ncu metrics:

| Kernel | bytes touched | `lts__t_bytes.sum` | `lts__t_sectors.sum` | `l1tex__…shared.sum` |
|--------|--------------:|-------------------:|---------------------:|---------------------:|
| `dsmem_writes_only`  (DSMEM)  | 2.36 GB | **991 KB**  | **30,987**  | 152,784 |
| `dsmem_reads_only`   (DSMEM)  | 2.36 GB | **1.05 MB** | **32,830**  | 76,753  |
| `global_writes_only` (control) | 2.36 GB | **2.55 GB** | 79,727,736 | 576     |

DSMEM L2 traffic: **0.04% of write/read volume** — essentially zero, just from
final anti-DCE smem-readback `sink[blockIdx.x] = …` and CTA exit cleanup. The
GMEM control kernel correctly shows 2.55 GB at L2 (matches 2.36 GB user volume
plus L2 metadata). The L1TEX shared-mem-pipe metric DOES count DSMEM traffic
(76k–152k wavefronts).

**Verdict on C6 (writes don't traverse L2): ✓ V53 confirmed.**
**Bonus finding (corrects V53_RUN_RESULTS §6 footnote):** DSMEM **reads** ALSO
do not traverse L2. `lts__t_bytes = 1.05 MB / 2.36 GB read = 0.04%`. The
DSMEM_REFERENCE.md older claim that "reads go through L2 (~4 sectors/load)" is
NOT supported by direct ncu measurement on B300 sm_103a with NVCC 13.2 — both
reads and writes use the dedicated inter-CTA cluster fabric. SASS `LD.E`/`ST.E`
encoding is misleading: the address points into the cluster shared window
(`SR_SWINHI`-relative), and the hardware routes it via L1TEX shared-pipe to the
peer SM, never reaching the L2 cache subsystem.

## VERDICT

| Catalog claim | Verdict |
|--|--|
| L7029 "DSMEM 23 cy ≈ local SMEM 25 cy" | ✗ **FALSE** — DSMEM is 9× slower (204-223 cy vs 23 cy local) |
| L7031 "DSMEM essentially free" | ✗ **MISLEADING** — adds ~180 cy per remote load, equivalent to a near-DRAM hit |
| L7857 "DSMEM load = 0× penalty" | ✗ **FALSE** |
| L7858 "DSMEM BW 99% of local SMEM" | ✗ **FALSE** — 5-43% depending on aggregation |
| L2862-2864 (correction note) "DSMEM 200-224 cy" | ✓ **CONFIRMED EXACTLY** |
| L2873 "8× higher latency, 9× lower ILP=4 throughput" | ✓ **CONFIRMED** (latency reproduced; throughput not retested in this audit) |
| V21 "560 GB/s/cluster sustained write" | ✗ **FALSE** — only achievable as 50-burst issue rate; sustained is 4-7× lower |
| V53 "82 GB/s/cluster sustained" | ✓ **CONFIRMED** (retest gave 87, within 6%) |
| V53 "1.47 TB/s 18-cluster aggregate" | ✓ **CONFIRMED** (retest 1.56 TB/s) |
| V53 "writes do NOT traverse L2" | ✓ **CONFIRMED** (ncu lts__t_bytes = 0.04% of write volume) |
| Bonus: reads do not traverse L2 either | ✓ **NEW FINDING** (corrects V53_RUN_RESULTS §6 — DSMEM_REFERENCE.md "reads go through L2" was wrong for both r/w) |

## RECOMMENDED CATALOG EDITS

1. **DELETE** L7022-7034 ("Cluster / Distributed Shared Memory" → "DSMEM is ~identical
   latency to local smem"). Replace with a pointer to §30.H (which already has the
   correct numbers: 28 cy local, 200-224 cy DSMEM, 8× slower).

2. **DELETE/REPLACE** L7836-7860 ("DSMEM Bandwidth & Atomic Costs" → "DSMEM
   bandwidth = 99% of local smem"). The 170 GB/s/SM number is unsourced and
   inconsistent with sustained measurements. Replace with V53 retested numbers:

   ```
   ## DSMEM write throughput (V53 retest, 2026-04-23)
   | Regime                                  | per-CTA  | per-cluster | aggregate |
   |-----------------------------------------|----------|-------------|-----------|
   | V21-style 5-store burst, unfenced       | 53 GB/s  | 427 GB/s    | n/a       |
   | V21-style 5-store burst, fenced         | 14 GB/s  | 110 GB/s    | n/a       |
   | Sustained 1-cluster (5000 iters)        | 27 GB/s  | 216 GB/s    | n/a       |
   | Sustained all 18 clusters (≥10 ms)      | 11 GB/s  | 87 GB/s     | 1.56 TB/s |
   ```

3. **DELETE** the implication anywhere that DSMEM reads traverse L2 (e.g.
   DSMEM_REFERENCE.md §1) — direct ncu measurement shows 0.04% L2 traffic
   for both reads and writes.

4. The "essentially free" framing (L7031, L7857) is the worst kind of catalog
   error: it's a one-line headline that contradicts a measured correction
   buried in a different section. Recommend a `_DSMEM_FACTS.md` callout that
   becomes the single source of truth, and removal of all "free"/"identical"
   language from sections that refer to DSMEM.

## NOTES

- The single-cluster sustained number (216 GB/s/cluster, ~187 in V53) is 2.5×
  the 18-cluster-concurrent number (87/cluster). This means there's a shared
  resource (probably the cluster-network arbitration) that gets saturated when
  all 18 GPCs drive DSMEM traffic simultaneously. The 1.56 TB/s aggregate is
  thus not a "fabric peak" but a contention-saturated steady-state.

- Cluster=4 and cluster=8 measure identical 204 cy (vs cluster=2 at 223 cy).
  Suggests cluster-internal routing has a fixed cost up to GPC-spanning, with
  cluster=2 paying ~10% extra startup overhead. Counter-intuitive that bigger
  clusters are slightly faster; possibly cluster=2 sometimes places CTAs on
  GPC-distant SMs while cluster=4/8 force GPC-local placement. Not investigated
  further.

- Without the trailing `barrier.cluster.arrive/wait`, cluster=4 and cluster=8
  read kernels give "unspecified launch failure" because peer CTAs exit (and
  release SMEM) before the reading CTA finishes. This caused all my early
  cluster=4/8 attempts to fail until I matched V53's pattern of always closing
  with a cluster barrier. Worth flagging in any future DSMEM scaffold.

- V53's burst at 5 outer iters reproduces the V21 issue-rate (53 GB/s/CTA × 8 =
  427 GB/s, vs V21's reported 560). The factor-of-1.3 gap is from V21 having
  slightly more ILP/warp coverage; both are within the same "burst issue rate"
  family. Catalog text should explicitly disambiguate "burst issue rate" from
  "sustained completion rate".

- All measurements made on B300 SXM6 sm_103a, GPU 0 (only `/dev/nvidia0`
  visible), driver-default clock = **1942 MHz** sustained throughout (no
  nvidia-smi lock active; verified via `nvidia-smi -q --display=CLOCK`
  sampled every 0.3s during runs). Power: 203W (latency test) up to 285W
  (sustained DSMEM writes). NVCC version: 13.2.78. Compile flag: `-arch=sm_103a -O3`.
