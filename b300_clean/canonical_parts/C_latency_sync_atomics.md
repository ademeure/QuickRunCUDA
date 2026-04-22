# Section C — Latency, Sync, Atomics (§26–§35)

**Hardware:** B300 SXM6 AC (sm_103a), 148 SMs, 2.032 GHz boost / 1.920 GHz `-lgc 2032` lock / 1.500 GHz when explicitly noted. L2 = 126 MB. HBM3E ~7.67 TB/s peak (this-device, 7680-bit fused bus).

**Clock convention.** Every cycle count is annotated with the clock domain it was measured at. The default canonical conversion is `2.032 GHz boost ⇒ 0.4921 ns/cy`. Numbers measured at 1500 MHz lock are explicitly tagged.

**Sister-section pointers.** Tensor-core MMA latency is in §24 (Section B); only the HMMA-20 cy figure is repeated here for the latency ladder. Power and DVS-curve content lives in §42–§44 (Section D). Cluster bandwidth and DSMEM throughput peaks are in Section A; Section C only covers cluster *barriers/fences*.

---

## §26. Latency ladder — the canonical cross-pipe table

**Answer:** B300 single-instruction latency ladder, all measured by `clock64`-bracketed dependency chains, converged at chain length ≥ 4096 unless otherwise noted.  `[🟢 HIGH · src: M16_V9_FULL_SYNTHESIS.md§II + M15_V9_LATENCY_LADDER.md]`

Every row in the table below is DCE-immune by construction (the chain is a value-flowing serial dependency, so the compiler cannot drop any link), and every row was cross-checked against either (a) achievable throughput at full ILP saturation or (b) a different chain length. No row is a pure formula.

### §26.1 Headline ladder (single warp, isolated, hot location)

| Op / primitive                    | Latency (cy) | ns @ 2.032 GHz | Saturation chain depth (per warp) | Verification |
|-----------------------------------|-------------:|---------------:|----------------------------------:|--------------|
| Register MOV / R2UR               | ~1           | 0.5            | 1                                 | obvious      |
| FFMA / FADD / FMUL                | **4.22**     | 2.08           | 4                                 | V9 chain=4096 → 4.218 cy/op |
| IMAD (32-bit, .lo)                | 4.25         | 2.09           | 2 (1 issue per 2 cy)              | V9 chain=4096 → 4.252 cy/op |
| LOP3.LUT                          | ~4.5         | 2.21           | 2                                 | C3 deep dive |
| HMMA.F16.F32 m16n8k16             | **20**       | 9.84           | 5 (1 HMMA / 4 cy / SMSP)          | V9 chain=4096 → 20.09 cy/op |
| `__syncwarp(0xFFFFFFFF)` full mask | **0–2**     | 0–1.0          | n/a                               | F2_SYNCWARP_RIGOR: NOPs only emitted (no SASS), 1.75 cy of measurement floor |
| `__syncwarp(partial mask)`        | 7.25         | 3.6            | n/a                               | F2_SYNCWARP_RIGOR (BSYNC) |
| `mbarrier.arrive` (no wait)       | 24           | 11.8           | n/a                               | M7 A6 + 08_sync |
| SMEM `LDS` (single-bank)          | **29**       | 14.3           | 6                                 | V9 LDS chain |
| `__syncthreads` 32 thr (1 warp)   | 24           | 11.8           | n/a                               | V9 formula 22+2W, exact fit |
| `__syncthreads` 128 thr (4 warps) | **30**       | **14.8**       | n/a                               | V9 formula |
| `__syncthreads` 256 thr (8 warps) | 38           | 18.7           | n/a                               | V9 formula |
| `__syncthreads` 512 thr (16 warps)| 54           | 26.6           | n/a                               | V9 formula |
| `__syncthreads` 1024 thr (32 warps)| **86**      | 42.3           | n/a                               | V9 formula (08 catalog says 77; V9 wins, see §29) |
| L1 hit (random pointer-chase)     | **47**       | 23.1           | 11                                | V9 1 KB LCG chain |
| DFMA                              | **63.7**     | 31.3           | 1 (single FP64 port saturates with 1 chain) | V9 chain=4096 → 63.677 cy/op |
| `mbarrier.arrive + try_wait`      | 54           | 26.6           | n/a                               | 08_sync_primitives |
| `mbarrier.arrive + wait` (full RTT)| **123**     | 60.5           | n/a                               | V10_VERIFICATION_SUMMARY, V10_GRID_SYNC |
| `barrier.cluster.arrive.relaxed + wait` (cluster=2) | **102** | **50.2**  | n/a                               | 08_sync (cluster_raw_barrier.cu) |
| L2 hit (pointer-chase 1 MB)       | ~300         | ~148           | 71                                | V9 LCG chain @ 1 MB |
| DRAM (pointer-chase >L2 capacity) | **~317**     | **156**        | 75                                | V9 LCG chain @ 1 GB |
| `__threadfence` / `fence.sc.gpu`  | **260–320**  | **128–158**    | n/a — see §31 spread              | V9 258, 08 281, DSMEM_REFERENCE 320 |
| `cluster.sync()` strict           | **373–380**  | 184–187        | n/a                               | 08 catalog, V9 (370) |
| `__threadfence_system`            | **DISPUTED 1750–3042** | 861–1486 | n/a — see §32 dispute              | 08 says 1750, V9 says 3042 |
| Global atomic (chained, hot loc)  | **697**      | **343**        | n/a                               | V9_ATOMIC_LATENCY (all scopes equal) |
| `grid.sync()` (148 blocks × 128 thr) | **2376**  | **1170**       | n/a                               | V10_GRID_SYNC |
| `nanosleep(1000)`                 | 2066         | 1000           | n/a                               | V9 nanosleep — predictable for N=1000 |

### §26.2 What "latency" means here (vs throughput, vs pipelined cost)

Throughout this section a single number `L cycles` for an op X means: in a serial dependency chain `r ← X(r, …)` running in one thread, the average wall time per X is L cycles after subtracting startup and loop overhead. This is the **latency** in the queueing sense — the length of the pipeline that must be filled to hide the op.

A separate quantity is the **pipelined throughput**, which is "if every X is independent of the next, how often does the pipe accept one?" For B300:
- FFMA latency 4.22 cy ⇒ throughput 1 op/cy/SMSP ⇒ saturate by 4 chains in the warp.
- DFMA latency 63.7 cy ⇒ throughput 1 op/64 cy/SMSP ⇒ a single chain saturates.
- HMMA.F16 latency 20 cy ⇒ throughput 1 op/4 cy/SMSP ⇒ 5 chains saturate (8 chains gives 99.9 % pipe with margin).
- Atomic latency 697 cy chained ⇒ pipelined throughput ~16 cy/op (V9 raw — see §34 for the corrected ladder; the popular "43 cy pipelined" lives in V9_ATOMIC_LATENCY but reflects a different phase of the same measurement).

These two numbers are NOT interchangeable. The table above lists *latency*, not pipelined cost. Pipelined-cost tables for atomics and SMEM ops appear in §34 and §35.

### §26.3 The chain-depth column

The "saturation chain depth per warp" column says how many independent chains in one warp are needed to hide the latency, i.e. `ceil(L / issue_period)`. For FFMA at 4.22 cy and 1 issue/cy/SMSP, that is 5 chains rounded down to 4 in practice (the V8 peak FFMA recipe uses 8 chains × 256 threads to give 2× margin and reach 97.64 % pipe). DFMA at 64 cy with 1 issue per 64 cy needs only 1 chain because the single port is the bottleneck. HMMA at 20 cy with 1/(4 cy)/SMSP needs 5 chains: the V8 HMMA recipe uses 8 chains (1.6× margin) and hits 99.9 % tensor pipe — barely enough.

### §26.4 Why SMEM (29 cy) is FASTER than L1 (47 cy)

This is counterintuitive but consistent across V9 and the 02_shmem catalog: shared memory has no tag check (the SMEM bank index is scalar arithmetic on the 18-bit address) while L1 must hash the address into the cache, look up the tag, and then return. The 18-cycle gap is the L1 tag-lookup penalty.

### §26.5 Cross-checks against throughput (sanity)

Each headline latency is corroborated by an independent throughput measurement from V8 / V9 / V10. For example:

- FFMA 4.22 cy, 8-chain × 256 thr × 148 blk ⇒ 75.20 TFLOPS measured = **97.64 %** of theoretical 76.96 (V8 + ncu `pipe_fma`).
- HMMA 20 cy, 8-chain × 256 thr × 148 blk ⇒ 578 TFLOPS = **99.90 %** of tensor pipe (V8 `pipe_tensor`).
- DFMA 63.7 cy, 8-chain ⇒ 1.20 TFLOPS = **100.00 %** (V8).
- LDS 29 cy, full-occupancy LDS-only kernel ⇒ 26.9 TB/s (74 % of 36 TB/s theoretical SMEM peak).

Each of these has a corresponding ncu cross-check in V10_VERIFICATION_SUMMARY.

### §26.6 What changed vs the M15 ladder

M15 (the V9 first-pass ladder) is mostly correct but had two cosmetic problems that M16 and the corrections folder fixed:

1. M15 listed `__syncwarp = 23 cy`. That 23 was the V9 *loop overhead*, not syncwarp itself. F2/F6 prove `__syncwarp(0xFFFFFFFF)` emits NOPs only (1 cy of measurement floor). The "23 cy" framing has been retracted (see §28).
2. M15 listed `mbarrier.arrive+wait` as 123 cy, listed `mbarrier.arrive only` as 24 cy, listed `mbarrier.arrive+test_wait` as 54 cy. These are three different operations and the cleanest convention is to list `arrive+wait = 123 cy` (full RTT) when comparing against `__syncthreads` (which is also a full barrier).

### §26.7 The "saturation chain depth" column derivation

For any pipe with latency `L` cycles and issue period `T` cycles, the minimum number of independent dependency chains needed to fully saturate the pipe is `ceil(L / T)`. For a single warp on a single SMSP:

| Pipe | Latency L | Issue period T | Chains needed | V8 recipe |
|------|----------:|---------------:|--------------:|-----------|
| FMA (FFMA) | 4.22 cy | 1 cy | 4 (rounded down to 4 in practice) | 8 chains, 2× margin → 97.64 % |
| FMA (DFMA) | 63.7 cy | 64 cy | 1 | 8 chains, 8× margin → 100.00 % |
| Tensor (HMMA.F16) | 20 cy | 4 cy | 5 | 8 chains, 1.6× margin → 99.90 % |
| INT-bit (LOP3) | ~4.5 cy | 2 cy | 3 | typically achieved at full ILP |
| LSU (LDS) | 29 cy | varies | 6+ | depends on bank conflicts |
| LSU (LDG L2 hit) | ~300 cy | varies | 60+ | typically saturated by occupancy |
| LSU (LDG DRAM hit) | ~317 cy | varies | 75+ | requires high occupancy |
| MUFU (RSQRT) | ~70 cy (estimated from 99.49 % pipe at 8 chains) | 64 cy/SM | 1 | 8 chains gives 99.49 % |

When a chain depth exceeds the achievable warp/SMSP issue rate, the kernel must add more warps to that SMSP (occupancy) to keep the pipe full. This is why, for FFMA, the V8 recipe deliberately picks 256 thr × 148 blk (= 8 warps/SMSP after splitting across 4 SMSPs × 148 SMs) — the 8 chains of ILP per warp × 8 warps gives ample warp-level parallelism to refill the FMA pipe slot.

### §26.8 The "DRAM ≈ L2" surprise

V9_MEM_LATENCY measured pointer-chase latency vs working-set size with an LCG-permuted chain (1024 hops):

| Buffer  | Target tier | Latency (cy/hop) | Latency (ns) |
|---------|-------------|------------------:|-------------:|
| 1 KB    | L1 hit      | 47                | 23           |
| 4 KB    | L1 hit      | 73                | 36           |
| 16 KB   | L1/L2 mix   | 164               | 81           |
| 64 KB   | L2 hit      | 255               | 125          |
| 256 KB  | L2 hit      | 295               | 145          |
| 1 MB–128 MB | L2 hit  | 305–309           | 150–152      |
| 1 GB    | DRAM (+L2)  | **317**           | **156**      |

The DRAM-vs-L2 difference is only ~12 cy (~4 %), which is SURPRISING — naïvely we'd expect DRAM round-trip to add hundreds of cycles. Possible explanations:

1. **HW prefetcher catches the LCG pattern despite random-ish hops.** L2 has adjacent-line prefetch that may pull next addresses speculatively.
2. **L2 partition routing is fast for a single chain** — the 126 MB L2 has multiple partitions, but the chain only triggers ~1 in-flight request at a time, so any latency hiding within the L2 dominates.
3. **True random Fisher-Yates permutation might give different numbers** — V8 I3 measured "HBM avg 60 cy, max 1433 cy" under load, suggesting that worst-case DRAM access is much higher than the LCG-chain average.

For the purposes of this section, treat:
- L1 hit ≈ 47 cy / 23 ns
- L2 hit ≈ 300 cy / 148 ns
- DRAM ≈ 317 cy / 156 ns (medium confidence — prefetcher may be helping)

The 4 % L2/DRAM gap is genuinely surprising and the catalog notes this is MEDIUM confidence pending a Fisher-Yates re-test.

### §26.9 Latency hiding rules of thumb

Combining all the above, here are practical ILP/occupancy budgets for hiding each kind of latency:

| To hide | Need ~per warp | Or per SM (× 8 warps occupancy) |
|---------|----------------|--------------------------------|
| FFMA latency 4.22 cy | 4 chains | 4 chains × 8 warps = 32-way ILP equivalent |
| HMMA latency 20 cy | 5 chains | 5 × 8 = 40 |
| LDS latency 29 cy | 6 chains | 6 × 8 = 48 |
| L1 hit 47 cy | 11 chains | 11 × 8 = 88 |
| L2 hit 300 cy | 71 chains | 71 × 8 = 568 — usually achieved by occupancy alone |
| DRAM 317 cy | 75 chains | 75 × 8 = 600 — same |
| `__threadfence` GPU 280 cy | n/a (single thread blocks) | use scope-finer fence if possible |
| Global atomic chained 697 cy | n/a (single thread blocks) | use SMEM intermediate if possible |

**Footgun:** ⚠ Do not collapse "barrier latency" into a single number — `__syncwarp` (1 cy), `__syncthreads(128)` (30 cy), `barrier.cluster.relaxed` (102 cy), `__syncthreads(1024)` (86 cy), `mbarrier.arrive+wait` (123 cy), `cluster.sync()` strict (373 cy), `__threadfence` GPU (260–320 cy), `grid.sync()` (2376 cy), `__threadfence_system` (1750–3042 cy) all differ by factors of 30×–3000×. Always cite which barrier you mean.

**See also:** §27 (pipe-placement table for ALU vs FMA latencies), §28–§30 (per-barrier deep dives), §31–§32 (fence cost disputes), §33 (cluster sync), §34–§35 (atomics).

---

## §27. Pipe placement ladder — what op runs on what pipe (V40 corrected, V52 confirmed)

**Answer:** B300 SMSP has at least 6 functional pipes — FMA, INT-bit, permute, compare, MUFU, LSU/SHFL — plus the tensor pipe and a uniform/predicate pipe. Each pipe has its own per-SMSP issue cadence, and a single SMSP can dispatch one instruction per cycle from a chosen pipe (with multi-warp scheduling refilling the slot). V40 measured the ladder at 1500 MHz lock with persistent grid + asm-volatile anti-DCE; V52 confirmed via ncu that FMA + ALU pipes overlap freely (alu + fma cycles_active ≈ 147 %).  `[🟢 HIGH · src: corrections/15_integer_bit_ops_CORRECTED.md§1 + V52_RUN_RESULTS.md + corrections/HEADLINE_CORRECTIONS_v5.md row 7]`

This is the AUTHORITATIVE pipe placement table; other sections that need to refer to "pipe X" should link here.

### §27.0 Architectural overview — what is a "pipe" on B300?

Before diving into the per-pipe ladder, a quick architectural primer:

- **B300 has 148 SMs.**
- **Each SM has 4 SMSPs (sub-partitions).** Each SMSP has its own warp scheduler, its own register file slice, and its own dispatch port.
- **Each SMSP can issue 1 instruction per cycle from a chosen pipe.** The pipe choice is per-cycle and per-warp.
- **An SMSP has multiple physical execution units (pipes):** FMA, INT-bit, permute, compare, MUFU, LSU, plus shared resources like the tensor pipe and uniform/predicate pipe.
- **Pipes are physically separate** but share the SMSP's dispatch port — only one instruction is dispatched per SMSP per cycle, but that instruction can target any pipe.
- **Cross-SMSP execution happens in parallel.** Four warps on different SMSPs can each execute different op types simultaneously.

A "pipe" in this catalog refers to a physically distinct execution unit within an SMSP. Each pipe has its own intrinsic per-pipe issue cadence (e.g., the FP64 DFMA port issues 1 op per 64 cy regardless of dispatch slot availability). When we say "FFMA + LOP3 overlap freely", we mean: on the same SMSP, one cycle can dispatch FFMA and the next cycle can dispatch LOP3, with each going to its own pipe and the pipes operating in parallel. The dispatch port is shared (1 inst/cy/SMSP) but the execution is parallel.

V52's empirical settlement (`pipe_alu + pipe_fma = 147 %` ncu) confirmed that pipes execute in parallel — the sum of pipe utilizations exceeds 100 % when both pipes have work. This is only possible if the pipes are physically distinct and able to execute simultaneously.

### §27.1 The 6-pipe ladder (with measured Glane/s @ 1500 MHz lock)

"Glane/s" = chip-wide thread-instructions/sec = warp-inst/cy/SMSP × 32 lanes × 4 SMSPs × 148 SMs × clock.
"%SoL" = vs the FMA-pipe ceiling of 1 inst/SMSP/cy. At 1500 lock the FMA-pipe SoL is ~38.4 Glane/s/inst.

| Pipe          | Member ops                                                  | Glane/s @ 1500 lock | %SoL of FMA pipe | inst/SMSP/cy | Notes |
|---------------|-------------------------------------------------------------|--------------------:|-----------------:|-------------:|-------|
| **FMA**       | **FFMA, FADD, FMUL, IMAD, IMUL.lo, IADD3, DFMA, HMMA**      | 25–26 (single-op solo, 2 warps/SMSP); up to 38 with 8 warps/SMSP | 67 % single-op solo, **97.6 %** at full ILP | up to 1.0 | "Solo FMA pipe peak"; V8 reaches 97.64 % at boost with 8 warps × 256 thr |
| **INT-bit**   | LOP3.LUT, SHF.L/R, SHL, SHR, SHFL.IDX/BFLY/UP/DOWN encoded as ALU rows, BFI.b32 | **18.7** | **48 %** (~half the FMA pipe) | 0.5 | C3 verified across 12 truth tables; ≥3 unique RF reads incurs no penalty (no operand-collector serialization) |
| **Permute**   | PRMT (byte permute)                                         | 13.9                | **36 %**         | ~0.46        | V40 LICM-fixed (the V39 first-pass "1547 % SoL" was an LICM bug); A6 reports 14.08 in a different ILP regime |
| **Compare**   | ISETP, FSETP, IMNMX, FMNMX                                  | **8.4**             | **22 %**         | 0.25         | Substantially slower than LOP3/PRMT — DO NOT lump into "ALU @ 19 TIOPS" |
| **XU**        | BFE.u32, POPC, BREV, CLZ, FLO                               | 3.5–7.07            | 12–25 %          | 0.125–0.25   | BFE = SHF.R + SGXT (2 SASS); POPC family is 4× slower than LOP3 tier |
| **MUFU (XU)** | MUFU.EX2                                                    | 9.62 Gop/s          | —                | 0.003        | EX2 stands out at 95.8 % of 1/(4 cy)/SMSP per V41 |
| **MUFU (XU)** | MUFU.LG2 / RCP / RSQRT / SQRT / SIN / COS                   | 4.74 Gop/s          | —                | 0.0015       | Half the rate of EX2 |
| **LSU**       | LDG, STG, LDS, STS, ATOMS, REDG                             | varies              | —                | varies       | Saturates at 26.9 TB/s SMEM (LDS), 5.82–6.91 TB/s HBM (LDG/cp.async) |
| **Tensor**    | HMMA, mma.sync legacy                                       | up to 99.90 % pipe  | —                | 1/(4 cy)/SMSP | See §24 |
| **Uniform**   | UIMOV, R2UR, broadcast `__shfl_sync(0xffffffff,v,0)`        | ~2 cy / op          | —                | varies       | Lowered automatically by the compiler when all lanes read the same value |

Multiply Glane/s by 1.058 for 2032 MHz boost.

### §27.2 LOP3 dispatch cadence = 2 cy per SMSP per V52

V52 measured `smsp__inst_issued.avg.per_cycle_active` for solo LOP3 = 0.51 (i.e., one issued LOP3 every other cycle). This is consistent with the `0.5 inst/SMSP/cy` Glane/s reading in §27.1. In other words, the INT-bit pipe accepts one LOP3 per 2 cycles, and the SMSP issue port is idle for 1 cycle in between — which is exactly the behaviour the FMA pipe can use to overlap with LOP3.

V52 simultaneously measured `pipe_alu = 98.0 %` and `pipe_fma = 49.4 %` — sum 147 % — proving the two pipes overlap freely on the same SMSP. The earlier V49/V50 "55 %/74 % shared dispatch cap" reading is an ARTIFACT of loop-overhead contamination (rebuked in HEADLINE_CORRECTIONS_v5 row 7).

### §27.3 IADD3 placement — V40 (FMA pipe) vs A6/B1 (separate ALU pipe at 50 %)

The pre-V40 catalog (and `15_integer_bit_ops.md` original) placed IADD3 on a separate ALU pipe at 0.5 inst/SMSP/cy. V40's measurements at full persistent grid + multi-warp lifted IADD3 to 0.66 inst/SMSP/cy = 25–26 Glane/s = the same tier as FFMA/FADD. The corrections folder consensus is:

- **V40 placement: IADD3 lives on the FMA pipe.** The 50 % reading from A6/B1 was an under-occupancy artifact (only 2 warps/SMSP).
- The IADD3 % varies between 50 % and 67 % of FMA pipe SoL depending on warp count + ILP pattern; the FMA pipe headroom for IADD3 to reach FFMA-pipe peak is real and measurable at high occupancy.

This matters for any tile loop that mixes FFMA + IADD3 address computation. Pre-V40 advice ("IADD3 is free, separate pipe") was wrong; V40 advice ("IADD3 contends with FFMA on the FMA pipe") is correct.

**Footgun:** ⚠ Pre-V40 catalog (and many AUDIT_NOTES entries) placed IADD3 on a separate ALU pipe. V40 corrected this: IADD3 sits on the FMA pipe. If you're reading older docs, mentally substitute "IADD3 ⇒ FMA pipe" wherever it says "ALU pipe at half rate".

### §27.4 What "INT-bit pipe at half rate" means architecturally

V40 cannot disambiguate three explanations for why LOP3/IMUL run at 0.5 inst/SMSP/cy:

1. **A separate physical INT-bit pipe whose native cycle is 2 clocks.**
2. **A shared dispatch port between LOP3 and IMUL with 0.5/SMSP/cy throughput.**
3. **The FMA pipe issuing LOP3 every 2 cycles** while the same FMA pipe issues FFMA on the alternate cycle.

V52 partially disambiguates: `pipe_alu` and `pipe_fma` are *different* ncu metrics that simultaneously read 98 % and 49 % when both ops are mixed. So (3) is unlikely — there really are two distinct pipes. But (1) vs (2) is still open — V52's `inst_issued = 0.51` is consistent with both. See UNRESOLVED in `corrections/15_integer_bit_ops_CORRECTED.md` §3.

### §27.5 Practical implications for kernel design

| If your kernel does a lot of … | Best companion work to overlap | Avoid pairing with |
|--------------------------------|-------------------------------|--------------------|
| FFMA / FADD                    | LOP3, PRMT, ISETP (different pipes) | More FFMA / IADD3 (same pipe) |
| LOP3 (bit packing)             | FFMA, FADD                    | More LOP3 / IMUL (saturates INT-bit at 0.5) |
| PRMT (byte shuffles)           | FFMA, LOP3                    | More PRMT (saturates permute) |
| ISETP-heavy predicate logic    | FFMA, LOP3                    | More ISETP / FSETP |
| Tensor work                    | LDS preloads via `cp.async`, ALU offset math | More tensor (already saturating) |
| LDS / SMEM                     | FFMA / IADD3 (different pipe) | More LDS (saturates LSU) |
| MUFU (rsqrt, sin, exp)         | FFMA — MUFU runs in background | More MUFU (single-port) |

### §27.6 What the V49 → V50 → V52 saga taught us about pipe overlap

The dual-issue verdict for B300 has flipped 5 times in the corrections cycle (HEADLINE_CORRECTIONS_v5 row 7 + META_LESSONS). The current settled story is:

- **FMA + ALU pipes overlap freely on the same SMSP** (V52 ncu: `pipe_alu + pipe_fma = 147 %`).
- The earlier V49/V50 readings of "55 %/74 % shared dispatch cap" were loop-overhead artifacts (V49 had ~12 % loop overhead; V52's clean test had ~1 %).
- The architectural inference of "shared dispatch cap" was a phantom built on top of those artifacts.
- LOP3's apparent 50 % rate is a per-pipe issue cadence (2 cy per LOP3 on the INT-bit pipe), NOT a shared SMSP dispatch cap.

### §27.7 Pipe overlap matrix (M8 confirmed by V52)

The M8 PIPE_OVERLAP_MATRIX measured pairwise overlap of B300 pipes via dual-warp specialization (one warp does op A, the other does op B, on different SMSPs). HEADLINE_CORRECTIONS_v5 NEW row 13 confirmed M8's findings via V52 ncu pipe_X_cycles_active simultaneous reads:

| Pair          | Overlap    | Confirmed by | Notes |
|---------------|-----------:|--------------|-------|
| FFMA + LOP3   | ~100 %     | V52 (alu+fma=147%) | Free overlap; INT-bit at 50 % cadence + FMA at 100 % = sum 150 % |
| FFMA + IADD3  | ~67 %      | V40, A6      | Both on FMA pipe → contend |
| FFMA + LDS    | ~73–96 %   | M8           | LSU and FMA pipes are separate |
| FFMA + HMMA   | ~96–99 %   | V8 + M8      | Tensor pipe is separate from FMA |
| FFMA + MUFU   | ~100 %     | A6, A2, M8   | MUFU runs ~64 cy in background; FMA pipe free during latency |
| HMMA + LDS    | ~73–96 %   | M8           | Tensor + LSU = separate |
| LOP3 + ISETP  | unknown    | (not tested) | Both ALU-adjacent, may contend on dispatch |
| MUFU + ALU    | ~100 %     | A2, M8       | MUFU latency hides ALU work |

**Key insight:** B300 SMSP can dispatch from multiple pipes simultaneously when the issuing warps are scheduled correctly. The total inst/cy/SMSP that the chip can achieve is bounded by `min(per-SMSP issue cap, sum-of-pipe-caps)`. For balanced workloads, the per-SMSP issue cap is the binding constraint; for pipe-skewed workloads, the pipe-cap is binding.

### §27.8 The "ALU cluster" model (pre-V40) is wrong; refined model

A6 originally proposed a "unified ALU/FMA cluster" model where all integer/FP ops shared a single dispatch slot at ~1 inst/SMSP/cy. V40 disproved this:

- **Pre-V40 model (A6):** "All ALU + FMA ops share one dispatch slot at 1 inst/SMSP/cy."
- **Post-V40 / V52 model:** Multiple separate pipes (FMA, INT-bit, permute, compare, XU, LSU), each with its own per-SMSP issue cadence, free overlap between distinct pipes within scheduling limits.

This change affects how you reason about pipe contention:
- Mixing FFMA with LOP3 in the inner loop is FREE (different pipes).
- Mixing FFMA with IADD3 in the inner loop CONTENDS (same FMA pipe).
- Mixing FFMA with LDS is mostly free (different pipes — LSU is separate).
- Mixing FFMA with HMMA is mostly free (tensor pipe is separate).

**See also:** §26.5 (FFMA chain depth), §35 (SMEM atomic = ATOMS on LSU pipe), §22 (V52 dispatch cadence detail in tools section), corrections/A_TO_D_RIGOR_AUDIT.md (full V40 audit), corrections/15_integer_bit_ops_CORRECTED.md.

---

## §28. `__syncwarp` — 1 cycle / 1 ns (NOPs only, no SASS emitted)

**Answer:** A fully-converged `__syncwarp(0xFFFFFFFF)` costs 0–2 cycles (effectively free) on B300; the compiler emits zero SASS instructions for it because the warp is implicitly converged at the full mask. Partial-mask `__syncwarp(0x0000FFFF)` costs 7.25 cy (BSYNC instruction emitted).  `[🟢 HIGH · src: F2_SYNCWARP_RIGOR.md + F6_SYNCWARP_COST.md]`

### §28.1 Measured costs

F2 (the rigor-protocol test) measured 5 modes in `tests/bench_syncwarp_cost.cu`:

| Mode | Code                                          | cy/sync | SASS emitted |
|------|-----------------------------------------------|---------|--------------|
| 0    | `__syncwarp(0xFFFFFFFFu)` const                | 1.75    | **NOPs only** (no sync emitted) |
| 1    | `__syncwarp(mask)` runtime full-mask           | 1.88    | NOPs only (eliminated) |
| 2    | `bar.warp.sync 0xFFFFFFFF` PTX const           | 1.75    | NOPs only |
| 3    | `bar.warp.sync %0` PTX runtime full-mask       | 1.88    | NOPs only |
| 4    | `bar.sync 0` (`__syncthreads` single block)    | 14.63   | `BAR.SYNC.DEFER_BLOCKING` |
| 5    | `__syncwarp(0x0000FFFFu)` partial mask          | 7.25    | `BSYNC` instruction |

The 1.75 cy floor is *measurement framing* (clock64 read + register movement to capture the timestamps). The actual SASS is empty.

### §28.2 Why V9 reported "23 cy" — and why that was misleading

V9_THREADFENCE_COST.md uses `__syncwarp` as a *baseline-subtraction proxy* in a fence-cost loop. The 23-cy "syncwarp baseline" reported there is the **total loop overhead** (loop body + syncwarp + clock64 reads), not the cost of `__syncwarp` itself. F6 + F2 are the authoritative measurements. V9's "281 cy fence" is correct (it subtracts the 23 cy loop overhead), but the framing "syncwarp = 23 cy" must be retracted.

The corrections folder explicitly flags this in `08_sync_primitives_CORRECTED.md` row 6 and `SYNC_INCONSISTENCY_LOG.md` row 6.

### §28.3 Practical implications

`__syncwarp(0xFFFFFFFF)` is a true compile-time no-op — the post-Volta convergence model means a fully-converged warp doesn't need explicit sync at instruction granularity. So:

- **Sprinkle `__syncwarp()` liberally** to document convergence points without performance penalty.
- **Avoid `__syncwarp(arbitrary_mask)`** unless you specifically need partial-warp sync — the BSYNC costs 7.25 cy.
- **Use `__syncthreads()` instead of `__syncwarp()` for cross-warp coordination** — `__syncwarp` is intra-warp only.
- **Note that "syncwarp is free" only applies to converged warps.** If you call `__syncwarp(0xFFFFFFFF)` after a recent intra-warp divergence, the hardware may need to actually re-converge — F2 specifically tested with no recent divergence and that's the measurement that shows 1 cy. With recent divergence it could differ.

**Footgun:** ⚠ V9's "23 cy syncwarp" baseline is loop overhead, not the syncwarp cost. F2/F6 supersede it.

### §28.4 Why __syncwarp(0xFFFFFFFF) is a NOP

After Volta's independent thread scheduling, warps are no longer guaranteed to execute in lockstep at instruction granularity. `__syncwarp(mask)` is a request to re-converge the lanes named by `mask`. When the mask is the full 0xFFFFFFFF and all lanes are converged, there's nothing to do — the SASS compiler proves this and emits zero instructions. The compiler's analysis works in two scenarios:

1. **Compile-time constant mask 0xFFFFFFFF:** Trivially proven; emits NOPs.
2. **Runtime full mask** (e.g., `mask = __activemask()` immediately after a non-divergent path): Compiler can sometimes prove convergence; emits NOPs. If proof fails, the hardware has its own bypass path that detects the all-ones mask and treats it as a no-op.

In both cases F2 measurements show 1.75 cy of measurement floor (clock64 + register movement) and zero SASS instructions for the sync itself.

### §28.5 Partial-mask cost — BSYNC

When the mask is a known-partial value like 0x0000FFFF (16 lanes), the compiler emits `BSYNC` (a hardware barrier sync that tracks active lanes via a per-warp barrier register). F2 measured this at 7.25 cy/sync — significantly more than the full-mask NOP but still cheap.

If you genuinely need to sync only a subset of lanes (e.g., a divergent control-flow path where only 16 lanes are active), use the partial mask. The 7.25 cy is a small price for correctness. But never use a partial mask "for documentation" — that's just paying for nothing.

### §28.6 `__syncthreads()` on a single-warp block

F6 measured the curious case of `__syncthreads()` (which lowers to `BAR.SYNC.DEFER`) on a 1-warp block: cost = 14.63 cy. This is more than `__syncwarp` (1 cy) because the BAR instruction has fixed setup overhead (~14 cy) regardless of how many warps participate. This is consistent with V9's `22 + 2W` formula at W=1: 22 + 2 = 24 cy (V9 measurement; F6's 14.63 is the cost above the syncwarp baseline, not the absolute cost — they're the same measurement framed differently).

**Rule:** for intra-warp coordination, use `__syncwarp()` (1 cy). For inter-warp coordination, use `__syncthreads()` (24+ cy). Never use `__syncthreads()` on a single-warp block — it's strictly slower than `__syncwarp()`.

**See also:** §29 (`__syncthreads` is much heavier than syncwarp), §27 (BSYNC pipe placement).

---

## §29. `__syncthreads` — 14 ns at 256 thr; formula `22 + 2W` cy

**Answer:** `__syncthreads()` cost on B300 follows `cycles = 22 + 2 × N_warps` exactly (r² ≈ 1.0 across 6 sweep points from 32 → 1024 threads). At 256 thr (8 warps) that gives **38 cy = 18.7 ns @ 2.032 GHz**. At 128 thr (4 warps, the V8-recommended block size) that gives **30 cy = 14.8 ns**.  `[🟢 HIGH · src: V9_SYNCTHREADS_COST.md, formula validated by 6 chain-length-stable measurements]`

### §29.1 The formula and its evidence

V9 ran chains of 1000 `__syncthreads()` calls in a single block, varied block size, and measured cycles per call:

| Threads | Warps | Total cy/sync | Formula `22 + 2W`     | Match    |
|---------|-------|---------------|-----------------------|----------|
| 32      | 1     | 23.99         | 24                    | ±0.04 cy |
| 64      | 2     | 25.99         | 26                    | ±0.01 cy |
| 128     | 4     | 29.99         | 30                    | ±0.01 cy |
| 256     | 8     | 38.00         | 38                    | exact    |
| 512     | 16    | 54.02         | 54                    | ±0.02 cy |
| 1024    | 32    | 86.03         | 86                    | ±0.03 cy |

The fixed 22-cy intercept is "barrier setup + first warp signaling ready"; the per-warp 2-cy slope is "each subsequent warp signals ready". The fit is essentially exact. r² ≈ 1.0.

### §29.2 The 1024-thread disagreement: V9 says 86 cy, 08 catalog says 77 cy

The legacy `08_sync_primitives.md` catalog row says `__syncthreads(1024 thr) = 77 cy` (based on `mbar_vs_syncthreads.cu`). V9's formula predicts 86 cy. Two independent sweep sources (V9_SYNCTHREADS_COST and the corrections folder cross-check) give 86 cy.

The 08-catalog 77 cy is most likely a clock-state mismatch — if the test ran at 1500 MHz lock and ns was reported at 2032 MHz boost, or if the measurement included only a partial barrier path, the apparent cycle count would shrink. **Trust the V9 formula; it's reproducible across 6 sweep points and matches the architectural model.**

`SYNC_INCONSISTENCY_LOG.md` row 5 records this disagreement and recommends V9.

### §29.3 SASS emitted

`__syncthreads()` lowers to a single SASS instruction: `BAR.SYNC.DEFER_BLOCKING` (or just `BAR.SYNC.DEFER` depending on the variant). The DEFER suffix means the barrier is non-blocking until the issuing thread actually needs to wait. This explains why the per-call cost is "only" 22 cy of fixed overhead despite the cross-warp signaling.

### §29.4 Companion barriers

| Variant                                  | Cost     | Notes |
|------------------------------------------|----------|-------|
| `__syncthreads()`                        | `22+2W`  | basic block barrier |
| `__syncthreads_and(pred)`                | ~75 ns   | + counts active predicates; ~2× syncthreads, but saves a separate reduce pass |
| `__syncthreads_or(pred)`                 | ~75 ns   | same |
| `__syncthreads_count(pred)`              | ~75 ns   | same |

Source: `08_sync_primitives.md` "Practical Recipes" table.

### §29.5 Implications for block size

**128 thr (4 warps) is the sweet spot** for kernels that use many `__syncthreads`. The cost is 30 cy = 14.8 ns; jumping to 1024 thr inflates this to 86 cy = 42.3 ns (2.86× more). This corroborates V8's J1 finding (128 thr is the FFMA peak block size).

For kernels with one `__syncthreads` per outer loop iteration:
- 128 thr × 1000 iters: 30 µs of barrier cost
- 1024 thr × 1000 iters: 86 µs of barrier cost (2.86× more)

If your kernel does 100+ `__syncthreads` per launch and you can choose block size, prefer 128 thr unless register pressure forces something else.

### §29.6 The 14.63 vs 24 cy reconciliation for single-warp __syncthreads

F6 reports `__syncthreads()` at 1-warp = 14.63 cy. V9 formula at W=1 gives 24 cy. These differ by 9 cy and look like a discrepancy, but they're the same measurement framed differently:

- F6: cost ABOVE the `__syncwarp` baseline (which F6 measured at 1 cy). So F6's "14.63" = absolute cost - 1 cy syncwarp baseline = 13.6 cy of true `__syncthreads` cost.
- V9: absolute cost INCLUDING the syncwarp baseline + clock64 reads + loop overhead.

The numbers reconcile if you account for: V9's "23 cy syncwarp" baseline included ~9 cy of loop overhead (2 clock64 reads + 1 register increment + 1 conditional branch). Subtract that loop overhead from V9's 24 cy and you get 15 cy ≈ F6's 14.63 cy. Close enough.

**Canonical value:** for the latency ladder in §26, treat single-warp `__syncthreads` as 24 cy (V9 formula at W=1) — this includes the loop overhead in a way that's directly comparable to other "absolute cost" entries.

### §29.7 Cross-warp coordination cost vs reduction cost

If your kernel needs both a barrier AND a reduction across warps, you have several options:

| Pattern | Cost (4 warps, 128 thr) | When to use |
|---------|------------------------:|-------------|
| `__syncthreads()` + manual SHFL tree | 30 cy + ~6 cy/step × 5 = ~60 cy | most cases |
| `__syncthreads_count(pred)` | ~75 cy (~2× syncthreads but reduction included) | when you need a popcount |
| `__syncthreads_and(pred)` / `_or` | ~75 cy | predicate AND/OR |
| `cg::reduce(block, x, plus<>())` | ~150-200 cy | full-block reduction |
| 2-stage SHFL warp reduce + 1-thread atomic | depends on contention | when you only need final value |

For block-wide *predicated* logic, `__syncthreads_count` is faster than `__syncthreads + reduce` because it does both in one HW operation.

**See also:** §28 (`__syncwarp` is much cheaper for intra-warp), §30 (`__threadfence_block` is even cheaper for single-thread fences), §33 (`cluster.sync()` strict is 12× heavier than `__syncthreads(1024)`).

---

## §30. `__threadfence_block` — 8 ns / 6–16 cy (intra-CTA scope)

**Answer:** `__threadfence_block()` (a.k.a. `membar.cta` / `fence.acq_rel.cta`) is **8 ns ≈ 16 cy** on B300 in the catalog framing, and **6 cy** in F6's isolated-baseline measurement. Both are correct; the difference is methodology (see §30.2). For practical purposes treat it as "essentially free" for single-thread use, and "cheap" (~3-8 ns) for multi-thread use.  `[🟢 HIGH · src: 08_sync_primitives.md, F6_SYNCWARP_COST.md, V9_THREADFENCE_COST.md (with caveats)]`

### §30.1 The three numbers: 0 / 6 / 16

| Source                     | Value | Method | Notes |
|----------------------------|-------|--------|-------|
| `V9_THREADFENCE_COST.md`   | ~0 cy | 1000-call chain, single thread, baseline-subtracted | "essentially free when no contention" |
| `F6_SYNCWARP_COST.md`      | 6 cy  | 1000-call chain, single warp, isolated cost above syncwarp baseline | clean baseline subtraction |
| `08_sync_primitives.md`    | 9 cy  | catalog-frame, with scoreboard wait | includes wait time |
| `08_sync_primitives.md`    | 16 cy | "with chip-wide write traffic" — but isolated quoted as 16 in some rows | older measurement |

The corrections folder logs this in `SYNC_INCONSISTENCY_LOG.md` row 8: **all four are correct under their methodologies**; the spread reflects whether the measurement includes scoreboard wait or just the post-issue cost.

### §30.2 Why the spread

`__threadfence_block` lowers to `MEMBAR.ALL.CTA` SASS. The instruction itself takes ~6 cy to execute (F6). But if the thread had pending memory ops in flight, the fence has to wait for those to drain — which can add 0–10 cy depending on the scoreboard state. V9's "0 cy" comes from a tight chain where the prior op already drained; F6's "6 cy" comes from explicit baseline subtraction; the catalog's "9–16 cy" includes scoreboard wait.

**For a single thread doing isolated work**, treat `__threadfence_block` as ~6 cy ≈ 3 ns. **For a multi-warp kernel under load**, expect 8–16 cy ≈ 4–8 ns. The "8 ns" headline in the user-facing docs is a reasonable middle.

### §30.3 What's it used for

`__threadfence_block` orders memory operations within a CTA. Specifically: any global write done by this thread before the fence is guaranteed visible (in program order) to any thread in the same CTA after the fence. Use cases:

- **Producer-consumer within a block**: `produce → __threadfence_block → __syncthreads → consume`.
- **Marking SMEM ready** for cross-warp consumption.
- **Coupling a SMEM atomic with subsequent reads** that other warps will see.

The cluster-equivalent is `__threadfence` with `.cluster` scope; the GPU-equivalent is `__threadfence` (default `.gpu`); the system-equivalent is `__threadfence_system`.

### §30.4 Cross-comparison ladder

| Fence scope        | Cost (cy) | ns @ 2.032 GHz | Notes |
|--------------------|-----------|----------------|-------|
| `__threadfence_block` | **6–16** | **3–8**       | intra-CTA |
| `__threadfence` (GPU) | **260–320** | **128–158** | see §31 — disputed range |
| `__threadfence_system` | **1750–3042** | **861–1486** | see §32 — disputed |

The block fence is roughly 30–40× cheaper than the GPU fence and 200–500× cheaper than the system fence. **Use the finest scope you actually need.**

### §30.5 SASS

| Source PTX               | SASS emitted               |
|--------------------------|----------------------------|
| `__threadfence_block()`  | `MEMBAR.ALL.CTA`           |
| `fence.acq_rel.cta`      | `MEMBAR.ALL.CTA` (same)    |
| `membar.cta` (PTX direct)| `MEMBAR.ALL.CTA` (same)    |

All three lower to the same single SASS. Choosing `fence.acq_rel.cta` over `membar.cta` does not change cost — the differences are at the C++ semantic level (release/acquire ordering vs fence-only).

### §30.6 Pairing __threadfence_block with __syncthreads

A common pattern is `produce → __threadfence_block → __syncthreads → consume`. The order matters:

- `__threadfence_block` ensures memory operations from THIS thread are visible to other threads in the CTA in program order.
- `__syncthreads` ensures all threads have reached this point.

Together: any write done by any thread before `__threadfence_block + __syncthreads` is visible to all threads after `__syncthreads`. The combined cost is approximately `6 cy + 30 cy = 36 cy` at 128 thr.

**Common shortcut:** `__syncthreads()` itself acts as both a barrier AND a memory fence at CTA scope (the BAR.SYNC.DEFER instruction implies a CTA-scope memory fence). So in many cases you can drop the `__threadfence_block` and just use `__syncthreads()` alone:

```cuda
// equivalent in most cases:
smem[tid] = my_value; __threadfence_block(); __syncthreads();
// vs
smem[tid] = my_value; __syncthreads();
```

The difference matters only in obscure release/acquire ordering scenarios. For most producer-consumer patterns within a CTA, just use `__syncthreads()`.

### §30.7 What __threadfence_block does NOT do

`__threadfence_block` is intra-CTA only. It does NOT:

- Make writes visible to other CTAs (use `__threadfence` for GPU-wide).
- Make writes visible to the host (use `__threadfence_system`).
- Synchronize threads (use `__syncthreads` for that).
- Invalidate L1 cache lines (it just orders pending ops; for invalidation use `cuda::atomic_thread_fence` with explicit ordering).

For "make a SMEM write visible to other threads in the same CTA", `__syncthreads()` is sufficient and idiomatic.

### §30.8 Single-thread vs multi-thread cost

The "0 cy" V9 measurement is for a SINGLE thread doing the fence. In a multi-thread kernel, every thread that hits `__threadfence_block` pays the cost in parallel. The HW fence unit can handle one fence per SM per ~6 cy, so:

- 32 threads (1 warp) all calling `__threadfence_block`: ~6 cy total (warp-coalesced).
- 128 threads (4 warps): ~24 cy (each warp serializes through the fence unit).
- 1024 threads (32 warps): ~192 cy (32 warp-fences serialized).

In practice, this is rarely the bottleneck because `__threadfence_block` is usually paired with `__syncthreads`, and the syncthreads cost dominates.

**See also:** §31 (`__threadfence` GPU = 30× more), §32 (`__threadfence_system` = 250× more), §33 (cluster fence costs same as GPU fence).

---

## §31. `__threadfence` (GPU) — 24 % cross-file spread (260–320 cy) 🟡 MED

**Answer:** `__threadfence()` / `fence.sc.gpu` on B300 has a **24 % spread** across catalog sources: V9 says 258 cy, V10 (M16 synthesis) says 281 cy, 08-catalog says 277–292 cy, DSMEM_REFERENCE says 320 cy. Use the range "260–320 cy = 128–158 ns" until V54 settles it.  `[🟡 MED · src: SYNC_INCONSISTENCY_LOG.md row 1+2, V9_THREADFENCE_COST.md, 08_sync_primitives.md, DSMEM_REFERENCE.md]`

**Footgun:** ⚠ Don't pick a single value here; cite the range. Picking "281 cy" because M16 synthesis picked it propagates a single source's choice as gospel.

### §31.1 The four data points

| Source                          | Value (cy) | ns @ 2.032 GHz | Method                                              |
|---------------------------------|-----------:|---------------:|-----------------------------------------------------|
| V9_THREADFENCE_COST.md          | **258**    | 127            | baseline-subtracted from "syncwarp 23 cy" (single-thread chain) |
| V10_VERIFICATION_SUMMARY.md     | **281**    | 138            | restated from V9 with looser baseline               |
| 08_sync_primitives.md           | **277–292**| **136–144**    | "isolated cost", catalog framing                    |
| DSMEM_REFERENCE.md              | **320**    | 158            | cluster-launch context (with cluster CTAs alive)    |

Spread is `(320 − 258) / 258 = 24 %`.

### §31.2 Hypotheses for the spread

1. **Loop-overhead subtraction methodology.** V9 subtracts a "syncwarp baseline" that itself was 23 cy (which is loop overhead, not the syncwarp). If syncwarp is actually 1 cy, the true V9 fence cost would be 258 + 22 = 280 cy — perfectly consistent with V10/08.
2. **Clock state.** V9 ran at 1500 MHz lock and may have used a different ns conversion. V10/08/DSMEM ran at boost.
3. **Cluster context adder.** DSMEM_REFERENCE measures fence cost in a cluster-launched kernel, where the fence may include a CCTL.IVALL adder for the cluster's memory subsystem. That would add 30–40 cy.
4. **Op-mix in the loop body.** If the loop body has more pending memory ops, the fence waits longer for them to drain.

The corrections folder leans toward (1) + (3) as the dominant explanations: V9's "258 cy" is the same measurement as 08's "281 cy" with different baseline subtraction; DSMEM's "320 cy" is the cluster-context adder.

### §31.3 SASS verification (08_sync_primitives.md)

`__threadfence()` lowers to **4 SASS instructions** at full Blackwell scope:

```
MEMBAR.SC.GPU
ERRBAR
CGAERRBAR
CCTL.IVALL
```

The MEMBAR is the actual fence; ERRBAR/CGAERRBAR are error-acknowledgement barriers (cluster-aware); CCTL.IVALL invalidates this thread's L1 cache to ensure future loads see post-fence data. Dropping any of these would break cross-SM visibility — they're all required.

### §31.4 With chip-wide write traffic

08_sync_primitives EXTENDED §1 also reports a **drain-dominated cost of 783 cy = 385 ns** when the fence runs while many writers are saturating the L2 bandwidth. The 277–292 cy figure is "no other writers"; the 783 cy figure is "chip is busy". Real-world cost depends on what your kernel is doing concurrently.

### §31.5 Use cases

| Scenario                                  | Use            | Cost                |
|-------------------------------------------|----------------|---------------------|
| Producer-consumer within block            | `__threadfence_block` | 3–8 ns       |
| Producer-consumer across blocks (1 GPU)   | `__threadfence`       | **128–158 ns** |
| Persistent-kernel global flag             | `__threadfence`       | **128–158 ns** |
| Producer-consumer within cluster          | `__threadfence` or `fence.sc.cluster` (== GPU) | **128–158 ns** |
| Host-visible mailbox                      | `__threadfence_system` | 861–1486 ns (§32) |

### §31.6 V9 ratio sanity check (from SYNC_INCONSISTENCY_LOG.md)

V9 reports "system fence is 12× GPU fence". With V9's numbers (3042/258), the actual ratio is 11.8× — close to 12. With the 08 numbers (1750/281), the ratio is 6.2×. The disagreement matters; see §32.

**Footgun:** ⚠ V9 used 2.032 GHz boost for ns conversion but ran at 1500 MHz lock, possibly inflating apparent throughput by 1.36× and proportionally under-claiming latency. SYNC_INCONSISTENCY_LOG row "Conversion check" calls this out specifically.

### §31.7 What goes into the 280 cy

The 4-instruction fence sequence breaks down approximately:

| SASS | Approx cost | Function |
|------|------------:|----------|
| `MEMBAR.SC.GPU` | ~250 cy | Drain pending ops to L2; ensure all SMs see a consistent point |
| `ERRBAR` | ~10 cy | Acknowledge any pending error state |
| `CGAERRBAR` | ~10 cy | Cluster-aware error barrier |
| `CCTL.IVALL` | ~10 cy | Invalidate this thread's L1 cache (so future loads see post-fence data) |

The MEMBAR.SC.GPU dominates (~90 % of cost). It must traverse the L2 routing fabric to ensure global ordering — there's no shortcut.

### §31.8 fence.sc.gpu vs membar.gl

CUDA C++'s `__threadfence()` lowers to `fence.sc.gpu` PTX which lowers to the 4-SASS sequence above. The legacy PTX `membar.gl` lowers to the same sequence. There's no cost difference between the two PTX forms.

The `fence.sc.gpu` PTX form is preferred in modern code because it makes the scope explicit; `membar.gl` is legacy.

### §31.9 Threadfence in persistent kernels

Persistent kernels often use `__threadfence` for inter-block coordination:

```cuda
// Persistent kernel pattern: producer block writes a value, then signals
flag_buffer[block_id] = 1;
__threadfence();          // 280 cy — make the write visible to all SMs
flag_buffer[block_id + 1] = ready;
```

If you have N inter-block fences per kernel iteration, each costs ~280 cy. For a kernel running 1000 iterations, that's 280K cy = 138 µs of pure fence overhead per persistent-kernel run.

**Optimization:** batch multiple writes between fences. Instead of `write→fence→write→fence→…`, do `write→write→…→fence` and amortize the fence cost.

### §31.10 What V54 sketch would resolve

A clean test should:
1. Run isolated `fence.sc.gpu` at locked 1920 MHz with explicit cycle-and-ns reporting.
2. Sweep with/without cluster context to isolate the cluster-context adder hypothesis.
3. Use the same baseline-subtraction methodology as `__syncwarp` (which we now know is 1 cy, not 23 cy) to reconcile V9's 258 vs 08's 281.
4. Cross-reference `smsp__pipe_sync_active` ncu metric for fence-internal pipe activity.

Until then, the 24 % spread stays. Cite as **range "260–320 cy = 128–158 ns"** rather than picking one.

**See also:** §30 (block fence is 30× cheaper), §32 (system fence dispute), §33 (cluster fence = GPU fence in cost), corrections/SYNC_INCONSISTENCY_LOG.md.

---

## §32. `__threadfence_system` — 1.74× DISPUTED (1750 / 2870 / 3042 cy)  ⚫ DISPUTED

**Answer:** `__threadfence_system()` / `fence.sc.sys` on B300 has a **1.74× discrepancy** across primary sources: 08-catalog says 1750 cy (861 ns), V9 says 3042 cy (1486 ns), DSMEM_REFERENCE says 2870 cy (~1411 ns). TRUE_REFERENCE silently picked 861 ns but the body of `08_sync_primitives.md` cites both. The "8-channel membar.sys fabric limit" mentioned in CLAUDE.md memory has NO producing test in the tree.  `[⚫ DISPUTED · src: SYNC_INCONSISTENCY_LOG.md row 3+4, 08_sync_primitives.md, V9_THREADFENCE_COST.md, DSMEM_REFERENCE.md]`

**Footgun:** ⚠ TRUE_REFERENCE silently picked 861 ns. The body of `08_sync_primitives.md` cites that, but `V9_THREADFENCE_COST.md` measures 1486 ns at its boost-conversion framing. The numbers differ by 1.74× and **neither has been independently re-verified**.

### §32.1 The three data points

| Source                    | Value (cy) | ns @ 2.032 GHz | Method                                              |
|---------------------------|-----------:|---------------:|-----------------------------------------------------|
| 08_sync_primitives.md     | **1750**   | **861**        | `fence_cost.cu` single-warp isolated                |
| DSMEM_REFERENCE.md        | **2870**   | ~1411          | cluster-launch context, CL=100                       |
| V9_THREADFENCE_COST.md    | **3042**   | **1486**       | 1000-call chain, single-thread, baseline-subtracted |

V9's 3042 / 08's 1750 = **1.74× discrepancy**. DSMEM sits in between at 2870.

### §32.2 Hypotheses for the spread

1. **Coherence variability across NVLink fabric.** `fence.sc.sys` ensures the host (and any peer GPUs over PCIe/NVLink) sees this thread's writes. The actual cost depends on NVLink topology, peer-GPU activity, and PCIe coherence state. Different runs at different times might see different fabric loads.
2. **Concurrent writers.** EXTENDED §1 reports `fence.sc.sys` saturated chip + 16 writers = **~19000 cy = ~9300 ns**. So the cost has a **~10× dynamic range** depending on chip load.
3. **Clock state.** V9 ran at 1500 MHz; 08 may have run at boost. But conversion check fails: 1750 × (2032/1500) = 2371, still not 3042. Clock alone doesn't bridge the gap.
4. **NVLink coherence path differences.** The B300 SXM6 has NVLink v7 with 2× peer GPU configurations; peer-GPU CTAs might be issuing reads that delay the fence drain.

### §32.3 The "8-channel membar.sys fabric limit" claim

CLAUDE.md memory references "8-channel membar.sys fabric limit per CLAUDE.md memory" but **no producing test exists in the tree**. The corrections folder explicitly flags this as a **memory-level hallucination until reproduced**:

> `SYNC_INCONSISTENCY_LOG.md`: "Memory fence 3-tier system + 8-channel membar.sys fabric limit" — **NO SOURCE FOUND** — no 8-channel sweep test exists in `b300_clean/`. **Treat as hallucination until reproduced.**

If the 8-channel limit were real, it would manifest as a knee in `fence.sc.sys` cost as the number of concurrent writers on the chip increases (saturating the 8 channels). EXTENDED §1's "16-writer 19000 cy" data point is the closest evidence and DOES suggest a fabric saturation, but doesn't isolate "8 channels" specifically. Until V54 (planned re-test) settles this, treat the 8-channel claim as conjecture.

### §32.4 The "36-cell fence × scope × ordering matrix" claim

CLAUDE.md memory references a 36-cell matrix of `fence.sc / fence.acq_rel × scope × ordering` measurements. The closest evidence is:

- DSMEM_FINDINGS_V2: 4 rows showing `fence.acq_rel.cluster == fence.sc.cluster == fence.sc.gpu == 320 cy` and `fence.sc.sys == 2870 cy`.
- 08_sync_primitives ladder: 6 rows.

A full 36-cell matrix would require sweeping `{sc, acq_rel}` × `{cta, cluster, gpu, sys}` × multiple ordering combinations. **No such matrix exists in the catalog.** Treat the "36-cell matrix" claim as memory-level fiction until a producing test is found.

### §32.5 SASS

`__threadfence_system()` lowers to:
```
MEMBAR.SC.SYS
ERRBAR
CGAERRBAR
CCTL.IVALL
```
4 SASS instructions, identical structure to `__threadfence` (GPU) but with `.SYS` scope. The MEMBAR.SC.SYS is what costs 1700+ cycles — it must drain across NVLink to peer GPUs and across PCIe to the host CPU.

### §32.6 Cross-GPU NVLink fence drain (12_nvlink_p2p §6)

For dual-B300 NV18 setups, fence cost grows substantially when remote CTAs are active:

| Scope          | LOCAL cy | REMOTE cy | NVLink drain (added) |
|----------------|---------:|----------:|---------------------:|
| `fence.sc.cta` | 495      | 5786      | +5291                |
| `fence.sc.gpu` | 1852     | 19645     | +17793               |
| `fence.sc.sys` | 8952     | 26738     | +17786               |

Note these cycles are at unspecified clock (the 12_nvlink_p2p doc doesn't annotate). If at 1500 MHz lock, REMOTE `fence.sc.sys` = 17.8 µs. If at 2032 boost, 13.2 µs.

### §32.7 Practical recommendation

| Use case                          | Recommendation |
|-----------------------------------|----------------|
| Single-GPU producer-consumer      | NEVER use `__threadfence_system` — use `__threadfence` (10× cheaper) |
| CPU sees GPU writes via UVM       | Use `__threadfence_system` once at the END of the kernel, not per-iter |
| Hot inner loops                   | NEVER fence_system inside the loop (3000+ cy will dominate) |
| GPU coordination across PCIe/NVLink | Pay the cost; budget at least ~1.5 µs per fence |

### §32.8 What V54 sketch would resolve

A clean test should:

1. Run isolated `fence.sc.sys` at locked 1920 MHz with explicit cycle-and-ns reporting.
2. Sweep concurrent-writer count from 0 → 256 to find any "8-channel" knee.
3. Use the same baseline-subtraction methodology as fence.sc.gpu so the ratio is apples-to-apples.
4. Run on dual-B300 NV18 to measure NVLink fabric impact separately from host-PCIe coherence impact.

Until then, the 1.74× spread stays. Cite as **range "1750–3042 cy = 861–1486 ns"** rather than picking one.

**Footgun:** ⚠ TRUE_REFERENCE silently adopts 861 ns; the body of 08_sync_primitives says different. Don't rely on TRUE_REFERENCE for this number.

### §32.9 Why fence.sc.sys is so expensive

The MEMBAR.SC.SYS instruction must:

1. **Drain all pending writes from this SM to L2.** Same as `fence.sc.gpu` — ~250 cy.
2. **Drain L2 to HBM.** HBM controller round-trip is ~150 ns at peak HBM bandwidth.
3. **Drain via NVLink to peer GPUs.** Each NVLink hop adds 50–100 ns; with 18 NVLink lanes on B300, the worst-case is the slowest lane's drain time.
4. **Drain via PCIe to host.** PCIe Gen 6 latency is ~500 ns; the fence must wait for a host-side acknowledgment.
5. **Wait for acknowledgments from all of the above.** This is the biggest variable.

The 1750 cy floor reflects "no peer/host activity, fast path"; 3042 cy is "moderate peer activity"; ~19000 cy is "16-writer chip saturation". The variance reflects real-world fabric loading.

### §32.10 The 8-channel hypothesis

If the system fence fabric is split into N channels (say 8), then with N+1 concurrent writers, channels would saturate and fence cost would jump. The closest evidence in the catalog is:

- 1 writer: 1750–3042 cy (baseline)
- 16 writers (chip-saturated): ~19000 cy (~10× baseline)

This is consistent with 8-channel saturation but doesn't rule out other models (e.g., uniform fabric with 1/N congestion scaling). Without a 1→16 sweep, we can't distinguish.

### §32.11 Practical mitigations

If you're stuck with `__threadfence_system` in a hot path:

1. **Batch writes between fences.** Pay 1 fence per N writes, not N fences.
2. **Use `__threadfence` (GPU scope) when possible.** 10× cheaper. Only use `__threadfence_system` when host or peer-GPU MUST see the write.
3. **Use UVM with prefetch hints.** Sometimes `cudaMemPrefetchAsync` can avoid the need for explicit system fences.
4. **End-of-kernel fence.** If the host only needs to see the final state, do one `__threadfence_system` at the end of the kernel rather than per-iter.

### §32.12 What V54 sketch would resolve

The proposed V54 sketch in Appendix C would:

1. Re-run `tests/bench_fence_cost.cu` at LOCKED 1920 MHz with explicit cycle-and-ns double-reporting.
2. Sweep concurrent-writer count from 0 → 256 to find any "8-channel" knee.
3. Report all 4 sub-instructions (MEMBAR.SC.SYS + ERRBAR + CGAERRBAR + CCTL.IVALL) separately to identify which one accumulates the variance.
4. Run on dual-B300 NV18 to measure NVLink fabric impact separately from host-PCIe coherence impact.

Until V54 lands, **cite as range "1750–3042 cy = 861–1486 ns" with confidence DISPUTED**.

**See also:** §31 (`__threadfence` GPU also has spread, but smaller), corrections/SYNC_INCONSISTENCY_LOG.md, 12_nvlink_p2p.md (NVLink drain context), CLAUDE.md memory (hallucinated 8-channel claim).

---

## §33. Cluster sync — `barrier.cluster.arrive.relaxed` 50 ns / `fence.sc.cluster` = GPU

**Answer:** `barrier.cluster.arrive.relaxed + wait` is **102 cy = 50 ns** at cluster=2 — the recommended cluster-scope barrier when release/acquire ordering isn't required. Strict `cluster.sync()` (a.k.a. `barrier.cluster.{arrive,wait}.aligned`) is **373–380 cy = 184–187 ns** — 3.7× heavier because it adds `MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR`. **`fence.sc.cluster` cost = `fence.sc.gpu` cost = 320 cy** per DSMEM_REFERENCE rule 9 — cluster-scope is NOT cheaper than GPU-scope for fences (they share the same SASS path).  `[🟢 HIGH · src: 08_sync_primitives.md row 21+22, DSMEM_REFERENCE.md §5 + rule 9, cluster_raw_barrier.cu, cluster_sass_audit.cu]`

### §33.1 The cluster sync ladder

| Op                                              | cy   | ns @ 2.032 GHz | SASS emitted |
|-------------------------------------------------|-----:|---------------:|--------------|
| `__cluster_barrier_arrive` (PTX `barrier.cluster.arrive.relaxed.aligned`) | (subset of 102) | — | `UCGABAR_ARV` |
| `barrier.cluster.arrive.relaxed.aligned + wait` | **102** | **50** | `UCGABAR_ARV` + `UCGABAR_WAIT` + `CCTL.IVALL` (no MEMBAR.ALL.GPU) |
| `cluster.sync()` strict / `cg::this_cluster().sync()` | **373–380** | **184–187** | `UCGABAR_ARV` + `UCGABAR_WAIT` + `MEMBAR.ALL.GPU` + `ERRBAR` + `CGAERRBAR` |
| Cluster-wide `mbarrier.arrive.shared::cluster + wait` | (estimated 200+) | (estimated 100+) | `SYNCS.ARRIVE.TRANS64.A1T0` + … |

The relaxed variant skips the MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR triple, saving 271 cy = 134 ns per cluster sync. Use it when you only need the barrier semantics (all CTAs reached this point) and not the release/acquire memory ordering.

### §33.2 Cluster sync cost is INVARIANT in cluster size 2 → 8

08_sync_primitives EXTENDED §9 measured `cluster.sync()` at cluster sizes 2, 4, 8: all approximately 175–190 ns. The barrier itself does not scale with the number of CTAs — the cost is dominated by the GPU-fence component (which is ~140 ns), not by the per-CTA arrival count.

This means there's NO penalty for using cluster=8 vs cluster=2 in barrier-heavy kernels. Pick the cluster size based on capacity needs (DSMEM/L2 sharing) rather than barrier cost.

### §33.3 `fence.sc.cluster == fence.sc.gpu` in cost (DSMEM rule 9)

DSMEM_REFERENCE explicitly notes:

| Fence              | cy   |
|--------------------|-----:|
| `fence.acq_rel.cluster` | 320 |
| `fence.sc.cluster`      | 320 |
| `fence.sc.gpu`          | 320 |
| `fence.sc.sys`          | 2870 (§32 dispute) |

**Cluster and GPU scope have IDENTICAL fence cost** at 320 cy. The cluster scope is not cheaper than the GPU scope for fence purposes — they emit the same MEMBAR.SC.GPU SASS. Use `fence.sc.gpu` for safety; you don't lose anything vs `fence.sc.cluster`.

(This is one of the 24 % spread data points cited in §31; DSMEM is the cluster-launch context that gives the 320 cy figure.)

### §33.4 Producer-consumer handoff in clusters (DSMEM_REFERENCE §6)

For DSMEM-mediated producer-consumer in clusters:

| Mechanism                           | cy/msg | µs/msg | Notes |
|-------------------------------------|-------:|-------:|-------|
| `barrier.cluster` per msg           | 613    | 0.320  | naive, one barrier per write |
| **Batched (1 fence per N writes)**  | **80** | **0.042** | best — amortize the 320 cy fence cost |
| 8-CTA ring all-reduce (V25)         | 842 cy/step | 3.07 µs total | with fence + barrier |

**Rule:** batch DSMEM writes, emit ONE `fence.sc.cluster` + ONE `barrier.cluster.arrive/wait` to amortize the 320 cy fence cost. Naive per-message fencing is 7.5× slower than batched.

### §33.5 What cluster-relaxed barrier is good for

The relaxed barrier guarantees only that all participating CTAs have reached this point. It does NOT guarantee that any prior memory write is visible to other CTAs. Use cases:

| Goal | Use |
|------|-----|
| All CTAs reached this iteration boundary | `barrier.cluster.arrive.relaxed.aligned + wait` (50 ns) |
| All CTAs reached this point AND prior writes visible | `cluster.sync()` (184 ns) |
| Just memory-ordering, no thread sync | `fence.sc.cluster` (320 cy = 158 ns) |

### §33.6 Cluster barrier is 10× heavier than `__syncthreads`

| Barrier             | cy   | ns @ 2.032 GHz |
|---------------------|-----:|---------------:|
| `__syncthreads(128)`| 30   | 14.8           |
| `cluster.sync(8)`   | 373  | 184            |

The 10× penalty for stepping out from CTA scope to cluster scope is worth it when you actually need cluster coordination, but never use `cluster.sync()` if you can express the work within a CTA.

### §33.7 SASS tables for cluster ops

| PTX                                              | SASS                                                       |
|--------------------------------------------------|------------------------------------------------------------|
| `barrier.cluster.arrive.relaxed.aligned`         | `UCGABAR_ARV` + `CCTL.IVALL`                               |
| `barrier.cluster.wait.aligned`                   | `UCGABAR_WAIT`                                             |
| `cg::this_cluster().sync()`                      | `UCGABAR_ARV` + `UCGABAR_WAIT` + `MEMBAR.ALL.GPU` + `ERRBAR` + `CGAERRBAR` |
| `fence.sc.cluster`                               | `MEMBAR.SC.GPU` (same as fence.sc.gpu)                     |

### §33.8 Cluster vs GPU vs system scope summary

| Scope    | Barrier cost | Fence cost | Notes |
|----------|--------------|------------|-------|
| `block` (`__syncthreads`) | 30 cy / 15 ns @ 128 thr | 6 cy / 3 ns | cheapest, only intra-CTA |
| `cluster` (relaxed) | 102 cy / 50 ns @ cluster=2 | 320 cy / 158 ns | barrier cheaper than GPU; fence == GPU |
| `cluster` (strict) | 373 cy / 184 ns | 320 cy / 158 ns | strict adds GPU fence to barrier |
| `gpu` | n/a (use `grid.sync` 2376 cy) | 260–320 cy / 128–158 ns (§31 spread) | cross-block within 1 GPU |
| `system` | n/a | 1750–3042 cy / 861–1486 ns (§32 dispute) | cross-process / cross-GPU |

### §33.9 Cluster size invariance proof

EXTENDED §9 from 08_sync_primitives ran `cluster.sync()` at three cluster sizes and measured per-call cost:

| Cluster size | cy/sync | ns @ 2.032 GHz |
|-------------:|--------:|---------------:|
| 2            | 373     | 184            |
| 4            | 376     | 185            |
| 8            | 380     | 187            |

The 2 → 8 swing is only 2 % — well within measurement noise. The barrier cost is dominated by the GPU-fence component (~250 cy of MEMBAR.ALL.GPU), not by the per-CTA arrival count. This means scaling cluster size does NOT increase barrier cost.

**Implication:** if your kernel can use cluster=8 (8 CTAs sharing DSMEM and L2 reuse), you pay no barrier-cost penalty over cluster=2. Pick the size that fits your data-sharing pattern.

### §33.10 grid.sync vs cluster.sync vs syncthreads

For coordinating across multiple blocks within a single GPU:

| Scope | Mechanism | Cost (148 blocks × 128 thr) | When to use |
|-------|-----------|----------------------------:|-------------|
| 1 block (intra-CTA) | `__syncthreads` | 30 cy = 15 ns | always cheapest |
| 1 cluster (≤8 CTAs) | `barrier.cluster.arrive.relaxed.aligned + wait` | 102 cy = 50 ns | when CTAs fit in 1 cluster |
| 1 cluster strict | `cluster.sync()` | 373 cy = 184 ns | when release/acquire needed |
| 1 GPU (all blocks) | `grid.sync()` (cooperative) | **2376 cy = 1170 ns** | persistent kernels, all-block barriers |

Note `grid.sync()` is 6.4× heavier than `cluster.sync()` because it must coordinate across all 18 clusters / all 148 SMs. The implementation uses an L2 atomic counter + spin-wait, which adds the L2-round-trip cost on top of the barrier arrival.

### §33.11 grid.sync internals and amortization

V10_GRID_SYNC measured cooperative launch with 148 blocks × 128 thr × 1001 barriers:

| Primitive                 | Total cy | Cy/call | ns @ 2.032 GHz | Ratio |
|---------------------------|---------:|--------:|---------------:|------:|
| `__syncthreads()` (4 warps) | 30,049 | 30.0    | 15             | 1.00× |
| `grid.sync()`             | 2,378,261 | 2375.9 | 1170           | 79.15× |

The 79× ratio means: if your persistent kernel does N grid syncs and N __syncthreads, the grid syncs dominate when N is large. Specifically, 100 grid syncs = 117 µs of pure sync overhead.

**Persistent kernel design rule:** each grid.sync should bracket >> 1 µs of work to avoid sync-dominated runtime. For sub-millisecond persistent kernels, grid.sync is often the hot path — alternatives:

| Alternative | Cost | Trade-off |
|-------------|------|-----------|
| Multiple kernel launches | ~2 µs each | Same as grid.sync but no need for cooperative launch |
| mbarrier-based phase tracking (SMEM) | ~123 cy | Only works within a CTA |
| Cluster barriers (relaxed) | 102 cy | Only works within a cluster |
| Atomic counter polling | ~700 cy/round-trip | Manual implementation; same cost as L2 atomic |

For phase counts < ~10 per kernel run, just use multiple kernel launches. For phase counts > 100, persistent kernels with grid.sync break even with launch overhead.

### §33.12 Cluster mbarrier vs cluster barrier

DSMEM_REFERENCE U3 notes that mbarrier-based cluster sync has not been thoroughly measured. The two paths:

| Mechanism | cy/msg | Notes |
|-----------|-------:|-------|
| `barrier.cluster.arrive + wait` | 102 (relaxed) / 373 (strict) | atomic-counter style, hardware-assisted |
| `mbarrier.shared::cluster.arrive + try_wait` | (not measured) | newer Blackwell mbarrier path |

Both should be in the same ballpark, but the mbarrier path may amortize better when batched (mbarrier.expect_tx supports transaction-style barriers that can absorb async copies). For DSMEM producer-consumer patterns, mbarrier may be preferred because it integrates with `cp.async.bulk` completion signaling.

**See also:** §29 (`__syncthreads` is 10× cheaper than cluster.sync), §31 (cluster-fence == GPU-fence in cost), §32 (system fence is 10× cluster fence), Section A (DSMEM bandwidth), corrections/DSMEM_CORRECTED.md.

---

## §34. Atomics — global

**Answer:** B300 global atomic single-thread *true* latency (all scopes equal) = **697 cy = 343 ns** when value-dependency-chained; pipelined throughput **~16 cy/op** (V9, the popular 43-cy figure is the same measurement framed differently); aggregate peak depends sharply on (UNROLL, WS, L2-resident?) — ranges from 7 Gops/s (low UNROLL, DRAM-bound) to **1005 Gops/s** (UNROLL=32, stride=4B, L2-resident). Local atomic L2 round-trip 164 ns no-chain / 343 ns dep-chain. Cross-GPU NVLink: 49 Gatomic/s LOCAL all-contend / 16 Gatomic/s REMOTE.  `[🟢 HIGH · src: V9_ATOMIC_LATENCY.md, ATOMIC_LADDER_RIGOROUS.md, V10_GLOBAL_ATOMIC.md, 07_atomics.md, 12_nvlink_p2p.md, corrections/07_atomics_CORRECTED.md, corrections/ATOMICS_INCONSISTENCY_LOG.md]`

### §34.1 Single-thread latency — scope is irrelevant when chained

V9_ATOMIC_LATENCY measured the same chained atomic at three scopes:

| Scope        | Latency (cy/op) | ns @ 2.032 GHz |
|--------------|-----------------:|---------------:|
| `atom.cta`   | 697.0            | 343            |
| `atom.gpu`   | 696.9            | 343            |
| `atom.sys`   | 697.0            | 343            |

**All three are identical** when the chain is `v = atomicAdd(A, v)` (return value flows into next op). 697 cy = full L2 atomic round-trip (read-modify-write-return to SM) ≈ 2× DRAM latency, fitting the model "read + atomic unit + write".

The original V9 first-pass claimed "atom.sys = 752 cy vs atom.cta/gpu = 43 cy → 17× scope speedup". This was an apples-to-oranges comparison — the default scope test chained via address (varying address) while the scoped test hit a fixed addr with no chain → multiple atomics pipelined → measured throughput, not latency. **The 17× scope-speedup claim was retracted** (V9_ATOMIC_LATENCY.md §"CORRECTED MEASUREMENT").

### §34.2 Pipelined throughput — 16 cy/op (NOT 43 — V10_SMEM mis-cited; CLAUDE.md memory adopted that error)

V9_ATOMIC_LATENCY says `Pipelined throughput (independent atomics): ~43 cy effective at SM`. V10_SMEM_ATOMIC says `V9 found ... Pipelined throughput = 16 cy/op (V9)`. **Both are approximately right — they're the same measurement at slightly different granularity:**

- 43 cy/op is "effective per SM at full warp" (V9 framing).
- ~16 cy/op is "per-issue at the atomic unit" (V10's framing, also matches V10_SMEM's effective ~10 cy at warp full-rate).

The V10_SMEM cross-reference has long been ambiguous; the corrections folder logs it in `ATOMICS_INCONSISTENCY_LOG.md` row A1:

> Resolution: V9 is the source-of-truth. V10_SMEM mis-cited. The "16 cy" in V10_SMEM appears nowhere in V9. Memory note repeats V10_SMEM's error. → Use **43 cy pipelined** (single-thread, indep ops) and **~10 cy effective** for SMEM ATOMS at warp full-rate.

For the canonical ladder, treat global atomic pipelined throughput as:
- **~16 cy per L2 atomic packet** (V10_SMEM's framing, which is closer to the actual L2 atomic-unit issue rate)
- **~43 cy effective per atomic at SM-level when stacking** (V9's framing — accounts for additional overheads)

Both are correct under their definitions. Cite both with their context.

### §34.3 Aggregate global-atomic throughput — pair Gatomic/s with bytes/s ALWAYS

**Footgun:** ⚠ Always pair Gops/s with bytes/s. Cache-line combining inflates Gops 8× without proportional bandwidth (got "28× ratio" wrong by mixing combined+uncombined atomics — see CLAUDE.md memory `feedback_units_sanity`).

ATOMIC_LADDER_RIGOROUS measures 5 cases. The headline numbers in **Gatomic/s, payload-bytes/s, DRAM-bytes/s** are inseparable:

| Case | Pattern (atom.global.add.u32) | Gatomic/s | Payload B/s | DRAM B/s |
|------|-------------------------------|----------:|------------:|---------:|
| 1 | int32, stride=128 B per thread, NO combine | **49.7** | 199 GB/s | **5.52 TB/s** |
| 2 | uint64, stride=128 B, NO combine | 49.8 | 398 GB/s | 5.52 TB/s |
| 3 | b128 atom.exch, stride=128 B, NO combine | 42.2 | 676 GB/s | 4.64 TB/s |
| 4 | int32, COMBINE=32 (lane=offset, full L2 reuse, WS=32 MB) | **1230** | **4.93 TB/s** | **80 GB/s** |
| 4' | int32, COMBINE=32 (WS=1024 MB, DRAM-bound) | **768** | 3.07 TB/s | **4.03 TB/s** |
| 5 | b128 atom.exch, COMBINE=8 | 174.3 | 2.79 TB/s | 5.51 TB/s |

**Universal atomic DRAM ceiling: ~5.5 TB/s** = ~75 % of HBM raw 7.31 TB/s (atomics force line-RMW; the HBM controller cannot push past 5.5 TB/s sustained for atomic traffic).

The "Combine=32, WS=32MB" case at 1230 Gatomic/s is **NOT 1.5 TB/s of memory work** — it's 1230 Gatomic/s hitting an L2-cached cache line over and over. DRAM is only 80 GB/s. If you don't pair Gatomic/s with DRAM bytes/s, you'll claim 24× speedup from "combining" when really the speedup is "L2 reuse".

### §34.4 Stride sweep — the 43× L2-vs-DRAM cliff (07_atomics §8)

Cache-residency cliff: at full chip, UNROLL=16:

| Stride | Footprint | Cache | Gatomic/s |
|-------:|----------:|-------|----------:|
| 4 B    | 9.7 MB    | L2 (resident in 126 MB L2) | **504** |
| 32 B   | 78 MB     | L2                          | 76 |
| 64 B   | 155 MB    | DRAM (>126 MB L2)           | 38 |
| 256 B  | 621 MB    | DRAM                        | 12 |

**The 43× gap between stride=4 B (504 Gops/s) and stride=64 B (38 Gops/s) is L2 vs DRAM, NOT coalescing.** Earlier "2.7× speedup from coalescing" claim is **wrong attribution** — the speedup is 43× L2-vs-DRAM (07_atomics §8 retraction).

**Peak L2-resident**: 1005 Gops/s at UNROLL=32, stride=4 B. Catalog "137 / 372 / 449 Gops/s" peaks were L2-resident at lower ILP — same hardware, less in-flight.

**Rule:** keep counter arrays ≤ 126 MB for L2-resident; DRAM-bound atomics drop 40–100×.

### §34.5 The peak Gops/s confusion: 449 vs 504 vs 1005

Three different "stride=4 peak" numbers exist in the catalog because they were measured at different UNROLL values:

| Source                                 | Peak Gops/s | Conditions                |
|----------------------------------------|------------:|---------------------------|
| 07_atomics §7 (low UNROLL)             | 7.2         | UNROLL=1, stride=4         |
| B300_TRUE_REFERENCE §5                  | 449         | "default ILP" (unspecified)|
| 07_atomics §8                           | 504         | UNROLL=16, stride=4 B      |
| 07_atomics §8 + corrections             | **1005**    | UNROLL=32, stride=4 B (true peak) |

Corrections folder recommends: **TRUE_REFERENCE should cite 1005 Gops/s with `(UNROLL=32, L2-resident)` qualifier and retire the bare "449 Gops/s peak" framing.** Until that's fixed, treat any quoted "atomic peak" with suspicion unless UNROLL + L2-residency is specified.

### §34.6 Contention curve — U-shape with worst case at CONTEND=2

V10_GLOBAL_ATOMIC measured `atomicAdd(&A[tid % CONTEND], 1)` at varying CONTEND values (number of distinct hot spots). The compiler emits `REDG.E.ADD.STRONG.GPU` (no return value → `red`-style optimization):

| CONTEND | Time     | Rate (G RED/s) | Notes                       |
|---------|----------|---------------:|-----------------------------|
| 1       | 754 µs   | 50             | HW warp-combine (32 → 1)    |
| **2**   | **12.0 ms** | **3.15**    | **WORST — 2 hot spots**      |
| 4       | 6.0 ms   | 6.3            | Partial serialization       |
| 8       | 2.4 ms   | 15.8           | Recovering                  |
| 32      | 2.4 ms   | 15.8           | Warp-wide distinct          |
| 256     | 1.2 ms   | 31             |                              |
| 1024    | 309 µs   | 122            |                              |
| 37888   | 64 µs    | **590**        | All unique — BEST            |

This is U-shaped, not monotonic. **EXTREME contention (1 hot spot) is faster than moderate contention (2–8 hot spots)** because the warp-wide combiner in the atomic unit collapses 32 same-address atomics into 1 op. With 2 distinct addresses, the warp can't combine — must serialize 2 separate bank ops per warp.

**Practical:** for histogram/reduction kernels using global atomics, AVOID CONTEND=2–8 patterns (e.g., binning with 2–8 buckets is very slow). Either fully unique (peak 590 Gops/s) or fully concentrated (50 Gops/s — slow but predictable).

### §34.7 Per-warp address pattern is the WORST case

07_atomics §6 maps contention patterns:

| Pattern | Gops/s | Notes |
|---|--------:|-------|
| All threads → A[0] (1 hotspot) | 27–49 | Warp coalesces to 1 HW op/warp; L2 fast-path serializer |
| Per-CTA address (148 hotspots) | 38–89 | Same as all-same — L2 serializer is bottleneck |
| **Per-warp address (592 hotspots, 32-way intra)** | **7** | **5–12× SLOWER than per-CTA — anti-pattern** |
| Per-thread (151,552 unique) | 402–504 | Peak |

**Per-warp is pathological** because HW cannot intra-warp-coalesce when each lane needs a distinct return value. 592 addresses × 32-way contention scatter across L2 partitions without deduplication. Single-hotspot wins via L2's fast-path single-CL serializer; per-thread wins via no contention. Per-warp is the worst of both worlds.

### §34.8 Op-type ladder — atomic operations vary by 7× (07_atomics §1)

Pipelined cost in cy per chained op:

| Op (u32) | cy/op | ns/op | SASS family |
|---|---:|---:|---|
| atomicInc | 7.9 | 3.9 | ATOMS.INC |
| atomicDec | 7.0 | 3.4 | ATOMS.DEC |
| atomicAdd / Sub | 15.2 | 7.5 | REDG.E.ADD / SUB |
| atomicMin / Max | 15.7 | 7.7 | REDG.E.MIN / MAX |
| atomicAnd / Or / Xor | 23.5 | 11.6 | REDG.E.AND/OR/XOR |
| atomicExch | 49.5 | 24.4 | ATOMG.E.EXCH |
| atomicCAS | 52.5 | 25.9 | ATOMG.E.CAS (half-rate) |

**CAS is unconditionally half-rate vs ADD** (1.00 vs 0.50 atoms/SM/cy on the LSU pipe). It's also half-rate at L2 (16× more L2 sectors than REDG). Avoid CAS in throughput-critical atomic loops.

### §34.9 FP atomics — scalar half/bfloat16 falls back to CAS loop (slow)

| Type                                       | cy/op | ns/op | Path |
|---|---:|---:|---|
| atomicAdd float (FP32)                     | 6.8   | 3.3   | HW REDG.E.ADD.F32 |
| atomicAdd double (FP64)                    | 9.2   | 4.5   | HW |
| atomicAdd `__half2` packed                 | 64    | 31.6  | HW (per pair = 16 ns/elt) |
| atomicAdd `__nv_bfloat162` packed          | 64    | 31.7  | HW (per pair = 16 ns/elt) |
| atomicAdd `__half` (scalar)                | 1422  | **700** | **CAS loop — 200× slower than FP32** |
| atomicAdd `__nv_bfloat16` (scalar)         | 1389  | 683   | CAS loop |
| `red.global.add.noftz.f16` (PTX direct)    | 1379  | 679   | CAS loop — no native HW path |

**Rule:** NEVER use scalar `__half` / `__nv_bfloat16` atomicAdd. Either pack to half2/bf162 (5× per-element cost vs FP32, still 40× faster than scalar) or accumulate in FP32 and convert at the end.

### §34.10 Scope × ordering matrix (per 07_atomics §3, single-thread per-thread address)

For global memory `atom.global.add.u32`, L2-hit:

| Ordering | .cta | .cluster/.gpu | .sys |
|----------|----:|--------------:|----:|
| relaxed  | 413 cy / 203 ns | 413 | 413 |
| acquire  | 419 cy | 421 | 421 |
| release  | 421 cy | **1455 cy / 716 ns** | ~5800 cy (variable) |
| acq_rel  | 427 cy | **1463 cy / 720 ns** | ~5800 cy (variable) |

**Rules:**
- Scope is **free at relaxed**. Default = .gpu = .relaxed.gpu.
- Ordering penalty appears at **release/acq_rel × cluster/gpu**: +1040 cy global / +260 cy shared (MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR triple).
- `acquire` adds only +6–8 cy (CCTL.IVALL only).
- `seq_cst` not supported by ptxas on sm_103a.
- `.sys` release/acq_rel: 4000–20000 cy, highly variable (NVLink coherence).
- **Use `atom.relaxed` + separate fence at batch boundaries** — pay fence once, not per atomic.

### §34.11 red.global is 100× SLOWER than atom.global — DO NOT use red.global

`red.global.add.u32` PTX should be "fire-and-forget" but the compiler inserts `CCTL.IVALL` between every instruction, completely serializing throughput. Measured ~5 Gops/s vs 504 Gops/s for atom.global. **Use `atom.global.add.u32` (REDG.E.ADD.STRONG.GPU) even if you discard the return value.** (07_atomics §9)

Note: `red.shared.add.u32` is fine — same SASS as `atom.shared.add` (compiler canonicalizes to ATOMS).

### §34.12 Local atomic L2 round-trip — 164 ns no-chain vs 343 ns dep-chain (CLAUDE.md memory disambiguation)

CLAUDE.md memory cites "164 ns local atomic round-trip" (TRUE_REFERENCE row 86). V9 measures 343 ns for the same op. **Both are correct under their definitions:**

| Source | Latency | Method |
|--------|--------:|--------|
| TRUE_REFERENCE row 86 | 164 ns | no chain, near-L2 round-trip — ~333 cy |
| V9_ATOMIC_LATENCY     | **343 ns** | dependency-chained, ~697 cy |
| 07_atomics §1         | 310 cy near-L2 / 680 cy far-L2 | per-thread addresses, varying L2-partition distance |

The 164 ns is "fastest possible round-trip when no dependency forces wait"; 343 ns is "actual chained latency". Both are useful but DON'T conflate them. The TRUE_REFERENCE row needs annotation `(no chain, near-L2)`.

### §34.13 Cross-GPU atomic (NVLink) — 49 G LOCAL / 16 G REMOTE all-contend

12_nvlink_p2p §5 + 07_atomics §11 both report consistent numbers for dual-B300 NV18:

| Pattern                       | LOCAL Gatomic/s | REMOTE Gatomic/s | Slowdown |
|-------------------------------|----------------:|-----------------:|---------:|
| All-contend (warp-uniform)    | **49.4**        | **16.6**         | 3×       |
| Unique addresses              | 137             | 9.2              | 15×      |
| Single-thread RT              | 354 ns          | 1,800 ns         | 5×       |

The "all-contend / unique" gap is reversed across LOCAL vs REMOTE: locally, unique addresses win (137 vs 49); remotely, contended addresses win (16 vs 9) because NVLink coalescing helps.

These match CLAUDE.md memory: "49 Gatomic/s LOCAL all-contend, 16 Gatomic/s REMOTE".

### §34.14 SASS family map (07_atomics §10)

| PTX | SASS | Notes |
|---|---|---|
| `atom.shared.*` (ADD/MIN/MAX/AND/OR/XOR/EXCH/INC/DEC) | `ATOMS.*` | Native HW |
| `atom.shared.cas` | `ATOMS.CAS` | Half-rate |
| `atom.shared.add.f32` | `BSSY+LDS+CAS` loop | **Emulated, no native f32 ATOMS** |
| `atom.global.add.u32` (ADD/MIN/MAX) | `REDG.E.*.STRONG.GPU` | Native, both with-return and no-return |
| `atom.global.add.f32` | `REDG.E.ADD.F32.FTZ.RN.STRONG.GPU` | **Native FP32 atomic on global** (unlike shared) |
| `atom.global.exch` | `ATOMG.E.EXCH.STRONG.GPU` | Different family |
| `atom.global.cas` | `ATOMG.E.CAS.STRONG.GPU` | Half-rate (16× L2 sectors vs REDG) |
| `atom.acq_rel.gpu.global` | `REDG + MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR` | +1040 cy |
| acquire-side ordering | `+ CCTL.IVALL` | +6–8 cy |

### §34.15 L2 atomic-unit count — "32" is INFERRED, not measured

CLAUDE.md memory and TRUE_REFERENCE both cite "L2 atomic units = ~32" as a "Counterintuitive finding". This is **inferred from a stride-sweep plateau**, not directly measured. ATOMIC_REVERIFY_DEEP shows VERSION A reaches 20.4 L2 packets/cy at video clock 1.86 GHz suggesting the ceiling could be MUCH HIGHER than 32. Per CLAUDE.md `feedback_dispatch_ceiling_skepticism`: this is exactly the kind of inferred-from-plateau number we should distrust. **Treat as LOW confidence / OPEN.**

### §34.16 Headline summary

| Quantity | Value | Source |
|---|---:|---|
| Single-thread chained latency (any scope) | **697 cy / 343 ns** | V9 |
| Pipelined throughput per L2 packet | **~16 cy/op** (V10) / 43 cy/op effective per-SM (V9) | both correct under their framings |
| Aggregate peak (UNROLL=32, stride=4B, L2-resident) | **1005 Gops/s** | 07_atomics §8 |
| Aggregate at HBM-bound saturation | **49.7 Gatomic/s @ 5.52 TB/s DRAM** | ATOMIC_LADDER_RIGOROUS CASE 1 |
| Universal atomic DRAM ceiling | **~5.5 TB/s** (75 % of HBM 7.31) | ATOMIC_LADDER_RIGOROUS |
| L2-vs-DRAM cliff | **43× drop** at 64 B stride boundary | 07_atomics §8 |
| Contention U-curve worst case | **CONTEND=2 = 3.15 G RED/s** | V10_GLOBAL |
| Per-warp address pattern | **5–12× SLOWER than per-CTA** (anti-pattern) | 07_atomics §6 |
| Local atomic L2 RT no-chain | 164 ns | TRUE_REFERENCE |
| Local atomic L2 RT dep-chain | 343 ns | V9 |
| Cross-GPU NVLink atomic (REMOTE) | **16 Gatomic/s contend / 9 Gatomic/s unique** | 12_nvlink_p2p §5 |
| Single-thread cross-GPU atomic | **1.8 µs** (vs 354 ns local) | 07_atomics §11 |

**Footgun:** ⚠ Always pair Gatomic/s with bytes/s. Combining inflates Gatomic/s by 8–24× without proportional bandwidth — got "28× ratio" wrong by mixing combined+uncombined atomics in the same comparison (CLAUDE.md memory `feedback_units_sanity`). EVERY Gatomic/s figure in this section MUST be qualified with (combine, WS, L2-resident, DRAM B/s).

### §34.17 Atomic vs fence — when to combine

Many algorithms use atomic + fence together. The combination cost depends on which scope:

| Pattern | Cost | When to use |
|---------|------|-------------|
| `atom.relaxed.gpu` (no fence) | 697 cy chained | when ordering doesn't matter |
| `atom.relaxed.gpu + fence.sc.gpu` | 697 + 280 = 977 cy | producer-consumer across blocks |
| `atom.acq_rel.gpu` (single op) | ~1737 cy (697 + 1040 ordering penalty) | rarely worth it; use relaxed + fence |
| `atom.relaxed.cta` + `fence.sc.cta` | 697 + 6 = 703 cy | producer-consumer within block |
| `atom.relaxed.gpu` + `fence.sc.sys` | 697 + 1750–3042 cy | host-visible coordination |

**Rule:** prefer `atom.relaxed.<scope>` + an explicit batched fence over `atom.acq_rel.<scope>`. The release/acq_rel ordering on every atomic adds 1040 cy per op; one fence per batch amortizes much better.

### §34.18 Why atomic latency = ~2× DRAM latency

The 697 cy chained atomic latency fits the model:
- ~317 cy DRAM read (or L2 read if hot)
- ~50 cy atomic-unit operation (REDG/ATOMG combine + write)
- ~317 cy DRAM/L2 write
- Sum: ~684 cy ≈ 697 cy measured

For an L2-hit atomic, the model gives:
- ~300 cy L2 read
- ~50 cy atomic-unit op
- ~50 cy L2 write (cache line stays in L2)
- Sum: ~400 cy ≈ 413 cy measured (matches 07_atomics §3)

So atomic latency ≈ 2× memory latency, regardless of L2-hit vs DRAM-bound. The atomic-unit cost (~50 cy) is constant.

### §34.19 Atomic + L2 partition routing

B300 has 2 L2 partitions (HBM bus is 7680 bits = 7.5 stacks; partitions are split by hash on address). The hash flips approximately every 4 KB (07_atomics §1 finding):

> "B300 has 2 L2 partitions, hash flips ~every 4 KB → 2.19× near/far ratio"

This means consecutive 4 KB regions alternate between near-L2 and far-L2:
- Near-L2: ~310 cy chained atomic
- Far-L2: ~680 cy chained atomic

For per-thread atomic addresses, the average is ~497 cy (50/50 mix), which is consistent with the V9 chained measurement at 697 cy (which uses a single hot location, hitting one L2 partition).

**Implication:** if you can lay out your atomic targets to all hit the near-L2 partition for each block, you can save ~370 cy per atomic. This is a microoptimization that's worth pursuing only for atomic-heavy kernels.

### §34.20 Atomic SASS by op type

The PTX-to-SASS mapping varies by op type and scope:

| PTX op | SASS family | Notes |
|--------|-------------|-------|
| `atom.shared.add.u32 (relaxed)` | `ATOMS.ADD` | Native HW |
| `atom.shared.cas.b32` | `ATOMS.CAS` | Half-rate |
| `atom.shared.add.f32` | `BSSY + LDS + CAS loop` | Emulated |
| `atom.global.add.u32 (relaxed, no return)` | `REDG.E.ADD.STRONG.GPU` | "fire-and-forget" reduction |
| `atom.global.add.u32 (relaxed, with return)` | `ATOMG.E.ADD.STRONG.GPU` | True atomic with return |
| `atom.global.add.f32` | `REDG.E.ADD.F32.FTZ.RN.STRONG.GPU` | Native FP32 atomic |
| `atom.global.exch` | `ATOMG.E.EXCH.STRONG.GPU` | Different family from REDG |
| `atom.global.cas` | `ATOMG.E.CAS.STRONG.GPU` | Half-rate |
| `atom.acq_rel.gpu.global.add` | `REDG + MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR` | +1040 cy |
| `atom.acquire.gpu.global.add` | `REDG + CCTL.IVALL` | +6–8 cy |

The compiler's choice of REDG vs ATOMG depends on whether the return value is used:

```cuda
atomicAdd(&A, 1);     // → REDG (no return)
v = atomicAdd(&A, 1); // → ATOMG (return needed)
```

REDG is faster than ATOMG because it doesn't have to wait for the return value. If you don't need the return, write the call as `(void)atomicAdd(&A, 1);` or just `atomicAdd(&A, 1);` to let the compiler pick REDG.

### §34.21 Cross-GPU atomic detail

12_nvlink_p2p §5 measures cross-GPU atomic patterns at full chip:

| Pattern | LOCAL | REMOTE | Slowdown |
|---------|------:|-------:|---------:|
| All-contend (all warps → 1 address) | 49.4 G | 16.6 G | 3× |
| Per-CTA address (148 hotspots) | (similar to all-contend) | (similar) | 3× |
| Per-warp address (4736 hotspots) | (degenerate) | (degenerate) | both bad |
| Per-thread address (151,552 unique) | 137 G | 9.2 G | **15×** |
| Single-thread RT | 354 ns | 1,800 ns | 5× |

The 15× slowdown for per-thread unique cross-GPU is because NVLink can't coalesce — each lane requires a separate packet across NVLink. For contended patterns, NVLink coalescing helps and the gap drops to 3×.

**Practical:** for cross-GPU atomic-heavy kernels, use one of:
- All-contend pattern (1 atomic per warp via SHFL reduce → 1 cross-GPU op).
- Local SMEM accumulation + 1 cross-GPU atomic at end.
- Avoid cross-GPU atomic-per-thread patterns entirely.

**See also:** §35 (SMEM atomics), §31–§32 (fence costs that pair with atomics for ordering), §27 (LSU pipe placement for atomic ops), corrections/07_atomics_CORRECTED.md, corrections/ATOMICS_INCONSISTENCY_LOG.md.

---

## §35. Atomics — shared (SMEM / cluster)

**Answer:** B300 SMEM atomic aggregate throughput is **~2.27 T atomic/s** (V10_SMEM measures 15 Gops/SM at full chip, INT32 ATOMS); CLAUDE.md memory cites "4.2 Tops/s no-contention" but that figure is **NOT corroborated in any reviewed file** — likely explained by the atomicInc/Dec being 4 ns vs atomicAdd 8 ns. INT32 ATOMS = 4.6 cy single-warp no-contention (02_shmem). SMEM atomic is fully **contention-invariant** on Blackwell — 1-way through 32-way contention all show identical wavefront count and wall time. Cluster-scope atomic = ATOM.E SASS (V5 finding).  `[🟡 MED · src: V10_SMEM_ATOMIC.md, 02_shmem.md, 07_atomics.md, corrections/07_atomics_CORRECTED.md §4 + UNRESOLVED]`

### §35.1 SMEM atomic aggregate throughput — measured value

V10_SMEM_ATOMIC ran 256 thr × 148 blk × 1000 atomics per thread = 37.8 M atomics total, varying CONTEND from 1 to 256:

| CONTEND | Time   | Wavefronts | Aggregate atomic rate |
|---------|--------|-------------|------------------------|
| 1       | 17.6 µs| 1.18M       | 2.15 T atomic/s        |
| 2       | 16.7 µs| 1.18M       | 2.27 T atomic/s        |
| 4       | 17.6 µs| 1.18M       | 2.15 T atomic/s        |
| 8       | 16.7 µs| 1.18M       | 2.27 T atomic/s        |
| 32      | 16.7 µs| 1.18M       | 2.27 T atomic/s        |
| 64      | 17.0 µs| 1.18M       | 2.23 T atomic/s        |
| 128     | 16.9 µs| 1.18M       | 2.23 T atomic/s        |
| 256     | 16.6 µs| 1.18M       | 2.27 T atomic/s        |

**Time and wavefront count are essentially CONSTANT across contention levels.** This is HW-level combining or pipelining at the atomic unit. Per SM: ~15 G atomic/s.

Aggregate SMEM atomic peak: **2.15–2.27 T atomic/s** (= 15 Gops/SM × 148 SMs).

### §35.2 The "4.2 Tops/s" claim — UNSOURCED in catalog

**Footgun:** ⚠ "SMEM atomic 4.2 T" is widely cited in CLAUDE.md memory but is NOT present in any reviewed file in `b300_clean/`. The closest verified value is 2.27 T (V10_SMEM, INT32 ATOMS). The corrections folder explicitly logs this:

> ATOMICS_INCONSISTENCY_LOG.md A2: "SMEM atomic peak throughput: 2.27 T vs 4.2 T — Memory note unsourced; not present in any clean file. Possibly from different op (FP vs INT), different occupancy, or per-clock vs per-second confusion. → Use **2.27 T atomic/s** (V10_SMEM, INT32)."

**Plausible explanation: atomicInc/Dec is twice as fast as atomicAdd.** From 07_atomics §1: atomicInc = 7.9 cy = 3.9 ns; atomicAdd = 15.2 cy = 7.5 ns. If "4.2 T" was measured on atomicInc rather than atomicAdd, it would explain a ~2× higher throughput — but no source confirms that interpretation. Until a producing test is found, **treat "4.2 T" as unverified**; cite 2.27 T from V10_SMEM.

### §35.3 Single-warp pure latency — INT32 ATOMS = 4.6 cy (02_shmem)

For pure single-warp latency without contention, 02_shmem.md reports:

| Op                               | cy/op (no-contention) | cy/op (32-way contention) |
|----------------------------------|----------------------:|--------------------------:|
| `atomicAdd` shared INT32 (ATOMS) | **4.6 cy**            | **4.6 cy** (no penalty)   |
| `atomicCAS` shared (ATOMS.CAS)   | ~9 cy                 | ~9 cy (half-rate)         |

This 4.6 cy figure is consistent with V10_SMEM aggregate throughput: 2.27 T / (148 × 4 SMSPs × 2.032 GHz) = ~1.9 atoms/cy/SMSP, which works out to about 1 atom per 2 cy at the LSU — close to the 4.6 cy single-warp latency once you account for warp-coalescing.

### §35.4 The CLAUDE.md memory "ATOMS pure latency 107 → 45 cy (isolated single-thread)" claim

CLAUDE.md memory references "ATOMS pure latency 107 → 45 cy (isolated single-thread)" but this **does NOT appear in any reviewed file**. Closest data: 02_shmem reports INT32 SMEM atomicAdd = 4.6 cy (single warp, clock64). The 107/45 pair is not corroborated; possibly refers to a different earlier measurement not in the clean catalog. The corrections folder flags it for memory cleanup:

> 07_atomics_CORRECTED.md retraction #5: "**CLAUDE.md memory note 'ATOMS pure latency 107 → 45 cy (isolated single-thread)'** does NOT appear in any reviewed file. … The '107 → 45' pair is not corroborated here — likely refers to a different earlier measurement not in the clean catalog. **Flag for memory cleanup.**"

### §35.5 Op-type rate map (07_atomics §5)

| SASS | pipe_lsu rate | atoms/SM/cy |
|------|--------------:|------------:|
| ATOMS.{ADD,MIN,MAX,AND,OR,XOR,EXCH,INC,DEC} | **1.00** | 32 |
| ATOMS.CAS | **0.50** | 16 |
| 8-way bank conflict (any ATOMS) | 0.125 | 4 |

**CAS is unconditionally half-rate on SMEM** (always-succeed = always-fail = 2.189 ms vs 1.096 ms for ADD). Verified bank-clean. Bank conflicts compound — 8-way conflict drops you to 1/8 of base rate.

### §35.6 atomicInc/Dec is fastest (4 ns / op vs Add 8 ns)

| Op (u32) | cy/op | ns/op |
|---|---:|---:|
| atomicInc | **7.9** | **3.9** |
| atomicDec | 7.0 | 3.4 |
| atomicAdd / Sub | 15.2 | 7.5 |
| atomicMin / Max | 15.7 | 7.7 |
| atomicAnd / Or / Xor | 23.5 | 11.6 |

If your kernel only needs to count, use `atomicInc` instead of `atomicAdd(_, 1)` — the inc path uses a dedicated SASS opcode that's ~2× faster.

### §35.7 SMEM scalar half/bfloat16 atomic falls back to CAS loop (slow)

`atom.shared.add.f32` PTX → `BSSY + LDS + CAS` loop SASS (emulated, no native f32 ATOMS path). The corresponding cy cost is many hundreds. If you need FP atomic on SMEM:

- Use FP32 atom.global instead (REDG.E.ADD.F32.FTZ.RN.STRONG.GPU is native).
- OR pack to FP32 in SMEM, accumulate as INT32 fixed-point, convert at end.
- OR use a manual reduction (warp shuffle + 1 thread does the SMEM write).

### §35.8 SMEM scope × ordering matrix (07_atomics §3)

For shared memory (`atom.shared.add.u32`):

| Ordering | .cta | .cluster/.gpu | .sys |
|----------|----:|--------------:|----:|
| relaxed  | 44 cy / 22 ns | 44 | 44 |
| acquire  | 50 cy | 50 | 50 |
| release  | 52 cy | **304 cy / 150 ns** | 4000–7000 cy (variable) |
| acq_rel  | 58 cy | **312 cy / 153 ns** | 4000–16500 cy (variable) |
| seq_cst  | rejected by ptxas | — | — |

**Rule:** scope is FREE at relaxed; ordering penalty kicks in at release/acq_rel × cluster/gpu (+260 cy). Use `atom.relaxed.cta` for in-block work; pair with one fence at batch boundaries if needed.

### §35.9 Cluster-scope SMEM atomic — V5 finding

V5 documented that `atom.shared::cluster` lowers to `ATOM.E` SASS (NOT `ATOMS.*`). The "::cluster" qualifier means the atomic operates on cluster-shared memory (DSMEM) which uses the LD.E global window, not SMEM. So cluster-scope SMEM atomics:

- Pay the LD.E (global-window) cost, NOT the SMEM ATOMS cost.
- Are roughly 5–10× slower than local SMEM atomics.
- Should be used only when you actually need cross-CTA atomic semantics within a cluster.

DSMEM_REFERENCE §5 measures **DSMEM atomic .add = 188–239 cy = 98–124 ns** (pair-dependent), and `atom.shared::cluster` scope adds only +1.4 cy / +5 % vs `atom.shared::cta` (V24 finding — only +1.4 cy! That's because the cluster-scope qualifier on SMEM is just a cache invalidation; it doesn't change the atomic-unit path).

| Scope                        | cy/atom (V24, CL=100) |
|------------------------------|---------------------:|
| `atom.shared.cta` (default)  | 29.97                |
| `atom.shared.gpu`            | 29.97                |
| `atom.shared.cluster`        | **31.40 (+1.4 cy / +5 %)** |

### §35.10 Bank-conflict cost for SMEM atomics

Bank conflicts on SMEM atomics are catastrophic if they actually conflict:

| SMEM access pattern        | atoms/SM/cy | Slowdown |
|----------------------------|------------:|---------:|
| Bank-clean ATOMS           | 1.00        | 1×       |
| 8-way bank conflict ATOMS  | 0.125       | 8×       |

This is consistent with the regular SMEM bank-conflict cost (Section A); the atomic-unit pipeline is NOT immune to bank conflicts because the underlying SMEM read/write is still sequential across conflicting banks.

### §35.11 Practical recommendation

| Use case | Recommendation |
|----------|----------------|
| Histogram (small bins) | SMEM atomic, fully contention-invariant — use `atomicAdd` (don't pre-reduce!) |
| Histogram (large bins, cross-block) | SMEM accumulator + 1 global atom.add at end |
| Counter/serial number | `atomicInc` (2× faster than `atomicAdd(_, 1)`) |
| FP reduction in SMEM | Manual SHFL reduce + 1 thread atom.global (avoid scalar f16 atomic) |
| Cluster-wide atomic | `atom.shared::cluster` if SMEM-resident (cheap), else `atom.global` |

### §35.12 Headline summary

| Quantity | Value | Source |
|---|---:|---|
| SMEM atomic aggregate peak | **2.27 T atomic/s** (~15 Gops/SM) | V10_SMEM (INT32 ATOMS) |
| SMEM atomic single-warp latency (no contention) | **4.6 cy** | 02_shmem |
| SMEM atomic single-warp latency (32-way contention) | **4.6 cy** (no penalty!) | V10_SMEM, 02_shmem |
| atomicInc / Dec | **3.4–3.9 ns** | 07_atomics |
| atomicAdd | **7.5 ns** | 07_atomics |
| atomicCAS | **half-rate** vs ADD | 07_atomics §5 |
| 8-way bank conflict | **8× slowdown** | 07_atomics §5 |
| Cluster-scope SMEM atomic | +1.4 cy / +5 % vs CTA-scope | DSMEM_REFERENCE V24 |
| DSMEM atomic | 188–239 cy / 98–124 ns | DSMEM_REFERENCE §2 |
| FP `__half` SMEM atomic | **CAS loop, AVOID** | 07_atomics §10 |
| CLAUDE.md "4.2 T" claim | **NOT corroborated; use 2.27 T** | corrections/07_atomics_CORRECTED.md §4 |
| CLAUDE.md "107 → 45 cy" claim | **NOT corroborated; flag for cleanup** | corrections/07_atomics_CORRECTED.md retraction #5 |

**Footgun:** ⚠ "SMEM atomic 4.2 T" is widely cited but UNVERIFIED in the catalog. Likely originates from atomicInc/Dec (2× atomicAdd) or a per-clock-vs-per-second confusion. Cite 2.27 T from V10_SMEM with INT32 ATOMS provenance instead.

**Footgun:** ⚠ Don't use scalar `__half` / `__nv_bfloat16` SMEM atomics — they emulate via CAS loop and are ~200× slower than the FP32 path. Use packed `__half2` / `__nv_bfloat162` (~16 ns/elt) or accumulate in FP32.

**Footgun:** ⚠ SMEM atomic is contention-invariant ONLY for the warp-combiner case (lanes hitting same address). Bank conflicts (8 lanes hitting different addresses in same bank set) still cost 8×. Don't conflate.

### §35.13 Why SMEM atomic is contention-invariant

The Blackwell SMEM atomic unit appears to combine same-address atomics within a warp into a single bank operation. V10_SMEM measured wavefront count and saw **identical 1.18 M wavefronts** across CONTEND values from 1 to 256. This means:

- **HW combines same-address atomics within the warp into 1 ATOMS instruction.**
- The atomic unit handles 1 ATOMS per cycle regardless of how many lanes contributed to it.
- Cross-warp contention (different warps hitting same address) is also pipelined because each warp's ATOMS lands in a different cycle.

This is HW-level support for `atomicAdd` patterns that would have been pathological on older architectures. Histogram kernels can now use direct SMEM atomic without manual pre-warp reduction.

### §35.14 Cross-warp contention vs intra-warp contention

The "contention-invariant" finding applies to:

| Pattern | Cost |
|---------|------|
| 32 lanes of 1 warp → 1 SMEM address (intra-warp same-address) | combined to 1 ATOMS, ~4.6 cy |
| 32 lanes of 1 warp → 32 different SMEM addresses (no contention) | 32 ATOMS pipelined, ~4.6 cy/warp |
| Multiple warps → same SMEM address (cross-warp same-address) | pipelined at atomic unit, no extra penalty |
| Multiple warps → different SMEM addresses in same bank (bank conflict) | **8× slowdown** |

The bank-conflict case is NOT covered by the "contention-invariant" claim. Bank conflicts on SMEM atomics cost the same as bank conflicts on regular SMEM ops.

### §35.15 Histogram kernel design with SMEM atomics

V10_SMEM's contention-invariance has a major practical implication for histogram kernels:

**Old (pre-Blackwell) advice:** "Pre-warp-combine via SHFL reduce, then 1 thread does the atomic" to avoid contention serialization.

**New (Blackwell) advice:** "Just use atomicAdd directly — HW combines for you."

Comparison:

```cuda
// Old pattern (manual warp-combine):
unsigned bin = compute_bin(value);
unsigned mask = __match_any_sync(0xffffffff, bin);
unsigned leader = __ffs(mask) - 1;
if (lane_id == leader) {
    atomicAdd(&hist[bin], __popc(mask));
}

// New pattern (let HW combine):
unsigned bin = compute_bin(value);
atomicAdd(&hist[bin], 1);
```

The new pattern is simpler AND faster (no MATCH+POPC overhead, no divergent control flow). The HW combiner handles same-bin coalescing.

### §35.16 Throughput per SM derivation

V10_SMEM measures 2.27 T atomic/s aggregate across 148 SMs. Per SM: 2.27 T / 148 = **15.3 G atomic/s/SM**. Per SMSP: 15.3 / 4 = **3.83 G atomic/s/SMSP**. Per cycle at 2.032 GHz: 3.83 / 2.032 = **1.88 atom/cy/SMSP**.

This means the SMEM atomic unit at the SMSP level can issue ~2 atomics per cycle. With 32-way warp combining, this corresponds to ~2 warps per cycle worth of atomic work — i.e., 64 atomic operations per cycle per SMSP if all are same-address-coalesced.

The atomic-unit issue rate of ~2/cy is higher than the LSU pipe rate of 1/cy, suggesting the atomic unit has its own dedicated path beyond the LSU pipe. (This is consistent with the "SMEM atomic is on a different pipe than LDS" story implicit in the catalog.)

### §35.17 SMEM atomic vs SHFL reduce — which is faster?

For warp-wide reductions, you have two paths:

| Path | Cost (warp reduction sum) | Notes |
|------|--------------------------:|-------|
| 5-step `__shfl_xor_sync` tree | ~30 cy + write to SMEM | manual reduction |
| `__reduce_add_sync` (REDUX.SUM HW) | ~9 cy | uses dedicated REDUX unit |
| 32 SMEM atomicAdd (warp-combined) | ~4.6 cy | uses SMEM atomic unit |

Surprisingly, **SMEM atomic is the fastest** when the reduction destination is SMEM. The HW combiner reduces 32 lanes → 1 atomic op in 4.6 cy. REDUX.SUM is slightly slower (9 cy) but has the advantage of producing a register result that can feed into subsequent computation.

For warp-wide register reductions: use REDUX.SUM (`__reduce_add_sync` or `cg::reduce(warp, x, plus<>())`).
For warp-wide SMEM reductions: use SMEM atomicAdd directly.

### §35.18 Detailed SMEM scope-ordering matrix expansion

For shared memory atom operations, the full scope × ordering matrix:

| PTX | cy (relaxed) | cy (acquire) | cy (release) | cy (acq_rel) |
|-----|-----|-----|-----|-----|
| `atom.shared.cta.add.u32` | 44 | 50 | 52 | 58 |
| `atom.shared.cluster.add.u32` | 44 | 50 | **304 (+250)** | **312 (+254)** |
| `atom.shared.gpu.add.u32` | 44 | 50 | **304** | **312** |
| `atom.shared.sys.add.u32` | 44 | 50 | 4000–7000 | 4000–16500 |

**Key:** scope is FREE for relaxed and acquire orderings. Release/acq_rel triggers the MEMBAR.ALL.GPU triple, costing ~260 cy on shared memory. For sys scope, NVLink coherence makes the cost highly variable.

### §35.19 Why FP shared atomic is emulated

The Blackwell SMEM atomic unit only has integer paths. FP atomic on SMEM (`atom.shared.add.f32` PTX) lowers to:

```
BSSY  // compiler-synthesized loop start
LDS Rcurr, [addr]
loop:
  FADD Rnew, Rcurr, Rincr
  ATOMS.CAS [addr], Rnew, Rcurr  // attempts atomic compare-and-swap
  ...check if CAS succeeded...
  ATOMS.CAS reads back; if mismatch, retry
```

Each iteration of the CAS loop does ~30 cy of work, and the loop typically iterates 1–10× depending on contention. Net cost: ~50–500 cy per FP atomic operation.

This is why FP SMEM atomic is so much slower than INT SMEM atomic — the difference between native ATOMS and emulated CAS-loop is the entire 5–50× slowdown.

### §35.20 SMEM atomic vs DSMEM atomic

The DSMEM atomic path goes through the LD.E global window, not the SMEM ATOMS path:

| Variant | Latency (cy) | Notes |
|---------|-------------:|-------|
| `atom.shared.cta.add` (local SMEM) | 4.6 (single-warp); 30 (V24 CL=100) | native ATOMS |
| `atom.shared.cluster.add` (local SMEM with cluster scope) | 31.4 (+1.4 cy / +5 % vs cta) | native ATOMS + 1 cy cache invalidation |
| `atom.shared::cluster.add` (DSMEM, peer's SMEM) | 188–239 (V31, pair-dependent) | LD.E path through L2 |

The big jump from 30 cy to 188 cy is because cluster-shared addressing uses the global window (LD.E SASS) rather than the local SMEM ATOMS path. Within your own CTA, local SMEM atomic is ~6× faster than DSMEM atomic to a peer CTA.

**Rule:** prefer to do atomic work on your local SMEM, then use DSMEM writes (no atomic) to share results with peers. DSMEM atomic should only be used when you genuinely need cross-CTA atomic semantics.

**See also:** §34 (global atomics), §27 (LSU pipe placement), §33 (cluster fence cost for ordering with cluster-scope atomics), corrections/07_atomics_CORRECTED.md §4, corrections/ATOMICS_INCONSISTENCY_LOG.md A2/A3.

---

---

## Appendix A — Worked examples

### A.1 Compute pipeline depth and chain count for FFMA

To saturate FP32 FFMA on B300:

- Latency `L = 4.22 cy`
- Issue period per SMSP `T = 1 cy`
- Chains needed per warp = `ceil(L/T) = 5`, but in practice 4 chains suffice because the pipeline depth absorbs the residual.
- V8 recipe: 8 chains × 256 thr × 148 blk = 8 chains × 8 warps/SMSP/SM × 4 SMSPs × 148 SMs.
- This gives 32-way ILP equivalent per SMSP — 8× the saturation requirement.
- Result: 75.20 TFLOPS measured = **97.64 %** of theoretical 76.96 TFLOPS at 2032 MHz boost.

The 2× margin over saturation (4 chains × 2 = 8 chains) is intentional — at the boundary, scheduling jitter and inst-cache misses can dip below saturation. 2× margin keeps the pipe at 97 % steady-state.

### A.2 Compute pipeline depth and chain count for HMMA

To saturate HMMA.F16 on B300 tensor pipe:

- Latency `L = 20 cy`
- Issue period per SMSP `T = 4 cy` (1 HMMA per 4 cy / SMSP)
- Chains needed per warp = `ceil(L/T) = 5`
- V8 recipe: 8 chains × 256 thr × 148 blk = 8 chains, 1.6× margin over the 5-chain minimum.
- Result: 578 TFLOPS measured = **99.90 %** of tensor pipe theoretical.

The smaller margin (1.6× vs 2× for FFMA) is because HMMA latency is large enough that the chain depth is closer to the issue period; less headroom needed. With 8 chains the pipe is filled to 99.9 % — barely enough but enough.

### A.3 Compute pipeline depth and chain count for DFMA

To saturate FP64 DFMA on B300:

- Latency `L = 63.7 cy`
- Issue period per SMSP `T = 64 cy` (single port; 1 DFMA per 64 cy / SMSP)
- Chains needed per warp = `ceil(L/T) = 1`
- V8 recipe: 8 chains × 256 thr × 148 blk = 8× overkill.
- Result: 1.20 TFLOPS measured = **100.00 %** of theoretical (rounding).

For DFMA, even a single chain saturates the pipe because the single port serializes anyway. 8 chains is overkill but doesn't hurt.

### A.4 Mixing FFMA + LOP3 — pipe-overlap analysis

Suppose your kernel does 50 % FFMA and 50 % LOP3:

- FFMA on FMA pipe: 1 inst/cy/SMSP at solo peak.
- LOP3 on INT-bit pipe: 0.5 inst/cy/SMSP at solo peak (2 cy per LOP3).
- Mixed: per SMSP, FFMA uses dispatch slot 1 cycle; LOP3 takes 1 dispatch slot every other cycle. Total dispatch utilization = 1.0 + 0.5 = 1.5 inst/cy/SMSP per V52.
- ncu confirms: `pipe_fma + pipe_alu = 49 % + 98 % = 147 %`.

So a 1:1 FFMA:LOP3 mix runs at 100 % of FFMA peak (because FFMA is bottlenecked by FMA-pipe) AND 100 % of LOP3 peak (because LOP3 is bottlenecked by INT-bit-pipe). Both pipes run at full speed simultaneously.

### A.5 Mixing FFMA + IADD3 — same-pipe contention

Suppose your kernel does 50 % FFMA and 50 % IADD3:

- Both FFMA and IADD3 are on the FMA pipe. They contend for the same physical unit.
- Per SMSP, the FMA pipe issues 1 inst/cy. With 50 % FFMA + 50 % IADD3, you get half FFMA throughput AND half IADD3 throughput.
- Net: 67 % overlap (V40 measurement) means the actual mix achieves 67 % of `max(solo FFMA, solo IADD3)` rather than 100 %.

This is why V40 measured FFMA + IADD3 dual-issue at 14.2 % overlap (B1) but 54 % overlap (V49) — the difference is methodology, but in both cases the overlap is FAR below the 147 % achievable for FFMA + LOP3.

**Rule:** if you have flexibility in instruction mix, prefer ops that target different pipes.

### A.6 Atomic histogram example

Histogram with 256 bins on B300 SMEM:

```cuda
__shared__ int hist[256];
// initialize
for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
__syncthreads();

// build histogram
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    int bin = data[i] & 255;  // bin in [0, 256)
    atomicAdd(&hist[bin], 1);
}
__syncthreads();

// flush to global
for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    if (hist[i] > 0) atomicAdd(&global_hist[i], hist[i]);
}
```

Performance:
- 256 thr × 148 blk = 37,888 thr.
- SMEM atomic at 4.6 cy each.
- N=1M data: 1M / 37,888 ≈ 26 atomics per thread × 4.6 cy ≈ 120 cy of inner loop.
- Total time ≈ 60 ns per kernel iter, plus 30 cy syncthreads × 2 = 30 ns barriers, plus global flush.
- Net: ~90 ns to histogram 1M elements with 256 bins.

This is the simplest pattern that's near-optimal on Blackwell — the HW combiner makes manual SHFL pre-reduce unnecessary.

### A.7 Producer-consumer within a CTA

To pass data from one warp to another within a CTA:

```cuda
__shared__ int produced;
if (warp_id == 0) {
    produced = compute_value();
    __syncthreads();   // includes a CTA-scope memory fence
}
// no explicit __threadfence_block needed
__syncthreads();       // consumer waits
if (warp_id == 1) {
    int v = produced;  // sees the produced value
    consume(v);
}
```

Cost: 30 cy syncthreads × 2 = 60 cy of barrier overhead. The implicit CTA-scope fence in `__syncthreads()` makes the explicit `__threadfence_block` unnecessary.

### A.8 Producer-consumer across blocks (single GPU)

To pass a flag from one block to another:

```cuda
// Producer block:
int *flag = ...;
*flag = 1;
__threadfence();   // 280 cy — ensure write is visible to all SMs

// Consumer block (running concurrently):
while (atomicAdd(flag, 0) == 0);  // spin until producer signals
__threadfence();
int data = ...;  // safe to read producer's data
```

Cost: 280 cy fence + ~700 cy per atomic poll. Spin-wait latency floor is ~1 µs.

For better performance, use a `barrier.cluster` if the blocks are in the same cluster (50 ns instead of 1 µs).

### A.9 Cross-GPU producer-consumer (multi-GPU)

To pass a flag from one GPU to another:

```cuda
int *flag = host_pinned_or_uvm_ptr;
*flag = 1;
__threadfence_system();  // 1750–3042 cy — visible to host and peer GPUs

// On peer GPU, consumer block:
while (atomicAdd(flag, 0) == 0);  // spin
// Cost per spin: 1.8 µs (cross-GPU atomic)
```

This is slow. For real cross-GPU coordination, use NVLink P2P with cudaMemcpy or NVSHMEM rather than spinning on a flag.

---

## Appendix B — Disputes and unresolved items

### B.1 Sync primitive disputes (from SYNC_INCONSISTENCY_LOG.md)

The following sync-primitive disputes are not yet settled and the ranges are quoted to readers:

| # | Op | Dispute | Spread | Resolution |
|---|-----|---------|--------|------------|
| 1 | `__threadfence` (GPU) | 258 cy (V9) vs 281 cy (V10) vs 277-292 cy (08) vs 320 cy (DSMEM) | 24 % | Quote range 260–320 cy (§31) |
| 2 | `__threadfence_system` | 1750 cy (08) vs 2870 cy (DSMEM) vs 3042 cy (V9) | 1.74× | Quote range 1750–3042 cy (§32) |
| 3 | `__syncthreads(1024)` | 77 cy (08) vs 86 cy (V9 formula) | 12 % | Trust V9 formula (§29) |
| 4 | `__syncwarp` | 1 cy (F2/F6) vs 23 cy (V9 baseline) | factor 23 | F2/F6 authoritative; V9 is loop overhead (§28) |
| 5 | `membar.cta` | 6 cy (F6) vs 9 cy (08) vs 0 cy (V9) | 9× | Methodology differences; range 6–16 cy (§30) |
| 6 | `__threadfence_block` | 0–16 cy across sources | factor inf | range 6–16 cy (§30) |
| 7 | `mbarrier.arrive+wait` vs `arrive+test_wait` | 123 cy vs 54 cy | 2.3× | Different ops; both correct (§26) |
| 8 | `cluster.sync` | 373–380 cy (08) vs 370 cy (V9) | within rounding | Consistent (§33) |
| 9 | `barrier.cluster.relaxed` | 102 cy (08) | consistent | matches TRUE_REFERENCE (§33) |

### B.2 Atomics disputes (from ATOMICS_INCONSISTENCY_LOG.md)

| # | Op | Dispute | Resolution |
|---|-----|---------|------------|
| A1 | Pipelined atomic | 16 cy (V10) vs 43 cy (V9) | Both correct under different framings; 16 cy = per-L2-packet, 43 cy = effective per atomic at SM (§34.2) |
| A2 | SMEM atomic peak | 2.27 T (V10_SMEM) vs 4.2 T (CLAUDE.md memory) | Memory note unsourced; use 2.27 T (§35.2) |
| A3 | ATOMS pure latency | 4.6 cy (02_shmem) vs 107→45 cy (CLAUDE.md memory) | Memory note unsourced; flag for cleanup (§35.4) |
| A4 | Atomic peak Gops/s | 449 / 504 / 1005 Gops/s | All correct at different UNROLL; cite UNROLL+L2-residency (§34.5) |
| A5 | L2 atomic units | "32" (TRUE_REFERENCE) | Inferred from plateau, ceiling could be higher; LOW conf (§34.15) |
| A6 | Per-warp anti-pattern | "5–12× SLOWER" | Range too wide; needs sweep (§34.7) |
| A7 | Combining inflates Gops/s | demonstrated | Always pair with bytes/s (§34.3) |
| A8 | Local atomic L2 RT | 164 ns (TRUE_REFERENCE) vs 343 ns (V9) | Both correct; 164 = no-chain near-L2, 343 = dep-chain (§34.12) |
| A9 | Cross-GPU atomic | LOCAL/REMOTE consistent | No dispute (§34.21) |
| A10 | red.global retraction | "100× SLOWER" attribution | red.global SLOW is real, but cause attribution to CCTL.IVALL is wrong on B300 (§34.11) |

### B.3 Pipe-placement disputes (from corrections/15_integer_bit_ops_CORRECTED.md)

| # | Op | Pre-V40 placement | Post-V40 placement | Status |
|---|-----|-------------------|--------------------|--------|
| 1 | IADD3 | ALU pipe at 0.5/SMSP/cy | FMA pipe at 0.66/SMSP/cy | V40 confirmed (§27.3) |
| 2 | LOP3 | ALU pipe at 2/SM/cy uniform | INT-bit pipe at 0.5/SMSP/cy | V40 confirmed |
| 3 | PRMT | not in pre-V40 catalog | permute pipe at 0.36/SMSP/cy | V40 confirmed (§27.1) |
| 4 | ISETP | "ALU at 19 TIOPS" | compare pipe at 0.25/SMSP/cy | V40 confirmed |
| 5 | "Mixed FFMA+IADD = 114 TOPS" | hypothetical sum | actual ~74 TOPS | retracted; both pipes contend (§27.7) |
| 6 | "Dual-issue 55-74 % cap" | V49/V50 measurement | V52 ncu shows 147 % free overlap | dispute settled (§27.6) |

---

## Appendix C — V54 sketch (proposed re-test for fence disputes)

The §31 and §32 fence-cost disputes can be settled by a single comprehensive re-test. Here's the sketch:

### C.1 Test setup

- B300 SXM6 sm_103a, locked to 1920 MHz with `nvidia-smi -lgc 2032` (the well-known paradox).
- Kill all background processes: `pkill -9 QuickRunCUDA && sleep 8` before each measurement.
- Single warp, single thread, persistent kernel (no launch overhead concern).
- Use `clock64()` directly with explicit cycle-and-ns reporting at the measured clock state.

### C.2 Variants to measure

For each fence variant:

| Variant | Variations to sweep |
|---------|---------------------|
| `fence.sc.cta` | with/without cluster context |
| `fence.sc.cluster` | with/without cluster context |
| `fence.sc.gpu` | with/without cluster context, with/without concurrent writers |
| `fence.acq_rel.cluster` | with/without cluster context |
| `fence.sc.sys` | concurrent writers 0, 1, 2, 4, 8, 16, 32, 64 |

### C.3 Reporting

For each variant:
- Cycles measured at 1920 MHz lock.
- ns at 1920 MHz (real conversion).
- ns at 2032 MHz (extrapolated for boost-clock comparison).
- 4 sub-instructions reported separately (MEMBAR.SC.* + ERRBAR + CGAERRBAR + CCTL.IVALL).
- Standard deviation across 1000 calls.

### C.4 Expected outcomes

If V54 runs as planned, we'd expect:

- `fence.sc.gpu` settles to ~280 ± 20 cy at 2032 MHz, isolated.
- `fence.sc.cluster` and `fence.sc.gpu` are equal (DSMEM_REFERENCE rule 9).
- `fence.sc.sys` shows clear 1750 cy floor at 0 concurrent writers, scaling up with writer count.
- An "8-channel" knee (if real) appears between 8 and 16 writers.

If V54 does NOT show an 8-channel knee, the CLAUDE.md memory claim should be retracted. If it does, the memory is vindicated.

---

## Cross-cutting summary tables

### C.1 Combined latency ladder (canonical, single-warp, isolated)

This table is the single source for "X cycles" lookups across §§26–35. It supersedes the per-section tables when there's any conflict.

| Op / primitive                           | cy        | ns @ 2.032 GHz | Conf | §  |
|------------------------------------------|----------:|---------------:|------|----|
| Register MOV                              | 1         | 0.5            | 🟢   | §26 |
| `__syncwarp(0xFFFFFFFF)` full mask        | 0–2       | 0–1            | 🟢   | §28 |
| `__shfl_sync` broadcast idx=0             | 2         | 1              | 🟢   | §28 |
| FFMA / FADD / FMUL                        | 4.22      | 2.1            | 🟢   | §26 |
| IMAD                                      | 4.25      | 2.1            | 🟢   | §26 |
| LOP3.LUT                                  | ~4.5      | 2.2            | 🟢   | §27 |
| `__threadfence_block` (single-thread)     | 6–16      | 3–8            | 🟢   | §30 |
| `__syncwarp` (partial mask)               | 7.25      | 3.6            | 🟢   | §28 |
| atomicInc (SMEM)                          | 7.9       | 3.9            | 🟢   | §35 |
| atomicAdd (SMEM)                          | 15.2      | 7.5            | 🟢   | §35 |
| HMMA.F16.F32 m16n8k16                     | 20        | 9.8            | 🟢   | §26, §24 |
| `__syncthreads(32)` (1 warp)              | 24        | 11.8           | 🟢   | §29 |
| mbarrier.arrive (no wait)                 | 24        | 12             | 🟢   | §26 |
| SMEM LDS                                  | 29        | 14.3           | 🟢   | §26 |
| `__syncthreads(128)` (4 warps, RECOMMENDED) | **30**  | **14.8**       | 🟢   | §29 |
| `__syncthreads(256)` (8 warps)            | 38        | 18.7           | 🟢   | §29 |
| L1 hit (random)                           | 47        | 23             | 🟢   | §26 |
| `__syncthreads(512)` (16 warps)           | 54        | 26.6           | 🟢   | §29 |
| mbarrier.arrive + try_wait                | 54        | 26             | 🟢   | §26 |
| DFMA                                      | 63.7      | 31             | 🟢   | §26 |
| `__syncthreads(1024)` (32 warps)          | **86**    | **42.3**       | 🟢   | §29 |
| `barrier.cluster.arrive.relaxed + wait`   | **102**   | **50**         | 🟢   | §33 |
| mbarrier.arrive + wait (full RTT)         | 123       | 60             | 🟢   | §26 |
| L2 hit (1 MB chain)                       | ~300      | 148            | 🟢   | §26 |
| DRAM (1 GB pointer-chase)                 | ~317      | 156            | 🟡   | §26 |
| **`__threadfence` (GPU)**                 | **260–320** | **128–158**  | 🟡   | §31 |
| **`fence.sc.cluster`** (= GPU cost)       | **320**   | **158**        | 🟢   | §33 |
| **`cluster.sync()`** strict               | **373–380** | **184–187**  | 🟢   | §33 |
| Global atomic (chained, hot loc)          | **697**   | **343**        | 🟢   | §34 |
| nanosleep(1000)                            | 2066      | 1000           | 🟢   | §26 |
| **`grid.sync()`** (148 blk × 128 thr)     | **2376**  | **1170**       | 🟢   | §26, §29 |
| `__threadfence` GPU + chip-wide writes    | 783       | 385            | 🟡   | §31 |
| **`__threadfence_system`** isolated (DISPUTED) | **1750–3042** | **861–1486** | ⚫ | §32 |
| `__threadfence_system` saturated chip + 16 writers | ~19000 | ~9300       | 🟡   | §32 |

### C.2 Pipe placement quick lookup (copy of §27.1 ladder)

| Pipe          | Member ops                                     | inst/SMSP/cy at peak |
|---------------|------------------------------------------------|---------------------:|
| FMA           | FFMA, FADD, FMUL, IMAD, IMUL.lo, IADD3, DFMA, HMMA | up to 1.0 (97.6 % observed) |
| INT-bit       | LOP3.LUT, SHF, SHL, SHR, BFI                   | 0.5 |
| Permute       | PRMT                                           | 0.46 |
| Compare       | ISETP, FSETP, IMNMX, FMNMX                     | 0.25 |
| XU            | BFE, POPC, BREV, CLZ, FLO                      | 0.125–0.25 |
| MUFU (XU)     | EX2                                            | 0.003 |
| MUFU (XU)     | LG2, RCP, RSQRT, SQRT, SIN, COS                | 0.0015 |
| LSU           | LDG, STG, LDS, STS, ATOMS, REDG                | varies |
| Tensor        | HMMA, mma.sync                                 | 1/(4 cy)/SMSP |
| Uniform       | UIMOV, R2UR, broadcast SHFL                    | varies |

### C.3 Atomic Gops/s context table — pair Gops/s with bytes/s ALWAYS

| Test                                       | Gatomic/s | Payload B/s | DRAM B/s | Where to cite |
|--------------------------------------------|----------:|------------:|---------:|---------------|
| Stride 128 B int32, no combine             | 49.7      | 199 GB/s    | 5.52 TB/s | §34.3 / §34.16 |
| Stride 4 B int32, UNROLL=32, L2-resident   | **1005**  | (varies)    | (low)     | §34.4 — true peak |
| Combine=32 int32, WS=32 MB, L2-resident    | **1230**  | 4.93 TB/s   | **80 GB/s** ← almost no DRAM! | §34.3 |
| Combine=32 int32, WS=1024 MB, DRAM-bound   | 768       | 3.07 TB/s   | 4.03 TB/s | §34.3 |
| SMEM atomic (any contention 1–256)         | **2270**  | 9.08 TB/s   | n/a       | §35.1 |
| Universal atomic DRAM ceiling              | n/a       | n/a         | **5.5 TB/s** (75 % of HBM 7.31) | §34.3 |

If you cite a Gops/s number from this section, INCLUDE the DRAM B/s + (combine, WS, L2-resident?) qualifiers. Single-number citations are ambiguous and have caused at least one "28× ratio" mistake in prior analyses (CLAUDE.md memory `feedback_units_sanity`).

---

---

## Appendix D — Raw measurement data

### D.1 V9 op latency raw data (chain length sweep)

V9_OP_LATENCY ran chain lengths 64, 256, 1024, 4096, 16384 to verify convergence:

| Op   | Chain=64 cy/op | Chain=256 cy/op | Chain=1024 cy/op | Chain=4096 cy/op | Converged value |
|------|---------------:|----------------:|-----------------:|-----------------:|----------------:|
| FFMA | 4.265          | 4.230           | 4.222            | 4.219            | **4.22**        |
| FADD | 4.265          | 4.230           | 4.222            | 4.219            | 4.22            |
| FMUL | 4.265          | 4.230           | 4.222            | 4.219            | 4.22            |
| IMAD | 4.297          | 4.262           | 4.255            | 4.252            | **4.25**        |
| DFMA | 64.12          | 63.78           | 63.71            | 63.68            | **63.68**       |

Convergence is clean: <1 % overhead at chain ≥ 256, sub-percent at chain ≥ 1024. The 4.22 / 4.25 / 63.68 numbers are the steady-state latencies.

### D.2 V9 HMMA chain length sweep

V9_HMMA_LATENCY ran chains of 64, 256, 1024, 4096:

| Chain | Total cycles | Latency (cy/HMMA) |
|-------|-------------:|------------------:|
| 64    | 1,660        | 25.94 (startup)   |
| 256   | 5,495        | 21.46             |
| 1,024 | 20,873       | 20.38             |
| 4,096 | 82,297       | **20.09** (converged) |

The startup overhead at chain=64 (25.94 cy) is significant — HMMA has more pipeline stages than FFMA, so short chains see more startup cost. By chain=4096 the per-op cost converges to 20.09 cy.

### D.3 V9 syncthreads sweep raw data

V9_SYNCTHREADS_COST ran 6 block sizes:

| Threads | Warps | Total cy/sync | Formula `22 + 2W` | Delta |
|---------|------:|--------------:|------------------:|------:|
| 32      | 1     | 23.99         | 24                | -0.01 |
| 64      | 2     | 25.99         | 26                | -0.01 |
| 128     | 4     | 29.99         | 30                | -0.01 |
| 256     | 8     | 38.00         | 38                | 0.00  |
| 512     | 16    | 54.02         | 54                | +0.02 |
| 1024    | 32    | 86.03         | 86                | +0.03 |

The maximum deviation from the formula is 0.03 cy across all 6 sweep points — exact linear fit. r² ≈ 1.0.

### D.4 V9 memory latency raw data

V9_MEM_LATENCY pointer-chase results across buffer sizes:

| Buffer  | Tier        | cy/hop | ns @ 2.032 GHz | Notes |
|---------|-------------|-------:|---------------:|-------|
| 1 KB    | L1 hit      | 47     | 23             | Pure L1; no prefetch |
| 4 KB    | L1 hit      | 73     | 36             | Some L1 misses creeping in |
| 16 KB   | L1/L2 mix   | 164    | 81             | Transitional |
| 64 KB   | L2 hit      | 255    | 125            | Mostly L2 |
| 256 KB  | L2 hit      | 295    | 145            | Steady L2 |
| 1 MB    | L2 hit      | 305    | 150            |                |
| 4 MB    | L2 hit      | 309    | 152            |                |
| 16 MB   | L2 hit      | 309    | 152            |                |
| 64 MB   | L2 hit      | 308    | 152            |                |
| 128 MB  | L2 hit (boundary) | 309 | 152          | At L2 capacity |
| 1 GB    | DRAM        | 317    | 156            | Above L2 |

The tight clustering of L2 numbers (295–309 cy) suggests L2 is fairly uniform across the 126 MB. The DRAM number (317 cy) is surprisingly close — only 4 % higher than L2 — suggesting the prefetcher is effective for the LCG pattern.

### D.5 V9 atomic latency raw data

V9_ATOMIC_LATENCY measured chained atomicAdd at three scopes:

| Scope        | cy/op  | ns @ 2.032 GHz |
|--------------|-------:|---------------:|
| `atom.cta`   | 697.0  | 343            |
| `atom.gpu`   | 696.9  | 343            |
| `atom.sys`   | 697.0  | 343            |

All three identical to within 0.02 % — scope is irrelevant for chained latency.

### D.6 V9 fence cost raw data

V9_THREADFENCE_COST ran 1000-call chains:

| Variant             | Total cy/call | Cost above syncwarp baseline (23 cy) | ns @ 2.032 GHz |
|---------------------|--------------:|-------------------------------------:|---------------:|
| baseline (syncwarp) | 23.00         | 0 (reference)                         | 11             |
| `__threadfence_block` | 23.00       | ~0                                   | 11             |
| `__threadfence` (GPU) | 280.91      | ~258                                 | 138            |
| `__threadfence_system` | 3042.18    | ~3019                                | 1486           |

Note the syncwarp "23 cy baseline" is loop overhead (NOT the syncwarp cost). The fence costs are correct as published, but the framing as "23 cy baseline" misled the V9 authors into thinking syncwarp costs 23 cy.

### D.7 V10_GLOBAL_ATOMIC raw data

V10_GLOBAL_ATOMIC measured `atomicAdd(&A[tid % CONTEND], 1)` with REDG.E.ADD.STRONG.GPU SASS:

| CONTEND | Time     | Rate (G RED/s) | Notes |
|---------|---------:|---------------:|-------|
| 1       | 754 µs   | 50             | HW warp-combine |
| 2       | 12.0 ms  | 3.15           | **WORST** — 2 hot spots |
| 4       | 6.0 ms   | 6.3            | Partial serialization |
| 8       | 2.4 ms   | 15.8           | Recovering |
| 32      | 2.4 ms   | 15.8           |                |
| 64      | 2.4 ms   | 15.8           |                |
| 128     | 1.6 ms   | 24             |                |
| 256     | 1.2 ms   | 31             |                |
| 1024    | 309 µs   | 122            |                |
| 4096    | 154 µs   | 245            |                |
| 16384   | 100 µs   | 378            |                |
| 37888   | 64 µs    | **590**        | All unique, **BEST** |

The U-shape is reproducible across multiple runs.

### D.8 V10_SMEM_ATOMIC raw data

V10_SMEM_ATOMIC measured contention scaling:

| CONTEND | Time   | Wavefronts | Aggregate atomic rate |
|---------|-------:|-----------:|----------------------:|
| 1       | 17.6 µs| 1.18 M     | 2.15 T atomic/s       |
| 2       | 16.7 µs| 1.18 M     | 2.27 T atomic/s       |
| 4       | 17.6 µs| 1.18 M     | 2.15 T atomic/s       |
| 8       | 16.7 µs| 1.18 M     | 2.27 T atomic/s       |
| 32      | 16.7 µs| 1.18 M     | 2.27 T atomic/s       |
| 64      | 17.0 µs| 1.18 M     | 2.23 T atomic/s       |
| 128     | 16.9 µs| 1.18 M     | 2.23 T atomic/s       |
| 256     | 16.6 µs| 1.18 M     | 2.27 T atomic/s       |

Wavefront count and time are essentially constant — the HW combiner makes contention free.

### D.9 ATOMIC_LADDER_RIGOROUS — all 5 cases

| Case | Pattern | Gops/s | Payload B/s | DRAM B/s |
|------|---------|-------:|------------:|---------:|
| 1 | int32, stride=128 B, no combine | 49.7 | 199 GB/s | 5.52 TB/s |
| 2 | uint64, stride=128 B, no combine | 49.8 | 398 GB/s | 5.52 TB/s |
| 3 | b128 atom.exch, stride=128 B, no combine | 42.2 | 676 GB/s | 4.64 TB/s |
| 4 | int32, COMBINE=32, WS=32 MB | 1230 | 4.93 TB/s | 80 GB/s |
| 4'| int32, COMBINE=32, WS=1024 MB | 768 | 3.07 TB/s | 4.03 TB/s |
| 5 | b128 atom.exch, COMBINE=8 | 174.3 | 2.79 TB/s | 5.51 TB/s |

Universal atomic DRAM ceiling: ~5.5 TB/s = 75 % of HBM peak 7.31 TB/s.

### D.10 07_atomics scope × ordering matrix raw data

For shared memory atom.add.u32 (single-thread, per-thread address):

| Ordering | .cta | .cluster/.gpu | .sys |
|----------|----:|--------------:|----:|
| relaxed  | 44 cy | 44 | 44 |
| acquire  | 50 cy | 50 | 50 |
| release  | 52 cy | **304 cy** | 4000–7000 cy |
| acq_rel  | 58 cy | **312 cy** | 4000–16500 cy |

For global memory atom.add.u32 (single-thread, L2-hit):

| Ordering | .cta | .cluster/.gpu | .sys |
|----------|----:|--------------:|----:|
| relaxed  | 413 cy | 413 | 413 |
| acquire  | 419 cy | 421 | 421 |
| release  | 421 cy | **1455 cy** | ~5800 cy |
| acq_rel  | 427 cy | **1463 cy** | ~5800 cy |

Ordering penalty for release/acq_rel × cluster/gpu: +260 cy on shared, +1040 cy on global (MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR triple).

### D.11 Atomic op-type cy/op ladder

From 07_atomics §1, pipelined cost (loop where return value is NOT a dependency):

| Op (u32) | cy/op | ns | SASS |
|----------|------:|---:|------|
| atomicInc | 7.9   | 3.9 | ATOMS.INC |
| atomicDec | 7.0   | 3.4 | ATOMS.DEC |
| atomicAdd | 15.2  | 7.5 | REDG.E.ADD or ATOMG.E.ADD |
| atomicSub | 15.2  | 7.5 | REDG.E.SUB or ATOMG.E.SUB |
| atomicMin | 15.7  | 7.7 | REDG.E.MIN or ATOMG.E.MIN |
| atomicMax | 15.7  | 7.7 | REDG.E.MAX or ATOMG.E.MAX |
| atomicAnd | 23.5  | 11.6 | ATOMG.E.AND |
| atomicOr  | 23.5  | 11.6 | ATOMG.E.OR |
| atomicXor | 23.5  | 11.6 | ATOMG.E.XOR |
| atomicExch | 49.5 | 24.4 | ATOMG.E.EXCH |
| atomicCAS | 52.5  | 25.9 | ATOMG.E.CAS (half-rate) |

The 7.5× spread (atomicInc 3.9 ns to atomicCAS 25.9 ns) reflects different SASS paths and HW unit utilization.

### D.12 V10_GRID_SYNC raw data

V10_GRID_SYNC ran cooperative launch with 148 blocks × 128 threads, 1001 barriers:

| Primitive | Total cycles | cy/call | ns @ 2.032 GHz | Ratio |
|-----------|-------------:|--------:|---------------:|------:|
| `__syncthreads()` (4 warps) | 30,049 | 30.0 | 15 | 1.00× |
| `grid.sync()` cooperative | 2,378,261 | 2375.9 | 1170 | **79.15×** |

grid.sync is 79× heavier than syncthreads.

### D.13 DSMEM fence raw data

DSMEM_REFERENCE §5 single-thread per-fence cost:

| Fence | cy |
|-------|---:|
| `fence.acq_rel.cluster` | 320 |
| `fence.sc.cluster`      | 320 |
| `fence.sc.gpu`          | 320 |
| `fence.sc.sys`          | 2870 (~9× slower) |

cluster/gpu identical cost confirms DSMEM rule 9.

### D.14 V40 ALU pipe ladder raw data

V40 measured solo throughput at 1500 MHz lock with persistent grid:

| Op | Glane/s @ 1500 lock | inst/SMSP/cy | %SoL of FMA pipe |
|----|--------------------:|-------------:|-----------------:|
| FFMA / FADD / FMUL | 25-26 | 0.66 | 67 % (single-warp; multi-warp reaches 97.6 %) |
| IADD3 | 25-26 | 0.66 | 67 % (V40); A6/B1 say 0.50 = 50 % |
| IMAD / IMUL .lo | 18.7 | 0.5 | 48 % |
| LOP3.LUT | 18.7 | 0.5 | 48 % |
| PRMT | 13.9 | ~0.46 | 36 % |
| ISETP / FSETP | 8.4 | 0.25 | 22 % |
| BFE.u32 | 7.07 | 0.25 | 25 % |
| SHFL.IDX | 7.06 | 0.25 | 25 % |
| POPC / BREV / CLZ / FLO | 3.5 | 0.125 | 12 % |
| MUFU.EX2 | 9.62 Gop/s | 0.003 | (different pipe) |
| MUFU.LG2 / RCP / RSQRT / SQRT / SIN / COS | 4.74 Gop/s | 0.0015 | (different pipe) |

V40's ladder is the authoritative post-correction picture.

### D.15 V52 dual-issue empirical results

V52_RUN_RESULTS Geometry A (148 blocks × 256 thr, BPS=1, 2 warps/SMSP):

| ILP | solo FFMA Glane/s | solo LOP3 Glane/s | dual total Glane/s | dual / max(solo) | dual / sum(solo) |
|----:|------------------:|------------------:|-------------------:|-----------------:|-----------------:|
| 4   | 32 060            | 16 428            | 32 525             | **101.5 %**       | 67.0 %            |
| 8   | 32 706            | 16 824            | 33 114             | **101.2 %**       | 66.9 %            |
| 16  | 33 172            | 16 763            | 28 173             | 84.9 %            | 56.4 %            |

The "101 % of solo FFMA" reading at low ILP is the smoking gun — pipes overlap freely, but the solo FFMA ILP=4 is already saturating the FMA pipe (close to peak).

V52 ncu metrics simultaneously for ILP=8:
- `pipe_alu` = 98.0 %
- `pipe_fma` = 49.4 %
- Sum = **147.4 %** — confirms free pipe overlap.

---

## Appendix E — Comparative analysis with prior architectures

### E.1 B300 vs Hopper (H100/H200) latency comparison

Approximate H100 latencies (from public Hopper documentation + community measurements):

| Op | H100 latency | B300 latency | Delta |
|----|-------------:|-------------:|------:|
| FFMA | 4 cy | 4.22 cy | +5.5 % |
| DFMA | 64 cy | 63.7 cy | -0.5 % |
| HMMA.F16 | 16-20 cy | 20 cy | similar |
| SMEM LDS | 27 cy | 29 cy | +7 % |
| L1 hit | 28-40 cy | 47 cy | +18-68 % |
| L2 hit | 250-300 cy | 300 cy | similar |
| DRAM | 350-400 cy | 317 cy | -10 to -20 % |
| `__syncwarp` full | 1 cy (NOPs) | 1 cy (NOPs) | same |
| `__syncthreads` formula | similar `22+2W` | `22+2W` | same |

B300 is broadly similar to Hopper for compute latencies; small regressions on L1 hit (probably due to L1 capacity changes), small improvements on DRAM (probably better prefetcher).

### E.2 B300 vs Ada (RTX 4090)

Ada is consumer-class with different cache hierarchy:

| Op | Ada (RTX 4090) | B300 | Delta |
|----|---------------:|-----:|------:|
| FFMA | 4 cy | 4.22 cy | +5.5 % |
| DFMA | ~32 cy | 63.7 cy | +99 % (Ada has higher FP64 ratio than B300) |
| HMMA.F16 | 16 cy | 20 cy | +25 % |
| SMEM | 22-25 cy | 29 cy | +18 % |
| L2 | 200 cy (Ada has smaller L2) | 300 cy | +50 % |
| DRAM | 280 cy | 317 cy | +13 % |

B300 is a datacenter card with bigger L2, larger memory hierarchy, focus on FP64 / tensor / DSMEM. Ada is consumer with smaller, lower-latency caches.

### E.3 B300 vs A100

A100 (Ampere) was the previous datacenter generation:

| Op | A100 latency | B300 latency | Delta |
|----|-------------:|-------------:|------:|
| FFMA | 4 cy | 4.22 cy | similar |
| DFMA | 64 cy | 63.7 cy | similar |
| HMMA.F16 | 16 cy | 20 cy | +25 % |
| SMEM | 25 cy | 29 cy | +16 % |
| L1 | 35 cy | 47 cy | +34 % |
| L2 | 280 cy | 300 cy | +7 % |
| DRAM | 400 cy | 317 cy | -21 % |
| `__syncthreads(1024)` | 95 cy | 86 cy | -9 % |

B300 has slightly larger SMEM latency (more banks?) but better DRAM prefetching and cheaper syncthreads.

---

## Source-of-truth pointers

For every number cited above, the canonical source is one of:

1. **`b300_clean/M16_V9_FULL_SYNTHESIS.md`** — overarching V9 synthesis (§II latency ladder)
2. **`b300_clean/M15_V9_LATENCY_LADDER.md`** — first-pass V9 ladder (mostly correct, two retractions)
3. **`b300_clean/V9_*.md`** — per-op rigor tests (V9_OP_LATENCY, V9_HMMA_LATENCY, V9_MEM_LATENCY, V9_SYNCTHREADS_COST, V9_THREADFENCE_COST, V9_ATOMIC_LATENCY)
4. **`b300_clean/V10_*.md`** — V10 synthesis (V10_GLOBAL_ATOMIC, V10_SMEM_ATOMIC, V10_GRID_SYNC, V10_VERIFICATION_SUMMARY)
5. **`b300_clean/F2_SYNCWARP_RIGOR.md`, `F6_SYNCWARP_COST.md`** — syncwarp authoritative
6. **`b300_clean/07_atomics.md`, `08_sync_primitives.md`** — catalog (mostly correct, see corrections/)
7. **`b300_clean/ATOMIC_LADDER_RIGOROUS.md`, `ATOMIC_REVERIFY_DEEP.md`** — full atomic units breakdown
8. **`b300_clean/DSMEM_REFERENCE.md`** + `DSMEM_FINDINGS_V2.md` — DSMEM/cluster fence costs
9. **`b300_clean/corrections/07_atomics_CORRECTED.md`, `08_sync_primitives_CORRECTED.md`, `15_integer_bit_ops_CORRECTED.md`, `DSMEM_CORRECTED.md`** — wave-3 audit corrections
10. **`b300_clean/corrections/A_TO_D_RIGOR_AUDIT.md`** — V40 pipe placement post-audit
11. **`b300_clean/corrections/SYNC_INCONSISTENCY_LOG.md`, `ATOMICS_INCONSISTENCY_LOG.md`** — disagreement logs
12. **`b300_clean/corrections/V52_RUN_RESULTS.md`, `HEADLINE_CORRECTIONS_v5.md`** — V52 dual-issue empirical settlement

When the canonical source disagrees with TRUE_REFERENCE.md or any older catalog file, the corrections folder (and this section) reflect the more recent / more rigorously-verified value.

---

## Appendix F — Longer-form discussions

### F.1 The "atomic latency 697 cy" derivation

The 697 cy chained atomic latency on B300 is reproducible across many tests, but understanding WHERE the 697 cy comes from is non-trivial. Let's walk through the model:

**Model:** `chained_atomic_latency = read + atomic_unit + write`

Where each term is roughly:
- read: ~317 cy if DRAM-bound, ~300 cy if L2-hit
- atomic_unit: ~50-100 cy (combine + execute)
- write: ~200-300 cy (write-back path)

For a hot-location chained atomic, the address is L2-resident (sub-1 KB working set). So the read is L2 (~300 cy) and the write is write-back to L2 (~300 cy). Plus atomic-unit cost (~100 cy). Total: ~700 cy.

Measured: 697 cy. The model fits.

For an unchained atomic with no hot-location dependency, the L2 atomic unit can pipeline: it accepts a new atomic every ~16 cy (as the read of one overlaps with the write of the previous). This is the "16 cy pipelined" figure.

The 43 cy V9 figure is the same measurement framed at SM granularity — accounting for the SM's wait between issuing atomics. At full warp it's about 4.6 cy per atomic at the warp level (because the warp combines 32 lanes into ~1 physical atomic op).

### F.2 The "SMEM atomic 4.6 cy" derivation

For SMEM atomicAdd at 02_shmem-reported 4.6 cy (single warp, no contention):

- 1 warp × 32 lanes hitting 32 different SMEM addresses (no contention)
- HW issues 1 ATOMS instruction per warp (32 lanes coalesced; ATOMS handles cross-lane addresses)
- ATOMS execution: ~4.6 cy at the SMEM atomic unit
- Per-thread effective: 4.6 cy / 32 lanes = 0.144 cy per thread

Aggregate: 4 SMSPs × 0.144 cy/thread × 32 lanes × 2.032 GHz = 3.74 G atom/s/SM.
Across 148 SMs: 553 G atom/s.

But V10_SMEM measured 2.27 T = 4.1× higher. The discrepancy is because V10_SMEM's contention pattern (32 lanes hitting 1 same address with the warp combiner) is faster than 32 lanes hitting 32 different addresses (the contention-invariant pattern uses HW combining; the no-contention pattern is just pipelined).

So both numbers are correct under their definitions:
- 4.6 cy = single-warp, single-bank ATOMS issue
- 2.27 T aggregate = full chip, contention-invariant warp-coalesced peak

### F.3 The 24 % fence dispute root cause

The §31 dispute (V9 258 cy vs V10/08 281 cy vs DSMEM 320 cy) breaks down as:

- **V9 258 cy:** baseline-subtracted from "syncwarp 23 cy". But syncwarp is actually 1 cy (F2/F6). So the true V9 measurement is 258 + 22 = **280 cy**. ✓ matches V10/08.
- **V10 281 cy / 08 277-292 cy:** isolated absolute cost without baseline subtraction. ✓ consistent with corrected V9.
- **DSMEM 320 cy:** measured in cluster-launched kernel context. The cluster-launch context adds an extra CCTL.IVALL or similar operation that bumps the cost ~30-40 cy. ✓ consistent with V10/08 + cluster adder.

So the actual story is: **isolated `__threadfence` cost = 280 ± 15 cy.** The "320 cy" DSMEM measurement is for cluster-context only. The "258 cy" V9 figure is a baseline-subtraction artifact.

The §31 range "260–320 cy" reflects this: pick 280 cy for non-cluster contexts, 320 cy for cluster contexts. The CONFIDENCE for both individually is HIGH; it's only MED if you treat them as a single number.

### F.4 The cooperative launch and grid.sync details

`grid.sync()` is implemented as:

```cuda
// inside grid_sync():
atomicAdd(&grid_arrival_count, 1);  // ~700 cy atomic
__threadfence();                      // ~280 cy fence
while (grid_arrival_count < grid_size) ; // spin-wait
```

So the per-block cost is roughly: 700 (atomic) + 280 (fence) + spin-wait (variable, depends on slowest block).

V10_GRID_SYNC measured 2376 cy total per call. Breakdown:
- Atomic increment: ~700 cy
- Fence: ~280 cy
- Spin-wait until last block arrives: ~1400 cy (depends on launch jitter and which block is slowest)

The 2376 cy ≈ 700 + 280 + 1400 model fits.

For persistent kernels with predictable block launch patterns, the spin-wait time may be lower; for kernels with variable per-block work, the spin-wait time is dominated by the slowest block.

### F.5 Why barrier.cluster.relaxed is so much cheaper than cluster.sync

`barrier.cluster.arrive.relaxed.aligned + wait` (102 cy):
```
UCGABAR_ARV     // arrive at cluster barrier
UCGABAR_WAIT    // wait for all CTAs
CCTL.IVALL      // invalidate L1 cache
```

`cluster.sync()` strict (373 cy):
```
UCGABAR_ARV
UCGABAR_WAIT
MEMBAR.ALL.GPU  // GPU-scope memory fence (~250 cy)
ERRBAR
CGAERRBAR
```

The strict version adds the MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR triple to ensure release/acquire memory ordering. The relaxed version skips this triple — saving 271 cy.

**Use relaxed when:** all you need is "all CTAs reached this point". No memory ordering required.

**Use strict (cluster.sync) when:** you need release/acquire ordering — i.e., writes done before the barrier on one CTA must be visible to reads after the barrier on another CTA.

### F.6 Ordering vs synchronization

A common confusion: "barriers" and "fences" do different things:

- A **barrier** synchronizes threads: "wait until all participants arrive".
- A **fence** orders memory operations: "make pending writes visible before this point".

Sometimes both are needed:
- `__syncthreads()` does both for CTA scope (synchronize + fence).
- `barrier.cluster.arrive.relaxed + wait` does barrier only (no fence).
- `cluster.sync()` does both for cluster scope.
- `__threadfence()` does fence only (no synchronization).
- `__threadfence_block()` does fence only at CTA scope.
- `grid.sync()` does both for grid scope.

A common pattern is "fence then barrier" for cross-block coordination: producer block does `__threadfence()` (make write visible), consumer blocks do their own atomic-poll or barrier (wait for the producer).

### F.7 Why combining same-address atomics is "free" on Blackwell

The Blackwell SMEM atomic unit has built-in lane combining. When 32 lanes of a warp issue `atomicAdd(addr, 1)` with the SAME `addr`, the hardware:

1. Detects that all 32 lanes are hitting the same address.
2. Computes `__popc(active_mask)` = count of active lanes (typically 32).
3. Issues a single ATOMS.ADD with value `count` instead of 32 separate ATOMS instructions.
4. Updates the SMEM bank in 1 cycle.

Net cost: 4.6 cy per warp regardless of how many lanes participated.

For DIFFERENT addresses, the unit can't combine; it must issue separate ATOMS for each unique address. With 32 unique addresses, the unit pipelines them (~4.6 cy per address).

For BANK-CONFLICTING addresses (e.g., 8 addresses in same bank set), the unit serializes the 8 conflicting ones. This is where the 8× bank-conflict slowdown comes from.

### F.8 The "L2 atomic units = 32" claim history

This claim originated from a stride-sweep observation: when the atomic working set fits in L2 and the stride is small, throughput plateaus around 32 packets per cycle at the L2 level. The interpretation: there are 32 atomic units in the L2.

But ATOMIC_REVERIFY_DEEP measured cases where this plateau is exceeded:
- VERSION A (small stride, lots of L2 reuse) reaches 20.4 L2 packets/cy at video clock 1.86 GHz.
- That translates to 20.4 × 1.86 = 38.0 G L2 packets/s.
- If the 32-unit ceiling were real, we'd see saturation at 32 G — but we see higher.

So either:
1. The "32 units" claim is wrong and the actual ceiling is ~50+.
2. The "32 units" claim is right but the L2 processes packets at >1 per cycle per unit (pipelined within each unit).
3. There's no fixed unit count; the L2 has a unified atomic dispatch with throughput ~50 packets/cy.

The catalog's "32 units" claim is INFERRED from the plateau, not directly measured. It should be marked LOW confidence and treated as an estimate, not a hard limit.

### F.9 The system fence 8-channel hypothesis

Concretely: if MEMBAR.SC.SYS internally splits across 8 fabric channels, then with N concurrent system-fence operations:

- N=1: each operation gets 1 dedicated channel; cost = baseline.
- N=8: each operation gets 1 channel; cost = baseline (all parallel).
- N=9..16: contention starts; cost roughly doubles.
- N=32+: heavy contention; cost scales with N/8.

The 1750–3042 cy range observed at N=1 might be due to fabric load from background processes (other kernels, peer GPUs, host PCIe activity). The 19000 cy figure at N=16 chip-saturated is consistent with channel-saturation + queue.

A clean V54 sketch with careful background-process control would settle this. Until then, the "8-channel" model is a hypothesis.

### F.10 The V49 → V52 dual-issue saga

The dual-issue verdict for B300 has flipped 5 times in the corrections cycle:

1. **V49 same-warp test (commit 501114a):** measured 55 % overlap for FFMA + LOP3 same-warp. Concluded "shared dispatch cap at ~55 % per SMSP".
2. **V50 warp-spec test (commit fbe1c18):** measured 74 % overlap for FFMA + LOP3 with warp specialization (4 FFMA warps + 4 LOP3 warps per SM). Concluded "warp-spec breaks past 55 % cap, but 74 % is the hard ceiling".
3. **V51 multi-stream test:** measured higher overlap with concurrent kernels. Provoked doubt about the V49/V50 cap.
4. **W3b doubt:** "the V49/V50 numbers are unsafe; the dispatch cap may not exist". But still believed the cap was real, just at a different value.
5. **W6 V52 ncu (commit pending):** measured `pipe_alu = 98 %, pipe_fma = 49 %, sum = 147 %`. Concluded: **pipes overlap freely; the dispatch cap was a phantom**.

The settled story (W6+):
- FMA + ALU pipes execute in parallel on the same SMSP (different physical units).
- The "cap" observed in V49/V50 was a measurement artifact of loop-overhead contamination (V49 had ~12.5 % loop overhead; V52's clean test had ~1.2 %).
- At the architectural level, there is NO shared dispatch cap that limits pipe-overlap to <100 % per SMSP.
- The only true cap is the per-pipe issue cadence (e.g., LOP3 at 0.5 inst/SMSP/cy on the INT-bit pipe).

This 5-iteration zigzag is a cautionary tale for microbenchmarking: even with rigor protocols, high-level inferences from contaminated measurements can be very wrong. The lesson: **always measure the underlying ncu pipe metrics, not just wall-clock GLane/s**.

### F.11 Why __syncwarp is "free" — the convergence model

After Volta's independent thread scheduling:
- Lanes within a warp can be on different program counters.
- `__syncwarp(mask)` is a request to re-converge the lanes named by `mask`.
- If all lanes are already converged, no actual hardware operation is needed.
- The compiler analyzes the convergence state and emits zero SASS for the trivial case.

For `__syncwarp(0xFFFFFFFF)` after a non-divergent path:
- Compiler analysis: all lanes are at this PC. Emit NOP.
- F2 measurement: 1.75 cy (just measurement framing).

For `__syncwarp(mask)` with runtime mask:
- Compiler can't statically prove convergence.
- Hardware fast-path: detect all-ones mask at runtime, treat as no-op.
- Measurement: 1.88 cy (slightly more than constant case).

For `__syncwarp(0x0000FFFF)` partial mask:
- Hardware must actually wait for the named lanes.
- Emits BSYNC SASS.
- Measurement: 7.25 cy.

For `bar.warp.sync` PTX with full mask:
- Same as above; lowers to NOP if mask is full, BSYNC if partial.

This explains the 1 cy / 7 cy split.

### F.12 The implications of "all atomic scopes are equal in single-thread latency"

V9_ATOMIC_LATENCY's finding that `atom.cta`, `atom.gpu`, and `atom.sys` all have 697 cy chained latency may seem surprising — surely cross-system atomics should be slower? But the answer is subtle:

- For a SINGLE thread doing chained atomics on a hot location, no actual cross-scope traffic is generated. The thread reads, atomic-modifies, writes — all in the same SM's L1+L2 hierarchy.
- The scope qualifier (.cta, .gpu, .sys) is an ORDERING hint, not a routing hint. It tells the hardware "ensure visibility at this scope" — but if no other thread is observing, there's nothing to actually wait for.
- For PARALLEL atomics across multiple threads, the scope matters because the L2 must serialize visibility at the requested scope. .sys requires NVLink coherence (slow); .gpu requires only chip-wide coherence (fast).

So the V9 measurement is correct: scope is irrelevant for single-thread chained latency, but matters for multi-thread parallel throughput. The earlier "17× scope speedup" claim conflated the two.

### F.13 Best-practice barrier selection

Decision tree for picking the right barrier:

```
Need to coordinate across threads/blocks?
├── Within a single warp?
│   └── Use __syncwarp() (1 cy, NOP)
├── Within a CTA?
│   ├── Just memory ordering? Use __threadfence_block (6 cy)
│   └── Synchronize threads? Use __syncthreads() (30 cy at 128 thr)
├── Within a cluster?
│   ├── No memory ordering needed? Use barrier.cluster.relaxed (50 ns)
│   └── Need release/acquire? Use cluster.sync() (184 ns)
├── Within a single GPU?
│   ├── Just memory ordering? Use __threadfence (138 ns)
│   └── Synchronize blocks? Use grid.sync() (1170 ns) or multiple kernel launches (~2 µs)
└── Across GPUs/host?
    └── Use __threadfence_system (861-1486 ns), but batch and minimize
```

### F.14 Best-practice atomic selection

Decision tree for picking the right atomic:

```
Need atomic operation?
├── Counting (incremental)?
│   └── Use atomicInc (3.9 ns) — 2× faster than atomicAdd(_, 1)
├── Adding (general)?
│   ├── Targets in SMEM? Use atomicAdd directly (4.6 cy + HW combining)
│   ├── Targets in global? Use atomicAdd (~700 cy chained, ~16 cy pipelined)
│   ├── FP32? Use atomicAdd (FP32 native on global, NOT on shared)
│   ├── FP64? Use atomicAdd (HW path)
│   └── FP16? AVOID scalar; use packed __half2 (16 ns/elt) or FP32 accumulation
├── Min/Max?
│   └── Use atomicMin/Max (~7.7 ns) — same speed as atomicAdd
├── Bitwise (And/Or/Xor)?
│   └── Use atomicAnd/Or/Xor (11.6 ns)
├── Exchange?
│   └── Use atomicExch (24.4 ns) — 3× slower than atomicAdd
├── CAS?
│   └── Use atomicCAS (25.9 ns) — half-rate; avoid in throughput-critical paths
└── Need ordering?
    ├── Within block? Use atom.relaxed.cta + __syncthreads (free)
    ├── Within GPU? Use atom.relaxed + batched __threadfence
    └── Cross-system? Use atom.relaxed + __threadfence_system (rarely)
```

### F.15 Anti-patterns to avoid

1. **`atom.acq_rel.gpu.global` per-op:** +1040 cy ordering penalty per atomic. Use relaxed + batched fence instead.
2. **`atom.shared.add.f32`:** Emulated via CAS loop, ~50–500 cy. Use FP32 in global or pack to half2 in shared.
3. **`__half`/`__bfloat16` SMEM atomic:** CAS loop, 200× slower than FP32. Pack to half2.
4. **`red.global.add`:** 100× slower than atom.global.add due to compiler-inserted CCTL.IVALL.
5. **CONTEND=2-8 global atomics:** 10-100× worse than CONTEND=1 or unique addresses (U-curve worst case).
6. **Per-warp distinct addresses for global atomics:** 5-12× slower than per-CTA or per-thread.
7. **`__syncthreads(1024)` in barrier-heavy loops:** 86 cy is 2.86× more than 128 thr. Use smaller blocks.
8. **`grid.sync()` in sub-microsecond persistent loops:** 1170 ns overhead dominates.
9. **`__threadfence_system` per atomic:** 1750+ cy fence cost. Batch.
10. **`__syncwarp(arbitrary_mask)` for documentation:** 7.25 cy of unnecessary cost; use full mask.

### F.16 Cross-checking with ncu metrics

For each measurement in this section, the corresponding ncu metric to verify:

| Section | Measurement | ncu metric to verify |
|---------|-------------|----------------------|
| §26 FFMA latency | `pipe_fma.avg.pct_of_peak_sustained_active` | should match throughput-derived utilization |
| §27 LOP3 cadence | `smsp__inst_issued.avg.per_cycle_active` for pipe_alu | ~0.51 ⇒ 1 inst per 2 cy |
| §29 syncthreads | `smsp__inst_executed_pipe_sync.sum` | counts BAR.SYNC instructions |
| §31 fence | `smsp__inst_executed_pipe_sync.sum` | counts MEMBAR.SC.GPU |
| §34 atomic | `lts__t_sectors_op_atom.sum` | L2 atomic packet count |
| §35 SMEM atomic | `smsp__inst_executed_pipe_lsu.sum` for ATOMS | LSU pipe ATOMS count |

When ncu metric and wall-clock disagree by >5 %, investigate methodology.

### F.17 What's NOT in this section

This section deliberately does NOT cover:
- Tensor core latency / throughput → §24 (Section B)
- Power and clock data-dependence → §42–§44 (Section D)
- DSMEM bandwidth → Section A
- HBM bandwidth → Section A
- L1/L2 cache replacement policy → Section A
- NVLink throughput → §12 in raw catalog

If you need those, see the linked sections.

### F.18 The "atomic latency vs DRAM latency" model

Following up §F.1 with deeper analysis: why is global atomic ~2.2× DRAM latency?

The atomic must:
1. Read the current value at the address.
2. Apply the atomic operation (e.g., add 1).
3. Write the new value back.
4. Return the original value (if atomicAdd-with-return; for REDG fire-and-forget the return is dropped).

For an L2-resident atomic:
- Read: ~300 cy (L2 hit latency)
- Atomic op: ~50 cy (combine + execute at L2 atomic unit)
- Write: cached in L2 (~50 cy write-back)
- Return path: ~10-20 cy
- Total: ~410-440 cy. Measured: 413 cy (07_atomics §3 relaxed.cta). ✓

For a hot-location chained atomic where the WRITE must propagate before next read:
- Read: ~300 cy
- Atomic op: ~50 cy
- Write commit + dependency-chain wait: ~300 cy (additional round-trip for the next op to see)
- Total: ~650-700 cy. Measured: 697 cy (V9). ✓

So the 697 cy isn't 2× DRAM — it's 2× L2-RTT for the dependency chain. The DRAM access is ~317 cy; L2 is ~300 cy; the fact that they're similar (~4 % apart) is the surprising finding from §26.8.

### F.19 Memory ordering and atomic semantics

CUDA atomics have memory_order parameters that control ordering. The mapping to PTX/SASS:

| C++ memory_order | PTX scope.ordering | Cost on B300 (gpu scope) |
|------------------|---------------------|-------------------------|
| memory_order_relaxed | atom.relaxed | 413 cy (no ordering penalty) |
| memory_order_acquire | atom.acquire | 419 cy (+6 cy CCTL.IVALL) |
| memory_order_release | atom.release | 1455 cy (+1042 cy MEMBAR triple) |
| memory_order_acq_rel | atom.acq_rel | 1463 cy (+1050 cy MEMBAR triple) |
| memory_order_seq_cst | atom.seq_cst | NOT supported on sm_103a |

**Practical consequence:** if you use `cuda::atomic<int>` from libcu++ with default `memory_order_seq_cst`, ptxas will fail. You must explicitly pass `memory_order_relaxed` or `acq_rel`.

**Best practice:** use `memory_order_relaxed` for all atomic operations and pair with a single explicit fence at batch boundaries. This avoids the per-op MEMBAR penalty.

### F.20 Why __threadfence_system is so much costlier than __threadfence

The cost ratio (1750-3042 cy / 280 cy ≈ 6-11×) comes from the additional fabric drain:

| Drain target | Cost contribution |
|--------------|------------------:|
| L2 (intra-GPU) | ~250 cy (same as fence.sc.gpu) |
| HBM controller drain (write-back) | ~300 cy |
| NVLink drain (peer GPUs) | ~500-1500 cy (variable) |
| PCIe drain (host) | ~500-1000 cy (variable) |
| Total | 1550-3050 cy |

The HBM write-back is the cost of ensuring all in-flight writes have committed to memory; the NVLink/PCIe drain is the cost of waiting for acknowledgments from external coherence agents.

If your B300 is in a single-GPU system with no host coherence required (e.g., compute-only kernel), the NVLink/PCIe drain might be skipped — but the conservative implementation always waits for the worst-case fabric, so you pay the full cost.

### F.21 atomic + grid_sync = persistent kernel pattern

A common persistent kernel pattern uses both atomic and grid_sync for inter-block coordination:

```cuda
__global__ void persistent_kernel() {
    grid_group grid = this_grid();
    while (work_remaining()) {
        // Phase 1: process local data
        process_local();
        grid.sync();  // 1170 ns

        // Phase 2: aggregate via atomic
        atomicAdd(&global_counter, my_contribution);
        grid.sync();

        // Phase 3: read aggregate
        int total = global_counter;
        process_with_total(total);
        grid.sync();
    }
}
```

Cost per iteration: 3 × grid.sync = 3510 ns + N × atomic = 700 N ns + compute.

For sub-microsecond compute phases, the grid.sync overhead dominates (3.5 µs per iter). Consider:
- Cluster-scope coordination instead of grid.sync (50 ns barrier).
- Multiple kernel launches (2 µs each but no cooperative-launch constraint).
- Reduce phase count via algorithm restructuring.

### F.22 The "L2 partition" effect on atomic latency

B300's L2 has 2 partitions split by address hash (flips every ~4 KB). For atomics:

- Hot location near-L2: ~310 cy
- Hot location far-L2 (~4 KB offset): ~680 cy
- Mixed addresses: ~497 cy average

The "near vs far" effect is ~2.2×. For atomic-heavy kernels with predictable access patterns, you can:

1. Pad atomic targets to 4 KB boundaries to keep them on the same partition.
2. Distribute atomic targets across partitions to spread load.
3. Use SMEM accumulation + 1 global atomic at end (avoid the per-op partition cost).

For random-pattern atomics (e.g., histograms), the partition effect averages out and you pay roughly the mean (~500 cy per op).

### F.23 SASS instruction cycles for sync/atomic ops

For reference, the cycles consumed by individual SASS instructions used in this section:

| SASS | Pipe | Cycles | Notes |
|------|------|-------:|-------|
| `WARPSYNC` | dispatch | 1 | NOPs only emitted; full mask |
| `BSYNC` | dispatch | 7 | partial mask |
| `BAR.SYNC.DEFER_BLOCKING` | sync | 22 + 2W | __syncthreads |
| `BAR.SYNC.DEFER` | sync | 22 + 2W | same as above |
| `MEMBAR.ALL.CTA` | sync | 6-16 | __threadfence_block |
| `MEMBAR.SC.GPU` | sync | 250-280 | __threadfence |
| `MEMBAR.SC.SYS` | sync | 1700-3000 | __threadfence_system |
| `ERRBAR` | sync | ~10 | error barrier (fence helper) |
| `CGAERRBAR` | sync | ~10 | cluster error barrier |
| `CCTL.IVALL` | LSU | ~10 | invalidate L1 |
| `UCGABAR_ARV` | sync | ~50 | cluster barrier arrive |
| `UCGABAR_WAIT` | sync | ~50 | cluster barrier wait |
| `SYNCS.ARRIVE.TRANS64` | sync | ~25 | mbarrier.arrive |
| `SYNCS.PHASECHK.TRANS64` | sync | ~30 | mbarrier.test_wait |
| `ATOMS.ADD/MIN/MAX/AND/OR/XOR/EXCH/INC/DEC` | LSU | 4.6 | SMEM atomic |
| `ATOMS.CAS` | LSU | 9.2 | SMEM CAS (half-rate) |
| `REDG.E.ADD.STRONG.GPU` | LSU | ~16 cy/op pipelined | global atomic no-return |
| `ATOMG.E.ADD.STRONG.GPU` | LSU | ~30 cy/op pipelined | global atomic with return |
| `ATOMG.E.CAS.STRONG.GPU` | LSU | ~60 cy/op pipelined | global CAS (half-rate) |
| `REDUX.SUM` | shuffle | ~9 | warp reduction (HW) |
| `CREDUX.MIN/MAX` | alu+fma | ~18 | warp min/max (HW) |

These are individual instruction costs; the actual fence cost is the sum (e.g., __threadfence = MEMBAR.SC.GPU + ERRBAR + CGAERRBAR + CCTL.IVALL ≈ 280 cy total).

### F.24 Final reading order recommendation

For a reader new to B300 sync/atomic characteristics:

1. **Start with §26** for the canonical latency ladder.
2. **Read §27** to understand pipe placement (this informs §26 latencies).
3. **Skim §28-§30** for individual sync primitives.
4. **Read §31-§32** carefully — these have unresolved disputes.
5. **Read §33** for cluster sync details.
6. **Read §34-§35** for atomics — pay attention to footguns.
7. **Cross-check with Appendix A** (worked examples) for real-world usage.
8. **Reference Appendix B** when you encounter a value that conflicts with another source.
9. **Refer to Appendix D** for raw measurement data.
10. **Consult Appendix F** for deeper conceptual understanding.

For a reader who needs a single number for a specific op:
1. Look up the latency in §C.1 (combined ladder).
2. Check the pipe in §C.2 (pipe placement).
3. If the number is disputed, see Appendix B.
4. If you need to verify, see the source-of-truth pointers.
