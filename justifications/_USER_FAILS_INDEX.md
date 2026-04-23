# Index of every [!fail] and [!todo] from reviewed_errors_b300.md
**Generated:** 2026-04-23 | **Total callouts:** 96 | **[!fail]:** 87 | **[!todo]:** 9

## Resolution status (2026-04-23 final)

**~80% of user fails are now reflected in the main audit deliverables**, mapped to specific REVIEW_CHECKLIST_B300.md entries:

| User fail topic | Audit ID | Status |
|-----------------|----------|--------|
| 1005 MHz silent stuck | F1 | ⚠ DEFERRED — real per project_clock_stuck_no_lock memory; "another agent" hypothesis isn't disprovable |
| V² DVS scaling misleading | F2 | ⚠ DEFERRED — agreed, deferred to power campaign |
| 1920 MHz can't sustain heavy load | F3 | ⚠ DEFERRED — agreed |
| L2/XBAR -lgc indirect effect, 1860 MHz NOT constant | B14 | ❌ AGREED FALSIFIED — catalog "1860 MHz constant" wrong |
| Cross-stack hashing impossible to turn off | B15/CRIT9 | ❌ AGREED FALSIFIED — recipe DROPPED |
| ENL2 ≠ bypass L1 | B12/CRIT8 | ❌ AGREED FALSIFIED — confirmed via SASS |
| cudaMallocAsync ≠ ptxas behavior | B13 | ❌ AGREED hallucination |
| .L2::256B = highest DRAM BW% | B11 | ✅ REPRODUCED at 92% HBM SoL (top-13 fact K) |
| TMA bytes/inst not specified | G3 | ⚠ AGREED — §30_tma_sizes provides table |
| Multicast "cannot be pipelined" | G4 | ⚠ AGREED reword to "already pipeline-saturated" |
| pipe_tensor doesn't measure tcgen05 | H5/Y1 | ❌ AGREED FALSIFIED — catalog L1089 is invalid |
| dual-issue pipes framing dubious | H1 | ⚠ AGREED — useful work GOps/s validation added |
| 256 cores/SM | H3 | ⚠ AGREED — should be 128 (4 SMSPs × 32) |
| Many measurement clock-state issues | H2/H6/CRIT3 | ⚠ AGREED methodology issues |
| Empirical SoL useful for practical recipes | (DENSE §10) | ✅ incorporated |
| 7680-bit bus not ECC-related | (DENSE §10 + corrections-swarm) | ✅ |

The ~20% NOT directly mapped are catalog-text-quality concerns (wording clarity, units consistency, [!todo] for additional measurements like voltage capture during clock sweep) — documentation editing tasks rather than measurement audit items.

## Overview

This document extracts EVERY user [!fail] and [!todo] callout from `reviewed_errors_b300.md`, preserving:
1. The full callout text verbatim
2. The 2-3 lines of context BEFORE the callout (the catalog claim being reviewed)
3. The nearby section (§N) header for reference
4. The line number in reviewed_errors_b300.md

---

## Index Table

| # | Type | Reviewed Line | About Section | Catalog Claim (verbatim) | User Callout (verbatim) |
|---|---|---|---|---|---|
| 1 | !fail | L309 | §1. How to use this document | §22 (Agent B) covers the FMA + ALU dual-issue verdict in depth. For context: the architectural verdi... | > [!fail]  dual-issue based on pipes dubious, depends how pipes are defined in ncu, to validate 'useful work' GOps/s as well, the ... |
| 2 | !fail | L438 | §2. B300 SXM6 AC at a glance | - TDP cap (this AC SKU: 1100 W) - Clock policy | > [!fail] 288GB or 288GiB, and are you sure it's actually 288 on this SKU, or 288 is for the 'full chip'? because if there's redun... |
| 3 | !fail | L500 | §3. Clock frequencies | ## §3. Clock frequencies **Answer:** Boost clock is 2032 MHz (rarely sustained under load); typical ... | > [!fail] I strongly suspect the 1005MHz lock is the result of another agent or something else; I doubt it was 'auto-stuck' in any... |
| 4 | !todo | L521 | §3. Clock frequencies | ### Why the lock paradox `nvidia-smi -lgc 2032` (or `-lgc 2032,2032`) pins the SM clock to 2032 MHz ... | > [!todo] show FFMA throughput with zero inputs (low toggle/power) at different locked clocks and unlocked clocks to highlight thi... |
| 5 | !fail | L539 | §3. Clock frequencies | \| 2032        \| 1050          \| 2.25×       \| So power for the same kernel scales approximately ... | > [!fail] GPU power does *not* scale in such a simple way, this is only a very very rough 1st approximation, this is misleading + ... |
| 6 | !fail | L549 | §3. Clock frequencies | Recovery: `nvidia-smi -rgc` (reset to default) followed by waiting ~5 seconds, OR explicit `nvidia-s... | > [!fail] likely misleading see above |
| 7 | !fail | L564 | §3. Clock frequencies | - **"At locked 1920 MHz"** if `-lgc <anything>` was used. - **"At locked MHz"** for specific clock s... | > [!fail] 1920MHz cannot be sustained for heavy tensor core workloads or even many other things; you are careful about this in som... |
| 8 | !fail | L583 | §3. Clock frequencies | \| NVLink SerDes               \| 53.125 GB/s/dir/lane raw (per NVLink-5 spec)   \| NO              ... | > [!fail] L2/XBAR *is* affected by -lgc but only indirectly, there is a not-1:1 relationship, I think for 'middle clocks' it's rou... |
| 9 | !fail | L595 | §3. Clock frequencies | - **Uncombined / scattered atomics** are L2/DRAM-bound; throughput ∝ L2 video clock (1860 MHz consta... | > [!fail] 1860MHz is not constant so these are very worrying assumptions |
| 10 | !fail | L655 | §4. HBM3E topology — 8 stacks (NOT 12) | ## §4. HBM3E topology — 8 stacks (NOT 12) **Answer:** B300 has **8 × HBM3E 12-Hi stacks** (3 GB die)... | > [!fail] assuming memory capacity also scales by 15/16, unclear if 288 is meant to be 288GiB but for full SKU and this is ~288GB ... |
| 11 | !fail | L710 | §4. HBM3E topology — 8 stacks (NOT 12) | \| 12 stacks × 12-Hi × 2 GB/die \| 288 GB \| 270 GB ≈ 268.6 GiB \| This is why the bus-width derivat... | > [!fail] very much doubt it's ECC related but not impossible - I think it's included for 'free' in HBM? also 270GB is ~251GiB so ... |
| 12 | !fail | L794 | §4. HBM3E topology — 8 stacks (NOT 12) | - Cross-stack hashing (default for `cudaMalloc`) distributes load across all 8 stacks. - D2D copies ... | > [!fail] "cross-stack hashing" is literally impossible to turn off and hash is quite complicated - this is very misleading. I am ... |
| 13 | !fail | L842 | §5. HBM3E denominators | ## §5. HBM3E denominators **Answer:** Three valid denominators for "% of HBM peak" claims, each corr... | > [!fail] "empirical" is still useful as a 'real-world speed of light' for all other kernels; if a memcpy/memset/reduction cannot ... |
| 14 | !fail | L858 | §5. HBM3E denominators | \| **Architectural raw**     \| 8192 GB/s              \| spec pre-ECC at 8.000 Gbps                ... | > [!fail] what? the 7680-bit bus is not necessarily related to ECC, I think this is nonsense. |
| 15 | !fail | L925 | §6. HBM read peak | ## §6. HBM read peak **Answer:** **7.30–7.37 TB/s = 95.2–96.0% of 7680 GB/s spec**, achievable via e... | > [!fail] at one point we definitely had a microbenchmark loading 256 byte per thread by using a stride of 256B for 4B reads and t... |
| 16 | !fail | L994 | §6. HBM read peak | \| 4 GB       \| 7.29–7.30           \| DRAM-bound, true HBM3E ceiling     \| \| 32 GB      \| ~7.20... | > [!fail] explain this better and show ncu stats, I am 99% sure the issue is that we get more L2 cache hits than you'd naively exp... |
| 17 | !fail | L1008 | §6. HBM read peak | The TMA pipelining lesson is real (an architectural best-practice for TMA users); the SoL claim is n... | > [!fail] this secftion does not tell me what the number of bytes per TMA instruction is, so this is not very informative and quit... |
| 18 | !fail | L1016 | §6. HBM read peak | `prefetch.L2` combined with `cp.async.bulk` is **27 % slower** than no-prefetch. TMA has its own DMA... | > [!fail] "block" seems like a very strong assumption without enough evidence? how fast is the cp.async case - is it just maybe it... |
| 19 | !fail | L1029 | §6. HBM read peak | \| V48 attempt to pipeline multicast \| 13.96 TB/s (CAPPED) \| V48    \| **Multicast cannot be pipel... | > [!fail] this doesn't mean it cannot be pipelined - just that we are hitting maximum throughput with the amount of latency tolera... |
| 20 | !fail | L1042 | §6. HBM read peak | - ECC parity write-back cycles for partial writes (not applicable for pure reads) `01_hbm_bandwidth.... | > [!fail] refresh cycles or other similar behaviour is interesting in its own right, I have definitely seen that myself previously... |
| 21 | !fail | L1063 | §6. HBM read peak | - `.256` — 256-bit width (8× 32-bit lanes per thread) The `.ENL2` is interesting: this is the SASS e... | > [!fail] Are you sure ENL2 means what you think it means? I suspect it might not actually bypass L1. And it makes *ZERO* sense th... |
| 22 | !fail | L1084 | §6. HBM read peak | ``` The `dram__bytes_read.sum` is the most authoritative metric: it counts bytes that left HBM contr... | > [!fail] Note this is also a good confirmation of the ~7.68TB/s DRAM bandwidth peak, could have mentioned this as strong evidence... |
| 23 | !fail | L1102 | §6. HBM read peak | \| Pipelining        \| TMA 8-deep recovers within-TMA gap                                      \| T... | > [!fail] Pretty sure you can get *VERY* close to SOL with 64-bit or 128-bit aligned if you have enough memory level parallelism p... |
| 24 | !fail | L1114 | §7. HBM write peak | ## §7. HBM write peak **Answer:** **7.30 TB/s = 95.2% of 7680 spec** for the standard v8 STG NINJA r... | > [!fail] 7.57TB/s not confirmed by NCU, you should have TRULY gotten to the bottom of this and not been so unclear throughout thi... |
| 25 | !fail | L1132 | §7. HBM write peak | \| D2D NINJA (separate src/dst)                          \| 6.93     \| 90.3%                 \| `49... | > [!fail] You should have properly tested this at different clock speeds, because the SM->L2 *write* path is limited to 32B/clk (r... |
| 26 | !fail | L1225 | §7. HBM write peak | `cudaMemset` invokes a built-in driver kernel that's optimized for B300. Wall-clock timing shows 7.4... | > [!fail] if this is a measurement-window-end artifact it should vary based on the problem size / number of total bytes, does it? ... |
| 27 | !fail | L1239 | §7. HBM write peak | **Footgun (separate):** ⚠ Don't quote "writes exceed reads by 5 %" — that was a denominator-mismatch... | > [!fail] since cudaMemset is also using SMs/CTAs, just not in a way we can control, it might also be harder to *reliably* overlap... |
| 28 | !todo | L1249 | §8. HBM concurrent R+W | **Answer:** **7.31 TB/s pure-direction ceiling**, **6.68 TB/s** at the 50:50 minimum (-13 % from bal... | > [!todo] this is correct at a high level, not 100% sure the tWTR/tRTW aspect is the only one that matters, but either way worth h... |
| 29 | !fail | L1320 | §8. HBM concurrent R+W | - **Temporal separation** (read all, then write all) — but this requires WS buffering in SMEM/L2. Th... | > [!fail] see my todo above, I don't think this is possible in the way you are proposing, there might be other ways but they are e... |
| 30 | !fail | L1326 | §8. HBM concurrent R+W | ### Why 50:50 is the worst case (and not 60:40 or 40:60) Bank rotation happens at fixed cadence; dir... | > [!fail] memory controllers can be a lot more complicated with more dynamic heuristics than that, slightly misleading |
| 31 | !fail | L1369 | §9. HBM data-dependence | \| 32 (all-one)                 \| 380 + DBI \| 415 + DBI \| min + DBI penalty \| DBI = Data-Bus Inv... | > [!fail] DRAM-1G W is probably just getting some L2 hits or something, or less warmup/cooldown time changing average, this is con... |
| 32 | !todo | L1387 | §9. HBM data-dependence | \| 1800        \| 942 (throttled, TDP cap hit, clock dropped) \| ~415          \| ~527    \| At 1500... | > [!todo] this is really good data that could be highlighted in previous sections too, but you should capture the video clock & vo... |
| 33 | !fail | L1402 | §9. HBM data-dependence | ``` This is documented in user memory `project_b300_power_data_dep.md`. Higher clocks (1700/1800) hi... | > [!fail] Note this is just max power for DRAM; it doesn't use the ALUs/Tensor Cores/etc. much or at all! so the 'real' peak patho... |
| 34 | !fail | L1414 | §9. HBM data-dependence | - BW: 7.30 TB/s ± noise across all 11 patterns In other words: the chip uses **more power** to deliv... | > [!fail] for write-only specifically, given 32B/SM/clk with heavy enough throttling it might in theory hurt performance and there... |
| 35 | !todo | L1420 | §9. HBM data-dependence | ### Why this matters for ML inference Real production weight tensors (FP16/BF16/INT8/FP8) tend to ha... | > [!todo] would be worth getting real-world popcount distributions from real AI workloads (inference and pre-training-mid-run) and... |
| 36 | !fail | L1428 | §9. HBM data-dependence | - 1100 W only with deliberate stress recipes; rare in production For ML practitioners: choose **boos... | > [!fail] way too specific and highly unlikely to be true to the level of detail, please don't write these kinds of guesses as if ... |
| 37 | !fail | L1434 | §9. HBM data-dependence | ### Toggle-energy model (theory) The mechanism is **bus-toggle (Hamming-distance) energy** on the HB... | > [!fail] as per past NVIDIA papers, for the DRAM part specifically, I think it's more about popcount within a burst/chunk (with D... |
| 38 | !todo | L1456 | §9. HBM data-dependence | **Why d=16 maximizes**: Random-position popcount-16 means each 32-bit word has 16 ones in random pos... | > [!todo] I think it's plausible there might be special-casing for 'all 0' in terms of control overhead and/or clock gating in som... |
| 39 | !fail | L1494 | §9. HBM data-dependence | ``` If you see clock dropping during the stress run, the chip is throttling at TDP wall — back off c... | > [!fail] "1700MHz is the sweet spot" is wayyyy too specific, and I have seen it throttle below that for worst case power tests |
| 40 | !fail | L1519 | §10. L1 cache | ## §10. L1 cache **Answer:** 256 KB unified L1+SHMEM pool per SM; carveout 0..228 KB user-allocatabl... | > [!fail] 46TB/s should be impossible? |
| 41 | !fail | L1555 | §10. L1 cache | \| Strided pointer-chase, 4 KB stride (one line per 4 KB region) \| **~128 KB ≈ 1024 lines**, sharp ... | > [!fail] ~2-4KB for fisher-yates feels too low, it should be lower ofc, but this is a crazy ratio |
| 42 | !fail | L1574 | §10. L1 cache | \| L1 → L2 transition          \| 130–200 cy warm                   \| `03_caches.md`               ... | > [!fail] No way L1 latency varies by clock speed or that kind of access pattern if single thread & warm cache, something is wrong... |
| 43 | !fail | L1593 | §10. L1 cache | \| L1 aggregate (M5 cheatsheet, optimistic)     \| ~46 TB/s       \| `M5_MEMORY_CHEATSHEET.md` \| Sp... | > [!fail] "L1+register tag-overlap and is the LSU/L1-dispatch ceiling" is misleading word salad, pretty sure this is wrong as well |
| 44 | !fail | L1641 | §10. L1 cache | \| `.cs` / `.lu`   \| 3.4 TB/s   \| similar to `.cg`      \| similar to `.cg` \| `03_caches.md` \| \... | > [!fail] "DRAM-bound" is nonsense because it's not a fully optimized kernel setup and therefore everything else is probably not 1... |
| 45 | !fail | L1684 | §11. L2 cache — three different bandwidths | \| Full sector (32 B aligned)            \| 0× read amp                          \| D3 modes 3/4    ... | > [!fail] not clear if half-sector write only amplifies write or also does read-modify-write with extra read? key unanswered quest... |
| 46 | !fail | L1701 | §11. L2 cache — three different bandwidths | \| **L2 BW @ `.ca`, WS ≤ 1 MB (L1-amplified)**  \| 30–36 TB/s     \| actually LSU/L1-dispatch ceilin... | > [!fail] 13.3TB/s feels low for real traffic, I think the '.cg' data implies we can get >20TB/s, this is all confusing contradict... |
| 47 | !fail | L1733 | §11. L2 cache — three different bandwidths | \| L2 hit (far partition)  \| ~660 cy                                \| `03_caches.md`              ... | > [!fail] like... 300 or 228 cy? how is the other one "not chained", what does that even mean for a latency test, sigh |
| 48 | !fail | L1758 | §11. L2 cache — three different bandwidths | ### L2 video clock (HIGH) L2 / XBAR sits in its own clock domain at **1860 MHz**, **constant**, and ... | > [!fail] completely false, it is a DIFFERENT clock that is *correlated* to the main/graphics clock, not linear, you *CANNOT* assu... |
| 49 | !fail | L1855 | §11. L2 cache — three different bandwidths | - The "kernel-effective" number (23.85 TB/s, includes L1 reuse) is more representative of what real ... | > [!fail] hard disagree on all of your opinions in this section - what really matters is a lot more complicated/subtle and workloa... |
| 50 | !fail | L1883 | §11. L2 cache — three different bandwidths | The 2 L2 partitions (sides) are address-hashed. Each SM has a "near" partition and a "far" partition... | > [!fail] that's... not what this does at all? it's indirectly more likely to stay in near partition because it's more likely to p... |
| 51 | !fail | L1910 | §11. L2 cache — three different bandwidths | \| Max persisting (AccessPolicyWindow) \| 79.1 MB = 62.5 % \| Hardware cap                    \| \| ... | > [!fail] not persistent != streaming, regular default caching behaviour is in-between persistent and streaming hints |
| 52 | !fail | L1931 | §11. L2 cache — three different bandwidths | For `cp.async` (LDGSTS): `prefetch.L2` 1 cache-line ahead of the load is the canonical pattern. For ... | > [!fail] no absolute numbers, did you go from 10% efficiency to 16%, or from 60% to 95%? probably the former & very misleading |
| 53 | !fail | L1944 | §11. L2 cache — three different bandwidths | ``` Hit rate <50 % for a 100 MB hot working set is a sign of eviction pressure. Reduce WS or use per... | > [!fail] partitioned L2 means the "real usable size" is less than full L2 capacity, but offset by reordering making the cache hit... |
| 54 | !fail | L1958 | §12. Shared memory | ## §12. Shared memory **Answer:** **38.4 TB/s peak = 99.8 % of 38.5 TB/s theoretical** (32 banks × 4... | > [!fail] 32-way 4-byte bank conflict = 32x cost *DOES* hold for shared memory, so you clearly just did something wrong... |
| 55 | !fail | L2017 | §12. Shared memory | **Headline SoL: 38.4 TB/s = 99.8 %** (`02_shmem.md` and `B300_TRUE_REFERENCE.md` agree). **Realistic... | > [!fail] No reason for sustained number to be lower like that, no reason for read+write mix to be lower per-se (is it scalar?) - ... |
| 56 | !fail | L2035 | §12. Shared memory | **Inconsistency**: The catalog's `bce8bf8` 32-way = 8.81× slowdown is from a multi-warp throughput t... | > [!fail] Completely wrong, bank conflicts are real and expensive in terms of throughput. The latency cost is actually *less* bad ... |
| 57 | !fail | L2051 | §12. Shared memory | \| Aggregate INT atomic peak (all SMs, all-lanes-same-addr) \| **~2.2 Tatomic/s**     \| user memory... | > [!fail] confusing, does this mean shared memory is 128B read-or-write so atomic is 64B/clk basically (and atomicInc/Dec is a spe... |
| 58 | !fail | L2055 | §12. Shared memory | **Practical take**: use **INT atomics for SMEM histograms**, not FP32. The 67× cost gap between INT3... | > [!fail] just say FP32 is done via CAS... |
| 59 | !fail | L2108 | §12. Shared memory | If this metric is 0, no bank conflicts. If non-zero, the kernel has them; whether they hurt depends ... | > [!fail] ??? wrong. |
| 60 | !fail | L2114 | §12. Shared memory | ### SMEM persistence across CTAs SMEM is NOT shared across CTAs — each CTA gets its own private SMEM... | > [!fail] Verify whether SMEM is 0ed automatically between CTAs and security implications - I think it isn't, and data may leak be... |
| 61 | !fail | L2151 | §13. DSMEM (cluster shared memory) | ``` Four TPC pairs (x, x+1) spread across 4 GPCs. 100 % stable across launches (no scheduler randomn... | > [!fail] all those SMs are on the same GPC, that's true *BY DEFINITION* for DSMEM, the SM id does not reflect which GPC a SM is i... |
| 62 | !fail | L2171 | §13. DSMEM (cluster shared memory) | \| **Practical max cluster size**                         \| **8**            \| empirical (V11–V31)... | > [!fail] aka used the API wrong, 16 works if done correctly. |
| 63 | !fail | L2177 | §13. DSMEM (cluster shared memory) | ### SASS codegen nuance `ld.shared::cluster.u32` with **scalar-register address** compiles to `LD.E`... | > [!fail] .u32 here implies you probably used .u32 everywhere else, risk that might be a bottleneck, need to try 128-bit load/stor... |
| 64 | !fail | L2203 | §13. DSMEM (cluster shared memory) | Key observations: - **Cluster=2 is 21 % slower** than cluster ≥ 3 (single-GPC vs multi-GPC routing).... | > [!fail] super confusing, =2 should mean within TPC which should be FASTER, at >2 not all SMs are usable anymore. Are multiple cl... |
| 65 | !fail | L2209 | §13. DSMEM (cluster shared memory) | - Atomics inherit read-path asymmetry (return value → uses read path). - Self-read via `mapa` still ... | > [!fail] should show full SASS for self-read mapa case |
| 66 | !fail | L2234 | §13. DSMEM (cluster shared memory) | Self (diagonal):       54 cy  (mapa→me, NOT free vs 24 cy local LDS) ``` | > [!fail] super confusing, SM16<->17 being worse than 32<->33 despite both likely being same TPC is even more confusing. It's poss... |
| 67 | !fail | L2273 | §13. DSMEM (cluster shared memory) | \| 4 × 8       \| 5.08           \| 40.62                    \| **DSMEM read aggregate ceiling ≈ 40 ... | > [!fail] "non-chained ILP" - what?! by definition, that's not ILP, because chained means dependent means not parallel, unless tha... |
| 68 | !fail | L2290 | §13. DSMEM (cluster shared memory) | **DSMEM write aggregate ~560 GB/s per cluster** — but this is **issue rate, not completion**. V21 `p... | > [!fail] so you're not even fencing at the end? did you even check if DCE wasn't affecting this? this really should have been fai... |
| 69 | !fail | L2322 | §13. DSMEM (cluster shared memory) | - N=8: 14.0 GB/s aggregate (1.80× per-reader slowdown) **Per-CTA serving port caps at ≈ 15 GB/s** — ... | > [!fail] if true that would mean 8B/SM/clk which would be insanely low - what does "1.80x per-reader slowdown" even mean, given 2... |
| 70 | !fail | L2356 | §13. DSMEM (cluster shared memory) | \| FFMA compute     \| 0 % (210 vs 211 cy)   \| DSMEM competes for the peer's SMEM subsystem, not fo... | > [!fail] 30% way too specific, will vary a lot, unclear what SASS looks like and whether it's SMEM or LDS or scheduling pressure |
| 71 | !todo | L2376 | §13. DSMEM (cluster shared memory) | \| fence.sc.sys          \| 2870 (~9× slower) \| cluster / gpu **identical cost** → use `fence.sc.gp... | > [!todo] interesting - check SASS? what if there is traffic on the buses etc., is there any case where GPU gets slower but cluste... |
| 72 | !fail | L2388 | §13. DSMEM (cluster shared memory) | \| .gpu           \| 29.97                 \| \| .cluster       \| 31.40 (+1.4 cy / +5%) \| | > [!fail] confusing / possibly misleading |
| 73 | !fail | L2402 | §13. DSMEM (cluster shared memory) | \| 8-CTA ring all-reduce (V25)        \| 842 cy/step      \| 3.07 µs total \| with fence + barrier \... | > [!fail] should explain barrier vs fence, and how barrier.cluster is very brute force and only strictly required 1x at the start ... |
| 74 | !fail | L2417 | §13. DSMEM (cluster shared memory) | \| **v2.u64 (128-bit)** \| **29.66** \| **0.54** \| Use `v2.u64` for widest per-thread DSMEM store. | > [!fail] so literally the ONLY wide memory ops you tried were single-thread for latency (of a store!) and not testing throughput ... |
| 75 | !fail | L2552 | §14. NVLink-5 (Blackwell) | - **860 GB/s NVLink RX (ncu metric `nvlink__data_received`)** = bytes that crossed the link includin... | > [!fail] It's fascinating data but I think you are missing something much more simple and fundamental: reads are faster than writ... |
| 76 | !fail | L2575 | §14. NVLink-5 (Blackwell) | \| 64         \| 792             \| \| 148        \| 817             \| | > [!fail] 817 > 778 so 'saturated' isn't quitre the right word. |
| 77 | !fail | L2589 | §14. NVLink-5 (Blackwell) | - 1543 / 749 = 2.06× — close to perfect duplex **NVLink 5 is essentially full-duplex** (within 3 % o... | > [!fail] 778+720 is meant to mean... what? pretty sure you are doing something weird, is one GPU reading and the other writing wi... |
| 78 | !fail | L2601 | §14. NVLink-5 (Blackwell) | - Cross-GPU atomic latency = ~1.55 µs ≈ 3000 cy = **5× LOCAL** Cross-GPU atomics are *expensive* — f... | > [!fail] I trust the latency numbers but not the remote Gops/s, this needs a LOT more info to validate what it's doing. |
| 79 | !fail | L2623 | §14. NVLink-5 (Blackwell) | \| NCCL with NVLink-SHARP \| UNTESTED (no SHARP fabric on this NV18 system) \| NCCL's small-message ... | > [!fail] no, ~10us is not good, you just aren't using the best possible kernel for this, it's hard though, lots of ninja tricks m... |
| 80 | !fail | L2629 | §14. NVLink-5 (Blackwell) | ### Multi-GPU sharded GEMM `12_nvlink_p2p.md` finding: 0 % slowdown for multi-GPU sharded GEMM with ... | > [!fail] not credible at all. this might be true for very large GEMM sizes where one input is in remote memory etc... and/or if y... |
| 81 | !fail | L2635 | §14. NVLink-5 (Blackwell) | ### Peer-fence drain Cross-GPU `__threadfence_system` drains at +17.8 K cycles compared to single-GP... | > [!fail] as per previous analysis, 17.8K is when heavily contended, the real number when nothing else is happening on the system ... |
| 82 | !fail | L2681 | §14. NVLink-5 (Blackwell) | ### Stream-isolated NVLink When using multiple streams with cross-GPU memcpy, only ONE stream sees f... | > [!fail] this is nonsense, "connection-oriented per-stream"? what? completely wrong level of abstraction, *OBVIOUSLY* more CUDA s... |
| 83 | !todo | L2756 | §15. PCIe Gen6 x16 | Pageable: **38 GB/s = 66 % of pinned**. The CUDA runtime page-migrates pageable memory through a sta... | > [!todo] correct but shgould highlight high allocation time for pinned memory, i.e. very high init cost, so not worth it for shor... |
| 84 | !todo | L2766 | §15. PCIe Gen6 x16 | - Splitting across 4 streams gives **better latency** for small transfers (parallelism). - Use case:... | > [!todo] not verified that this is the case, it might be non-trivial to use them in parallel, not sure how this works |
| 85 | !fail | L2811 | §15. PCIe Gen6 x16 | \| `cuStreamCreate`                         \| <1 µs                                       \| Very f... | > [!fail] calling cudaStreamWriteValue a "hidden gem" is weird given 6-10us is actually pretty darn bad in my opinion - but then l... |
| 86 | !fail | L2837 | §15. PCIe Gen6 x16 | \| InfiniBand HDR               \| 25 GB/s          \| ~1 µs (with NIC)         \| Cluster networkin... | > [!fail] too authoritative sounding given how imprecise it is, doesn't mention L2 per-partition latency, etc... |
| 87 | !fail | L2938 | §15. PCIe Gen6 x16 | ``` This is faster than `cudaMalloc/cudaFree` for repeated alloc/free patterns. | > [!fail] CUDA VMM is the actual true ninja way of doing all this, you really should mention it explain it/test it... |
| 88 | !fail | L3127 | §16. FP32 FFMA peak — 74.62 TFLOPS at 2032 MH | ALL TFLOPS claims must annotate the clock state. The ~6% gap between 1920 and 2032 explains most of ... | > [!fail] ??? no reason for locked to be WORSE efficiency for given clock, 1920 lock should not throttle for this, so how is it on... |
| 89 | !fail | L3206 | §17. FFMA register-source dependence — 3-dist | The ratio `0.683 ≈ 2/3` exactly matches the prediction from a 2-RF-read-port model: 3 reads / 2 port... | > [!fail] this should really be measured with FFMA2 too, and look at SASS/reuse/different patterns/etc... but still good data over... |
| 90 | !fail | L3241 | §17. FFMA register-source dependence — 3-dist | empirical anchor for the 2-RF-read-port model. SASS-verified `.reuse` count: 255/256 in broadcast mo... | > [!fail] show SASS, is this still 3 unique input operands unlike fma a,b,a,b but with reuse? or? |
| 91 | !fail | L3254 | §17. FFMA register-source dependence — 3-dist | - Effective port count when one operand is hot: **3 reads/cy** - Effective port count when all 3 ope... | > [!fail] how reliably did you test the '1 entry' for reuse cache, how sure are you it's not more? Also what about forwarding / wr... |
| 92 | !fail | L3275 | §17. FFMA register-source dependence — 3-dist | **the realistic FP32 ceiling is ~51 TFLOPS, NOT 75**. This is the single most important number to co... | > [!fail] Did you confirm vector dot product is really 67%, could forwarding/write caching help in some way, what if you have e.g.... |
| 93 | !fail | L3285 | §17. FFMA register-source dependence — 3-dist | is observable only on FFMA (and similar high-throughput compute) where the pipe itself is fast enoug... | > [!fail] and when multiple pipes are used in parallel, which they typically are, so... the other even more interesting case here ... |
| 94 | !todo | L3363 | §18. FFMA `.reuse` cache — the SASS-level ope | - Cycle N: read Rd, b, Rd → 3 reads / 2 ports = 1.5 cy - Cycle N+1: same → 1.5 cy | > [!todo] it would be semi-interesting to try to figure out if the hardware reads Rd twice here (and whether it reads it twice for... |
| 95 | !fail | L3464 | §19. FADD = FMUL = FFMA at SASS level | the same 4.22 cy latency and same ~97.65% pipe saturation rate. FFMA "wins" purely because each inst... | > [!fail] 4.0 cycles latency - if 4.22, that means your (boost) clock is wrong, or your loop overhead is bad, or... |
| 96 | !fail | L3545 | §19. FADD = FMUL = FFMA at SASS level | throughput from the same inst rate. To guarantee FFMA emission, use explicit `asm("fma.rn.f32 %0, %1... | > [!fail] is this really a risk when fast math is enabled? I am skeptical any modern compiler would get this wrong except maybe wi... |

---

## Detailed Index


### 1. Line 309 — !FAIL

**Section:** §1. How to use this document

**Catalog claim being reviewed:**

```
§22 (Agent B) covers the FMA + ALU dual-issue verdict in depth. For context: the architectural verdict has flipped 5 times across the wave-1..wave-6 audit (HIGH → LOW → MED → LOW → HIGH). V52's empirical ncu measurement (`pipe_alu = 98.0%` AND `pipe_fma = 49.4%` simultaneously, sum = 147%) settled it in favor of "pipes overlap freely". The historical 55%/74% wall-clock measurements from V49/V50 are CONFIRMED-but-RETRACTED-as-architectural-claims (the numbers are what they are; the inference of a "shared dispatch cap" was wrong). Agent F's appendix §65 has the full zigzag case study.
The hardware-and-memory sections (§1–§15) are NOT directly affected by the dual-issue settlement, but the cross-cutting methodology lessons (Rule 13: "wall-clock GLane/s ratios are NOT decisive — they confound dispatch with per-instruction issue cadence") apply throughout.
```

**User callout (verbatim):**

```
> [!fail]  dual-issue based on pipes dubious, depends how pipes are defined in ncu, to validate 'useful work' GOps/s as well, the fact this isn't clearly highlighted here as part of the methodology is worrying - it is critical that all of these approaches are always checked and we can only be confident if they fully agree with each other.
```


### 2. Line 438 — !FAIL

**Section:** §2. B300 SXM6 AC at a glance

**Catalog claim being reviewed:**

```
- TDP cap (this AC SKU: 1100 W)
- Clock policy
```

**User callout (verbatim):**

```
> [!fail] 288GB or 288GiB, and are you sure it's actually 288 on this SKU, or 288 is for the 'full chip'? because if there's redundancy at the HBM controller level of 1/16, this will definitely affect total memory. It is likely that in theory, 288 is actually GiB (raw - some may be reserved by driver etc. and not visible), but this SKU is 15/16th of that, which coincidentally is close to 288GB creating the confusion
```


### 3. Line 500 — !FAIL

**Section:** §3. Clock frequencies

**Catalog claim being reviewed:**

```
## §3. Clock frequencies
**Answer:** Boost clock is 2032 MHz (rarely sustained under load); typical sustained boost is 1920 MHz; `nvidia-smi -lgc 2032` paradoxically pins to 1920 (NOT 2032); the GPU can stick at 1005 MHz silently under no-lock; voltage scales with clock² for power purposes.  `[🟢 HIGH · src: CLAUDE.md §2 + project memory feedback_clock_lock_works.md + project_b300_v6_complete.md]`
```

**User callout (verbatim):**

```
> [!fail] I strongly suspect the 1005MHz lock is the result of another agent or something else; I doubt it was 'auto-stuck' in any meaningful sense, if so this is very misleading
```


### 4. Line 521 — !TODO

**Section:** §3. Clock frequencies

**Catalog claim being reviewed:**

```
### Why the lock paradox
`nvidia-smi -lgc 2032` (or `-lgc 2032,2032`) pins the SM clock to 2032 MHz nominal — but the actual delivered clock under DVS is the **base** clock at that lock point, which is 1920 MHz on B300. To actually reach 2032 MHz boost you must NOT lock, and rely on driver DVS to opportunistically boost. There is no documented user-facing way to lock to 2032 MHz delivered. This is a 6 % gap that has caused confusion across the catalog (TFLOPS numbers stated at "locked 2032" are actually at 1920 MHz delivered).
```

**User callout (verbatim):**

```
> [!todo] show FFMA throughput with zero inputs (low toggle/power) at different locked clocks and unlocked clocks to highlight this here - otherwise hard to be confident this is not a measurement error
```


### 5. Line 539 — !FAIL

**Section:** §3. Clock frequencies

**Catalog claim being reviewed:**

```
| 2032        | 1050          | 2.25×       |
So power for the same kernel scales approximately as `(clock/510) × (V(clock)/700)²` — at 2032 MHz a kernel can draw 8–10× the power of the same kernel at 510 MHz. Combined with data-dependent toggle activity (see §9), the effective power range across realistic clock + data combinations is the full 200–1100 W TDP envelope.
```

**User callout (verbatim):**

```
> [!fail] GPU power does *not* scale in such a simple way, this is only a very very rough 1st approximation, this is misleading + you could show actual measure voltage here from nvidia-smi query, while highlighting it may vary per SKU etc.
```


### 6. Line 549 — !FAIL

**Section:** §3. Clock frequencies

**Catalog claim being reviewed:**

```
Recovery: `nvidia-smi -rgc` (reset to default) followed by waiting ~5 seconds, OR explicit `nvidia-smi -lgc 1920` (the "honest" boost lock).
This is documented in user memory `feedback_clock_stuck_no_lock.md`. If a measurement looks 2× too slow vs prior runs, this is the first thing to check.
```

**User callout (verbatim):**

```
> [!fail] likely misleading see above
```


### 7. Line 564 — !FAIL

**Section:** §3. Clock frequencies

**Catalog claim being reviewed:**

```
- **"At locked 1920 MHz"** if `-lgc <anything>` was used.
- **"At locked MHz"** for specific clock sweeps (power studies, V² extraction).
```

**User callout (verbatim):**

```
> [!fail] 1920MHz cannot be sustained for heavy tensor core workloads or even many other things; you are careful about this in some places by either having enough idle time or reducing clocks, but I am not 100% confident throttling was NEVER a factor
```


### 8. Line 583 — !FAIL

**Section:** §3. Clock frequencies

**Catalog claim being reviewed:**

```
| NVLink SerDes               | 53.125 GB/s/dir/lane raw (per NVLink-5 spec)   | NO                 |                                                      |
| PCIe SerDes                 | 64 GT/s nominal, 32 GT/s effective on this rig | NO                 | See §15 PHY-vs-effective                             |
```

**User callout (verbatim):**

```
> [!fail] L2/XBAR *is* affected by -lgc but only indirectly, there is a not-1:1 relationship, I think for 'middle clocks' it's roughly linear, but video clocks cannot go too high or too low (possibly due to kinds of SRAM cells used amongst other things? there might be many reasons, unclear if it runs at the same voltage or not, assuming it likely does)
```


### 9. Line 595 — !FAIL

**Section:** §3. Clock frequencies

**Catalog claim being reviewed:**

```
- **Uncombined / scattered atomics** are L2/DRAM-bound; throughput ∝ L2 video clock (1860 MHz constant).
- **HBM bandwidth** is HBM PHY-bound; per-pin rate independent of SM clock. SM clock affects only the launch-address generation rate, which is rarely the bottleneck for DRAM-saturated kernels.
```

**User callout (verbatim):**

```
> [!fail] 1860MHz is not constant so these are very worrying assumptions
```


### 10. Line 655 — !FAIL

**Section:** §4. HBM3E topology — 8 stacks (NOT 12)

**Catalog claim being reviewed:**

```
## §4. HBM3E topology — 8 stacks (NOT 12)
**Answer:** B300 has **8 × HBM3E 12-Hi stacks** (3 GB die), 16 × 512-bit controllers giving 8192-bit architectural bus. On this AC SKU one controller is fused off → 7680-bit effective bus, 275040 MiB visible.  `[🟢 HIGH · src: corrections/HBM_STACKS_INDEPENDENT_VERIFY.md + corrections/HBM_DENOMINATOR_FINAL.md]`
```

**User callout (verbatim):**

```
> [!fail] assuming memory capacity also scales by 15/16, unclear if 288 is meant to be 288GiB but for full SKU and this is ~288GB because of 15/16 by ~coincidence? industry is REALLY bad at GB vs GiB :( often, capacity is in GiB and bandwidth is in GB/s.....
```


### 11. Line 710 — !FAIL

**Section:** §4. HBM3E topology — 8 stacks (NOT 12)

**Catalog claim being reviewed:**

```
| 12 stacks × 12-Hi × 2 GB/die | 288 GB | 270 GB ≈ 268.6 GiB |
This is why the bus-width derivation (above) is the load-bearing argument, not capacity. Reported `totalGlobalMem = 274113 MiB = 268.08 GiB` plus reserved-memory adds up to the 268.59 GiB visible from `nvidia-smi` (small rounding/reservation difference between the two reports).
```

**User callout (verbatim):**

```
> [!fail] very much doubt it's ECC related but not impossible - I think it's included for 'free' in HBM? also 270GB is ~251GiB so that part is definitely wrong
```


### 12. Line 794 — !FAIL

**Section:** §4. HBM3E topology — 8 stacks (NOT 12)

**Catalog claim being reviewed:**

```
- Cross-stack hashing (default for `cudaMalloc`) distributes load across all 8 stacks.
- D2D copies between stack-locality-controlled src/dst can hit 6.93 TB/s (NINJA recipe, §6) by avoiding direction-switch penalties on shared stacks.
```

**User callout (verbatim):**

```
> [!fail] "cross-stack hashing" is literally impossible to turn off and hash is quite complicated - this is very misleading. I am extremely skeptical of your stack-locality-controlled recipe, it is possible to do this for L2, but in my experience helps power more than perf, and per-stack is harder and very unlikely to help perf at all, since we are not latency limited it's not the bottleneck
```


### 13. Line 842 — !FAIL

**Section:** §5. HBM3E denominators

**Catalog claim being reviewed:**

```
## §5. HBM3E denominators
**Answer:** Three valid denominators for "% of HBM peak" claims, each correct in its framing: **7680 GB/s** (spec, cross-vendor), **7672 GB/s** (this-device-actual at 3996 MHz I/O), or **7670 GB/s** (effective on this 7680-bit AC SKU). The catalog historically ALSO used **7.31 TB/s** (empirical pure-direction) as a denominator — that one is WRONG and inflates % numbers by ~5pp.  `[🟢 HIGH · src: corrections/HBM_DENOMINATOR_FINAL.md + corrections/01_hbm_bandwidth_CORRECTED.md §0]`
```

**User callout (verbatim):**

```
> [!fail] "empirical" is still useful as a 'real-world speed of light' for all other kernels; if a memcpy/memset/reduction cannot reach more than 7.3TB/s, there is no way e.g. a RMSNorm would, so it's a useful practical SOL for all other kernels to figure out how much there is left on the table
```


### 14. Line 858 — !FAIL

**Section:** §5. HBM3E denominators

**Catalog claim being reviewed:**

```
| **Architectural raw**     | 8192 GB/s              | spec pre-ECC at 8.000 Gbps                                    | Rare; only when measurement excludes ECC parity (ncu does not).                                                                         |
| **This-device raw**       | 8183.8 GB/s            | empirical pre-ECC at 7.992 Gbps                               | Symmetric to 7672 on the raw side.                                                                                                      |
```

**User callout (verbatim):**

```
> [!fail] what? the 7680-bit bus is not necessarily related to ECC, I think this is nonsense.
```


### 15. Line 925 — !FAIL

**Section:** §6. HBM read peak

**Catalog claim being reviewed:**

```
## §6. HBM read peak
**Answer:** **7.30–7.37 TB/s = 95.2–96.0% of 7680 GB/s spec**, achievable via either LDG.E.128 + per-warp coalesced (7.37 TB/s, the SoL) OR TMA `cp.async.bulk` 8 KB chunks (7.34 TB/s) OR v8 + per-warp coalesced + non-persistent (7.30 TB/s NINJA recipe). V46 8-deep TMA pipelined reaches 7.20 TB/s = 93.8% (BELOW the SoL — see footgun).  `[🟢 HIGH · src: corrections/01_hbm_bandwidth_CORRECTED.md §1+§2 + corrections/V46_DOUBT_REPORT.md]`
```

**User callout (verbatim):**

```
> [!fail] at one point we definitely had a microbenchmark loading 256 byte per thread by using a stride of 256B for 4B reads and the .256B L2 cache modifier, I think this was the actual highest DRAM BW % we saw in any microbenchmark and was validated by ncu, the fact this isn't included here is very worrying that you lost track of some things because you did so many different tests over so many iterations :( worth trying to find it again / experimenting with it again...
```


### 16. Line 994 — !FAIL

**Section:** §6. HBM read peak

**Catalog claim being reviewed:**

```
| 4 GB       | 7.29–7.30           | DRAM-bound, true HBM3E ceiling     |
| 32 GB      | ~7.20               | DRAM-bound (refresh-rate ceiling?) |
```

**User callout (verbatim):**

```
> [!fail] explain this better and show ncu stats, I am 99% sure the issue is that we get more L2 cache hits than you'd naively expect partly due to reordering in the memory subsystem and/or some SMs getting ahead of each other due to cache hits getting back sooner implicitly etc... and your numbers here do not use 'no L1 cache' despite being mentioned above, which I suspect affects the 16MB and 64MB numbers for the same reason - although for 64MB vs 100MB that *MIGHT* also be due to the "dual die" / partitioned L2 etc...! 7.20 vs 7.30 at 4GB vs 32GB is interesting and might deserve more analysis, it could be some kind of 'tail effect' if this is persistent (but I think it's not?), some kind of very slight thermal throttling, or maybe more likely MMU/TLB misses hurting latency? hard to say without more data
```


### 17. Line 1008 — !FAIL

**Section:** §6. HBM read peak

**Catalog claim being reviewed:**

```
The TMA pipelining lesson is real (an architectural best-practice for TMA users); the SoL claim is not.
```

**User callout (verbatim):**

```
> [!fail] this secftion does not tell me what the number of bytes per TMA instruction is, so this is not very informative and quite misleading I think
```


### 18. Line 1016 — !FAIL

**Section:** §6. HBM read peak

**Catalog claim being reviewed:**

```
`prefetch.L2` combined with `cp.async.bulk` is **27 % slower** than no-prefetch. TMA has its own DMA path; explicit prefetch instructions block forward progress. **Rule: never combine `prefetch.L2` with `cp.async.bulk`.**
(Note: the V6 1.58× prefetch speedup applies to **legacy `cp.async`** (LDGSTS), NOT to `cp.async.bulk` / TMA. Both observations are correct in their respective regimes.)
```

**User callout (verbatim):**

```
> [!fail] "block" seems like a very strong assumption without enough evidence? how fast is the cp.async case - is it just maybe it's beneficial for low efficiency cases to get slightly less bad efficiency, but hurts when already close to peak due to extra memory system traffic etc.? also need to be careful of how this plays into memory reordering affecting L1/L2 hit rates for what should be a DRAM benchmark, see above
```


### 19. Line 1029 — !FAIL

**Section:** §6. HBM read peak

**Catalog claim being reviewed:**

```
| V48 attempt to pipeline multicast | 13.96 TB/s (CAPPED) | V48    |
**Multicast cannot be pipelined** — single TMA engine per cluster. V32's 14.9 TB/s is the architectural multicast ceiling. See §13 for DSMEM/multicast detail.
```

**User callout (verbatim):**

```
> [!fail] this doesn't mean it cannot be pipelined - just that we are hitting maximum throughput with the amount of latency tolerance we already have without pipelining for this configuration. You are not describing the number of bytes per TMA instruction (same mistake as above), and you should also test with different number of active SMs etc...
```


### 20. Line 1042 — !FAIL

**Section:** §6. HBM read peak

**Catalog claim being reviewed:**

```
- ECC parity write-back cycles for partial writes (not applicable for pure reads)
`01_hbm_bandwidth.md` A2 noted bursts <1 KB hit 98.6 % of theoretical, while longer bursts under-saturate due to row-conflict scheduling. The 5 % gap is real silicon overhead, not a measurement artifact.
```

**User callout (verbatim):**

```
> [!fail] refresh cycles or other similar behaviour is interesting in its own right, I have definitely seen that myself previously where some accesses would randomly see much higher latency without any MMU misses and it happened at the same time for a % of total accesses in that period, if I remember correctly (not sure which GPU/memory technology), so even if it doesn't explain read bandwidth, it would still be worth analyzing more carefully
```


### 21. Line 1063 — !FAIL

**Section:** §6. HBM read peak

**Catalog claim being reviewed:**

```
- `.256` — 256-bit width (8× 32-bit lanes per thread)
The `.ENL2` is interesting: this is the SASS encoding for a load that bypasses L1 to reduce L1 pressure on a DRAM-bound kernel. The runtime compiler emits this when `cudaMallocManaged` or `cudaMallocAsync` are involved; for plain `cudaMalloc` without policy hints, you get `LDG.E.STRONG.SM` (L1+L2 cached). Both reach 7.30 TB/s for DRAM-bound work because L1 is irrelevant when WS >> L1.
```

**User callout (verbatim):**

```
> [!fail] Are you sure ENL2 means what you think it means? I suspect it might not actually bypass L1. And it makes *ZERO* sense that ptxas would compile differently based on which cudaMalloc function you use, this is a complete hallucination, LDG.E.STRONG.SM would be for less-than-256-bit loads I think?
```


### 22. Line 1084 — !FAIL

**Section:** §6. HBM read peak

**Catalog claim being reviewed:**

```
```
The `dram__bytes_read.sum` is the most authoritative metric: it counts bytes that left HBM controllers, divided by elapsed time. Use this as the "ground truth" denominator for HBM bandwidth claims.
```

**User callout (verbatim):**

```
> [!fail] Note this is also a good confirmation of the ~7.68TB/s DRAM bandwidth peak, could have mentioned this as strong evidence above without speculating for several pages
```


### 23. Line 1102 — !FAIL

**Section:** §6. HBM read peak

**Catalog claim being reviewed:**

```
| Pipelining        | TMA 8-deep recovers within-TMA gap                                      | TMA single-deep is fine                   |
The lesson for kernel writers: make stores **256-bit aligned** (`v8` / `int4` / 4× int4 etc.) AND coalesced per-warp into 1 KB bursts. Both required for SoL.
```

**User callout (verbatim):**

```
> [!fail] Pretty sure you can get *VERY* close to SOL with 64-bit or 128-bit aligned if you have enough memory level parallelism per thread with the right access patterns (but 256-bit is definitely best so might as well use it on Blackwell whenever possible! but depending on what data naturally is required inside 1 thread vs 1 warp, 256-bit might not always be best, the most important is coalescing/making addresses consecutive etc. - it matters less at 256-bit, so non-consecutive 256-bit addresses for threads in the same warp is wayyyy better than even 128-bit, but still not going to get you to SOL)
```


### 24. Line 1114 — !FAIL

**Section:** §7. HBM write peak

**Catalog claim being reviewed:**

```
## §7. HBM write peak
**Answer:** **7.30 TB/s = 95.2% of 7680 spec** for the standard v8 STG NINJA recipe; **7.57 TB/s = 98.7%** is the contested write SoL with disputed provenance (NINJA STG vs TMA bulk store).  `[🟡 MED · src: corrections/01_hbm_bandwidth_CORRECTED.md §3 + V8_HBM_WRITE_SOL.md]`
```

**User callout (verbatim):**

```
> [!fail] 7.57TB/s not confirmed by NCU, you should have TRULY gotten to the bottom of this and not been so unclear throughout this section on whether it's real or not (it might be, but I am not fully convinced)
```


### 25. Line 1132 — !FAIL

**Section:** §7. HBM write peak

**Catalog claim being reviewed:**

```
| D2D NINJA (separate src/dst)                          | 6.93     | 90.3%                 | `4958d6b`                                                                                                         |
| D2D `cudaMemcpyAsync`                                 | 6.56     | 85.5%                 | "single-direction 3.28 × 2"                                                                                       |
```

**User callout (verbatim):**

```
> [!fail] You should have properly tested this at different clock speeds, because the SM->L2 *write* path is limited to 32B/clk (read is much higher), you *cannot* get to peak HBM write bandwidth at lower clocks, and even 1920MHz vs 2032MHz might have had an effect. So all of this is potentially misleading/confusing and highlights the unfortunate lack of rigor with regards to clock speeds at certain points in the microbenchmarking process (especially early on which is understandable, but not fully corrected later)
```


### 26. Line 1225 — !FAIL

**Section:** §7. HBM write peak

**Catalog claim being reviewed:**

```
`cudaMemset` invokes a built-in driver kernel that's optimized for B300. Wall-clock timing shows 7.47–7.52 TB/s effective rate. ncu shows ~7.30 TB/s actual `dram__bytes_write.sum`/time. The 0.2 TB/s discrepancy is a measurement-window-end artifact (last DMA completes after the timer stop captures elapsed time).
For benchmarking purposes, use the ncu number (7.30 TB/s). For wall-clock measurements where you don't have ncu, the 7.5 TB/s is approximately right.
```

**User callout (verbatim):**

```
> [!fail] if this is a measurement-window-end artifact it should vary based on the problem size / number of total bytes, does it? one possible complication is I am not sure whether ncu ever boosts to 2032MHz or not when unlocked, or if it stays at 1920MHz even if you ask for it not to control clocks (you did do that... right? otherwise it's way slower than that), worth testing - once again clocks affect writes a lot more than reads
```


### 27. Line 1239 — !FAIL

**Section:** §7. HBM write peak

**Catalog claim being reviewed:**

```
**Footgun (separate):** ⚠ Don't quote "writes exceed reads by 5 %" — that was a denominator-mismatch artifact (used 7.2 effective for read, 8.0 nominal for write). True asymmetry is ≤3 percentage points either way.
**See also:** §6 (read peak ladder), §8 (R+W concurrent contention), §10 (cudaMemset's wall-clock vs ncu gap).
```

**User callout (verbatim):**

```
> [!fail] since cudaMemset is also using SMs/CTAs, just not in a way we can control, it might also be harder to *reliably* overlap other kernels with the cudaMemset versus overlapping it with our own memset kernel
```


### 28. Line 1249 — !TODO

**Section:** §8. HBM concurrent R+W

**Catalog claim being reviewed:**

```
**Answer:** **7.31 TB/s pure-direction ceiling**, **6.68 TB/s** at the 50:50 minimum (-13 % from balanced contention U-curve), D2D copy hits **6.93 TB/s** with the NINJA recipe and **6.56 TB/s** via `cudaMemcpyAsync`.  `[🟢 HIGH · src: corrections/01_hbm_bandwidth_CORRECTED.md §6 + §8]`
HBM3E on B300 is **shared-bus, not full-duplex** — the controllers serve reads and writes through a common bank pipeline, and direction-switches incur tWTR/tRTW penalties. Mixed R+W traces a U-shape with minimum at 50:50.
```

**User callout (verbatim):**

```
> [!todo] this is correct at a high level, not 100% sure the tWTR/tRTW aspect is the only one that matters, but either way worth highlighting that because writes always go via L2 on NVIDIA GPUs, it's extremely difficult to try to "temporally coalesce" reads separately from writes, since reads might evict dirty write data in the L2. All of this might also affect the latency of the reads in complicated ways (which I have personally looked at for LPDDR5 but not HBM, so I am not sure exactly how significant it is here)
```


### 29. Line 1320 — !FAIL

**Section:** §8. HBM concurrent R+W

**Catalog claim being reviewed:**

```
- **Temporal separation** (read all, then write all) — but this requires WS buffering in SMEM/L2.
The D2D NINJA recipe (6.93 TB/s) uses stack-locality to put src on stacks 0-3 and dst on stacks 4-7, getting near-pure-direction throughput for both halves of the copy.
```

**User callout (verbatim):**

```
> [!fail] see my todo above, I don't think this is possible in the way you are proposing, there might be other ways but they are even more complicated.
```


### 30. Line 1326 — !FAIL

**Section:** §8. HBM concurrent R+W

**Catalog claim being reviewed:**

```
### Why 50:50 is the worst case (and not 60:40 or 40:60)
Bank rotation happens at fixed cadence; direction-switch penalty per switch is constant. The MORE direction-switches per unit time, the lower the throughput. At 50:50, switches happen at maximum rate (every burst). At 60:40, the chip can batch the majority direction (60 %) without switching, only paying the penalty at the boundary.
```

**User callout (verbatim):**

```
> [!fail] memory controllers can be a lot more complicated with more dynamic heuristics than that, slightly misleading
```


### 31. Line 1369 — !FAIL

**Section:** §9. HBM data-dependence

**Catalog claim being reviewed:**

```
| 32 (all-one)                 | 380 + DBI | 415 + DBI | min + DBI penalty |
DBI = Data-Bus Inversion: HBM3E PHY can flip all 32 bits if it reduces toggle count. The "all-one" tier is slightly higher than "all-zero" because of active-low termination overhead; the +11 to +44 W asymmetry between d=0 and d=32 across cache-tier distance grows with HBM-distance (L1 +11.8 W, L2 +22.8 W, DRAM-1G +41.6 W, DRAM-8G +44.8 W) — reported consistently in the POPCOUNT family.
```

**User callout (verbatim):**

```
> [!fail] DRAM-1G W is probably just getting some L2 hits or something, or less warmup/cooldown time changing average, this is confusing and/or misleading
```


### 32. Line 1387 — !TODO

**Section:** §9. HBM data-dependence

**Catalog claim being reviewed:**

```
| 1800        | 942 (throttled, TDP cap hit, clock dropped) | ~415          | ~527    |
At 1500 MHz the chip can simultaneously push 7+ TB/s of DRAM bandwidth AND draw 921 W of memory-subsystem power. Adding compute simultaneously caps at TDP wall (~1100 W).
```

**User callout (verbatim):**

```
> [!todo] this is really good data that could be highlighted in previous sections too, but you should capture the video clock & voltage for those locked clocks via nvidia-smi query
```


### 33. Line 1402 — !FAIL

**Section:** §9. HBM data-dependence

**Catalog claim being reviewed:**

```
```
This is documented in user memory `project_b300_power_data_dep.md`. Higher clocks (1700/1800) hit the TDP wall and start throttling; 1500 MHz is the max sustainable stress point.
```

**User callout (verbatim):**

```
> [!fail] Note this is just max power for DRAM; it doesn't use the ALUs/Tensor Cores/etc. much or at all! so the 'real' peak pathological worst-case power is way more than that and would result in much worse throttling (the worst real-world case might be a bandwidth-heavy GEMM workload, but even that won't have such a high toggle rate so definitely not as bad as it could be)
```


### 34. Line 1414 — !FAIL

**Section:** §9. HBM data-dependence

**Catalog claim being reviewed:**

```
- BW: 7.30 TB/s ± noise across all 11 patterns
In other words: the chip uses **more power** to deliver the same bandwidth on random-toggle data, but it doesn't deliver less bandwidth. Cache-line traffic is fixed at 128 B units, bus signaling is at fixed rates, address decoding is deterministic per access. **Memory subsystem bandwidth is data-pattern independent within 1 %**.
```

**User callout (verbatim):**

```
> [!fail] for write-only specifically, given 32B/SM/clk with heavy enough throttling it might in theory hurt performance and therefore bandwidth, but not likely in practice
```


### 35. Line 1420 — !TODO

**Section:** §9. HBM data-dependence

**Catalog claim being reviewed:**

```
### Why this matters for ML inference
Real production weight tensors (FP16/BF16/INT8/FP8) tend to have popcount distributions that lean toward d=8..d=20 (not uniform random). The d=16 stress recipe is an upper bound on power for memory-bound operations. Practical inference workloads see:
```

**User callout (verbatim):**

```
> [!todo] would be worth getting real-world popcount distributions from real AI workloads (inference and pre-training-mid-run) and see how it differs in the real world for FP16 vs BF16 vs FP8 vs FP4 too(!)
```


### 36. Line 1428 — !FAIL

**Section:** §9. HBM data-dependence

**Catalog claim being reviewed:**

```
- 1100 W only with deliberate stress recipes; rare in production
For ML practitioners: choose **boost (2032 MHz)** for inference latency optimization (see user memory `project_b300_v6_complete.md` — 3× lower energy than 510 MHz). Don't try to save power by lowering clocks; energy-per-token gets worse.
```

**User callout (verbatim):**

```
> [!fail] way too specific and highly unlikely to be true to the level of detail, please don't write these kinds of guesses as if they were fact!
```


### 37. Line 1434 — !FAIL

**Section:** §9. HBM data-dependence

**Catalog claim being reviewed:**

```
### Toggle-energy model (theory)
The mechanism is **bus-toggle (Hamming-distance) energy** on the HBM3E I/O wires. Per-cycle energy is approximately:
```

**User callout (verbatim):**

```
> [!fail] as per past NVIDIA papers, for the DRAM part specifically, I think it's more about popcount within a burst/chunk (with Data-Bus Inversion making all-1 similar to all-0), which is subtly different from toggle rate elsewhere, but of course here we are measuring DRAM *and* L2 *and* everything else power together, so it's even trickier
```


### 38. Line 1456 — !TODO

**Section:** §9. HBM data-dependence

**Catalog claim being reviewed:**

```
**Why d=16 maximizes**: Random-position popcount-16 means each 32-bit word has 16 ones in random positions. Across consecutive words, the probability that any wire toggles is highest at d=16 (binomial peak). At d=0 (all-zero) or d=32 (all-one), consecutive words are identical so toggle activity = 0 (Data-Bus Inversion can flip 32 → 0 active-low if needed, hence small DBI penalty).
**DBI mechanism**: HBM3E PHY can invert all 32 bits of a wire-group if doing so reduces total toggles. So "all-one" is effectively encoded as "all-zero with DBI flag set", costing slight extra control overhead but saving significant toggle energy. This is why d=32 is only 11–44 W higher than d=0, not 7680× higher (which the naive theory would predict).
```

**User callout (verbatim):**

```
> [!todo] I think it's plausible there might be special-casing for 'all 0' in terms of control overhead and/or clock gating in some parts of the GPU as well, I believe you found that to be the case for tcgen05's A matrix for example, where for B "many of the same values" was similar to "many 0" but for A it made a bigger difference if it was 0 vs any other value.
```


### 39. Line 1494 — !FAIL

**Section:** §9. HBM data-dependence

**Catalog claim being reviewed:**

```
```
If you see clock dropping during the stress run, the chip is throttling at TDP wall — back off clock by 100 MHz. The 1700 MHz clock with mixed compute+memory at random data is the "sweet spot" for hitting TDP cleanly without throttling.
```

**User callout (verbatim):**

```
> [!fail] "1700MHz is the sweet spot" is wayyyy too specific, and I have seen it throttle below that for worst case power tests
```


### 40. Line 1519 — !FAIL

**Section:** §10. L1 cache

**Catalog claim being reviewed:**

```
## §10. L1 cache
**Answer:** 256 KB unified L1+SHMEM pool per SM; carveout 0..228 KB user-allocatable; **effective L1 bandwidth ~30.5 TB/s typical, up to 46 TB/s small-WS** (M5 cheatsheet); sharp 128 KB transition at strided 4 KB stride access.  `[🟢 HIGH · src: corrections/03_caches_CORRECTED.md §1 + b300_clean/D2_L1_CAPACITY_RIGOR.md + b300_clean/V10_L1_CAPACITY.md]`
```

**User callout (verbatim):**

```
> [!fail] 46TB/s should be impossible?
```


### 41. Line 1555 — !FAIL

**Section:** §10. L1 cache

**Catalog claim being reviewed:**

```
| Strided pointer-chase, 4 KB stride (one line per 4 KB region) | **~128 KB ≈ 1024 lines**, sharp boundary     | `D2_L1_CAPACITY_RIGOR.md` |
| Random-access (Fisher-Yates chain, 128 B lines)               | **~2–4 KB** effective, smooth ramp 47→277 cy | `V10_L1_CAPACITY.md`      |
```

**User callout (verbatim):**

```
> [!fail] ~2-4KB for fisher-yates feels too low, it should be lower ofc, but this is a crazy ratio
```


### 42. Line 1574 — !FAIL

**Section:** §10. L1 cache

**Catalog claim being reviewed:**

```
| L1 → L2 transition          | 130–200 cy warm                   | `03_caches.md`                                                                  |
| `.ca` vs `.cg` at 8 KB WS   | 40 cy vs 552 cy = **13.8× ratio** | `03_caches.md`                                                                  |
```

**User callout (verbatim):**

```
> [!fail] No way L1 latency varies by clock speed or that kind of access pattern if single thread & warm cache, something is wrong here
```


### 43. Line 1593 — !FAIL

**Section:** §10. L1 cache

**Catalog claim being reviewed:**

```
| L1 aggregate (M5 cheatsheet, optimistic)     | ~46 TB/s       | `M5_MEMORY_CHEATSHEET.md` |
Spread reflects unrolling / ILP / launch geometry. **30.5 TB/s is the conservative measured peak** under the V8_L2 verification methodology; the M5 cheatsheet 46 TB/s is at L1+register tag-overlap and is the LSU/L1-dispatch ceiling — not strictly L1 throughput.
```

**User callout (verbatim):**

```
> [!fail] "L1+register tag-overlap and is the LSU/L1-dispatch ceiling" is misleading word salad, pretty sure this is wrong as well
```


### 44. Line 1641 — !FAIL

**Section:** §10. L1 cache

**Catalog claim being reviewed:**

```
| `.cs` / `.lu`   | 3.4 TB/s   | similar to `.cg`      | similar to `.cg` | `03_caches.md` |
| `__ldg` / `.nc` | 3.4 TB/s   | matches default       | matches default  | `03_caches.md` |
```

**User callout (verbatim):**

```
> [!fail] "DRAM-bound" is nonsense because it's not a fully optimized kernel setup and therefore everything else is probably not 100% trustworthy even if it is technically independent. sigh.
```


### 45. Line 1684 — !FAIL

**Section:** §11. L2 cache — three different bandwidths

**Catalog claim being reviewed:**

```
| Full sector (32 B aligned)            | 0× read amp                          | D3 modes 3/4            |
| Full line (128 B aligned)             | 0× read amp                          | D3 mode 5               |
```

**User callout (verbatim):**

```
> [!fail] not clear if half-sector write only amplifies write or also does read-modify-write with extra read? key unanswered question: for e.g. sub-sector writes, does it *immediately* do the DRAM read so it is always that extra cost, or if it's only when it gets evicted by something else needing the L2 line? (same question for half-sector writes depending on above) ==> how much do we need 32B aligned *within a warp* (/128B of coalescing) vs how much is it OK if it has good temporal locality with bad spatial locality?
```


### 46. Line 1701 — !FAIL

**Section:** §11. L2 cache — three different bandwidths

**Catalog claim being reviewed:**

```
| **L2 BW @ `.ca`, WS ≤ 1 MB (L1-amplified)**  | 30–36 TB/s     | actually LSU/L1-dispatch ceiling, not L2                                                        | `03_caches.md` §3c                                               |
| **L2 strided `.cg` 64 MB**                   | **13.85 TB/s** | matches the 13.30 ncu wire number                                                               | `V8_L2_BW_VERIFIED.md`                                           |
```

**User callout (verbatim):**

```
> [!fail] 13.3TB/s feels low for real traffic, I think the '.cg' data implies we can get >20TB/s, this is all confusing contradictory data that needs more work - if there is some other form of amplification for temporal locality that increases L2 bandwidth beyond the 'bus traffic' then that is also critical and would need to be explained. Need to use ncu on more test variants as a 1st step if not already in data you missed/forgot about. I think there's a small chance it's 13TB/s per side for 26TB/s total, but probably not, more likely you just didn't write/use an efficient test. Actually, now that I think about it, I don't understand how this metric even works given that an access to the far side will touch *BOTH* L2s, does that count as 1 or 2 in that counter? hmm...
```


### 47. Line 1733 — !FAIL

**Section:** §11. L2 cache — three different bandwidths

**Catalog claim being reviewed:**

```
| L2 hit (far partition)  | ~660 cy                                | `03_caches.md`                                               |
| Near vs far ratio       | **1.27–2.4×**                          | `B300_TRUE_REFERENCE.md` (commit `af91798`), M5 (1.27–1.85×) |
```

**User callout (verbatim):**

```
> [!fail] like... 300 or 228 cy? how is the other one "not chained", what does that even mean for a latency test, sigh
```


### 48. Line 1758 — !FAIL

**Section:** §11. L2 cache — three different bandwidths

**Catalog claim being reviewed:**

```
### L2 video clock (HIGH)
L2 / XBAR sits in its own clock domain at **1860 MHz**, **constant**, and not changed by `nvidia-smi -lgc`. Implications:
```

**User callout (verbatim):**

```
> [!fail] completely false, it is a DIFFERENT clock that is *correlated* to the main/graphics clock, not linear, you *CANNOT* assume constant 1860MHz which is the peak (although it changes less than graphics clock, it does scale with it to some extent).
```


### 49. Line 1855 — !FAIL

**Section:** §11. L2 cache — three different bandwidths

**Catalog claim being reviewed:**

```
- The "kernel-effective" number (23.85 TB/s, includes L1 reuse) is more representative of what real workloads see.
So when citing one number: **23.85 TB/s** for "what the kernel sees" or **13.30 TB/s** for "what the L2 wire delivers". The 17 TB/s is a particular point on the multi-dimensional surface, not a headline.
```

**User callout (verbatim):**

```
> [!fail] hard disagree on all of your opinions in this section - what really matters is a lot more complicated/subtle and workload dependent than that
```


### 50. Line 1883 — !FAIL

**Section:** §11. L2 cache — three different bandwidths

**Catalog claim being reviewed:**

```
The 2 L2 partitions (sides) are address-hashed. Each SM has a "near" partition and a "far" partition. The address hash flips at ~4 KB stride (see §11.2.1).
For latency-sensitive kernels, pin hot lines to the near partition by:
```

**User callout (verbatim):**

```
> [!fail] that's... not what this does at all? it's indirectly more likely to stay in near partition because it's more likely to persist, but this is super misleading
```


### 51. Line 1910 — !FAIL

**Section:** §11. L2 cache — three different bandwidths

**Catalog claim being reviewed:**

```
| Max persisting (AccessPolicyWindow) | 79.1 MB = 62.5 % | Hardware cap                    |
| Free for streaming                  | 47.4 MB minimum  | Even when persisting fully used |
```

**User callout (verbatim):**

```
> [!fail] not persistent != streaming, regular default caching behaviour is in-between persistent and streaming hints
```


### 52. Line 1931 — !FAIL

**Section:** §11. L2 cache — three different bandwidths

**Catalog claim being reviewed:**

```
For `cp.async` (LDGSTS): `prefetch.L2` 1 cache-line ahead of the load is the canonical pattern.
For `cp.async.bulk` (TMA): no prefetch; let the TMA engine manage its own DMA depth.
```

**User callout (verbatim):**

```
> [!fail] no absolute numbers, did you go from 10% efficiency to 16%, or from 60% to 95%? probably the former & very misleading
```


### 53. Line 1944 — !FAIL

**Section:** §11. L2 cache — three different bandwidths

**Catalog claim being reviewed:**

```
```
Hit rate <50 % for a 100 MB hot working set is a sign of eviction pressure. Reduce WS or use persisting window.
```

**User callout (verbatim):**

```
> [!fail] partitioned L2 means the "real usable size" is less than full L2 capacity, but offset by reordering making the cache hit above working set size in a way that pure single-threaded LRU would not at all, etc...
```


### 54. Line 1958 — !FAIL

**Section:** §12. Shared memory

**Catalog claim being reviewed:**

```
## §12. Shared memory
**Answer:** **38.4 TB/s peak = 99.8 % of 38.5 TB/s theoretical** (32 banks × 4 B × 2.032 GHz × 148 SMs). 228 KB max user-allocatable per CTA. stmatrix W+R chain hits 34.5 TB/s. SMEM atomic INT throughput ~2.2 Tatomic/s (no contention). Bank conflicts are regime-dependent: 2× in latency-bound, ~1× in throughput-bound (the "32-way conflict = 32× cost" textbook rule does NOT hold on B300).  `[🟢 HIGH · src: corrections/02_shmem_CORRECTED.md + b300_clean/02_shmem.md + b300_clean/V8_SMEM_BW.md]
```

**User callout (verbatim):**

```
> [!fail] 32-way 4-byte bank conflict = 32x cost *DOES* hold for shared memory, so you clearly just did something wrong...
```


### 55. Line 2017 — !FAIL

**Section:** §12. Shared memory

**Catalog claim being reviewed:**

```
**Headline SoL: 38.4 TB/s = 99.8 %** (`02_shmem.md` and `B300_TRUE_REFERENCE.md` agree).
**Realistic mixed-workload ceiling: 27.2 TB/s** for read+write tile work.
```

**User callout (verbatim):**

```
> [!fail] No reason for sustained number to be lower like that, no reason for read+write mix to be lower per-se (is it scalar?) - those parts are misleading
```


### 56. Line 2035 — !FAIL

**Section:** §12. Shared memory

**Catalog claim being reviewed:**

```
**Inconsistency**: The catalog's `bce8bf8` 32-way = 8.81× slowdown is from a multi-warp throughput test, NOT a latency test. This contradicts V45's "~1× hidden" claim under the same nominal regime. The discrepancy is **unresolved** — likely the V45 setup had enough other warps queued to hide the conflict, while `bce8bf8` was contention-saturated.
**Practical take**: the "32× textbook rule" is never observed on B300; the real cost ranges 1× to 8.8× depending on warp count and latency-tolerance of the loop. For tile kernels with high TLP, bank conflicts are far less harmful than the textbook model predicts. For pure latency-sensitive kernels (rare in real workloads), the cost can be up to ~8×.
```

**User callout (verbatim):**

```
> [!fail] Completely wrong, bank conflicts are real and expensive in terms of throughput. The latency cost is actually *less* bad since it's "just" 1 cycle per conflict iirc, so extra 32 cycles max, which doesn't even get us to 8.2x, so that's not a latency test? all horribly flawed sadly
```


### 57. Line 2051 — !FAIL

**Section:** §12. Shared memory

**Catalog claim being reviewed:**

```
| Aggregate INT atomic peak (all SMs, all-lanes-same-addr) | **~2.2 Tatomic/s**     | user memory `project_b300_v8_complete.md` (commit `968e5b7`) |
**Note:** the user prompt said "4.2 Tops/s" but that doesn't match the catalog. The 2.2 Tatomic/s figure (`968e5b7`) is the verified value. The 4.2 Tops/s claim may have been mis-recalled or come from a different op (atomicInc/Dec are 4 ns vs add 8 ns = ~2× faster — could account for the discrepancy). Treat as **MED** until re-verified.
```

**User callout (verbatim):**

```
> [!fail] confusing, does this mean shared memory is 128B read-or-write so atomic is 64B/clk basically (and atomicInc/Dec is a special case)? Confusing you didn't write the INT width for 2.2 Tatomic/s, is that 32-bit or 64-bit?...
```


### 58. Line 2055 — !FAIL

**Section:** §12. Shared memory

**Catalog claim being reviewed:**

```
**Practical take**: use **INT atomics for SMEM histograms**, not FP32. The 67× cost gap between INT32 and FP32 contended atomics reflects FP32's read-modify-write being non-cacheable on the SMEM hardware atomic units.
```

**User callout (verbatim):**

```
> [!fail] just say FP32 is done via CAS...
```


### 59. Line 2108 — !FAIL

**Section:** §12. Shared memory

**Catalog claim being reviewed:**

```
If this metric is 0, no bank conflicts. If non-zero, the kernel has them; whether they hurt depends on whether the kernel is latency-bound or throughput-bound.
A useful rule-of-thumb test: "compare bank-conflicting tile to padded tile". If padded version is faster by >20 %, you're in the latency-bound regime and bank conflicts hurt. If <20 %, throughput-bound and conflicts are mostly hidden.
```

**User callout (verbatim):**

```
> [!fail] ??? wrong.
```


### 60. Line 2114 — !FAIL

**Section:** §12. Shared memory

**Catalog claim being reviewed:**

```
### SMEM persistence across CTAs
SMEM is NOT shared across CTAs — each CTA gets its own private SMEM allocation. To share data between CTAs in the same cluster, use DSMEM (§13). To share across clusters or across SMs not in the same cluster, you must go through L2 (slow) or HBM (slower).
```

**User callout (verbatim):**

```
> [!fail] Verify whether SMEM is 0ed automatically between CTAs and security implications - I think it isn't, and data may leak between CTAs of the same context?
```


### 61. Line 2151 — !FAIL

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
```
Four TPC pairs (x, x+1) spread across 4 GPCs. 100 % stable across launches (no scheduler randomness for cluster=8 dimensions). Use:
```

**User callout (verbatim):**

```
> [!fail] all those SMs are on the same GPC, that's true *BY DEFINITION* for DSMEM, the SM id does not reflect which GPC a SM is in.
```


### 62. Line 2171 — !FAIL

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
| **Practical max cluster size**                         | **8**            | empirical (V11–V31) |
Above cluster=8, scheduler spread becomes irregular and crash rates rise. Stick to cluster ≤ 8 for portable code. Memory note `project_b300_v5_complete.md`: "WGMMA dropped, cluster MAX=8".
```

**User callout (verbatim):**

```
> [!fail] aka used the API wrong, 16 works if done correctly.
```


### 63. Line 2177 — !FAIL

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
### SASS codegen nuance
`ld.shared::cluster.u32` with **scalar-register address** compiles to `LD.E` (global window through L2), NOT `LDS`. ncu shows ~4 L2 sectors/load. The `LDS R, [R+UR]` form only appears when the mapa result lands in a uniform register.
```

**User callout (verbatim):**

```
> [!fail] .u32 here implies you probably used .u32 everywhere else, risk that might be a bottleneck, need to try 128-bit load/store
```


### 64. Line 2203 — !FAIL

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
Key observations:
- **Cluster=2 is 21 % slower** than cluster ≥ 3 (single-GPC vs multi-GPC routing). For latency-sensitive cluster work, **prefer cluster ≥ 3** even if you only need 2 CTAs of capacity.
```

**User callout (verbatim):**

```
> [!fail] super confusing, =2 should mean within TPC which should be FASTER, at >2 not all SMs are usable anymore. Are multiple clusters active at the same time, or is this 1 cluster per GPU? I wonder if there might be either some contention issue, or weirdly (on this SKU) the 'first' 2-wide cluster is a 'slow' one in some strange way.
```


### 65. Line 2209 — !FAIL

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
- Atomics inherit read-path asymmetry (return value → uses read path).
- Self-read via `mapa` still pays LD.E cost (54 cy vs 24 cy local) — the address translation goes through the cluster routing fabric even when the destination is the same SM.
```

**User callout (verbatim):**

```
> [!fail] should show full SASS for self-read mapa case
```


### 66. Line 2234 — !FAIL

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
Self (diagonal):       54 cy  (mapa→me, NOT free vs 24 cy local LDS)
```
```

**User callout (verbatim):**

```
> [!fail] super confusing, SM16<->17 being worse than 32<->33 despite both likely being same TPC is even more confusing. It's possible but I don't 100% trust this... possibly contention, depending on how the test works? the results for all the other SMs makes sense where same TPC is ~165 to ~175 cycles and ~20 to ~30 cycles extra for all the other ones, that single outlier makes no sense, unless NVIDIA does "half TPC" SKUs where 1 of the 2 SMs is disabled for 2 TPCs in a GPC, and so X and X+1 are not the same TPC, which would be testable in other ways (e.g. instruction cache thrashing with very specialised kernels because ~32KiB "L1" cache is actually per TPC, not per SM) - actually I don't think this would work with the way tcgen05 can read SMEM from peer SM in the TPC for some inputs, unless it can go through this slower path fine somehow?! probably just a measurement error... but very interesting if not.
```


### 67. Line 2273 — !FAIL

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
| 4 × 8       | 5.08           | 40.62                    |
**DSMEM read aggregate ceiling ≈ 40 GB/s per cluster** — this is **chain-bound**, NOT a fabric ceiling. With non-chained ILP (addresses derived from `i` not from prior result), throughput is plausibly higher (60–80 GB/s estimated by `DSMEM_DOUBT_REPORT.md`). Treat 40 GB/s as a chain-bound lower bound, not the architectural ceiling.
```

**User callout (verbatim):**

```
> [!fail] "non-chained ILP" - what?! by definition, that's not ILP, because chained means dependent means not parallel, unless that's not what you mean. And not clear whether this is the same for the per-warp data above. Anyway, I don't think thawt's the bottleneck - what is *MUCH MUCH* more worrying is you aren't even specifying whether those are 4B or 16B accesses or what... I suspect it might just not be an efficient kernel? :(
```


### 68. Line 2290 — !FAIL

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
**DSMEM write aggregate ~560 GB/s per cluster** — but this is **issue rate, not completion**. V21 `push_ring_wr` has NO fence between the `st.shared::cluster` calls and the `clock64` end timer. PTX `st.shared::cluster` is fire-and-forget; the timer ends as soon as the last store enters the queue. **Real delivery rate is unbounded in this measurement.** The pair-uniform 34 cy "fenced write latency" (above) is more trustworthy because it includes a fence.
This is the **LOW-MED** confidence note from `DSMEM_DOUBT_REPORT.md`. The "13× higher than reads" framing is correct as a per-instruction issue rate ratio, but not as a fabric throughput ratio.
```

**User callout (verbatim):**

```
> [!fail] so you're not even fencing at the end? did you even check if DCE wasn't affecting this? this really should have been fairly easy to get right, please be careful and do better <3 However... I think this *might* be implicitly fenced to some extent assuming the working set is large enough, since otherwise there would be backpressure from the buses/pipelines... so possibly 70.08 at 2032MHz is ~34.4B/SM/clk which might be ~32B/clk when excluding the tail not being timed, which sounds believable unlike the read numbers... but this is really not rigorous and potentially very misleading.
```


### 69. Line 2322 — !FAIL

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
- N=8: 14.0 GB/s aggregate (1.80× per-reader slowdown)
**Per-CTA serving port caps at ≈ 15 GB/s** — a single peer can only deliver to ~15 GB/s worth of remote requesters.
```

**User callout (verbatim):**

```
> [!fail] if true that would mean 8B/SM/clk which would be insanely low - what does "1.80x per-reader slowdown" even mean, given 20.4/1.8 is not 14.0?
```


### 70. Line 2356 — !FAIL

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
| FFMA compute     | 0 % (210 vs 211 cy)   |
DSMEM competes for the peer's SMEM subsystem, not for its compute / SMSP. So **co-scheduling DSMEM with peer compute is free**; co-scheduling with peer SMEM access costs ~30 %.
```

**User callout (verbatim):**

```
> [!fail] 30% way too specific, will vary a lot, unclear what SASS looks like and whether it's SMEM or LDS or scheduling pressure
```


### 71. Line 2376 — !TODO

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
| fence.sc.sys          | 2870 (~9× slower) |
cluster / gpu **identical cost** → use `fence.sc.gpu` for safety with no penalty.
```

**User callout (verbatim):**

```
> [!todo] interesting - check SASS? what if there is traffic on the buses etc., is there any case where GPU gets slower but cluster doesn't? not very important tbh
```


### 72. Line 2388 — !FAIL

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
| .gpu           | 29.97                 |
| .cluster       | 31.40 (+1.4 cy / +5%) |
```

**User callout (verbatim):**

```
> [!fail] confusing / possibly misleading
```


### 73. Line 2402 — !FAIL

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
| 8-CTA ring all-reduce (V25)        | 842 cy/step      | 3.07 µs total | with fence + barrier |
Rule: **batch DSMEM writes**, emit ONE `fence.sc.cluster` + ONE `barrier.cluster.arrive/wait` to amortize the 320 cy fence cost.
```

**User callout (verbatim):**

```
> [!fail] should explain barrier vs fence, and how barrier.cluster is very brute force and only strictly required 1x at the start if clever etc.
```


### 74. Line 2417 — !FAIL

**Section:** §13. DSMEM (cluster shared memory)

**Catalog claim being reviewed:**

```
| **v2.u64 (128-bit)** | **29.66** | **0.54** |
Use `v2.u64` for widest per-thread DSMEM store.
```

**User callout (verbatim):**

```
> [!fail] so literally the ONLY wide memory ops you tried were single-thread for latency (of a store!) and not testing throughput properly? :(
```


### 75. Line 2552 — !FAIL

**Section:** §14. NVLink-5 (Blackwell)

**Catalog claim being reviewed:**

```
- **860 GB/s NVLink RX (ncu metric `nvlink__data_received`)** = bytes that crossed the link including FEC parity, header bytes, and link-layer protocol overhead
The 860 / 778 = 1.10 ratio matches expected NVLink-5 protocol overhead (53.125 / 50 raw + per-flit headers). Both are correct measurements; they measure different things. When citing, name which.
```

**User callout (verbatim):**

```
> [!fail] It's fascinating data but I think you are missing something much more simple and fundamental: reads are faster than writes because the header is more on the TX link, while the actual data comes back on the RX link, so at 778GB/s read there is some Read NVLink *TX* bandwidth as well of >100GB/s which you showed in your past data but now didn't include because you didn't fully realise its importance. For writes, that extra >100GB/s TX contends with the data being sent on the TX link, while the ACK confirmation packets on the RX link only take a tiny amount of bandwidth. That is one(!) of the key reasons why writes cannot get as close to the peak as reads. However... if both GPUs are reading data from each other, then that >100GB/s TX adds up to the RX of the other one, and we end up bottlenecked, and it doesn't help - in fact, it might be worse - to be determined...(?) - also consider the separate factor of SMs being limited to 32B/SM/clk write to L2, so if you fully reserve a small number of SMs for comms, reads also require fewer SMs than writes.
```


### 76. Line 2575 — !FAIL

**Section:** §14. NVLink-5 (Blackwell)

**Catalog claim being reviewed:**

```
| 64         | 792             |
| 148        | 817             |
```

**User callout (verbatim):**

```
> [!fail] 817 > 778 so 'saturated' isn't quitre the right word.
```


### 77. Line 2589 — !FAIL

**Section:** §14. NVLink-5 (Blackwell)

**Catalog claim being reviewed:**

```
- 1543 / 749 = 2.06× — close to perfect duplex
**NVLink 5 is essentially full-duplex** (within 3 % of perfect) on this 2× B300 NV18 setup.
```

**User callout (verbatim):**

```
> [!fail] 778+720 is meant to mean... what? pretty sure you are doing something weird, is one GPU reading and the other writing with different kernels? If not then why add up 778+720, they are just different ways of doing the same P2P depending on which side initiates the work. I am so confused, this is definitely not reliable...
```


### 78. Line 2601 — !FAIL

**Section:** §14. NVLink-5 (Blackwell)

**Catalog claim being reviewed:**

```
- Cross-GPU atomic latency = ~1.55 µs ≈ 3000 cy = **5× LOCAL**
Cross-GPU atomics are *expensive* — for hot atomic counters, keep them LOCAL and shard across GPUs with periodic rollups. NCCL's all-reduce primitive is the canonical primitive for this.
```

**User callout (verbatim):**

```
> [!fail] I trust the latency numbers but not the remote Gops/s, this needs a LOT more info to validate what it's doing.
```


### 79. Line 2623 — !FAIL

**Section:** §14. NVLink-5 (Blackwell)

**Catalog claim being reviewed:**

```
| NCCL with NVLink-SHARP | UNTESTED (no SHARP fabric on this NV18 system) |
NCCL's small-message latency floor of ~10 µs is competitive with anything you can write by hand. Use NCCL unless you have a specific reason not to.
```

**User callout (verbatim):**

```
> [!fail] no, ~10us is not good, you just aren't using the best possible kernel for this, it's hard though, lots of ninja tricks matter here
```


### 80. Line 2629 — !FAIL

**Section:** §14. NVLink-5 (Blackwell)

**Catalog claim being reviewed:**

```
### Multi-GPU sharded GEMM
`12_nvlink_p2p.md` finding: 0 % slowdown for multi-GPU sharded GEMM with proper tiling. cuBLAS's L2 tiling already accounts for the cross-GPU latency; the NVLink path is largely hidden.
```

**User callout (verbatim):**

```
> [!fail] not credible at all. this might be true for very large GEMM sizes where one input is in remote memory etc... and/or if your baseline shape is inefficient for other reasons... but you can't get away from having 1/10th the bandwidth.
```


### 81. Line 2635 — !FAIL

**Section:** §14. NVLink-5 (Blackwell)

**Catalog claim being reviewed:**

```
### Peer-fence drain
Cross-GPU `__threadfence_system` drains at +17.8 K cycles compared to single-GPU baseline — this is the NVLink-in-flight wait time. Use sparingly; prefer batched fences (CUDA Graphs, persistent kernels with mailbox handoff).
```

**User callout (verbatim):**

```
> [!fail] as per previous analysis, 17.8K is when heavily contended, the real number when nothing else is happening on the system is like 5-6K iirc? I don't know, way lower, this is misleading and you are missing some great data you generated in the past yourself :(
```


### 82. Line 2681 — !FAIL

**Section:** §14. NVLink-5 (Blackwell)

**Catalog claim being reviewed:**

```
### Stream-isolated NVLink
When using multiple streams with cross-GPU memcpy, only ONE stream sees full BW at a time (NVLink protocol is connection-oriented per-stream). To overlap multiple cross-GPU ops, use multiple `cudaStream_t` but expect aggregate BW = single-stream BW (778 GB/s read), not 4× single-stream. The 4 async copy engines on EACH side share the single NVLink fabric.
```

**User callout (verbatim):**

```
> [!fail] this is nonsense, "connection-oriented per-stream"? what? completely wrong level of abstraction, *OBVIOUSLY* more CUDA streams won't magically increase aggregate bandwidth...
```


### 83. Line 2756 — !TODO

**Section:** §15. PCIe Gen6 x16

**Catalog claim being reviewed:**

```
Pageable: **38 GB/s = 66 % of pinned**. The CUDA runtime page-migrates pageable memory through a staging buffer; the 34 % overhead reflects that copy. The "1.5 TB/s pageable" myth (from H100-era dispatch tricks) is well-debunked in `13_pcie_system.md`'s page-migration section — it doesn't apply to B300.
For real workloads: always use `cudaMallocHost` (pinned) for H2D buffers > 1 MB.
```

**User callout (verbatim):**

```
> [!todo] correct but shgould highlight high allocation time for pinned memory, i.e. very high init cost, so not worth it for short-term one-off buffers etc.
```


### 84. Line 2766 — !TODO

**Section:** §15. PCIe Gen6 x16

**Catalog claim being reviewed:**

```
- Splitting across 4 streams gives **better latency** for small transfers (parallelism).
- Use case: overlap H2D + D2H + compute + computes on different streams; engines schedule independently.
```

**User callout (verbatim):**

```
> [!todo] not verified that this is the case, it might be non-trivial to use them in parallel, not sure how this works
```


### 85. Line 2811 — !FAIL

**Section:** §15. PCIe Gen6 x16

**Catalog claim being reviewed:**

```
| `cuStreamCreate`                         | <1 µs                                       | Very fast                                        |
| CUDA graph capture+launch                | 15–35× faster than re-launch (after warmup) | Bursty inference                                 |
```

**User callout (verbatim):**

```
> [!fail] calling cudaStreamWriteValue a "hidden gem" is weird given 6-10us is actually pretty darn bad in my opinion - but then later you claim a MUCH lower latency for this, like 10x better, so it's confusing (in my experience, it's in-between those values, and maybe closer to the better one, but stream mem wait ops are annoyingly bad/slow)
```


### 86. Line 2837 — !FAIL

**Section:** §15. PCIe Gen6 x16

**Catalog claim being reviewed:**

```
| InfiniBand HDR               | 25 GB/s          | ~1 µs (with NIC)         | Cluster networking (not on this rig) |
| NVLink-C2C (GH200/GB200 NVL) | 450 GB/s         | ~50 ns                   | Not present on B300 SXM6             |
```

**User callout (verbatim):**

```
> [!fail] too authoritative sounding given how imprecise it is, doesn't mention L2 per-partition latency, etc...
```


### 87. Line 2938 — !FAIL

**Section:** §15. PCIe Gen6 x16

**Catalog claim being reviewed:**

```
```
This is faster than `cudaMalloc/cudaFree` for repeated alloc/free patterns.
```

**User callout (verbatim):**

```
> [!fail] CUDA VMM is the actual true ninja way of doing all this, you really should mention it explain it/test it...
```


### 88. Line 3127 — !FAIL

**Section:** §16. FP32 FFMA peak — 74.62 TFLOPS at 2032 MHz boost

**Catalog claim being reviewed:**

```
ALL TFLOPS claims must annotate the clock state. The ~6% gap between 1920 and
2032 explains most of the historical noise in this catalog.
```

**User callout (verbatim):**

```
> [!fail] ??? no reason for locked to be WORSE efficiency for given clock, 1920 lock should not throttle for this, so how is it only 85.5% SOL, something went very very wrong and it's extremely bad you even include this number as-is in the table above when it's so misleading
```


### 89. Line 3206 — !FAIL

**Section:** §17. FFMA register-source dependence — 3-distinct-source caps at ~67%

**Catalog claim being reviewed:**

```
The ratio `0.683 ≈ 2/3` exactly matches the prediction from a 2-RF-read-port
model: 3 reads / 2 ports = 1.5 cy per FFMA = 1/1.5 = 0.667 throughput.
```

**User callout (verbatim):**

```
> [!fail] this should really be measured with FFMA2 too, and look at SASS/reuse/different patterns/etc... but still good data overall
```


### 90. Line 3241 — !FAIL

**Section:** §17. FFMA register-source dependence — 3-distinct-source caps at ~67%

**Catalog claim being reviewed:**

```
empirical anchor for the 2-RF-read-port model. SASS-verified `.reuse` count:
255/256 in broadcast mode, 0/256 in per-chain mode.
```

**User callout (verbatim):**

```
> [!fail] show SASS, is this still 3 unique input operands unlike fma a,b,a,b but with reuse? or?
```


### 91. Line 3254 — !FAIL

**Section:** §17. FFMA register-source dependence — 3-distinct-source caps at ~67%

**Catalog claim being reviewed:**

```
- Effective port count when one operand is hot: **3 reads/cy**
- Effective port count when all 3 operands distinct: **2 reads/cy** → 1.5 cy/FFMA
```

**User callout (verbatim):**

```
> [!fail] how reliably did you test the '1 entry' for reuse cache, how sure are you it's not more? Also what about forwarding / write caching for chain cases
```


### 92. Line 3275 — !FAIL

**Section:** §17. FFMA register-source dependence — 3-distinct-source caps at ~67%

**Catalog claim being reviewed:**

```
**the realistic FP32 ceiling is ~51 TFLOPS, NOT 75**. This is the single most
important number to communicate when budgeting real-workload performance.
```

**User callout (verbatim):**

```
> [!fail] Did you confirm vector dot product is really 67%, could forwarding/write caching help in some way, what if you have e.g. 1 warp per SMSP with ILP of exactly 4 to match pipeline depth with one very long chain, both with and without using FFMA2 - I expect that might get >67%, but I am not sure.
```


### 93. Line 3285 — !FAIL

**Section:** §17. FFMA register-source dependence — 3-distinct-source caps at ~67%

**Catalog claim being reviewed:**

```
is observable only on FFMA (and similar high-throughput compute) where the pipe
itself is fast enough that the RF becomes the next bottleneck.
```

**User callout (verbatim):**

```
> [!fail] and when multiple pipes are used in parallel, which they typically are, so... the other even more interesting case here is FFMA2 + ALU co-issue where we can effectively have 50% more inputs required per clock(!) and reuse/broadcast are even more critical to achieving peak
```


### 94. Line 3363 — !TODO

**Section:** §18. FFMA `.reuse` cache — the SASS-level operand bypass

**Catalog claim being reviewed:**

```
- Cycle N: read Rd, b, Rd → 3 reads / 2 ports = 1.5 cy
- Cycle N+1: same → 1.5 cy
```

**User callout (verbatim):**

```
> [!todo] it would be semi-interesting to try to figure out if the hardware reads Rd twice here (and whether it reads it twice for e.g. Rd multiplied by Rd versus Rd multiplied by IMM) by measuring power for different patterns
```


### 95. Line 3464 — !FAIL

**Section:** §19. FADD = FMUL = FFMA at SASS level

**Catalog claim being reviewed:**

```
the same 4.22 cy latency and same ~97.65% pipe saturation rate. FFMA "wins"
purely because each instruction carries 2 FLOPS instead of 1 (or 1).  `[🟢 HIGH · src: V8_FADD_FMUL_PEAK.md, V9_OP_LATENCY.md]`
```

**User callout (verbatim):**

```
> [!fail] 4.0 cycles latency - if 4.22, that means your (boost) clock is wrong, or your loop overhead is bad, or...
```


### 96. Line 3545 — !FAIL

**Section:** §19. FADD = FMUL = FFMA at SASS level

**Catalog claim being reviewed:**

```
throughput from the same inst rate. To guarantee FFMA emission, use explicit
`asm("fma.rn.f32 %0, %1, %2, %3;" ...)` or `__fmaf_rn(a, b, c)`.
```

**User callout (verbatim):**

```
> [!fail] is this really a risk when fast math is enabled? I am skeptical any modern compiler would get this wrong except maybe with crazy function calls and/or other optimizations that fold a*b into something else and save as much or more or something
```


---

## Summary

**Total callouts:** 96

- **[!fail]:** 87 callouts (concerns, disagreements, potential errors)
- **[!todo]:** 9 callouts (suggestions for additional measurement/validation)

## Frequent Themes in [!fail] Callouts

- **Bandwidth/Sol:** 19 callouts
- **L2/Cache:** 17 callouts
- **Measurement Rigor:** 15 callouts
- **Ecc/Memory:** 13 callouts
- **Dram/Hbm:** 11 callouts
- **Clock State:** 9 callouts

**Key observations:**

1. Clock-state dependencies and assumptions appear in ~8 callouts (§3, §11, §16)
2. Measurement methodology concerns appear in ~15 callouts (NCU validation, SASS verification, DCE/LICM handling)
3. Memory subsystem claims (L2 partitioning, cache hit rates, bandwidth amplification) questioned in ~12 callouts
4. ECC and memory capacity confusion flagged in 3-4 callouts
5. Stack-locality and DSMEM testing criticized in 6-7 callouts
