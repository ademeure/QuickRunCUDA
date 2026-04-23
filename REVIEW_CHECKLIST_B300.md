# B300 Catalog — Review Checklist (one-line yes/no per claim)

> **What this is:** My OWN skeptical review of `B300_PIPE_CATALOG.md` claims, sectioned for fast yes/no scanning. The user's `reviewed_errors_b300.md` notes were on a DIFFERENT (lower-quality) doc — I use them as INSPIRATION for the kinds of issues to look for here (clock-state assumptions, formula-as-measurement, DCE, denominator mixing, agent-hallucinations of names/mechanisms, etc.), but the claims listed below are the catalog's own.
>
> **Instructions for user:** Each row is a single claim I am NOT 100% certain of. Mark with `[x]` for "yes, this is correct", `[ ]` for "no, wrong", and add a `// short note` after the line if you want to leave a comment. The structure is grouped by catalog section so you can stop after any group without losing context.
>
> **Format:**
> - `[ ]` claim text — context (numeric value, regime if relevant) — `[ref: B300_PIPE_CATALOG.md:Lxxx]` — `[reason for uncertainty]`
> - One blank line between groups.
>
> **Color key for `[reason for uncertainty]`:**
> - `unverified`: I haven't been able to replicate it yet
> - `clock-mismatch`: number was at a different clock than headline implies
> - `formula`: claim looks like it's computed from spec, not measured
> - `DCE-suspect`: pattern looks vulnerable to compiler eliminating the loop
> - `agent-hearsay`: number came from an LLM swarm without independent re-verify, or names a mechanism (SASS opcode, memory path) that I haven't seen documentation for
> - `inconsistent`: same topic reported with different number elsewhere in the catalog
> - `regime-narrow`: holds only under one combination of (warps, BS, working set) and may not generalize
> - `unit-confusion`: GB vs GiB, wire-rate vs effective, packed-element-count vs SASS-inst-count
> - `superseded-suspect`: I think a later test contradicted this but haven't traced
>
> **Note on cross-references:** Where an entry says `[ref: reviewed_errors L###]`, that's because the user raised an ANALOGOUS skepticism point on the canonical doc — used here as the inspiration for flagging the catalog's similar claim. The catalog claim itself is the load-bearing thing being reviewed.

---

## Group A — Headline FP32 / FFMA

- [ ] **A1** "FP32 FFMA peak = 71.8 TFLOPS = 98.8% of theoretical 72.7 TFLOPS at 1.92 GHz" — the 72.7 denominator uses 1920 MHz; CLAUDE.md says theoretical at 2032 MHz boost is 76.96 TFLOPS — `[ref: B300_PIPE_CATALOG.md:30]` — `[clock-mismatch]`
- [ ] **A2** "256 FLOPS/clk/SM" formula assumes scalar FFMA dispatches to BOTH fma sub-pipes simultaneously — V52 confirmed this for FFMA but also showed "dispatch cap=4" framing is misleading (alu+fma=147%) — `[ref: B300_PIPE_CATALOG.md:30,210]` — `[regime-narrow]`
- [ ] **A3** "FP32 via FFMA2 packed = 72.3 TFLOPS = 99.4%" — same chip-FLOPS as scalar FFMA per claim; needs SASS verification that FFMA2 is actually emitted (not FFMA scalar pair) — `[ref: B300_PIPE_CATALOG.md:31]` — `[unverified]`
- [ ] **A4** "FP16 via HFMA2 = 72.3 TFLOPS-FP16, no extra throughput on scalar path" — claim is that compiler packs into HFMA2 anyway; verify with SASS for both packed and scalar tests — `[ref: B300_PIPE_CATALOG.md:32,33]` — `[unverified]`
- [ ] **A5** "FP64 DFMA = 0.95 TFLOPS = 1/76× of FFMA" — CLAUDE.md says 1.20 TFLOPS theoretical (1:64 ratio per SingleToDoublePrecisionPerfRatio); 0.95 < 1.20 implies under-saturation — `[ref: B300_PIPE_CATALOG.md:35]` — `[inconsistent]`
- [ ] **A6** "FFMA dispatch ceiling = 4.00 sm_inst/SM/cy is hard" — V52 demonstrated alu+fma sum = 147% which means total can exceed dispatch (different pipes overlap); need to clarify what "dispatch" means — `[ref: B300_PIPE_CATALOG.md:480]` — `[regime-narrow]`

## Group B — Memory hierarchy

- [ ] **B1** "smem 35.6 TB/s = 98% theoretical (128 B/clk/SM × 148 × 1.92)" — uses 1.92 GHz; if test ran at 2032 boost the SoL would be 38.4 TB/s. Need clock state captured — `[ref: B300_PIPE_CATALOG.md:41]` — `[clock-mismatch]`
- [ ] **B2** "L1 hit (.ca, WS≤1MB) = 36.1 TB/s" but "L1 = 28.7 TB/s" — same tier, different numbers; suggests L1 was sometimes measuring L2-bypass effects. Mechanism unclear — `[ref: B300_PIPE_CATALOG.md:42,43]` — `[inconsistent]`
- [ ] **B3** "L2 plateau 22-26 TB/s — was wrongly 10.2, under-occupied launch" — the 4× spread suggests methodology was historically broken; need to verify whether 22-26 is now stable across launches — `[ref: B300_PIPE_CATALOG.md:44]` — `[regime-narrow]`
- [ ] **B4** "L2 knee gradual 23 → 22 → 20 → 11 TB/s" at "1 GB → 11 TB/s" — but pure HBM is 7.18 TB/s, so 11 TB/s at 1 GB implies L2-amortization at boundary. User asks for ncu split — `[ref: B300_PIPE_CATALOG.md:45]` — `[unit-confusion]`
- [ ] **B5** "DRAM (HBM3E) read = 7.18 TB/s ncu-verified" with "WS≥1GB→8GB consistent" — likely correct; should denominator be spec 7680 or this-device 7672? — `[ref: B300_PIPE_CATALOG.md:46]` — `[unit-confusion]`
- [ ] **B6** "DRAM write = 7.09 TB/s" — User flag: "SM→L2 *write* path is limited to 32B/clk, you cannot get peak HBM write at lower clocks; 1920 vs 2032 might affect this" — `[ref: B300_PIPE_CATALOG.md:46, reviewed_errors L1132]` — `[clock-mismatch]`
- [ ] **B7** "TMEM read 55.92 TB/s, drops to 31 with 4R/iter" — surprisingly high; my back-of-env says theoretical TMEM read is bounded by tcgen05.ld throughput per warp × SMs which gives much less. Methodology + SASS dump needed — `[ref: B300_PIPE_CATALOG.md:50]` — `[unverified]`
- [ ] **B8** "TMEM write 97.93 → 131 TB/s" — even more suspect; needs first-principles bound check — `[ref: B300_PIPE_CATALOG.md:50]` — `[unverified]`
- [ ] **B9** "Constant memory broadcast LDC.32 = 17.8 TB/s eff (~0.55 TB/s actual cache traffic)" — broadcast amplification × 32 lanes; verify denominator — `[ref: B300_PIPE_CATALOG.md:47]` — `[unit-confusion]`
- [ ] **B10** "Local (register spill) 1.3 TB/s = 52× slower than smem" — likely correct but needs spill-depth-controlled test (catalog notes spill cliff at 32 vars) — `[ref: B300_PIPE_CATALOG.md:49]` — `[regime-narrow]`
- [ ] **B11** "256-byte stride v8 .256B L2 cache modifier" test — user remembers achieving HIGHEST DRAM BW % in any microbenchmark with this; catalog has lost track — needs to be re-located in tests/ — `[ref: reviewed_errors L925]` — `[superseded-suspect]`
- [ ] **B12** "LDG.E.ENL2.256 means bypassing L1" — user is skeptical: "Are you sure ENL2 means what you think it means?" — needs ISA reference check — `[ref: B300_PIPE_CATALOG.md, reviewed_errors L1063]` — `[agent-hearsay]`
- [ ] **B13** "ptxas compiles differently based on cudaMalloc vs cudaMallocAsync" — user: "complete hallucination" — `[ref: reviewed_errors L1063]` — `[agent-hearsay]`
- [ ] **B14** "L2/XBAR clock = 1860 MHz constant, NOT affected by -lgc" — user: "1860 MHz is NOT constant" and separately "L2/XBAR *is* affected by -lgc but only indirectly" — needs clock sweep test — `[ref: B300_PIPE_CATALOG.md (canonical L576), reviewed_errors L583,L595]` — `[agent-hearsay]`
- [ ] **B15** "Per-stack BW independence + cross-stack hashing controllable" — user: "cross-stack hashing is literally impossible to turn off" — stack-locality recipes for D2D 6.93 TB/s are dubious — `[ref: reviewed_errors L794,L1320]` — `[agent-hearsay]`

## Group C — Pipe topology / dispatch ceiling

- [ ] **C1** "Dispatch ceiling = 4.00 warp-inst/SM/cy" — V52 shows pipes overlap freely (alu+fma=147%), so the "hard cap" framing is misleading. Per-SMSP dispatch cap of 1 is the real story — `[ref: B300_PIPE_CATALOG.md:189,210,480]` — `[regime-narrow]` — **2026-04-23 update:** ncu re-confirmed (justifications/01_pipe_topology.md): pure FFMA hits 3.88, dual hits 3.95; alu+fma sum = 145%. Cap IS a hard 4.00 in strict-total sense; "free overlap" claim was always about cross-pipe sums, not within-pipe. Catalog framing OK; only L218 wording needs fix (see new C8 below).
- [ ] **C8** "FFMA → uniquely uses BOTH fma sub-pipes simultaneously" — **FALSIFIED 2026-04-23.** ncu shows in dual mode pipe_fmalite=93% but pipe_fmaheavy=4.5% — FFMA dispatches to ONE sub-pipe per cycle, scheduler-chosen, not both. — `[ref: B300_PIPE_CATALOG.md:218]` — `[agent-hearsay]` — Catalog wording needs to change to "FFMA can use EITHER sub-pipe per cycle, alternating freely". Doesn't change the 256 FLOPS/SM/cy peak (4 SMSPs × 1 dispatch × 32 lanes × 2 FLOPS = 256), but the mechanism story is wrong.
- [ ] **C2** "pipe_alu cap 2.00" — needs ncu verification of `sm__inst_executed_pipe_alu` for pure-LOP3 test; only V52 explicitly tested this — `[ref: B300_PIPE_CATALOG.md:196]` — `[unverified]`
- [ ] **C3** "pipe_xu cap 0.50 compound, 1.00 simple" — the compound vs simple distinction needs explicit test (MUFU.SIN vs MUFU.EX2) — `[ref: B300_PIPE_CATALOG.md:200]` — `[unverified]`
- [ ] **C4** "pipe_uniform handles ACTIVEMASK and LDSM" — partial verification; the full Blackwell uniform datapath claims (UFFMA, UFADD, etc.) are inferred from NVIDIA docs not measured here — `[ref: B300_PIPE_CATALOG.md:530]` — `[unverified]`
- [ ] **C5** "FFMA2 + UNPACK gives u=1.67 (16% SMSP friction specific to F2FP)" — catalog §3 contention rule; PRMT+FFMA2 hits u=1.95. Need replication — `[ref: B300_PIPE_CATALOG.md:478]` — `[regime-narrow]`
- [ ] **C6** "match-any-sync 375 cy = 20× other warp ops" — likely correct but value matters; should be in latency table — `[ref: B300_PIPE_CATALOG.md:85]` — `[unverified]`

## Group D — Tensor cores

- [ ] **D1** "FP16/BF16 mma.sync m16n8k16 = 577 TFLOPS, 4-chain 574 / 8-chain 577 = 101% of 569 SOL estimate" — 101% of estimate suggests SoL estimate is wrong; needs first-principles bound — `[ref: B300_PIPE_CATALOG.md:25]` — `[formula]`
- [ ] **D2** "TF32 mma.sync m16n8k8 = 288 TFLOPS, 'catalog previously wrongly listed 141'" — earlier catalog version had 141, now claims 288. Suggests methodology is volatile — `[ref: B300_PIPE_CATALOG.md:26]` — `[superseded-suspect]`
- [ ] **D3** "FP8 mma.sync = 276 TFLOPS emulated via F2FP+HMMA" — earlier 2336/2247 numbers were FADD artifacts (DCE'd 99.99% of mma chain). Needs current-test SASS verification — `[ref: B300_PIPE_CATALOG.md:27]` — `[DCE-suspect]`
- [ ] **D4** "INT8 mma.sync IMMA = 142 TOPS, 45× slower than FP8" — needs verification; B300 deliberately deprecates INT8 per claim — `[ref: B300_PIPE_CATALOG.md:28,88]` — `[unverified]`
- [ ] **D5** "tcgen05.mma all formats = 128 cy at M=128 N=256" — clean claim but needs replication for ≥3 formats (FP16, FP8, FP4) — `[ref: B300_PIPE_CATALOG.md:120-130]` — `[unverified]`
- [ ] **D6** "FP4 NVFP4 K=64 = 9856 TFLOPS chip TFLOPS" — needs verification; the K=96 ULTRA path that gives 1.5× is mentioned elsewhere as inaccessible in public libs — `[ref: B300_PIPE_CATALOG.md:128]` — `[regime-narrow]`
- [ ] **D7** "P2P GEMM remote weights via NVLink: zero penalty" — based on cuBLAS L2-tiling; needs confirmation that "tiles fit in L2" explanation is what's actually happening (vs bandwidth-bound) — `[ref: B300_PIPE_CATALOG.md:178-183]` — `[regime-narrow]`

## Group E — Latency / sync / atomics

- [ ] **E1** "FFMA latency = 4 cy" — generally correct; needs to specify whether single-chain or RAW dependency — `[ref: B300_PIPE_CATALOG.md:101]` — `[regime-narrow]`
- [ ] **E2** "DFMA latency = 92 cy, no ILP" — likely correct (heavily throttled FP64); needs SASS dump — `[ref: B300_PIPE_CATALOG.md:103]` — `[unverified]`
- [ ] **E3** "MUFU.sin latency = 24 cy, ILP throughput 8.4 cy with 3 chains" — needs replication; user separately notes EX2 has split issue/result-availability latencies — `[ref: B300_PIPE_CATALOG.md:106]` — `[unverified]`
- [ ] **E4** "fence.sc.gpu = 274 cy" — V54 settled at 267 cy + 280 cy first-fence-after-write penalty; catalog needs update — `[ref: B300_PIPE_CATALOG.md:115]` — `[superseded-suspect]`
- [ ] **E5** "__syncthreads at BS=512 cost 45 cy, BS=1024 cost 89 cy" — formula `12 + 2W` cy from L116 doesn't match these (would give 44 / 76 cy). Inconsistent — `[ref: B300_PIPE_CATALOG.md:74,75,116]` — `[inconsistent]`
- [ ] **E6** "Atomic single-address chip-wide is 5× FASTER than per-warp atomic hotspot" — per claim L87, the per-warp hotspot is SLOWER; verify mechanism (cache-line combining) — `[ref: B300_PIPE_CATALOG.md:87]` — `[unverified]`
- [ ] **E7** "All-reduce ≤1 MB floor = 21 µs, NCCL = 10 µs" — multi-GPU; needs MGFenceBench + nccl-tests verification at this rig — `[ref: B300_PIPE_CATALOG.md:157,168]` — `[unverified]`

## Group F — Power / clock / DVS

- [ ] **F1** "1005 MHz silent stuck mode" — user: "I strongly suspect this was the result of another agent or something else" — was it actually architectural or just process leftover? — `[ref: reviewed_errors L500]` — `[agent-hearsay]`
- [ ] **F2** "V² DVS scaling: V scales linearly with clock" — user: "GPU power does not scale in such a simple way, this is only a very rough 1st approximation, misleading" — should sample actual nvidia-smi voltage — `[ref: reviewed_errors L539]` — `[formula]`
- [ ] **F3** "1920 MHz is sustained boost" — user: "1920 MHz cannot be sustained for heavy tensor core workloads or even many other things" — throttling concerns — `[ref: reviewed_errors L564]` — `[regime-narrow]`
- [ ] **F4** "POPCOUNT bell curve peak at d=16" — user: "DRAM-1G W is probably just getting some L2 hits, this is confusing/misleading" — methodology has L2 amortization confound — `[ref: reviewed_errors L1369]` — `[regime-narrow]`
- [ ] **F5** "1071 W stress recipe at 1500 MHz lock + d=16 random" — user wants voltage capture — `[ref: reviewed_errors L1387]` — `[unverified]`
- [ ] **F6** "Clock-vs-power table 510→2032 MHz V scaling" — entire table approximate; user says misleading — `[ref: B300_PIPE_CATALOG.md (canonical L528-535)]` — `[formula]`

## Group G — TMA / mbarrier / cluster

- [ ] **G1** "TMA cp.async.bulk issue rate = 48 cy/inst floor (size-independent)" — needs replication; the size-independence is critical claim — `[ref: B300_PIPE_CATALOG.md:58]` — `[unverified]`
- [ ] **G2** "TMA chip-wide realistic = 29.2 TB/s, but ncu confirms only 12.6 GB/s actual DRAM (so this is L2→smem, not DRAM)" — important caveat; verify ncu dram__bytes_read numbers separate from "TMA pipe BW" — `[ref: B300_PIPE_CATALOG.md:61]` — `[unit-confusion]`
- [ ] **G3** "TMA bytes per instruction not specified" — user [!fail]: "this section does not tell me what the number of bytes per TMA instruction is, so this is not very informative" — need explicit bytes/inst breakdown for each TMA test — `[ref: reviewed_errors L1008]` — `[unit-confusion]`
- [ ] **G4** "Multicast cannot be pipelined" claim — user: "this doesn't mean it cannot be pipelined - just that we are hitting maximum throughput with the amount of latency tolerance we already have" — `[ref: reviewed_errors L1029]` — `[regime-narrow]`
- [ ] **G5** "fence.proxy.async.shared::cta lowers to MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S" — needs SASS verification — `[ref: B300_PIPE_CATALOG.md:81]` — `[unverified]`
- [ ] **G6** "mbarrier RTT = 54 cy single-thread count=1" — needs replication — `[ref: B300_PIPE_CATALOG.md:73]` — `[unverified]`
- [ ] **G7** "228 KB hardware max smem per-SM, 200 KB per CTA without opt-in" — likely from cudaDeviceProp; verify on this machine — `[ref: B300_PIPE_CATALOG.md:84]` — `[unverified]`

## Group H — Methodology assumptions baked into many measurements

- [ ] **H1** Whole "dual-issue based on pipes" framing is dubious — user [!fail]: "depends how pipes are defined in ncu, to validate 'useful work' GOps/s as well, the fact this isn't clearly highlighted here as part of the methodology is worrying" — `[ref: reviewed_errors L309]` — `[agent-hearsay]`
- [ ] **H2** "Catalog mixes 1800/1920/2032 MHz across runs; cycle counts are clock-independent" claim at top of catalog — user concerns suggest cycle counts ARE often clock-dependent due to memory subsystem — `[ref: B300_PIPE_CATALOG.md:14]` — `[regime-narrow]`
- [ ] **H3** "256 cores/SM" vs "128 cores/SM" — CLAUDE.md is emphatic about 128. Catalog L30 uses "256 FLOPS/clk/SM" which assumes the FFMA-dispatches-to-2-pipes behavior. Anywhere else in catalog using "256 cores" should be flagged — `[ref: CLAUDE.md vs B300_PIPE_CATALOG.md:30]` — `[unit-confusion]`
- [ ] **H4** Wall-clock measurements affected by launch overhead for kernels <100 µs; catalog has many such tests — needs runtime ≥10 ms verification per test — `[ref: CLAUDE.md §3]` — `[regime-narrow]`
- [ ] **H5** Many tests use `ncu pipe_tensor` for tcgen05.mma — but that counter only measures legacy mma.sync (HMMA). Any tcgen05.mma claim measured via pipe_tensor is wrong — `[ref: project memory]` — `[agent-hearsay]`
- [ ] **H6** "Tests verified with ncu" claims often lack exact metric name — should always state which ncu counter — `[ref: catalog throughout]` — `[unverified]`
- [ ] **H7** "I have not manually checked most of this" warning at top is honest but means the catalog has many propagated agent-hallucinations — every claim needs SASS+ncu re-verify — `[ref: B300_PIPE_CATALOG.md:5]` — `[agent-hearsay]`

---

## Summary by uncertainty type (for quick scan)

| Uncertainty type | Count | Most-affected groups |
|---|--:|---|
| `unverified` | 22 | C, D, G |
| `clock-mismatch` | 4 | A, B, F |
| `regime-narrow` | 12 | A, C, F, H |
| `agent-hearsay` | 8 | B, F, H |
| `formula` | 3 | D, F |
| `inconsistent` | 3 | A, B, E |
| `unit-confusion` | 5 | B, G, H |
| `superseded-suspect` | 3 | B, D, E |
| `DCE-suspect` | 1 | D |

Note: a single line can carry multiple flags; most-load-bearing flag listed.
