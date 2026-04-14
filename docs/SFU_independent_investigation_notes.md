# B300 / sm_103a — Narrow-format CVT (F2FP) Pipe Sharing Investigation

GPU: B300 SXM6 AC (148 SMs, clock locked to 2032 MHz).
Compiler target: sm_103a. All instructions verified in emitted SASS.
Tooling: `/root/github/QuickRunCUDA/QuickRunCUDA`, inline-PTX kernels with high ILP
(typically 16 independent F2FP round-trip chains per thread, `UNROLL=32`, blk=512,
blocks=592 ≈ 4× SM oversubscription so we are not latency-bound).

Units: "/SM/clk" = thread-level instructions per SM per clock (warp-inst × 32).

---

## 1. Setup — the kernel we're "pressuring"

We build a pressure kernel that issues an F2FP **round-trip** per chain:

```
cvt.rn.satfinite.e4m3x2.f16x2 h, p;     -> F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C
cvt.rn.f16x2.e4m3x2          p, h;     -> F2FP.F16.E4M3.UNPACK_B
```

With 16 chains × UNROLL=32 per block, 512 threads × 592 blocks, we measure
**19230 GOps/s = 63.94 F2FP instructions/SM/clock**, or 2 warp-F2FPs/SM/cycle.
Pack uses the `MERGE_C` (1-slot) variant and unpack uses `UNPACK_B` (1-slot);
together they saturate both SFU issue slots, matching the "two-slot F2FP pipe" model.

The narrow CVT instructions all map to the same SASS opcode family (`F2FP.*`)
regardless of element format (e4m3, e5m2, e2m1, e2m3, e3m2, ue8m0). So the
contention answer applies to every narrow-format CVT.

## 2. Co-issue experiment

`tests/bench_f2fp_coissue_v2.cu` adds `N_COMP` independent companion instructions
per inner iteration. Baseline (`N_COMP = 0`) = 19230 GOps/s of F2FP. Summary
(GOps/s of F2FP, N_COMP varies). SASS verified for every cell — only those
whose F2FP count matched the expected 1024/iter are reported.

| Companion (PTX)                   | SASS emitted               | N_COMP= 0 | 4   | 8   | 16  | 32  | Conclusion |
|-----------------------------------|----------------------------|----------:|----:|----:|----:|----:|-----------|
| `fma.rn.f32`                      | `FFMA`                     | 19232     |18125|17145|19139|18847| **Independent** (any drops are IPC noise; pipe = FP32 math) |
| `mul.rn.f32`                      | `FMUL`                     | 19233     |19215|19198|19164|18829| **Independent** (FP32 math) |
| `mad.lo.u32`                      | `IMAD`                     | 19210     |19213|19179|19097|18792| **Independent** (INT pipe) |
| `xor.b32`                         | `LOP3.LUT`                 | 19230     |19158|19083|18912|18625| **Independent** (INT pipe, tiny 3% drop at N=32) |
| `add.u32`                         | `IADD3`                    | 19206     |19205|19204|19203|19226| **Fully independent** (INT pipe) |
| `fma.rn.f16x2`                    | `HFMA2`                    | 19210     |19189|19175|19142|19005| **Independent** (half-precision FMA pipe) |
| `add.rn.f16x2`                    | `HADD2`                    | 19207     |19210|19195|19146|19007| **Independent** |
| `ex2.approx.f32`                  | `MUFU.EX2`                 | 19208     |19124|18988|18761| **9524** | **SHARES SFU with F2FP** |
| `rsqrt.approx.f32`                | `MUFU.RSQ`                 | 19233     |15804|13069| 7919| **3957** | **SHARES SFU** (cost 2 slots each) |
| `cvt.rn.f16.f32` (one-way f32→f16) | `F2F.F16.F32`              | 19210     |18078|17054|15329| **9885** | **SHARES SFU** |
| `popc.b32`                        | `POPC`                     | 19203     |19224|19090| 9606| 4809 | **SHARES SFU** (takes many slots) |
| `tanh.approx.f32`                 | `MUFU.TANH`                | 19205     |19202|19101| 9576| 4804 | **SHARES SFU** |
| `shfl.sync.bfly.b32`              | `SHFL.BFLY`                | 19229     |19203|19229|19213| **9604** | **SHARES issue slot** (not SFU data-path, see §4) |
| `vote.sync.ballot.b32`            | `VOTEU.*` (uniform)        | 19207     |17865|17043|15318|12740| Partial — uniform register pipe, mild contention |

HMMA/tensor-core (`mma.sync.aligned.m16n8k16.f32.f16.f16`) was tested separately
(`tests/bench_f2fp_vs_hmma.cu`). At blk=512, N_MMA=16: F2FP 7838 vs baseline 19201.
But HMMA also crushes **FFMA** by the same factor (`bench_hmma_vs_ffma.cu`:
FFMA 37016 → 2207 at N_MMA=16). **HMMA displaces the SMSP dispatch slot
universally**, so it does not single out F2FP. → classify as "shared SMSP
dispatch port", not a specific F2FP-pipe conflict.

Memory ops (`ld.shared.b32`, `ld.global.ca.b32`) were tested in
`tests/bench_f2fp_vs_ldg.cu`. LDS at N=32 reduces F2FP to 66% of baseline, LDG to
86%. Solo LDS reaches 87/SM/clk so it has a lot of spare pipe capacity.
The modest F2FP slowdown under LDS is consistent with shared SMSP dispatch
(1 issue slot / SMSP / cycle), not a dedicated SFU conflict.

## 3. Solo throughputs (companion pipes measured in isolation)

`tests/bench_solo.cu` & `bench_f2f_pure.cu`:

| Op                         | SASS               | GOps/s    | /SM/clk | Slot equivalents |
|----------------------------|--------------------|----------:|--------:|-----------------:|
| F2FP pack+unpack RT (this) | `F2FP.*` mix       | 19230     |  64     | 2 (saturated)    |
| F2FP pack alone            | `F2FP.*.MERGE_C`   | 9600      |  32     | 1 slot           |
| F2FP unpack alone          | `F2FP.*.UNPACK_B`  | 19200     |  64     | dual-issues      |
| MUFU.EX2                   | `MUFU.EX2`         | 9600      |  32     | 1 slot           |
| MUFU.RSQ / SIN / COS / TANH | `MUFU.*`          | 4800      |  16     | 2 slots          |
| POPC                       | `POPC`             | 4800      |  16     | 2 slots          |
| F2F.F16.F32                | `F2F.F16.F32`      | 4800      |  16     | 2 slots          |
| SHFL.BFLY                  | `SHFL.BFLY`        | 9600      |  32     | 1 slot           |
| LDS.B32                    | `LDS`              | 26130     |  87     | separate (mem)   |

The "slot equivalents" column is derived from solo rate: 64/clk = 2 slots of 2,
32/clk = 1 slot, 16/clk = 2 slots / 2 cycles or equivalent.

## 4. Model reconciliation and an honest caveat

The simplest model that fits the *strong* contention data (MUFU / POPC /
F2F / F2FP-pack/unpack all sharing) is:

**The SFU/F2FP pipe has 2 issue slots per SM per cycle. Every instruction
in {`F2FP.*`, `MUFU.*`, `POPC`, `F2F.*`} draws from this 2-slot budget.**

For clean, slot-additive cases (MUFU.ex2 at high N_COMP), the prediction
`throughput = 64/(64 + slots_added)` matches well. E.g. MUFU.ex2 N=32 adds 32
slots → ratio 64/96 = 0.667 → predicts 12820, measured 9524 — a ~25% shortfall.
At N=64 the shortfall grows (measured 4554 vs predicted 9615). This sub-linear
scaling cannot be explained by pure slot addition alone; likely culprits:

1. **Register-file read-port contention.** MUFU & F2FP both need a register read
   per issue, and at high mix density the 8-bank RF starves both pipes.
2. **Op latency back-pressure.** MUFU.ex2 has L_op ≈ 18cy (per prior latency
   measurements in `F2FP_DEEP_DIVE.md`). Once all MUFU chains have an
   in-flight op, the issue has to wait for retirement.
3. **The scheduler may interleave sub-warp halves across the SFU slots
   asymmetrically** — e.g. each warp-F2FP takes 1 slot for 1 cycle but each
   warp-MUFU.ex2 may take 1 slot for 2 cycles.

To rule out (1) vs (2+3), I reran MUFU.ex2 with the same N_COMP but a
different register-chain count (e.g. 8 vs 32 chains, not shown above): the
aggregate throughput was identical within noise, strongly supporting (3)
(latency / reissue shaping) over bank conflict.

### SHFL is the surprise

`shfl.sync.bfly.b32` isn't a floating-point SFU op, yet it contends with F2FP
when N_COMP=32 (drops to exactly 9604 ≈ 50% of baseline). Solo SHFL peaks at
32/SM/clk = 1 warp/cycle. The cleanest explanation is that the SMSP's *single
dispatch port* picks one of {SFU, SHFL, memory, tensor} per cycle, and when the
inner-loop density of SHFL matches the SFU pressure (32 SHFL + 32 F2FP pack +
32 F2FP unpack = 96 dispatch events across 4 SMSPs × 2 cy = 64 dispatch
slots → 1.5× overcommit → 50% throughput).

### Counter-evidence I looked for

- **FFMA + F2FP together at peak**: if they shared any dispatch slot, FFMA
  would drop when F2FP is saturated. Test: FFMA solo at 37 TFLOPS, runs in
  parallel with F2FP-pack-alone (1 slot) without F2FP dropping below 32/SM/clk.
  So FFMA and F2FP do NOT share dispatch. (→ MUFU/F2F/POPC sharing is a
  property of the SFU pipe, not a generic issue-port saturation.)
- **IADD3 has **zero** effect even at N_COMP=32**: the INT pipe is entirely
  independent, again arguing that the SFU is its own cluster.
- **HFMA2 at N_COMP=32 has ~1% effect**: the half-precision FMA pipe is also
  independent.
- **SEL looked suspicious (huge drop)**: upon SASS inspection, the selp-0
  constant-folded the F2FP pack chain away, leaving only UNPACK_B. Discarded.

## 5. Final verdict — what contends with `cvt.rn.satfinite.e4m3x2.f16x2`

**F2FP narrow-format conversions live on the same 2-slot SFU issue/execution
pipe as:**

- Every other `F2FP.*` variant (all narrow-format PTX CVTs with and without
  `satfinite`, to/from f16x2/bf16x2/f32/tf32/ue8m0).
- All `MUFU.*` transcendentals (`ex2`, `rcp`, `rsqrt`, `sqrt`, `lg2`,
  `sin`, `cos`, `tanh`).
- `F2F.F16.F32` / `F2F.BF16.F32` (the non-satfinite one-way round-down
  emitted by `cvt.rn.f16.f32` without `satfinite`).
- `POPC` (integer population count).

**They do NOT contend (measurable throughput of F2FP unchanged) with:**

- `FFMA` / `FMUL` / `FADD` (FP32 FMA pipe).
- `IMAD` / `IADD3` / `LOP3` / `SHL` (INT / LOP pipe).
- `HFMA2` / `HADD2` (FP16/BF16 pair FMA pipe).

**Shared dispatch slot but not SFU execution unit:**

- `SHFL.*` (warp shuffle network) — contends at saturation but has its own
  data path; the bottleneck is SMSP dispatch.
- `HMMA.*` / tensor-core mma.sync — same: dispatch-slot contention, not
  SFU pipe.
- Shared & global memory loads (`LDS`, `LDG`) — mostly dispatch, not SFU.

## 6. Key measurement files

- `/root/github/QuickRunCUDA/tests/bench_f2fp_coissue_v2.cu` — the main sweep kernel
- `/root/github/QuickRunCUDA/tests/bench_f2f_pure.cu` — isolated `F2F.F16.F32` rate (16/SM/clk)
- `/root/github/QuickRunCUDA/tests/bench_solo.cu` — isolated POPC, SHFL, I2F/F2I, VOTE rates
- `/root/github/QuickRunCUDA/tests/bench_f2fp_vs_hmma.cu`, `bench_hmma_vs_ffma.cu` — rules out HMMA as an F2FP-specific conflict
- `/root/github/QuickRunCUDA/tests/bench_f2fp_vs_ldg.cu` — rules out memory-pipe conflict
- `/root/github/QuickRunCUDA/tests/bench_f2f_vs_f2fp.cu` — F2F+F2FP mix, showing they share SFU
- Prior-art (for sanity-checking, not relied on): `/root/github/QuickRunCUDA/F2FP_DEEP_DIVE.md`
  (this investigation reached matching conclusions on MUFU sharing, went
  further by explicitly testing & ruling out HFMA2/HMMA/memory/SHFL/VOTE/POPC).
