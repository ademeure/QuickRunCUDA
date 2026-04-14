# Report #8 — NCU hardware metrics reveal the real pipe structure

Using `ncu --metrics sm__inst_executed_pipe_*.sum` directly on isolated
single-op kernels gives the **hardware ground truth** for which pipe each
SASS instruction uses. Runs: 4,850,848 dynamic instructions each (with
148 SMs × 128 threads × 8192×16 ops per thread = 4.85M total).

## Raw NCU pipe-attribution table

| Kernel | pipe_alu | pipe_fma(lite) | pipe_fmaheavy | pipe_xu | pipe_tensor |
|---|---:|---:|---:|---:|---:|
| FFMA | 76,960 | **4,850,848** | 592 | 0 | 0 |
| EX2 (MUFU) | 152,736 | 152,736 | 592 | **4,849,664** | 0 |
| RSQ (MUFU) | **4,850,848** | **9,700,512** (2×) | 592 | **4,849,664** | 0 |
| F2FP unpack | **4,850,848** | 76,368 | 76,368 | 0 | 0 |
| F2FP pack | **9,700,512** (2×) | 592 | 592 | 0 | 0 |

## Pipe-per-instruction decomposition

| PTX | SASS | Pipes used | Ops per pipe per PTX inst |
|---|---|---|---|
| `fma.rn.f32` | `FFMA` | `fmalite` | 1 |
| `ex2.approx.f32` | `MUFU.EX2` | `xu` | 1 |
| `rsqrt.approx.f32` | `MUFU.RSQ` | `xu` + `alu` + `fmalite` | 1 + 1 + 2 |
| `cvt.rn.f16x2.e4m3x2` (unpack) | `F2FP.F16.E4M3.UNPACK_B` | `alu` | 1 |
| `cvt.rn.satfinite.e4m3x2.f16x2` (pack) | `F2FP.*.UNPACK_B_MERGE_C` | `alu` | **2** |

**F2FP is an ALU-pipe op**, not an XU-pipe op. My earlier "2-slot SFU" model
was a functional approximation — the real structure is that F2FP competes
for **ALU pipe** cycles, and pack costs 2 ALU cycles per issue while unpack
costs 1.

**MUFU.RSQ is a compound operation** — each RSQ instruction dispatches work
to *four* pipe stages: XU (lookup), ALU, and two FMA-lite stages (final
refinement math). That's why RSQ runs at 16/SM/clk: 2 FMA-lite issues per
RSQ × 32 F2FP/ALU peak = bottleneck at 32/clk ÷ 2 FMA-lite-ops = 16/clk.

**MUFU.EX2 is pure XU** with ~no ALU use — that's why it's the fastest MUFU
and doesn't contend with F2FP the same way RSQ does (F2FP alu + EX2 xu =
parallel).

## Reconciling with my earlier observations

I previously thought:
- F2FP, MUFU all share "2-slot SFU pipe"
- Read-port count in F2FP determines dual-issue

NCU now shows:
- F2FP → `pipe_alu` (NOT `pipe_xu`)
- MUFU.EX2 → `pipe_xu`, MUFU.RSQ → `pipe_xu` + `pipe_alu` + 2× `pipe_fmalite`
- Pack = 2 ALU issues per PTX inst (matches the "2 read ports" rule — they
  show up as 2 pipe_alu instructions because each occupies 2 cycles of the
  ALU pipe)

**Unified explanation:** pipe_alu has 2 issue slots per cycle per SM. An
unpack takes 1 ALU issue → 2 unpacks/cycle = 64/SM/clk. A pack takes 2 ALU
issues → 1 pack/cycle = 32/SM/clk. This is the same "1 read port / 2 read
ports" model, but the unit being counted is ALU pipe slots, not some abstract
SFU slot.

## Re-explaining the contention results

- **F2FP + EX2**: F2FP on ALU, EX2 on XU — different pipes. But at N=32 EX2
  per iter, the SCHEDULER can't dispatch both in parallel fast enough. That's
  dispatch-port contention, not pipe execution.
- **F2FP + RSQ**: both use ALU pipe (RSQ uses 1 ALU per issue). So RSQ
  competes for ALU cycles — direct pipe sharing. Stronger contention.
- **F2FP + FFMA**: FFMA on `fmalite`, F2FP on `alu`. Different pipes. Some
  contention at saturation = dispatch-port.
- **F2FP + LOP3/IADD3**: LOP3/IADD3 on `pipe_adu` or `pipe_alu`... need to
  recheck. Actually my measurement showed LOP3 barely touches F2FP. Let me
  re-check with NCU.

## Open question: where do LOP3 / IADD3 live?

Per my cross-matrix Report #6, LOP3/IADD3 barely affect F2FP. If F2FP is on
`pipe_alu` and LOP3 were also on `pipe_alu`, they'd contend. Let me check
their NCU pipe attribution to finalize the model.

## Practical takeaway update

The bottom-line throughputs stand unchanged:
- F2FP unpack: 64 PTX-inst/SM/clk, 128 elements/SM/clk
- F2FP pack: 32 PTX-inst/SM/clk, 64 elements/SM/clk
- MUFU.EX2: 32/SM/clk
- MUFU.RSQ/SIN/...: 16/SM/clk

But the *why* is now clearer:
- **Pack counts as 2 ALU issues** (the "2nd read port" of MERGE_C / PACK_AB)
- **RSQ counts as 4 pipe issues across 3 pipes** (expensive to mix with F2FP)
- **EX2 counts as 1 XU issue** (cheap from ALU's perspective)

## Update — LOP3/IADD3/IMAD pipe attribution (DCE-resistant kernels)

Re-ran with loop-variant inputs so ptxas can't optimize away:

| Kernel | pipe_alu | pipe_fmaheavy | pipe_fmalite | pipe_xu |
|---|---:|---:|---:|---:|
| **LOP3** (xor.b32) | **3.71M** | **3.56M** | 0 | 0 |
| **IADD3** (add.u32) | **4.78M** | 152K | 0 | 0 |
| **IMAD** (mad.lo.u32) | **4.85M** | **4.85M** | 592 | 0 |

- LOP3 uses `pipe_alu` + `pipe_fmaheavy` (both at ~1× each).
- IMAD uses `pipe_alu` + `pipe_fmaheavy` at 1× each (heavy compound op).
- IADD3 is primarily `pipe_alu`, with minimal `pipe_fmaheavy`.

This complicates the pure "F2FP → pipe_alu" story:
- F2FP unpack uses pipe_alu 1×
- LOP3 uses pipe_alu 1× + fmaheavy 1×
- IADD3 uses pipe_alu 1×
- IMAD uses pipe_alu 1× + fmaheavy 1×

If pipe_alu were a simple resource, F2FP+LOP3 should contend (both at pipe_alu
saturation rate). But my cross-matrix showed F2FP+LOP3 has only mild (~10%)
drop. So pipe_alu must have either:
1. Multiple execution lanes (e.g., 4-wide alu vs 2-wide scheduler slot),
2. Different "types" of alu ops that don't serialize,
3. pipe_alu counts instructions that touch the register file at all, not
   actual execution slot usage.

Option 3 is most consistent: pipe_alu might be the "dispatch" counter (every
op that uses an ALU input touches it), while **functional execution happens
on the separate fmalite/fmaheavy/xu pipes**. Only ops that EXECUTE on
pipe_alu (like F2FP) compete for the actual ALU functional unit.

## Final model (as refined by NCU metrics)

Each SM has these execution units:
- **pipe_fmalite** — FP32 FMA (1 per cycle per SMSP × 4 = 4/SM/cycle, but 32-wide SIMT = 128 thread-ops/SM/cycle)
- **pipe_fmaheavy** — integer FMA / LOP3 / heavy int (similar rate)
- **pipe_alu** — F2FP, many housekeeping ops; pack costs 2× unpack here
- **pipe_xu** — MUFU (EX2 at 32/SM/clk, RSQ compound at 16)
- **pipe_tensor** — HMMA/BMMA
- **pipe_lsu** — memory
- **pipe_adu** — branches/address compute

And NCU counts are "instructions flowing to that pipe" which can be larger
than 1× for compound ops (RSQ, pack) that send work to multiple pipes.
