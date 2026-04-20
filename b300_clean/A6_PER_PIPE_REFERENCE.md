# Per-pipe instruction throughput + dispatch model — V4 / A6

**Date: 2026-04-20.** 1500 MHz clock-locked, persistent grid (1 block/SM
× 256 threads = 8 warps/SM = 2 warps/SMSP), 8 chains × 16 unroll,
`asm volatile` for anti-DCE. pkill+sleep 6 between measurements.

## Per-instruction throughput table (single-op-type sweep)

All numbers measured directly on B300 SXM6 (sm_103a). Clock is LOCKED
at 1500 MHz (not 2032 boost) — scale by 2032/1500 = 1.355× for boost.

| Instruction | TIPS_inst (@ 1500) | inst/SMSP/cy | Pipe   | Commit  |
|-------------|-------------------:|-------------:|--------|---------|
| FFMA        | 18.87              | 0.66         | FMA (unified) | 22b06b3 |
| IADD3       | 14.13              | 0.50         | ALU (unified) | 22b06b3 |
| LOP3.LUT    | 14.16              | 0.50         | ALU (unified) | 22b06b3 |
| SHF.L/R     | 14.12              | 0.50         | ALU    | this md |
| PRMT        | 14.08              | 0.50         | ALU    | this md |
| BFI         | 13.15              | 0.46         | ALU    | this md |
| BFE         | 7.07               | 0.25         | XU     | this md |
| SHFL.IDX    | 7.06               | 0.25         | LSU/SHFL | this md |
| BREV        | 3.54               | 0.125        | XU     | this md |
| POPC        | 3.54               | 0.125        | XU     | this md |
| CLZ / FLO   | 3.53               | 0.125        | XU     | this md |
| MUFU.EX2    | 9.62 Gops/s        | 0.003        | MUFU   | 5f70214 |
| VABSDIFF2   | emulated (~9 inst) | —            | N/A    | — |

**Convert to boost (2032 MHz)**: multiply TIPS_inst by 1.355.
E.g. LOP3 = 14.16 × 1.355 = **19.18 TIPS_inst at 2032 MHz**.

## Inferred pipe structure (per SMSP)

Each SMSP can dispatch **1 warp-instruction per cycle** (single issue
port). The instruction then enters one of the pipes:

- **FMA/INT unified cluster** (ratio 0.5/SMSP/cy for INT, 1.0 for FFMA
  theoretical but we hit 0.66 at 2 warps/SMSP — needs 4+ warps):
  - FFMA, IMAD, IADD3, LOP3, SHF, PRMT, BFI
  - All share one dispatch port on the unified cluster
  - Confirmed via Jarmusch et al. arXiv:2507.10789 (sm_120, likely
    applies to sm_103a given similar architecture).

- **XU / bit-manip pipe** (ratio 0.125/SMSP/cy = 8 cy per inst):
  - BREV, POPC, CLZ / FLO
  - These are 4× slower per-inst than LOP3 → the XU pipe is narrower.

- **LSU / SHFL pipe** (ratio 0.25/SMSP/cy = 4 cy per inst):
  - SHFL.IDX (warp shuffle via cross-lane)

- **MUFU / SFU pipe** (slow):
  - EX2, LOG2, SIN, COS, RCP, RSQRT

## Dispatch model: mixed-pipe overlap measurements

How much does mixing pipes reduce wall-clock? Method: run 2 chains
(one per op type) in same loop, compare to sum-of-isolated.

| Mix            | T_a (ms) | T_b (ms) | Sum   | Measured | Overlap |
|----------------|---------:|---------:|------:|---------:|--------:|
| FFMA + IADD3   | 1.607    | 2.146    | 3.753 | 3.219    | **14.2%** |
| FFMA + SHFL    | 1.608    | 4.294    | 5.902 | 5.034    | **14.7%** |
| FFMA + MUFU    | —        | ~12.6    | —     | ~12.6    | **~100%** |

**Interpretation:**
- FFMA + IADD3 (both unified cluster): limited overlap (14.2%), since
  they compete for the SMSP issue port AND the unified ALU/FMA pipe.
- FFMA + SHFL (different pipes): surprisingly little overlap (14.7%).
  Possibility: SHFL occupies the SMSP issue port for multiple cycles
  during cross-lane routing, blocking FFMA issue.
- FFMA + MUFU (previous commit `8012b98`): essentially 100% overlap —
  MUFU issues quickly then runs long in background, leaving issue port
  free for FFMA.

**Key insight**: **issue-port pressure, not pipe diversity, determines
co-issue potential.** If one op occupies the port for only 1 cy after
entering the pipe, FFMA can fill adjacent cycles freely. MUFU fits
this model. SHFL does not (possibly multi-cycle dispatch).

## Why FFMA only hits 0.66/SMSP/cy, not 1.0

With 2 warps/SMSP (1 persistent block of 256 threads/SM) and FFMA
4-cy latency chain, each warp's achievable rate is bounded by
chain-latency. At NC=8, ILP = 8/4 cy = 2 inst/cy/warp max, but SMSP
issue port = 1/cy/SMSP = shared across 2 warps = 0.5/cy/warp. Two
warps alternating gives 1.0/SMSP/cy ideally; we measure 0.66/SMSP/cy
(33% bubble).

Existing `04_fp32_peak.md` shows the cure: **need 4+ warps/SMSP
(block_size ≥ 512) to get 98% of peak**. At 2 warps, warp-scheduler
alternation can't cleanly issue every cycle.

## Confidence

- **HIGH** for all per-inst TIPS at 1500 MHz (3+ trials, stable)
- **HIGH** for inst/SMSP/cy inference (cross-checked with theoretical)
- **HIGH** for unified-cluster model (matches Jarmusch paper + local
  `04_fp32_peak.md` + mixed overlap results)
- **MED** for pipe-occupancy explanation of SHFL vs MUFU overlap
  asymmetry — need separate direct test (SHFL pipe-slot duration
  measurement).

## Files

- `tests/bench_lop3_lut_sweep.cu` — LOP3 imm sweep
- `tests/bench_lop3_port_pressure.cu` — RF port pressure  
- `tests/bench_iadd3_throughput.cu` — IADD3
- `tests/bench_dual_pipe_ffma_iadd3.cu` — FFMA+IADD3 mix
- `tests/bench_int_pipes_sweep.cu` — BREV/POPC/CLZ/BFE/SHF/PRMT/SHFL/BFI/VABSDIFF2
- `tests/bench_ffma_shfl_dual.cu` — FFMA+SHFL mix
