# SASS Binary Modification for four_six_fp4 on SM103 (B300)

## Summary

Post-compilation binary patching of NVIDIA GPU SASS (native assembly) to remove
redundant instructions that the compiler generates but can't optimize away.

**Result: NC=2 (two-candidate quantization) improved from 5.25 → 5.64 TB/s (+7.5%)**

## The Redundant Instructions

The `four_six_fp4` kernel quantizes BF16→NVFP4 (e2m1). The compiler generates
`F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C` to convert float→e2m1 and pack the
result into a register. When the merge operand C=RZ (zero register), the upper
bytes of the output are guaranteed zero.

Despite this, the compiler inserts `LOP3.LUT Rdst, Rsrc, 0xff, RZ, 0xc0`
(Rdst = Rsrc & 0xFF) after each PACK — a byte-mask that is mathematically
redundant since the upper bytes are already zero. These serve as register copies
(Rdst ≠ Rsrc in 25/32 cases) to free the PACK's destination register.

## Proof of Redundancy

Three methods independently verified (all produce bit-identical output):

1. **Identity mask**: Change immediate from `0xff` to `0xffffffff`
   (Rdst = Rsrc & 0xFFFFFFFF = Rsrc). Same LOP3 instruction, wider mask.

2. **LUT passthrough**: Change LUT from `0xc0` (A&B) to `0xf0` (passthrough A).
   The mask immediate becomes irrelevant.

3. **Register rename + NOP**: NOP the LOP3, redirect the downstream
   `F2FP.F16.E2M1.UNPACK_B` consumer to read from the original source
   register directly.

## The Patching Pipeline

### Step 1: Compile and save the original cubin
```bash
./QuickRunCUDA tests/four_six_fp4.cu -H "..." -t 128 -b N ...
cp output.cubin original.cubin
```

### Step 2: Identify targets
```bash
cuobjdump --dump-sass original.cubin | grep "LOP3.*0xff,"
```
Each target has: `LOP3.LUT Rdst, Rsrc, 0xff, RZ, 0xc0` at a known SASS offset.

### Step 3: For each target, find the consumer
The consumer is always `F2FP.F16.E2M1.UNPACK_B Rout, Rdst` — the instruction
that reads the LOP3's output to convert the byte back to half2 for error
computation.

**Safety check**: verify Rsrc (the LOP3's source, = PACK_AB output) is not
overwritten between the PACK and the UNPACK. If overwritten, the rename is
unsafe (6 of 32 in the GPT=2 build).

### Step 4: Binary patch
For each safe target:
1. **NOP the LOP3**: Replace bytes [0:8] with NOP encoding
   (`0x1879000000000000`), clear bytes [8:10] (operand/LUT fields).
2. **Rename the UNPACK source**: Change byte 4 of the UNPACK instruction
   from Rdst to Rsrc.

### Step 5: Delete the NOPs
Shift all subsequent instructions up by 16 bytes. Pad the freed space at the
end with trailing NOPs.

**Caveats**:
- Some deletions break scoreboard timing (instruction-position-sensitive
  stall counts). Test each deletion individually.
- The BRA self-loop at the end uses relative offset — self-referencing loops
  survive shifting unchanged.
- Register reuse cache flags (bits 2-4 of control word byte 15) reference
  the preceding instruction's register reads. Shifting changes the predecessor,
  potentially using stale cached values. In practice, most deletions work
  because the reuse flags are conservative.

## Results

### GPT=1 (1 group/thread, 233 SASS instructions)
| Config | Instructions | TB/s | vs baseline |
|--------|-------------|------|-------------|
| Original | 233 | 4.94 | — |
| 14 LOP3→NOP+rename | 233 (14 NOP) | 5.08 | +2.7% |
| 14 LOP3 deleted | 219 | 5.29 | +7.1% |

### GPT=2 (2 groups/thread, 465 SASS instructions, stride1K, mbps=8)
| Config | Removed | TB/s | vs baseline |
|--------|---------|------|-------------|
| Original | 0 | 5.25 | — |
| 26 NOP+rename | 0 deleted | 5.30 | +0.9% |
| 22 deleted + 4 NOP | 22 | **5.64** | **+7.5%** |

### ncu Pipeline Shift (GPT=2 NC=2)
| Metric | Before | After |
|--------|--------|-------|
| ALU pipe | 69% | 64% |
| FMA pipe | 46% | 50% |
| DRAM throughput | 68% | 73% |

## Encoding Reference (SM103)

Each SASS instruction is 128 bits (16 bytes):
- Bytes 0-7: opcode + operands ("hi8" in LE)
- Bytes 8-15: control word ("lo8" in LE)

### LOP3.LUT encoding
```
hi8: [opcode:2=0x7812] [Rdst:1] [Rsrc_a:1] [immediate:4]
lo8: [Rsrc_c:1] [LUT:1] [flags:6]
```
- LUT 0xc0 = A & B
- LUT 0xf0 = passthrough A

### NOP encoding
```
hi8: 0x1879000000000000
lo8: 0x000fc00000000000 (or any valid control word)
```

### F2FP.UNPACK_B encoding
```
hi8: [opcode:2=0x723e] [Rdst:1] [modifier:1] [Rsrc:1] [flags:3]
```
Byte 4 = source register. Change this for register rename.

## Files

- `rename_lop3.py`: Automated NOP+rename+delete pipeline
- `patch_lop3.py`: Simpler NOP-only or identity-mask patcher
- `SASS_MODIFICATION.md`: This document

## Why adj_mbps0 Doesn't Tolerate Deletions

The deletion safety depends on the **control word scheduling byte** (lo[5]):

| Config | Regs | lo[5] pattern | Deletable |
|--------|------|---------------|-----------|
| stride1K_mbps8 | 64 | uniform 0xe2 | 22/26 |
| adj_mbps0 | 47 | mixed 0xe2/0xe4/0xc6 | 0/30 |

**Explanation**: Lower register count → compiler must pack instructions
more tightly → position-sensitive dependency chains in the control word.
When an instruction is deleted and subsequent code shifts, the control
word's dependency reference (which implicitly targets the preceding
instruction) now points to a different instruction, breaking the
scoreboard timing.

The 64-reg build has "scheduling slack" — uniform control words that
tolerate position changes. The 47-reg build has no slack — every
instruction's position is critical for the dependency chain.

**Implication**: For SASS deletion to work, prefer builds with higher
register counts (e.g., `MIN_BLOCKS_PER_SM=8` with 128 threads) that
generate looser, position-tolerant scheduling.

## SM103 Pipe Architecture: f16x2 SIMD is ALU, NOT a Separate Pipe

Tested using ncu `sm__inst_executed_pipe_fp16` → returns n/a.

HFMA2, HMUL2, HADD2 all run on the **ALU pipe** alongside LOP3, SHF, PRMT, IMAD.
This means moving computation from f32 FMA (FMUL2/FFMA2, on FMA pipe at 47%)
to f16x2 (HFMA2, on ALU pipe at 70%) HURTS performance.

**Correct strategy**: keep computation in f32 (FMA pipe has headroom), minimize
ALU work (LOP3/SHF byte manipulations are the real bottleneck).

## SM103 Pipe Map (Confirmed via ncu Microbenchmarks)

| Instruction | Pipe | Throughput | ncu Evidence |
|---|---|---|---|
| FMUL, FFMA, FMUL2, FFMA2, FHFMA, IMAD | **FMA** | ~2× ALU | 30-31% in isolation |
| LOP3, SHF, PRMT, FSETP, FSEL, IADD3, HADD | **ALU** | ~1× | 65% in isolation |
| HFMA2, HMUL2, HADD2, HMNMX2 | **ALU** | ~1× | 65% ALU (NOT a separate f16 pipe) |
| F2FP (all CVT variants) | **F2FP** (own pipe) | ? | ~0% on ALU/FMA |
| FMNMX, FMNMX3 | **F2FP** (own pipe) | ? | ~0% on ALU/FMA |
| MUFU (rcp, rsqrt) | **XU** | 1/4× | Standard |

Key architecture insight: **FMA pipe has ~2× the throughput of ALU on SM103.**
With 99 ALU ops (70% utilized) and 179 FMA ops (47% utilized), the kernel is
ALU-throughput limited despite having fewer ALU instructions.

This means:
- f32 FMA path is optimal (FHFMA, FMUL2, FFMA2 on high-throughput FMA pipe)
- f16x2 SIMD (HFMA2/HMUL2) HURTS because it runs on the low-throughput ALU pipe
- Moving work from ALU→FMA is valuable; moving from FMA→ALU is harmful

## bf16 Input is Better Than f16 for ALU-Bound Kernels

bf16→f32: `shl 16` (IMAD on FMA pipe) + `and 0xffff0000` (LOP3 on ALU) = 1 ALU + 1 FMA per pair
f16→f32:  `HADD2.F32` (ALU pipe) × 2 per pair = 2 ALU per pair

bf16 format is just f32 with lower mantissa bits zeroed → pure bit operation that uses the
high-throughput FMA pipe (via IMAD × 0x10000). f16→f32 requires actual exponent remapping
which runs on the low-throughput ALU pipe via HADD2.F32.

For ALU-bound kernels: bf16 input ≈ 50% less ALU for extraction vs f16 input.

## Final NC=2 Results (6.04 TB/s, 85% of peak)

All optimizations combined:
- CONTIGUOUS_SCALES: eliminates ~24 ALU swizzle ops, drops regs 63→44
- Fused quant_dequant_fused: keeps .b8 in asm, eliminates 24 LOP3
- Combined scale constants: folds 2 sequential FMULs into 1
- s_round==0 check removed: eliminates 4 FSETP + 4 FSEL
- Interleaved NC=2 candidates: FFMA2 cross-candidate error accumulation

SASS: 358 instructions, 44 regs, 78 ALU at 69%, 162 FMA at 48%, DRAM at 78%

Best config: `-H "#define CONTIGUOUS_SCALES #define MIN_BLOCKS_PER_SM 0"` -t 64

Approaches that DON'T help further:
- bf16x2 absmax: adds 9 ALU without removing any (extraction still needed)
- f16x2 error path: HFMA2 runs on ALU, not separate pipe
- f16 input: HADD2.F32 for f16→f32 is all-ALU (worse than bf16's IMAD+LOP3 split)
- SASS deletion: adj_m0 (44 regs) doesn't tolerate position shifts
- Pack-only-winner: compiler generates same total work either way
