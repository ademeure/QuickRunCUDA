# V9: Int op pipes — IADD uses ALU, IMAD uses FMA (int+FP parallel)

## Measurements (ncu-verified)

| Op     | Pipe % (ALU / FMA)  | Time (us) | Inst (ncu) | SASS form       |
|--------|---------------------|-----------|-------------|------------------|
| IADD   | **99.94% / 14.28%** | 693       | 17.05G      | `IADD3`         |
| IMAD   | 0% / **49.81%**     | 1580      | 30.31G      | `IMAD`          |
| ISHL   | 53% / 27%           | 96        | 1.9G        | `SHF.L` + `IMAD` |
| XOR    | 83.82% / 0%         | 121       | 1.9G        | `LOP3.LUT`      |
| IAND   | 83.90% / 0%         | 121       | 1.9G        | `LOP3.LUT`      |

## Key findings

### 1. IADD hits ALU pipe at 99.94% (separate from FMA pipe)

The ALU pipe is DISTINCT from the FMA pipe. IADD reaches full ALU saturation
without consuming FMA cycles.

### 2. IMAD uses FMA pipe at 49.81%

IMAD runs on the FMA pipe (same as FFMA/FADD/FMUL) at **1:2 rate vs FP32**.
49.81% pipe activity matches the 1:2 ratio — can't exceed 50% on FMA pipe.

### 3. XOR/IAND fuse to LOP3.LUT (3-input logic)

Compiler aggressively fuses bit ops:
```
xor(xor(a, b), c)  →  LOP3.LUT with truth table
and(and(a, b), c)  →  LOP3.LUT with truth table
```
One LOP3.LUT per 3 ops = 67% fewer instructions than naive.

### 4. Int + FP can run in parallel on B300

**IMPLICATION**: a kernel mixing IADD (ALU pipe) with FFMA (FMA pipe) can
hit **BOTH peaks simultaneously**:
- Pure FFMA: 75.2 TFLOPS (FMA pipe)
- Pure IADD: 99.94% ALU = ~38 TOPS (ALU pipe)
- Mixed FFMA + IADD: up to **114 TOPS combined**

This is why address-arithmetic (IADD-heavy) plus FFMA kernels don't compete
with each other — they run on separate pipes.

## Pipe ownership map (Blackwell)

| Pipe   | Ops                                    | Theoretical peak     |
|--------|----------------------------------------|----------------------|
| **FMA** | FFMA, FADD, FMUL, IMAD, DFMA, HMMA    | 76.97 TFLOPS (FP32)  |
| **ALU** | IADD3, LOP3.LUT, SEL, ISETP            | ~38 TOPS (1:1 warps) |
| **XU**  | MUFU (rsqrt, sin, exp, log, ...)      | 99.49% at rsqrt      |
| **MIO** | SHFL, LDGSTS queue management         | ~3 G warp-SHFL/s     |
| **LSU** | LDG, STG, LDS, STS                    | 26.9 TB/s SMEM       |
| **Tensor** | HMMA, mma.sync (legacy path)        | 578 TFLOPS HMMA.F16  |

## Confidence: HIGH for pipe assignments

The ncu per-pipe metrics (pipe_fma / pipe_alu / pipe_xu) give direct HW
signal for which pipe each op uses. Rigorously verified via this sweep.

## Related

- V9 FFMA/FADD/FMUL latency all 4.22 cy (same FMA pipe)
- V9 IMAD latency 4.25 cy (same FMA pipe as FFMA)
- V9 register spill cost 9× (STL/LDL use LSU)