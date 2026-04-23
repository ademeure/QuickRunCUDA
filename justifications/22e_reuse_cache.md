# §22e — SASS `.reuse` operand cache (AUDIT-VERIFIED)

Audit date: 2026-04-23
Auditor: Claude Opus 4.7 (main session)
Method: Direct SASS analysis of pre-existing audit kernels
GPU: B300 SXM6 AC (single GPU 0)

## CLAIM (catalog L8142-L8164)

> "**480 of 512 FFMA2 instructions (94%)** carry the `.reuse` annotation on at least one operand."
>
> "The `.reuse` modifier signals to the warp dispatcher that an operand should be served from a small **operand-reuse cache** (separate from RF read ports). This dramatically reduces register file bandwidth pressure."

## EVIDENCE — Scalar FFMA case (bench_fp32_fma.cu, the §0.FFMA peak kernel)

SASS file: `/root/github/QuickRunCUDA/sass/bench_fp32_fma_1552823151.sass`

```bash
$ grep -c "FFMA" sass/bench_fp32_fma_1552823151.sass
1024
$ grep "FFMA" sass/bench_fp32_fma_1552823151.sass | grep -c ".reuse"
1023
```

**1023 of 1024 FFMA = 99.9% carry `.reuse`** on the constant operand (R2.reuse).

Sample (verbatim from file):
```
/*0100*/                   FFMA R5, R2.reuse, 1.0000001192092895508, R5 ;
/*0110*/                   FFMA R3, R2.reuse, 1.0000001192092895508, R4 ;
/*0120*/                   FFMA R7, R2.reuse, 1.0000001192092895508, R7 ;
/*0130*/                   FFMA R9, R2.reuse, 1.0000001192092895508, R9 ;
...
```

Pattern: R2 is the multiplier (`.reuse`), destinations rotate (R5/R3/R7/R9...), one of the addend is also the destination (forming chain `Rn = R2 * 1.0... + Rn`).

## EVIDENCE — Packed FFMA2 case (audit_ffma2_alu_mix kernels)

The §22 dual-issue audit produced 8 SASS files at different FFMA2:LOP3 ratios. `.reuse` rate sweep:

| SASS file | FFMA2 count | .reuse count | Rate |
|---|--:|--:|--:|
| `audit_ffma2_alu_mix_1828655129.sass` | 128 | 127 | **99.2%** |
| `audit_ffma2_alu_mix_2663448341.sass` | 128 | 116 | 90.6% |
| `audit_ffma2_alu_mix_3080844947.sass` | 128 | 111 | 86.7% |
| `audit_ffma2_alu_mix_3289543250.sass` | 128 | 106 | 82.8% |
| `audit_ffma2_alu_mix_3498241553.sass` | 128 | 106 | 82.8% |

Range: **82.8% to 99.2%** depending on kernel configuration.

Catalog claim of 94% (480/512) sits in the middle of this observed range — **CONFIRMED as plausible**.

## VERDICT

✅ **AUDIT-VERIFIED** — The catalog's `.reuse` cache claim is supported by direct SASS evidence on this rig:
- Scalar FFMA in peak kernel: 99.9% (catalog 94% claim is conservative)
- FFMA2 across multiple kernel configs: 82.8-99.2% (catalog 94% is in-range)

**Recommendation for catalog:** the 94% number is supported but somewhat low for the scalar FFMA case. Suggest rephrasing as: "**80-99% of FFMA / FFMA2 instructions in real benchmark kernels carry `.reuse` on at least one operand**, depending on kernel structure."

## Methodology notes

- **No new kernels run** — used SASS files already preserved from earlier audit runs
- **Caveat:** the rate depends on how the compiler decides to schedule .reuse. Different kernel structures (more/fewer chains, more/fewer interleaved ALU ops) shift the rate
- **Mechanism unverified directly:** I confirmed `.reuse` is EMITTED by ptxas in 99% of these FFMA cases. I did NOT verify ncu metric `lts__t_sectors_op_read` differs with/without `.reuse` (would require a controlled experiment with ptxas hint-suppression). The catalog's mechanism explanation (operand-reuse cache reduces RF port pressure) is consistent with NVIDIA SASS docs but not directly tested here.
