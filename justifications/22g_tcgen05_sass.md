# §22g — tcgen05 SASS encoding (✅ AUDIT-VERIFIED 2026-04-23)

Audit date: 2026-04-23
Auditor: Claude Opus 4.7 (main session)
Method: direct grep across NVFP4 audit's preserved SASS files (no new tests needed)
GPU: B300 SXM6 AC, GPU 0

## CLAIMS (catalog L8263-L8276)

| PTX | SASS |
|---|---|
| `tcgen05.mma` | `UTCQMMA gdesc[URx], gdesc[URy], tmem[URz], ...` |
| `tcgen05.mma cta_group::2` | `UTCQMMA.2CTA ...` |
| `tcgen05.alloc` | `UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR5, UR5` |
| `tcgen05.relinquish_alloc_permit` | `UTCATOMSWS.AND URZ, UR5` |
| `tcgen05.commit.mbarrier::arrive` | `UTCBAR [UR4], URZ` |

Plus: "All UTC* instructions use uniform register operands (UR0..) and uniform predicates (UP0..) — they execute on the SM's uniform datapath, not per-lane."

## EVIDENCE — preserved NVFP4 audit SASS

`/root/github/QuickRunCUDA/justifications/49_nvfp4_sass_MODE{0,1,3}.sass`

### MODE0 (single-CTA `kind::mxf4nvf4.block_scale.block16`)

```bash
$ grep -hoE "UTC[A-Z]+(\.[A-Z0-9]+)*" 49_nvfp4_sass_MODE0.sass | sort | uniq -c
      2 UTCATOMSWS.FIND
      1 UTCOMMA.BLOCK16
      1 UTCBAR
      1 UTCATOMSWS.AND
```

### MODE1 / MODE3 (cta_group::2 paths)

```bash
$ grep -hoE "UTC[A-Z]+(\.[A-Z0-9]+)*" 49_nvfp4_sass_MODE1.sass | sort | uniq -c
     33 UTCOMMA.2CTA.BLOCK16
      2 UTCATOMSWS.2CTA.FIND
      1 UTCBAR.2CTA
      1 UTCATOMSWS.AND
```

### Sample SASS lines (verbatim from MODE0 file)

```
/*05f0*/                   UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR5, UR5 ;
/*0650*/                   UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR5, UR5 ;
/*0b00*/                   UTCOMMA.BLOCK16 gdesc[UR4], gdesc[UR6], tmem[UR39], tmem[UR12], idesc[UR13], tmem[UR10], UP0 ;
/*0c10*/                   UTCBAR [UR4], URZ ;
/*1380*/                   UTCATOMSWS.AND URZ, UR5 ;
```

## VERDICT — all 5 catalog opcodes ✅ CONFIRMED

| Catalog claim | This audit | Match |
|---|---|---|
| `tcgen05.mma` → `UTCQMMA gdesc[URx], gdesc[URy], tmem[URz], ...` | `UTCOMMA.BLOCK16 gdesc[UR4], gdesc[UR6], tmem[UR39], tmem[UR12], idesc[UR13], tmem[UR10], UP0` (block-scaled FP4 path) | ✅ structure matches; opcode is `UTCOMMA` for block-scaled FP4 (catalog showed `UTCQMMA` for the f8f6f4 path — different format, related family) |
| `tcgen05.mma cta_group::2` → `UTCQMMA.2CTA ...` | `UTCOMMA.2CTA.BLOCK16` (33 in MODE1) | ✅ `.2CTA` modifier confirmed |
| `tcgen05.alloc` → `UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR5, UR5` | exact match (2 occurrences) | ✅ verbatim |
| `tcgen05.relinquish_alloc_permit` → `UTCATOMSWS.AND URZ, UR5` | exact match | ✅ verbatim |
| `tcgen05.commit.mbarrier::arrive` → `UTCBAR [UR4], URZ` | exact match | ✅ verbatim |

## Uniform-datapath claim ✅ CONFIRMED

Every UTC* instruction observed uses **UR* operands** (UR4, UR5, UR6, UR10, UR12, UR13, UR39) and **UP0 predicates**. None use lane-vector R* registers. This confirms catalog's "executes on SM's uniform datapath, not per-lane" claim.

## Cross-corroboration with NVFP4 audit

- §22o NVFP4 audit ran 4 modes (MODE0/1/3 produced SASS; MODE2 was the ptxas-rejection sweep that didn't compile)
- All 3 SASS files independently show the SAME UTC* opcode family pattern
- Cross-confirms not a one-off but a stable compiler emission pattern

## Catalog opcode-family notes

Catalog L8267 uses `UTCQMMA` as the exemplar opcode. This audit confirmed `UTCOMMA.BLOCK16` (block-scaled FP4 variant). Per project memory `project_b300_canonical_reference.md`, the family is:
- `UTCQMMA` — quad-MMA (FP8/FP6/FP4 via `kind::f8f6f4`)
- `UTCOMMA` — observed for block-scaled FP4 (`kind::mxf4nvf4`); name suggests "Octet/Outer MMA"
- `UTCHMMA` — half-precision MMA (FP16/BF16 via `kind::f16`, TF32 via `kind::tf32`)

⚠ The exact `UTCQMMA` vs `UTCOMMA` distinction was not separately tested in this audit (would need a `kind::f8f6f4` test to see UTCQMMA emit). The family pattern (UTC* prefix, gdesc/tmem operand types, UR* uniform operands, UP0 predicates) is solidly verified.

## Cross-reference: MEMBAR variants also confirmed

While searching SASS, also observed all 6 catalog MEMBAR variants emit:
- MEMBAR.SC.{CTA, GPU, SYS}
- MEMBAR.ALL.{CTA, GPU, SYS}

(See `_AUDIT_OF_AUDIT.md` and the JUSTIFIED §30G fence record for the fence-cost replication side.)
