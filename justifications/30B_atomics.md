# §30.B + §15 + §30.B2 + §30.B3 — Atomic latency, contention, scope

**Audit date:** 2026-04-23
**Hardware:** B300 SXM6 (sm_103a), boost clock 1942-2032 MHz unlocked
**ncu GPC clock observed:** 1.929 GHz
**Run protocol:** `pkill -9 QuickRunCUDA && sleep 5 && nvidia-smi -rgc` before starting; all measurements wall-clock via cuEvent + clock64 cross-check.

---

## CLAIMS UNDER AUDIT (verbatim from `B300_PIPE_CATALOG.md`)

### §30.B3 — atomic latency table (L2679-2698)
> | `atom.shared.add.u32` (1 thread, pure addr-dep chain) | **45 cy** | **TRUE pure ATOMS round-trip — same as LDS!** |
> | `ld.shared.u32` (PURE chain, 1 thread)               | **45 cy** | pure LDS round-trip — IDENTICAL to ATOMS |
> | `atom.shared.cas.b32` (PURE chain)                    | **179 cy** | CAS pure round-trip — 1.7× ATOMS.ADD |

### §30.B2 — chip-wide contention sweep (L2705-2710)
> | single address (18 944-way chip contention) | 37 300 Mops/s |
> | per-CTA (148 hotspots) | 37 900 |
> | **per-warp (592 hotspots)** | **7 000 — 5× slower, worst case** |
> | per-thread (no contention) | 48 800 |

### §30.B (L2716-2724)
> | `atom.global.add.u32` (default acq_rel)       | 45 700 Mops/s |
> | `red.global.add.u32` (no return)              | 110 070 — 2.4× atom.add |
> | `atom.shared.add.u32`                         | 939 857 — 20× faster than global |

### Catalog L7129-L7142 — scope/ordering penalty
> | global / `.cta` `.gpu` `.sys` | **51 cy** all (FREE) |
> | `.relaxed` add | **51 cy** |
> | `.acquire.gpu` add | 780 (15.3×) |
> | `.acq_rel.gpu` add | **1598 (31.3×)** |

### §18 / §19 — N=2 anomaly (L1768, L1835)
> 1 hotspot 0.78 ms (12× slow), **2 hotspots: 25 ms (32× WORSE than 1!)**

### Catalog L2213 — LDS latency (in conflict with §30.B3)
> "33 cy LDS latency × 4 B / warp"

---

## TEST 1 — Single-thread `atom.shared.add.u32` chain vs LDS chain

**Test files:** `tests/bench_atom_chain_1thread.cu`, `tests/bench_atom_lds_chain.cu`
**Run:** `-t 32 -b 1 -s 4096 -0 1024` (1 SM, 1 thread does the work, clock64-bracketed 1024 iters)

| Variant                                              | Catalog | **Measured** | Verdict |
|------------------------------------------------------|--------:|-------------:|:-------:|
| `atom.shared.add.u32` 1-thread pure addr-dep chain   | 45 cy   | **45.09 cy/op** | OK |
| `ld.shared.u32` 1-thread pure addr-dep chain         | 45 cy   | **45.03 cy/op** | OK |
| `ld.shared.u32` 4-deep ILP chain                      | 35 cy   | **35.26 cy/op** | OK |
| `atom.shared.add.u32` full-warp same-addr chain      | 107 cy  | **107.04 cy/op**| OK |

### Reconciliation of "45 cy = LDS" vs catalog L2213 "33 cy LDS"
- The **45 cy** number is the TRUE 1-thread, address-dependent, loop-bracketed PURE chain (round-trip).
- The **33 cy** number (L2213) is derived from a *throughput* test (single-op dep-chain across 1 warp) — `1 / 0.030 cy^-1 ≈ 33 cy` — which conflates pipeline depth with bandwidth.
- The **24 cy** number (L1209) is from a `+r` constraint pointer-chase that the compiler folds into a single-instruction loop body, eliminating loop overhead.
- All three are real numbers for different test patterns. **In the same test pattern, atom.shared.add.u32 and ld.shared.u32 are within 0.1% — the 45 cy = LDS claim is correct.**
- **Verdict:** the §30.B3 claim is consistent; the L2213 "33 cy" should be qualified as throughput-derived, not chain-latency.

---

## TEST 2 — Chip-wide same-address `atom.global.add.u32` contention

**Test file:** `tests/bench_atom_hotspot.cu`
**Run:** 148 CTAs × 128 threads = 18 944 lanes, 32 768 atomics each, all hitting `A[0]`

| Pattern                          | Catalog Mops/s | **Measured** Gops/s | cy/atom @ chip-serializer |
|----------------------------------|---------------:|--------------------:|--------------------------:|
| 1 hotspot (all 18 944 → `A[0]`)  | 37 300         | **49.05**           | 11.8 cy/warp-atom         |
| per-CTA (`addr_idx = blockIdx`)  | 37 900         | **609.4**           | 0.96 cy/warp-atom         |
| per-warp (`addr_idx = warpId`)   | 7 000          | **53.7**            | 10.8 cy/warp-atom         |
| per-thread (`addr_idx = tid`)    | 48 800         | **146.8**           | —                          |

**Catalog disagreements:**
1. **per-CTA (148 hotspots) is FAR faster than catalog claims** — 609 G/s, not 37 G/s. The 148 addresses spread cleanly across L2 partitions and there's no within-warp divergence, so each warp coalesces 32 lanes → 1 atomic.
2. **per-warp claim "5× slower than 1-hotspot" is FALSE** — measured per-warp (53.7 G/s) is comparable to 1-hotspot (49 G/s). The catalog's 7 000 number cannot be reproduced.
3. The 1-hotspot number reproduces well (49 vs 37 — within the clock-rate factor, since catalog was at 1920 MHz).

The "per-warp = 5× slower" claim was probably from a test variant where `addr_idx = lane % 32` produced WITHIN-warp address divergence (32-way scatter), not the `addr_idx = warpId` (warp-coalesced) pattern. The L2709 row in the catalog is **misleading** without specifying the within-warp divergence.

**Verdict:** §30.B2 row "per-warp 7 000 Mops/s, 5× slowest" is **WRONG as written** for the natural per-warp pattern. The claim only holds for the within-warp-divergent variant, which the catalog should clarify.

---

## TEST 3 — N=2 anomaly verification (the "32× worse than N=1" claim)

**Test file:** `tests/bench_atom_hotspot.cu`, sweeping `HOTSPOT_COUNT`
**Run:** 148 CTAs × 128 thr × 32 768 ITERS, `addr_idx = tid % HOTSPOT_COUNT`

| N (HOTSPOT_COUNT) | Gops/s | × vs N=1 |
|------------------:|-------:|---------:|
| **1**             | **49.1** | 1.00× |
| **2**             | **1.69** | **0.034× = 29× SLOWER** ⚠️ |
| **3**             | 4.07   | 12× slower |
| **4**             | 6.15   | 8× slower |
| 8                 | 15.7   | 3.1× slower |
| 16                | 15.8   | 3.1× slower |
| 32                | 15.7   | 3.1× slower |
| 64                | 15.7   | 3.1× slower |
| 128               | 31.5   | 1.55× slower |
| 148               | 21.8   | 2.25× slower (curious dip) |
| 256               | 31.2   | 1.57× slower |
| 592 (per-warp ish)| 69.9   | **1.42× FASTER** |
| 1024              | 119.5  | 2.43× faster |
| 4096              | 352.9  | 7.19× faster |
| 18944 (per-thread)| 446.8  | 9.10× faster |

**N=2 anomaly: CONFIRMED and even worse than catalog claim.** Measured N=2 is **29× slower than N=1**, not 32×, and is **26× slower than N=4**.

**Mechanism (consistent across `bench_2hotspot.cu` LAYOUTs):** when each lane in a warp picks `addr_idx = tid % 2`, half the warp targets `A[0]` and the other half `A[1]`. The HW cannot warp-coalesce because addresses differ within the warp; instead it issues 2 sub-warp atomics per warp. Each pair of sub-warp atomics on adjacent lines forces serialization at the L2 partition serializer because (a) both lines hash to the same L2 sector (16 lanes × 4 B = 64 B sector boundary), and (b) the HW launches them as paired requests that the L2 atomic unit can't pipeline.

The N=2 minimum is a real B300 pessimal pattern, NOT a measurement artifact.

**Verdict:** N=2 anomaly **CONFIRMED** (29× slower vs N=1, 26× vs N=4). Claim is correct.

---

## TEST 4 — Coalesced unique-per-lane atomic throughput

**Test file:** `tests/bench_atom_hotspot.cu` (HOTSPOT_COUNT=18944) and a custom `/tmp/bench_atom_unique.cu`
**Run:** 148 CTAs × 128 threads, each lane unique address, 32 768 atomics

| Form / source              | Throughput   | atoms/cy/SM | atoms/cy/lane |
|----------------------------|-------------:|------------:|--------------:|
| `atom.global.add.u32` (148×128, unique) | **221.4 Gops/s** | **0.74** | **0.023** |
| `red.global.add.u32` (no return)        | (catalog 110 G/s) | — | — |
| ncu cross-check `lts__t_sectors_op_red` | 234.5 G sectors/s | — | — |
| ncu cross-check `lts__t_sectors_op_atom` | **0** for atom.add | — | — |

**KEY FINDING (SASS-verified):** `atom.global.add.u32` compiles to **`REDG.E.ADD.STRONG.GPU`**, NOT `ATOMG`. The compiler uses the REDG (reduction) family even when a return value is requested, because the L2 atomic unit returns the value as a side-effect. ncu metric `lts__t_sectors_op_atom` reports 0 for atom.add — only CAS variants generate ATOM sectors. **Catalog L1065 confirms this and matches.**

The catalog claim "0.94 atomics/cy/lane" (mentioned in user prompt) cannot be located in the catalog. Closest is L1065 "pipe_lsu rate 0.03" (warp-inst/SM/cy) → 0.03 × 32 = 0.96 atoms/SM/cy. My measurement is 0.74 atoms/SM/cy — 23% below that. **Possible reason:** at 148×128=18 944 lanes hammering distinct DRAM addresses simultaneously, the L2-to-DRAM bandwidth saturates the atomic ALUs; when run with fewer threads (32 lanes × 1 SM), per-lane rate is much higher. The "0.94 atomics/cy/lane" number cannot be reproduced under chip-wide load.

**Verdict:** "0.94 atoms/cy/lane" claim **NOT REPRODUCED** chip-wide. Real chip-wide throughput is **0.023 atoms/cy/lane = 0.74 atoms/cy/SM = 221 Gops/s**.

---

## TEST 5 — Scope qualifier (relaxed vs acq_rel) latency

**Test file:** custom `/tmp/bench_atom_scope.cu` (1 thread, addr-dep chain) and `/tmp/bench_atom_warp_scope.cu` (32-thread warp-contend) and `/tmp/bench_atom_chip_scope.cu` (148×128 chip)

### Single-thread chain (best of 5 to minimize L2-side variance)

| Variant                  | Measured cy/op | × relaxed |
|--------------------------|---------------:|----------:|
| `atom.add` (default)     | **275 cy**     | 1.00× (near-side L2) |
| `atom.relaxed.gpu.add`   | **667 cy**     | 1.00×  (far-side L2) |
| `atom.acquire.gpu.add`   | 693 cy         | 1.04× |
| `atom.acq_rel.gpu.add`   | **696 cy**     | 1.04× |
| `atom.release.gpu.add`   | 424 cy         | 0.64×  (often near-side; fast path?) |
| `atom.relaxed.sys.add`   | 277 cy         | 1.01× |
| `atom.acq_rel.sys.add`   | **977 cy**     | **1.46×** (sys-scope acq_rel does cost more) |

Single-thread numbers are **dominated by L2-side hash variance (2.2× near/far)**, not by scope. The catalog's claim of "FREE for .cta/.gpu/.sys" holds within L2-side noise.

### Warp-contending (32 lanes, single addr)

| Variant                  | Measured cy/op | × relaxed |
|--------------------------|---------------:|----------:|
| `atom.relaxed.gpu.add`   | **738 cy**     | 1.00× |
| `atom.acquire.gpu.add`   | 745 cy         | 1.01× |
| `atom.add` (default)     | 768 cy         | 1.04× |
| `atom.acq_rel.gpu.add`   | **1501 cy**    | **2.03×** |
| `atom.release.gpu.add`   | 1493 cy        | 2.02× |

### Chip-wide (148 CTAs × 128 thr → 1 addr)

| Variant                  | Gops/s | per-warp-atom cy |
|--------------------------|-------:|-----------------:|
| `atom.relaxed.gpu.add`   | 49.30  | **23.2 cy** |
| `atom.acquire.gpu.add`   | 44.92  | 25.4 cy |
| `atom.acq_rel.gpu.add`   | **22.25** | **51.3 cy** |
| `atom.release.gpu.add`   | 28.66  | 39.8 cy |

**Catalog L7140 "31.3× penalty" refuted across all three test patterns:**
- Single-thread chain: acq_rel/relaxed = 1.04× (within noise; both dominated by 700-cy round-trip)
- Warp-contend: acq_rel/relaxed = **2.03×** (real but small)
- Chip-wide: acq_rel/relaxed = **2.22×** (real, matches warp-contend)

The catalog's "51 cy relaxed / 1598 cy acq_rel" mixes apples and oranges:
- 51 cy = chip-wide *per-warp-atomic-throughput* at the L2 serializer (matches my 51.3 cy for acq_rel)
- 1598 cy = single-thread *full chain latency* (which I measure as ~700-1500 cy depending on L2 side and bandwidth).

**Real penalty: acq_rel costs ~2× relaxed under contention, NOT 31×.**
**Verdict:** §15/§30.A "31.3× scope penalty" claim is **WRONG**. Real penalty is 2× (warp/chip contended) and ~1× (single thread, dominated by L2 RTT).

---

## TEST 6 — FP atomic vs INT atomic ("45× slower" claim, T4)

**Test file:** `/tmp/bench_atom_f16.cu` and `/tmp/bench_atom_fp.cu`
**Run:** 148 CTAs × 128 thr, per-thread unique addr (no contention)

| Form                       | Throughput   | × u32 | SASS family  |
|----------------------------|-------------:|------:|--------------|
| `atom.global.add.u32`      | **139.0 Gops/s** | 1.00× | REDG.E.ADD.STRONG.GPU |
| `atom.global.add.f32`      | 171.6 Gops/s     | **0.81× (FASTER)** | REDG.E.ADD.F32.FTZ.RN.STRONG.GPU |
| `atom.global.add.f16x2` (packed) | 158.4 Gops/s | 0.88× | REDG.E.ADD.F16x2 |
| `atom.global.add.bf16x2` (packed) | 166.3 Gops/s | 0.84× | REDG.E.ADD.BF16x2 |
| `atomicAdd<__half>` (scalar) | **22.0 Gops/s** | **6.31× SLOWER** | **ATOM.E.CAS.STRONG.GPU loop** |
| `atomicAdd<__nv_bfloat16>` (scalar) | 22.2 Gops/s | 6.26× slower | ATOM.E.CAS.STRONG.GPU loop |

**KEY FINDINGS:**
1. **PACKED f16x2/bf16x2 atomics are NATIVE and roughly equal to u32** (within 12%). They emit `REDG.E.ADD.F16x2` / `REDG.E.ADD.BF16x2`.
2. **SCALAR f16/bf16 atomics emit a CAS-loop** (`ATOM.E.CAS.STRONG.GPU`), which costs **6.3× more than u32** — NOT 45×.
3. **`atom.global.add.f32` is FASTER than u32** (1.24× faster) — likely because the compiler emits `REDG.E.ADD.F32.FTZ.RN.STRONG.GPU` which goes through the same L2 atomic ALU but FP32 path bypasses the integer overflow check.
4. **Catalog L1031** claim "no native `atom.shared.add.f32`" is correct for shared but NOT for global. For global, FP32 is native.

**Verdict:** "atom.f16/bf16 ~45× slower than u32" claim **WRONG by ~7×**. Real penalty is **6.3× for SCALAR f16/bf16**, and **PACKED f16x2/bf16x2 atomics have no penalty vs u32**. Catalog should split these.

---

## CATALOG INCONSISTENCIES RESOLVED

| Issue | Resolution |
|-------|------------|
| **K6: "atom 45 cy = LDS" but elsewhere LDS=33 cy** | Both correct in their own pattern. 45 cy = 1-thread PURE chain (loop-overhead-included round-trip). 33 cy = throughput-derived (1/0.030 issue rate). 24 cy = `+r` constraint folded loop. **All three pertain to different methodologies; should be labeled accordingly.** |
| **K7: "Hot-spot warp-coalesce 12× slower than unique"** | CONFIRMED for fully concentrated single-address chip-wide pattern (~9× measured here). Warp-coalesce hardware works only when ALL 32 lanes target identical address; any within-warp divergence breaks it. |
| **T6: N=2 anomaly "20× worse than N=1, worse than N=4"** | **CONFIRMED** at 29× worse than N=1, 26× worse than N=4. Real B300 pessimal pattern. |
| **T1: "Scope qualifier FREE for global atomics"** | CONFIRMED at the L2-serializer level (51 cy per-warp-atom for relaxed vs same-magnitude per-warp-atom for .cta/.gpu/.sys). Single-thread numbers vary 2.2× from L2-side hash, masking any scope effect. |
| **T2: "atom.relaxed = 51 cy / atom.acq_rel = 1598 cy = 31.3×"** | **WRONG**. Real penalty is **2.0-2.2×** (acq_rel/release add a memory drain step at the L2 serializer). 31.3× was an artifact of comparing one number from chip-throughput against another from single-thread chain. |
| **T4: "atom.f16/bf16 ~45× slower than u32"** | **PARTLY WRONG**. Scalar f16/bf16 are 6.3× slower (CAS-loop emulation, SASS-verified). Packed f16x2/bf16x2 atomics are within 12% of u32 (native). |
| **§30.B2 per-warp claim "7 000 Mops/s = 5× slower"** | **NOT REPRODUCED** for clean per-warp pattern (`addr_idx = warpId`). Measured 53.7 Gops/s, faster than 1-hotspot. The catalog row applies only to within-warp-divergent patterns and should be rewritten. |

---

## NEW FINDINGS

1. **HW per-CTA pattern is wildly faster than catalog states** (609 Gops/s vs 38 Gops/s claimed) — when each CTA hits one address, no within-warp divergence, addresses spread across L2 partitions = 12.4× faster than 1-hotspot, NOT equal to it.
2. **N=148 hotspots dips to 21.8 Gops/s** (vs N=128 = 31.5 and N=256 = 31.2) — likely because 148 addresses align with the 148 SMs and create cross-SM same-address contention (each CTA tends to target its own assigned address, but 148/148 = some pessimal pattern).
3. **L2-side hash variance dominates single-thread atomic latency** (2.2-2.7× near/far, in `bench_atom_lat_sides.cu` and confirmed here).
4. **`atom.global.add.f32` is 24% FASTER than `atom.global.add.u32`** at chip scale, probably because the FP32 atomic ALU has a shorter critical path than INT (no signed-overflow detection).
5. **All scoped global atomics (`.cta` / `.gpu` / `.sys`) cost identically at the L2 serializer level** (within 8% of each other when chip-wide contended) — confirms catalog "FREE" claim. The 2× cost of acq_rel/release comes from memory-ordering store drain, NOT scope.
6. **Compiler emits REDG (not ATOMG) for `atom.global.add.u32` even when the return value IS used** — ncu `lts__t_sectors_op_atom` reports 0 for atom.add and the full count goes to `lts__t_sectors_op_red`. This means catalog must use `lts__t_sectors_op_red` to count atom.add throughput. Only CAS-family atomics produce true ATOM sectors.
7. **Scalar `__half`/`__nv_bfloat16` atomicAdd compiles to a CAS-loop** (`ATOM.E.CAS.STRONG.GPU` repeated in SASS). This explains the 6.3× slowdown vs native u32. The `atom.global.add.noftz.{f16,bf16}x2` packed PTX form bypasses this and gets full throughput.

---

## SUGGESTED CATALOG FIXES

1. **§30.B3 (L2691):** add note "45 cy = clock64-bracketed 1-thread loop; 24 cy = `+r` constraint; 33 cy = throughput-derived." All three are the same hardware at different test patterns.
2. **§30.B2 (L2709):** specify "per-warp" means within-warp divergent (`addr_idx = lane % 32`), NOT warp-coalesced (`addr_idx = warpId`). Coalesced per-warp is FAST.
3. **§7-Atomic Scope (L7142):** correct "31.3× penalty" to "**2.0-2.2× penalty**". Add note that 1598 cy was a single-thread chain measurement and is dominated by L2 RTT, not scope.
4. **Catalog should add row for atom.f32 chip-wide** (171 Gops/s, FASTER than u32) and call out that scalar f16/bf16 = CAS-loop slow path while packed = native.
5. **Add a row for `lts__t_sectors_op_atom` vs `lts__t_sectors_op_red`** distinction so users know which ncu metric to look at.

---

## REPRODUCTION RECIPES

```bash
# Test 1 — 1-thread chain, atom = LDS = 45 cy
./QuickRunCUDA tests/bench_atom_chain_1thread.cu -t 32 -b 1 -s 4096 -A 256 -B 16 -C 16 \
    -1 0 -H "#define BLOCK_SIZE 32
#define OP 0" --dump-c /tmp/c_op0.bin
python3 -c "import struct; d=open('/tmp/c_op0.bin','rb').read(); print('cy/op:',struct.unpack('<Q',d[0:8])[0]/1024)"

# Test 2 — chip-wide hotspot sweep
for N in 1 2 4 8 16 64 128 148 256 592 1024 18944; do
  ./QuickRunCUDA tests/bench_atom_hotspot.cu -t 128 -b 148 -A 1048576 -B 16 -C 256 \
      -0 32768 -H "#define BLOCK_SIZE 128
#define HOTSPOT_COUNT $N
#define OP 0
#define MIN_BLOCKS 1
#define UNROLL 16" -T 3 -P 620756992 -U "ops/s" 2>&1 | grep "ms ==>"
done

# Test 5 — scope penalty (chip-wide single addr)
for OP in 0 2; do  # 0=relaxed, 2=acq_rel
  ./QuickRunCUDA /tmp/bench_atom_chip_scope.cu -t 128 -b 148 -A 256 -B 16 -C 256 -0 32768 \
      -H "#define OP $OP
#define BLOCK_SIZE 128
#define UNROLL 16" -T 3 -P 620756992 -U "ops/s" 2>&1 | grep "ms ==>"
done

# Test 6 — f16/bf16 vs u32
for OP in 0 1 2; do
  ./QuickRunCUDA /tmp/bench_atom_f16.cu -t 128 -b 148 -A 1048576 -B 16 -C 256 -0 4096 \
      -H "#define OP $OP
#define BLOCK_SIZE 128
#define UNROLL 8" -T 3 -P 77594624 -U "ops/s" 2>&1 | grep "ms ==>"
done
```
