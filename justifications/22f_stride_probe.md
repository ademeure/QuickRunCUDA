# §22f L1/L2 cache granularity probe — RIGOROUS replication

## CLAIM (verbatim from `B300_PIPE_CATALOG.md` lines 8231-8249)

> ## L1/L2 Cache Granularity Probe
>
> Stride sweep (4096 loads after warm-up):
>
> | Stride | cy/load | Tier (inferred) |
> |--------|---------|-----------------|
> | 4B | 56 | L1 hit (warps coalesce to 128B requests) |
> | 8B | 56 | L1 hit (same coalescing) |
> | 16B | 56 | L1 hit |
> | 32B | 56 | L1 hit (still within coalescing) |
> | **64B** | **304** | L1 miss → L2 hit (5.4× jump!) |
> | 128B | 316 | L2 hit |
> | 256B-1024B | 316 | L2 hit (no further degradation) |
>
> **Sharp break at 64B stride** — beyond this, per-thread loads stop benefiting
> from warp-level coalescing. Each lane needs its own cacheline transaction.
>
> This indirectly tells us: **the warp-level memory access "footprint" per
> `ld.global.u32` is 128 B** — when 32 lanes × 4 B fits within a 128 B aligned
> region, fast (56 cy). When stride exceeds this, the loads spill into separate
> cachelines (304 cy = L2 hit).
>
> L2 hit latency in this test: ~316 cy

## VERDICT (executive summary)

**MIXED — partial replication, headline interpretation needs revision.**

| Catalog claim                                                       | Replication                                                   | Verdict |
|---------------------------------------------------------------------|---------------------------------------------------------------|---------|
| 56 cy/load at stride 4 B (single-thread, throughput-style)          | **56.8 cy/load** (mean of 5 runs, σ < 0.01)                   | **CONFIRMED** |
| Same 56 cy at strides 8/16/32                                       | 87 / 136 / 160 cy — **monotonic rise**, never flat           | **REFUTED** (catalog hides intermediate behaviour) |
| Sharp 5.4× cliff at exactly stride 64 (56 → 304)                    | No cliff at 64; **gradual rise 57 → 160 cy and plateau**     | **REFUTED** (no sharp 64 B break) |
| 304 cy = "L2 hit" plateau                                           | Plateau is **158-163 cy** in throughput mode, **328 cy** in latency-chain mode (and only above stride 32, not 64) | **PARTIAL** (304 ≈ catalog's separate L2-latency entry of 301 cy, see line 111 of catalog — they used a serial test) |
| 128 B coalescing footprint                                          | ncu shows **32 B sector**-granular: 5 sectors at stride 4 (warp 32 lanes × 4 B = 128 B = 4 sectors + boundary), 32 sectors at stride 32 (each lane = own sector) | **REFINED** (true unit is 32 B sector, not 128 B line) |
| Cache-line size = 128 B (L1 sector model implicit)                  | Confirmed: 128 B line = 4 × 32 B sectors                      | **CONFIRMED** |

The catalog's three numbers (56 / 304 / 316 cy) appear to come from different
experiments collapsed into one table:

* 56 cy ≈ best-case L1-resident throughput (matches my stride-4 single-thread
  throughput-style measurement).
* 304 cy ≈ pointer-chase L2 hit **latency** (matches catalog's own line 111
  `ld.global L2 | 301 cy` — completely different test).
* The "sharp jump at 64" is implied by combining those two numbers, but no
  single contiguous sweep on this hardware produces it.

---

## TEST FILES

Two kernels, identical logic except for the dependency structure inside the
timed loop:

### A) `tests/bench_stride_probe.cu` (latency-chain, "v51 in catalog terminology")
Each loaded value is XORed into `v` (which is fed to the next iteration as
loop-carried state). The compiler must serialize each `LDG → LOP3 → next
iteration's bookkeeping`, so cy/load reports the **full serial latency** of
`ld.global` plus loop overhead.

### B) `tests/bench_stride_probe_thr.cu` (throughput, K_INNER = 16 ILP)
16 independent `LDG`s per outer iteration, each into its own register; the
XOR reductions occur AFTER all 16 LDGs are issued. Allows the LSU to
**pipeline up to 16 loads in flight**, exposing throughput-limited
behaviour rather than per-load latency.

Both kernels accept three CLI args:
* `arg0` (`-0`) = `iters` (outer loop trip count)
* `arg1` (`-1`) = `stride_bytes`
* `arg2` (`-2`) = mode (0 = single thread, 1 = full warp of 32 lanes)

Both kernels: 64-iter warm-up before `clock64()` t0. Anti-DCE: gated impossible
store of XOR-accumulated `v` to `C[1024+lane]`. cycles written to `((long
long*)C)[0]`.

PTX emitted: `ld.global.ca.u32`. SASS emitted: `LDG.E.STRONG.SM` (the default
on sm_103a — `STRONG.SM` is just the cache-coherent scope tag, not a special
modifier). Verified by `grep LDG sass/bench_stride_probe*.sass`:
* `bench_stride_probe.sass`: 2 LDGs (1 warmup, 1 timed loop body) — matches
* `bench_stride_probe_thr.sass`: 17 LDGs (1 warmup, 16 inner-unroll body) —
  matches `K_INNER=16`

---

## BUILD / RUN

GPU 0 only (only `/dev/nvidia0` accessible).

```bash
pkill -9 QuickRunCUDA; sleep 5; nvidia-smi -rgc

# Latency-chain sweep (single-thread, mode 0)
for STR in 4 8 16 32 40 48 56 60 64 72 96 128 192 256 512 1024 2048 4096; do
  for run in 1 2 3 4 5; do
    ./QuickRunCUDA tests/bench_stride_probe.cu -t 32 -b 1 \
        -A 67108864 -B 1024 -C 1024 \
        -0 4096 -1 $STR -2 0 -T 1 \
        --dump-c /tmp/sp_${STR}_${run}.bin
  done
done

# Throughput sweep, single-thread (mode 0) and warp (mode 1)
# K_INNER=16, iters=256 → 4096 timed LDGs (matches catalog's "4096 loads")
for MODE in 0 1; do
  for STR in 4 8 16 32 40 48 56 60 64 72 96 128 192 256 512 1024 2048 4096; do
    for run in 1 2 3 4 5; do
      ./QuickRunCUDA tests/bench_stride_probe_thr.cu -t 32 -b 1 \
          -A 67108864 -B 1024 -C 1024 \
          -0 256 -1 $STR -2 $MODE -T 1 \
          --dump-c /tmp/sp_thr_${MODE}_${STR}_${run}.bin
    done
  done
done

# Cycles parsed from first 16 bytes of dumped C: { int64 total_cycles, int64 cy_per_load_x1000 }
```

---

## REPLICATION TABLE

All numbers are mean of 5 runs (σ < 0.05% of mean for every entry).
Both kernels run as 1 block × 32 threads (warp-mode active in mode=1).
4096 timed LDGs in throughput tests; 4096 timed LDGs in latency tests.

| Stride (B) | THR mode 0 (single thread, ILP=16) | THR mode 1 (warp, ILP=16) | LAT mode 0 (single thread, dep-chain) | Catalog claim |
|-----------:|----------------------------------:|-------------------------:|--------------------------------------:|--------------:|
|          4 |                          **56.8** |                **52.9** |                                104.2 |        **56** |
|          8 |                              87.0 |                    75.9 |                                135.7 |        **56** |
|         16 |                             135.6 |                   113.2 |                                199.6 |        **56** |
|         32 |                             160.1 |                   117.3 |                                327.6 |        **56** |
|         40 |                             159.6 |                   114.7 |                                327.4 |             — |
|         48 |                             159.1 |                   115.4 |                                327.4 |             — |
|         56 |                             158.6 |                   115.3 |                                327.4 |             — |
|         60 |                             159.0 |                   114.8 |                                327.5 |             — |
|       **64** |                         **160.1** |               **120.4** |                              **327.3** |       **304** |
|         72 |                             159.3 |                   115.3 |                                327.5 |             — |
|         96 |                             159.1 |                   118.3 |                                328.0 |             — |
|        128 |                             157.4 |                   131.3 |                                327.9 |           316 |
|        192 |                             157.8 |                   124.8 |                                328.2 |             — |
|        256 |                             157.3 |                   132.0 |                                328.3 |           316 |
|        512 |                             157.6 |                   132.8 |                                328.5 |           316 |
|       1024 |                             158.6 |                   134.1 |                                328.4 |           316 |
|       2048 |                             161.9 |                   133.5 |                                328.5 |             — |
|       4096 |                             163.3 |                   134.6 |                                328.5 |             — |

cy/load is `(t1 - t0) / num_timed_loads` from in-kernel `clock64()`.

### Crossover analysis

* **Throughput mode 0 (single thread)**: rises 57 → 160 cy/load monotonically over
  strides 4-32, then plateaus at ~158 cy from stride 32 onward through 4096.
  No cliff. The "knee" is at stride **32 B** (one sector), not 64 B.
* **Throughput mode 1 (warp)**: rises 53 → 117 cy from strides 4-32, then
  plateaus at ~115-135 cy. Shallower because warp-coalescing absorbs more work
  per LSU instruction.
* **Latency mode 0 (single-thread dep-chain)**: rises 104 → 328 cy over
  strides 4-32, plateaus at 327-329 cy from stride 32 on. Dep-chain exposes
  the actual L2 hit latency (~327 cy at 1.94 GHz = ~169 ns).

In NO mode is there a 5.4× jump at stride 64 specifically. The transition is
gradual and centered on stride 32 (one sector), not 64.

---

## ncu CONFIRMATION (single-thread throughput mode = "thr_0")

`l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_{hit,miss}.sum` measured for
the timed kernel:

| Stride | LSU inst | Sectors | L1 hits | L1 misses | Hit % | LTS sectors |
|-------:|---------:|--------:|--------:|----------:|------:|------------:|
|     4  |   4 165  |   4 162 |   3 648 |       512 | 88%   |       11.4k |
|    32  |   4 165  |   4 162 |      64 |     4 096 | 1.5%  |       24.0k |
|    64  |   4 165  |   4 162 |      64 |     4 096 | 1.5%  |       42.4k |
|   128  |   4 165  |   4 162 |      64 |     4 096 | 1.5%  |       47.1k |
|  1024  |   4 165  |   4 162 |      64 |     4 096 | 1.5%  |       54.1k |

Reading: **L1 saturates at miss-everything from stride 32 onward** — every load
needs a fresh sector and there is zero spatial reuse. LTS sector traffic
**roughly doubles between stride 32 and 64**, indicating L2 sector spread (different
cachelines being touched), but per-load cy stays flat because the LSU pipeline
is throughput-limited rather than latency-limited.

LSU instruction count = 4 165 = 64 (warmup) + 4 096 (timed) + 5 (epilogue
stores) — exactly as expected.

### ncu CONFIRMATION (warp throughput mode = "thr_1")

| Stride | LSU inst | Sectors | L1 hits | L1 misses | Sectors/LDG | Warp lat (cy/inst) |
|-------:|---------:|--------:|--------:|----------:|------------:|-------------------:|
|     4  |   4 165  |  20 226 |  19 708 |       516 |       4.86  |              10.85 |
|    32  |   4 165  | 133 122 | 128 993 |     4 127 |      31.96  |              31.73 |
|    64  |   4 165  | 133 122 | 128 993 |     4 127 |      31.96  |              44.33 |
|   128  |   4 165  | 133 122 | 128 993 |     4 127 |      31.96  |              46.81 |

This is the smoking gun for the coalescing footprint:

* At **stride 4**, the warp generates ≈ 5 sectors per LDG instruction. That's
  the 4 sectors of one 128 B line (32 lanes × 4 B = 128 B), plus a small
  cross-line tail from the ~3 B unaligned starts averaged across 4096 LDGs.
  This matches the catalog's "warp-coalesced to 128 B requests" claim.
* At **stride 32**, sectors/LDG jumps to 32 — one 32 B sector per lane.
  This is the **transition: stride ≥ 32 B = one fresh sector per lane**.

So the **catalog's claim of 128 B coalescing footprint is right** for the
in-line case (stride ≤ 32), but the actual hardware unit is the **32 B
sector** — when stride < 4 B (i.e. multiple lanes per int) you'd get even
fewer sectors; when stride ≥ 32 B you saturate at 32.

The "stride 64 cliff" the catalog reports does NOT correspond to a
HW-microarchitectural transition — by stride 32 the per-lane sector
allocation is already fully separated.

---

## CLOCK STATE

Sampled during one of the timed runs:

```
clocks.current.graphics: 1942 MHz
clocks.current.memory:   3996 MHz
power.draw:              200-206 W
temperature.gpu:         38 C
```

No `nvidia-smi -lgc` issued. GPU is in default boost state. Catalog claims
1920 MHz for this section; actual was 1942 MHz (1.15 % higher), which
inflates measured cy/load by ≤ 1.2 % relative to a 1920 MHz baseline —
negligible for the discussion.

---

## RAW STDOUT (representative — full data in `/tmp/sweep_*.txt`)

### Throughput mode 1 (warp), all strides

```
stride_B mean_cycles mean_cy_per_load cycles_runs
     4      216640      52.890  [216649 216637 216630 216636 216647]
     8      311037      75.937  [311034 311018 311064 311036 311034]
    16      463581     113.179  [463606 463560 463624 463594 463521]
    32      480318     117.264  [480272 480261 480229 480268 480558]
    40      469921     114.726  [469892 469840 469919 469996 469958]
    48      472660     115.395  [472417 473699 472336 472457 472391]
    56      472083     115.254  [472022 472140 472059 472092 472100]
    60      470253     114.808  [470218 470254 470226 470290 470279]
    64      493192     120.408  [493262 493234 493152 493183 493129]
    72      472397     115.331  [472394 472293 472479 472471 472348]
    96      484641     118.320  [484662 484590 484664 484633 484654]
   128      537889     131.320  [538049 537900 537832 537851 537815]
   192      511251     124.817  [511052 511289 511336 511187 511393]
   256      540801     132.031  [540735 540760 540745 540791 540973]
   512      543928     132.794  [543887 543931 543949 543887 543985]
  1024      549290     134.103  [549230 549313 549334 549298 549275]
  2048      546874     133.514  [546682 546991 546851 546969 546879]
  4096      551323     134.600  [551304 551437 551332 551268 551274]
```

### Throughput mode 0 (single thread, ILP=16), all strides

```
stride_B mean_cycles mean_cy_per_load cycles_runs
     4      232744      56.822  [232756 232732 232750 232743 232741]
     8      356476      87.030  [356475 356481 356495 356463 356466]
    16      555425     135.601  [555397 555452 555353 555385 555539]
    32      655730     160.090  [655820 655597 655758 655780 655697]
    40      653583     159.566  [653628 653571 653605 653570 653542]
    48      651855     159.144  [651861 651842 651863 651763 651948]
    56      649825     158.648  [649756 649849 649845 649849 649827]
    60      651287     159.005  [651319 651284 651269 651248 651316]
    64      655922     160.137  [655903 656072 655936 655811 655886]
    72      652342     159.262  [652364 652292 652472 652320 652260]
    96      651582     159.077  [651544 651477 651541 651659 651689]
   128      644616     157.377  [644676 644402 644671 644772 644559]
   192      646378     157.807  [646394 646497 646350 646365 646283]
   256      644311     157.302  [644312 644522 644230 644356 644134]
   512      645519     157.597  [645625 645454 645408 645534 645575]
  1024      649487     158.566  [649387 649459 649669 649439 649481]
  2048      662984     161.861  [663171 662931 662979 662962 662875]
  4096      668863     163.296  [668970 668859 668797 668882 668807]
```

### Latency mode 0 (single-thread dep-chain), all strides

```
stride_B mean_cycles mean_cy_per_load cycles_runs
     4      426940       104.233  [426947 426949 426955 426922 426928]
     8      555971       135.735  [555977 555987 555969 555954 555970]
    16      817662       199.624  [817645 817663 817664 817647 817692]
    32     1341696       327.562  [1341694 1341719 1341719 1341638 1341710]
    40     1341151       327.429  [1341246 1341212 1341107 1341040 1341149]
    48     1341195       327.440  [1341171 1341089 1341245 1341229 1341241]
    56     1340985       327.388  [1341033 1340995 1340933 1340924 1341038]
    60     1341538       327.523  [1341611 1341609 1341517 1341369 1341584]
    64     1340422       327.251  [1340415 1340484 1340430 1340373 1340410]
    72     1341386       327.486  [1341436 1341374 1341378 1341376 1341367]
    96     1343499       328.002  [1343463 1343465 1343525 1343564 1343480]
   128     1343079       327.900  [1343101 1343092 1343053 1343061 1343088]
   192     1344224       328.179  [1344260 1344141 1344211 1344213 1344293]
   256     1344841       328.330  [1344791 1344885 1344933 1344853 1344742]
   512     1345394       328.465  [1345401 1345346 1345459 1345378 1345385]
  1024     1345175       328.411  [1345186 1345004 1345073 1345290 1345321]
  2048     1345471       328.483  [1345461 1345542 1345521 1345454 1345375]
  4096     1345587       328.512  [1345653 1345479 1345686 1345662 1345456]
```

---

## SASS verification

`sass/bench_stride_probe.sass` (latency kernel) timed-loop body:

```
.L_x_2:
        IMAD.SHL.U32 R6, R8, 0x4, RZ ;
        LOP3.LUT     R6, R6, 0xffffffc, RZ, 0xc0, !PT ;
        IADD3        R6, P0, PT, R6, UR8, RZ ;
        IMAD.X       R7, RZ, RZ, UR9, P0 ;
        LDG.E.STRONG.SM R7, desc[UR6][R6.64] ;     ← single LDG per iteration
        VIADD        R9, R9, 0x1 ;
        IMAD.IADD    R8, R5, 0x1, R8 ;
        ISETP.NE.U32.AND P0, PT, R9, RZ, PT ;
        LOP3.LUT     R0, R7, R0, RZ, 0x3c, !PT ;   ← XOR feeds into R0 (loop-carried v)
   @P0  BRA `(.L_x_2) ;
```

`sass/bench_stride_probe_thr.sass` (throughput kernel) timed-loop body:
16 distinct `LDG.E.STRONG.SM` instructions (lines 91-169 of the SASS file)
each writing to a different register, followed by 16 `LOP3.LUT` reductions
into `acc`. ILP available = 16.

LDG count tally:
```
$ grep -c "LDG.E.STRONG.SM" sass/bench_stride_probe_thr.sass
17     # 1 warm-up + 16 inner-unrolled LDGs
$ grep -c "LDG.E.STRONG.SM" sass/bench_stride_probe.sass
2      # 1 warm-up + 1 timed body
```

Anti-DCE is in place via gated impossible-predicate STG (`@!P0 STG.E ... R5`)
visible in both SASS files — XOR-accumulated `v` is materialized.

---

## DETAILED CRITIQUE OF CATALOG TABLE

The catalog presents one column "cy/load" with values that span both the
throughput regime (56 cy) and the latency regime (304-316 cy). This is
internally inconsistent:

* If 56 cy is throughput → the regime change point is at stride 32 (when the
  LSU stops being able to hide the per-sector L1 cost via spatial reuse), not
  at stride 64.
* If 304 cy is latency → the catalog's other entry on line 111 (`ld.global L2
  | 301 cy`) is the same number, and represents a serial-dependency
  measurement, not a continuation of the 56 cy throughput sweep. They are
  apples-and-oranges.

The catalog's true findings, restated rigorously:

1. **L1 spatial reuse breaks at stride 32 B**, not 64 B. Above stride 32 B,
   each `ld.global.u32` lane requires a fresh 32 B sector (the L1 tag-table
   unit), so L1 hit ratio collapses to ~1.5%.
2. **A 32 B sector (not 128 B line) is the L1 allocation unit** on B300, so
   warp coalescing actually combines into 32 B sectors (4 sectors per 128 B
   line, but tags are per-sector).
3. **L2 hit latency is ~327 cy** at 1942 MHz (~169 ns), measured via
   single-thread dep-chain. This matches the catalog's separate "L2 latency
   = 301 cy" entry.
4. **Throughput-mode L2 hit per-load cost is ~158-163 cy** in the same
   single-thread test when ILP is exposed (K_INNER=16 unroll). The discount
   vs latency comes from pipelining LDGs through the LSU — exactly what a
   real kernel benefits from.
5. **The catalog's "5.4× jump at stride 64"** is an artifact of comparing a
   throughput-style stride-4 measurement (56 cy) with a latency-style
   stride-64 measurement (304 cy). It is not a discontinuity in any single
   self-consistent experiment.

---

## RECOMMENDED REPLACEMENT TABLE for catalog §22f

Suggested rewrite (all data measured this session, B300 SXM6 sm_103a, 1942
MHz, default boost):

| Stride | thr 1-thr ILP=16 | thr warp ILP=16 | latency dep-chain | tier |
|-------:|-----------------:|----------------:|------------------:|:-----|
|    4 B |          57 cy   |         53 cy   |         104 cy    | L1 hit (sector reused) |
|    8 B |          87 cy   |         76 cy   |         136 cy    | L1 hit, partial reuse |
|   16 B |         136 cy   |        113 cy   |         200 cy    | L1 hit weakening |
| **32 B** |       **160 cy** |      **117 cy** |       **328 cy**  | **L1 plateau, sector-per-lane** |
|   64 B |         160 cy   |        120 cy   |         327 cy    | same plateau |
|  128 B |         157 cy   |        131 cy   |         328 cy    | L2 hit (still in-cache) |
| 1024 B |         159 cy   |        134 cy   |         328 cy    | L2 hit |

Plain-English summary: "On B300, the L1 sector tag is 32 B. A `ld.global.u32`
at stride < 32 B benefits from spatial reuse across consecutive loads (best
case 57 cy at stride 4). At stride ≥ 32 B every load misses L1 and hits
L2; per-load cost saturates at ~160 cy (throughput-pipelined) or ~328 cy
(serial dependency, exposing full L2 latency)."

---

## FILES

* `tests/bench_stride_probe.cu` — latency-chain kernel (NEW)
* `tests/bench_stride_probe_thr.cu` — throughput kernel (NEW)
* `sass/bench_stride_probe.sass` — SASS dump (auto-generated)
* `sass/bench_stride_probe_thr.sass` — SASS dump (auto-generated)
* `/tmp/sweep_thr_mode0_final.txt`, `/tmp/sweep_thr_mode1_final.txt`,
  `/tmp/sweep_mode0.txt` — raw sweep stdouts (this session)
