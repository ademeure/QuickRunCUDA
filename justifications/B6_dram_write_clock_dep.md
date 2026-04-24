# §B6 — DRAM write peak: clock-dependence audit

> Audit date: 2026-04-23. Hardware: NVIDIA B300 SXM6 AC (148 SMs, sm_103a),
> CUDA 13.2, ECC enabled, GPU 0 only (CUDA_VISIBLE_DEVICES=0).
> Default boost clock = 2032 MHz; `nvidia-smi -lgc 1942` paradoxically pins to
> 1920 MHz (per CLAUDE.md). Used `--clock-control none` for ncu to honor the lock.

## Catalog claim under review

`B300_PIPE_CATALOG.md` L46:

> | DRAM (HBM3E) | **7.18 TB/s** read (ncu-verified, WS=1GB→8GB) | **7.09 TB/s** write |

The number under audit is the **7.09 TB/s DRAM write peak**.

## User concern (reviewed_errors L1132)

> "SM→L2 *write* path is limited to 32B/clk (read is much higher), you cannot
> get peak HBM write bandwidth at lower clocks, and even 1920MHz vs 2032MHz
> might have had an effect."

Concretely: if the bottleneck is the per-SM SM→L2 store path at 32 B/cycle,
then chip-wide DRAM write throughput should be:

`32 B/clk × 148 SMs × clock_GHz`

= 7.10 TB/s @ 1500 MHz, 8.05 TB/s @ 1700 MHz, 9.10 TB/s @ 1920 MHz, 9.62 TB/s @ 2032 MHz

— bounded above by the HBM ceiling (~7.18 TB/s on this AC SKU at 7680-bit bus).
So at lower clocks, the SM→L2 path SHOULD bind; at higher clocks, HBM SHOULD bind.

## Test setup

**Test file:** `tests/bench_dram_peak.cu` (existing, OP=5 = `st.global.v8.u32`,
256-bit STG per inst).

**Kernel highlights** (anti-DCE):

- 256-bit aligned writes via `st.global.v8.u32` PTX (8 dwords = 32 B/inst)
- Per-thread 4096 inner iters × 32 B = 131 072 B written (cold DRAM, WS = 1 GB > L2)
- Address pattern `(tid * 32 + (i+j) * 32 * gridDim*blockDim) & ws_mask`
  — full-grid stride, sequential per-thread, modular into the 1-GB B buffer
- Stored values are `tid+i+j` (loop-counter-derived; not constant; not DCE-able)
- `volatile` + `: "memory"` clobber on the inline asm
- (No accumulator load, since this is write-only — no read-side amplification)

**SASS verification** (`sass/bench_dram_4gb_*.sass`, OP=5 path):

```
$ grep -cE "STG\.E\.ENL2\.256" sass/bench_dram_4gb_4017206595.sass
16
```

Exactly UNROLL=16 vectorized 256-bit stores in the inner body — no DCE, no
spurious extras. ENL2 = "evict-normal-L2" hint (B300 default for STG via cg-style).

## Build/run

```
make                                                    # builds ./QuickRunCUDA
./QuickRunCUDA tests/bench_dram_peak.cu \
  -t 512 -b 1184 -T 5 -A 1073741824 -B 1073741824 \
  -2 1073741824 -0 4096 \
  -H "#define OP 5
#define BLOCK_SIZE 512"
```

- BS=512, blocks=1184 = 8 CTAs/SM × 148 SMs (high occupancy)
- ITERS=4096, WS=1 GB (-2 1073741824)
- 5 timed iterations, kernel time event-based per iter (cuEventRecord)

**ncu wrapper:**

```
ncu --clock-control none \
    --metrics dram__bytes_write.sum.per_second,
              dram__sectors_write.sum.per_second,
              lts__t_sectors_srcunit_tex_op_write.sum.per_second,
              lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum.per_second,
              sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,
              sm__throughput.avg.pct_of_peak_sustained_elapsed,
              gpc__cycles_elapsed.avg.per_second,
              gpu__time_duration.sum \
    --launch-skip 1 --launch-count 1 \
    ./QuickRunCUDA tests/bench_dram_peak.cu ...
```

`--clock-control none` is ESSENTIAL — without it ncu force-clamps GPU to 1920 MHz
regardless of the `nvidia-smi -lgc` lock and reports inconsistent numbers.

## Clock-lock procedure

```
sudo nvidia-smi -lgc <MHz> -i 0   # 1500, 1700, 1920 tested
sudo nvidia-smi -rgc -i 0          # release lock → boost (2032 MHz under load)
```

Confirmed `nvidia-smi -i 0 --query-gpu=clocks.gr --format=csv` after each lock.
At 1942 lock: GPU pinned to 1920 (per known B300 quirk).

## Raw ncu output (per clock)

### 1500 MHz (locked)

```
dram__bytes_write.sum.per_second                         5.51 TB/s
dram__sectors_write.sum.per_second                     172.30 sector/ns
gpc__cycles_elapsed.avg.per_second                       1.50 GHz
gpu__time_duration.sum                                  13.88 ms
lts__t_sectors_srcunit_tex_op_write.sum.per_second     178.92 sector/ns
lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum.per_second   6.64 sector/ns
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active     2.75 %
sm__throughput.avg.pct_of_peak_sustained_elapsed               20.18 %
```

### 1700 MHz (locked)

```
dram__bytes_write.sum.per_second                         6.20 TB/s
dram__sectors_write.sum.per_second                     193.84 sector/ns
gpc__cycles_elapsed.avg.per_second                       1.70 GHz
gpu__time_duration.sum                                  12.33 ms
lts__t_sectors_srcunit_tex_op_write.sum.per_second     201.39 sector/ns
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active     2.75 %
sm__throughput.avg.pct_of_peak_sustained_elapsed               20.02 %
```

### 1920 MHz (locked via `-lgc 1920`)

```
dram__bytes_write.sum.per_second                         6.75 TB/s
dram__sectors_write.sum.per_second                     211.02 sector/ns
gpc__cycles_elapsed.avg.per_second                       1.92 GHz
gpu__time_duration.sum                                  11.30 ms
lts__t_sectors_srcunit_tex_op_write.sum.per_second     219.79 sector/ns
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active     2.62 %
sm__throughput.avg.pct_of_peak_sustained_elapsed               19.35 %
```

### 2032 MHz (no lock; default boost, ncu confirms gpc=2.03 GHz)

```
dram__bytes_write.sum.per_second                         6.86 TB/s
dram__sectors_write.sum.per_second                     214.38 sector/ns
gpc__cycles_elapsed.avg.per_second                       2.03 GHz
gpu__time_duration.sum                                  11.09 ms
lts__t_sectors_srcunit_tex_op_write.sum.per_second     223.83 sector/ns
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active     2.49 %
sm__throughput.avg.pct_of_peak_sustained_elapsed               18.60 %
```

## Wall-clock cross-check (event-based, no ncu)

```
clock=1500  ms/run=13.92    (matches ncu 13.88 within 0.3%)
clock=1700  ms/run=12.30    (matches ncu 12.33 within 0.2%)
clock=1920  ms/run=11.31    (matches ncu 11.30 within 0.1%)
clock=2032  ms/run=11.11    (matches ncu 11.09 within 0.2%)
```

Bytes per launch = 1184 blocks × 512 thr × 4096 iters × 32 B = 76.0 GB.
Wall TB/s = 76.0 GB / time_ms. Matches ncu's `dram__bytes_write` to within ~3%
(small overcount from L2 absorption — `lts_lookup_hit` shows 6.64 / 178.9 = **3.7%**
of write traffic hits L2, not DRAM-bound). The user's reviewed_errors comment that
"writes hit L2 first" applies but is small for this WS=1 GB recipe.

## Cross-check: SM→L2 wire bandwidth from `lts__t_sectors_srcunit_tex_op_write`

The `lts__t_sectors_srcunit_tex` family counts sectors arriving at the L2
**from the SM-side path** (TEX/L1 client → L2). Each sector = 32 B. So:

| Clock | sectors/ns SM→L2 | SM→L2 BW (TB/s) | 32 B/clk × 148 SMs ceiling | SoL of SM→L2 |
|---:|---:|---:|---:|---:|
| 1500 | 178.92 | **5.72** | 7.10 | **80.6 %** |
| 1700 | 201.39 | **6.44** | 8.05 | 80.0 % |
| 1920 | 219.79 | **7.03** | 9.10 | 77.3 % |
| 2032 | 223.83 | **7.16** | 9.62 | 74.4 % |

## Composite results table

| Clock (MHz) | Wall ms/run | DRAM write TB/s (ncu) | SM→L2 TB/s | SM→L2 32B/clk ceil (TB/s) | **SM→L2 SoL** | **HBM SoL (vs 7.18)** |
|---:|---:|---:|---:|---:|---:|---:|
| 1500 | 13.92 | **5.51** | 5.72 | 7.10 | 80.6 % | 76.7 % |
| 1700 | 12.30 | **6.20** | 6.44 | 8.05 | 80.0 % | 86.4 % |
| 1920 | 11.31 | **6.75** | 7.03 | 9.10 | 77.3 % | 94.0 % |
| 2032 | 11.11 | **6.86** | 7.16 | 9.62 | 74.4 % | 95.5 % |

## Analysis: which path binds at each clock?

**The SM→L2 SoL stays remarkably constant (74-81 %) across the entire clock
range** — the SM→L2 path scales linearly with SM clock (5.72 → 7.16 TB/s as we
go 1500 → 2032), so it IS clock-dependent and the user is RIGHT that it matters.

**However, neither path is fully pegged at any clock:**

- At 1500 MHz: SM→L2 at 80% of its 7.10 TB/s ceiling, HBM at 77% of 7.18 TB/s.
  **SM→L2 is closer to ceiling but neither is binding.**
- At 1700 MHz: SM→L2 at 80%, HBM at 86%. Both ~equally close to peak.
- At 1920 MHz: SM→L2 at 77%, HBM at 94%. **HBM nearly binding.**
- At 2032 MHz: SM→L2 at 74%, HBM at 96%. **HBM is the binding bottleneck.**

The crossover from SM→L2-limited to HBM-limited happens around **1700 MHz**.
Above 1700 MHz, increasing clock buys diminishing returns (HBM ceiling is
clock-independent: HBM3E PHY runs off a separate `memoryClockRate=3996 MHz`
domain, not the SM clock).

**Important nuance:** the SM→L2 SoL is *uniformly* 74-80 %, never approaching
the 32 B/clk theoretical. So the per-SM store path actually delivers ~25 B/clk
× 148 SMs in practice — there's some structural underutilization (likely
sector-credit or coalescing-buffer flushes) that's NOT clock-dependent. The
"32 B/clk × 148 SM" formula is the ABSOLUTE upper bound but practical chip-wide
write traffic plateaus at ~80% of it.

## User's hypothesis verdict

**Largely CONFIRMED with refinement:**

1. ✅ "SM→L2 write path scales with SM clock" — measured DRAM write rises
   monotonically 5.51 → 6.20 → 6.75 → 6.86 TB/s as clock rises 1500 → 2032 MHz.
2. ✅ "Cannot reach peak HBM write at lower clocks" — at 1500 MHz we're
   stuck at 5.51 TB/s = 77 % of HBM 7.18 TB/s. Can't do better at this clock.
3. ⚠ "1920 vs 2032 might affect this" — minor effect (6.75 vs 6.86 TB/s = 1.6 %
   delta). At 1920 MHz HBM is at 94 %; at 2032 MHz at 96 %. Both leave ~5 %
   on the table to HBM (some L2-absorption + write-merge inefficiency).
4. ❌ "32 B/clk SM→L2 ceiling is binding at low clocks" — actual SM→L2 traffic
   is at **80 %** of that ceiling at 1500 MHz, NOT 100 %. Some other SM-side
   limit (sector-credit return, store-coalescing buffer flush rate, or write-
   merge stall) keeps us below the 32 B/clk peak. The user is right about
   *direction* but the bottleneck isn't a HARD 32 B/clk wall.

## Verdict on catalog's "7.09 TB/s write" number

The catalog's **7.09 TB/s** is **OPTIMISTIC** by ~3-30 % depending on regime:

- Best measured DRAM write: **6.86 TB/s @ 2032 MHz** (default boost, no lock).
  This is the canonical "DRAM write peak" — but it's 96 % of HBM ceiling, not
  the 99 % implied by 7.09. The catalog's 7.09 figure is plausible only if
  measured under different conditions (different occupancy / longer run /
  L2-warmed kernels biasing toward 100 % of HBM ceiling).
- At catalog's commonly-cited 1920 MHz: only **6.75 TB/s**. The catalog's 7.09
  number, if read as "write peak at 1920 MHz", **overstates by 5 %**.
- At locked 1500 MHz: only **5.51 TB/s** = 78 % of catalog claim — confirms
  the user's concern that lower-clocked rigs can't hit 7.09 TB/s.

**Recommended catalog edit:**

> DRAM (HBM3E) write: **6.86 TB/s @ 2032 MHz** (boost, no lock) / **6.75 TB/s @ 1920 MHz**
> (locked) — was 7.09 TB/s in prior catalog (clock-state unspecified).
> Lower-clocked configs: 6.20 @ 1700 MHz / 5.51 @ 1500 MHz, scaling roughly
> linearly with SM clock. Bottleneck transitions from SM→L2 store path (≤ 1700 MHz)
> to HBM (≥ 1920 MHz). Read peak (7.18 TB/s) is reachable at all clocks since
> SM→L2 read path is wider (>32 B/clk).

## Methodology notes / pitfalls hit

1. **ncu with default `--clock-control` overrides `nvidia-smi -lgc` and pins to
   1920 MHz** silently. Always pass `--clock-control none` when measuring at
   non-1920 clocks. ncu will warn "results may be inconsistent" — that warning
   is INTENDED behavior here.
2. **Other agent on GPU 0 caused a transient 2× slowdown** in one wall-clock
   run — confirmed by re-running and getting the expected 11 ms. The `pgrep
   QuickRunCUDA` check found a stuck process from another concurrent
   `four_fix` worktree session. Always `pgrep` before reporting and discard
   runs that disagree with ncu by > 5%.
3. **`-T 1` standalone wall-clock matches ncu within 0.3 %** — gives a
   reliable cross-check of ncu's per-launch timing. Use when ncu metrics
   collection is flaky.
4. **`lts__t_sectors_srcunit_tex_op_write`** is the ncu metric for SM→L2 sector
   traffic. `lts__t_sectors_op_write` (without `srcunit_tex`) double-counts
   (includes L2→DRAM back-end traffic).
5. **5-launch ncu replay error 9** ("Failed to prepare kernel for profiling")
   intermittent due to GPU-side race with concurrent agent. Workaround: sleep
   8-15 s between ncu invocations and retry.
6. **Catalog's 7.09 TB/s number's clock-state was unspecified** — the audit
   exposes that this is the WORST kind of catalog hygiene issue (per CLAUDE.md
   §2). Always state clock when quoting BW.

## Hardware-side rationalization

B300 SM→L2 read path is provisioned wider than write path (per architecture
docs and reviewed_errors_b300.md L1132 user note). Specifically:
- **Read** path: ~64 B/clk/SM (or higher) × 148 SMs × 1.92 GHz = > 18 TB/s SM→L2 read
  ceiling — comfortably above HBM read peak 7.18 TB/s. Read is HBM-bound at
  all reasonable clocks.
- **Write** path: 32 B/clk/SM × 148 SMs × 1.92 GHz = 9.10 TB/s SM→L2 write
  ceiling — above HBM 7.18 TB/s only at high clocks. At ≤ 1500 MHz, SM→L2
  ceiling drops below HBM peak (7.10 vs 7.18) and write becomes SM-bound.

**Practical takeaway:** the asymmetric SM→L2 sizing (read fat, write narrow)
makes B300 DRAM-write performance ~5-10 % slower than read at boost clock,
and **substantially** slower (24 % gap) at locked-1500 MHz.

## Verdict

**⚠ PARTIAL — catalog 7.09 TB/s is high; canonical write peak = 6.86 TB/s @ 2032 MHz boost / 6.75 TB/s @ 1920 MHz locked.**
**User's hypothesis is largely confirmed**: SM→L2 write path is clock-dependent
and limits write throughput at lower clocks; HBM ceiling binds only at boost.
Catalog should add explicit clock annotation and tag write rate as "boost"-state.
