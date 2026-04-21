# M2 — B300 Energy Ladder (2026-04-21 synthesis)

**Power and energy reference for B300 SXM6 AC sustained at 1500 MHz lock.**
Synthesizes H1-H10 + R1-R4 findings into a single ladder.

System: 148 SMs, idle baseline 164-167 W, TDP ceiling ~1100 W.

---

## 1. Idle ladder (no compute, only allocation)

| State | Power | Δ above true idle | Source |
|-------|------:|-------------------:|--------|
| GPU true idle (no kernel) | 164.7 W | 0 | — |
| GPU + 1 SM kernel "alive" | 165.4 W | +0.7 W | H6 (#e642e65) |
| GPU + all 148 SMs spinning (no work) | 172.0 W | +7.3 W = **0.05 W/SM** | H6 |
| GPU + 148 SMs allocated TMEM (idle) | 172.0 W | +7.3 W (no extra) | H7 (#c245f13) |
| GPU + 148 SMs in mbarrier.try_wait | 171.6 W | +6.9 W | R2 (#c30248c) |
| GPU + 148 SMs spinning on managed flag | 173.1 W | +8.4 W | R2 |
| GPU + 148 SMs in __syncthreads loop | 173.2 W | +8.5 W | R2 |

Key: TMEM allocation has ZERO power overhead. mbarrier.try_wait saves
~25% power vs spin (4.3 vs 5.8 W delta on 148 SMs).

---

## 2. Per-pipe active power (148 SMs, ITERS for ~100ms+ steady-state)

| Pipe / Op | Δ power | Per-FFMA-eq energy | Source |
|-----------|--------:|--------------------:|--------|
| Idle (loop only, DCE'd) | 0 W | 0 | H1 |
| FFMA (single chain) | +10 W | 0.6 pJ/FLOP | H1 (#dedd2b1) |
| FFMA (16 chains, 2-RF reads) | +24 W | 0.6 pJ/FFMA | H9 (#b489c02) |
| FFMA (16 chains, 3-RF reads, no .reuse) | +27 W | 0.91 pJ/FFMA | H9 |
| IMAD chain | +37 W | ~6.5 J / 1M ops | H2 (#2713af5) |
| LOP3 chain | +22 W | ~3.85 J / 1M ops | H2 |
| MUFU rsqrt.ftz | +24 W | similar to LOP3 | H3 |
| MUFU sin (no .ftz) | +39 W | 1.5× MUFU rsqrt | H3 |
| **LDG (memory)** | **+177 W** | **~96.6 J / 1M ops (15× IMAD)** | H4 |
| LDS (32-bit shared load) | +22 W | 4.4× lower than LDG | H8 (#7b6ec38) |
| STS (32-bit shared store) | +31 W | +41% vs LDS | H8 |
| LDS.128 (vec load) | +56 W | 2.5× scalar | H8 |
| STS.128 (vec store) | +74 W | 2.4× scalar | H8 |

**KEY: Memory pipe (LDG +177 W) is by FAR the dominant power consumer.**
SMEM is 5× lower power than HBM. Cache-blocking saves both time AND energy.

---

## 3. Per-FLOP/FLOP-eq energy

| Operation | pJ per op | Notes |
|-----------|----------:|-------|
| FFMA (with .reuse, broadcast operand) | **2.2 pJ/FLOP = 4.4 pJ/FFMA** | H1 baseline |
| FFMA (no .reuse, 3 unique RF reads) | 6.5 pJ/FFMA (1.49× more) | H9 |
| RF read (incremental) | **0.3 pJ/read** | H9 |
| LDS u32 read | ~5 pJ (estimated from H8 ratio) | derived |
| LDG u32 read | ~75 pJ (estimated from H4 ratio) | derived |
| Branch (predictable) | 3.36× FFMA energy = 14.8 pJ | H5 (#356f0be) |
| Branch (divergent half-warp) | 4.20× FFMA energy = 18.5 pJ | H5 |
| @p IMAD.IADD (vs bare IADD3) | 2× cost | M2 (#c2c26db) |

---

## 4. Compiler flag energy impact

| Flag combo | Baseline runtime | With flag | Δ time | Δ energy | Source |
|------------|-----------------:|---------:|-------:|---------:|--------|
| -use_fast_math (vs no fast_math) | 348 ms | 106 ms | 3.28× faster | **4.05× less energy** | G2 (#5a5a393) |
| __forceinline__ (vs __noinline__) | 182 ms | 12 ms | 15.2× faster | ~15× less energy | G7 (#21fa032) |
| .reuse on broadcast (vs no .reuse) | (in same kernel) | -32% time | -19% power | **-49% energy** | H9 |
| -Xptxas=-O3 (vs -O0) | 53 ms | 12 ms | 4.4× faster | ~4× less energy | G1 (#d657f69) |

---

## 5. Static vs dynamic power breakdown

- **Static "alive" SM**: 0.05 W
- **Dynamic FFMA SM**: +0.4 W (8-9× ratio dynamic/static)
- **Implication**: B300 uses mature process; **leakage is ~10% of switching**.
  Power-gating idle SMs would save only ~7 W (out of ~1100 W TDP).
  Clock/voltage scaling is the main lever for power management.

---

## 6. Lane-level granularity

- Predicating off lanes does NOT reduce per-warp power (R4 #de87c5a).
  - 32 → 1 active lanes via @p: 171 → 170.6 W (delta < 1 W)
  - Per-warp issue + RF dominates per-lane compute power
- True lane-level power gating requires BRA divergence (warp-level skip).

---

## 7. Practical energy recipes

### Maximum FFMA throughput at minimum energy
1. Use `-use_fast_math` (4× energy savings)
2. `__forceinline__` device helpers (15× speedup)
3. Use broadcast operand pattern → emit `.reuse` (49% energy savings via D6/H9)
4. Target compute-bound (FFMA dominant) over memory-bound (LDG dominant) — 5× lower energy

### Persistent kernels waiting on host
1. Use `mbarrier.try_wait` not spin loop (25% lower power per R2)
2. Or just exit and re-launch (kernel launch is 2 µs; spin-waits cost more)

### Multi-GPU coordination
- NVLink one-way 1.55 µs (J1)
- Use `cuStreamWriteValue32` not kernel-write (8× faster, J2)
- For CPU↔GPU signaling: managed mem + CPU spin (4.4 µs RT, L1)

---

## 8. Methodology caveats

1. **nvidia-smi power = 33 Hz max CLI** sample rate (S4). Need 5+ sec sustained kernels for steady-state.
2. **NVRTC harness uses fast_math by default** — many H/R measurements include FTZ behavior (memory note).
3. **Per-pipe isolation is hard** (R1 partial). Use ncu pipe metrics where possible.

For complete commit hashes and rigor docs, see `M1_V4_DEEP_DIVE_INDEX.md`.
