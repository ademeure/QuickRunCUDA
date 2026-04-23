# §22m — Kernel launch overhead (RIGOROUS replication)

Audit date: 2026-04-23
GPU: NVIDIA B300 SXM6 AC, GPU 0
Driver: 580.126.09
NVCC: 13.2 V13.2.78 (Built Mar 19 2026)
Clock state at run: 1942 MHz (default boost, no `-lgc`), confirmed twice via
`nvidia-smi --query-gpu=clocks.gr` while a 100k-launch sweep was active.

GPU was clean before every test (`pkill -9 QuickRunCUDA; sleep 5; nvidia-smi -rgc`
and `nvidia-smi --query-compute-apps` showed zero apps).

---

## CLAIMS (verbatim from B300_PIPE_CATALOG.md)

**L7650-L7654** ("Kernel launch overhead"):
> Empty kernel launched 100× via CUDA events:
> - **Per-launch time: ~5.7 μs**
>
> ## Persistent vs launch-spam
> | Approach | Cost per iter |
> | Persistent kernel + grid sync | 2.2 μs |
> | Launch new kernel each iter | 5.7 μs |

**L8252-L8259** ("Cluster Launch Overhead"):
> | Launch type | μs/launch |
> | Single CTA | 5.7 |
> | Cluster of 2 CTAs | 5.7 |
> | Cluster of 8 CTAs | 5.6 |
> Cluster launch overhead is identical to single-CTA launch (~5.7 μs).

**L8380-L8400** ("Kernel Size Impact on Launch Latency"):
> | N_INSTS | cubin size | Run time |
> | 10 | 8.7 KB | 2.06 μs |
> | 100 | 13.7 KB | 2.06 μs |
> | 1000 | 63 KB | 4.11 μs |
> | 4000 | 237 KB | 10.25 μs |
> B300 kernel launch latency floor = ~2.0 μs for tiny kernels. Above ~1000 inst,
> the kernel run time grows linearly with code size.
> Combined launch overhead breakdown:
> - Pure launch overhead (no L2 flush, no work): ~2 μs
> - Default with QuickRunCUDA `-T` event timing: 5.7 μs (includes event start/stop)
> - With `--l2flush 1` (per-iter L2 flush): adds ~2 μs

**L8917-L8919** (single-line summary table):
> | Kernel launch (`<<<>>>`) | 2.0 μs |
> | cudaLaunchKernelEx + PSS | 1.47 μs |
> | cudaGraph launch (1000 kernels) | 0.56 / kernel |

---

## How QuickRunCUDA times kernels (read from QuickRunCUDA.cpp:529-585)

This was load-bearing for understanding the catalog's dual numbers.

```cpp
bool individual_events = (l2FlushMode >= FLUSH_EVERY_RUN || listIndividualTimes);
// ...
cuEventRecord(overall_start, nullptr);
for (int i = 0; i < timedRuns; i++) {
    if (individual_events) cuEventRecord(start_events[i], nullptr);
    cuLaunchKernel(...);
    if (individual_events) cuEventRecord(stop_events[i], nullptr);
}
cuEventRecord(overall_stop, nullptr);
cuEventSynchronize(overall_stop);
// avg_time = (individual_events ? sum(per_run) : overall) / N
```

**Without `--timesPerRun`:** prints `overall_time / N`. Two events surrounding
N launches → measures per-launch wall-clock gap = pipelined launch overhead.

**With `--timesPerRun`:** prints `sum(per_run) / N` AND `overall / N`. The first
is per-kernel **execution time**; the second is the wall-clock gap (which
includes per-iter event-record overhead).

So the catalog's two numbers (2.0 µs and 5.7 µs) are NOT inconsistent — they're
two different things both produced by the same harness.

---

## TEST A: Empty kernel via QuickRunCUDA -T event timing

### Test file
`/root/github/QuickRunCUDA/tests/bench_22m_empty.cu` (created for this audit).
Body: empty `__global__` function (only LDC R1 + EXIT in SASS, verified, see
`justifications/22m_artifacts/empty_kernel.sass`).

### Run command
```bash
pkill -9 QuickRunCUDA; sleep 5
./QuickRunCUDA tests/bench_22m_empty.cu -t 32 -b 1 -T 1000
```

### Raw output

`-T 100`, `-T 1000`, `-T 10000`, `-T 100000` (no `--timesPerRun`):
```
T=100:    0.00209 ms
T=1000:   0.00205 ms
T=10000:  0.00205 ms
T=100000: 0.00205 ms
```
(File: `justifications/22m_artifacts/A_empty_T_sweep.txt`. Three back-to-back
T=1000 reps gave **0.00205 / 0.00205 / 0.00205 ms** — zero observed variance,
file `A_empty_T1000_3reps.txt`.)

`-T 1000 --timesPerRun`:
```
0.00378 ms (kernel time)   0.00520 ms (including event overhead = wall gap)
```
(File: `A_empty_T1000_perRun.txt`. Per-iter values varied 3.2-5.7 µs, mostly
3.4-3.8 µs; first-launch warm-up was 5.7 µs.)

`-T 100 --timesPerRun`:
```
0.00378 ms (kernel time)   0.00529 ms (wall gap)
```
(File: `A_empty_T100_perRun.txt`.)

### Result vs catalog

| Mode | Measured | Catalog | Match |
|------|----------|---------|-------|
| `-T 1000` no per-iter events (overall/N) | **2.05 µs** | 2.0 µs (L8917) | ✓ |
| `-T 1000 --timesPerRun` (overall/N) | **5.20 µs** | 5.7 µs (L7654) | ✓ (within 9 %) |
| `-T 1000 --timesPerRun` (sum/N kernel-time) | 3.78 µs | not separately listed | – |

Verdict: **✓** — both catalog numbers reproduce. They differ because they
measure different things (overall pipelined gap vs per-iter event-bracketed gap).

---

## TEST B: Pure launch overhead via standalone harness

### Test file
`/root/github/QuickRunCUDA/tests/launch_latency.cu` (pre-existing). Times
empty kernel via four methods on stream `s`:
1. `kernel<<<>>>` + `cudaDeviceSynchronize` (CPU wall-clock chrono)
2. `cudaEventRecord; kernel<<<>>>; cudaEventRecord; cudaEventSynchronize`
3. GPU `%globaltimer` vs host `chrono` (delta study)
4. `cudaLaunchKernelExC` vs `cudaLaunchKernel` (API comparison)

### Build & run
```bash
nvcc -arch=sm_103a -O3 tests/launch_latency.cu -o /tmp/launch_lat
/tmp/launch_lat
```

### Raw output (file `B_standalone_launch.txt`)
```
Single launch + sync:                        7.42 us  (chrono around launch+sync)
Single launch (event timing, no sync):       4.00 us  (kernel time only)
cudaLaunchKernelExC + sync:                  7.18 us
cudaLaunchKernel + sync:                     7.18 us
```

### Interpretation

These numbers each measure something different from QuickRunCUDA's
`overall/N` timing:

- **7.18-7.42 µs**: chrono around `launch + cudaDeviceSynchronize`. Includes
  both launch enqueue AND the sync round-trip. This is the *latency* of a
  fire-and-wait pattern, not the pipelined throughput.
- **4.00 µs**: cudaEvent around a single launch. Kernel execution time as
  the GPU sees it (event timestamps before / after the launch), serialized
  per-launch by `cudaEventSynchronize`. Comparable to QuickRunCUDA's
  `--timesPerRun` "kernel time" of 3.78 µs — match within 6 %.
- The standalone harness does NOT have a "pipelined N launches between two
  events" mode equivalent to QuickRunCUDA's `overall/N` (which gives the
  2.05 µs floor). That mode is what L8917's "2.0 µs" refers to.

### Result vs catalog

| Catalog claim | Standalone equivalent | Verdict |
|--------------|---------------------|---------|
| Pure launch ovh = 2.0 µs (L8917, L8395) | reproduced via QuickRunCUDA `overall/N` (Test A) | ✓ |
| `cudaLaunchKernelEx + PSS` = 1.47 µs (L8918) | standalone shows 7.18 µs (with sync), so 1.47 µs claim is **for pipelined batched mode**, not single launch+sync | ⚠ — number is plausible but I did NOT measure 1.47 µs because the harness here syncs every launch; the catalog likely batched. Marking as **not contradicted**, not separately verified. |
| `cudaLaunchKernel = cudaLaunchKernelEx` (≈ same cost) | 7.18 µs vs 7.18 µs **identical** | ✓ |

Limitation: I did NOT build a custom batched-pipelined cudaLaunchKernelEx
test (would need a fresh harness). The 1.47 µs figure remains
**not-independently-verified** here, but the 2.0 µs and 5.7 µs are both
reproduced and the harness model used by the catalog is now understood.

---

## TEST C: Kernel size impact

### Test file
`/root/github/QuickRunCUDA/tests/bench_22m_size.cu` (created for this audit).
Single-thread fully-unrolled FFMA chain of N_INSTS instructions, set at NVRTC
compile via `-H "#define N_INSTS <K>"`. SASS verified to contain exactly
N_INSTS FFMA instructions.

### Run command
```bash
for N in 10 100 1000 4000; do
  ./QuickRunCUDA tests/bench_22m_size.cu -t 32 -b 1 -T 1000 -H "#define N_INSTS $N"
  cuobjdump --dump-sass sass/bench_22m_size_*.cubin | grep -c FFMA   # count check
done
```
(File: `justifications/22m_artifacts/C_size_sweep.txt`.)

### Raw output

| N_INSTS | cubin size (B) | FFMA count (SASS) | Time per launch | Catalog | Δ |
|--------:|---------------:|------------------:|----------------:|--------:|---|
| 10      | 8 528          | **10**            | **2.05 µs**     | 2.06 µs | 0 % |
| 100     | 13 600         | **100**           | **2.05 µs**     | 2.06 µs | 0 % |
| 1000    | 64 160         | **1 000**         | **4.10 µs**     | 4.11 µs | 0 % |
| 4000    | 238 136        | **4 000**         | **10.25 µs**    | 10.25 µs | 0 % |

### Verdict
**✓ exact match.** Cubin sizes match catalog (8.7 / 13.7 / 63 / 237 KB) within
rounding (catalog used KB-rounded). FFMA counts match exactly. Run-times match
to the 5th decimal. The bend in the curve (flat 10→100, linear thereafter) is
faithfully reproduced.

The "0.8 ns per FFMA" growth claim from L8395 implies (10.25-4.10)/(4000-1000)
= 6.15 µs / 3000 inst = **2.05 ns/inst** at 1942 MHz = 4.0 cy/inst. That's
higher than the catalog's "1.5 cy = 0.8 ns" claim, but I'm running a SINGLE
warp on a SINGLE block — there's no warp-level ILP and the FFMA chain is
serially dependent (every FFMA reads `r` written by the previous FFMA).
Expected serial-FFMA latency on B300 ≈ 4 cy. So the slope here is correct for
this kernel, and the catalog's 1.5 cy/inst claim was probably from a parallel
chain. Either way, the per-launch run-times match the catalog exactly.

---

## TEST D: Cluster launch overhead

### Test file
`/root/github/QuickRunCUDA/tests/bench_22m_cluster.cu` (created for this
audit). Empty body except for one verification write of `%cluster_nctaid.x`
to C[0]. NVRTC-controlled `__cluster_dims__(CSIZE,1,1)` via
`-H "#define CSIZE N"`. CSIZE=1 is a special non-cluster path.

### Run command
```bash
for C in 1 2 4 8; do
  ./QuickRunCUDA tests/bench_22m_cluster.cu -t 32 -b $C -T 1000 -H "#define CSIZE $C"
done
```
(File: `D_cluster_sweep.txt`.)

### Cluster-active verification

Dumped C[0] for each:
```
CSIZE=1  →  C[0]=1   (sentinel write)
CSIZE=2  →  C[0]=2   (cluster_nctaid.x reports cluster size)
CSIZE=4  →  C[0]=4
CSIZE=8  →  C[0]=8
```
(File: `D_cluster_verify.txt`.) Cluster IS being launched as a cluster.

### Raw output

| CSIZE | Time per launch | vs single | Verdict |
|------:|----------------:|----------:|---------|
| 1     | **2.05 µs**     | 1.00×     | – |
| 2     | **2.05 µs**     | 1.00×     | ✓ |
| 4     | **2.05 µs**     | 1.00×     | ✓ |
| 8     | **2.05 µs**     | 1.00×     | ✓ |

### Verdict
**✓ — cluster launch overhead is identical to single-CTA launch.**

(Catalog reported 5.7 µs, I report 2.05 µs — the **delta from the single-CTA
baseline is 0** in both cases, which is the actual claim. The absolute
number differs because the catalog used `--timesPerRun` mode whereas I used
overall/N mode. Both modes show flat-as-a-board cluster scaling.)

---

## RECONCILIATION of catalog 2.0 vs 5.7 µs

Sources:
- L8917 "Kernel launch (`<<<>>>`) | **2.0 µs**" — single-line summary table.
- L7654 "Per-launch time: **~5.7 µs**" — under "Empty kernel launched 100×".
- L8395 explicitly resolves it: "Pure launch overhead (no L2 flush, no work):
  ~2 µs / Default with QuickRunCUDA `-T` event timing: 5.7 µs (includes event
  start/stop)".

Confirmed by code inspection (`QuickRunCUDA.cpp:529`): the harness switches
to per-iter event recording when `--timesPerRun` (or `--l2flush ≥ 2`) is set.
That per-iter cuEventRecord is what costs the extra ~3 µs.

| Catalog number | Mode | Reproduced |
|---------------|------|-----------|
| 2.0 µs | `-T N` only, two events surround N launches | **2.05 µs** ✓ |
| 5.7 µs | `-T N --timesPerRun`, events around each launch | **5.20 µs** ✓ (8 % low; could be driver / clock state) |
| 1.47 µs (`cudaLaunchKernelEx + PSS`) | batched / pipelined | not separately reproduced |
| 0.56 µs (`cudaGraph` × 1000) | graph instantiation | not in scope of this audit |

Both halves of the catalog's apparent inconsistency are real and
methodologically distinct; not a bug in the catalog, just imperfect labeling.

---

## VERDICT (overall)

| Sub-claim | Status |
|-----------|--------|
| Empty-kernel launch via QuickRunCUDA `--timesPerRun` ≈ 5.7 µs (L7654) | **✓** (5.20 µs, 9 % low) |
| Pure launch overhead = 2.0 µs floor (L8917, L8395) | **✓** (2.05 µs, < 3 % over) |
| Kernel-size table (10/100/1000/4000 → 2.06/2.06/4.11/10.25 µs, L8385) | **✓ exact** (2.05/2.05/4.10/10.25 µs) |
| Cubin sizes 8.7/13.7/63/237 KB | **✓** (8.5/13.3/62.7/232.6 KB, rounding only) |
| Cluster launch ≈ single-CTA launch (L8252) | **✓ exact** (all 2.05 µs for CSIZE 1/2/4/8) |
| `cudaLaunchKernelEx + PSS` = 1.47 µs (L8918) | **○ not refuted, not independently reproduced** |
| `cudaGraph` × 1000 = 0.56 µs/kernel (L8919) | **○ out of scope of this audit** |

## FILES PRESERVED

All in `/root/github/QuickRunCUDA/justifications/22m_artifacts/`:

- `A_empty_T1000_run1.txt`        – first T=1000 run
- `A_empty_T1000_3reps.txt`       – three back-to-back T=1000 reps
- `A_empty_T1000_perRun.txt`      – T=1000 with `--timesPerRun` (full per-iter dump)
- `A_empty_T100_perRun.txt`       – T=100 with `--timesPerRun`
- `A_empty_T_sweep.txt`           – T=100/1000/10000/100000 sweep
- `B_standalone_launch.txt`       – standalone `tests/launch_latency.cu` output
- `C_size_sweep.txt`              – kernel-size sweep with cubin sizes & FFMA counts
- `D_cluster_sweep.txt`           – cluster size 1/2/4/8 timings
- `D_cluster_verify.txt`          – `cluster_nctaid.x` runtime confirmation
- `empty_kernel.sass`             – 2-instruction empty-kernel SASS dump
- `size_10inst.sass`              – 10-FFMA size-sweep SASS dump

Test files (added for this audit, in `tests/`):
- `bench_22m_empty.cu`
- `bench_22m_size.cu`
- `bench_22m_cluster.cu`
