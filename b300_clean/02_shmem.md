# 02 — Shared Memory (SHMEM, DSMEM, banks, atomics)

**GPU:** B300 SXM6, sm_103a, 148 SMs, 2.032 GHz boost (1920 MHz when `-lgc 2032` is set — see 01_clock).

All theoretical numbers in this doc are quoted at the **2032 MHz boost clock** unless otherwise noted. Subtract ~6% for the 1920 MHz "locked" state.

---

## Headline numbers

| Quantity | Value | % of theory | Confidence |
|---|---:|---:|---|
| **Pure-read peak (LDS.128, RAW chain)** | **38.4 TB/s** | **99.8%** | **HIGH** |
| Read+write mix peak (4 reads + 1 write/iter) | 27.2 TB/s | 71% | HIGH |
| ldmatrix.x4.b16 peak (tensor feed path) | 33-35 TB/s | 91% | HIGH |
| float4 (non-volatile) typical | 35-36 TB/s | 92% | HIGH |
| Naive 8-scalar-LDS pattern | 19-26 TB/s | 50-67% | HIGH |
| Local SMEM latency (1-warp dependent chain) | 28 cy / 14.6 ns | — | HIGH |
| DSMEM latency (cluster=2, dependent chain) | 224 cy / 117 ns = **8× local** | — | HIGH |
| DSMEM throughput at ILP=4 vs local | **9× slower** | — | HIGH |

**Theoretical SHMEM peak**: 32 banks × 4 B/cycle × 148 SMs × 2.032 GHz = **38.49 TB/s** (260 GB/s/SM). HIGH.

---

## Optimal recipe — pure-read SoL (38.4 TB/s = 99.8%)

Source: commit **d41c38c** (`investigations/rigor_smem_sol.cu`). ncu-verified, SASS-verified. **HIGH** confidence.

```cpp
__launch_bounds__(256, 8) __global__ void smem_read(int4 *out, int iters, int seed) {
    __shared__ int4 smem[256];                                 // 4 KB — small footprint
    int tid = threadIdx.x;
    smem[tid] = make_int4(tid+seed, tid+seed+1, tid+seed+2, tid+seed+3);
    __syncthreads();

    int4 r0 = smem[tid];
    int4 r1 = smem[(tid+1) & 255];
    int4 r2 = smem[(tid+2) & 255];
    int4 r3 = smem[(tid+3) & 255];

    for (int i = 0; i < iters; i++) {
        // RAW dependency: next addr depends on prior reads — defeats LICM/DCE
        int o = (r0.x ^ r1.x ^ r2.x ^ r3.x) & 0xff;
        int4 n0 = smem[(tid + o + 0) & 255];
        int4 n1 = smem[(tid + o + 1) & 255];
        int4 n2 = smem[(tid + o + 2) & 255];
        int4 n3 = smem[(tid + o + 3) & 255];
        r0 = n0; r1 = n1; r2 = n2; r3 = n3;
    }
    if ((r0.x ^ r1.x ^ r2.x ^ r3.x) == 0xdeadbeef)
        out[blockIdx.x*blockDim.x + tid] = r0;                 // anti-DCE
}
// Launch: <<<148, 256>>>, iters=100000, no clock lock (let it boost to 2032).
```

Why it hits 99.8%:
- **LDS.128** (vector load — 16 B per thread per instruction). SASS shows `LDS.128 R8, [R2]`.
- **No writes interleaved** — pure-read pattern keeps the bank port busy on a single phase.
- **True RAW dependency on the address** prevents ptxas from hoisting/eliminating loads.
- **1 block/SM** (148×256 with `__launch_bounds__(256,8)`) = low occupancy but enough warps to saturate the LSU pipe.
- **Short, fast kernel** — completes before thermal throttle drops 2032→1920 MHz.

ncu cross-check: `sm_cycles_active = 99.84%`, `smsp__inst_executed_pipe_lsu = 0.06 inst/cy/SMSP` → 4 SMSPs × 0.06 × 16 B × 2.032 GHz × 148 SMs ≈ 36.95 TB/s. Matches measurement (gap is the 4-LDS-per-iter address-compute overhead). **HIGH**.

---

## Sub-optimal patterns (and why)

| Pattern | BW | Why it falls short |
|---|---:|---|
| `ld.shared.v4.u32` non-volatile | 37.6 TB/s | **Same** as volatile v4 — both emit identical `LDS.128`. The "volatile is faster" claim is **WRONG**. (HIGH, commit d41c38c & 07_smem_peak.md) |
| `float4 v = smem_f4[i]` | 35-36 TB/s | LDS.128, but address compute overhead per iter | 
| `ldmatrix.sync.aligned.x4.b16` | 33-35 TB/s | LDSM via LSU — at ~2.3 cy/warp issue rate (HIGH, commit 4ccda4f). Matches MMA tile-load workload. |
| 8 × scalar `LDS.32` (float-by-float) | 19-26 TB/s | 4× more issue-queue slots for same data volume. The compiler does NOT vectorize a sequence of `smem[k]` reads with non-contiguous indices. (HIGH, commit 4503a17) |
| 4 reads + 1 write per iter (mixed) | 27.2 TB/s | Write port competes with read port. Realistic kernel ceiling for read+modify+write tile work. (HIGH, commit 4503a17) |
| UNROLL=32 + LDS.128 | 18 TB/s | ALU-bound on per-step address arithmetic, NOT LSU-bound (HIGH, 07_smem_peak.md T4) |
| stmatrix.x4.b16 | ~14 B/cy/warp = 36 cy/store | Write-side smem hazards make stmatrix ~14× slower per-instruction than ldmatrix (HIGH, commit ede88fb) |

**`ld.volatile.shared.v4.u32` does NOT bypass any cache or force re-reads on B300.** It only forces the compiler to use the vector form of the load. Same SASS, same BW. (HIGH, SASS-verified in 07_smem_peak.md.)

---

## Bank conflicts — scaling and the stride-33 trick

Source: commit **bce8bf8** (`investigations/banks_proper.cu`). 148 × 128 thr × 10k iter × (4 reads + 1 write) per iter. anti-DCE verified. **HIGH** confidence.

| Stride | Conflict degree | Gops/s | Slowdown vs stride-1 |
|---:|---|---:|---:|
| 1 | none | 2624 | **1.0×** baseline |
| 2 | 2-way | 2498 | 1.05× |
| 4 | 4-way | 1996 | 1.32× |
| 8 | 8-way | 1159 | 2.26× |
| 16 | 16-way | 590 | 4.45× |
| **32** | **32-way (worst)** | **298** | **8.81×** |
| **33** | **none (coprime)** | **2562** | **1.02×** |

Conclusions (HIGH):
- N-way conflict scales roughly N/4 for moderate N, approaching N for full conflict.
- 2-way conflict is ~5% — usually not worth chasing.
- **Stride 33 (coprime with 32) eliminates conflicts.** The classic `tile[N][N+1]` padding trick recovers full BW.
- A separate column-access test (commit 10f2d4a) confirms `[32][33]` gives **5.8× speedup** for column reads vs `[32][32]` (1024 cy → 176 cy).

---

## Per-block SHMEM limits and carveout

Source: commits **ee884ca**, **20291f7**. Device-attribute verified. **HIGH** confidence.

| Limit | Value |
|---|---:|
| `cudaDevAttrMaxSharedMemoryPerBlockOptin` | **232,448 B = 227 KB** |
| `cudaDevAttrReservedSharedMemoryPerBlock` | **1024 B** |
| Total SRAM per SM (L1+SHMEM unified) | **256 KB** |
| Total SRAM chip-wide | 148 × 256 KB = 38 MB |

Requests above 227 KB return `cudaSuccess` **but produce 0 blocks/SM (silent no-op)**. Hard cap is 227 KB.

### Occupancy curve (256 thr/block, full SHMEM carveout)

| SHMEM/block | Blocks/SM | Theoretical max occupancy |
|---:|---:|---|
| 0-16 KB | 8 | 100% (8 × 256 = 2048 thr/SM) |
| 32 KB | 6 | 75% |
| 56 KB | 4 | 50% |
| 64 KB | 3 | 38% |
| 100 KB | 2 | 25% |
| 128-227 KB | 1 | 12.5% |

### Carveout knob (`cudaFuncAttributePreferredSharedMemoryCarveout`)

With 32 KB SHMEM/block requested:

| Carveout % | Blocks/SM at 32 KB |
|---:|---:|
| 0 (MaxL1) | 1 — only 16 KB SHMEM available |
| 25 | 1 |
| 50 | 4 |
| 75 | 5 |
| **100 (MaxShared)** | **6** |

Notes (MED):
- The L1/SHMEM partition is **non-monotonic** in subtle ways (commit 879f942): smem=8-40 KB can disable L1 cache entirely (555 cy = pure L2 latency), while smem=80-160 KB leaves L1 functional (42 cy). The partition depends on which physical SRAM banks are allocated, not just total size.
- Practical: avoid SHMEM in the 8-40 KB band if L1 caching matters.

### "Steal reserved 1 KB" trick (LOW — works, has caveats)

Commits **5e2ac2e**, **f80ba3a**, **b4d6556**, **f76ad82**:
- Tell compiler `MaxDynamicSharedMemorySize=56 KB`, then write to the reserved 1 KB at offsets [0..1023] via raw `st.shared.u8` PTX.
- Net per block: 57 KB, packing **4 blocks × 57 KB = exactly 228 KB** on each SM.
- Verified: 148 SMs × 4 blocks = 592 blocks, **0 corruption**.
- **CAVEATS**: cluster.sync, mbarrier, TMA, async-copy, and PDL all use the reserved 1 KB and **WILL corrupt** the stolen region. Pure compute kernels are safe. Performance: hot-loop access has overhead — best for setup/constant data. Future CUDA versions may add new uses → fragile.

---

## SHMEM atomics

Source: commit **baeef1f**. Single warp (32 threads), measured via `clock64`. **HIGH** confidence.

| Operation | No contention (cy) | 32-way contention (cy) |
|---|---:|---:|
| **INT32 atomicAdd** | **4.6** | **4.6** (zero penalty) |
| FP32 atomicAdd | 85 | 5729 |
| INT32 atomicCAS | 6.2 | — |

- **INT32 smem atomics: 18× faster than FP32** and zero contention overhead — HW processes all 32 lanes simultaneously even when targeting the same address.
- **FP32 contention scaling is super-linear**: 1→85, 2-way→172 (2.0×), 4-way→346 (4.1×), 8-way→1120 (13×), 16-way→2666 (31×), 32-way→5729 (**67× — not 32×**). FP non-associativity forces serialized RMW.
- **Recommendation**: use `redux.sync.add` (56 cy on integer types) instead of FP32 smem atomics where possible. Note: `redux.sync` is **integer-only** on sm_103a (commit 46c73ab).

---

## DSMEM (distributed shared memory across cluster)

Source: investigation **04_dsmem_overhead.md**, commit **870b7c2**. SASS-verified. **HIGH** confidence.

### Latency (1-warp dependent chain, runtime cluster barrier excluded)

| Memory | Cluster size | cy/load | ns/load |
|---|---:|---:|---:|
| Local SMEM | 1 | **28** | 14.6 |
| DSMEM | 2 | 224 | 117 |
| DSMEM | 4 | 201 | 105 |
| DSMEM | 8 | 201 | 105 |

**DSMEM latency = 7-8× local SMEM latency.** Not 0.8% (LICM artifact in old `bench_dsmem.cu`), not 4.7× (wrong FADD-serialized metric in old `dsmem_v2.cu`).

### Throughput (ILP=4, 4 independent dependent chains)

| Memory | cy/load | loads/cy | Ratio |
|---|---:|---:|---:|
| Local SMEM ILP=4 | 7.0 | 0.143 | 1.0× |
| DSMEM ILP=4 | 63.5 | 0.016 | **9.1× slower** |

### Mechanism (HIGH — SASS-verified)

`ld.shared::cluster.u32` with a scalar-register address compiles to **`LD.E` (global memory load), not `LDS`**. The peer SHMEM is reached via a global-address window built from `mapa` + `PRMT` + `IMAD.IADD` + `SR_SWINHI`. The latency penalty (~8×) reflects the L2/interconnect path vs the local crossbar path. The `LDS R,[R+UR]` form only appears when the mapa result lands in a uniform register (UR), which requires both inputs to be warp-uniform AND recognized as such by ptxas.

### DSMEM crash behavior (LOW — workaround documented)

Dependent DSMEM chains crash non-deterministically with "unspecified launch failure":
- Cluster=2: ~50% crash rate at 50+ iter
- Cluster=4: ~50% at 10 iter, 100% at 15+
- Cluster=8: similar

Workaround: small iteration counts (≤8 for cluster=4, ≤50 for cluster=2). Local SMEM never crashes. Likely related to the LD.E/cluster-tracking interaction.

---

## ldmatrix / stmatrix throughput

Source: commits **664a67b**, **4ccda4f**, **ede88fb**, **347208b**. **HIGH** for ldmatrix; stmatrix is **MED** (the 32 cy/warp number was on consecutive same-address stores, which may overstate cost in realistic tensor-output patterns).

### ldmatrix.sync.aligned.{x1,x2,x4}.m8n8.shared.b16 (per warp)

| Variant | Bytes | cy/load | B/cy | Efficiency |
|---|---:|---:|---:|---:|
| x1 | 128 | 28.0 | 4.6 | 1.0× |
| x2 | 256 | 27.0 | 9.5 | 2.1× |
| **x4** | **512** | **29.0** | **17.7** | **3.9×** |

**Use x4 always.** Barely more cycles, 4× the data. `.trans` form has identical throughput. Per-warp issue rate ≈ 0.43 ldmatrix/cy.

All FP4/FP6/FP8 LDSM variants run at the **same 2.30 cy/warp** as standard b16 — Blackwell uses the same HW path regardless of element width. (HIGH)

`x8 b16` is rejected by ptxas. (HIGH)

### stmatrix.sync.aligned.x{1,2,4}.m8n8.shared.b16 (per warp)

| Variant | Bytes | cy/store | B/cy |
|---|---:|---:|---:|
| x1 | 128 | 30.0 | 4.3 |
| x2 | 256 | 32.0 | 8.0 |
| x4 | 512 | 36.0 | 14.2 |

stmatrix is **~14× slower per instruction** than ldmatrix — write-side smem pipeline hazards on consecutive same-address stores. **MED**: realistic tensor-store patterns hit different addresses; per-inst cost likely lower in practice. The "stmatrix DCE'd" issue from commit 6077386 is a separate measurement bug — these numbers are after the bug was fixed.

---

## Findings being retired or corrected

- **"SHMEM peak = 27.2 TB/s = 71%"** (commit 4503a17) — NOT retired. Correct for **read+write mix**; the new 38.4 TB/s is for **pure reads**. Both numbers stand for their respective workloads. Use 38.4 TB/s as the headline SoL; cite 27.2 TB/s as the realistic mixed-workload ceiling.
- **"19.85 TB/s aggregate (52% of theoretical)"** (AUDIT_NOTES) — **RETIRED as a peak number**. Correct as the throughput of a scalar-LDS-×8 access pattern, not a hardware ceiling.
- **"ld.volatile.shared.v4.u32 unlocks 1.8× more BW vs ld.shared.v4.u32"** (B300_PIPE_CATALOG §0) — **WRONG**. Both emit identical `LDS.128`, both deliver 37.63 TB/s. The original comparison conflated **scalar vs vector** with **non-volatile vs volatile**. **RETIRE this explanation.** (HIGH, SASS-verified)
- **"DSMEM only 0.8% slower than local SMEM"** (B300_PIPE_CATALOG §30.H, commit b478bb0) — **WRONG**. LICM hoisted the load out of the loop in both DSMEM and local-SMEM variants; the test was measuring loop overhead. True ratio is 7-8×. SASS in `sass/bench_dsmem_1891246005.sass` proves it. **RETIRE.**
- **"DSMEM 4.7× slower than local SMEM"** (`tests/dsmem_v2.cu`) — **WRONG framing**. Used FADD-serialized accumulator that turned the test into a single-thread latency measurement; the 4.7× number is wrong-by-factor for the bandwidth claim. True latency ratio is 7-8×, throughput ratio is 9×. **RETIRE the 4.7× number.**
- **"DSMEM = 1035 GB/s remote across all cluster sizes"** (commit b478bb0, 5f3edca) — **uncertain**. Likely measuring a workload-specific number, not a hardware peak. Better numbers are in 04_dsmem_overhead.md.
- **"L1 carveout disables L1 above 8 KB SHMEM"** — actually non-monotonic (commit 879f942). MED-confidence finding stands; needs more characterization to predict from `cudaFuncSetAttribute` settings.
- **ldmatrix_test.cu (commit d714801)** — **DCE'd, results invalid**, kept in repo for reference only. The 4ccda4f / 664a67b / ede88fb runs are the valid throughput numbers (these used proper anti-DCE).

---

## Open questions

- **Realistic stmatrix cost in tensor-output patterns** (different addresses, write-coalesced). Current 32-36 cy/warp number is from same-address back-to-back stores — likely too pessimistic for real GEMM epilogues. **MED** — needs a workload-mode test.
- **Why does DSMEM throughput at ILP=4 still pay 9× vs local?** Dependent latency is 8×; an ILP=4 ought to hide more if it's pure latency. The fact that throughput penalty matches latency penalty suggests a per-load cross-SM serialization on the L2 path — not characterized.
- **Sustained vs burst BW transition.** At ~2 ms the chip drops 2032→1920 MHz; further sustained runs (>~8000 iter) settle at **17-21 TB/s** (~50% of 1920 MHz theoretical). The extra 50% reduction beyond clock alone is **unexplained** — power management, but no specific signal pinpointed. Use **short runs (≤5000 iter, ≤2 ms)** for true peak. (07_smem_peak.md §6, MED)
- **Why does DSMEM cluster=2 pay more (224 cy) than cluster=4/8 (201 cy)?** Counter-intuitive — possibly fewer cluster routing options at size 2. **LOW** importance.
- **Bank-conflict atomics.** The 32-way INT atomic test reported zero contention overhead — but bank-aware contention scaling for atomics has not been mapped. **OPEN.**
- **Steal-reserved durability across CUDA toolkit versions.** Currently safe on CUDA 13.x driver / sm_103a; no contract from NVIDIA. **LOW** confidence as a long-term technique.

---

## Verification status (per claim)

| Claim | SASS | ncu | Cross-checked sources |
|---|:---:|:---:|---|
| 38.4 TB/s pure-read peak | YES (LDS.128) | YES (sm_active 99.84%, lsu 0.06 inst/cy/SMSP) | rigor_smem_sol.cu + 07_smem_peak.md |
| volatile == non-volatile (LDS.128) | YES (identical opcodes) | implicit | 07_smem_peak.md apples-to-apples |
| 27.2 TB/s read+write mix | NO SASS dump in commit, but pattern matches | NO | shmem_peak.cu (commit 4503a17) |
| Bank conflict 1×/9× scaling | NO | NO | banks_proper.cu (commit bce8bf8) |
| 227 KB / 1024 B reserved | device-attr query (HIGH) | N/A | reserved_smem.cu, ee884ca |
| Carveout occupancy curve | N/A | implicit (occupancy API) | carveout_smem.cu, 20291f7 |
| INT32 atomic 4.6 cy / FP32 67× | clock64 in-kernel | NO | baeef1f |
| DSMEM 7-8× latency | YES (LD.E vs LDS) | NO | 04_dsmem_overhead.md |
| ldmatrix x4 = 17.7 B/cy | YES (LDSM x4) | NO | 664a67b, 4ccda4f |
| stmatrix x4 = 14.2 B/cy | YES (STSM x4) | NO | 664a67b, ede88fb |

---

## MIO Pipe Architecture — per-SMSP and per-SM caps (HIGH, 2026-04-18 session)

The MIO ("Memory Input/Output") pipe per SM handles **STS, LDS, SHFL, ATOMS,
REDUX (sum/and/or/xor), and BAR**. Discovered by combo experiments where any
two MIO consumers cap at ~1 inst/SM/cy total in any mix.

### Per-SMSP unit rates (high ILP, saturated)

| Op            | per-SMSP cap   | per-SM cap     | SMSPs needed to saturate per-SM |
|---------------|----------------|----------------|---------------------------------|
| SHFL.bfly     | 16 thr-op/cy   | 32 (1 inst/cy) | 2                               |
| SHFL.up/down  | 16 thr-op/cy   | 32             | 2                               |
| SHFL.idx      | ~15 thr-op/cy  | 30 (5% slower) | 2                               |
| LDS.32        | 16 thr-op/cy   | 32             | 2                               |
| ATOM.INC      | 16 thr-op/cy   | 32             | 2                               |
| STS.32        | 8  thr-op/cy   | 32             | 4                               |
| CREDUX.MIN/MAX| 8  thr-op/cy   | 32             | 4 (separate pipe — see below)   |
| REDUX.SUM/AND/OR/XOR | 8 thr-op/cy | **16 (0.5 inst/cy)** | 2 (tight cap) |

Test: `investigations/ninja_ilp_sweep.cu` — clean ILP sweep at 1/2/4 SMSPs
across all 6 ops. Time-scaling sanity check passes (linear in N_iters).

### Vector LDS/STS — fewer SMSPs to saturate SMEM port

|         | 1 SMSP   | 2 SMSPs  | 4 SMSPs  | % of 38.5 TB/s peak |
|---------|----------|----------|----------|---------------------|
| LDS.32  |  9.5 TB/s | 18.9 TB/s| 35.7 TB/s | 93%                |
| LDS.64  | 18.5 TB/s | 33.9 TB/s| 37.5 TB/s | 97%                |
| LDS.128 | **22.9** | **35.4** | **38.0** TB/s | **99%**         |
| STS.32  |  9.2     | 18.5     | 36.9 TB/s | 96%                 |
| STS.64  | 12.5     | 24.9     | 37.7 TB/s | 98%                 |
| STS.128 | 15.1     | 29.3     | 32.4 TB/s | 84% (degraded — RF write port?) |

**LDS.128 with just 2 SMSPs reaches 92% of peak.** Vector LDS lets fewer
SMSPs saturate the SMEM port — important for kernels with limited warp count.

Test: `investigations/ninja_smsp_vec.cu`.

### MIO contention combos (4 SMSPs, 4 chains/warp)

All combinations cap at ~30 thr-op/SM/cy = 1.0 inst/SM/cy:

  STS-only            : 0.97 inst/SM/cy (32 thr-op/SM/cy)
  ATOMS-only          : 0.97 inst/SM/cy
  STS + SHFL          : 0.96 inst/SM/cy total (each ≈ 0.48)
  ATOMS + SHFL        : 0.95 inst/SM/cy
  ATOMS + STS         : 0.99 inst/SM/cy
  ATOMS + STS + SHFL  : 0.93 inst/SM/cy
  LDS + SHFL          : 0.94 inst/SM/cy

Conclusion: STS, LDS, SHFL, ATOMS share ONE per-SM MIO port at 1 inst/cy.
Tests: `investigations/ninja_smsp_combo.cu`, `ninja_smsp_mio.cu`.

### REDUX vs CREDUX (SASS-level distinction)

SASS reveals two distinct opcodes:
  REDUX.SUM/AND/OR/XOR  — uses MIO (caps at 0.49 inst/SM/cy = 1 inst per 2 cy)
  CREDUX.MIN/MAX        — uses DIFFERENT pipe (0.78 inst/SM/cy alone, >1 in combo)

Verified: CREDUX.MIN + SHFL = 1.16 inst/SM/cy total > MIO ceiling 1.0.
Test: `investigations/ninja_redux_credux.cu`.

### Pipes orthogonal to MIO

Tested with X+STS combo time vs max(t(X), t(STS)) and sum:

| Op | t(X) | t(STS) | t(X+STS) | max | sum | verdict |
|----|------|--------|----------|-----|-----|---------|
| FFMA  | 0.21 | 0.81 | 0.81 ms | 0.81 | 1.02 | **INDEPENDENT** |
| MUFU  | 5.60 | 0.81 | 5.87 ms | 5.60 | 6.41 | **INDEPENDENT** |
| DFMA  | 15.33| 0.81 | 15.48 ms| 15.33| 16.14| **INDEPENDENT** |
| HMMA  | 0.45 | 0.81 | 1.08 ms | 0.81 | 1.26 | PARTIAL (40% overlap) |

FFMA, MUFU, DFMA: per-SMSP units, fully orthogonal to MIO.
HMMA (raw mma.sync): partially shares with MIO (~60% serialization).
Test: `investigations/ninja_pipe_matrix.cu`, `ninja_hmma_raw.cu`.

### mma.sync warp-count saturation (S1+S4 resolved)

Per-SMSP tensor units exist (4 per SM). Each saturates at ~0.46 mma/cy with
4+ warps per SMSP for pipeline fill. Peak BF16 = **2.26 PFLOPS = 90.5% of
NVIDIA's 2.5 PF spec** at 16 warps/SM (4 per SMSP). Refutes prior "23% of
spec" claim. Test: `investigations/ninja_mma_warp_sweep.cu`.

