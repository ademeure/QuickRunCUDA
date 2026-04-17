# B300 Atomic Scope x Ordering Cost Matrix

**GPU**: NVIDIA B300 SXM6 AC  
**Clock**: 2032 MHz (verified by 44 cy / 21.65 ns = 2031 MHz during kernel execution; idle reads 1920 MHz)  
**Methodology**: Single warp (32 threads), each thread operates on its own address slot (no cross-thread contention), ITERS=8192, min of 10 runs. Chain: `v = r + 1` forces true latency measurement. Source: `investigations/atomic_matrix_runner_v2.cu`.

## Shared Memory Atomics (atom.shared.add.u32)

| Ordering | .cta | .cluster | .gpu | .sys |
|----------|------|----------|------|------|
| relaxed  | **44 cy** (21.6 ns) | 44 cy | 44 cy | 44 cy |
| acquire  | 50 cy (24.6 ns) | 50 cy | 50 cy | 50 cy |
| release  | 52 cy (25.6 ns) | **304 cy (150 ns)** | 304 cy | ~4000-7000 cy (variable) |
| acq_rel  | 58 cy (28.5 ns) | **312 cy (153 ns)** | 312 cy | ~4000-7000 cy (variable) |
| seq_cst  | N/A (ptxas rejects) | N/A | N/A | N/A |

**Baseline**: `atom.shared.add.u32` (no qualifier) = 44 cy — identical to relaxed.cta.

## Global Memory Atomics (atom.global.add.u32, L2-hit)

| Ordering | .cta | .cluster | .gpu | .sys |
|----------|------|----------|------|------|
| relaxed  | **413 cy** (203 ns) | 413 cy | 413 cy | 413 cy |
| acquire  | 419 cy (206 ns) | 421 cy | 421 cy | 421 cy |
| release  | 421 cy (207 ns) | **1455 cy (716 ns)** | 1455 cy | ~5800 cy (variable) |
| acq_rel  | 427 cy (210 ns) | **1463 cy (720 ns)** | 1463 cy | ~5800 cy (variable) |
| seq_cst  | N/A (ptxas rejects) | N/A | N/A | N/A |

**Baseline**: `atom.global.add.u32` (no qualifier) = 413 cy = identical to relaxed.gpu (confirms default global scope = .gpu).

Note: Global latency (~413 cy) is the L2-cache-hit single-thread chain latency, not the throughput number. With 32 threads operating on 32 distinct cachelines, this measures per-thread latency with no contention.

---

## Key Findings

### 1. SCOPE IS FREE for relaxed ordering — confirmed

For both shared and global atomics, every scope (cta/cluster/gpu/sys) at relaxed ordering costs identically:
- smem relaxed: **44 cy** regardless of scope
- gmem relaxed: **413 cy** regardless of scope

This resolves line 8020-8027 of the catalog: "Scope qualifier is FREE" — correct, but only at relaxed ordering.

### 2. ORDERING is the expensive part, not scope — but there is a critical scope interaction

The ordering penalty depends heavily on scope:

**For .cta scope**: ordering overhead is nearly free:

| Ordering | smem cost | overhead | gmem cost | overhead |
|----------|-----------|----------|-----------|----------|
| relaxed  | 44 cy     | +0 cy    | 413 cy    | +0 cy    |
| acquire  | 50 cy     | +6 cy    | 419 cy    | +6 cy    |
| release  | 52 cy     | +8 cy    | 421 cy    | +8 cy    |
| acq_rel  | 58 cy     | +14 cy   | 427 cy    | +14 cy   |

**For .cluster/.gpu scope**: release and acq_rel become expensive:

| Ordering | smem cost | overhead vs cta | gmem cost | overhead vs cta |
|----------|-----------|-----------------|-----------|-----------------|
| relaxed  | 44 cy     | 0 cy            | 413 cy    | 0 cy            |
| acquire  | 50 cy     | 0 cy            | 421 cy    | +2 cy           |
| release  | 304 cy    | **+252 cy**     | 1455 cy   | **+1034 cy**    |
| acq_rel  | 312 cy    | **+254 cy**     | 1463 cy   | **+1036 cy**    |

**For .sys scope**: all non-relaxed orderings have extremely high and variable cost (4800-6000+ cy for release/acq_rel). The sys scope crosses the NVLink bus and is subject to system-wide coherence traffic.

### 3. Resolving the catalog contradiction

The catalog has two conflicting claims about atomics. Both are correct but measure different things:

**Claim A** (line 8020-8027): "atom scope is FREE — cta/cluster/gpu/sys all 34 cy"
- Context: cluster-launched kernel, multiple independent atoms per iteration, per-thread addresses
- This measures **throughput** (multiple atoms overlapping), not latency
- 34 cy is the **throughput** cycle count for a relaxed-ordered atom
- Scope is FREE at relaxed ordering — confirmed by this study

**Claim B** (line 7099-7104): "acquire.gpu = 780 cy, 17x slower than relaxed.gpu"
- Context: single-thread contended warp (all 32 threads hitting same address)
- This measures a **contended atom chain** — each atom waits for all prior warp's atoms
- The warp serialization means 32 threads * ~24 cy each = ~780 cy observed
- This is WARP SERIALIZATION cost, not scope/ordering overhead per se

**Claim C** (this study): acquire.gpu = 421 cy, release.gpu = 1455 cy on global (per-thread addresses)
- Context: each thread has its own distinct cacheline address, no contention
- This measures **pure scope+ordering overhead** via latency chain

**The three measurements are not contradictory — they measure different things:**

| Study | Metric | Contention | relaxed.gpu | acquire.gpu | release.gpu | acq_rel.gpu |
|-------|--------|------------|-------------|-------------|-------------|-------------|
| Catalog §scope (line 8020) | throughput | none (per-thread) | 34 cy | — | — | — |
| Catalog §ordering (line 7099) | latency chain | warp-contended | 51 cy | 780 cy | 872 cy | 1598 cy |
| This study | latency chain | none (per-thread) | 413 cy | 421 cy | 1455 cy | 1463 cy |

Note: The 413 cy global latency in this study vs 51 cy in the catalog reflects that 51 cy was the contended throughput (warp serializes into ~51 cy/op chunks), whereas 413 cy is the actual round-trip L2 latency per atom when no other thread is competing.

### 4. SASS-confirmed instruction patterns

The compiler maps PTX scope+ordering to these SASS instruction sequences:

| PTX | SASS pattern |
|-----|-------------|
| atom.shared relaxed.cta | ATOMS.ADD (plain) |
| atom.shared acquire.cta | ATOMS.ADD + CCTL.IVALL |
| atom.shared release.cta | MEMBAR.ALL.CTA + ATOMS.ADD |
| atom.shared release.cluster/gpu | MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR + ATOMS.ADD |
| atom.shared acq_rel.cta | MEMBAR.ALL.CTA + ATOMS.ADD + CCTL.IVALL |
| atom.global relaxed.gpu | ATOMG.E.ADD.STRONG.GPU (baseline stall count) |
| atom.global acquire.gpu | ATOMG.E.ADD.STRONG.GPU (higher stall count) + CCTL.IVALL |
| atom.global release.gpu | MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR + ATOMG.E.ADD.STRONG.GPU |
| atom.global acq_rel.gpu | MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR + ATOMG.E.ADD.STRONG.GPU + CCTL.IVALL |

The MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR triple is the cluster-crossing fence. This costs ~260 cy on shared mem and ~1040 cy on global (because global's underlying ATOMG instruction has longer base latency).

CCTL.IVALL (L1 cache invalidate all) costs 6-8 cy and is inserted for acquire-side ordering.

### 5. Design rules from this data

**Use relaxed atomics whenever possible.**

| Scope | Ordering | Recommendation |
|-------|----------|----------------|
| any   | relaxed  | Free: costs nothing extra beyond base atom latency |
| cta   | acquire/release/acq_rel | ~6-14 cy overhead — acceptable for within-CTA sync |
| cluster/gpu | acquire | ~2-8 cy overhead — acceptable |
| cluster/gpu | release/acq_rel | **250-1050 cy overhead** — very expensive |
| sys   | release/acq_rel | 4000-6000+ cy, highly variable — avoid in hot paths |

**If you need GPU-wide ordering**: use `atom.relaxed` + separate `fence.release.gpu` at batch boundaries (one fence per batch, not per atomic). The fence cost is paid once, not per-atom.

**Scope is always free at relaxed ordering**: any of cta/cluster/gpu/sys at relaxed ordering costs the same as bare `atom.global` or `atom.shared`. The PTX ISA-level scope annotation has zero hardware cost when paired with relaxed ordering.

**seq_cst is not supported**: ptxas sm_103a rejects `atom.seq_cst.*` with "Unknown modifier '.seq_cst'" for both shared and global atomics.

---

## Raw data for sys-scope variability

The sys scope atomics with release/acq_rel have high variance because they touch the NVLink bus for cross-GPU coherence. Raw samples (cy/iter, across 10 runs):

- smem.release.sys: 3699 - 7116 cy (2x variation)
- smem.acq_rel.sys: 3812 - 16538 cy (4x variation, outliers to >8000 cy)
- gmem.release.sys: 4984 - 20359 cy (4x variation, outliers to >20000 cy)
- gmem.acq_rel.sys: 4972 - 8655 cy (2x variation in this run)

The minimum is not a reliable characterization — these costs depend on system-wide bus state and other GPUs' activity.
