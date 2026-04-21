# V10: DSMEM WRITE BW — 4× slower than reads

## Measurements (st.shared::cluster.u32, 144 blocks × 128 thr × 10K iters)

### Cluster size sweep (NL=8)

| Cluster | Read BW (prior)  | Write BW (new)   | Ratio (R/W) |
|---------|------------------|-------------------|--------------|
| 2       | 48.5 TB/s        | **11.8 TB/s**     | **4.1×**     |
| 4       | 32.6 TB/s        | 6.5 TB/s          | 5.0×         |
| 8       | 32.6 TB/s        | 6.0 TB/s          | 5.4×         |

### Cluster=2 write ILP sweep

| NL | BW          | Speedup vs NL=1 |
|----|-------------|------------------|
| 1  | 5.4 TB/s    | 1.00×            |
| 4  | 11.7 TB/s   | 2.17× (saturated) |
| 8  | 11.8 TB/s   | 2.20×            |
| 16 | 11.9 TB/s   | 2.21×            |

**Writes saturate at NL=4**, unlike reads which scale to NL=8-16.

## Key findings

1. **Reads are 4-5× faster than writes** for DSMEM at all cluster sizes.
2. **Write BW saturates early** (NL=4) — can't push higher with more ILP.
3. **Cluster=2 optimal for both**, but writes show only 4% of cluster=2
   read BW at cluster=2 write config.

## Practical implications

**For cluster GEMM/reductions:**
- Prefer **producer = local, consumer reads peer** pattern (read-dominant)
- AVOID patterns where CTA writes to peer's SMEM frequently
- If write-dominant, cluster=2 is still best but expect ~12 TB/s ceiling

**For cooperative algorithms:**
- Read-based reductions (each CTA reads from neighbors) → use DSMEM
- Write-based scatter (CTA writes to all peers) → consider local SMEM + sync

## Why are writes slow?

Hypothesis (unverified, needs HW spec reference):
- Writes propagate through cluster interconnect WITH invalidation
- Must maintain coherence across cluster
- Read-only bus has no invalidate/ordering — fast

## Confidence

HIGH for the 4-5× read/write ratio.
MED for absolute write peak 12 TB/s (may also have L1 artifact).

## Combined DSMEM picture

| Operation       | Cluster=2  | Cluster=8 |
|-----------------|------------|-----------|
| Read NL=8       | 48.5 TB/s  | 32.6 TB/s |
| Read NL=16      | 58.7 TB/s  | 42.3 TB/s |
| **Write NL=8**  | **11.8 TB/s** | 6.0 TB/s |
| Write NL=16     | 11.9 TB/s  | ~same      |

For symmetric cluster communication (both R and W), **read-side throughput bounds** the min side.