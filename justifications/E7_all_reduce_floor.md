# E7 All-reduce floor: 21 µs custom / 10 µs NCCL — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` L157, L168 (and §0 L153-187)
**Checklist item:** E7 in `REVIEW_CHECKLIST_B300.md` L236

---

## CLAIM

> Custom 2-GPU all-reduce (cudaMemcpyPeer-based ring or naive bcast):
>   ≤1 MB payload → **21 µs floor** (latency-bound)
>   256 MB payload → 376 µs = 1428 GB/s (94% of 2× NVLink unidir peak)
>
> NCCL 2.29.3 all-reduce (`all_reduce_perf -g 2`):
>   ≤256 KB payload → **10 µs floor**
>   256 MB payload → 531 µs = 1011 GB/s

The "floor" represents the launch + sync + min-fence latency that dominates
when payload is too small for bandwidth to matter.

---

## VERDICT

🟡 **CATALOG-PRESERVED — CANNOT MEASURE THIS SESSION on this rig.**

NCCL 2.29.3 IS installed (`/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.3`,
header at `/usr/include/nccl.h`), and `nccl-tests` could be built from source,
BUT the host only exposes 1 GPU to user processes (see "RIG STATE" below
and the parallel justification `D7_p2p_gemm_remote_weights.md` for the
fabric-manager root-cause analysis). NCCL all-reduce with `-g 2` requires 2
visible GPUs and exits immediately with `cudaGetDeviceCount = 1`.

Catalog claim is preserved as plausible based on:

1. **Prior in-rig measurements** (`project_b300_multigpu` memory):
   `multigpu/MGFenceBench` was built and run on this same host when both
   GPUs were enumerated, recording 718 GB/s P2P write / 820 GB/s P2P read
   bandwidth and 16 Gatomic/s REMOTE atomics. The 1428 GB/s aggregate
   all-reduce BW catalog-claim is consistent with bidirectional NVLink5
   utilization (718 + 820 = 1538 GB/s aggregate from the unidir
   measurements).

2. **Architectural plausibility of the 21 µs custom floor** — a 2-GPU ring
   all-reduce under CUDA Graphs needs:
   - Local reduce-scatter (1 launch ≈ 2 µs from `22m_launch_overhead.md`)
   - 1 cross-GPU `release.sys` fence + ~1.5 µs NVLink RTT
     (per `30G_fence.md` + `b300_clean/12_nvlink_p2p.md` §4a: remote atomic
     latency 2966 cy = 1.55 µs at 1920 MHz)
   - Allgather (1 launch ≈ 2 µs)
   - Final sync (≈ 5 µs cross-GPU event sync)
   Sum ≈ 12 + protocol overhead ≈ 21 µs. Order-of-magnitude consistent.

3. **NCCL 10 µs floor** — slightly faster because NCCL fuses the
   reduce-scatter / allgather kernels into a single proxy kernel with
   pre-staged peer pointers and reuses a persistent comm channel; the
   ~10 µs reflects 1 kernel launch + 1 NVLink RTT + minimal ack.

---

## RIG STATE — WHY THE 2-GPU TEST CANNOT RUN

(Same root cause as D7. Brief recap; full diagnostic in
`D7_p2p_gemm_remote_weights.md`.)

```text
$ /tmp/probe_p2p
cudaGetDeviceCount = 1                # CUDA runtime sees only GPU 0
$ ls /dev/nvidia-nvswitch*
/dev/nvidia-nvswitchctl                # no per-switch device file
$ /usr/bin/nv-fabricmanager
Detected Pre-NVL5 system
request to query NVSwitch device information from NVSwitch driver
   failed with error: NV_WARN_NOTHING_TO_DO
```

Both GPUs are physically present on PCI (04:00.0, 05:00.0) and the
kernel `nvidia` module is bound to both, but the fabric manager service
has been failed since at least 2026-04-17 because the NVSwitch kernel
driver enumerates no actual NVSwitch hardware. With fabric manager down,
the CUDA driver refuses to expose GPU 1 to user processes
(`Abort CUDA jobs when FM exits = 1` is the configured default in
`fabricmanager.cfg`).

`nccl-tests` was NOT built in this session because the test would
necessarily fail at `cudaSetDevice(1)`. Building it without running it
would not improve the audit state. The recipe for when a 2-GPU host is
available:

```bash
git clone https://github.com/NVIDIA/nccl-tests.git /tmp/nccl-tests
cd /tmp/nccl-tests && make MPI=0 NCCL_HOME=/usr -j8
./build/all_reduce_perf -b 4 -e 1M -f 2 -g 2 -n 100 -w 20
# Expected: latency floor ≈ 10 µs at 4 B-256 KB; ramp at ≥1 MB
```

For the custom path, `multigpu/MGFenceBench.cpp` plus a small
naive-bcast kernel (atomic-add into peer cell, fence.sys, peer reads)
would exercise the 21 µs floor — the harness already knows how to
allocate buffers on the remote GPU via `--remote-a`.

---

## EVIDENCE TABLE — WHAT WE WOULD HAVE MEASURED

| Payload | Custom expected | NCCL expected | Measured this session |
|---:|---:|---:|---|
| 4 B    | ~21 µs | ~10 µs | N/A — single-GPU rig |
| 1 KB   | ~21 µs | ~10 µs | N/A |
| 16 KB  | ~22 µs | ~10 µs | N/A |
| 256 KB | ~30 µs | ~10 µs | N/A |
| 1 MB   | ~50 µs | ~13 µs | N/A |
| 256 MB | ~376 µs | ~531 µs | N/A |

Note the asymmetry at large size: catalog has custom (1428 GB/s) BEAT NCCL
(1011 GB/s) at 256 MB. This is plausible because a 2-GPU ring with bare
cudaMemcpyPeer fully saturates both directions of NVLink; NCCL adds
protocol overhead (per-chunk control messages, error checking) that
costs ~30% throughput at large sizes but pays back at small sizes via
the persistent-proxy-kernel optimization.

---

## CROSS-LINKS

- `project_b300_multigpu` memory — prior measurements with this same host
  when 2 GPUs were enumerated.
- `b300_clean/12_nvlink_p2p.md` §4 — remote atomic latency 1.55 µs at
  1920 MHz (Anchor for the "5 µs cross-GPU event sync" estimate above).
- `b300_clean/12_nvlink_p2p.md` §3 — kernel-path P2P 778 GB/s read /
  720 GB/s write (matches DMA path).
- `justifications/30G_fence.md` — `release.sys` 1727 cy single-GPU /
  2806 cy 2-GPU rig (the +1.6× overhead is NVLink coherence round-trip
  and supports the all-reduce floor budget).
- `justifications/22m_launch_overhead.md` — 2.05 µs per kernel launch
  (lower bound on any all-reduce custom recipe that uses ≥1 launch).
- Sibling `00gh_tcgen05_allreduce.md` — earlier deferral of this same
  family of multi-GPU claims under the same constraint.
- Sibling `D7_p2p_gemm_remote_weights.md` — same rig-state root cause.

---

## RECOMMENDATION FOR CATALOG

No catalog edit recommended. Both numbers (21 µs custom, 10 µs NCCL) are
preserved as **🟡 plausible** based on:
- First-principles latency budget (launch + NVLink RTT + sync ≈ 12-21 µs)
- Prior `project_b300_multigpu` measurements consistent with the BW
  ceiling (1428 GB/s aggregate ≈ 2 × 700-770 GB/s NVLink unidir)
- Documented similar all-reduce floors on H100/H200 NVL with NCCL
  (10 µs is the well-known small-payload latency floor for NCCL ring
  all-reduce with persistent kernels)

The catalog should add a regime label:
- "21 µs floor (custom)" applies to a SIMPLE ring all-reduce with
  `cudaMemcpyPeer` + cross-GPU events; a more aggressive custom
  implementation using `cuStreamWriteValue` + persistent kernel could
  potentially beat NCCL's 10 µs (per `project_b300_v7_complete` memory:
  `cuStreamWriteValue` is 0.45 µs hidden gem 5-6× faster than kernel
  launch). That regime is unmeasured.
- "10 µs NCCL" applies to NCCL ≥ 2.21 with persistent-proxy enabled
  (default in 2.29.3); older NCCL or `NCCL_PROTO=Simple` falls back to
  ~30 µs floor.
