# D7 P2P GEMM remote weights via NVLink: zero penalty — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` L178-183 (and §0 L153-187)
**Checklist item:** D7 in `REVIEW_CHECKLIST_B300.md` L221

---

## CLAIM

> "P2P GEMM remote weights via NVLink: zero penalty (1.00-1.01× slowdown)"
> i.e. cuBLAS BF16 GEMM with weights placed in peer-GPU HBM accessed via NVLink
> shows essentially the same wall-clock as the local-weights baseline.

Cited mechanism: cuBLAS L2-tiles GEMM into ≤126 MiB chunks; after the first
fetch over NVLink, subsequent inner iterations hit local L2 not peer HBM,
making the "remote-vs-local" distinction invisible.

---

## VERDICT

🟡 **CATALOG-PRESERVED — CANNOT MEASURE THIS SESSION on this rig.**

The host has 2 physical B300 SXM6 AC GPUs (PCI 04:00.0 and 05:00.0, kernel
`nvidia` driver bound to both, both UUIDs valid in `/proc/driver/nvidia/gpus/`),
but only GPU 0 is exposed to CUDA user processes. The 2-GPU measurement
requested by the audit task therefore cannot be performed in this session.

Catalog claim is preserved as plausible based on:
1. Prior in-rig measurements when both GPUs were available
   (`project_b300_multigpu` memory: 718 GB/s P2P write, 820 GB/s P2P read,
   NV18 NVLink5 fabric working).
2. The proposed mechanism (cuBLAS L2-tiling) is sound — at M=N=K=4096 BF16
   the weight matrix is 32 MiB and fits inside L2 (126 MiB) with room for
   activations and accumulators, so after the cold first fetch every inner
   tile hits local L2 at ~20 TB/s rather than going back to peer HBM at
   ~7 TB/s (or NVLink at ~770 GB/s). Marginal cost of the cold fetch is
   amortized over ~2K reuses per element.

---

## RIG STATE — WHY THE 2-GPU TEST CANNOT RUN

Direct evidence captured in this session:

```text
$ nvidia-smi -L
GPU 0: NVIDIA B300 SXM6 AC (UUID: GPU-219ab314-7ddc-ea5f-7cea-86315d45b67c)
$ CUDA_VISIBLE_DEVICES=0,1 nvidia-smi -L
GPU 0: NVIDIA B300 SXM6 AC (UUID: GPU-219ab314-7ddc-ea5f-7cea-86315d45b67c)
$ /tmp/check_devs    # cuInit + cuDeviceGetCount
cuDeviceGetCount = 1
Device 0: NVIDIA B300 SXM6 AC
$ /tmp/probe_p2p     # cudaGetDeviceCount via runtime
cudaGetDeviceCount = 1
$ lspci | grep -i nvidia
04:00.0 3D controller: NVIDIA Corporation Device 3182 (rev a1)
05:00.0 3D controller: NVIDIA Corporation Device 3182 (rev a1)   <- physically present
$ ls /proc/driver/nvidia/gpus/
0000:04:00.0  0000:05:00.0                                       <- both bound
$ cat /proc/driver/nvidia/gpus/0000:05:00.0/information | head -2
Model: NVIDIA B300 SXM6 AC
GPU UUID: GPU-add062e2-de33-1d91-dc77-0df9b9a765d3
```

Both GPUs exist on PCI and are bound to the `nvidia` kernel module, but the
CUDA driver / `nvidia-smi` only enumerate GPU 0. Root cause:

```text
$ systemctl status nvidia-fabricmanager
× nvidia-fabricmanager.service - failed (exit-code) since 2026-04-17
$ /usr/bin/nv-fabricmanager -c /usr/share/nvidia/nvswitch/fabricmanager.cfg
Detected Pre-NVL5 system
request to query NVSwitch device information from NVSwitch driver
   failed with error:WARNING Nothing to do [NV_WARN_NOTHING_TO_DO]
$ ls /dev/nvidia-nvswitch*
/dev/nvidia-nvswitchctl                # no /dev/nvidia-nvswitch0
$ lspci -nn | grep -i 10de
04:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:3182]
05:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:3182]
                                        # no NVSwitch PCI device IDs present
```

The `fabricmanager-start.sh` precheck classifies the system as "Pre-NVL5"
(meaning it expects NVSwitches), then asks the NVSwitch kernel driver for the
switch list and gets `NV_WARN_NOTHING_TO_DO` because no switches are
enumerated on PCI. With fabric manager refusing to start, the CUDA driver
hides GPU 1 from user processes (`Abort CUDA jobs when FM exits = 1`,
`fabricmanager.cfg` default).

The audit task's premise "BOTH GPUs now available" is therefore not actually
satisfied on this host. Recovery would require either (a) physically
re-seating / replacing the missing NVSwitch boards, (b) editing
`fabricmanager-start.sh` precheck to bypass the NVSwitch query and run in
direct-PCIe-P2P mode (unsupported configuration on SXM6), or (c) booting on
a different host with a functional NV18 fabric. None of these are
in-session-actionable.

---

## WHAT THE EXISTING HARNESS WOULD DO IF GPU 1 WERE LIVE

The repo already has `multigpu/MGFenceBench.cpp` from the
`project_b300_multigpu` campaign. It performs the per-buffer placement
(`--remote-a` puts buffer A on GPU 1, accessible from GPU 0 via P2P) and
event-timed launches that the D7 measurement would need. To extend it to a
GEMM rather than a generic kernel one would:

1. Replace the NVRTC custom kernel with a `cublasLtMatmul` call, with
   the weight buffer (matrix B, conventionally) allocated on the remote GPU
   and registered via `cudaDeviceEnablePeerAccess`.
2. Run M=N=K=4096 BF16 (4 + 4 + 4 = 12 MiB problem, weight matrix 32 MiB)
   plus M=N=K=8192 BF16 (256 MiB weight — exceeds L2 → exposes the L2-tile
   theory directly).
3. Wrap the matmul in a `cudaGraph` to amortize launch overhead, run
   100+ iterations under cudaEvent timing.
4. Parallel ncu pass: `dram__bytes_read.sum.per_second` (local HBM
   utilization) + `nvlink__rxbytes` (NVLink ingress on GPU 0).
5. Hypothesis to test: at 4096³ BF16, slowdown ≤ 1.01× because weight tile
   fits in L2; at 8192³ BF16 (weight 256 MB > 126 MB L2), slowdown
   significant because the L2-reuse argument fails.

This investigation is preserved for a future session that has functional
2-GPU enumeration.

---

## CROSS-LINKS

- `project_b300_multigpu` memory — confirmed prior 718 GB/s W / 820 GB/s R
  P2P bandwidth on this rig with both GPUs enumerated; supports the catalog
  NVLink ceiling.
- `b300_clean/12_nvlink_p2p.md` §3 — kernel-path P2P read 778 GB/s = 9× slower
  than local HBM (7 TB/s); the L2-tiling argument is what bridges this 9×
  gap to "zero" in the GEMM regime.
- `b300_clean/B300_CANONICAL_REFERENCE.md` L178-183 — original claim source.
- Sibling justification `00gh_tcgen05_allreduce.md` (which already preserved
  this same family of multi-GPU claims under the same constraint).

---

## RECOMMENDATION FOR CATALOG

No catalog edit recommended. The "1.00-1.01× remote-vs-local" claim is
preserved as **🟡 plausible** from the L2-tiling first-principles argument
plus the verified `project_b300_multigpu` NVLink measurements on this rig.
The catalog should retain a regime label noting it requires:
- Tile size ≤ L2 capacity (126 MiB), and
- Sufficient compute reuse per fetched element to amortize the cold NVLink
  hop (≥ ~hundreds of FLOPs per byte fetched is comfortable for BF16 GEMM
  at K ≥ 1024).

For very-large-N GEMMs where the weight matrix exceeds 126 MiB
(BF16 weight matrix needs 8192² × 2 = 128 MiB just for one operand at
N=K=8192), the "zero penalty" claim should NOT be assumed — the L2
amortization argument breaks down and one expects a substantive slowdown
proportional to (HBM_BW / NVLink_BW) ≈ 9×.
