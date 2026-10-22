# Roofline analysis: where the milliseconds actually are

This page condenses the profiling campaign behind the numbers in the README: what is
memory-bound vs compute-bound on each platform, what quantization can and cannot move,
and why certain tempting optimizations were measured and rejected. Every claim below
was measured on the named device with NSYS (kernel timings) + NCU (per-kernel hardware
counters), not inferred.

## Orion-Lite on Jetson AGX Orin (64 GB)

Measured LPDDR5 bandwidth: 204.8 GB/s. The first thing to check on Orin is whether the
workload is bandwidth-bound or compute-bound, because the two have opposite remedies.
The quickest experiment: compile the same engine in FP16 and FP32. Latency was
identical (218 vs 217 ms on the 12L@512 vision engine) - the classic bandwidth-bound
signature, and consistent with the 4090-to-Orin latency ratio (6.2x) tracking the
bandwidth ratio (~5x).

Breakdown of the 217 ms 12L@512 vision-module latency on Orin:

| Component | Time |
|---|---|
| ViT MLP GEMMs | 57 ms |
| MHA / attention | 34 ms |
| Tensor transfers (copies, window slicing) | 52 ms |
| LayerNorm + other elementwise | 55 ms |
| Remaining matmul/attention glue | 18 ms |

NCU per-op verdicts (E-note, 12L@512 engine):

| Op class | Key counters | Verdict |
|---|---|---|
| MLP GEMMs (45% of time) | tensor pipe 82.9%, SM 83% | compute-bound, near the Ampere ceiling (~85%) - nothing left |
| Elementwise chains (softmax+LN fusion, 20+ ops) | SM 58%, DRAM 55% | balanced, TRT already fuses deeply - no cross-kernel fusion headroom |
| LayerNorm mean/var | DRAM 69.8%, L2 80.6%, occ 91% | memory-bound; fusing saves 1-2% e2e at most |
| Fused attention | occupancy 16.3% (226 reg/thread), SM 47%, DRAM 59% | the one real candidate: 1.5x occupancy would be ~5% e2e - not worth a custom plugin |
| Window-slice chain | SM 69%, occ 87% | compute-bound, architectural - only a model change moves it |

Conclusion: after the resolution cut and the layer reduction, the 12L engine was within
a few percent of the hardware floor, and every remaining candidate was quantified and
rejected on cost/benefit. The 85.2 ms final variant came from model-side changes
(6 layers, 384 input), not from further kernel work.

## Quantization vs the bottleneck type

Quantization only pays where the bottleneck lives. The two chains split cleanly:

- Lite vision engines (both x86 and Orin): weight-only quantization (W4A16/W8A16)
  shrinks weight traffic but leaves activation traffic untouched; activations dominate
  a ViT workload. Measured: calibrated INT8 ran within noise of FP16 on x86
  (128.8 vs 129.7 ms) and no gain on Orin (Q/DQ overhead cancelled the GEMM savings). The
  shipped INT8 tier exists for completeness; FP16 is the recommended tier.
- 7B LLM prefill: the reverse. W8A16/INT4 weight-only did not move prefill (64.6 vs
  65.6/64.9 ms - the GEMMs are activation-bandwidth-bound at this batch size), while
  W8A8 SmoothQuant cuts activation traffic and drops the LLM part 2.5x (64.6 to
  26.0 ms).

Platform transfer is not free: a torch INT8 matmul measured 1.70x on the 4090 but
0.39x on Orin sm_87 (the fused INT8 path is not supported there), so an INT8 win on
desktop says nothing about Jetson until it is measured there. And technical speedup is
not system viability: even with a working INT8 GEMM, the 7B LLM on Orin sits around
350 ms - the platform verdict was no, independent of any kernel result.

## Server vs laptop 4090

The 7B numbers in this repo were taken on an RTX 4090 server GPU (24 GB, 128 SM,
~920 GB/s measured). The same engines on a 16 GB laptop 4090 (76 SM, ~506 GB/s) run
1.94-2.02x slower, matching the hardware ratio (bandwidth 1.82x times compute 2.08x) -
a pure hardware difference, not a tuning gap. When comparing across GPUs, quantify the
spec gap first (bandwidthTest + cudaGetDeviceProperties) so hardware is not mistaken
for headroom.
