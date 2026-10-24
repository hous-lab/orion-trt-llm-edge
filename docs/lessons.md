# Lessons from the deployment campaign

Distilled from the project retrospective; each entry follows observation, evidence,
principle. The ones below are the deployment-relevant subset. The number anchors
(E-notes) referenced here live in the project retrospective, not in this repo.

## Numerical verification

1. Deployed-model accuracy must be validated on real inputs. Random inputs are fine
   for shape/build smoke tests only: with random inputs this project read bbox cos
   0.88 while real data showed 0.999 - and random inputs hid an input-ordering bug
   that spilled NaNs into the engine.
2. Multi-input engine interfaces need explicit, auditable input mapping. A zip that
   paired 21 input names with 22 tensors shifted every input by one slot (NaNs);
   index tables or name checks beat implicit ordering, especially at large IO counts.
3. Flattened cosine similarity has a blind spot: it is scale-invariant. An overall
   cos of 0.9995 hid a single canbus token at 20-38% scale deviation, which cost 24%
   plan_L2. Always pair cos with a norm ratio or per-token analysis when scale errors
   are possible.
4. Mixed-splice ablation localizes root cause precisely: replace segments of the
   engine input with official ground-truth values one at a time; the segment whose
   restoration fixes the output (here: the canbus token, cos 0.9897 to 0.9992) is the
   bug.
5. Verify the expensive hypothesis last: one trtexec --fp32 build (72 seconds) proved
   precision was not the problem and saved days of numerics debugging. Cheap
   disproof first, authoritative measurement last (NCU counters to close the case).
6. Reproducing stateful logic (cross-frame memory) needs a field-level checklist
   against the official implementation - one missed field (memory_egopose) stalled
   plan_L2 at 0.857 for two days. State logic is where stateless engines fail.
7. A reference implementation must be stateless and in-place-safe: the PyTorch proxy
   mutated its input (memory_timestamp +=), poisoning every subsequent frame and
   faking an "engine degradation" result. ONNX Runtime graph execution makes a better
   reference.

## Performance on memory-bound platforms (Jetson Orin)

8. Diagnose the bound regime before optimizing kernels. The dominant sign was FP16
   vs FP32 latency being nearly identical (218 vs 217 ms): compute would halve,
   bandwidth doubles the tensor count - equal time means the workload is moving
   bytes, not doing math. Orin LPDDR5 (204.8 GB/s measured) is roughly 4.5x slower
   than a desktop 4090, so every activation-sized transfer is on the critical path.
9. Roofline arithmetic sets the achievable floor. The 12-layer tower at 512 input
   moves an estimated minimum of ~44 GB effective traffic; at 204.8 GB/s that bounds
   e2e at roughly 215 ms against a measured 217 ms - there was no headroom left for
   kernel-level heroics, which redirected effort from micro-optimization to
   architecture (fewer layers, lower resolution, distillation).
10. NCU closes optimization debates. Per-kernel counters ended several threads: the
    ViT MLP kernels run at 82.9% tensor-pipe utilization (compute-bound, near the
    Ampere-class ceiling), elementwise ops are balanced (SM 58% / DRAM 55%), LN
    kernels are DRAM-bound at 69.8% with 91% occupancy (fusion upside measured at
    1-2%, not worth the code), and fused attention runs at 16.3% occupancy
    (226 registers/thread) but is worth under 5% end-to-end. Measure before
    writing a custom plugin.
11. Quantization is not a latency lever on this platform. Within-noise INT8 results
    on x86 (128.8 vs 129.7 ms) plus no gain on Orin (Q/DQ insertion costs what the
    narrower math saves) mean the standard "just quantize it" advice does not
    transfer to bandwidth-bound small accelerators. Weight-only W8A16/W4A16 on the
    7B LLM likewise left prefill flat (64.6 / 65.6 / 64.9 ms) because prefill is
    activation-bound; full W8A8 SmoothQuant halved activation traffic and dropped
    the LLM to 26 ms. Quantize activations, not just weights.
12. Platform results do not transfer. The same INT8 matmul measured 1.70x speedup
    on a 4090 and 0.39x on Orin sm_87; a server 4090 (128 SM, ~920 GB/s) runs the
    identical engine 1.94-2.02x faster than a 16 GB laptop 4090 (76 SM, ~506 GB/s),
    tracking the bandwidth (1.82x) and SM (2.08x) ratios. Quote latencies only
    against the exact device and power mode they were measured on.
13. Power mode is a first-class lever on Jetson. Raising the 6-layer tower from 50W
    to MAXN cut vision-module latency 118.96 to 84.76 ms (-29%) and e2e from ~154
    to ~121 ms (-21%) - a larger, cheaper win than any kernel work attempted in the
    project. Early numbers measured at 50W silently mislead every later comparison;
    lock the power mode before benchmarking.

## Distillation acceptance

14. Freeze what you are not distilling. An earlier 6-layer round retrained the
    planning student jointly with the backbone and regressed to 14.82 m plan_L2 -
    an 18x regression that looked like "6 layers does not work". Feeding both the
    6L and 24L towers into the same untouched 8L student (differing by only 0.34 m)
    located the real culprit: the retrained student had been bent by the small
    distillation set, not the backbone. The working recipe distills the backbone
    only and keeps the student and heads frozen.
15. A/B the student, not just the backbone. The student is hypersensitive to its
    input distribution; a candidate backbone must be evaluated through the frozen
    downstream student, or the experiment measures distillation-side drift instead
    of the compression being tested.
16. Long online distillation diverges numerically: feat=nan appeared around 18k-25k
    steps on the transformer-decoder path. Short runs with frequent checkpointing,
    monitored per-run, beat one long run.
17. Intermediate similarity metrics are training monitors, not acceptance gates. A
    12-layer tower with healthy embedding cos (0.88/0.89/0.85 across the three
    output heads) still shifted detection score calibration (high-confidence recall
    dropped to ~60%), while planning improved. Feature fidelity and task alignment
    decouple; acceptance runs the downstream task metric through the real pipeline.
18. Proxy evaluations degrade every model systematically and can invert rankings.
    Zeroing the cross-frame memory took the 24-layer baseline from 0.868 m to 2.34 m
    proxy plan_L2, and relative ordering between variants shifted with it. Any
    cross-variant comparison in this project runs the full real pipeline with real
    memory evolution, and contaminated frames (polluted labels in the eval slice)
    are excluded by the documented rules.

## Working habits that paid off

19. Run an accuracy gate before any performance work. Everything above rests on
    having a trustworthy baseline metric to compare against; a broken eval harness
    converts every optimization into noise.
20. Keep export depth and config in sync: the exported tensor set (hidden states
    vs logits) must match what the downstream consumer reads, or the pipeline runs
    with stale bindings that look like numerics bugs.
21. "Works, but too slow" and "meets the bar" are different release states. A
    functionally correct engine that misses the latency bar is not deployable;
    state both axes explicitly in comparisons.
22. Hardware differences are not optimization headroom. A 2x gap between two GPUs
    with 2x the SM count and bandwidth is arithmetic, not an optimization backlog.

## Evaluation harness reference

The Lite accuracy numbers in this repo (README, [docs/model_variants.md](model_variants.md)) all come
from one harness: the 173-frame Bench2Drive val subset, evaluated end-to-end with
real images and real cross-frame memory evolution. Of the 173 frames, 133 have
valid future trajectory; the committed per-variant eval JSONs under [lite/results/](../lite/results/)
report the raw (all), no-pollution (valid_no_pollute) and future-valid
(fut_valid_only) subsets separately, and the quoted plan_L2 values are the
future-valid-only subset. Frames whose labels are polluted in the eval
slice are excluded from the quoted metric; the exclusion list and the
zeroed-memory proxy pitfall that motivated it are described in lesson 18.

The 7B numbers use the same Bench2Drive val subset through the TRT-LLM pipeline
described in [docs/pipeline_guide.md](pipeline_guide.md); the official PyTorch reference (0.690) uses
the upstream harness, which is why the two tables are read independently.
