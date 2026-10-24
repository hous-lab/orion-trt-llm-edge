# Power modes on Jetson AGX Orin

Orin power modes move the numbers more than most kernel-level work. Measured on the
6L@448 Lite variant (trtexec, 2026-08 re-measurement):

| Metric | MODE_50W | MODE_MAXN | Delta |
|---|---|---|---|
| Vision module | 118.96 ms | 84.76 ms | -29% |
| Student head | 16.75 ms | 15.71 ms | -6% |
| End-to-end | ~154 ms | ~121 ms | -21% |

The 50W mode also explains an otherwise-confusing piece of history: earlier rounds of
the project recorded the 6L vision module at ~119 ms and concluded the 100 ms bar was
out of reach at that resolution; those numbers were all 50W measurements. On MAXN the
same engine is 84.76 ms, and the final 6L@384 variant re-verified on the shipped
artifacts at 81.2 ms end-to-end (the earlier inputs-resident harness summed 69.7 ms
vision + 15.5 ms student = 85.2 ms).

Practical notes:

- Set the mode explicitly on both the host and the device side (nvpmodel -q to check,
  sudo nvpmodel -m 0 for MAXN, plus sudo jetson_clocks to pin clocks) and record it
  with every benchmark - a latency table without the power mode is not reproducible.
- Clock pinning matters for A/B comparisons: unpinned clocks let thermal state bleed
  into the numbers. Lock before measuring, and re-check the mode after reboots (it is
  persistent, but worth verifying).
