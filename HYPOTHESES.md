# Pre-Registered E2E Hypotheses

This file was written before launching the end-to-end OPD-GRPO runs in this
workspace. The current validation pass intentionally uses one seed only:
`seed=42`. Multi-seed runs remain required before any final statistical claim.

1. OURS Pass@1 >= B3 Pass@1 - 1pt, so correctness is not sacrificed.
2. OURS Pass@8 >= B1 Pass@8 - a small margin, while B2 Pass@8 is much lower than B1.
3. OURS peak memory is lower than B3 peak memory; OURS net step time is no worse than B3.
4. B2 and/or B3 show late-stage instability; OURS does not.
5. B4 Pass@1/Pass@8 < OURS, and B4 <= B3. Routing on OPD divergence is the refutation target.
6. OURS > B4b, showing the GRPO sparse-support routing criterion is informative beyond random routing.
7. OURS- isolates the routing gain; OURS minus OURS- isolates the compression gain.

Risk acknowledgement: if B4 is approximately equal to OURS end-to-end, the
gradient-level artifact may not transmit to downstream loss. In that case the
paper framing should shift from "end-to-end better" to "diagnostic methodology
for safe gradient compression / routing source."
