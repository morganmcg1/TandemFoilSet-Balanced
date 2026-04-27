# SENPAI Research State — willow-r5

- **Last updated:** 2026-04-27 15:35 UTC
- **Most recent direction from human team:** No active issues — no overrides at this time.

## Current research focus

**Round 1 — establish willow-r5 single-variable baselines.** Eight diverse hypothesis families are out for solo verification on the bare Transolver baseline. Each PR tests ONE controllable variable so attribution stays clean for an eventual ICML appendix table.

### Why this round, this way

- The willow-r5 track starts from the bare Transolver default — none of the parallel `kagent_v_students` improvements are merged in. Replicating each parallel-track winner solo gives us un-aliased ablations for the appendix.
- Family coverage in round 1: loss (H1, H8), conditioning/data-repr (H3, H9), capacity/arch (H4, H5, H6), optimizer/systems (H7).
- All eight are single-variable so the round-2 compound chain (Huber + Fourier+FiLM + sn=16 + nl=3 + SwiGLU + AMP, the parallel track's converged recipe) can be applied with each component's marginal gain measurable.

### Round 1 assignments (in flight — pending PR creation)

| Student | Hypothesis | Family | Predicted Δ |
|---------|-----------|--------|-------------|
| willow5-frieren | H1 huber-loss-delta-sweep | loss | −10 to −20% |
| willow5-alphonse | H3 fourier-pe-film-re | conditioning | −15 to −25% |
| willow5-edward | H4 slice-num-down-sweep | capacity | −5 to −10% |
| willow5-nezuko | H5 n-layers-down-sweep | capacity | −3 to −8% |
| willow5-fern | H6 swiglu-feedforward | architecture | −3 to −7% |
| willow5-tanjiro | H7 amp-bf16-throughput | optimizer/systems | −2 to −6% |
| willow5-thorfinn | H8 surf-weight-sweep | loss | −3 to −8% |
| willow5-askeladd | H9 domain-aware-conditioning-tokens | conditioning (NEW) | −5 to −12% |

## Potential next directions (round 2+)

- **Compound chain.** Apply the parallel track's converged recipe (Huber + Fourier+FiLM + sn=16 + nl=3 + SwiGLU + AMP) and report each component's marginal contribution.
- **Reserve ideas (round 2):** H2 loss head-to-head (MSE/L1/Huber clean 3-bar), H10 learnable-output-scale-per-channel, H11 attn-only dropout, H12 grad-clip-only.
- **Augmentation revival:** parallel track closed horizontal-flip aug as a dead end (#15) under its baseline — but it is physics-exact for the aerial cruise domain. Worth re-attempting after the loss/architecture floor is established.
- **Output reparameterization:** asinh on pressure was closed (#9) but a per-domain rescale aware of `y_std` magnitude jumps (10× per Re decade) hasn't been tried.
- **Conditioning extensions:** if H3 wins, FiLM on AoA / NACA(M, P, T) / gap+stagger as additional conditioning streams.
- **Best-checkpoint test discipline.** All round-1 PRs must report `test_avg/mae_surf_p` at the best `val_avg/mae_surf_p` checkpoint, not the terminal epoch. The trainer already does this — confirm in PR review.

## Plateau watch

Not in plateau yet (round 1 is the first round). Plateau protocol activates after 5 consecutive non-improving rounds.
