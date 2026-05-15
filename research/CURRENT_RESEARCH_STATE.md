# SENPAI Research State

- **Last updated:** 2026-05-15 14:30 (after PR #3208 merge; fern's round-2 assigned)
- **Most recent research direction from human researcher team:** none (no open issues).
- **Current research focus and themes:** 7 round-1 PRs still in flight; fern now on round-2 optimizer PR. De facto baseline established at `val_avg/mae_surf_p` = **116.61** via Huber loss.

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`, round-2 advisor branch for the Charlie launch. Local JSONL metrics only. PRs target this branch.

## Established baseline
- **val_avg/mae_surf_p = 116.61** (PR #3208, Huber loss, best epoch 13, 14 epochs / 30-min cap)
- Per-split: single 161.69 | geom_rc 117.56 | geom_cruise 85.67 | re_rand 101.53
- Loss form: SmoothL1 (Huber β=1.0) — merged into HEAD. All subsequent experiments must beat 116.61.

## Active PRs
| PR | Student | Family | Hypothesis | Status |
|----|---------|--------|-----------|--------|
| #3179 | alphonse | Arch (width) | n_hidden 128 → 192 | WIP round 1 |
| #3183 | askeladd | Arch (depth) | n_layers 5 → 8 | WIP round 1 |
| #3205 | edward | Arch (attention) | slice_num 64 → 192, n_head 4 → 8 | WIP round 1 |
| #3208 | fern | Loss | MSE → SmoothL1 (Huber) | **MERGED** — new baseline |
| #3214 | frieren | Loss/Bias | surf_weight 10 → 30 + 2× pressure channel weight | WIP round 1 |
| #3216 | nezuko | Feature | 32-frequency Fourier features over (x, z) | WIP round 1 |
| #3220 | tanjiro | Schedule | 100 epochs, linear warmup 5 + cosine, lr 5e-4 → 7e-4 | WIP round 1 |
| #3223 | thorfinn | Throughput | BF16 autocast + batch_size 4 → 8 | WIP round 1 |
| #3276 | fern | Optimizer | Gradient clip (max_norm=1.0) + AdamW selective decay + NaN guard | WIP round 2 |

## Round 2 candidate pool (from researcher-agent, `research/RESEARCH_IDEAS_2026-05-15_initial.md`)
| ID | Direction | Notes |
|----|-----------|-------|
| H2 | Per-sample output scale normalization | Targets per-sample y-std variability (40×). Most relevant to `val_re_rand`. |
| H1 | FiLM global-parameter conditioning | Targets `val_geom_camber_*` splits. |
| H3 | Separate surface/volume decoder heads | Removes shared-decoder bottleneck. |
| H4 | Per-channel loss weighting (p upweighted) | 2-line change; see also frieren #3214 results. |
| H10 | Hierarchical two-level PhysicsAttention | Captures boundary layer + large-scale field. |
| H13 | Log-scale pressure loss in train only | Compresses pressure dynamic range. |

H15 (gradient clip + selective decay) is now assigned to fern (#3276).

## Open questions
- Which scaling axis (width / depth / attention slots) gives the most return? → #3179, #3183, #3205.
- Does surf-weight bias (#3214) complement or compete with Huber loss (now baseline)?
- Does Fourier pos-encoding (#3216) preferentially help the geometry-interpolation splits?
- Does 100-epoch schedule (#3220) beat the 30-min-capped 14-epoch Huber baseline?
- Does BF16 + batch8 (#3223) give meaningful wall-clock savings and variance reduction?
- Does gradient clipping + selective decay (#3276) move val_avg/mae_surf_p 1–4% from 116.61?

## Plateau watch
Not applicable — only 1 result in so far. Reassess after 4+ round-1 results land.
