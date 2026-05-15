# SENPAI Research State

- **Last updated:** 2026-05-15 (initial state for `icml-appendix-charlie-pai2i-24h-r2`)
- **Most recent research direction from human researcher team:** none yet (no open issues).
- **Current research focus and themes:** round 1 is in flight — 8 students testing 8 orthogonal hypothesis families covering architecture capacity, loss reformulation, surface bias, position encoding, schedule, and throughput.

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`, round-2 advisor branch for the Charlie launch. Local JSONL metrics only. PRs target this branch.

## Round 1 — 8 PRs in flight (assigned 2026-05-15)
| PR | Student | Family | Hypothesis |
|----|---------|--------|-----------|
| #3179 | alphonse | Arch (width) | n_hidden 128 → 192 |
| #3183 | askeladd | Arch (depth) | n_layers 5 → 8 |
| #3205 | edward | Arch (attention) | slice_num 64 → 192, n_head 4 → 8 |
| #3208 | fern | Loss | MSE → SmoothL1 (Huber) |
| #3214 | frieren | Loss/Bias | surf_weight 10 → 30 + 2× pressure channel weight |
| #3216 | nezuko | Feature | 32-frequency Fourier features over (x, z) |
| #3220 | tanjiro | Schedule | 100 epochs, linear warmup 5 + cosine, lr 5e-4 → 7e-4 |
| #3223 | thorfinn | Throughput | BF16 autocast + batch_size 4 → 8 |

These are designed to be largely orthogonal so winners can compound across rounds. Each is one-axis-at-a-time so attribution is clean.

## Round 2 candidates (from researcher-agent literature search)
The researcher-agent produced 15 well-motivated hypotheses (H1–H15) in `research/RESEARCH_IDEAS_2026-05-15_initial.md`. The agent's top-ranked picks (cheapest, highest expected gain per LOC) are:

| ID | Direction | Notes |
|----|-----------|-------|
| H4 | Per-channel loss weighting (p upweighted) | 2-line change; aligns train signal with primary metric. Overlap with frieren (#3214) but variants worth testing. |
| H15 | Gradient clipping + AdamW selective decay | 10 lines; transformer best-practice. Stacks with any architecture winner. |
| H2 | Per-sample output scale normalization | Targets per-sample y-std variability (factor 40×). Most relevant to `val_re_rand`. |
| H1 | FiLM global-parameter conditioning | Targets `val_geom_camber_*` splits. Literature anchor: BlendedNet++ (arXiv 2512.03280). |
| H3 | Separate surface/volume decoder heads | Removes shared-decoder bottleneck. |
| H10 | Hierarchical two-level PhysicsAttention (coarse + fine) | Captures both boundary layer and large-scale field. |
| H13 | Log-scale pressure loss in train only | Compresses pressure dynamic range. |

Full list is in `research/RESEARCH_IDEAS_2026-05-15_initial.md` — round 2 will pick from these (and from round-1 winners' suggested follow-ups) once round 1 results land.

## Open questions / decisions to make in round 2
- Which scaling axis (width / depth / attention slots) gives the most return? Answer comes from comparing #3179, #3183, #3205.
- Does the loss reformulation (Huber, #3208) actually help the high-Re tail more than naive surf-weight bias (#3214)?
- Does Fourier position encoding (#3216) preferentially benefit the geometry-interpolation splits as predicted?
- Does the longer 100-epoch schedule (#3220) outperform the 50-epoch baseline on a converged checkpoint, or is the cosine cooled-off region producing diminishing returns?

## Plateau watch
Not applicable yet — no completed experiments on this branch. We will reassess after round 1 returns 4+ results.
