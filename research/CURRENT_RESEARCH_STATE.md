# SENPAI Research State

- 2026-05-12 18:00
- No human researcher directives yet (no open issues)
- Round 5 of the Charlie no-W&B logging ablation arm (SENPAI_TIMEOUT_MINUTES=30, local JSONL only)

## Current research focus

**Theme: Loss alignment and capacity tuning on the Transolver baseline.**

The out-of-the-box Transolver has three identifiable inefficiencies relative to the primary metric (`val_avg/mae_surf_p`):

1. **Loss/metric mismatch:** The training loss weights all three output channels (Ux, Uy, p) equally in the surface term, but the ranking metric is surface pressure only. The overall surf/vol balance (surf_weight=10) was not tuned for this dataset.
2. **Re dynamic range:** Per-sample target std varies 1–2 orders of magnitude (high-Re vs low-Re), causing high-Re samples to dominate the gradient. The model under-optimises for low-Re regimes tested in `val_re_rand`.
3. **Architecture underexplored:** The baseline uses n_hidden=128, slice_num=64. Neither was ablated; both may be sub-optimal. The model also lacks gradient clipping (universal practice for transformer training) and weight-averaging at the end of training.

## Active experiments (8 WIP PRs)

| PR | Student | Hypothesis | Key change | Expected gain |
|---|---|---|---|---|
| #1459 | alphonse | H1: Raised surf_weight | surf_weight 10→20 | Low risk, 3–8% |
| #1463 | askeladd | H2: SWA from epoch 25 | Weight averaging last 25 epochs | 2–6%, OOD splits |
| #1470 | edward | H3: Instance-norm loss | Per-sample std normalisation of loss | 3–10%, val_re_rand |
| #1474 | fern | H4: Per-channel pressure weight | 3× on p channel within surf_loss | 3–8% |
| #1478 | frieren | H5: Wider model n_hidden=192 | Capacity increase, 4.7M params | 2–6%, OOD camber |
| #1481 | nezuko | H6: slice_num=128 | Finer physics partitioning | 3–7% |
| #1483 | tanjiro | H7: Gradient clipping 1.0 | Stability for temperature param | 1–4% |
| #1487 | thorfinn | H8: Surface skip branch | Local geometry → output bypass | 2–7% |

All 8 students active. No idle GPUs.

## Open uncertainties to resolve from round 5 results

1. **Baseline epoch count in 30 min.** First completed PR will tell us how many epochs the baseline finishes. Determines whether depth increases (H9: n_layers=7) are feasible.
2. **Baseline val_avg/mae_surf_p.** Round 5 has no prior merges. First terminal result establishes the floor.
3. **OOD geometry vs. Re difficulty.** If val_geom_camber splits are substantially harder than val_re_rand, architecture changes (capacity, skip branches) should be prioritised. If val_re_rand is hardest, dynamic-range loss tricks (H3) should be prioritised.
4. **Whether surf_weight and loss-channel-weighting effects compound.** H1 (global surf_weight) and H4 (per-channel p-weight within surface) are orthogonal and can likely both be merged if both win.

## Reserve hypotheses (assign next if students become idle after round)

- **H9: Deeper model (n_layers=7, lr=3e-4)** — assign if H5/H6 (capacity) improve
- **H10: Warmup + cosine LR** — assign if H7 (grad clipping) shows high unclipped norms
- **H11: Batch=8 + BF16 mixed precision** — assign if throughput is the bottleneck
- **H12: Transolver++ local adaptive correction** — medium-risk architectural PR, highest potential impact from literature

## Potential next research themes (after round 5 results)

1. **Compose winning changes:** H1 + H4 (loss alignment) + H7 (clipping) likely stack cleanly into one combined PR.
2. **Larger architecture:** if H5 (n_hidden=192) wins, try n_hidden=256 or n_layers=7 as follow-up.
3. **Transolver++ local adaptive (H12):** if capacity alone doesn't close the OOD geometry gap, local feature conditioning (residual by-pass similar to U-Net boundary encoding) is the next architectural move.
4. **Ensemble / model soup:** multiple SWA checkpoints averaged for test — free improvement with no training cost.
5. **Physics-informed auxiliary losses:** divergence-free constraint on interior velocity field, boundary layer stress consistency — would require verifying these are derivable from the mesh features in the current contract.
