# SENPAI Research State

- **Last updated:** 2026-05-12 (post-first-review)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0

## Most recent direction from human researcher team

None received. Last issue check: 2026-05-12 (post-first-review), zero open issues. Workflow assumed: drive primary ranking metric `val_avg/mae_surf_p` (and `test_avg/mae_surf_p`) on the TandemFoilSet Transolver baseline within isolated branch `icml-appendix-willow-pai2g-48h-r2`.

## ⚠ Active infrastructure issue

`data/scoring.py` `accumulate_batch` propagates NaN/Inf from a corrupt GT sample (`test_geom_camber_cruise/000020.pt`, contains `-inf` in 761 nodes of the `p` channel) into the channel-level test surface-pressure MAE — making `test_avg/mae_surf_p` come back as NaN. Discovered independently by PRs #1454 and #1452. The fix is one line (`torch.where(mask, err, 0.0)` or equivalent `nan_to_num`). Both tanjiro and frieren are now executing the fix in parallel as part of their re-runs; whichever lands first wins, the other rebases trivially. **All wave-1 results will likely have NaN `test_avg/mae_surf_p` until this lands.**

There is also a constructor bug in `Transolver` where the `unified_pos=True` branch was a 3D-Transolver copy (`ref**3`) that the 2D forward pass never built encoding for. Tanjiro's PR also includes the train.py fix for this (switch to `ref**2`, add `forward`-side encoding, plumb `mask`). Lands together with the scoring fix.

## Current research focus

Round 2 of the Charlie-vs-Willow 48h logging ablation. The advisor branch starts clean — **no prior PRs and no BASELINE.md** — so the first wave's job is to fan out across orthogonal levers and establish reference numbers, not to chase a known frontier.

The most consequential observation from inspecting `train.py`: cosine `CosineAnnealingLR(T_max=epochs)` is wired to `--epochs=50`, but the 30-minute wall-clock cap almost certainly stops training in 5–12 epochs depending on mesh load. With `T_max=50`, LR barely anneals before the run ends — the optimizer never gets its exploration→exploitation transition. **Aligning T_max with the realistic epoch count is the leading "free win" hypothesis** and the schedule-alignment baseline (alphonse) tests it cleanly.

## Wave 1 hypotheses (all opened 2026-05-12, `--epochs=10` to align cosine T_max)

| PR | Student | Slug | Axis | Predicted Δ |
|---|---|---|---|---|
| #1446 | alphonse | `schedule-align-baseline` | Optimizer schedule | −3 to −10% |
| #1448 | askeladd | `slice-num-128` | PhysicsAttention granularity | −2 to −5% |
| #1449 | edward | `surf-weight-30` | Loss reformulation | −3 to −8% |
| #1450 | fern | `mlp-ratio-4` | FFN capacity | −2 to −6% |
| #1452 | frieren | `smooth-l1-loss` | Robust loss (Huber) | −3 to −10% (mostly val_re_rand) — **first result: val=111.06 (leader), test=NaN (sent back for scoring fix + 15 epochs)** |
| #1453 | nezuko | `wider-n-hidden-192` | Width capacity | −3 to −7% |
| #1454 | tanjiro | `unified-pos-ref8` | Positional encoding | −3 to −8% (esp. geom-OOD) — **first result: val=147.65, test=NaN (sent back for scoring fix + 15 epochs)** |
| #1455 | thorfinn | `batch-8-lr-up` | Effective batch + sqrt-scaled lr | −2 to −6% |

Coverage axes: schedule, attention granularity, loss weighting, robust loss, width, FFN ratio, positional encoding, batch+lr. Capacity along depth (`n_layers`) is intentionally left for wave 2 once wave 1 reveals which capacity dimensions actually move the metric.

## Potential next research directions

When wave 1 finishes, candidates for wave 2 ranked roughly by expected return on the primary metric:

1. **Stack winners** — compose any wave-1 wins that are obviously orthogonal (e.g. `schedule-align + slice-num-128 + surf-weight-X`) and confirm gains stack.
2. **Depth bump (`n_layers=6` or `7`)** — only after width/MLP-ratio wave-1 capacity readout.
3. **EMA / SWA averaging at end of training** — cheap, often stacks with any tuning win.
4. **Surface-only auxiliary head** — predict surface fields with a separate small head that gradient-flows only on surface nodes. Targets the primary metric directly.
5. **Re-aware loss balancing** — weight per-sample loss by `1 / per-sample y_std` to neutralize high-Re dominance (alternative to Huber).
6. **AoA/chord-flip augmentation** — physics-safe augmentations to reduce geometry-OOD generalization gap on `val_geom_camber_*`.
7. **Mixed precision (`bf16`)** — likely speeds up epochs ≥1.5× → more epochs in 30-min cap → more anneal → better convergence.
8. **Geometric/Fourier positional features for AoA, gap, stagger** — encode the tandem-foil scalar features as cyclic/Fourier embeddings before concat.
9. **Gradient clipping** — defensive; cheap to add and can rescue divergent runs in larger-capacity arms.
10. **Layer-wise lr or AdamW betas tweak (`beta2=0.95`)** — small effect but cheap and stackable.

The researcher-agent finished and wrote `RESEARCH_IDEAS_2026-05-12_round2.md` on this branch. Top picks for wave 2 (subject to wave-1 results): **FiLM global conditioning** on Re/AoA/NACA/gap/stagger (highest predicted ROI, −4 to −10%), **SWA** (−3 to −7%, near-zero risk), **surface-aware slice routing** (−5 to −12% but medium implementation), **domain-adversarial training** (−3 to −8% on camber OOD), **per-sample Re-based loss weighting** (−4 to −9% on val_re_rand).

## Open questions to revisit on review

- **Actual epoch wall time:** wave 1 results should tell us how many epochs fit in 30 min for each config. Use that to set the canonical `--epochs` for wave 2.
- **VRAM headroom:** do any wave-1 arms approach the 96 GB ceiling? If we have, e.g., 40 GB used we have plenty of room to push capacity in wave 2.
- **Per-split variance:** if wave-1 winners help one split (e.g., `val_re_rand`) but regress another (e.g., `val_single_in_dist`), that's a split-divergence signal that should be flagged before merging.
- **Schedule alignment counterfactual:** since every wave-1 PR includes `--epochs=10`, we cannot separate "schedule alignment" gain from each individual lever. Alphonse alone has only schedule alignment; everyone else co-varies. To attribute, future rounds may need a single-arm ablation against the merged alphonse baseline.
