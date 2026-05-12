# SENPAI Research State

- **Last updated:** 2026-05-12 20:06 (post-wave-1-merge, wave-2 launched)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 actively assigned)

## ⭐ Current baseline (PR #1452 merged 2026-05-12 20:02 UTC)

- **val_avg/mae_surf_p:** **100.7659** (best, epoch 14)
- **test_avg/mae_surf_p:** **90.3840** (4-split, all finite)
- **Config:** Transolver baseline + Smooth-L1 (Huber β=1.0) + `data/scoring.py` NaN-safe fix + CosineAnnealingLR(T_max=15)
- **W&B run:** `lo8vp7rj`
- See `BASELINE.md` for the full reproducible spec.

## Most recent direction from human researcher team

None received. Last issue check: 2026-05-12 20:01 UTC, zero open issues. Workflow assumed: drive primary ranking metric `val_avg/mae_surf_p` (and `test_avg/mae_surf_p`) on the TandemFoilSet Transolver baseline within isolated branch `icml-appendix-willow-pai2g-48h-r2`.

## ✓ Resolved infrastructure issues

- **`data/scoring.py` NaN propagation:** FIXED via PR #1452 (frieren's `torch.where(mask, err, zero)`). All future PRs inherit the fix from the merged baseline.
- **`Transolver` `unified_pos=True` constructor bug:** still present in baseline (not in scope for #1452). Will be re-applied as part of PR #1551 (tanjiro stack-test) if it lands.

## ⚠ Active operational note

GraphQL API was rate-limited from ~19:25 UTC; reset at ~19:48 UTC. During the rate-limit window, the 5 wave-1 student pods (alphonse, askeladd, edward, fern, nezuko) saw "no assignments" in their poll output and idled, despite having open assigned PRs. After the reset (~19:53 UTC onwards) all 5 picked up their assignments and started training. Their results will land ~20:20-20:30 UTC if everything works.

Current GraphQL bucket: ~2117/5000 remaining at 20:06 UTC; next reset ~20:48 UTC. Continue using REST helpers (`gh api .../issues/.../comments`, `gh api .../pulls/X --method PATCH`) where possible to preserve GraphQL budget.

## Current research focus

**Wave 1 (closed):** 8 orthogonal single-variable hypotheses against the original advisor-branch baseline. Three completed:

- **#1452 frieren — Smooth-L1 (Huber β=1.0)** → val=100.77 / test=90.38 → **MERGED as baseline**
- #1454 tanjiro — unified-pos ref=8 → val=128.78 / test=117.33 → CLOSED (worse than new baseline; stack-test as #1551)
- #1455 thorfinn — batch=8, lr=7.1e-4 → val=162.39 (e10) → rerun pending

**Wave 2 (in flight):** Stack tests on top of the merged Huber baseline:

| PR | Student | Slug | Hypothesis | Predicted Δ |
|---|---|---|---|---|
| #1551 | tanjiro | `unified-pos-on-huber` | unified_pos=True ref=8 + forward-side encoding, on Huber baseline | −3 to −8% (~92–98 val) |
| #1554 | frieren | `swa-on-huber` | SWA on final 4/15 epochs, swa_lr=1e-4, terminal test eval uses `swa_model` | −3 to −7% (~94–98 val) |

**Wave 1 still in flight (now training on the pre-merge MSE baseline):**

| PR | Student | Slug | Lever |
|---|---|---|---|
| #1446 | alphonse | `schedule-align-baseline` | CosineAnnealingLR(T_max=epochs=10) — was confounded with Huber in #1452 |
| #1448 | askeladd | `slice-num-128` | Double PhysicsAttention slice_num |
| #1449 | edward | `surf-weight-30` | Bias loss toward surface metric |
| #1450 | fern | `mlp-ratio-4` | Restore canonical Transolver MLP FFN capacity |
| #1453 | nezuko | `wider-n-hidden-192` | Widen Transolver hidden dim |
| #1455 | thorfinn | `batch-8-lr-up` (rerun) | Batch=8 + sqrt(2)-scaled lr, --epochs=15 + scoring fix |

These 6 are running on MSE + their respective lever. Since Huber alone gave a ~25% improvement (147.65 → 111.06), any wave-1-MSE hypothesis that lands above ~108 val will likely not beat the new baseline. **However, several of these levers (schedule-align in particular) may still be measuring something distinct from Huber — schedule alignment is implicit in the 15-epoch reruns but isn't explicitly tested.** Evaluate each on its own merits when it lands, not just on the absolute metric.

## Potential next research directions (wave 3+)

Ranked by expected ROI on `val_avg/mae_surf_p` if wave 2 hits expected improvements:

1. **Stack #1551 (unified-pos) + #1554 (SWA) + Huber** — three orthogonal wins compounded. Expected delta if both hit: 100.77 × 0.94 × 0.94 ≈ 89 val.
2. **β sweep on Huber** (β ∈ {0.1, 0.3, 1.0, 3.0}) — β=1.0 was a guess; lower β acts more like L1, higher more like MSE. May yield another ~3-5% by itself.
3. **Surface-aware slice routing** (research-ideas H2) — −5 to −12% predicted but medium implementation effort (thread `is_surface` through every block).
4. **FiLM global conditioning** on Re/AoA/NACA/gap/stagger (research-ideas H5) — −4 to −10% predicted, new module.
5. **Per-sample Re-based loss weighting** (research-ideas H4) — −4 to −9% on val_re_rand specifically.
6. **Surface-only Huber + MSE on volume** — split loss by node type; surface is the headline metric.
7. **Domain-adversarial training** — −3 to −8% on camber OOD specifically (research-ideas).
8. **Per-channel β** — pressure has wider normalized range than Ux/Uy in normalized space.
9. **EMA averaging instead of SWA** — different averaging schedule; cheap variant if SWA doesn't deliver.
10. **Gradient clipping** — defensive; cheap to add, can rescue divergent runs in larger-capacity arms.

The researcher-agent's `RESEARCH_IDEAS_2026-05-12_round2.md` doc on this branch contains H1–H10 with concrete implementation specs.

## Open questions to revisit on review

- **Stacking gain:** when wave-2 results land, compute the gain ratio of each stack vs. predicted compound. If actual gain falls below predicted, the levers may be partially correlated (e.g. unified-pos and Huber both helping mostly via the same OOD samples). That's diagnostic.
- **Per-split divergence post-merge:** with Huber baseline, `val_geom_camber_cruise` (80.90) and `val_re_rand` (93.04) are the easiest splits; the two raceCar splits (val_single_in_dist=119.74, val_geom_camber_rc=109.38) are the hardest. Future architectural levers may target the hard splits specifically.
- **Schedule alignment vs. Huber decoupling:** wave-1 PR #1452 conflates "Huber loss" with "--epochs=15 = T_max-aligned schedule". A pure schedule-alignment-only comparison (alphonse #1446) lands soon — that result will let us attribute how much of the 25%+ wave-1 win was schedule alignment vs. loss reformulation.
- **VRAM headroom:** #1455 thorfinn ran at 84.2 GB peak with batch=8. We have ~12 GB headroom for capacity bumps — useful for future hypotheses.
- **What to do with the 6 wave-1-in-flight PRs that may not beat Huber baseline:** evaluate each individually; some levers (e.g. surf_weight=30) target the metric directly and may still help even on the new baseline.
