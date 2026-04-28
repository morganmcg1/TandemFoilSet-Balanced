# SENPAI Research State

- **Updated:** 2026-04-28 19:40 UTC
- **Track:** `icml-appendix-willow-pai2e-r1` (TandemFoilSet ICML appendix, Willow PAI2E Round 1)
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1`
- **Most recent direction from human researcher team:** _(none yet — no GitHub issues)_

## Current research focus

Round 1 is in flight across 8 hypotheses. PR #771 (uncertainty weighting) has been reviewed and closed as a dead end — the mechanism redistributes capacity away from the pressure channel, which is exactly backwards for our metric. A critical NaN-propagation bug in `data/scoring.py` and `train.py` was found by the student and fixed in commit `49c55ed` on the advisor branch; all future runs will have correct `test_geom_camber_cruise` metrics.

No unmodified baseline exists yet. `willowpai2e1-edward` has been assigned PR #846 to run the clean unmodified default config and establish the Round 1 reference number.

**Active Round 1 hypotheses (WIP):**

1. **Loss reformulation** (alphonse — Huber)
2. **Architecture / capacity** (frieren — width=256, slices=128; tanjiro — depth=8/10; askeladd — surface-aware slice routing)
3. **Optimization / training stability** (nezuko — LR warmup + grad clip; fern — EMA weights)
4. **Data augmentation** (thorfinn — log(Re) jitter)
5. **Unmodified baseline reference** (edward — PR #846, clean default config)

Once the unmodified baseline lands (PR #846), we will have a true reference number for all Round 1 comparisons.

Surface-pressure MAE is the ranking metric, so each hypothesis is scored against effect on `val_avg/mae_surf_p` and follow-up paper-facing `test_avg/mae_surf_p`. Per-split disagreements are flagged as information (which split a hypothesis helps tells us about its mechanism).

## Potential next research directions (Round 2+)

Pending Round 1 outcomes, candidate themes for Round 2:

- **Compounding wins.** If Huber+EMA+grad-clip all win independently, stack them. Architecture and optimization wins are usually orthogonal.
- **Pressure-channel-only training tail.** A second-stage fine-tune that drops the Ux/Uy loss entirely — only the metric-of-record is optimized.
- **Better surface conditioning.** Cross-attention between surface and volume nodes; surface-pinned skip connections; explicit boundary-layer inductive biases.
- **Stronger geometry generalization.** Random foil reflection (cruise-style symmetric flow), camber/thickness perturbation augmentation, mesh subsampling for regularization.
- **Architectural alternatives.** Swap PhysicsAttention for Galerkin attention, or try a hybrid Transolver + GNN message-passing module on surface nodes.
- **Multi-task curriculum.** Train Ux/Uy first, freeze, fine-tune p — or domain-specific heads (raceCar single vs. tandem vs. cruise).
- **Loss shape on tails.** If Huber wins at delta=2.0, try log-cosh or asymmetric losses tuned to pressure's positive/negative skew.
- **Mixed precision + larger effective batch.** AMP + gradient accumulation to free wall-clock for higher epoch counts.
- **Test-time scoring with TTA.** Mesh perturbation TTA averaged at inference.

## Standing constraints

- 30 min wall-clock per run (`SENPAI_TIMEOUT_MINUTES`), 50-epoch cap.
- 96 GB VRAM per GPU, batch_size=4 default; meshes up to 242K nodes.
- No edits to `data/`. All augmentation/sampling lives in `train.py`.
- One hypothesis per PR. Compound only after each isolated win is verified.
- Prefer common-recipe changes that survive across all four splits over hacks that only improve one.
