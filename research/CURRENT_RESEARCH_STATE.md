# SENPAI Research State

- **Updated:** 2026-04-28 21:58 UTC
- **Track:** `icml-appendix-willow-pai2e-r1` (TandemFoilSet ICML appendix, Willow PAI2E Round 1)
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1`
- **Most recent direction from human researcher team:** _(none yet — no GitHub issues)_

## Current best

**PR #773 (EMA decay=0.99) — MERGED:** `val_avg/mae_surf_p = 119.35`, `test_avg/mae_surf_p = 108.79`  
EMA is now part of the advisor baseline. All future student runs will use `--ema_decay 0.99` by default.

## Current research focus

Round 1 results are coming in. Two PRs reviewed this cycle:
- **PR #773 (fern — EMA):** MERGED. EMA decay=0.99 beats live-model selection by +6% on val_avg across all 4 splits. Geometry OOD splits benefit most. New baseline.
- **PR #777 (thorfinn — log-Re jitter):** CLOSED. Input augmentation slows convergence; in our 14-epoch budget the regularization benefit never materializes. Control (no-jitter) wins on all splits including the targeted val_re_rand.

**Budget insight:** Every run so far times out at epoch 13-14 (30-min wall clock). The EMA decay=0.99 is optimal for this budget because it has a fast-enough half-life to integrate useful signal. All future hypotheses should be evaluated with `--ema_decay 0.99` as the default, and schedule/optimizer hypotheses should account for the mismatch between the 50-epoch cosine schedule and the ~14-epoch actual run.

**Still in flight (WIP):**

1. **alphonse #769** — Huber loss (loss reformulation)
2. **askeladd #770** — surface-aware slice routing (architecture)
3. **edward #846** — unmodified baseline reference
4. **frieren #774** — wider model (n_hidden=256, slice_num=128)
5. **nezuko #775** — LR warmup + grad clip (optimization)
6. **tanjiro #776** — deeper model (n_layers=8)

**Newly assigned (this cycle):**
- **fern** — surface weight scan (surf_weight ∈ {20, 50, 100}) with EMA decay=0.99
- **thorfinn** — schedule alignment: shorten cosine T_max to match actual budget (~14 epochs) and OneCycleLR comparison

Once the unmodified baseline (PR #846) lands, all Round 1 results will have a valid reference number.

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
