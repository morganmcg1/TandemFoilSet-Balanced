# BASELINE — icml-appendix-charlie-pai2i-24h-r1

Active advisor branch baseline. Updated after each merged winner. All
val/test MAE numbers below come from the committed `models/<experiment>/metrics.jsonl`
on the listed PR.

## Current best — PR #3136 (charliepai2i24h1-frieren / surfw25)

- **val_avg/mae_surf_p**: **126.3241** (best at epoch 14; 14 of 50 epochs realized under the per-run `SENPAI_TIMEOUT_MINUTES=30` wall-clock cap)
- **test_avg/mae_surf_p**: **NaN** — same `data/scoring.py::accumulate_batch` × corrupt-GT bug that has affected every PR so far (cruise test sample 20 has 761 `Inf` values in GT `p`). The other three test splits averaged **123.43**. Tanjiro's NaN-safe `eval_test_clean.py` pattern (#3141) should be the standard for cleaning up test numbers until a dedicated `data/scoring.py` fix lands (queued; needs advisor waiver).
- **Per-val-split mae_surf_p** (best epoch 14):
  - val_single_in_dist: 158.79
  - val_geom_camber_rc: 127.26
  - val_geom_camber_cruise: 102.20
  - val_re_rand: 117.04
- **Per-test-split mae_surf_p** (from best-val EMA checkpoint):
  - test_single_in_dist: 136.67
  - test_geom_camber_rc: 117.86
  - test_geom_camber_cruise: NaN (corrupt GT, see above)
  - test_re_rand: 115.75
- **peak_memory_gb**: 42.11
- **Metric artifacts**: `models/model-surfw25-20260515-132819/metrics.jsonl`, `models/model-surfw25-20260515-132819/metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --experiment_name baseline-merged --agent baseline`

### Stacking note (trust-in-orthogonality)

Frieren's measured 126.3241 was on the **pre-#3130** trunk config (`n_hidden=128, n_head=4`) plus `surf_weight=25` (no EMA). The squash-merge layered just the surf_weight line change onto the already-merged wider+EMA config. The active advisor recipe is now **`n_hidden=192, n_head=6, EMA decay=0.999, surf_weight=25.0`** — a 3-axis stack that has never been measured end-to-end. We're trusting orthogonality across width, EMA, and loss-weight. The first round-2 PR to actually be based on the post-#3136 advisor branch will produce the first true measurement of this combined recipe; if it underperforms 126.32 by more than 2-3%, the orthogonality assumption is failing and we'll need a dedicated clean-combined-baseline run.

## Active model configuration

| Component | Value |
|---|---|
| Architecture | Transolver with physics-aware attention |
| n_hidden | **192** (from #3130) |
| n_layers | 5 |
| n_head | **6** (from #3130) |
| slice_num | 64 |
| mlp_ratio | 2 |
| **EMA decay** | **0.999** (from #3137) — applied to eval/test/checkpoint |
| Optimizer | AdamW |
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| **surf_weight** | **25.0** (from #3136) |
| epochs (configured) | 50 |
| schedule | CosineAnnealingLR (T_max=epochs) |
| sampler | WeightedRandomSampler (3-domain balanced) |
| loss | MSE on normalized targets, `vol_loss + surf_weight * surf_loss` |

## Metrics contract

- **Primary ranking (val)**: `val_avg/mae_surf_p` — equal-weight surface pressure MAE across the four val splits.
- **Paper-facing (test)**: `test_avg/mae_surf_p` — same aggregation over the four test splits, computed from the best-val checkpoint.
- **Per-split diagnostics**: `{split}/mae_{surf,vol}_{Ux,Uy,p}` and `{split}/{vol,surf,total}_loss`.
- Direction: **lower is better**.

## Known issues / systemic constraints

1. **Schedule misalignment under 30-min cap.** `SENPAI_TIMEOUT_MINUTES=30` lets ~9-15 epochs complete per run; cosine `T_max=50` only anneals ~18-30% of its schedule. The model is evaluated near peak LR rather than after a low-LR fine-tune. For a paper-quality absolute number we will need a separate round with `--epochs ≈ realized_budget` and `T_max=epochs` so the schedule actually anneals. Thorfinn's send-back of #3144 includes this recipe.
2. **Cruise-test pressure NaN — root cause identified (GT corruption + scoring bug).** Cruise test sample 20 has 761 `Inf` values in ground-truth `p`. `data/scoring.py::accumulate_batch` computes `err = (pred - y).abs()` BEFORE masking, then multiplies by `surf_mask/vol_mask`. IEEE-754 `Inf * 0 = NaN` causes the pressure accumulator to go NaN. **Affects every PR.** Until patched, students should produce a NaN-safe test re-evaluation à la tanjiro's `eval_test_clean.py` (#3141) and report the 3-split mean. Fix is one-line in `data/scoring.py` (`err.nan_to_num_(0.0, posinf=0.0, neginf=0.0)` after the subtraction, or `torch.where(mask, err, zeros)` before the sum), but the file is marked read-only in `program.md` — requires advisor waiver. **Queued as a dedicated bug-fix PR.**

## How students should report

Students commit `models/<experiment>/metrics.jsonl` and a terminal `SENPAI-RESULT`
marker in the PR with both `val_avg/mae_surf_p` and `test_avg/mae_surf_p`.
For the test number, run the NaN-safe re-eval (tanjiro's pattern from #3141)
and report the 3-split finite mean alongside the raw value.

## Compounding

When a PR beats this baseline, it gets merged and this file is updated with
the new best metrics. Future hypotheses are then layered on top of the
new advisor configuration.

## Merge history

| Date | PR | Student | Change | val_avg/mae_surf_p |
|---|---|---|---|---|
| 2026-05-15 12:44 | #3130 | edward | Wider: n_hidden 128→192, n_head 4→6 | 166.5037 |
| 2026-05-15 14:24 | #3137 | nezuko | EMA decay=0.999 on eval/test/ckpt | 129.4217 |
| 2026-05-15 14:35 | #3136 | frieren | surf_weight 10→25 | **126.3241** |
