# SENPAI Research Results — icml-appendix-charlie-pai2e-r3

## 2026-04-28 22:30 — Round 1 review summary (PRs #830-#837)

Round 1 dispatched 8 hypotheses across the default Transolver baseline. Two PRs (#832 edward, #834 frieren) remained WIP at review time. Of the 6 reviewed, **only PR #835 (nezuko, MAE/L1 loss) beat the placeholder baseline**, and is now the merged baseline.

| PR  | Student   | Hypothesis                                  | val_avg/mae_surf_p | Δ vs new baseline (104.058) | Decision      |
|-----|-----------|---------------------------------------------|--------------------|-----------------------------|---------------|
| 835 | nezuko    | MAE/L1 loss (replace MSE)                   | **104.058**        | merged (new baseline)       | **MERGED**    |
| 831 | askeladd  | surf_weight 10 → 50                         | 122.96             | +18.2%                      | sent back     |
| 836 | tanjiro   | mlp_ratio 2 → 4                             | 126.996            | +22.0%                      | closed        |
| 833 | fern      | 5-epoch warmup + lr=1e-3                    | 145.25             | +39.6%                      | closed        |
| 837 | thorfinn  | pressure channel weight [1,1,5]             | 144.08             | +38.5%                      | closed        |
| 830 | alphonse  | larger model n_hidden=256/n_layers=7        | 183.06             | +76.0%                      | closed        |
| 834 | frieren   | dropout=0.1 (attn + FFN)                    | 139.08             | +33.7%                      | closed        |

### PR #835 — Nezuko: MAE/L1 loss (MERGED, new baseline)
- Branch: `charliepai2e3-nezuko/mae-loss`
- Hypothesis: MSE over-penalizes high-Re pressure outliers; switching to MAE makes training robust and improves surface pressure metric.
- Results: `val_avg/mae_surf_p=104.058`, `test_avg/mae_surf_p=92.608` (NaN-sample-skipped workaround).
  - `target/runs/mae-loss-metrics/metrics.jsonl`
- Conclusion: Confirmed — MAE is superior to MSE for this surrogate. MAE is now the baseline loss. As a bonus, nezuko also implemented the `data/scoring.py` NaN-poisoning workaround (skip non-finite GT samples) directly in `train.py`, fixing the `test_geom_camber_cruise` evaluation for all subsequent experiments on this branch.

### PR #831 — Askeladd: surf_weight=50
- Hypothesis: Surface metric drives the score; up-weighting surface loss should focus the model.
- Result: 122.96 (still improving at 14/50 epoch cutoff). The MAE loss change (#835) addresses the same pressure-tail concern more directly. **Sent back** for an MAE-based variation.

### PR #836 — Tanjiro: mlp_ratio=4
- Result: 126.996 val, 117.46 test. Doubled FFN params (0.66M→0.99M), epochs slowed to ~149s. No improvement. **Closed.**

### PR #833 — Fern: 5-epoch warmup + lr=1e-3
- Result: 145.25. 5 of 14 epochs spent in warmup (36% of budget), wasting effective training time within the 30min cap. **Closed.**

### PR #837 — Thorfinn: pressure channel weight [1,1,5]
- Result: 144.08. Effectively double-weights pressure on top of `surf_weight=10`, over-emphasizing pressure at the expense of velocity learning. **Closed.**

### PR #834 — Frieren: dropout=0.1
- Result: 139.08 val, 125.29 test (with NaN-sample workaround). 13/50 epochs.
- Frieren independently identified and worked around the scoring.py NaN-poisoning bug — same fix already merged via #835. Heavy regularization on an under-fit 14-epoch model adds noise without payoff. **Closed.**

### PR #830 — Alphonse: larger model (256/7/8)
- Result: 183.06. OOM at batch_size=4 (~94GB) forced fallback to bs=2, only 6/50 epochs at ~5.7min each. Larger model cannot be trained within budget without bf16/AMP. **Closed.**

### Cross-cutting findings from Round 1
1. **MAE loss is a clean win** — every future experiment must build on it. The fix is in `train.py`.
2. **The 30-min/14-epoch budget is the binding constraint.** Schedules tuned for 50 epochs (cosine T_max=50, 5-epoch warmup) are mistuned. Future experiments must adjust schedules to the realized ~14-epoch budget.
3. **scoring.py NaN bug** is now worked around in `train.py` (skip non-finite GT). All future students inherit this fix automatically by branching from the advisor branch.
4. **Capacity scaling needs AMP** — the default model already uses ~22GB per sample. Any architecture larger than (128, 5, 4) needs bf16 mixed precision to fit at bs=4 in the budget.
5. **Camber-holdout split is hardest** (val_geom_camber_rc=116.84 vs val_geom_camber_cruise=76.93 in MAE baseline) — the rc-camber unseen-front-foil generalization gap is the largest source of error and the highest-leverage place to focus.
