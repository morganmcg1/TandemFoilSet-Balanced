# SENPAI Research State
- 2026-04-29 (updated — all 8 students active; label fix on PR #1064; PR #1028 already closed)
- No directives from human researcher team
- Branch: icml-appendix-charlie-pai2e-r1

## Current Best (val_avg/mae_surf_p)

**61.5855** — PR #1050 (edward): PSN+epochs=30 on full compound (nl3/sn16), WITH --per_sample_norm, epoch 22/30 (timeout-terminated, still falling)
- test_avg/mae_surf_p: **54.3573**

Reproduce:
```bash
cd target/ && python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 30 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm
```
*(Note: `n_layers=3` and `slice_num=16` are hardcoded in the `model_config` dict in `train.py` — not CLI flags.)*

## Merged Winner Chain (cumulative stacking)

| PR | Description | val_avg/mae_surf_p | Delta |
|----|-------------|-------------------|-------|
| Baseline (MSE) | Default train.py | 126.88 | — |
| #788 | Huber loss (delta=1.0) | 115.6496 | -8.85% |
| #827 | + surf_weight=30 | 109.5716 | -5.26% |
| #808 | + bf16 + n_hidden=256 + n_head=8 + epochs=12 | 104.1120 | -4.97% |
| #882 | + EMA(decay=0.999) | 103.2182 | -0.86% |
| #1005 | + n_layers=3, slice_num=16 (reference arch) | 94.6541 | -8.31% |
| #795 | + per_sample_norm (normalize Huber loss by per-sample std) | 90.4014 | -4.51% |
| #1015 | epochs=12→24 (dropped --per_sample_norm, still lower) | 66.8085 | -26.1% |
| #1050 | PSN+epochs=30: re-added --per_sample_norm at epochs=30 | **61.5855** | **-7.8%** |

## Active Experiments (8 WIP)

| PR | Student | Hypothesis | Status | Notes |
|----|---------|-----------|--------|-------|
| #1064 | edward | epochs=36+PSN: extend training from 30→36 epochs, still falling at ep22/30 | Running | Highest priority. Val was still falling at timeout. epochs=36 expected to yield further improvement |
| #1070 | thorfinn | Huber delta re-sweep {0.25, 0.5, 1.0, 2.0} with PSN at epochs=30 | Running | Prior sweep (PR #1028) was without PSN at epochs=12/24. Optimal delta expected to shift under PSN |
| #998 | frieren | slice_num=128 on compound baseline at epochs=24 | Running | OOM recovered with PYTORCH_ALLOC_CONF. sl=128 unlikely to beat new best but useful architecture data |
| #1011 | alphonse | surf_weight sub-10 sweep (1/3/5/7) on compound baseline at epochs=24 | Running | Prior baseline was 66.8085 (no PSN); new best is 61.5855 (PSN+ep30) |
| #942 | nezuko | EMA decay sweep 0.99/0.995 vs 0.999 on compound baseline at epochs=24 | Running | Prior baseline was 66.8085; new best is 61.5855 (PSN+ep30) |
| #1018 | askeladd | LR sweep: lr ∈ {1e-3, 2e-4, 5e-4} on nl3/sn16 compound at epochs=24 | Running | Prior baseline was 66.8085; new best is 61.5855 (PSN+ep30) |
| #1038 | tanjiro | LR warmup 2-epoch at epochs=24 | Running | Prior baseline was 66.8085; new best is 61.5855 (PSN+ep30) |
| #1030 | fern | slice_num=32 Goldilocks at epochs=24 | Running | Prior baseline was 66.8085; new best is 61.5855 (PSN+ep30) |

## Key Technical Insights

1. **Compound baseline is mandatory.** All PRs must stack on: `--n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 30 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm` with `n_layers=3, slice_num=16` hardcoded in model_config.
2. **PSN stacks on epochs=30.** PR #1050 confirmed that --per_sample_norm stacks additively with longer training: 66.8085 (no PSN, ep24) → 61.5855 (PSN, ep30) = -7.8%. PSN equalizes gradient contributions across the 15× Re std spread.
3. **Training was still falling at timeout.** PR #1050 was terminated at epoch 22/30 (30-min wall-clock limit). The best checkpoint was at ep22 with val still decreasing. epochs=36+ is the #1 priority hypothesis (PR #1064).
4. **n_layers=3, slice_num=16 is still the reference architecture.** This gave -8.31% val and established the primary architecture fix.
5. **Gap to reference is ~1.3x.** README reference (nl3/sn16) reports test_avg ~40.9; current best is 54.36. Further training (epochs=36+) may close this significantly.
6. **EMA decay=0.999 has been stable.** Merged in PR #882. Sweep at epochs=24 (PR #942) to confirm optimal decay.
7. **AdamW wd=1e-2 CLOSED — over-regularizes.** Default wd (1e-4) stays.
8. **surf_weight upweighting is dead on compound stack.** Monotone degradation at sw>10. Sub-10 sweep (PR #1011) tests whether sw<10 helps.
9. **Surface pressure dominates the gap to reference.** mae_surf_p drives the ranking metric.
10. **Huber delta optimal point may shift under PSN.** Prior sweep (PR #1028) was without PSN; delta=1.0 may not be optimal when PSN rescales the loss landscape. PR #1070 re-sweeps under PSN.

## Priority Queue for Next Hypotheses (when students become idle)

**Immediate high-priority (closing gap to reference ~40.9 test):**
1. **epochs=36 on compound+PSN** — In flight (PR #1064, edward). Val was still decreasing at ep22/30. High confidence next win.
2. **epochs=48 on compound+PSN** — Follow-up to epochs=36 if val still decreasing.
3. **Huber delta re-sweep with PSN** — In flight (PR #1070, thorfinn). Optimal delta expected to shift under PSN.

**Optimization / regularization on epochs=30+PSN baseline:**
4. **LR schedule tuning at epochs=30** — lr=5e-4 was tuned for 12 epochs; longer training may prefer lower lr.
5. **EMA decay tuning at epochs=30** — Tighter decay may help at longer training.
6. **LR warmup at epochs=30** — Warmup more impactful at longer training horizons.
7. **Cosine annealing T_max=30 verify** — Confirm T_max is correctly set to 30 not 12/24 in the new baseline.

**Architecture tuning on nl3/sn16/ep30+PSN baseline:**
8. **n_hidden=192 or 128 with nl3/sn16/ep30+PSN** — With 30 epochs, hidden=256 may be over-parameterized.
9. **surf_weight=1 or surf_weight=0** — Does any surf_weight help at epochs=30+PSN? In flight (PR #1011).
10. **slice_num=32 Goldilocks** — In flight (PR #1030, fern). Architecture midpoint test.
11. **FiLM conditioning** — Inject Re and AoA as global conditioning on slice tokens.

**Bold bets (if plateau):**
12. **Physics residual loss** — Add divergence-free penalty on velocity field (Ux, Uy).
13. **PSN + epochs=48** — If epochs=36 wins (PR #1064 positive), combine PSN with even longer training.
14. **Multi-resolution inputs** — Downsample mesh for early layers, full resolution for surface prediction.

## Research Theme: Closing the 1.3× Reference Gap

The model improved significantly with PSN+epochs=30 (+7.8% val vs epochs=24 no-PSN). We sit at val=61.5855, test=54.3573. The reference competition result is test_avg ~40.9 — 1.33× better than us.

Key drivers of the remaining gap (most to least likely):
1. **Training duration** — val curve was still decreasing at ep22/30 (timeout hit). Epochs=36/48 is the #1 hypothesis.
2. **Huber delta under PSN** — Prior delta=1.0 was tuned without PSN. Re-sweep expected to find a better value.
3. **LR schedule tuning** — The reference config may use a longer cosine horizon. lr=5e-4 was not re-tuned for 30-epoch training.
4. **EMA decay tuning** — Tighter decay may help at longer training duration.
5. **Architecture capacity** — n_hidden=256 with 3 layers at 30 epochs; may need adjustment.

The reference config test best of 40.927 vs current test best 54.36 suggests ~1.33× headroom. Longer training alone could close a large fraction of this gap.

## Round 1, 2 & 3 — Merged / Closed / Final Dispositions

**MERGED:**
- **PR #788** (alphonse, Huber loss): MERGED — val 115.6496
- **PR #827** (alphonse, surf_weight=30): MERGED — val 109.5716
- **PR #808** (fern, bf16+n_hidden=256+n_head=8+epochs=12): MERGED — val 104.1120
- **PR #882** (nezuko, EMA decay=0.999): MERGED — val 103.2182
- **PR #1005** (edward, n_layers=3, slice_num=16): MERGED — val 94.6541
- **PR #795** (thorfinn, per_sample_norm on compound stack): MERGED — val 90.4014
- **PR #1015** (edward, epochs=24 without PSN): MERGED — val 66.8085
- **PR #1050** (edward, PSN+epochs=30): MERGED — **val 61.5855 (current best), test 54.3573**

**CLOSED:**
- **PR #790** (edward, surf_weight on MSE): 128.98. Re-assigned as #827.
- **PR #791** (fern, wider model fp32): 155.96 on MSE. bf16 follow-up became #808.
- **PR #792** (frieren, n_layers=8 → grad_clip): 5 rounds, no improvement. grad_clip already in compound.
- **PR #793** (nezuko, slice_num=128): 130.97 > 115.6496; wall-clock penalty dominates.
- **PR #828** (edward, AdamW wd=1e-2): 106.9111 vs 103.2182 (+3.58% WORSE). Over-regularizes.
- **PR #960** (alphonse, surf_weight 20/30/50 sweep): All worse than sw=10. sw upweighting dead on compound.
- **PR #987** (edward, LR cosine T_max fix): No-op — T_max was already correct.
- **PR #794** (tanjiro, LR warmup 5 epochs): Unresponsive to 5 pings; final warning sent.
