# SENPAI Research State
- 2026-04-29 (updated)
- No directives from human researcher team
- Branch: icml-appendix-charlie-pai2e-r1

## Current Best (val_avg/mae_surf_p)

**66.8085** — PR #1015 (edward): longer training epochs=24 on compound stack (nl3/sn16), epoch 22/24 (hit 30-min timeout, val still falling)
**test_avg/mae_surf_p:** 58.7266

Reproduce:
```bash
cd target/ && python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 24 --grad_clip 1.0 --ema_decay 0.999
```
*(Note: `n_layers=3` and `slice_num=16` are hardcoded in the `model_config` dict in `train.py` — not CLI flags. Run was without `--per_sample_norm`.)*

## Merged Winner Chain (cumulative stacking)

| PR | Description | val_avg/mae_surf_p | Delta |
|----|-------------|-------------------|-------|
| Baseline (MSE) | Default train.py | 126.88 | — |
| #788 | Huber loss (delta=1.0) | 115.6496 | -8.85% |
| #827 | + surf_weight=30 | 109.5716 | -5.26% |
| #808 | + bf16 + n_hidden=256 + n_head=8 + epochs=12 | 104.1120 | -4.97% |
| #882 | + EMA(decay=0.999) | 103.2182 | -0.86% |
| #1005 | + n_layers=3, slice_num=16 (reference arch) | 94.6541 | -8.31% |
| #795 | + per-sample loss normalization | 90.4014 | -4.50% |
| **#1015** | **epochs=24 (longer training)** | **66.8085** | **-26.1%** |

## Active Experiments (WIP — 7 students, 1 idle: thorfinn)

| PR | Student | Hypothesis | Status | Notes |
|----|---------|-----------|--------|-------|
| #1050 | edward | PSN + epochs=24 compound stack (per_sample_norm + longer training stacked) | Running | Testing if PSN stacks on top of epochs=24 winner; PR #1015 was run without --per_sample_norm |
| #998 | frieren | slice_num 64→128 on compound baseline (wider PhysicsAttn) | Running | Testing if more slices improves spatial resolution of physics attention |
| #1011 | alphonse | surf_weight sub-10 sweep (1/3/5/7) on compound baseline | Running | PR #960 showed monotone degradation: sw=10 < sw=20 < sw=30 < sw=50. Optimum may be below 10 |
| #942 | nezuko | EMA decay sweep: 0.99/0.995 vs 0.999 on compound | Running | Tighter decay may fix in-dist/OOD asymmetry |
| #1030 | fern | slice_num=32 Goldilocks sweep: midpoint between sl=16 and sl=64 | Running | sl=16 is current best; sl=64 was worse; sl=32 may hit a sweet spot |
| #1038 | tanjiro | LR warmup 2 epochs + cosine annealing on compound baseline | Running | 2-epoch LinearLR warmup before cosine annealing to reduce early gradient instability |
| #1018 | askeladd | LR sweep on nl3/sn16 compound: 1e-3 / 2e-4 / 5e-4 | Running | Reference config LR investigation — optimal LR likely shifted with nl3/sn16+per_sample_norm |
| — | **thorfinn** | **IDLE — needs assignment** | Awaiting | PR #1028 closed: δ=0.25 wins under PSN at epochs=12 (86.40 val_avg, ~29% above baseline). δ=0.25 finding routed into edward's PR #1050. |

## Key Technical Insights

1. **Compound baseline is mandatory.** New baseline is: `--n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 24 --grad_clip 1.0 --ema_decay 0.999` with `n_layers=3, slice_num=16` hardcoded in model_config.
2. **Longer training is the biggest single win.** PR #1015 gave -26.1% val and -27.0% test improvement. Val was still decreasing at epoch 22 when the 30-min timeout hit (LR had decayed to 8.5e-6). Epochs 30-36 are next priority.
3. **Per-sample loss normalization (PSN) to stack.** PR #795 gave -4.50% on epochs=12 compound. The PR #1015 epochs=24 run was done WITHOUT PSN. Edward is now testing PSN+epochs=24 stack (PR #1050).
4. **n_layers=3, slice_num=16 is the biggest architectural win.** PR #1005 gave -8.31% val and -9.43% test vs compound baseline. Over-partitioned physics attention (slice_num=64) was hurting generalization.
5. **EMA merged.** decay=0.999 gives -0.86% val overall.
6. **Gap to reference is ~1.4x on test.** README reference reports test_avg ~40.9; current best is 58.7266. Previously 2× gap. Longer training closed substantial headroom.
7. **AdamW wd=1e-2 CLOSED — over-regularizes on full compound.** Default wd (1e-4) is appropriate.
8. **surf_weight upweighting is dead on compound stack.** PR #960 showed clean monotone: sw=10 < sw=20 < sw=30 < sw=50. Sub-10 values being tested (PR #1011).
9. **Surface pressure dominates the gap to reference.** Velocity errors (Ux ~0.9, Uy ~0.45) are reasonable; `mae_surf_p` drives the ranking metric.

## Priority Queue for Next Hypotheses (when students become idle)

**Immediate high-priority (closing gap to reference ~40.9 test):**
1. **epochs=30 or epochs=36 on nl3/sn16 compound + PSN** — Val was still falling at ep22. After PR #1050 lands, push epochs budget further (extend PSN+epochs run to 30-36 if timeout is hit again).
2. **n_layers=4 or n_layers=5** — 3 layers already optimal vs 8 layers, but trying n_layers=4 or 5 on the new longer-training baseline may give an incremental win.
3. **Cyclic / warm-restart LR** — With 24+ epochs now standard, cosine warm restart may help find better minima than single cosine decay.
4. **Learning rate from 5e-4 to 1e-3** — LR sweep (PR #1018) may resolve this; if 1e-3 wins, new baseline should use it with PSN + epochs=24.

**Architecture tuning on nl3/sn16 baseline:**
5. **n_hidden=192 or 128 with nl3/sn16** — With 3 layers, hidden=256 may now be over-parameterized. Smaller hidden could reduce overfitting further.
6. **FiLM conditioning** — Inject Re and AoA as global conditioning on slice tokens (physics-informed).
7. **Separate pressure decoder** — Surface pressure (p) may benefit from its own decoder head vs shared decoder.

**Optimization / regularization:**
8. **EMA decay 0.99 or 0.995** — In flight (PR #942, nezuko). Tighter decay may help with per-split asymmetry.
9. **Huber delta tuning with PSN — RESOLVED (PR #1028 closed).** δ=0.25 wins monotonically over δ ∈ {0.5, 1.0, 2.0} under PSN at epochs=12 (best 86.40, ~29% above current baseline). Outcome doesn't beat baseline alone; δ=0.25 should be applied to PSN+longer-training (PR #1050, edward) and could be pushed sub-0.25 (δ ∈ {0.05, 0.1, 0.15}) once PSN viability is confirmed.
10. **AdamW beta2=0.95** — Faster momentum decay for noisy gradients.

**Bold bets (if plateau):**
11. **Physics residual loss** — Add divergence-free penalty on velocity field (Ux, Uy).
12. **Multi-resolution inputs** — Downsample mesh for early layers, full resolution for surface prediction.
13. **Ensemble of EMA checkpoints** — Average top-K val checkpoints by score; potentially 1-2% free improvement.

## Research Theme: Closing the 1.4× Remaining Reference Gap

The training duration discovery (epochs 12→24) was the biggest single jump (+26.1% val, +27.0% test). We now sit at val=66.8085, test=58.7266. The reference competition result is test_avg ~40.9 — ~1.4× better than us.

Key drivers of the remaining gap (most to least likely):
1. **Training duration** — val curve was still decreasing at epoch 22 when timeout hit (LR=8.5e-6). Epochs 30-36 should close more gap.
2. **PSN stacking on epochs=24** — PR #1050 (edward) is testing this. Could give another 4-5% gain.
3. **LR schedule tuning** — Default lr=5e-4 was not tuned for nl3/sn16 + longer training. A higher initial LR (1e-3) may allow the model to escape earlier local minima.
4. **Architecture capacity** — n_hidden=256 with 3 layers and 24 epochs may now be optimal or could benefit from a smaller hidden dimension.
5. **Loss formulation** — Huber delta=1.0 was tuned on epochs=12; optimal delta may shift with longer training and PSN.

## Merged / Closed History

**MERGED:**
- **PR #788** (alphonse, Huber loss): MERGED — val 115.6496
- **PR #827** (alphonse, surf_weight=30): MERGED — val 109.5716
- **PR #808** (fern, bf16+n_hidden=256+n_head=8+epochs=12): MERGED — val 104.1120
- **PR #882** (nezuko, EMA decay=0.999): MERGED — val 103.2182
- **PR #1005** (edward, n_layers=3, slice_num=16): MERGED — val 94.6541, test 83.7608
- **PR #795** (thorfinn, per-sample loss normalization): MERGED — val 90.4014, test 80.3748
- **PR #1015** (edward, epochs=24 longer training): MERGED — **val 66.8085, test 58.7266 (current best)**

**CLOSED:**
- **PR #790** (edward, surf_weight on MSE): 128.98, above Huber baseline.
- **PR #791** (fern, wider model fp32): 155.96 on MSE.
- **PR #792** (frieren, n_layers=8 → grad_clip focus): 5 rounds; grad_clip already in compound.
- **PR #793** (nezuko, slice_num=128): 130.97 > 115.6496 baseline.
- **PR #828** (edward, AdamW wd=1e-2): 106.9111 vs 103.2182 (+3.58% WORSE). Over-regularizes.
- **PR #960** (alphonse, surf_weight 20/30/50 sweep): All worse than sw=10.
- **PR #987** (edward, LR cosine T_max fix): No-op — T_max was already correct.
- **PR #1028** (thorfinn, Huber δ sweep under PSN, epochs=12): Best δ=0.25 → val 86.40, test 76.71. ~29% above current baseline (66.8085). Direction confirmed (smaller δ wins under PSN, monotonically). δ=0.25 routed to PR #1050.
