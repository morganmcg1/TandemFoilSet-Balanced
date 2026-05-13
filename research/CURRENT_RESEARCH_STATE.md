# SENPAI Research State

- **Date:** 2026-05-13 09:45
- **Branch:** `icml-appendix-charlie-pai2g-24h-r2`
- **Track:** Charlie no-W&B 24h/48h logging-ablation arm (round 2/3)
- **Most recent human researcher direction:** none on this branch

## Current floor

**val_avg/mae_surf_p = 84.5393** (PR #1477, merged 2026-05-13 08:49)
Config: 3-ep warmup + lr=7.5e-4 + cosine(T_max=12) + gradclip(max_norm=1.0) + Huber β=0.3 + chan_w=[1,1,5] + **AMP bf16** + non-finite-y prefilter, wd=1e-4, ~0.66M model, 15 epochs (24.6 min — FULL cosine)
bs=1 clean test_avg = **74.9122** | bs=4 clean test_avg = **74.6655** (prefilter fixes NaN)
Floor progression: 122.70 → 111.15 → 105.68 → 85.93 → **84.54**

**Key change:** AMP bf16 cuts epoch time 37% and VRAM 22%, enabling full 15-epoch cosine completion within 24.6 min. Non-finite-y prefilter resolves bs=4 test NaN — no more eval_bs1.py fallback needed.

## Active experiments (WIP — all on AMP floor unless noted)

| PR | Student | Hypothesis | Lever | Status |
|---|---|---|---|---|
| #2140 | askeladd | bs=8 with AMP bf16: lr=7.5e-4 and lr=1e-3 | Batch scaling | Just assigned |
| #1947 | alphonse | chan_w sweep: [1,1,3] vs [1,1,7] under β=0.3 | Loss tuning | Rebasing onto AMP |
| #1927 | edward | Huber β=0.1 | Loss tuning | Rebasing onto AMP |
| #1489 | thorfinn | AoA flip p=0.25 | Augmentation | Rebasing onto AMP |
| #1681 | nezuko | Weight decay 5e-4 | Regularization | Rebasing onto AMP |
| #2096 | fern | n_hidden=128→160 (wider Transolver) | Architecture | In flight (post-AMP) |
| #2116 | tanjiro | Dropout sweep (p=0.05/0.1/0.2 in attn+MLP) | Regularization | Just assigned |
| #2129 | frieren | SWA-lite (avg last 3-5 ckpts at eval) | Optimization | Just assigned |

## Recent decisions

- **#1477 (fern AMP bf16) MERGED — NEW FLOOR**: val_avg 85.93→84.54 (−1.6%). AMP bf16 structural win: full cosine, 9 GB VRAM freed, clean bs=4 test. Two seeds both beat floor.
- **#2061 (tanjiro mlp_ratio=4) CLOSED**: +13.4% val regression. Bigger model didn't converge in 30-min budget (11/11 epochs, still descending at timer). Same throughput-dominance lesson as grad-accum/OneCycleLR.
- **#2019 (frieren cosine bracket) CLOSED**: Arm 1 (T_max=11) confirmed cosine-completion gain at fp32 (val=85.78 < 85.93) but **superseded by AMP floor #1477** (val=84.54). Schedule-completion lever now fully resolved.
- **#1751 (frieren T_max=12) MERGED**: val_avg 105.68→85.93 (−18.7%). Schedule calibration was the dominant lever.
- **#1891 (tanjiro OneCycleLR) CLOSED**: +3.32% regression. Structurally mismatched to 14-epoch budget.
- **#1927 (edward β=0.1) SENT BACK**: val=85.57 beats old floor 85.93 but below new fern floor 84.54. Needs rebase onto AMP.
- **#1536 (askeladd NaN guard) CLOSED**: Superseded by AMP floor's non-finite-y prefilter (#1477). Askeladd reassigned to bs=8 + AMP sweep (#2140).

## Key findings so far

1. **chan_w=[1,1,5]** confirmed win (+6.4%, PR #1464)
2. **Warmup + lr=7.5e-4 + gradclip** confirmed wins (stacked ~8.6%, PRs #1482, #1573)
3. **Huber β=1.0 → β=0.3** confirmed wins (stacked −14.3%, PRs #1801, #1849)
4. **T_max=12 cosine alignment** largest single win (−18.7%, PR #1751)
5. **AMP bf16 + non-finite-y prefilter** structural win (−1.6% val, −3.5% test, full cosine, PR #1477)
6. **β=0.1 confirmed positive** at T_max=47 and T_max=12 floors (edward #1927, needs AMP rebase)
7. **Eight wins stacked**: chan_w + warmup + gradclip + Huber + β=0.3 + T_max=12 + AMP bf16 + prefilter
8. **EMA / Lookahead / grad-accum / OneCycleLR fail under timeout-cut**: step throughput dominates
9. **β trend monotone**: β=1.0 > β=0.5 > β=0.3 > β=0.1 (at all floors tested so far)

## Round 7 hypothesis pipeline

### Critical
- **edward β=0.1 rebase on AMP floor** (#1927): First β=0.1 + AMP combination. β trend is monotone, high probability of continued improvement.
- **alphonse chan_w rebase on AMP floor** (#1947): [1,1,3] vs [1,1,7]. AMP changes VRAM budget, may shift the optimal channel weighting.
- **fern n_hidden=160** (#2096): Wider Transolver. AMP unlocks the VRAM for larger model in the same wall-clock budget.

### Just assigned (this cycle)
- **tanjiro dropout sweep**: Untested regularization lever. Current Transolver dropout=0. Sweep p∈{0.05, 0.1, 0.2} in attention+MLP. Free in epoch budget (no compute change). May reduce overfitting on the high-residual splits (camber_rc, single_in_dist).
- **frieren SWA-lite (checkpoint averaging)**: Save last 5 checkpoints during AMP training, average weights at eval. Known free win in literature. Matches frieren's schedule-expertise. Uses the full 15-epoch AMP cosine — the late-cosine ckpts should be high-quality.

### In flight (rebasing onto AMP)
- **askeladd NaN guard** (#1536): Bug fix + clean floor measurement. Prefilter already in AMP floor, so askeladd's NaN guard may be partially superseded.
- **thorfinn AoA flip** (#1489): Augmentation on AMP floor.
- **nezuko WD=5e-4** (#1681): Regularization on AMP floor.

### Next round queue
- **torch.compile reduce-overhead**: Independent of AMP, should reduce epoch time further (more epochs in budget)
- **bs=8 with bf16**: ~50 GB projected VRAM; more gradient smoothing per step
- **β=0.05 or pure L1 (β→0)**: Continue the monotone β trend (after β=0.1 confirms or breaks the trend on AMP)
- **EMA on top of SWA-lite**: If SWA-lite wins, layer EMA decay for online ensemble
- **Per-channel β** (Ux=Uy=0.1, p=0.5): edward's noted split-level β interaction from #1849 — low β hurts low-residual splits
- **Mixup/CutMix on geometry features**: Augmentation in feature space rather than physical AoA flip
- **Lookahead optimizer (k=5, α=0.5)**: Outer-loop averaging combines with SWA-lite
- **T_max=13/14 now moot**: AMP completes T_max=12 cleanly
