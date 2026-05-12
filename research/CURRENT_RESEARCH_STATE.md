# SENPAI Research State

- 2026-05-12 21:00
- No human researcher directives (no open issues)
- Round 5 Charlie no-W&B arm — 30-min wall-clock cap, local JSONL only

## Merged baseline

**val_avg/mae_surf_p = 114.40** (PR #1519 — warmup3 + cosine T_max=13)
- test_avg NaN due to 1 corrupted GT sample (cruise test sample 20 has Inf in y)
- 3-split clean test = 112.63
- Model still improving at epoch 13 → more compute headroom

## Key round-5 findings to date

| Finding | Impact |
|---|---|
| Baseline does ~13 epochs in 30 min (not 50) | All schedule hyperparams must be matched to budget |
| CosineAnnealingLR(T_max=50) barely decays in 13 epochs | Matching T_max=13 alone gave 8.6% improvement |
| 3-epoch warmup stabilises early training | Composes cleanly with schedule fix |
| SWA mechanism works (27.5% within-run gain on val) | Not yet beating merged baseline; need warmup+SWA compose |
| surf_weight=20 regresses at 14 epochs | Budget too short for the changed loss landscape |
| p-channel 3x weight regresses | Surface Ux/Uy is NOT a free task; equal-weight is already right |
| GT NaN at cruise test sample 20 (idx=20, y contains Inf) | Data-side bug; model predictions are healthy |
| **Surface skip branch -6.2% within-run on pre-warmup baseline** (#1487) | Mechanism works; needs composition with merged baseline (~107 expected) |
| Root cause of GT-NaN: `err * mask` in `data/scoring.py` returns NaN at masked pos | NaN * 0 = NaN in IEEE; needs `torch.where(mask, err, 0)` upstream fix |
| GitHub API rate-limited 4 student pods for ~1hr | Resolved at ~20:50; lost some experiment throughput |

## Active PRs (all 8 students assigned)

| PR | Student | Hypothesis | Status | Target |
|---|---|---|---|---|
| #1463 | askeladd | Warmup+cosine+SWA composed | WIP rerun (needs rebase) | Beat 114.40 |
| #1470 | edward | Instance-norm loss | WIP active | Beat 114.40 |
| #1478 | frieren | Wider model n_hidden=192 | WIP active | Beat 114.40 |
| #1481 | nezuko | slice_num=128 | WIP active | Beat 114.40 |
| #1483 | tanjiro | Gradient clipping max_norm=1.0 | WIP active | Beat 114.40 |
| #1487 | thorfinn | Surface skip composed with merged baseline | WIP rerun | Beat 114.40 (~107?) |
| #1564 | alphonse | GT-NaN fix → clean test number | WIP active | val ≈114.40, test finite |
| #1565 | fern | BF16 + batch=8 → 20 epochs in budget | WIP active (87 GB VRAM) | Beat 114.40 |

## Open questions from active experiments

1. **Does the surface skip compose with warmup+cosine?** (#1487 rerun) — High-prior YES; within-run delta of -6.2% on stale baseline. Expected ~107.
2. **Does gradient clipping help?** (#1483 tanjiro) — Would stabilise training for all future experiments. May explain why model is still improving at epoch 13.
3. **Does wider model help?** (#1478 frieren) — 192 hidden dims; slower per-epoch so fewer epochs in budget; may or may not pay off.
4. **Does instance-norm loss help?** (#1470 edward) — Addresses Re dynamic range gradient dominance; could improve val_re_rand specifically.
5. **Does BF16+batch=8 unlock more epochs?** (#1565 fern) — If this gives 20 epochs for free, almost every future hypothesis gets a throughput boost.
6. **Does composed warmup+SWA beat warmup alone?** (#1463 askeladd) — SWA was clearly working; composing with the warmup recipe is the most promising pending hypothesis.

## Next hypotheses to queue (when students go idle)

1. **Data-side scoring.py fix** — thorfinn volunteered separate PR (`torch.where(mask, err, 0)` in `data/scoring.py`). Cleanest fix; doesn't require duplicating logic in train.py.
2. **Surface skip + EVERYTHING that wins** — if both #1487 (skip) and any of (#1483 clipping, #1463 SWA, #1565 BF16) win, the next compose layer is skip + winner.
3. **Skip hidden=64/128** — student suggested in #1487 follow-up; current skip is 32 hidden, may have unused capacity.
4. **Volume-side skip** — analogous to surf_skip but for volume nodes (different feature subset, no signed-arc-length).
5. **EMA of weights** (alternative to SWA): applies per gradient step, no LR valley requirement, tends to be better in low-iteration regimes.
6. **LR increase:** with warmup+cosine and gradient clipping (if merged), might be able to push lr=1e-3 and get faster convergence.
7. **Transolver++ local adaptive correction (H12):** highest expected gain from literature, moderate engineering effort.
8. **n_layers=7 with BF16:** if BF16 (#1565) wins, depth increase becomes viable within budget.
