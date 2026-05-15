# SENPAI Research State

- **Last updated:** 2026-05-15 ~22:35 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline

**val_avg/mae_surf_p = 100.5275, test_avg/mae_surf_p = 90.1489** — from PR #3089 (alphonse, L1 loss + warmup + clip + lr=1e-3), merged 2026-05-15 ~22:31 UTC. See `BASELINE.md` for full details.

Previous baseline was 109.42 (PR #3091 edward). Alphonse's L1 loss delivered −8.1% on val and also landed the canonical scoring fix, making `test_avg/mae_surf_p` finite for the first time.

## Most recent research direction from human researcher team

No GitHub Issues open for this track. Proceeding from the program contract only.

## Cross-cutting findings (apply to ALL in-flight PRs)

1. **Cosine schedule mis-tuned:** `SENPAI_TIMEOUT_MINUTES=30` → ~14-15 epochs realized; `T_max=50` means LR barely anneals. Future PRs should pass `--epochs 10` to get proper cosine decay within budget. Confirmed by both fern (#3092) and edward (#3091).
2. **Scoring NaN fixed (canonical):** PR #3089 (alphonse) merged — `torch.isfinite` per-sample mask in `train.py`'s `evaluate_split`. All branches rebasing onto advisor tip after this point get the fix automatically.
3. **Grad norm:** Pre-clip gradient norm was 160 at lr=5e-4 in edward's Arm A. Now clipped at max_norm=1.0 (merged). Future PRs benefit automatically.
4. **Model is not converged** at the 30-min timeout — edward's Arm B best epoch was the last completed epoch (14/15). Significant headroom at longer training or larger epoch budgets.
5. **L1 loss is the new default** (Config.loss_type = "l1"). All new experiment branches rebasing on advisor tip use L1 automatically.

## Active in-flight PRs

| # | Student | Hypothesis | State | val_avg/mae_surf_p (W&B-verified) | Δ vs 100.53 |
|---|---|---|---|---|---|
| **#3372** | **askeladd** | **Fourier PE on (x,z) coords, 4-freq** | wip (awaiting SENPAI-RESULT) | **94.491** (run xmcndd46) | **−6.0% 🏆 best** |
| #3371 | thorfinn | EMA decay=0.9999 | wip (awaiting SENPAI-RESULT) | 106.22 (run rlgnspig) | +5.7% ✗ vs new baseline |
| #3469 | tanjiro | Depth n_layers=5→6 | wip (running ~22:00+) | — | — |
| #3479 | frieren | Per-channel output heads (p, Ux, Uy split) | wip (newly assigned 22:23) | — | — |
| #3092 | fern | slice_num=64 baseline vs 128 (rebased) | wip (awaiting results) | — | — |
| #3288 | edward | Scoring fix + lr default bump | wip (may be obsolete after #3089 merge) | — | — |
| **#NEW** | **nezuko** | **L1 LR sweep: lr ∈ {3e-4, 2e-3, 4e-3}** | **being assigned** | — | — |

## Round 4/5 confirmation arm results (full table)

| # | Student | Hypothesis | W&B run | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs 109.42 | Status |
|---|---|---|---|---|---|---|---|
| **#3089** | **alphonse** | **L1 + warmup + clip + lr=1e-3** | 14w7wdyb | **100.527** | **90.149** | **−8.1% 🏆 MERGED** | ✓ merged new baseline |
| **#3372** | **askeladd** | **Fourier PE 4-freq** | xmcndd46 | **94.491** | n/a | **−13.6%** | waiting SENPAI-RESULT |
| #3371 | thorfinn | EMA decay=0.9999 | rlgnspig | 106.22 | n/a | −2.9% vs old | +5.7% vs new baseline |
| #3092 | fern | slice_num=64 rerun | nr0e4l1p | ~109.23 | n/a | ~tied old baseline | waiting rerun |
| ~~#3095~~ | ~~nezuko~~ | ~~surf_weight=20~~ | 6amjj7jr | 111.92 | 97.70 | +2.3% old / +11.3% new | **CLOSED** |
| ~~#3414~~ | ~~tanjiro~~ | ~~SWA last K~~ | udfmekyw | swa=109.48 | — | ~flat | **CLOSED** |
| ~~#3093~~ | ~~frieren~~ | ~~bf16+bs=8~~ | wnnq2o3x | 120.43 | n/a | +10% | **CLOSED** |

## Merged wins

| PR | Description | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|---|
| #3089 | L1 loss + scoring fix (alphonse) | **100.5275** ← current baseline | **90.1489** |
| #3091 | LR warmup + clip + lr=1e-3 (edward) | 109.42 ← previous baseline | NaN (pre-fix) |

## Next decisions (queued; awaiting student SENPAI-RESULT comments)

### PRs needing terminal SENPAI-RESULT before merge decision

1. **#3372 askeladd Fourier PE (val=94.491)** — BIGGEST WIN vs new baseline (−6.0%). Already nudged; awaiting student to post SENPAI-RESULT terminal marker. Will merge as next baseline once marker lands.
2. **#3371 thorfinn EMA 0.9999 (val=106.22)** — +5.7% ABOVE new baseline of 100.53. Once terminal SENPAI-RESULT lands, close as dead end (EMA doesn't help here; model not converged enough to benefit from averaging).
3. **#3092 fern** — slice_num=64 vs 128 rebased. Awaiting results. If 128 > 64 on val AND finite test metric, merge. Otherwise close.
4. **#3288 edward** — scoring fix + lr default: the scoring fix is now canonical via alphonse's merge. The lr default flip is trivial but non-urgent. Will review once student posts results; may close as superseded.

### PRs to monitor (training, no results yet)
- **#3469 tanjiro depth n_layers=6** — newly running; expect results within ~40 min from assignment
- **#3479 frieren per-channel heads** — newly assigned; expect results 30+ min after pod picks up

### Edward #3288 disposition
The scoring fix is now merged via #3089. Edward's PR only needs to land the lr default flip (5e-4→1e-3 in Config). The run he started already at val=115.76 was on the pre-#3089 baseline — irrelevant now. Ask him to trim to lr-default-only and resubmit, or close as superseded since all new branches on the advisor tip already inherit lr=1e-3 from the existing Config default.

## Potential next research directions (round 5+)

1. **L1 + LR sweep** — lr=1e-3 was tuned for MSE; with L1 loss the optimal may differ. Try {3e-4, 2e-3, 4e-3} → nezuko #NEW.
2. **Fourier PE (askeladd #3372)** — best single result so far at val=94.49; needs merge.
3. **Width increase n_hidden=128→160** — ~25% more params (~830K params); L1+warmup+clip stabilizes training so larger models may converge better. Zero code complexity.
4. **Depth n_layers=6 (tanjiro #3469)** — +20% cost per step; within budget at 10 epochs.
5. **Per-channel heads (frieren #3479)** — separate p/Ux/Uy decoders; zero parameter cost.
6. **Input feature engineering** — append chord length, angle of attack, Re number as global conditioning tokens.
7. **Physics-aware loss** — divergence-free penalty, near-surface gradient consistency as auxiliary loss term.
8. **Mixed precision training (bf16) with schedule re-tune** — #3093 failed because bf16 numerics + lr=1e-3 + no warmup was incompatible. With proper warmup + clip + L1, bf16 might work. The speed unlock (2× epochs in same wall-clock) is very valuable.
9. **CosineAnnealingWarmRestarts** — T_0=5, T_mult=1 → 2 full cycles in 10-epoch budget; may escape local optima.
10. **Dropout regularization** — add small dropout (0.1) in MLP layers; model may be overfitting within 10 epochs.

## Cross-cutting observations

- **Every architecture change tried so far (depth, width) is runtime-budget-bound**, not capacity-bound. The model's biggest val drops happen at epochs 10–14; slower models can't reach that regime in 30 min.
- **L1 loss is the most impactful single change** (+8.1% on val, first clean test metric). The loss function change aligns the training objective with the evaluation metric.
- **Fourier PE is the best single experiment result** (val=94.49, −13.6% vs old baseline, −6.0% vs current). Geometric features in high-frequency space are valuable for airfoil CFD.
- **surf_weight tuning does not help** above the default of 10. The optimizer already focuses on surface-p via the baseline weighting; additional emphasis regresses both val and test.
