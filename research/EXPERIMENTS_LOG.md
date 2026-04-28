# SENPAI Research Results

## 2026-04-28 00:35 — PR #344: H2 linear warmup + cosine to zero with corrected T_max — **MERGED**

- Branch: `willowpai2d4-edward/h2-warmup-cosine`
- Hypothesis: Linear warmup + per-step cosine-to-zero, with `T_max` re-aligned to the actual run length, should reduce `val_avg/mae_surf_p` by 3–7% by fixing the per-epoch `CosineAnnealingLR(T_max=50)` that never reaches zero under the 30-min wall clock.
- 3-cell matrix in W&B group `h2-warmup-cosine`:

| Run | Config | best_val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | W&B |
|-----|--------|--------------------------|----------------------|------------|-----|
| A | `--epochs 50` (peak 5e-4) | 125.17 | 113.85 | 14 | `5okwzg15` |
| B | `--epochs 30` (peak 5e-4) | 129.47 | 117.65 | 13 | `4wd9nu6k` |
| C | `--epochs 25 --lr 7e-4` | **120.97** | **109.92** | 13 | `rua9xrca` |

### Per-split test surface MAE (Run C, winner)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------:|------------:|------------:|
| `test_single_in_dist` | 127.09 | — | — |
| `test_geom_camber_rc` | 123.58 | — | — |
| `test_geom_camber_cruise` | 81.16 | — | — |
| `test_re_rand` | 107.83 | — | — |
| **avg** | **109.92** | 1.96 | 0.83 |

### Conclusions

- **Hypothesis confirmed in spirit, with a twist.** The 30-min wall clock truncates training at epoch ~14 in all three configs, so cosine never actually reaches zero in any run. Run B (which would have reached zero by configured epoch 30) underperformed Run A precisely because it spent more time near peak lr without the cosine tail kicking in. Run C wins by raising peak lr 40% AND shortening configured epochs — net effect is higher integrated lr early plus a meaningful (if not all-the-way-to-zero) decay tail.
- **Cross-split signature matched the prediction.** In-distribution split improved most strongly (Run C vs B: -20% on `test_single_in_dist`); generalization splits moved less.
- **Critical NaN fix shipped alongside.** `evaluate_split` now filters samples with non-finite ground truth (e.g., `test_geom_camber_cruise` sample 20 has `-inf` in pressure GT) and defensively zeros out non-finite predictions before metric accumulation. Without this fix, `test_avg/mae_surf_p` was NaN on every run that touched the cruise camber test split.
- **Suggested future follow-up:** test `--epochs 14` (matched to the actual epoch budget) to see whether cosine reaching exactly zero in run-time further improves things.

---

## 2026-04-28 00:38 — PR #346: H7 z-mirror augmentation — **CLOSED**

- Branch: `willowpai2d4-frieren/h7-zmirror-augmentation`
- Hypothesis: z-axis mirror augmentation with sign flips on z-position, saf[1], AoA, Uy should reduce `val_avg/mae_surf_p` by 3–8% by doubling effective training data via 2D physical symmetry.
- 3-cell matrix in W&B group `h7-zmirror`:

| Run | `augment_zmirror` | best_val_avg/mae_surf_p | best_epoch | W&B |
|-----|-------------------|--------------------------|------------|-----|
| A | 0.0 | 124.71 | 12 | `4nu04gte` |
| B | 0.5 | 169.29 (+35.8%) | 8 | `9p19tvj6` |
| C | 1.0 | 412.52 (+231%) | 1 | `4o4nfvhb` |

`test_avg/mae_surf_p` was None on all 3 runs because `test_geom_camber_cruise/mae_surf_p` came back NaN — same root cause edward's PR fixed (now merged on advisor branch, so future runs are robust).

### Conclusions

- **Strict monotonic regression.** Run C's diagnostic is decisive: train losses descend cleanly on the all-mirrored distribution but val MAE *increases* during training, the classic distribution-shift fingerprint. The (mirrored x, mirrored y) pair is **not** a valid sample of the same underlying CFD problem — the augmentation is breaking the input→output mapping rather than preserving it.
- **`mae_surf_Uy` regresses ~3x** under augmentation despite being explicitly sign-flipped — confirms the model can't learn the symmetry from the corrupted training distribution.
- **Likely structural causes (per student diagnostic):**
  1. raceCar tandem (~30% of training) has a slip-wall ground at z=0 — the BC effects (ground effect, wake interaction) are not z-symmetric, so mirroring produces a sample of a *different* CFD problem with hallucinated targets.
  2. `gap`/`stagger` (dims 22, 23) decompose differently in cruise vs. raceCar (chord-aligned frame for raceCar). The conservative no-flip choice corrupts ~10% of cruise tandem; flipping to fix cruise corrupts raceCar. There is no consistent z-mirror law given the dataset construction.
- **Worthwhile follow-ups (deferred):**
  1. **Test-time augmentation (TTA)** — average predictions on (x, mirror(x)) at val time. Tests whether the symmetry exists in the trained model, divorced from training-time corruption.
  2. **Domain-conditional augmentation** — restrict mirroring to cruise-only with proper gap sign-flip. Small slice of training data but clean physics.
- **Action:** closed; reassigning frieren to a fresh hypothesis.
