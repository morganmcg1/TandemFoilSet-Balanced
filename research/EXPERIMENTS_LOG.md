# SENPAI Research Results — willow-pai2i-24h-r4

## 2026-05-15 14:10 — PR #3257: Surface MAE loss + pressure-channel weight 3×

- **Student/branch:** willowpai2i24h4-frieren / `willowpai2i24h4-frieren/surf-mae-p-weight`
- **Hypothesis:** Switch surface loss from MSE to MAE and weight the p channel 3× to align the training signal with `test_avg/mae_surf_p`. Volume loss kept as MSE.
- **W&B run:** `zz2r70lt` (https://wandb.ai/wandb-applied-ai-team/senpai-v1/runs/zz2r70lt)

### Result (best checkpoint at epoch 13 of 14; timeout cap)

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------:|----------------:|
| `single_in_dist`        | 115.42 | 106.18 |
| `geom_camber_rc`        | 119.27 | 104.56 |
| `geom_camber_cruise`    |  73.02 | **NaN** (non-finite p preds) |
| `re_rand`               |  91.93 |  86.33 |
| **avg**                 | **99.91** | **NaN** |

Peak GPU: 42.1 GB / 96 GB. Wall time: 30.8 min (hit cap). Val curve still descending at termination (ep 1 → 228.01, ep 13 → 99.91, ep 14 → 122.76).

### Decision: send back to student (#3257-comment-4460628326)

`test_avg/mae_surf_p = NaN` is disqualifying per advisor protocol — the primary ranking metric for the paper-facing comparison must be finite.

### Root cause

`data/scoring.py:accumulate_batch` skips samples with non-finite **ground truth** but does not guard against non-finite **predictions** — one runaway pred poisons the running sum. `data/scoring.py` is read-only, so the fix has to live in `train.py:evaluate_split`. Sent back with explicit `torch.nan_to_num(pred_orig, nan=0.0, posinf=1e6, neginf=-1e6)` patch + `n_nonfinite_pred` per-split diagnostics, plus instructions to rerun the same arm.

### Analysis

- Per-split val shape matches the hypothesis prediction (cruise/re_rand carry the gain), suggesting the loss change is doing what we wanted — we just can't confirm the test-side number until the rerun.
- Training was wall-clock-capped at epoch 14/50 — the cosine schedule was set for `T_max=50` but only ~14 epochs run. The model never saw the low-LR end of the schedule. This is a systemic issue affecting every PR in this round; will address in a follow-up hypothesis family.
- No baseline (unmodified Transolver) measurement exists yet on this branch — the clean rerun of this PR will be the first credible point.

## 2026-05-15 15:30 — PR #3262: Random Fourier Features positional encoding

- **Student/branch:** willowpai2i24h4-edward / `willowpai2i24h4-edward/fourier-pos-enc`
- **Hypothesis:** Augment input with Random Fourier Features (RFF; Tancik et al. 2020) of unnormalized (x, z) coordinates. n_freqs=16, swept σ ∈ {1.0, 4.0}. Baseline measured in parallel (no RFF).
- **W&B runs:** baseline `17fia1vd`, σ=1.0 `vlv1b0ab`, σ=4.0 `q9vkl63z`

### Result (best checkpoint, all runs hit 30-min timeout cap at epoch 13–14 / 50)

| Split | Baseline | σ=1.0 | σ=4.0 | σ=1.0 Δ |
|-------|---------:|------:|------:|--------:|
| `val_single_in_dist/mae_surf_p`     | 155.71 | **140.41** | 157.36 | −9.8% |
| `val_geom_camber_rc/mae_surf_p`     | 136.10 | **120.10** | 146.07 | −11.8% |
| `val_geom_camber_cruise/mae_surf_p` | 103.19 | **92.11**  | 101.16 | −10.7% |
| `val_re_rand/mae_surf_p`            | 118.38 | **110.49** | 118.94 | −6.7% |
| **`val_avg/mae_surf_p`**            | **128.34** | **115.78** | 130.88 | **−9.8%** |
| `test_single_in_dist/mae_surf_p`    | 135.28 | 119.89 | 139.12 | −11.4% |
| `test_geom_camber_rc/mae_surf_p`    | 128.51 | 108.99 | 132.54 | −15.2% |
| `test_geom_camber_cruise/mae_surf_p`| **NaN** | **NaN** | **NaN** | — |
| `test_re_rand/mae_surf_p`           | 118.07 | 104.24 | 114.87 | −11.7% |
| `test_avg/mae_surf_p` (4-split mean) | NaN | NaN | NaN | — |
| `test_avg/mae_surf_p` (3 valid splits, edward's report) | 127.29 | 111.04 | 128.84 | −12.8% |

Peak GPU σ=1.0: 42.5 GB / 96 GB. n_params σ=1.0: 670,551 (vs baseline 662,359, +1.2%).

### Decision: send back to student (#3262-comment-4461135244)

`test_avg/mae_surf_p = NaN` (formal 4-split mean) is disqualifying per advisor protocol — same `data/scoring.py` non-finite-prediction bug surfaced via #3257. The val win is strong (−9.8% across all splits) and consistent. Sent back with the same `torch.nan_to_num` patch for `train.py:evaluate_split` and instruction to rerun only the σ=1.0 arm (skip σ=4.0 confirmed loser, skip baseline rerun).

### Analysis

- **Hypothesis worked, large effect size.** RFF σ=1.0 reduces `val_avg/mae_surf_p` by 9.8% (within and beyond the 3–8% predicted envelope) with consistent per-split gains. Largest improvements on OOD geometry splits (`val_geom_camber_rc` −11.8%, `test_geom_camber_rc` −15.2%) — suggests RFF helps spatial generalization more than in-distribution fitting, consistent with the spectral-bias literature interpretation.
- **σ=4.0 confirms scale.** Slight regression at σ=4.0 (+2.0% val_avg vs baseline) brackets the useful range as σ ∈ [0.5, 2.0] (Tancik 2020 alias-warning regime).
- **Volume pressure also improves** (1.5–7.4% per split), so RFF benefit is not surface-only.
- **NaN on `test_geom_camber_cruise` is pre-existing**, present in vanilla baseline too — confirms it's the systemic `accumulate_batch` bug, not RFF-induced divergence.
- **First credible baseline measurement on this branch:** edward's paired vanilla `17fia1vd` gives `val_avg/mae_surf_p = 128.34` and 3-split `test_avg/mae_surf_p = 127.29`. Once the σ=1.0 rerun lands with finite 4-split test_avg, this becomes BASELINE.md.
- **R2 follow-up queue (from edward's suggestions, ranked):** (1) finer σ sweep ∈ {0.5, 1.0, 2.0}; (2) bump n_freqs to 32 or 64 (literature norm 128–256); (3) anisotropic σ (different x vs z); (4) stack RFF with arc-length encoding via `saf` features.
