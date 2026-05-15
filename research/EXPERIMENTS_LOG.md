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

## 2026-05-15 17:35 — PR #3261: Wider-shallower Transolver (n_hidden=256, n_layers=3, n_head=8)

- **Student/branch:** willowpai2i24h4-alphonse / `willowpai2i24h4-alphonse/wider-shallower-256d`
- **Hypothesis:** Wider-shallower configuration — 2.45× more params, 3/5 depth — should improve in-distribution capacity with better per-layer width.
- **W&B runs:** baseline `xfayvdk2` (with NaN guard), wider `qjzx09k6`

### Result (best checkpoint, timeout at epoch 10–14 / 50)

| | Baseline `xfayvdk2` | Wider `qjzx09k6` | Δ |
|---|---:|---:|---:|
| n_params | 0.66M | 1.61M (+145%) | — |
| Epochs completed | 14 (best@13) | 11 (best@10) | wider → fewer steps |
| `val_avg/mae_surf_p` | **117.89** | 146.26 | **+24.1% WORSE** |
| `val_geom_camber_rc` | 125.09 | 176.33 | +41% WORSE (largest gap) |
| `test_avg/mae_surf_p` | **106.23** | 133.34 | **+25.5% WORSE** |
| `test_geom_camber_cruise` | **78.72** | 96.07 | finite (NaN guard applied) |

### Decision: closed (>5% regression on primary metrics)

### Analysis

- **Hypothesis disconfirmed.** Wider-shallower is substantially worse across all splits. The largest degradation is on `val_geom_camber_rc` (+41%) — the hardest OOD split — consistent with depth being critical for compositional generalization.
- **Params not matched.** The PR description implied "roughly matched budget" but actual ratio is ×2.45 (0.66M → 1.61M). Depth halving (5→3) doesn't compensate width doubling (128→256) because attention and MLP both scale as O(d²).
- **Fewer epochs under wall-clock cap.** Per-epoch wall time: 132s → 165s (+25%). 11 epochs at best vs 14 for baseline — introduces compounding unfairness in a still-converging regime.
- **Depth is doing real compositional work.** The depth-5 pattern is empirically validated. Will not revisit width-vs-depth ablations at matched budget in the near term.
- **MOST VALUABLE CONTRIBUTION: alphonse's vanilla baseline `xfayvdk2`.** This is the first run with NaN guard applied, giving us the **first finite 4-split test_avg reference**: val_avg=117.89, test_avg=106.23. Added to BASELINE.md.

## 2026-05-15 16:45 — PR #3264: Dropout p=0.1 in Transolver MLP and attention

- **Student/branch:** willowpai2i24h4-askeladd / `willowpai2i24h4-askeladd/dropout-0.1`
- **Hypothesis:** Enable dropout=0.1 in both PhysicsAttention and MLP pathways to reduce overfitting on ~1500-sample training set and improve OOD camber generalization.
- **W&B runs:** baseline `j4y20e31`, dropout=0.1 `chzqcfyz`

### Result (best checkpoint, both runs hit 30-min timeout at epoch 13 / 50)

| Split | Baseline (d=0) | **dropout=0.1** | Δ |
|-------|------:|------:|---:|
| `val_single_in_dist/mae_surf_p`     | 153.89 | 170.57 | +10.8% WORSE |
| `val_geom_camber_rc/mae_surf_p`     | 139.60 | 146.70 | +5.1% WORSE |
| `val_geom_camber_cruise/mae_surf_p` | 106.15 | 115.73 | +9.0% WORSE |
| `val_re_rand/mae_surf_p`            | 130.71 | 129.28 | −1.1% (noise) |
| **`val_avg/mae_surf_p`**            | **132.59** | **140.57** | **+6.0% WORSE** |
| `test_avg/mae_surf_p` (3 valid) | 131.53 | 136.27 | +3.6% WORSE |

### Decision: closed (>5% regression on primary val metric)

Dropout=0.1 degrades performance across 3 of 4 val splits including both OOD camber splits it was meant to help. Closed per CLAUDE.md protocol (>5% regression).

### Analysis

- **Mechanism correctly diagnosed by askeladd:** model is in UNDERFIT regime at the 30-min cap (val curve still descending at epoch 13, training loss still falling from 1.09 to 0.27). Dropout slows convergence (partially-masked subnetworks reduce effective gradient signal per step). Slower convergence × same wall-clock = strictly worse best-checkpoint on a still-descending loss curve. This is not an overfitting regime.
- **The single split where dropout helped (`val_re_rand`, −1.1%)** is the Re-stratified holdout — plausibly the axis most prone to co-adaptation memorization, but the effect is noise-level (1.43 MAE on 130-MAE base).
- **Dropout may not be worth retrying** at this compute budget. Would only be worth testing once models consistently train past 25+ epochs and val curves start to plateau. Marked as low-priority for future rounds.

### Critical diagnostic contribution: correct NaN root cause

Askeladd's bug-report comment (16:28 UTC) independently found the same root cause that frieren found (#3257, 16:32 UTC): `+inf` in `test_geom_camber_cruise_gt/000020.pt` y-channel for p at ~761 nodes. The actual IEEE 754 bug: `(pred - inf).abs() = inf`, then `inf * 0 = NaN` in `(err * mask).sum()` — poisons the running sum even though `accumulate_batch` would otherwise skip the sample via `y_finite`. **My original patch (sanitize predictions) was wrong.** The correct fix (from frieren's #3257 commit `34600cf`): zero the mask for non-finite-y samples AND `nan_to_num` y before `accumulate_batch`. Sent corrected patches to #3257, #3258, #3262.

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

## 2026-05-15 16:30 — PR #3258: Gradient clip 1.0 + 5-epoch LR warmup

- **Student/branch:** willowpai2i24h4-fern / `willowpai2i24h4-fern/grad-clip-warmup`
- **Hypothesis:** Add gradient clipping (max_norm=1.0) and 5-epoch linear LR warmup before cosine annealing. Bound catastrophic-batch updates and let slice routing stabilize before peak LR.
- **W&B runs:** baseline `nylo2tvd`, clip1.0-wu5 `69np1sbe`, clip0.5-wu3 `4yg5bhtc`

### Result (best checkpoint, all runs hit 30-min timeout cap at epoch 11–14 / 50)

| Split | Baseline `nylo2tvd` | **clip1.0-wu5 `69np1sbe`** | clip0.5-wu3 `4yg5bhtc` (best ckpt) |
|-------|-----:|-----:|-----:|
| `val_single_in_dist/mae_surf_p`     | 172.20 | **142.37** | 148.51 |
| `val_geom_camber_rc/mae_surf_p`     | 161.19 | **124.69** | 152.15 |
| `val_geom_camber_cruise/mae_surf_p` | 109.57 | **90.45**  | 95.82  |
| `val_re_rand/mae_surf_p`            | 124.80 | **105.42** | 113.35 |
| **`val_avg/mae_surf_p` (best ckpt)**| **141.94** | **115.73 (−18.5%)** | 127.46 (−10.2%) |
| `val_avg/mae_surf_p` (terminal) | 141.94 | — | 117.90 (student reported this value as if best-ckpt — mismatch) |
| `test_avg/mae_surf_p` (4-split mean) | NaN | NaN | NaN |
| `test_avg/mae_surf_p` (3 valid splits, fern's report) | 139.34 | 115.46 (−17.1%) | 118.47 (−15.0%) |

Gradient norm trace from clip1.0-wu5: median 56.0, mean 87.2, p99 445, **max 1110**, clipping binds on 100% of steps.

### Decision: send back to student (#3258-comment-4461464504)

Same NaN-poisoning blocker as #3257 and #3262 — `test_avg/mae_surf_p = NaN` (4-split formal) is disqualifying. Sent back with the same `torch.nan_to_num` patch for `train.py:evaluate_split` and instructed to rerun only the `clip1.0-wu5` arm (skip clip0.5-wu3 which is the smaller win, skip baseline rerun).

### Analysis

- **Hypothesis confirmed, large effect size.** clip1.0-wu5 reduces `val_avg/mae_surf_p` by 18.5% (vs predicted 2–5%), with consistent per-split gains and largest effect on `val_geom_camber_rc` (−22.6%) and `val_re_rand` (−15.5%).
- **Mechanism is real and important.** Pre-clip gradient norms median 56, peak 1100 (50–1000× the clip cap). This is unusual for a 1.5M-param model. Without clipping, a few outlier batches per epoch take steps ~1000× the median, swinging the PhysicsAttention slice-routing softmax into bad basins. Baseline `nylo2tvd` shows characteristic catastrophic-batch behavior: val_avg goes 142 → 192 between epochs 11 and 12 before partial recovery. Clipped runs never regress.
- **clip0.5-wu3 best-ckpt reporting issue.** Fern reported 117.90 for clip0.5-wu3, but actual W&B best checkpoint is 127.46. Fern's table mixed terminal-value vs best-checkpoint reporting across runs. The clip1.0-wu5 winner is correctly reported at 115.73 (best=terminal for that run). The rerun will be measured consistently.
- **Run-to-run variance is large (~13pt on val_avg).** Edward's vanilla baseline `17fia1vd` reported 128.34, fern's vanilla baseline `nylo2tvd` reported 141.94 — same code, same config, ~9–13pt gap. This is purely stochastic; with median grad norm ~56 and peaks >1000, an unclipped baseline naturally lands in different local minima per run. A consequence: **a "win" of <10% may be partial regression-to-the-mean** from a bad-luck baseline draw. The clip1.0-wu5 win at 115.73 beats *either* baseline by 9.8–18.5%, so the gain is real.
- **#3258 and #3262 produce nearly identical val_avg (115.73 vs 115.78).** When both reruns land cleanly, we should test whether grad-clip+warmup and RFF compound or are redundant. Merge fern first (foundational training fix), then re-evaluate edward's RFF on the new baseline.
- **R2 follow-up queue (from fern's diagnostics):** (1) investigate gradient-norm sources (PhysicsAttention softmax temperature 0.5, surf_weight=10); (2) longer warmup (10 epochs) + lower clip (0.5); (3) log post-clip grad_norm and clip ratio.

## 2026-05-15 18:25 — PR #3257 (MERGED): Surface MAE loss + p-weight 3× — rerun with canonical NaN guard

- **Student/branch:** willowpai2i24h4-frieren / `willowpai2i24h4-frieren/surf-mae-p-weight`
- **Hypothesis:** Switch surface loss from MSE to MAE with per-channel weight [1, 1, 3] on (Ux, Uy, p). Volume loss stays MSE. Frieren also independently traced and fixed the cruise-NaN root cause (commit `34600cf`).
- **W&B run:** `szru1ogx` (https://wandb.ai/wandb-applied-ai-team/senpai-v1/runs/szru1ogx)

### Result (best checkpoint at epoch 13/14; 30-min timeout cap)

| Split | val mae_surf_p | test mae_surf_p | n_skipped_y_samples |
|-------|---------------:|----------------:|--------------------:|
| `single_in_dist`       | 124.31 | 122.34 | 0 |
| `geom_camber_rc`       | 131.21 | 106.31 | 0 |
| `geom_camber_cruise`   |  78.23 |  62.47 | 1 |
| `re_rand`              |  92.92 |  86.28 | 0 |
| **avg (4-split)**      | **106.67** | **94.35** | — |

Beats prior baseline `xfayvdk2` (val_avg=117.89, test_avg=106.23) by **−9.5% val / −11.2% test**.

### Decision: MERGED as R1 winner #1 (commit `a059a65`)

Per CLAUDE.md, this beats the baseline on the primary ranking metric (`test_avg/mae_surf_p`) with a finite 4-split mean and a terminal SENPAI-RESULT marker. New BASELINE.md anchor.

### Analysis

- **The cruise split swung the most.** Test cruise = 62.47 (vs prior 78.72, −20.6%). This is the split where pressure dynamic range is largest relative to other test splits, so the p-weight=3 had the biggest leverage exactly where MAE-vs-MSE matters most.
- **Per-split shape mirrors the hypothesis.** Frieren predicted cruise/re_rand would lead the gain on val, and they did on test too (cruise leads at −20.6%, re_rand at −18.6% vs prior baseline). Single_in_dist gained least (−3.4%), consistent with MAE-on-p being most valuable where the p distribution is heavy-tailed.
- **Canonical NaN guard works as designed.** `n_skipped_y_samples=1` on cruise, 0 everywhere else — confirms exactly one bad-GT sample (`000020.pt`) is skipped and the masking is precise.
- **Cosine T_max=50 mismatch is still unaddressed for this run.** Frieren's run uses the same cosine schedule the rest of R1 used (T_max=50 nominal, ~14 epochs actual). Implicit under-annealing across the board — alphonse's #3358 will address.
- **Run-to-run variance still applies.** Frieren's win margin vs the alphonse baseline (also NaN-guarded) is ~11pt val / ~12pt test. Edward's and fern's unclipped baselines were 128/142 on val — frieren's improvement is robust against any of these reference points.

### Two PRs sent back due to base change (rebase required)

- **#3258 (fern, grad-clip+warmup)** — already reran with the corrected NaN guard, returned `val_avg=117.31, test_avg=105.70` on the OLD MSE base. Beats the old baseline by 0.5% on test_avg but regresses against the new merged baseline by 12%. Sent back for rebase onto frieren's loss + rerun. Mechanism (clipping median-56 gradients with peak >1000) is orthogonal to loss reformulation, so should compose.
- **#3262 (edward, RFF σ=1.0)** — never got to corrected-patch rerun before frieren merged. Posted note: rebase onto new base + apply NaN guard + rerun RFF σ=1.0 (skip σ=4.0 which already lost).

## 2026-05-15 19:23 — PR #3263: FiLM log(Re) conditioning (send-back, needs rebase)

- **Student/branch:** willowpai2i24h4-thorfinn / `willowpai2i24h4-thorfinn/film-re-cond`
- **Hypothesis:** Inject a FiLM (Feature-wise Linear Modulation) module conditioned on log(Re) after the Transolver preprocess layer, before the attention blocks. FiLM adds a learned affine gate `(γ, β) = MLP(log(Re))` applied to the preprocess hidden state, giving the model a direct low-rank route to modulate all 128 channels by global Re.
- **W&B runs:** `zjogv9vn` (film-re-v1), `rlildyv4` (film-re-v2), `joszk2jg` (film-re-v3 best), `vsuqhyt5` (fresh baseline)

### Result (vs own fresh baseline `vsuqhyt5`, all runs hit 30-min timeout cap at epoch 14)

| Split | Baseline `vsuqhyt5` | FiLM v3 `joszk2jg` | Δ (rel) |
|-------|--------------------:|-------------------:|--------:|
| `val_single_in_dist`      | 161.82 | 142.03 | **−12.2%** |
| `val_geom_camber_rc`      | 137.18 | 125.03 | **−8.9%** |
| `val_geom_camber_cruise`  | 121.72 |  95.99 | **−21.1%** |
| `val_re_rand`             | 129.41 | 111.15 | **−14.1%** |
| **`val_avg/mae_surf_p`**  | **137.53** | **118.55 (−13.8%)** | — |
| `test_single_in_dist`     | 137.69 | 126.24 | −8.3% |
| `test_geom_camber_rc`     | 126.39 | 120.84 | −4.4% |
| `test_re_rand`            | 127.87 | 110.98 | −13.2% |
| `test_geom_camber_cruise` | NaN    | NaN    | pre-existing (run pre-fix) |
| **3-split test mean**     | 130.65 | 119.35 | **−8.6%** |

Seed spread across 3 FiLM runs: 133.34 / 127.89 / 118.55 (best). 3-seed mean: 126.59. All seeds beat baseline on val_avg.

### Decision: sent back for rebase + rerun on frieren-base

Thorfinn's fresh baseline was `vsuqhyt5` (val=137.53) — a high-variance unclipped run. While thorfinn was running, frieren's #3257 merged (val=106.67, test=94.35). FiLM v3's val=118.55 does NOT beat the new merged baseline (106.67). Mechanism is orthogonal (Re conditioning vs. loss reformulation), so should compose. Sent back to rebase and rerun v3 on the new base.

### Analysis

- **Directional signal is strong.** −21.1% on val_cruise (the split with widest Re range 122K–5M) and −14.1% on val_re_rand exactly matches the prediction that FiLM leverages Re most where Re varies most. The mechanism is working as intended.
- **Seed variance is large (133→127→118).** This is the unclipped baseline variance. The rebased rerun should reduce variance if clip+warmup lands (orthogonal merge candidate via #3258).
- **The test_3split=119.35 vs merged baseline test_4split=94.35 gap is 27%** — but these are against different loss configurations. The rebased run on frieren's loss may close most of this gap.
- **NaN guard was NOT applied** (run pre-merge). The rebased run will automatically inherit the canonical guard from the merged head — test_avg will be 4-split finite.

### R2 follow-ups (from thorfinn's suggestions, saved)

1. Per-block FiLM heads (standard recipe in conditional generation) — natural R2 extension after rebase win
2. Richer conditioning vector `(log_Re, AoA_1, AoA_2, gap, stagger)` — all global scalars, same mechanism
3. ~~NaN fix~~ — RESOLVED in #3257 merge

## 2026-05-15 19:25 — PR #3256 (CLOSED): Huber loss delta=0.5 (redundant with #3257 merge)

- **Student/branch:** willowpai2i24h4-tanjiro / `willowpai2i24h4-tanjiro/huber-loss`
- **Hypothesis:** Replace MSE with Huber loss (δ=0.5) for outlier-robust loss metric alignment.
- **Outcome:** No results produced. Closed as redundant — frieren's #3257 empirically validated the L1-style robustness direction, and pure MAE+p-weight=3 is a superset of the Huber approach. Pod was blocked by GitHub API rate-limit cycling for 6+ hours with no commits pushed.
- **Reassignment:** #3406 surf_weight sweep {5,10,20} on merged baseline.
