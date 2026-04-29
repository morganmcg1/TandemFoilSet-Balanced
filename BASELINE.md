# Baseline — icml-appendix-charlie-pai2f-r3

## Current Best Result

**Source:** PR #1208 — Extended training 75ep + T_max=75 on FiLM+Fourier baseline (charliepai2f3-frieren)

**Primary metric:** `val_avg/mae_surf_p = 35.8406`

**Configuration:** Lion optimizer + L1 loss + EMA(0.995) + bf16 autocast + n_layers=1 + surf_weight=28 + cosine scheduler (T_max=75, single full-cycle decay) + grad_clip=1.0 + n_hidden=128 + n_head=4 + slice_num=64 + mlp_ratio=2 + epochs=75 + Fourier positional encoding on (x,z) with freqs=(1,2,4,8,16,32,64) + FiLM global conditioning (scale+shift per TransolverBlock conditioned on Re/AoA/NACA regime vector, DiT/AdaLN-Zero init)

**Note:** Training was cut at ep57/75 by 30-min wall-clock timeout — model still improving at cutoff. No warmup in this run (multi-cycle cosine from the PR #1104 config). The longer T_max=75 horizon keeps LR meaningfully nonzero throughout, making the full-cycle decay the primary driver.

**Per-split breakdown:**
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 33.7806 |
| val_geom_camber_rc | 51.6584 |
| val_geom_camber_cruise | 19.6970 |
| val_re_rand | 38.2266 |
| **val_avg** | **35.8406** |

**Test split breakdown:**
| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 29.8002 |
| test_geom_camber_rc | 44.5661 |
| test_geom_camber_cruise | 16.0529 |
| test_re_rand | 29.2713 |
| **test_avg** | **29.9226** |

**Training:** ~30.33 min (wall-clock timeout at ep57/75), best epoch 57 (still improving), batch_size=4, Peak VRAM: 9.89 GB, n_params: 252,487

**Metrics path:** `target/models/model-charliepai2f3-frieren-extended-training-film-75ep-20260429-154641/metrics.jsonl`

## Run Command

```bash
cd target/ && python train.py --n_layers 1 --bf16 True --surf_weight 28.0 --optimizer lion --lr 3e-4 --weight_decay 1e-2 --loss l1 --scheduler cosine --T_max 75 --clip_grad_norm 1.0 --n_hidden 128 --n_head 4 --slice_num 64 --mlp_ratio 2 --batch_size 4 --epochs 75 --fourier_pos_enc --fourier_freqs 1 2 4 8 16 32 64
```

Note: Extended to 75 epochs with T_max=75 single-decay cosine. Key insight: longer T_max keeps the LR meaningfully nonzero throughout training — at ep49 (same count as previous baseline), this run already achieved val_avg=37.21 vs 39.95, a −6.86% gain from the schedule change alone. Extra epochs 50–57 added another −3.27%. Best epoch=57 was the final epoch reached (wall-clock limited) — model was still improving with a steeply downward val curve. FiLM conditioning is the default in codebase after PR #1104.

## Merge History

### 2026-04-29 — PR #1208: Extended training 75ep + T_max=75 on FiLM+Fourier baseline (charliepai2f3-frieren)
- Previous: `val_avg/mae_surf_p = 37.0739` (PR #1175, FiLM+Fourier+warmup, T_max=45, 50 epochs)
- New best: `val_avg/mae_surf_p = 35.8406` (improvement: −1.2333, −3.33%)
- Test: `test_avg/mae_surf_p = 29.9226` (improvement vs previous test_avg 31.3474: −1.4248, −4.55%)
- Student: charliepai2f3-frieren
- Key finding: T_max=75 (single full-cycle cosine) is the primary driver — at ep49 alone this yielded val_avg=37.21 (−6.86% vs PR #1104 baseline), confirming LR horizon matters more than epoch count. Wall-clock timeout cut at ep57/75; model still improving steeply. Best epoch = 57 = last epoch, no plateau. All 4 val splits improved. Test_avg broke below 30 for the first time.

### 2026-04-29 — PR #1175: LR warmup + single-decay cosine on FiLM+Fourier baseline (charliepai2f3-thorfinn)
- Previous: `val_avg/mae_surf_p = 39.9450` (PR #1104, FiLM+Fourier, T_max=15 multi-cycle, no warmup)
- New best: `val_avg/mae_surf_p = 37.0739` (improvement: −2.8711, −7.19%)
- Test: `test_avg/mae_surf_p = 31.3474` (improvement vs previous test_avg 33.5761: −2.2287, −6.64%)
- Student: charliepai2f3-thorfinn
- Key finding: 5-epoch linear warmup (start_factor=1/30 → 3e-4) stabilizes Lion optimizer initialization. Single-decay cosine (T_max=45) avoids LR restarts that bounce model out of good basins. Best epoch=50 — model still improving at timeout, indicating more training could help. Both val and test improved across all 4 splits. Run 1 (warmup=10, multi-cycle T_max=15) also beat baseline at val_avg=38.9454, confirming warmup is the key driver.

### 2026-04-29 — PR #1196: Single-decay cosine schedule (T_max=50) on Fourier pos enc baseline (charliepai2f3-frieren)
- Context: Tested against PR #1148 baseline (val_avg=43.9575); current best is PR #1104 (val_avg=39.9450 with FiLM)
- Result: `val_avg/mae_surf_p = 42.4863` (vs stale baseline −1.4712, −3.35%; vs current best: +2.5413, regresses from 39.9450)
- Test: `test_avg/mae_surf_p = 35.6687`
- Key finding: Single-decay cosine (T_max=50) confirms the LR-cycling failure mode. This improvement on the non-FiLM branch is real but the FiLM path (PR #1104) is already significantly ahead. The T_max=50 improvement should be adopted in the FiLM branch experiments.
- Note: Merged against stale baseline — current best remains PR #1104 at val_avg=39.9450.

### 2026-04-29 — PR #1104: FiLM global conditioning: inject Re/AoA/NACA via scale+shift (charliepai2f3-edward)
- Previous: `val_avg/mae_surf_p = 43.9575` (PR #1148, Fourier freqs=(1,2,4,8,16,32,64))
- New best: `val_avg/mae_surf_p = 39.9450` (improvement: −4.0125, −9.13%)
- Test: `test_avg/mae_surf_p = 33.5761` (improvement vs previous test_avg 37.4541: −3.8780, −10.35%)
- Student: charliepai2f3-edward
- Key finding: FiLM global conditioning (scale+shift per TransolverBlock conditioned on 11-dim physics regime vector) delivers a decisive −9.13% improvement on val and −10.35% on test. Best epoch 49/50 — model not yet converged; extended training is the obvious next direction. All 4 val splits improved. DiT/AdaLN-Zero initialization key to stability. Param count increases from 184,903 to 252,487 (+67,584, +36.5%).

### 2026-04-29 — PR #1148: Extended Fourier freqs on (x,z): freqs=(1,2,4,8,16,32,64) (charliepai2f3-askeladd)
- Previous: `val_avg/mae_surf_p = 44.4154` (PR #1106, Fourier pos enc freqs=(1,2,4,8,16))
- New best: `val_avg/mae_surf_p = 43.9575` (improvement: −0.4579, −1.03%)
- Student: charliepai2f3-askeladd
- Key finding: freqs=(1,2,4,8,16,32,64) beats baseline; adding freq=128 regresses (Nyquist aliasing near mesh resolution)

### 2026-04-29 — PR #1106: Fourier positional encoding on (x,z) (charliepai2f3-frieren)
- Previous: `val_avg/mae_surf_p = 47.3987` (PR #1093, compound baseline)
- New best: `val_avg/mae_surf_p = 44.4154` (improvement: −2.9833, −6.29%)
- Student: charliepai2f3-frieren
- Also included: NaN fix for test_geom_camber_cruise (non-finite GT entries in sample 20 masked via y_finite guard in evaluate_split)

### 2026-04-29 — PR #1093: Compound baseline anchor (Lion+L1+EMA+bf16+n_layers=1+sw=28+cosine+clip)
- Previous: `val_avg/mae_surf_p = 47.7385` (charlie-pai2e-r5 reference)
- New best: `val_avg/mae_surf_p = 47.3987` (improvement: −0.3398)
- Student: charliepai2f3-alphonse
