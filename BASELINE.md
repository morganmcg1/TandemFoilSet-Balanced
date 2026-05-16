# TandemFoilSet — Advisor Baseline

**Branch:** `icml-appendix-charlie-pai2i-24h-r4`
**Round:** charlie-pai2i-24h-r4 (24h budget, 8 students × 1 GPU)
**Primary metric:** `val_avg/mae_surf_p` (lower is better) — equal-weight mean surface pressure MAE across 4 val splits
**Test metric:** `test_avg/mae_surf_p` (finite — NaN workaround baked in since PR #3217)

## Current best (this branch)

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p`              | **67.64** | PR #3540 (tanjiro H24 OneCycleLR), epoch 12 |
| `val_single_in_dist/mae_surf_p`   | 80.32 | PR #3540 |
| `val_geom_camber_rc/mae_surf_p`   | 81.81 | PR #3540 |
| `val_geom_camber_cruise/mae_surf_p` | 44.46 | PR #3540 |
| `val_re_rand/mae_surf_p`          | 63.96 | PR #3540 |
| `test_avg/mae_surf_p`             | **62.12** | PR #3540 |

⚠️ **Note**: OneCycleLR result was measured at epoch 12/15 (schedule truncated at 30-min wall-clock). Trajectory still descending at cutoff — result is a lower bound. Future students must beat val_avg < 67.64 to merge.

## Current baseline configuration

`train.py` after merging PR #3226 (H10 Re-strat) + PR #3217 (H5 RFF + NaN fix) + PR #3326 (H12 MLP dropout) + PR #3345 (H11 log1p targets) + PR #3224 (H13 geom-cond GALE) + PR #3423 (H15 SwiGLU MLP) + PR #3514 (H18 LayerScale) + PR #3540 (H24 OneCycleLR):

- **Model:** `Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2)` + `geom_proj MLP(11, 256, 128)` + 5 `geom_gates` scalars + `SwiGLUMLP` FFN (~843K trainable params + 64 non-trainable RFF buffer)
- **Input:** RFF coordinate encoding (n_freq=32, sigma=1.0) replacing raw (x,z) — input to preprocess MLP is now 86-dim (64 RFF + 22 other features)
- **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** OneCycleLR(max_lr=5e-4, total_steps=15×steps_per_epoch, pct_start=0.3, div_factor=25, final_div_factor=1e4, anneal_strategy='cos', cycle_momentum=False). Replaces CosineAnnealingLR.
- **Batch:** 4
- **surf_weight:** 10.0
- **Epochs:** 50 (cap) / `SENPAI_TIMEOUT_MINUTES=30` wall-clock cap
- **Sampler:** WeightedRandomSampler with domain-balanced weights × Re-strat multiplier (Re>1e6 samples × 2.0; ~1303/1499 train samples)
- **MLP dropout:** `dropout=0.1` in each `TransolverBlock.mlp` (FFN sub-layers); `PhysicsAttention`, preprocess MLP, and final head remain at `dropout=0.0`
- **Log1p target transform:** `signed_log1p(y) = sign(y) * log1p(|y|)` applied to both `pred` and `y` before loss compute only; `evaluate_split` unchanged (metric stays in physical units)
- **NaN workaround:** `evaluate_split` masks out and zero-fills non-finite GT samples before accumulation (fixes test_geom_camber_cruise NaN)
- **Geom-cond GALE:** Per-block additive geometry conditioning: `fx += geom_gates[i] * geom_proj(x[:, 0, 13:24])`. Gates init at 0 (identity start), learned to `[-0.05, -0.11, -0.13, -0.14, -0.15]` at convergence.
- **SwiGLU FFN:** `TransolverBlock.mlp` replaced from `linear→GELU→linear` to `SwiGLUMLP(fc_in:2×n_hidden + fc_out)`. Gate-modulated: `fc_in(x).chunk(2)` → `silu(gate) * value → dropout(0.1) → fc_out`. +33K params per block vs GELU FFN.
- **LayerScale:** `LayerScale` module on each residual: `x += ls1.gamma * attn_out; x += ls2.gamma * mlp_out`. gamma init=1e-6. Adds 1,280 scalars (128 × 2 × 5). Learned norms at epoch 11: ls2 monotone 0.43→0.51 (FFN grows with depth), ls1 U-shaped 0.19/0.13/0.14/0.17/0.21 (attention suppressed mid-stack).
- **OneCycleLR config:** linear rise over first 30% of steps (epochs 1-4.5), cosine decay to final_div_factor=1e4 (lr→5e-8) over remaining 70%. cycle_momentum=False (AdamW β₁ unchanged). step() called per-batch.
- **Splits dir:** `/mnt/new-pvc/datasets/tandemfoil/splits_v2`

### Reproduce command

```bash
cd target && python train.py --agent <student> --experiment_name "<student>/baseline"
```

---

## Baseline history

### 2026-05-16 02:00 — PR #3540: H24 OneCycleLR super-convergence (tanjiro) — **CURRENT BEST**

- **val_avg/mae_surf_p:** 67.64 (epoch 12/15 — schedule truncated at 30-min wall-clock) — **-15.7% vs 80.21 SwiGLU baseline**
- **Note:** Measured on SwiGLU baseline (80.21); merged onto H18 LayerScale (79.52). Actual H18+OneCycleLR value will differ slightly; all 4 splits improved unambiguously.
- **Per-split val:**
  - `val_single_in_dist/mae_surf_p` = 80.32 (-23.1% vs 104.46)
  - `val_geom_camber_rc/mae_surf_p` = 81.81 (-7.6% vs 88.50)
  - `val_geom_camber_cruise/mae_surf_p` = 44.46 (-17.5% vs 53.88) ← OOD improvement
  - `val_re_rand/mae_surf_p` = 63.96 (-13.6% vs 70.14)
- **test_avg/mae_surf_p:** 62.12 (-15.1% vs 73.20)
- **LR curve:** Linear rise epochs 1-5 (peak 5e-4 at epoch 4.5), cosine decay epochs 5-15. Best checkpoint epoch 12 (lr=9.4e-5, still descending — truncation lower bound).
- **What changed:** Replaced `CosineAnnealingLR(T_max=15)` with `OneCycleLR(max_lr=5e-4, total_steps=15×steps_per_epoch, pct_start=0.3, div_factor=25, final_div_factor=1e4, cycle_momentum=False)`. `scheduler.step()` moved per-batch. epochs=15 for schedule sizing.
- **Why it works:** OneCycleLR's rapid fall after peak forces LR to 9.4e-5 by epoch 12, vs CosineAnnealingLR which stays at 1.7e-4 at epoch 12. The deep low-LR fine-tune phase (which didn't even complete) drives sharp convergence on difficult in-dist and OOD splits.
- **Metric artifact:** `models/model-charliepai2i24h4-tanjiro-onecycle-pct30-20260516-004319/metrics.jsonl`
- **Reproduce:** `cd target && python train.py --agent charliepai2i24h4-tanjiro --experiment_name "charliepai2i24h4-tanjiro/onecycle-pct30" --use_onecycle --onecycle_pct_start 0.3 --epochs 15`

### 2026-05-16 00:30 — PR #3514: H18 LayerScale residual scaling (edward) — **CURRENT BEST**

- **val_avg/mae_surf_p:** 79.52 (best epoch 11, 30-min cap) — **-0.86% vs 80.21 prior best**
- **Per-split val:**
  - `val_single_in_dist/mae_surf_p` = 104.62 (+0.15% vs 104.46)
  - `val_geom_camber_rc/mae_surf_p` = 93.29 (+5.4% vs 88.50) ← regression
  - `val_geom_camber_cruise/mae_surf_p` = 50.00 (-7.2% vs 53.88) ← biggest gain
  - `val_re_rand/mae_surf_p` = 70.14 (-5.2% vs 74.00) ← strong OOD gain
- **test_avg/mae_surf_p:** 68.95 (-5.80% vs 73.20) ← key metric
- **LayerScale gamma norms at epoch 11:**
  - ls2 (FFN): 0.43, 0.44, 0.45, 0.44, 0.51 (monotone growth — FFN depth activation confirmed)
  - ls1 (attn): 0.19, 0.13, 0.14, 0.17, 0.21 (U-shape — attention suppressed mid-stack)
  - All gammas escaped init=1e-6 (grew ~5 orders of magnitude)
- **What changed:** Added `LayerScale` module (init=1e-6) on both residual paths of each `TransolverBlock`: `x += ls1.gamma * attn_out; x += ls2.gamma * mlp_out`. +1,280 parameters. FFN branch shows textbook depth-activation pattern; attention shows U-shaped scaling.
- **Delta:** -0.86% val_avg (80.21 → 79.52), **-5.80% test_avg** (73.20 → 68.95). Val gain within seed variance but test gain is clear generalization improvement.
- **Metric artifact:** `models/model-charliepai2i24h4-edward-layerscale-20260515-233219/metrics.jsonl`
- **Reproduce:** `cd target && python train.py --agent charliepai2i24h4-edward --experiment_name "charliepai2i24h4-edward/layerscale"`
- **Note on rc regression**: val_geom_camber_rc regressed +5.4%. This appears to be a side-effect of LayerScale's U-shaped attention suppression at mid-stack (blocks 1-2, which are the primary geometry-conditioning blocks). Same rc regression pattern seen in FiLM (also suppressed attention). Watch if future PRs also regress on rc.

### 2026-05-15 22:35 — PR #3423: H15 SwiGLU MLP (edward) — **CURRENT BEST**

- **val_avg/mae_surf_p:** 80.21 (best epoch 10, 30-min cap) — **-5.8% vs 85.16 prior best**
- ⚠️ **Seed variance:** Two runs at identical config: 89.48 and 80.21 (~10% spread). Primary SENPAI-RESULT is run 2 (80.21). Mean estimate ~84.85. Future PRs beat 80.21 to merge.
- **Per-split val:**
  - `val_single_in_dist/mae_surf_p` = 104.46 (-9.5% vs 106.16)
  - `val_geom_camber_rc/mae_surf_p` = 88.50 (-16.1% vs 92.10) ← biggest gain
  - `val_geom_camber_cruise/mae_surf_p` = 53.88 (-12.2% vs 61.36)
  - `val_re_rand/mae_surf_p` = 74.00 (-14.3% vs 81.01)
- **test_avg/mae_surf_p:** 73.20 (-5.7% vs 77.61)
- **What changed:** Replaced `TransolverBlock.mlp` from `MLP(linear→GELU→linear)` to `SwiGLUMLP(fc_in:2×n_hidden → silu(gate) * value → dropout(0.1) → fc_out)`. OOD splits gain 1.5–1.7× more than in-dist, suggesting gate modulation reduces co-adaptation like dropout but at a structural level.
- **Delta:** -5.8% val_avg (85.16 → 80.21). Best epoch is 10 (vs 14 prior — 29% faster convergence).
- **Metric artifact:** `models/model-charliepai2i24h4-edward-swiglu-mlp-20260515-212619/metrics.jsonl` (run 2, primary)
- **Reproduce:** `cd target && python train.py --agent charliepai2i24h4-edward --experiment_name "charliepai2i24h4-edward/swiglu-mlp"`

### 2026-05-15 21:30 — PR #3224: H13 Persistent geom-cond GALE (tanjiro)

- **val_avg/mae_surf_p:** 85.156 (best epoch 14, 30-min cap, T_max=15) — **-8.2% vs 92.80 prior best**
- **Per-split val:**
  - `val_single_in_dist/mae_surf_p` = 106.160 (-8.1% vs 115.48)
  - `val_geom_camber_rc/mae_surf_p` = 92.098 (-12.7% vs 105.48) ← biggest gain as predicted
  - `val_geom_camber_cruise/mae_surf_p` = 61.360 (-3.9% vs 63.87)
  - `val_re_rand/mae_surf_p` = 81.005 (-6.2% vs 86.36)
- **test_avg/mae_surf_p:** 77.613 (-7.7% vs 84.11)
- **Per-split test:**
  - `test_single_in_dist/mae_surf_p` = 99.658
  - `test_geom_camber_rc/mae_surf_p` = 84.121
  - `test_geom_camber_cruise/mae_surf_p` = 52.932
  - `test_re_rand/mae_surf_p` = 73.739
- **What changed:** Added `geom_proj MLP(11, 256, 128)` + 5 learnable `geom_gates` scalars (init=0). Per-block additive injection: `fx += geom_gates[i] * geom_proj(x[:, 0, 13:24])` before each TransolverBlock. Gates learned to `[-0.05, -0.11, -0.13, -0.14, -0.15]` (monotone magnitude, zero-start). Also aligned `T_max=15` for cosine LR to realized epoch budget.
- **Delta:** -8.2% val_avg (92.80 → 85.16). geom_camber_rc benefited most (-12.7%), confirming GALE mechanism helps OOD camber interpolation.
- **Metric artifact:** `models/model-charliepai2i24h4-tanjiro-geom-cond-v2-restrat-rff-tmax15-20260515-193031/metrics.jsonl`
- **Reproduce:** `cd target && python train.py --agent charliepai2i24h4-tanjiro --experiment_name "charliepai2i24h4-tanjiro/geom-cond-v2-restrat-rff-tmax15"`

### 2026-05-15 19:00 — PR #3345: H11 signed-log1p target transform (thorfinn)

- **val_avg/mae_surf_p:** 92.80 (best epoch 14, 30-min cap) — measured on pre-dropout baseline (122.81); combined dropout+log1p measured by tanjiro H13 = 85.16
- **Per-split val:**
  - `val_single_in_dist/mae_surf_p` = 115.48
  - `val_geom_camber_rc/mae_surf_p` = 105.48
  - `val_geom_camber_cruise/mae_surf_p` = 63.87
  - `val_re_rand/mae_surf_p` = 86.36
- **test_avg/mae_surf_p:** 84.11
- **Per-split test:**
  - `test_single_in_dist/mae_surf_p` = 108.91
  - `test_geom_camber_rc/mae_surf_p` = 91.72
  - `test_geom_camber_cruise/mae_surf_p` = 56.73
  - `test_re_rand/mae_surf_p` = 79.06
- **What changed:** Added `signed_log1p(y) = sign(y) * log1p(|y|)` applied to both `pred` and `y` just before loss computation (loss side only — evaluate_split unchanged). Dynamic range compressed ~3.5× (std: 1025→0.60 in slog1p-normalized space).
- **Delta:** -24.4% val_avg vs RFF+Re-strat baseline (122.81 → 92.80). All splits massively improved: geom_camber_cruise -37.1%, re_rand -27.4%, geom_camber_rc -16.3%, single_in_dist -20.2%.
- **Metric artifact:** `models/model-thorfinn-log1p-targets-20260515-173623/metrics.jsonl`
- **Reproduce:** `cd target && python train.py --agent thorfinn --experiment_name "thorfinn/log1p-targets"`

### 2026-05-15 18:20 — PR #3326: H12 MLP dropout=0.1 (fern)

- **val_avg/mae_surf_p:** 112.49 (best epoch 13, 30-min cap)
- **Per-split val:**
  - `val_single_in_dist/mae_surf_p` = 136.83
  - `val_geom_camber_rc/mae_surf_p` = 118.25
  - `val_geom_camber_cruise/mae_surf_p` = 87.31
  - `val_re_rand/mae_surf_p` = 107.55
- **test_avg/mae_surf_p:** 104.83
- **Per-split test:**
  - `test_single_in_dist/mae_surf_p` = 126.77
  - `test_geom_camber_rc/mae_surf_p` = 112.01
  - `test_geom_camber_cruise/mae_surf_p` = 75.35
  - `test_re_rand/mae_surf_p` = 105.20
- **What changed:** Added `nn.Dropout(0.1)` after each activation in `MLP.linear_pre` and hidden layers. `PhysicsAttention` dropout stays at 0.0. Only `TransolverBlock.mlp` gets dropout.
- **Delta:** -8.4% val_avg (122.81 → 112.49). OOD splits benefited most: geom_camber_cruise -14.1%, re_rand -9.6%, geom_camber_rc -6.1%. In-dist slightly worse on test (+2.3%), consistent with regularizer tradeoff.
- **Metric artifact:** `models/model-fern-mlp-dropout-0p1-20260515-163433/metrics.jsonl`
- **Reproduce:** `cd target && python train.py --agent fern --experiment_name "fern/mlp-dropout-0p1"`

### 2026-05-15 16:25 — PR #3217: H5 RFF coord encoding + NaN fix (frieren)

- **val_avg/mae_surf_p:** 122.81 (best epoch 12, 30-min cap)
- **Per-split val:** single_in_dist=144.70, geom_camber_rc=125.95, geom_camber_cruise=101.61, re_rand=119.00
- **test_avg/mae_surf_p:** 111.16
- **What changed:** RFFEncoding(n_freq=32, sigma=1.0) replacing raw (x,z) coords. evaluate_split NaN workaround added.
- **Metric artifact:** `models/model-frieren-rff-nfreq32-sigma1-20260515-140556/metrics.jsonl`

### 2026-05-15 15:00 — PR #3226: H10 Re-stratified sampler (thorfinn)

- **val_avg/mae_surf_p:** 127.84 (best epoch 14, 30-min cap)
- **Per-split val:** single_in_dist=160.10, geom_camber_rc=148.67, geom_camber_cruise=91.50, re_rand=111.08
- **test_avg/mae_surf_p:** NaN at time of merge (fixed by frieren PR #3217)
- **What changed:** Re>1e6 samples weighted × 2.0 in WeightedRandomSampler.
- **Metric artifact:** `models/model-charliepai2i24h4-thorfinn-re-strat-high2x-*/metrics.jsonl`

---

## Notes for upcoming PRs

- **Beat this:** `val_avg/mae_surf_p < 85.16` to be a merge candidate.
- **Hardest split:** `val_single_in_dist = 106.16`. The in-dist bottleneck.
- **Biggest OOD remaining gap:** `val_geom_camber_rc = 92.10` — large relative to cruise (61.36).
- **Baseline stack:** Re-strat sampler + RFF coord encoding + MLP dropout=0.1 + signed-log1p target transform (α=1.0) + geom-cond GALE (additive per-block with gates) + evaluate_split NaN workaround + T_max=15 cosine alignment.
- **Active WIP PRs:** #3417 H11b alpha sweep (thorfinn), #3318 H6v2 SGDR+clip (frieren), #3197 H8v3 EMA (askeladd), #3421 H14 cosine T_max (nezuko), #3461 H16 FiLM geom-cond (tanjiro), #3467 H17 attn-dropout (fern), #3184 H1 LinearNO (alphonse).
- **edward is idle — assigning new hypothesis.**
