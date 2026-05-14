# Baseline — `icml-appendix-charlie-pai2g-48h-r5`

This branch is the **Charlie no-W&B logging ablation, round 5 (charlie-pai2g-48h-r5)**.

Experiment metrics are written to local JSONL only (`models/<experiment>/metrics.jsonl`).
**Do not** add or query W&B / wandb experiment logging for this arm.

## Primary ranking metric

- **Validation:** `val_avg/mae_surf_p` — equal-weight mean of surface pressure MAE
  across the four val tracks (`val_single_in_dist`, `val_geom_camber_rc`,
  `val_geom_camber_cruise`, `val_re_rand`). Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` from the best-val checkpoint.

> ✅ **Round-5 scoring bug fixed (merged via PR #1532):** `test_geom_camber_cruise/000020.pt`
> contains ±Inf values in the `p` channel. The `train.py:evaluate_split` workaround
> (batch-level `y_finite_mask` filter before `accumulate_batch`) is now on the
> advisor branch. All subsequent PRs must include this fix on their branch and
> should report **finite `test_avg/mae_surf_p`**. Round-5 ranking remains
> `val_avg/mae_surf_p` as the primary metric.

## Reference configuration (train.py defaults)

```
lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50
model_config = dict(
    space_dim=2, fun_dim=22, out_dim=3,
    n_hidden=128, n_layers=5, n_head=4,
    slice_num=64, mlp_ratio=2,
)
optimizer = AdamW; scheduler = CosineAnnealingLR(T_max=epochs)
```

Each training execution is hard-capped by `SENPAI_TIMEOUT_MINUTES=30` (wall clock).
`--epochs 50` is an upper bound; runs typically reach 12-16 epochs under the
30-min cap at the default model size.

## Current best (val)

| Metric | Value | PR | Config | Notes |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **33.0195** | #2692 | Lion lr=1.5e-4 + FiLM + SE per-block (reduction=8, zero-init fc2); all other params from #2614 unchanged | ep65/70 (timeout-truncated, 5 epochs missing); **−1.06% vs #2614** (33.3722); val OOD splits WIN 3/4; in_dist regresses +4.31%; test essentially flat (−0.06%) |
| `test_avg/mae_surf_p` | **28.3562** | #2692 | — | test from best-val checkpoint ep65; essentially flat −0.06% vs #2614 (28.3736) |

All subsequent PRs must beat `val_avg/mae_surf_p < 33.0195` to be merged.

## 2026-05-14 [Round 84] — PR #2692: Squeeze-Excitation per-block: val OOD WIN −1.06% (NEW BEST)

- **Student:** charliepai2g48h5-tanjiro
- **Best epoch:** 65 of 70 (timeout-truncated at 30 min; 5 cosine epochs missing)
- **Epochs reached:** 65 (~28 s/epoch; +1-2 s vs baseline due to SE compute)
- **Peak GPU memory:** 15.5 GB (slight increase from 14 GB baseline)
- **Param count:** 338,267 (328,619 baseline + 9,648 SE = +2.94%)
- **SE gate stats (terminal, sample batch):** Block 3 engages most — std=0.076, range 0.27–0.71; blocks 1-2 near identity (std ~0.022-0.030); no saturation observed
- **LayerScale γ (terminal):** γ_attn ~0.008-0.020, γ_mlp ~0.051-0.078 — depth-progressive compression at block 3 indicating SE/LayerScale depth trade-off

| Split | val mae_surf_p | Δ vs #2614 (33.3722) |
|---|---|---|
| `val_single_in_dist` | 26.4221 | **+4.31%** REGRESSION |
| `val_geom_camber_rc` | **48.3191** | **−2.54%** |
| `val_geom_camber_cruise` | **20.2953** | **−0.60%** |
| `val_re_rand` | **37.0415** | **−2.94%** |
| **val_avg** | **33.0195** | **−1.06%** |

| Split | test mae_surf_p |
|---|---|
| `test_single_in_dist` | 24.7135 |
| `test_geom_camber_rc` | 42.9508 |
| `test_geom_camber_cruise` | 16.0798 |
| `test_re_rand` | 29.6808 |
| **test_avg** | **28.3562** |

- **Mechanism:** SE finds useful signal at depth (block 3 std 0.076 >> blocks 1-2 std 0.022-0.030). Global-pool gating suppresses decision-irrelevant channels for OOD samples; mildly harmful for in-dist where features are already well-routed. Depth-progressive: deeper blocks lean on SE, shallower on LayerScale (γ_mlp block 3 = 0.051 vs typical ~0.08).
- **Metric artifacts:**
  `models/model-charliepai2g48h5-tanjiro-se-r8-20260514-010915/metrics.jsonl`
  `models/model-charliepai2g48h5-tanjiro-se-r8-20260514-010915/metrics.yaml`
- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-tanjiro \
      --experiment_name "charliepai2g48h5-tanjiro/se-r8" \
      --lr 1.5e-4 \
      --weight_decay 3e-4 \
      --epochs 70
  ```
  (SE module now on advisor branch — stacks with FiLM + LayerScale + all prior wins)

---

## 2026-05-14 00:00 — PR #2614: FiLM feature-stream gate: uniform test WIN −0.89% (NEW BEST)

- **Student:** charliepai2g48h5-alphonse
- **Best epoch:** 70 of 70 (best=terminal; still monotonically descending at cosine endpoint)
- **Epochs reached:** 70 (full schedule; ~25.7 s/epoch)
- **Peak GPU memory:** 14.1 GB (unchanged)
- **Param count:** 328,619 (+384 FiLM params; +0.12% over #2553 baseline 328,235)
- **FiLM diagnostics:** film.weight.norm=2.6245 (from 0), film.bias.norm=0.8504 (from 0); modulation factor ~0.92 on val_re_rand batch

| Split | val mae_surf_p | Δ vs #2553 (33.4935) |
|---|---|---|
| `val_single_in_dist` | **25.3293** | **−1.71%** |
| `val_geom_camber_rc` | **49.5771** | **−1.93%** |
| `val_geom_camber_cruise` | 20.4181 | +0.67% (wash) |
| `val_re_rand` | 38.1642 | +2.12% (mild regression; noise on n=100) |
| **val_avg** | **33.3722** | **−0.36%** |

| Split | test mae_surf_p | Δ vs #2553 |
|---|---|---|
| `test_single_in_dist` | **24.4830** | **−0.90%** |
| `test_geom_camber_rc` | **43.3910** | **−1.04%** |
| `test_geom_camber_cruise` | **16.8389** | **~0%** |
| `test_re_rand` | **28.7816** | **−1.16%** |
| **test_avg** | **28.3736** | **−0.89%** |

- **Mechanism:** Single shared FiLM gate applied before block 0. Re/AoA scalar inputs → Linear(3, 96) → `fx = fx * (1 + gamma(c))`. Zero-init gate grows to norm 2.62 — optimizer found Re/AoA as useful routing signal. Geometric OOD splits (rc, in-dist) benefit most. val_re_rand regression appears to be n=100 small-sample noise (test_re_rand improves −1.16%).
- **Metric artifacts:**
  `models/model-charliepai2g48h5-alphonse-film-feature-stream-20260513-222603/metrics.jsonl`
  `models/model-charliepai2g48h5-alphonse-film-feature-stream-20260513-222603/metrics.yaml`
- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-alphonse \
      --experiment_name "charliepai2g48h5-alphonse/film-feature-stream" \
      --lr 1.5e-4 \
      --weight_decay 3e-4 \
      --epochs 70
  ```
  (FiLM gate now on advisor branch — stacks with all prior wins)

---

## 2026-05-14 21:00 — PR #2553: Lion lr=1.5e-4 sweep: uniform WIN −8.05% (NEW BEST)

- **Student:** charliepai2g48h5-edward
- **Best epoch:** 70 of 70 (best=terminal; still monotonically descending at terminal, LR annealed to 0)
- **Epochs reached:** 70 (full schedule; ~25.6 s/epoch, ~30 min wall-clock)
- **Peak GPU memory:** 14.01 GB (unchanged from #2524)
- **Param count:** 328,235 (unchanged — only lr changed)
- **Lion momentum non-zero fraction:** 0.9958 (fully populated; consistent with #2524 0.9986)

| Split | val mae_surf_p | Δ vs #2524 (36.3994) | Δ vs AdamW #2307 (42.3455) |
|---|---|---|---|
| `val_single_in_dist` | **25.7691** | **−9.60%** | −27.36% |
| `val_geom_camber_rc` | **50.5514** | **−3.50%** | −16.90% |
| `val_geom_camber_cruise` | **20.2827** | **−14.36%** | −26.65% |
| `val_re_rand` | **37.3708** | **−8.90%** | −17.73% |
| **val_avg** | **33.4935** | **−8.05%** | −20.91% |

| Split | test mae_surf_p |
|---|---|
| `test_single_in_dist` | **24.7056** |
| `test_geom_camber_rc` | **43.8462** |
| `test_geom_camber_cruise` | **16.8409** |
| `test_re_rand` | **29.1189** |
| **test_avg** | **28.6279** |

- **Note on test_geom_camber_cruise/loss=NaN:** bf16 vol_loss overflow on one sample's squared-loss accumulation during test eval; MAE values (FP64 accumulator) are valid and consistent with val pattern. Not blocking.
- **Mechanism:** Higher LR (1.5× from 1e-4) unlocked a deeper minimum. Best epoch moved LATER (ep70 vs ep65 for lr=1e-4) confirming the model continued exploring more terrain in mid-training before cosine cooldown. Lion's sign-step is most powerful when LR is generous enough to take large steps before schedule contracts. First beat baseline: epoch 54.
- **Config change:** `--lr 1.5e-4` vs #2524's `--lr 1e-4`. All other params unchanged (wd=3e-4, betas=(0.9, 0.99), epochs=70).
- **Metric artifacts:**
  `models/model-charliepai2g48h5-edward-lion-lr15e-5-20260513-200129/metrics.jsonl`
  `models/model-charliepai2g48h5-edward-lion-lr15e-5-20260513-200129/metrics.yaml`
- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-edward \
      --experiment_name "charliepai2g48h5-edward/lion-lr15e-5" \
      --lr 1.5e-4 \
      --weight_decay 3e-4 \
      --epochs 70
  ```

---

## 2026-05-14 05:50 — PR #2524: Lion optimizer (lr=1e-4, wd=3e-4, betas=(0.9, 0.99)): uniform WIN −14.05%

- **Student:** charliepai2g48h5-edward
- **Best epoch:** 65 of 67 (best≠terminal; still improving monotonically in cosine tail at timeout)
- **Epochs reached:** 67 (30-min wall-clock cutoff at ~25-30 s/epoch)
- **Peak GPU memory:** 14.02 GB (≈AdamW; no v_hat state, but activation memory dominates)
- **Param count:** 328,235 (unchanged vs AdamW; pure optimizer swap)

| Split | val mae_surf_p | Δ vs #2307 (42.3455 baseline) |
|---|---|---|
| `val_single_in_dist` | **28.5065** | **−19.65%** |
| `val_geom_camber_rc` | **52.3873** | **−13.88%** |
| `val_geom_camber_cruise` | **23.6834** | **−14.35%** |
| `val_re_rand` | **41.0204** | **−9.69%** |
| **val_avg** | **36.3994** | **−14.05%** |

| Split | test mae_surf_p | Δ vs #2307 |
|---|---|---|
| `test_single_in_dist` | **27.2726** | −24.43% |
| `test_geom_camber_rc` | **46.1996** | −19.43% |
| `test_geom_camber_cruise` | **19.1050** | −13.44% |
| `test_re_rand` | **32.3027** | −16.10% |
| **test_avg** | **31.2200** | **−18.92%** |

- **CRITICAL diagnostic:** Lion momentum non-zero fraction = 0.9986 → fully populated; no stuck/zero-update pathology.
- **Mechanism:** L1 loss already provides sign-based gradients. AdamW's 2nd-moment normalization is approximately redundant for this sign-gradient regime. Lion's `sign(β₁m + (1-β₁)g)` commits to sign-direction signal directly. Additionally, Lion has no `v_hat` to track in bf16 precision, whereas AdamW's 2nd-moment EMA can accumulate numerical errors in bf16.
- **Test improves MORE than val** (−18.92% vs −14.05%) → consistent with Lion's uniform step magnitudes providing slight regularization effect; warmup-cosine provides the magnitude tapering Lion lacks.
- **val_geom_camber_rc: 60.83 → 52.39 (−13.88%)** — LARGEST single-PR movement on rc OOD bottleneck since launch.
- **Config change:** Replace `torch.optim.AdamW(lr=5e-4, wd=1e-4)` with `Lion(lr=1e-4, wd=3e-4, betas=(0.9, 0.99))` in `train.py`. All other hyperparameters unchanged.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-edward-lion-lr1e-4-20260513-191121/metrics.jsonl`
  `models/model-charliepai2g48h5-edward-lion-lr1e-4-20260513-191121/metrics.yaml`
- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-edward \
      --experiment_name "charliepai2g48h5-edward/lion-lr1e-4" \
      --lr 1e-4 \
      --weight_decay 3e-4 \
      --epochs 70
  ```
  (Lion optimizer now on advisor branch; stacks with all prior wins: L1 + compile + bf16 + slice_num=24 + warmup-3-cosine + n_head=2 + LayerScale + n_layers=4 + n_hidden=96 + mlp_ratio=2)

---

## 2026-05-13 20:30 — PR #2307: slice_num 32→24 (PhysicsAttention granularity-down): routing-quality WIN −9.61%

- **Student:** charliepai2g48h5-askeladd
- **Best epoch:** 57 of 58 (best≠terminal: ep57=42.35, ep58=45.07; 1-epoch bounce; cosine T_max=67 slightly long)
- **Epochs reached:** 58 (~30.80 s/epoch; **only −2% vs #2268's ~31.4 s/epoch** — NOT the predicted −8 to −12%)
- **Peak GPU memory:** 18.30 GB (slightly above #2268 16.55 GB — compile-cache variance; well within 96 GB)
- **Param count:** 576,875 (vs #2268 577,931 → **−0.18%; slice_num affects intermediate tensors only, weight matrices unchanged**)

| Split | val mae_surf_p | Δ vs #2268 |
|---|---|---|
| `val_single_in_dist` | **35.4776** | **−14.93%** |
| `val_geom_camber_rc` | **60.8311** | **−5.94%** |
| `val_geom_camber_cruise` | **27.6517** | **−12.43%** |
| `val_re_rand` | **45.4214** | **−8.11%** |
| **val_avg** | **42.3455** | **−9.61%** |

| Split | test mae_surf_p | Δ vs #2268 |
|---|---|---|
| `test_single_in_dist` | **36.0730** | **−6.68%** |
| `test_geom_camber_rc` | **57.3635** | **−1.74%** |
| `test_geom_camber_cruise` | **22.0773** | **−14.19%** |
| `test_re_rand` | **38.5100** | **−4.90%** |
| **test_avg** | **38.5059** | **−5.66%** |

- **Config change:** `slice_num: 32 → 24` in model_config. `--epochs 70` to give headroom.
- **NOTE:** PR #2307 branched off #2268 (n_hidden=128) before #2290 merged. slice_num=24 was squash-merged onto current advisor which has n_hidden=96. Advisor now has BOTH n_hidden=96 + slice_num=24.
- **CRITICAL FINDING — mechanism is NOT budget-bound, it is ROUTING QUALITY:** Per-epoch cost barely changed (−2% only). The −9.61% gain comes from **fewer slices = better geometric routing**. slice_num=32 was ABOVE the routing optimum; reducing to 24 sharpened partitioning without removing required capacity. This is a regularization / inductive-bias effect. Prediction of budget-gain mechanism was WRONG; routing-quality mechanism was FOUND. **This is the largest single-PR gain since round-1 warmup merge.**
- **val_geom_camber_rc (the historic OOD bottleneck that barely moved since round-1) improved −5.94%** — first significant movement on this split since LayerScale in PR #2195 (−2.22%). slice_num reduction directly addresses the geometric routing quality that this split requires.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-askeladd-slice-num-24-20260513-132247/metrics.jsonl`
  `models/model-charliepai2g48h5-askeladd-slice-num-24-20260513-132247/metrics.yaml`
- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-askeladd \
      --experiment_name "charliepai2g48h5-askeladd/slice-num-24" \
      --epochs 70
  ```
  (slice_num=24 now on advisor branch; stacks with LayerScale + n_layers=4 + n_hidden=96)

---

## 2026-05-13 20:00 — PR #2290: n_hidden 128→96 (width-down, --epochs 90): budget-bound 2-for-2

- **Student:** charliepai2g48h5-frieren
- **Best epoch:** 67 of 70 (best≠terminal; cosine schedule converged cleanly within 30-min cap)
- **Epochs reached:** 70 (~25.4 s/epoch; ~15% savings vs n_layers=4 baseline ~30 s/epoch)
- **Peak GPU memory:** 14.27 GB (-13% vs n_layers=4 ~16.4 GB)
- **Param count:** 329,803 (~330K; -43% vs n_layers=4 ~578K)

| Split | val mae_surf_p | Δ vs #2268 (n_layers=4) |
|---|---|---|
| `val_single_in_dist` | **40.7757** | **-2.22%** |
| `val_geom_camber_rc` | **64.2563** | **-0.64%** |
| `val_geom_camber_cruise` | **31.5251** | **-0.16%** |
| `val_re_rand` | **48.8877** | **-1.10%** |
| **val_avg** | **46.3612** | **-1.04%** |

| Split | test mae_surf_p | Δ vs #2268 |
|---|---|---|
| `test_single_in_dist` | **38.2124** | — |
| `test_geom_camber_rc` | **56.6784** | — |
| `test_geom_camber_cruise` | **25.2465** | — |
| `test_re_rand` | **41.2846** | — |
| **test_avg** | **40.3555** | **-1.12%** |

- **Config change:** `n_hidden: 128 → 96` in model_config. `--epochs 90` to give headroom; cosine completed at ep67.
- **Budget-bound 2-for-2:** width-down WIN follows depth-down WIN (#2268). All 4 val splits improve uniformly — classic budget-bound signature. Per-epoch savings ~15% (not predicted 40-45%: slice ops scale ~linearly in n_hidden, not as n_hidden²). 17% more cosine epochs (60→70) drove improvement.
- **LayerScale γ shape:** adapted automatically from (128,) to (96,). Convergence preserved: MLP γ ~0.063–0.078 vs attn γ ~0.004–0.012 (6–10× ratio, consistent with PR #2195 pattern).
- **Budget-bound signal:** best epoch 67 ≠ terminal 70 — cosine converged cleanly; no plateau overshoot. Width axis open further (n_hidden=64 is the natural next probe).
- **Metric artifacts:**
  `models/model-charliepai2g48h5-frieren-n-hidden-96-20260513-130602/metrics.jsonl`
  `models/model-charliepai2g48h5-frieren-n-hidden-96-20260513-130602/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-frieren \
      --experiment_name "charliepai2g48h5-frieren/n-hidden-96" \
      --epochs 90
  ```
  (n_hidden=96 now on advisor branch; stacks with LayerScale + n_layers=4)

---

## 2026-05-13 17:30 — PR #2268: n_layers 5→4 (depth-down, --epochs 60): budget-bound regime confirmed

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 58 of 58 — terminal (run timed out at 30.4 min / ep58; val descending at cutoff)
- **Epochs reached:** 58 (~31.4 s/epoch, -20% vs baseline — one fewer TransolverBlock)
- **Peak GPU memory:** 16.55 GB (lower than baseline)
- **Param count:** 577,931 (baseline 657,079 → -79,148 = -12% relative to pre-LayerScale 708K; **-18.4% vs assigned baseline**)

| Split | val mae_surf_p | Δ vs #2195 (LayerScale) |
|---|---|---|
| `val_single_in_dist` | **41.7031** | **-6.51%** |
| `val_geom_camber_rc` | **64.6729** | **-1.93%** |
| `val_geom_camber_cruise` | **31.5759** | **-4.99%** |
| `val_re_rand` | **49.4322** | **-1.67%** |
| **val_avg** | **46.8460** | **-3.44%** |

| Split | test mae_surf_p | Δ vs #2195 |
|---|---|---|
| `test_single_in_dist` | **38.6569** | **-3.73%** |
| `test_geom_camber_rc` | **58.3783** | **-3.45%** |
| `test_geom_camber_cruise` | **25.7282** | **-6.59%** |
| `test_re_rand` | **40.4925** | **-6.09%** |
| **test_avg** | **40.8140** | **-4.70%** |

- **Config change:** `n_layers: 5 → 4` in model_config (one fewer TransolverBlock). `--epochs 50 → 60` to use the ~20% wall-clock savings.
- **IMPORTANT — this PR did NOT include LayerScale** (branched before #2195 merge). However, squash-merge onto current advisor (which has LayerScale) succeeds cleanly — the n_layers=4 diff is orthogonal to LayerScale γ params. Advisor now has both.
- **Mechanism — BUDGET-BOUND CONFIRMED:** The ~20% per-epoch wall-clock savings bought 8-10 extra epochs of cosine refinement. All 4 val splits improved, consistent with "budget-gain dominates capacity-loss." Depth=5 was carrying compute waste, not load-bearing geometric capacity for this hidden_dim=128 regime.
- **Budget-bound diagnostic confirmed:** Best epoch = terminal; cosine LR still decaying at ep58 (LR ~3e-6). Model never plateaued within 30-min budget. Further depth reduction (n_layers=3) or wider schedule (--epochs 75) could yield additional gains.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-n-layers-4-20260513-122122/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-n-layers-4-20260513-122122/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/n-layers-4" \
      --epochs 60
  ```
  (n_layers=4 now on advisor branch; stacks with LayerScale γ=1e-4 from PR #2195)

---

## 2026-05-13 16:30 — PR #2195: LayerScale (CaiT-style learnable residual gain γ, init=1e-4)

- **Student:** charliepai2g48h5-askeladd
- **Best epoch:** 42 of 43 — **best≠terminal** (ep42=48.52, ep43=49.41; first convergence-before-timeout in several rounds)
- **Epochs reached:** 43 (~42.25 s/epoch; same as baseline — LayerScale adds only per-channel multiply)
- **Peak GPU memory:** 23.85 GB (unchanged)
- **Param count:** 658,359 (baseline 657,079 → +1,280 = +0.19% — 5 blocks × 2 branches × 128 dims)

| Split | val mae_surf_p | Δ vs #2173 |
|---|---|---|
| `val_single_in_dist` | **44.6149** | **-3.62%** |
| `val_geom_camber_rc` | **65.9411** | **-2.22% (FIRST MOVE since round-1!)** |
| `val_geom_camber_cruise` | 33.2325 | +1.95% (slight) |
| `val_re_rand` | **50.2756** | **-4.95%** |
| **val_avg** | **48.5160** | **-2.59%** |

| Split | test mae_surf_p | Δ vs #2173 |
|---|---|---|
| `test_single_in_dist` | **40.1418** | **-1.27%** |
| `test_geom_camber_rc` | **60.4713** | **-1.67%** |
| `test_geom_camber_cruise` | **27.5452** | **-0.38%** |
| `test_re_rand` | **43.1065** | **-2.82%** |
| **test_avg** | **42.8162** | **-1.66%** |

- **Config change:** added per-channel learnable `nn.Parameter` γ (init=1e-4) to each TransolverBlock's attention and MLP residual branches. Multiplies each branch output before residual add: `x = x + gamma * branch(x)`.
- **Mechanism:** CaiT-style (Touvron et al. 2021) residual scaling. Init=1e-4 starts each branch near-zero (residual stream = identity), then learned γ grows per-channel to selectively amplify useful features. Forces the model to learn "how much to trust each branch at each channel" vs fixed uniform residual integration.
- **Trained γ diagnostics:** MLP branches activated 4-8× stronger than attention branches (γ_mlp abs_mean ≈ 0.025-0.05 vs γ_attn ≈ 0.003-0.011). Block 3 attention notably underweighted. Signs mixed within each γ vector (selective per-channel gating). Non-trivial structure.
- **Failure-mode signature check:** ALL splits improve (not bimodal-averaging), val_re_rand NOT worst (not broadcast-scalar corruption), pattern is uniform-direction architectural WIN — 8th distinct failure-mode taxon NOT triggered.
- **MAJOR DIAGNOSTIC:** `val_geom_camber_rc` (67.44 → 65.94) moved for the FIRST TIME since round-1. This is the hardest OOD split that has been flat across every other intervention (warmup, n_head, normalization, augmentation). LayerScale is the first mechanism to crack it.
- **Best epoch 42 ≠ terminal 43** — first run in recent rounds where the model converged within the 30-min budget (ep42=48.52, ep43=49.41). LayerScale may be enabling faster convergence via the residual-scaling degree of freedom.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-askeladd-layerscale-1e-4-20260513-110555/metrics.jsonl`
  `models/model-charliepai2g48h5-askeladd-layerscale-1e-4-20260513-110555/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-askeladd \
      --experiment_name "charliepai2g48h5-askeladd/layerscale-1e-4" \
      --epochs 50
  ```
  (LayerScale γ=1e-4 now on advisor branch — see PR #2195 diff)

---

## 2026-05-13 14:30 — PR #2173: n_head 4→2 (dim_head 32→64): architectural head-rank probe

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 47 of 47 — terminal (still improving when 30-min timeout hit; ep45→47 val: 50.91→49.97→49.81)
- **Epochs reached:** 47 (~37.5 s/epoch, -9% vs baseline — fewer head projections)
- **Peak GPU memory:** 20.05 GB (-6.1% vs baseline)
- **Param count:** 708,269 (unchanged — head projections are hidden_dim × hidden_dim regardless)

| Split | val mae_surf_p | Δ vs #2033 |
|---|---|---|
| `val_single_in_dist` | **46.2915** | **-3.44%** |
| `val_geom_camber_rc` | 67.4416 | +0.11% (wash) |
| `val_geom_camber_cruise` | **32.5963** | **-5.09%** |
| `val_re_rand` | 52.8918 | +0.27% (wash) |
| **val_avg** | **49.8053** | **-1.57%** |

| Split | test mae_surf_p | Δ vs #2033 |
|---|---|---|
| `test_single_in_dist` | **40.6576** | **-3.64%** |
| `test_geom_camber_rc` | **61.4956** | -0.17% |
| `test_geom_camber_cruise` | **27.6519** | **-2.03%** |
| `test_re_rand` | 44.3531 | +1.14% (mild) |
| **test_avg** | **43.5396** | **-0.97%** |

- **Config change:** `n_head: 4 → 2` in model_config (dim_head: 32 → 64). Same hidden_dim=128, same param count.
- **Mechanism:** with n_head=4 / dim_head=32, each attention head's query/key subspace is rank-32 — too narrow to encode geometric/physical relations. n_head=2 / dim_head=64 doubles each head's rank for the same FLOPs/params. Matches literature optimum (dim_head≈64, Vaswani 2017, LLaMA, T5).
- **Per-split pattern:** improvements concentrated in val_single_in_dist (-3.4%) and val_geom_camber_cruise (-5.1%) — in-dist and "easy" OOD benefit most. Harder OOD splits (val_geom_camber_rc +0.1%, val_re_rand +0.3%) are washed — data/regularization-limited, not head-rank-limited.
- **Best=terminal again** — training still descending at 30-min timeout. Both #2033 and now #2173 are budget-limited; true ceiling is below 49.8.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-n-head-2-20260513-101936/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-n-head-2-20260513-101936/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/n-head-2" \
      --epochs 50
  ```
  (n_head=2 now on advisor branch)

---

## 2026-05-13 13:00 — PR #2033: Linear warmup 3ep + monotone cosine (T_max=47)

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 44 of 44 — terminal (still improving when 30-min timeout hit)
- **Epochs reached:** 44 (~41.0 s/epoch, -6% vs slice=32 baseline — warmup startup slightly more stable)
- **Peak GPU memory:** 21.35 GB (unchanged)

| Split | val mae_surf_p | Δ vs #1846 |
|---|---|---|
| `val_single_in_dist` | **47.9418** | **-18.87%** |
| `val_geom_camber_rc` | 67.3675 | -0.11% |
| `val_geom_camber_cruise` | **34.3430** | -3.85% |
| `val_re_rand` | **52.7481** | -1.89% |
| **val_avg** | **50.6001** | **-6.31%** |

| Split | test mae_surf_p | Δ vs #1846 |
|---|---|---|
| `test_single_in_dist` | **42.1940** | — |
| `test_geom_camber_rc` | **61.5999** | — |
| `test_geom_camber_cruise` | **28.2251** | — |
| `test_re_rand` | **43.8531** | — |
| **test_avg** | **43.9680** | **-7.68%** |

- **LR schedule:** linear warmup ep1-3 (5e-5 → 5e-4) then CosineAnnealingLR(T_max=47, eta_min=0). Warmup confirmed by logged LR: ep1=5e-5, peak at ep4, monotone cosine descent to ~2e-5 by ep44.
- **Mechanism:** warmup gives optimizer 2-3 sub-peak-LR epochs to select a better loss basin before cosine descent locks in. Largest gain is val_single_in_dist (-18.9%) where basin quality is most sensitive; OOD splits move less but don't regress.
- **Late-stage settling preserved:** unlike SGDR (#1989 LOSS), the monotone cosine tail lets L1 sign-gradient regime fine-tune in. val improved all the way to ep44 — gap ep41→44 still -4.9%.
- **Why it works:** L1's sign-gradient property benefits from a two-phase schedule: warmup ("find the basin"), cosine tail ("fine-tune within it"). Complementary design.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-warmup-3-cosine-20260513-072010/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-warmup-3-cosine-20260513-072010/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/warmup-3-cosine" \
      --epochs 50
  ```
  (warmup-3-cosine schedule now on advisor branch)

---

## 2026-05-13 05:45 — PR #1846: slice_num 64 → 32 (tighter attention bottleneck)

- **Student:** charliepai2g48h5-frieren
- **Best epoch:** 40 of 41 — **first run where best ≠ terminal** (model converged within budget!)
- **Epochs reached:** 41 (~43.5 s/epoch, -12.3% vs baseline — slice_num=32 is faster)
- **Peak GPU memory:** 21.35 GB (-10.4% vs baseline)
- **Param count:** 657,079 (~5K fewer than baseline 662K — <1% change)

| Split | val mae_surf_p | Δ vs #1700 L1 baseline |
|---|---|---|
| `val_single_in_dist` | **59.0943** | -8.93% |
| `val_geom_camber_rc` | 67.4450 | -8.91% |
| `val_geom_camber_cruise` | **35.7197** | **-10.63%** |
| `val_re_rand` | **53.7616** | -9.25% |
| **val_avg** | **54.0051** | **-9.30%** |

| Split | test mae_surf_p | Δ vs #1700 |
|---|---|---|
| `test_single_in_dist` | **53.2538** | -4.27% |
| `test_geom_camber_rc` | **62.8744** | -5.86% |
| `test_geom_camber_cruise` | **29.4777** | -12.22% |
| `test_re_rand` | **44.8988** | -9.97% |
| **test_avg** | **47.6261** | **-7.46%** |

- **All 4 val splits and all 4 test splits improved uniformly (~9%)** — this is a global inductive-bias benefit, not one-split coincidence.
- **Why it works:** slice_num=64 was over-allocated for TandemFoilSet's natural spatial regimes (~10-20 CFD motifs). slice_num=32 forces tighter routing → regularization at the information-bottleneck level + ~4 extra epochs from budget gain.
- **Key diagnostic:** first run in round 5 where best_epoch ≠ terminal (ep 40 < ep 41) — the model now converges within the 30-min cap. Smaller bottleneck = faster optimization.
- **⚠️ Caveat:** Measured on L1-only base (#1700, 59.54). Post-merge advisor includes sampler 2× single (#1619). True stacked baseline revealed by future runs.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-frieren-slice-num-32-20260513-030801/metrics.jsonl`
  `models/model-charliepai2g48h5-frieren-slice-num-32-20260513-030801/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-frieren \
      --experiment_name "charliepai2g48h5-frieren/slice-num-32" \
      --epochs 50
  ```
  (slice_num=32 now on advisor branch)

---

## 2026-05-13 05:10 — PR #1619: Sampler 2× single boost on L1 baseline

- **Student:** charliepai2g48h5-nezuko
- **Best epoch:** 39 (wall-clock-bound at 30 min; best == terminal; trajectory still descending: ep 38→39: 57.74→56.62)
- **Epochs reached:** 39 (~46.3 s/epoch, unchanged vs L1 baseline)
- **Peak GPU memory:** 23.83 GB (unchanged)

| Split | val mae_surf_p | Δ vs #1700 |
|---|---|---|
| `val_single_in_dist` | **56.1237** | **-13.51%** |
| `val_geom_camber_rc` | 71.0701 | -4.02% |
| `val_geom_camber_cruise` | 41.6906 | +4.31% |
| `val_re_rand` | **57.6024** | -2.76% |
| **val_avg** | **56.6217** | **-4.89%** |

| Split | test mae_surf_p | Δ vs #1700 |
|---|---|---|
| `test_single_in_dist` | **50.0812** | **-9.97%** |
| `test_geom_camber_rc` | 66.9241 | +0.20% |
| `test_geom_camber_cruise` | 34.1808 | +1.78% |
| `test_re_rand` | 50.5377 | +1.34% |
| **test_avg** | **50.4310** | **-2.01%** |

- **Sampler intervention:** `racecar_single` boost factor 2× → 50% / 25% / 25% share (single / tandem / cruise).
- **Lever validated across 3 baselines:** β=1.0 (-2.80%), β=1.0+compile (-2.25%), **L1+compile (-4.89%)**. Win grows as loss gets sharper.
- **Three of four val splits improve;** only `val_geom_camber_cruise` regresses (+4.31%) — mechanistically expected (cruise loses 25% of training mass to the 2× boost).
- **Metric artifacts:**
  `models/model-charliepai2g48h5-nezuko-sampler-2x-on-l1-20260513-021352/metrics.jsonl`
  `models/model-charliepai2g48h5-nezuko-sampler-2x-on-l1-20260513-021352/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-nezuko \
      --experiment_name "charliepai2g48h5-nezuko/sampler-2x-on-l1" \
      --epochs 50
  ```
  (sampler-reweight block now on advisor branch — see PR #1619 diff)

---

## 2026-05-13 02:10 — PR #1700: Pure L1 loss (β sweep → β=0 limit wins)

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 37 (wall-clock-bound at 30 min; best == terminal; still descending)
- **Epochs reached:** 37 (~49.64 s/epoch, unchanged vs #1633)
- **Peak GPU memory:** 23.83 GB (unchanged)

| Split | val mae_surf_p | Δ vs #1633 |
|---|---|---|
| `val_single_in_dist` | 64.8899 | -10.6% |
| `val_geom_camber_rc` | 74.0437 | -5.5% |
| `val_geom_camber_cruise` | **39.9687** | **-7.9%** |
| `val_re_rand` | 59.2391 | -4.5% |
| **val_avg** | **59.5354** | **-7.08%** |

| Split | test mae_surf_p |
|---|---|
| `test_single_in_dist` | 55.6271 |
| `test_geom_camber_rc` | 66.7873 |
| `test_geom_camber_cruise` | **33.5816** |
| `test_re_rand` | 49.8704 |
| **test_avg** | **51.4666** |

- **β sweep summary:** β=2.0→1.0→0.5→0.25→0 monotone improvement: 77.81→69.83→64.07→60.76→**59.54**. Diminishing returns with each halving (+8.2% → +5.2% → +2.0%), but L1 is the best point on the curve.
- **Key code change:** `F.smooth_l1_loss(pred, y_norm, beta=0.5, reduction='none')` → `F.l1_loss(pred, y_norm, reduction='none')` at both call sites (lines 246, 485).
- **Arm A (β=0.25):** val_avg=60.7558, test_avg=52.3312.
- **Critical diagnostic:** both arms best_epoch == terminal — undertrained, not overfit.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-l1-loss-20260513-005443/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-l1-loss-20260513-005443/metrics.yaml`
  `models/model-charliepai2g48h5-thorfinn-huber-beta-0.25-20260513-000538/metrics.jsonl`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/l1-loss" \
      --epochs 50
  ```
  (L1 loss now on advisor branch — both `smooth_l1_loss` call sites replaced with `F.l1_loss`)

---

## 2026-05-13 00:50 — PR #1633: Huber β=0.5 (sharper loss function)

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 37 (wall-clock-bound at 30 min; model still descending at timeout)
- **Epochs reached:** 37 (~49.5 s/epoch, same as compile baseline — sharper β adds no compute)
- **Peak GPU memory:** 23.83 GB (unchanged)

| Split | val mae_surf_p | Δ vs #1568 baseline |
|---|---|---|
| `val_single_in_dist` | 72.5692 | -5.9% |
| `val_geom_camber_rc` | 78.3209 | -6.2% |
| `val_geom_camber_cruise` | **43.3744** | **-14.4%** |
| `val_re_rand` | 62.0174 | -8.9% |
| **val_avg** | **64.0705** | **-8.2%** |

| Split | test mae_surf_p |
|---|---|
| `test_single_in_dist` | 63.0824 |
| `test_geom_camber_rc` | 69.4136 |
| `test_geom_camber_cruise` | **36.1544** |
| `test_re_rand` | 53.3341 |
| **test_avg** | **55.4961** |

- **Key code change:** `F.smooth_l1_loss(..., beta=0.5)` (was β=1.0). Sharper β makes the loss linear for a wider range of medium-magnitude residuals, down-weighting outlier gradients — directly suited to TandemFoil's heavy-tailed surface pressure residual distribution.
- **Monotone signal:** β=2.0 (val=77.81, +11.4%), β=1.0 (val=69.83, baseline), β=0.5 (val=64.07, -8.2%). Clear direction: sweep further toward β=0.25 / L1.
- **Note:** best_epoch=37=terminal — model was still improving at the timeout. Sweeping β=0.25 is the next logical step.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-huber-beta-0.5-20260512-221022/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-huber-beta-0.5-20260512-221022/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/huber-beta-0.5" \
      --epochs 50
  ```
  (change both `smooth_l1_loss` call sites in `train.py` to `beta=0.5` — see PR #1633 diff)

---

## 2026-05-12 22:10 — PR #1568: torch.compile + bf16 AMP for additional throughput

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 36 (wall-clock-bound at 30 min; model still descending at timeout)
- **Epochs reached:** 36 (~2.0× faster than bf16 baseline: ~49.5 s/epoch vs ~98 s)
- **Peak GPU memory:** 23.8 GB

| Split | val mae_surf_p | Δ vs #1532 baseline |
|---|---|---|
| `val_single_in_dist` | 77.10 | -35.8% |
| `val_geom_camber_rc` | 83.49 | -22.0% |
| `val_geom_camber_cruise` | 50.64 | -38.9% |
| `val_re_rand` | 68.10 | -28.0% |
| **val_avg** | **69.8316** | **-30.9%** |

| Split | test mae_surf_p |
|---|---|
| `test_single_in_dist` | 67.81 |
| `test_geom_camber_rc` | 77.68 |
| `test_geom_camber_cruise` | 41.98 |
| `test_re_rand` | 59.99 |
| **test_avg** | **61.8652** |

- **Key code change:** `torch.compile(model, dynamic=True)` applied after model construction; `dynamic=True` prevents recompilation on variable mesh batch sizes. No recompilation stalls observed across 36 epochs.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-torch-compile-bf16-20260512-205152/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-torch-compile-bf16-20260512-205152/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/torch-compile-bf16" \
      --epochs 50
  ```
  (`torch.compile(model, dynamic=True)` now on advisor branch — see PR #1568 diff)

---

## 2026-05-12 20:01 — PR #1532: bf16 AMP for 2x epoch throughput + scoring-NaN fix

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 17 (wall-clock-bound at 30 min; model still improving at epoch 19)
- **Epochs reached:** 19 (~25% faster than fp32: ~98 s/epoch vs ~131 s)
- **Peak GPU memory:** 32.95 GB (well under 96 GB limit)

| Split | val mae_surf_p | Δ vs #1444 |
|---|---|---|
| `val_single_in_dist` | 120.0176 | -15.14 |
| `val_geom_camber_rc` | 107.0980 | -21.98 |
| `val_geom_camber_cruise` | 82.8425 | +5.14 |
| `val_re_rand` | 94.5268 | -6.57 |
| **val_avg** | **101.1212** | **-9.64** |

| Split | test mae_surf_p |
|---|---|
| `test_single_in_dist` | 105.4434 |
| `test_geom_camber_rc` | 99.9931 |
| `test_geom_camber_cruise` | 69.2841 |
| `test_re_rand` | 91.2844 |
| **test_avg** | **91.5013** |

- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-bf16-amp-scoring-fix-20260512-192502/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-bf16-amp-scoring-fix-20260512-192502/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/bf16-amp-scoring-fix" \
      --epochs 50
  ```
  (bf16 AMP via `torch.autocast` + scoring workaround — see PR #1532 diff)

---

## 2026-05-12 — PR #1444: Swap MSE → Smooth-L1 (Huber, beta=1.0)

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 14 (wall-clock-bound at 30 min; model still improving)
- **Peak GPU memory:** 42.1 GB
- **Time per epoch:** ~131 s

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 135.16 | 1.719 | 0.769 |
| `val_geom_camber_rc` | 129.08 | 2.104 | 0.988 |
| `val_geom_camber_cruise` | 77.70 | 1.047 | 0.555 |
| `val_re_rand` | 101.10 | 1.607 | 0.740 |
| **val_avg** | **110.76** | — | — |

- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-smooth-l1-loss-20260512-180133/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-smooth-l1-loss-20260512-180133/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/smooth-l1-loss" \
      --epochs 50
  ```
  (plus the Smooth-L1 substitution in `train.py` — see PR #1444 diff)

## Reproduce command (reference defaults)

```bash
cd target && python train.py \
    --agent <student> \
    --experiment_name "<student>/<short-description>" \
    --epochs 50
```

Commit `models/<experiment>/metrics.jsonl` and `metrics.yaml` with the PR and
quote the key values in the PR results comment plus the
`SENPAI-RESULT` terminal marker.
