<!-- Advisor-owned research log for the TandemFoilSet surrogate track.
     Maintained on the advisor branch only. Newest baseline at the top. -->

# Research log — TandemFoilSet surrogate (advisor branch)

Primary metric: **`test_avg/mae_surf_p`** (equal-weight mean surface-pressure MAE
over the 4 test splits; lower is better). Iteration/checkpoint metric:
**`val_avg/mae_surf_p`**.

> ⚠️ Track-wide scoring bug: cruise **test** GT sample `000020.pt` has non-finite
> `p`, and read-only `data/scoring.py` does `err*mask` → `nan*0=nan`, so W&B
> `test_avg/mae_surf_p` is **NaN for every run** (val + Ux/Uy clean). Rank on
> clean `val_avg/mae_surf_p`; recover a test number by re-evaluating the best
> checkpoint on the cruise test split with a `torch.where`-based skip of the
> non-finite sample. **Do not patch `data/`.**

## Current baseline

| item | value |
|------|-------|
| model | Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, ~0.66M params) |
| optimizer | AdamW lr=5e-4, wd=1e-4, bs=4, CosineAnnealingLR(T_max=epochs) |
| **loss** | **L1 (surf_weight=10, masked per-node mean, normalized target space)** ← R2 winner |
| `val_avg/mae_surf_p` | **≈ 112** (111.98, E=9 screening) |
| corrected `test_avg/mae_surf_p` | **≈ 102** (102.01) |
| branch SHA | `75eb555` (after #4604) |

Reproduce: `python train.py --agent <name> --loss l1 --epochs 9`

## History (newest first)

### R2 — Loss function: L1 vs Huber vs MSE — ✅ MERGED (#4604, tf6h-frieren)
- **Change:** added `--loss {mse,l1,huber}` + `--delta` (SmoothL1 β) to `train.py`;
  applied elementwise to both vol+surf terms; set `Config.loss` default = `l1`.
  `train.py`-only; `data/` untouched; MAE metric accumulation unchanged.
- **Hypothesis:** ranked metric is MAE(L1) but training minimized MSE(L2); high-Re
  outlier magnitudes (±29k) dominate L2 gradients. L1/Huber should match the metric.
- **Result (E=9 screening, ~132 s/epoch @4-way, peak 42.2 GB):**
  `val_avg/mae_surf_p` — mse 141.50 · **l1 111.98** · huber δ1.0 114.65 · huber δ0.3 113.92.
  corrected `test_avg/mae_surf_p` — mse 130.28 · **l1 102.01** · huber δ1.0 103.66 · huber δ0.3 104.03.
- **Robustness:** L1 lowest on **all 4 val splits AND all 4 test splits**; monotone,
  no instability. 3 independent L1-family seeds all ~20% below MSE → not seed noise.
- **vs recorded reference baseline** (MSE, val 133.71 / test 119.81): L1 = **−16% val, −15% test**.
- **Seed caveat:** this run's MSE self-check landed val 141.50 vs reference 133.71
  (~6% seed spread; `train.py` sets no seed). L1 gain ≫ that spread.
- W&B: mse `bakex6hg` · **l1 `3rjbrod5`** · huber-d1.0 `fo1jn4yz` · huber-d0.3 `jcwij9z5`
  (`wandb-applied-ai-team/senpai-v1`).

### R2 — slice_num (physics-token count) {32,64,96,128} — ⏸️ closed, promoted (#4605, tf6h-fern)
- **Finding (E=8, MSE base, equal-epoch):** clean **monotone** per-epoch win — 128 (139.17)
  beats 64 (149.11) by **6.66%** robustly on all 4 val splits; 96 intermediate; 32 worse.
  Cost/VRAM ~linear: sec/epoch 112/131/151/172; peak GB 37.2/42.2/47.6/54.6 (64→128 = +31% time, +29% VRAM).
- **Not merged (2 reasons):** (a) ran on the *superseded MSE* baseline; (b) compared at **equal
  epochs**, but our limit is a 30-min **wall-clock** cap — 128's +31%/epoch means ~31% fewer
  epochs under the cap. Epochs are valuable (64: ~149@E8→~133.5@E10), and large-slice models
  overtake only late (128 trailed 64 at ep5-6), so the equal-epoch win may not survive equal wall-clock.
- **⚠️ Programme learning:** for the deployment decision under a fixed wall-clock cap, compare
  slice_num / any compute-changing knob at **equal wall-clock** (each config at its own max E), not equal epochs.
- **Promoted to R3 #4607** (fern): equal-wall-clock slice_num {64,96,128,160} under **L1** baseline.
- W&B: sl32 `poqnbxcs` · sl64 `kpafdlke` · sl96 `67ymc395` · sl128 `154my62p`.

### R1 — Learning rate {3e-4,5e-4,1e-3,2e-3} — ❌ closed negative (#4602, tf6h-fern)
Flat/split-inconsistent optimum; best (1e-3=131.50) only ~1.5% under baseline. Kept lr=5e-4.
Model is **not optimizer-limited**.

### R1 — Surface loss weight {5,10,20,40} — ❌ closed negative (#4603, tf6h-frieren)
`surf_weight=10` optimal robustly. Kept. (This student first found the scoring bug.)

## Notes / open threads
- **Screening noise floor ≈ 1.5%** (single-seed, E=9-10) → marginal gains need multi-seed confirm.
- Compute-bound: ~10 epochs max under the 30-min/process cap; ~6-hour cluster budget.
- **Wall-clock discipline:** every compute-changing knob (slice_num, n_layers, n_hidden, bs…)
  must be judged at **equal wall-clock** under the 30-min cap, not equal epochs.
- **In flight:**
  - #4606 (frieren) — pressure-channel loss weight `{1,2,4,8}` on L1 (metric is surface-p only).
  - #4607 (fern) — equal-wall-clock slice_num `{64,96,128,160}` on L1 (settles the R2 deployment question).
- Round-4 ideas: combine merged winners (L1 + best p_weight + best slice_num); physics-informed
  target normalization/scaling (Re / dynamic pressure; `single` split has highest error);
  longer budget (L1 still improving at E=9).
