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

### R3 — Pressure-channel loss weight {1,2,4,8} on L1 — ❌ closed inconclusive (#4606, tf6h-frieren)
- **Change:** added `--p_weight` (channel weight `[1,1,p_weight]` applied to the elementwise
  L1 term **before** the vol/surf masked-mean split → pure channel axis, orthogonal to
  `surf_weight`), identical in train + `evaluate_split`; checkpoint metric unchanged. `train.py`-only.
- **Hypothesis:** metric is surface-**p** only, so up-weighting the p channel (vs Ux/Uy) is
  metric-aligned and should lower `val_avg/mae_surf_p`. Competing: Ux/Uy + volume act as
  beneficial multi-task auxiliaries.
- **Result (E=9, L1 base, ~132 s/ep, 42.2 GB):** `val_avg/mae_surf_p` — p1 111.55 ·
  **p2 109.94 (−1.44%)** · p4 112.32 · p8 122.23. Shallow minimum near p=1–2, then worse (not monotone).
- **Verdict:** pre-registered bar (>1.5% robust across all 4 val splits) **NOT met** — p2's
  −1.44% is below the ~1.5% noise floor AND `single_in_dist` regresses (+1.2%). Kept
  `p_weight=1.0` (no flag merged; baseline stays simple).
- **Confirms auxiliaries matter:** surface-Ux MAE degrades monotonically with p_weight
  (1.27→1.37→1.65→2.05 for p=1/2/4/8) → velocity/volume channels are beneficial regularizers.
- **⚠️ Low-priority thread:** on *corrected test* (the paper-facing metric) p2 beats baseline on
  **all 4** splits (avg 98.98 vs 101.27, **−2.27%**) despite the sub-threshold/non-robust val.
  Single 9-ep seed within ~6% seed noise → only revisit p=2 (2–3 seeds + longer training) if spare capacity.
- W&B: p1 `0coijmec` · **p2 `usbpz9t4`** · p4 `5vjki9p6` · p8 `o55pxzfi`.

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
  - #4607 (fern) — equal-wall-clock slice_num `{64,96,128,160}` on L1. **Preliminary (mid-run,
    ~20 min in):** slice-64 leading (val ~114, still descending) vs 96/128/160 (~129/134/131) →
    the epoch budget appears to dominate the per-epoch quality gain, so equal-wall-clock will
    likely **keep slice_num=64** — the opposite of the equal-epoch #4605 ranking, exactly the
    confound #4607 was built to expose. Awaiting final result + review.
  - #4608 (frieren, R4) — physics-informed pressure target normalization: per-sample Re-based
    amplitude reweight on the p channel via `--pnorm_exp {0,1,2,3}` (exp=0 = identity self-check,
    must reproduce ~112). Motivated by `p ∝ ½U² ∝ Re²`; win bar >1.5% robust across all 4 val
    splits, watch OOD splits (`re_rand`, `single_in_dist`).
- Round-5+ ideas: combine merged winners (L1 + any slice/pnorm winner); model capacity
  (n_hidden/n_layers) judged at **equal wall-clock**; longer budget (L1 still improving at E=9);
  revisit p_weight=2 multi-seed if the corrected-test signal recurs.
