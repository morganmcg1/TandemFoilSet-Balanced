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

### R4 — Physics-informed pressure target normalization (Re-scaling) — ❌ closed decisive negative (#4608, tf6h-frieren)
- **Change:** added `--pnorm_exp` (default 0.0). Per-sample `f=Re^(-exp)/ref` (unit-mean over train, inputs-only,
  no leakage) rescales the **p-channel normalized target** for the L1 loss; inverts `pred/f` before scoring so
  the tensor handed to scoring stays global-normalized. Ux/Uy untouched. `train.py`-only. exp=0 = exact identity.
- **Physics fit (read-only):** regressed `log(p_surf_rms)` on `log(Re)` over 240 balanced-train samples →
  **k=1.99** (R²=0.89; per-domain 1.95–2.07) → confirms `p ∝ Re²`. Swept exps {0,1,2,3} bracketing k.
- **Result (E=9, ~131.5 s/ep, 42.2 GB):** `val_avg/mae_surf_p` — exp0 **115.61** · exp1 120.82 · exp2 173.75 ·
  exp3 474.94. **Monotone degradation on ALL 4 val splits** (incl. the OOD splits the hypothesis predicted
  would improve: re_rand 102→106→152→431; single 156→162→225→533). Decisive falsification.
- **Plumbing verified:** offline unit tests (identity no-op, round-trip, unit-mean) pass; identity arm reproduces
  baseline within seed variance (val 115.61; corrected test 104.30) → arms B–D trustworthy; nan-safe corrected-test
  matches W&B exactly on the 3 clean splits.
- **Decision:** keep `pnorm_exp=0.0` (no merge). **Banked lesson:** the *absolute global-normalized* pressure
  target is the correct representation — down-weighting high-|p|/high-Re amplitude removes signal the equal-weighted
  metric rewards. With R3 p_weight, this **rules out the relative/normalized-loss reparametrization class.**
- W&B: **exp0 `xc9whgtt`** · exp1 `eljfwpfb` · exp2 `oj9st882` · exp3 `xjl40hrd`.

### R3 — slice_num at EQUAL WALL-CLOCK {64,96,128,160} on L1 — ❌ closed negative, KEPT 64 (#4607, tf6h-fern)
- **Change:** re-added `--slice_num` (threaded into `model_config`), default 64. `train.py`-only.
- **Question:** under the 30-min wall-clock cap, which slice_num minimizes `val_avg/mae_surf_p` when each
  config runs its max affordable epochs? (Resolves the #4605 equal-epoch confound.)
- **Result (equal ~26-min wall-clock, L1; each at its own max E):**
  slice64 E11 → **103.08 (WINNER, best on all 4 splits)** · slice128 E9 → 108.00 · slice96 E10 → 111.37 · slice160 E8 → 116.69.
  Per-split single/rc/cruise/re_rand — slice64: 127.30/111.88/78.06/95.07 (wins every split; +4.56% avg over #128).
- **Ranking FLIP confirmed:** at a common epoch (E=8) slice128 (111.02) beats slice64 (114.13) by 2.7% — the
  per-epoch gain is real (reproduces #4605). But **all arms compute-bound (best@final epoch, none plateaued)**,
  so the config affording the most epochs wins: slice64→E11 beats slice128→E9. Equal-epoch champion **loses** at equal wall-clock.
- **Decision:** keep `slice_num=64` (no merge — winning config = incumbent). Larger slices don't earn their
  permanent per-experiment compute tax. Retroactively validates closing #4605 unmerged.
- **⚠️ Key programme insight (fern):** model is compute-bound & still descending at the cap → **the real lever at
  fixed budget is training throughput / #epochs, not capacity.** Drives R4 #4609 (precision/throughput).
- **Deployment ceiling:** slice64 at max-E-under-cap (E11) = **val≈103**, vs the E=9 screening reference ≈112
  (~8% left on the table by the screening budget; E=9 stays the fair apples-to-apples sweep budget).
- W&B: **slice64 `v9s9yyhp`** · slice96 `qh01r4se` · slice128 `37iheam0` · slice160 `t2t114gj`.

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
- **Compute-bound regime (fern R3):** the model does **not plateau** within the cap (best val @ final
  epoch for every arm) → **throughput and #epochs are first-class levers**: AMP/TF32/`torch.compile`,
  faster data paths, anything raising epochs-per-wall-clock. Capacity knobs (slice_num, and likely
  width/depth) do **not** pay at equal wall-clock — slice_num settled at 64. `train.py` is currently pure fp32.
- **Representation levers exhausted (mostly):** loss=L1 is the only representation change that helped (merged).
  p_weight (channel) and pnorm (target Re-scaling) both fail → **absolute global-normalized target + L1 is the
  right representation; relative/normalized-loss class is ruled out.** Next value is in throughput/optimization.
- **In flight (both throughput/optimization levers — the compute-bound epochs lever):**
  - #4609 (fern, R4) — training throughput/precision `--precision {fp32,tf32,bf16,bf16_compile}` at
    **equal wall-clock**. **Emerging DECISIVE WIN** (mid-run, ~885s): steps fp32 2522 / tf32 2895 / bf16 3375 /
    **bf16_compile 6599 (2.6×)**; val fp32 138.7 → **bf16_compile 93.7** and dropping (below the ~103 fp32 ceiling).
    If confirmed → **mergeable faster re-baseline**; verify fp32 arm ≈103 (no-op), bf16 numerically stable,
    >1.5% robust across 4 val splits. Review-ready ~12:35.
  - #4610 (frieren, R5) — **batch size** `{2,4,8,16}` at equal wall-clock (never swept; `--batch_size` already a flag).
    Larger bs → more GPU utilization → more epochs, vs fewer steps/undertraining. bs=4 arm = reference (~112–116).
    Win >1.5% robust across 4 val splits → re-baseline bs.
- Round-5+ ideas: combine merged winners (L1 + precision + bs); **with a throughput win, revisit LR/schedule at
  the now-affordable higher epoch counts** (R1's flat LR was at ~9 ep; likely different at ~25 ep); revisit
  p_weight=2 multi-seed if the corrected-test signal recurs. Capacity/representation knobs **deprioritized**.
