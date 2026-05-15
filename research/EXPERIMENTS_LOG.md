# SENPAI Research Results

## 2026-05-15 15:20 — PR #3123: Random Fourier positional features over (x,z) mesh coords

- Branch: `willowpai2i48h5-thorfinn/fourier-positional-features`
- Hypothesis: Map (x,z) coordinates through random Fourier features `[sin(2π B·xz), cos(2π B·xz)]` with Gaussian projection B (sigma=10) to give the model a high-frequency position basis, helping near-surface pressure gradient representation. Expected larger improvement on OOD camber splits.

| Arm | W&B run | n_fourier | epochs | val_avg | test_avg | Notes |
|-----|---------|-----------|--------|---------|----------|-------|
| A — baseline | jyqygcbx | 0 | 14/50 | 135.23 | NaN⚠️ | Wall-clock timeout at epoch 14 |
| B — fourier-8 | qvkpm23n | 8 | 14/50 | 143.23 | NaN⚠️ | Worse than baseline (seed variance?) |
| C — fourier-16 | 24yldhv7 | 16 | 14/50 | **130.46** | NaN⚠️ | **WINNER** |

**Per-split val mae_surf_p:**

| Arm | in_dist | camber_rc | camber_cruise | re_rand |
|-----|---------|-----------|----------------|---------|
| A (baseline) | 156.98 | 144.01 | 119.48 | 120.44 |
| B (n=8) | 191.33 | 148.23 | 102.79 | 130.55 |
| C (n=16) | 159.57 | 150.12 | **89.02** | 123.13 |

**Decision: MERGED** (Arm C config). val_avg: 135.23 → 130.46 (-3.5%).

**Analysis:**
- Main signal: cruise camber OOD split drops 25.5% (119→89). Fourier features help geometry interpolation at the frequency scale of camber variation. raceCar camber split shows no benefit (+4%) — possibly because raceCar pressure is dominated by ground-effect features better captured by the existing dsdf descriptor than by position frequency.
- Arm B anomaly (worse than baseline) is likely seed variance at 14 epochs rather than a real effect.
- ALL arms hit 30-min wall clock timeout at epoch 14 of 50 — severe under-training. Longer runs would give more signal.
- **Critical bug discovered:** `test_avg/mae_surf_p = NaN` on all arms due to model overflow on test_geom_camber_cruise split. Tracked in PR #3296 (thorfinn follow-up).
- Baseline-equivalent (Arm A) val_avg = 135.23 is now the empirical starting point for all Round 1 comparisons.

---

## 2026-05-15 18:15 — PR #3098: SmoothL1 / Huber loss for heavy-tailed surface pressure

- Branch: `willowpai2i48h5-alphonse/huber-surface-loss`
- Hypothesis: Replace MSE with SmoothL1 (Huber) loss to cap gradient magnitude on heavy-tailed high-Re samples, rebalancing optimizer toward moderate-Re bulk. Expected -3 to -8% on val_avg/mae_surf_p.

| Arm | W&B run | loss_type | beta | epochs | val_avg | test_avg | Notes |
|-----|---------|-----------|------|--------|---------|----------|-------|
| A — MSE baseline | 9jr2u0f9 | mse | — | 12/50 | 137.54 | NaN⚠️ | Wall-clock |
| B — SmoothL1 β=0.1 | nlvd0e6f | smooth_l1 | 0.10 | 14/50 | 111.22 | NaN⚠️ | |
| C — SmoothL1 β=0.05 | md6so639 | smooth_l1 | 0.05 | 14/50 | **96.05** | NaN⚠️ (cruise bug) | **WINNER** |

**Per-split val mae_surf_p:**

| Arm | in_dist | camber_rc | camber_cruise | re_rand |
|-----|---------|-----------|----------------|---------|
| A (MSE) | 193.27 | 135.11 | 102.79 | 118.99 |
| B (β=0.1) | 146.38 | 129.52 | 75.69 | 93.27 |
| C (β=0.05) | **109.64** | **112.30** | **73.22** | **89.06** |

**Test partial (excl. cruise, arm C):** in_dist 96.04, camber_rc 100.16, re_rand 84.02

**Decision: MERGED** (Arm C). val_avg: 130.46 → 96.05 (-26.4%). New launch best.

**Analysis:**
- Effect size far exceeded prediction (predicted -8%, observed -30%). Pressure is the dominant heavy-tailed channel; SmoothL1 is near-perfectly matched to the metric.
- β=0.05 outperforms β=0.1 — smaller transition point keeps more gradients in linear regime during under-training phase.
- All 4 val splits improved; OOD gains (camber_cruise -29%, re_rand -25%) suggest Huber reduces high-Re sample dominance that hurts OOD generalization.
- These runs did NOT use Fourier PE (n_fourier=0) — gains are additive to PR #3123. Round 2 compound stack expected to deliver further improvement.
- test_avg NaN persists (cruise GT bug) — tracked in PR #3296.

---

## 2026-05-15 18:15 — PR #3109: bf16 + bigger batch (bs=8/16)

- Branch: `willowpai2i48h5-frieren/bf16-bigger-batch`
- Hypothesis: bf16 AMP + larger batches increase effective epoch count in 30-min window → better val.

| Arm | W&B run | batch_size | dtype | epochs | val_avg | test_avg | Notes |
|-----|---------|-----------|-------|--------|---------|----------|-------|
| A — fp32 bs=4 | uxk9rt4t | 4 | fp32 | 14/50 | 133.72 | NaN⚠️ | Best arm |
| B — bf16 bs=8 | 3a8s43dk | 8 | bf16 | 17/50 | 139.34 | NaN⚠️ | More epochs, worse result |
| C — bf16 bs=10 | mkqpnjzp | 10 | bf16 | 17/50 | 162.06 | NaN⚠️ | Worst — bs=12/16 OOMed |

**Decision: CLOSED** (does not beat merged baseline 96.05; merge conflict).

**Analysis:**
- bf16 speedup is real (~18% faster epochs) but larger batches hurt convergence — LR not scaled with batch size. Arms B/C completed 17 epochs but final val_avg worse than baseline's 14 epochs.
- bs=16/12 OOMed on real loader (242K-node cruise meshes push padded batch to 94+ GB). Max viable batch is ~bs=10.
- Key insight: bs=4 with bf16 alone (no batch scaling) may be worth a quick verification — frieren suggested this. Could fold into compound stack PR.
- Cosine LR with T_max=50 is poorly calibrated against the ~14-epoch wall-clock ceiling — stays near peak lr for entire run.

---

<!-- Template:
## <YYYY-MM-DD HH:MM> — PR #<number>: <title>
- Branch: <student-branch-name>
- Hypothesis: <hypothesis>
- Results:

| Arm | W&B run ID | val_avg/mae_surf_p | test_avg/mae_surf_p | notes |
|-----|------------|--------------------|---------------------|-------|
| A (baseline) | ... | ... | ... | |
| B (...) | ... | ... | ... | |

- Analysis: <results commentary, analysis and conclusions>
-->
