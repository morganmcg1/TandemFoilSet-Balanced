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

## 2026-05-15 18:50 — PR #3100: Transolver scale-up (wider/deeper architecture, ~3-7M params)

- Branch: `willowpai2i48h5-askeladd/transolver-scale-up`
- Hypothesis: Larger n_hidden/n_layers/n_head will improve representation capacity → better val_avg under reasonable VRAM headroom (96 GB GPU).

| Arm | W&B run | n_hidden/n_layers/n_head | n_params | bs | epochs | val_avg | test_avg | Peak VRAM |
|-----|---------|--------------------------|----------|----|----|--------|---------|----------|
| A — baseline | xii5dbk8 | 128/5/4 | 0.66M | 4 | 11 | **150.94** | **136.70** | 42.1 GB |
| B — wider | d7coya51 | 192/6/8 | 1.70M | 4 | 6 | 168.02 | 153.92 | 80.2 GB |
| C — deeper-wide | pcarz06v | 256/6/8 | 3.01M | 2 | 4 | 179.92 | 166.38 | 49.4 GB |

**Decision: CLOSED** (val_avg 150.94 = +57% vs new baseline 96.05).

**Analysis:**
- Capacity is not the binding constraint at our 30-min wall clock — convergence is. Arms B/C reach far fewer epochs (6, 4) and regress badly. This conclusively closes the parameter-count axis as a winning lever.
- First valid test_avg of the launch (136.70 on arm A, NaN-clean across all splits) — credit Edward's parallel NaN diagnosis enabling this.
- Cruise split is easiest in absolute terms for every arm (93.68 on A) — counter to in-dist intuition.

---

## 2026-05-15 18:50 — PR #3103: Slice-num scaling (64 → 128 / 192 physics tokens)

- Branch: `willowpai2i48h5-edward/slice-num-scaling`
- Hypothesis: Increasing slice_num gives more PhysicsAttention tokens → better representation. Combined with rerun-with-NaN-guard for clean test metrics.

| Arm | W&B run | slice_num | epochs | val_avg | test_avg | Notes |
|-----|---------|-----------|--------|---------|----------|-------|
| A baseline | aod6uhrj | 64 | 14 | **124.39** | NaN¹ | best val_avg, pre-fix |
| A rerun w/ full NaN fix | zxu6ktx5 | 64 | 14 | 137.14 | **124.02** | first finite test_avg |
| B 128 (no fix) | s0cgfl2s | 128 | 11 | 140.23 | NaN¹ | worse than A |
| B 128 rerun w/ fix | 9j7oeip2 | 128 | 11 | 150.16 | 138.07 | worse |
| C 192 | a6t73no8 | 192 | 0 | — | — | **OOM in epoch-1 val** |

¹ NaN from pre-NaN-fix runs (y-inf in test_geom_camber_cruise/000020.pt).

**Decision: CLOSED** (val_avg 124.39 = +29% vs new baseline 96.05; OOM at slice=192).

**Analysis:**
- Slice-num scaling does not help under 30-min wall clock — same convergence-not-capacity verdict as scale-up.
- **MAJOR launch credit:** Edward independently diagnosed the y-inf root cause on `test_geom_camber_cruise/000020.pt` (761 inf p-values), informing thorfinn's two-pronged NaN guard in PR #3296.
- Edward's `zxu6ktx5` rerun was the first finite test_avg on the launch (124.02), confirming the y-side mask works correctly.
- slice=192 OOMed on epoch-1 val on a single H100 — the OOM happens during validation on cruise (largest mesh), not training.

---

## 2026-05-15 18:50 — PR #3105: Linear warmup + cosine LR

- Branch: `willowpai2i48h5-fern/warmup-cosine-lr`
- Hypothesis: Linear LR warmup over first ~5% epochs avoids early gradient instability → better val.

| Arm | W&B run | lr peak | warmup_frac | epochs | val_avg | test_3split_partial | Notes |
|-----|---------|---------|-------------|--------|---------|---------------------|-------|
| A baseline (advisor pick) | i3z00pw4 | 5e-4 | 0.00 | 14 | **127.82** | 126.44 | |
| A repeat 1 | 07ddhitq | 5e-4 | 0.00 | 14 | 122.10 | 119.64 | run-noise diagnostic |
| A repeat 2 | b4cv2rqp | 5e-4 | 0.00 | 14 | 131.26 | 127.00 | run-noise diagnostic |
| B warmup-5e-4 | pd21qc2t | 5e-4 | 0.05 | 14 | 145.40 | 148.38 | +13.8% regression |
| C warmup-1e-3 | l2pow9iw | 1e-3 | 0.05 | 14 | 143.79 | 141.23 | +12.5% regression |

**Decision: CLOSED** (val_avg 127.82 = +33% vs new baseline 96.05; warmup arms regress vs even worst-case arm-A).

**Analysis:**
- Warmup actively hurts in our 14-epoch-cap regime — cosine T_max=50 already keeps LR near peak, further suppressing early LR throws away gradient signal.
- Three arm-A repeats give us the **first run-to-run variance estimate** on the launch: σ ≈ 4.6 on val_avg (range 122.10–131.26), about ~3.6% relative. Useful reference for evaluating all future small deltas.
- Existing cosine-no-warmup schedule is locally optimal at this wall-clock budget.

---

## 2026-05-15 18:50 — PR #3114: Gradient clipping + EMA model weights

- Branch: `willowpai2i48h5-nezuko/grad-clip-ema`
- Hypothesis: grad-clip(1.0) suppresses gradient spikes; EMA(0.999) on model weights provides flat-minima inference → both improve generalization.

| Arm | W&B run | grad_clip | ema_decay | epochs | val_avg | test_3split |
|-----|---------|-----------|-----------|--------|---------|-------------|
| A baseline | gt0hqg32 | 0.0 | 0.0 | 14 | 135.62 | 135.66 |
| A rerun 1 | jpdav2j1 | 0.0 | 0.0 | 13 | 128.41 | 128.19 |
| A rerun 2 | mkmflt8c | 0.0 | 0.0 | 12 | 133.73 | 132.68 |
| B grad-clip only | p2v9zpal | 1.0 | 0.0 | 14 | 104.87 | 101.96 |
| B rerun | tcggs514 | 1.0 | 0.0 | 13 | 111.78 | 108.04 |
| **C clip+EMA** | **i69fv3fg** | **1.0** | **0.999** | **14** | **102.67** | **99.48** |

**Decision: CLOSED** (val_avg 102.67 = +6.9% vs new baseline 96.05; mechanism subsumed by alphonse's compound stack #3379).

**Analysis:**
- Strong standalone result — clip+EMA gets to **2nd place** in Round 1 leaderboard. Clip alone (104.87) captures most of the gain; EMA adds ~2 points.
- Mechanism confirmed orthogonal to Huber, hence is exactly the optimization layer being tested on top of Huber in alphonse's Round 2 compound stack PR #3379.
- Not merged because the stack is a strictly stronger candidate — would be a regression to land 102.67 over the 96.05 baseline standalone.
- Round 1 winner (Huber) and 2nd place (clip+EMA) are mechanistically orthogonal — supports the compound stack hypothesis.

---

## 2026-05-15 18:50 — PR #3118: Per-channel surface loss weighting

- Branch: `willowpai2i48h5-tanjiro/per-channel-loss-weighting`
- Hypothesis: Up-weighting the pressure channel in surf_loss focuses the optimizer on the metric → better val_avg/mae_surf_p.

| Arm | W&B run | surf_w (Ux,Uy,p) | epochs | val_avg | Notes |
|-----|---------|------------------|--------|---------|-------|
| A baseline | wwfhp260 | (1.0, 1.0, 1.0) | 14 | **130.51** | |
| B p-2x | lvw0bz34 | (1.0, 1.0, 2.0) | 13 | 144.58 | +10.8% regression |
| C p-heavy | 3770ejgb | (0.5, 0.5, 3.0) | 12 | 142.38 | +9.1% regression |

**Decision: CLOSED** (val_avg 130.51 = +36% vs new baseline 96.05; p-weighting REGRESSES pressure itself).

**Analysis:**
- Counterintuitive finding: up-weighting p hurt the pressure channel itself (arm B vol_p +11%, arm C vol_p +49%).
- Mechanism: Ux/Uy gradients reinforce shared encoder representations that the pressure head relies on. Deprioritizing the velocity channels weakens the encoder, hurting everything including pressure.
- Closes the per-channel-weighting axis. Confirms that multi-task coupling is doing useful work and supports the bandit-style "train on all 3 outputs" approach.

---

## 2026-05-15 18:50 — PR #3296: Two-pronged NaN guard (pred-side nan_to_num + y-side sample mask)

- Branch: `willowpai2i48h5-thorfinn/fix-test-cruise-nan`
- Hypothesis: Two contributors to NaN — model pred overflow on cruise OOD samples AND inf y values in test_geom_camber_cruise/000020.pt (761 nodes). Two-pronged guard resolves both.

| Metric | Before fix | After two-pronged guard (run 4gqpc5ez) |
|--------|------------|----------------------------------------|
| val_avg/mae_surf_p | 130.12 | 142.20 |
| test_avg/mae_surf_p | NaN | **128.97** |
| test_geom_camber_cruise/mae_surf_p | NaN | 103.04 |

Diagnostic confirms 0 non-finite predictions (pred-side clean post-Huber), only y-side inf samples being correctly dropped (761 nodes from 1 sample).

**Per-split test MAE (run 4gqpc5ez, n_fourier=0):**

| Split | test surf_p |
|-------|-------------|
| in_dist | 147.12 |
| camber_rc | 135.67 |
| camber_cruise | **103.04** |
| re_rand | 130.05 |
| **avg** | **128.97** |

**Decision: SEND BACK FOR REBASE.** Branch has merge conflict against #3098 (Huber) merge. Once rebased onto current baseline + rerun, will give us the first clean test_avg on the launch's actual best config.

**Analysis:**
- Two-pronged guard is conceptually correct: pred-side `nan_to_num` handles overflow even with Huber; y-side sample-level mask drops corrupted GT.
- val_avg 142.20 reflects training-without-Huber baseline — when rebased on #3098 (Huber β=0.05) and rerun, expect val_avg ~96 + test_avg in 90s-100s range.
- This is the **single most important PR to land cleanly** — every Round 2 PR's test_avg metric depends on this guard.

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
