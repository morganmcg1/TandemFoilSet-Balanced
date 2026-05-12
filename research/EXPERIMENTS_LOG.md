# SENPAI Research Results — Charlie pai2g 24h r3

Advisor branch: `icml-appendix-charlie-pai2g-24h-r3`.
Records of reviewed and merged experiment PRs. Add a new section under the
appropriate heading whenever an experiment terminal-completes.

---

## Round 2 — build on merged stack (ongoing)

All experiments in this round must rebase on `icml-appendix-charlie-pai2g-24h-r3` (includes grad_clip=1.0, wd=1e-3, OneCycleLR, EMA=0.999) before running. Current baseline: **val_avg/mae_surf_p = 112.546** (PR #1520).

---

### 2026-05-12 19:53 — PR #1520: OneCycleLR + EMA weights (fern)
**Branch:** `charliepai2g24h3-fern/onecycle-lr-ema` | **Status: MERGED** ⭐

- **Hypothesis:** Replace CosineAnnealingLR (T_max=50, only ~5% annealed in 14 epochs) with OneCycleLR (auto-adapts to actual steps). Add EMA(0.999) weights for evaluation to eliminate checkpoint-selection jitter.
- **val_avg/mae_surf_p: 112.546** (epoch 14/14) — **−2.5% vs baseline 115.403**.
- **Per-split:** single=125.10, rc=136.04, **cruise=86.31** (best camber), re_rand=102.73.
- **Test (3-split proxy):** single=113.89, rc=118.86, re_rand=99.84 → **110.862** (−3.7% vs baseline 115.13).
- **Analysis:** Val curve was strictly monotone decreasing (epoch 14 still the best at cap). EMA eliminated the regression fern's own baseline saw at epochs 13–14 (115.40 → 119.37 → 126.42). OneCycleLR peak LR = 5e-3 reached at epoch 3; lr=4.3e-3 at cap (barely in cosine anneal phase). val_re_rand improved 6.4% (109.76 → 102.73) — EMA smoothing particularly helps the Re-generalization split. Both effects (OneCycleLR + EMA) are compounding.
- **Artifacts:** `models/model-charliepai2g24h3-fern-onecycle-ema-decay999-20260512-191518/{metrics.jsonl,metrics.yaml}`

---

### 2026-05-12 19:54 — PR #1484: Huber loss delta=0.5 and delta=1.0 (alphonse)
**Branch:** `charliepai2g24h3-alphonse/huber-pressure-loss` | **Status: SENT BACK**

- **Hypothesis:** Huber loss (delta=1.0 in normalized space) clips high-Re gradient extremes and improves `val_avg/mae_surf_p` by 3–8%.
- **val_avg/mae_surf_p:** delta=0.5: **108.097**, delta=1.0: **108.104** (both epoch 14; tie within noise).
- **Per-split (delta=0.5 best):** single=146.2, rc=113.3, cruise=79.0, re_rand=93.8.
- **Test (3-split proxy):** delta=0.5: single=124.8, rc=98.4, re_rand=90.4 → ~104.55. delta=1.0: single=111.3, rc=105.3, re_rand=102.3 → ~106.27.
- **Analysis:** Best raw number of all round-1 runs (108.10 vs 115.40 baseline), but ran on pre-merge base (no grad_clip + wd, no OneCycleLR + EMA). Merge conflict blocked direct merge. Key concern: delta=0.5 helps cruise/re_rand but hurts single_in_dist (146.2 vs 122.6 at d=1.0) — aggressive clipping removes signal needed for high-Re single-foil. Bug analysis of cruise NaN matches tanjiro's independent trace.
- **Why sent back:** Merge conflicts (DIRTY state). Pre-merge base comparison unfair to new baseline 112.546. Instructed: rebase, run both arms (d=0.5, d=1.0) with full merged stack (OneCycleLR + EMA + clip + wd), pass criterion val_avg < 112.546.
- **Artifacts:** `models/model-huber-delta-{1p0,0p5}-20260512-*/metrics.yaml`

---

### 2026-05-12 22:17 — PR #1493 v2: slice_num 64→128 (nezuko) — CLOSED (regression)
**Branch:** `charliepai2g24h3-nezuko/more-slices-128` | **Status: CLOSED**

- **Hypothesis:** Doubling PhysicsAttention slice_num gives the model more abstract "physics tokens" for soft node-set assignment. Predicted −5 to −10% on val_avg.
- **val_avg/mae_surf_p: 121.354** (epoch 11/11, fully annealed) — **+17.70% over baseline 103.10**.
- **Per-split:** single=135.46, rc=144.30, cruise=94.89, re_rand=110.76. Uniform regression across all 4 splits.
- **Test (4-split safe re-eval):** **113.686** vs PR #1495's 94.757 → **+20.0%**.
- **Analysis:** Clean run on full merged stack (be35472: PR #1491+#1520+#1495). Conclusion: at 11-epoch budget, doubling slice_num adds ~40K params to `in_project_slice` layers (5×4×32×64 extra) that have to learn useful routing patterns from scratch — undertrained → noise. The baseline slice_num=64 is already converged for its size within 11 epochs. Transolver paper's slice-num sensitivity result was at a different architecture/budget regime where slice_num was actually bottlenecked.
- **Why closed:** >5% regression on every metric; large margin; root cause well-understood by student. Not worth a v3.
- **Credit:** Nezuko's analysis of the cruise NaN trace (boolean→0.0→Inf*0=NaN mechanics) is the canonical explanation. Used the safe re-eval pattern correctly for paper-facing test metric.
- **Artifacts:** `models/model-more-slices-128-v2-20260512-205438/{metrics.jsonl,metrics.yaml,test_safe_eval.{jsonl,log}}`

---

### 2026-05-12 22:25 — PR #1662: Fourier mesh positional encoding (nezuko) — WIP (assigned)
**Branch:** `charliepai2g24h3-nezuko/fourier-mesh-positional-encoding` | **Status: WIP**

- **Hypothesis:** Replace raw (x, y) mesh coordinates with sinusoidal Fourier features γ(x) = [sin(2π·2ᵏ·x), cos(...)] for k=0..5. Standard NeRF-style positional encoding. Gives attention direct access to high-frequency spatial signals (boundary layers, wake structure). Predicted −3 to −8% on val_avg.
- **Expected per-split:** Largest gains on boundary-layer-dominated splits (single_in_dist, re_rand). Zero parameter cost.
- **Artifacts:** TBD

---

### 2026-05-12 20:55 — PR #1543: Log-cosh loss on merged stack v1 (fern) — SENT BACK
**Branch:** `charliepai2g24h3-fern/logcosh-loss` | **Status: SENT BACK**

- **Hypothesis:** Log-cosh loss is a smooth, threshold-free Huber alternative. Expected −3 to −8% on val_avg.
- **val_avg/mae_surf_p: 106.682** (epoch 14/14) — **−5.21% vs PR #1520 (112.55)**, but **+3.5% over current merged baseline 103.10 (PR #1495)**.
- **Per-split:** single=124.05, rc=129.81, **cruise=75.92** (best split — gradient saturation effect on high-Re), re_rand=96.95.
- **Test (4-split safe re-eval):** **100.373**.
- **Analysis:** Effect shape matches hypothesis exactly. Cruise (high-Re heavy-tail) gets the biggest gain (−12.0%), single_in_dist (mid-magnitude residuals) barely moves (−0.8%). Gradient saturation via `tanh(r)` is doing what Huber does, without the δ knob. BUT run was on `git_commit=29893da` (post-#1520, pre-#1495) — 6 min after #1495 merged. Stale base.
- **Why sent back:** Result doesn't beat current baseline 103.10. Log-cosh effect likely orthogonal to augmentation (different mechanism — loss-curvature vs data-OOD). Rebase + re-run with augmentation default ON should land near 97-100 if the effects compound. Single-arm rerun instructed.
- **Artifacts:** `models/model-logcosh-onecycle-ema-20260512-200805/{metrics.jsonl,metrics.yaml,safe_eval.json,config.yaml}`

---

### 2026-05-12 19:59 — PR #1495: AoA + NACA camber jitter v2 (thorfinn) — rebase
**Branch:** `charliepai2g24h3-thorfinn/geometry-aoa-augmentation` | **Status: MERGED** ⭐

- **Hypothesis:** Online ±0.5° AoA jitter + ±0.002 NACA camber jitter on training inputs should improve OOD camber generalization.
- **val_avg/mae_surf_p: 103.100** (epoch 14/14) — **−8.4% vs PR #1520 baseline 112.546**.
- **Per-split:** single=125.91, rc=114.35, **cruise=77.99** (best split, exactly as hypothesized), re_rand=94.15.
- **Test (safe re-eval, 4-split):** single=105.14, rc=100.58, cruise=83.48 (199/200), re_rand=89.83 → **94.757**.
- **Test (3-split proxy):** 98.520.
- **Analysis:** Augmentation behaves as predicted on camber-OOD: cruise (M=2-4 held out) interpolates cleanly at 77.99 (best of all splits). val curve was monotone descending (244 → 195 → 183 → ... → 103). Ran with cosine T_max=14 (no OneCycleLR/EMA), so the merged baseline number 103.10 is for this specific config. Composability with OneCycleLR+EMA is untested. Thorfinn also wrote a reusable safe re-eval side script (`safe_re_eval.py`) that is now canonical for paper-facing test reporting.
- **Artifacts:** `models/model-geom-aoa-augment-r2-20260512-190924/{metrics.jsonl,metrics.yaml,test_safe_eval.{jsonl,log},safe_re_eval.py}`

---

### 2026-05-12 20:02 — PR #1494: FiLM conditioning v2 (tanjiro) — SENT BACK (best raw result)
**Branch:** `charliepai2g24h3-tanjiro/re-film-conditioning` | **Status: SENT BACK**

- **Hypothesis:** FiLM (γ·h + β) per TransolverBlock conditioned on log(Re) for cross-regime generalization.
- **val_avg/mae_surf_p: 100.987** (epoch 14/14) — best raw number this track has produced. 12.6% better than #1520, 2.1% better than current #1495 baseline.
- **Per-split:** single=122.17, rc=112.23, cruise=76.64, **re_rand=92.90** (best split, exactly as hypothesized).
- **Test (safe re-eval):** single=108.58, rc=99.88, cruise=63.97 (199/200), re_rand=88.70 → **90.281**.
- **FiLM diagnostics:** Block0 weight norm 0 → 4.38, block4 0 → 2.09 over 14 epochs. Conditioning is being learned (monotonic growth). Earlier blocks acquire larger modulation. Train grad_clip_fire_rate stayed at 1.0 (max_norm=1.0 actively binding).
- **Why sent back:** mergeStateStatus=DIRTY/CONFLICTING. Branch rebased on c7f371c (pre-#1520); missing OneCycleLR+EMA (#1520) AND augmentation (#1495). Need v3 rebase onto post-#1495 base + run two arms: (A) full stack including augmentation+OneCycleLR+EMA, (B) FiLM + augmentation cosine T_max=14 matching #1495 setup.
- **Artifacts:** `models/model-re-film-conditioning-v2-20260512-191851/{metrics.jsonl,metrics.yaml,test_safe_eval.log}`

---

### 2026-05-12 19:58 — PR #1488: Decoupled per-channel heads + surf_weight_p=20 (askeladd)
**Branch:** `charliepai2g24h3-askeladd/decoupled-channel-heads` | **Status: SENT BACK**

- **Hypothesis:** Replace shared mlp2 with three independent linear heads (Ux, Uy, p) + per-channel surface weights `[10, 10, 20]` to amplify pressure gradient signal.
- **val_avg/mae_surf_p: 132.340** (epoch 13/14) — 28% WORSE than current baseline 103.10.
- **Per-split:** single=158.49, rc=152.18, cruise=99.93, re_rand=118.75.
- **Test (3-split proxy):** ~119.81. Cruise NaN per usual.
- **Analysis:** Ran on pre-#1491 base (no grad_clip+wd, no OneCycleLR+EMA, no augmentation). Cosine T_max=50 mismatch caused noisy val curve (oscillating around minimum: 202 → 152 → 132 → 163). Architecture itself (decoupled heads, −16K params from removing mlp2) is fine; the bad result is the missing optimization stack + scheduler mismatch, not the head decoupling. Askeladd correctly identified the literal /3 normalization in instructions would have given effective `[3.33, 3.33, 6.67]` weights and made the right judgment call to drop /3 — implementation was `[10, 10, 20]` as intended. Also included a non-finite-y pre-filter in evaluate_split (parallel to thorfinn's safe re-eval).
- **Why sent back:** Need rebase + re-run with full merged stack (two arms: A=full stack, B=cosine T_max=14 matching #1495). Behavior of decoupled heads cannot be assessed until the optimization stack is matched.
- **Artifacts:** `models/model-charliepai2g24h3-askeladd-decoupled-heads-surf-p20-20260512-190233/{metrics.jsonl,metrics.yaml}`

---

## Round 1 — broad coverage (assigned 2026-05-12)

Hypotheses sourced from `/research/RESEARCH_IDEAS_2026-05-12_18:00.md`.

**Cross-round findings (apply to all round 1 results):**
- All 5 reviewed runs hit the 30-min timeout. With `--epochs 50` and CosineAnnealingLR `T_max=50`, only 7-14 epochs ran → LR barely annealed (~93-95% of peak).
- `test_geom_camber_cruise/mae_surf_p` is NaN for all runs. Root cause traced by tanjiro (PR #1494): `splits_v2/.test_geom_camber_cruise_gt/000020.pt` contains 761 `+Inf` values in `y[:, 2]`. In `data/scoring.py`, the subtraction `pred - y` happens before the sample-skip mask is applied, so `Inf * 0 = NaN` poisons the accumulator. File is read-only; use a safe re-eval side script (zero-fill non-finite `y` before subtraction) or the 3-split proxy.
- grad_clip=1.0 fires on 100% of training batches (real norms 41-115). This is unit-norm SGD + AdamW adaptive scaling, not "spike clipping" — but it works.

---

### 2026-05-12 18:56 — PR #1491: Gradient clipping + weight_decay tuned (fern)
**Branch:** `charliepai2g24h3-fern/grad-clip-adamw-tuned` | **Status: MERGED** ⭐

- **Hypothesis:** grad_clip=1.0 + weight_decay 1e-4→1e-3 would stabilize training on high-Re outliers.
- **val_avg/mae_surf_p: 115.403** (epoch 12/14)
- **Per-split:** single=133.09, rc=129.76, **cruise=88.99** (best), re_rand=109.76
- **Test (3-split proxy):** single=116.98, rc=119.26, re_rand=109.15 → proxy avg ~115.1
- **Analysis:** Clipping fired on 100% of batches (norms 41-115). Produces the smoothest val trajectory of round 1 (249 → 115 over 12 epochs, nearly monotone). The wd=1e-3 + clip combination outperforms all other round-1 variants. This is the new baseline.
- **Artifacts:** `models/model-grad-clip-wd1e-3-20260512-181000/metrics.jsonl`

---

### 2026-05-12 18:56 — PR #1495: AoA + NACA camber jitter augmentation (thorfinn)
**Branch:** `charliepai2g24h3-thorfinn/geometry-aoa-augmentation` | **Status: SENT BACK**

- **Hypothesis:** Online jitter of AoA (±0.5°) and NACA camber (±0.002) improves OOD camber splits.
- **val_avg/mae_surf_p: 129.694** (epoch 12/14)
- **Per-split:** single=155.30, rc=141.33, **cruise=102.93** (OOD best), re_rand=119.22
- **Test (3-split proxy):** single=139.75, rc=129.04, re_rand=122.90 → ~130.56
- **Analysis:** 12% worse than fern's run, but same epoch budget. Camber OOD splits are NOT the worst — single-foil in-dist (155.3) is worst, possibly because extreme high-Re raceCar pressures dominate. Cannot isolate augmentation effect without equal-budget no-aug control. Cosine T_max mismatch (same as all round 1). Sent back: rebase on #1491 baseline + fix T_max.
- **Artifacts:** `models/model-geom-aoa-augment-20260512-181104/metrics.jsonl`

---

### 2026-05-12 18:53 — PR #1492: mlp_ratio 2→4 wider FFN (frieren)
**Branch:** `charliepai2g24h3-frieren/mlp-ratio-4-wider-ffn` | **Status: SENT BACK**

- **Hypothesis:** Restoring mlp_ratio to the paper's default (4) improves FFN capacity.
- **val_avg/mae_surf_p: 144.334** (epoch 11/13)
- **Per-split:** single=183.46, rc=153.62, cruise=105.23, re_rand=135.03
- **Test (3-split proxy):** single=155.33, rc=139.70, re_rand=125.49 → ~140.18
- **Analysis:** 25% worse than fern's run. Same 30-min timeout issue. mlp_ratio=4 is ~21% slower per epoch, so fewer epochs completed. Without proper cosine annealing the comparison is unfair. Sent back: rebase on #1491 + set --epochs 12 to match actual budget.
- **Artifacts:** `models/model-mlp-ratio-4-20260512-180817/metrics.jsonl`

---

### 2026-05-12 19:09 — PR #1494: FiLM conditioning on log(Re) (tanjiro)
**Branch:** `charliepai2g24h3-tanjiro/re-film-conditioning` | **Status: SENT BACK**

- **Hypothesis:** Inject FiLM (γ·h + β) per TransolverBlock conditioned on log(Re); should help cross-Re generalization (val_re_rand).
- **val_avg/mae_surf_p: 129.94** (epoch 12/14) — 12.6% worse than #1491 baseline.
- **Per-split:** single=156.91, rc=140.57, cruise=106.23, **re_rand=116.04 (best)**.
- **Test (safe re-eval):** single=138.96, rc=123.33, cruise=90.24 (199/200 samples), re_rand=120.40 → **test_avg=118.23**.
- **FiLM diagnostics:** γ/β weight norms grow monotonically from zero (block0 0→5.97, block4 0→3.01 over 12 epochs). Conditioning IS being learned. val_re_rand becomes the best-of-4 split — consistent with the FiLM hypothesis.
- **Why sent back, not closed:** Ran on pre-merge base (no grad_clip + wd=1e-3); not a fair comparison to merged baseline. Same cosine T_max=50 mismatch. Need rebase + --epochs 14 re-run.
- **Bonus:** Tanjiro's bug analysis on the cruise NaN is the source of the safe re-eval pattern now in BASELINE.md.
- **Artifacts:** `models/model-re-film-conditioning-20260512-182128/{metrics.jsonl,metrics.yaml,test_safe_eval.log}`

---

### 2026-05-12 19:22 — PR #1493: PhysicsAttention slice_num 64→128 (nezuko)
**Branch:** `charliepai2g24h3-nezuko/more-slices-128` | **Status: SENT BACK**

- **Hypothesis:** Doubling slice_num gives PhysicsAttention more token capacity to represent surface vs. volume regions in 74-242K-node meshes.
- **val_avg/mae_surf_p: 138.317** (epoch 10/11) — 19.9% worse than #1491 baseline.
- **Per-split:** single=175.88, rc=147.04, cruise=108.51, re_rand=121.83.
- **Test (3-split proxy):** single=146.80, rc=135.49, re_rand=123.74 → ~135.01.
- **Memory:** Peak 54.5 / 96 GB — slice_num=128 is cheap. Room for slice_num=192 or 256 in a later round.
- **Cruise NaN trace:** Independently identified the same 761-Inf bug as tanjiro (PR #1494). Clearest write-up of the boolean→float cast mechanics. Credited alongside tanjiro.
- **Why sent back, not closed:** Ran on pre-merge base (no grad_clip + wd=1e-3); not a fair comparison to merged baseline. Same cosine T_max=50 mismatch (11 epochs only). Need rebase on #1491 + --epochs 11 re-run.
- **Artifacts:** `models/model-more-slices-128-20260512-180855/{metrics.jsonl,metrics.yaml}`

---

### 2026-05-12 18:53 — PR #1490: Scale model n_hidden=256, n_head=8 (edward)
**Branch:** `charliepai2g24h3-edward/scale-model-256` | **Status: SENT BACK**

- **Hypothesis:** n_hidden 128→256, n_head 4→8 (~2.54M params) improves capacity.
- **val_avg/mae_surf_p: 172.262** (epoch 6/7; 30-min cap after 7 epochs)
- **Per-split:** single=199.15, rc=194.97, cruise=131.46, re_rand=163.47
- **Test:** NaN overall; 3-split proxy: single=191.41, rc=186.78, re_rand=159.10 → ~179.09
- **Analysis:** Severely under-budgeted — ~260 s/epoch means only 7 epochs in 30 min. Model trending down (172 → 176 at epoch 7) but far from converged. Also: model is 2.54M not the predicted ~6M (mlp_ratio=2 was not changed). OOM risk (83.9GB peak). Sent back: scale down to n_hidden=192 + set --epochs 10 + rebase on #1491.
- **Artifacts:** `models/model-scale-model-256-20260512-180850/metrics.jsonl`
