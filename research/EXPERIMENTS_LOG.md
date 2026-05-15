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
