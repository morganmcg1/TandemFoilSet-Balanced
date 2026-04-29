# SENPAI Research State
- 2026-04-29 17:30 (icml-appendix-charlie-pai2f-r3)
- No recent research directives from the human researcher team

## Current Best Baseline

**val_avg/mae_surf_p = 35.8406** (PR #1208 frieren, merged 2026-04-29)

Configuration: Lion + L1 + EMA(0.995) + bf16 + n_layers=1 + surf_weight=28 + cosine T_max=75 (single full-cycle decay, **no warmup**) + clip_grad=1.0 + n_hidden=128 + n_head=4 + slice_num=64 + mlp_ratio=2 + batch_size=4 + 75 epochs + Fourier pos enc on (x,z) freqs=(1,2,4,8,16,32,64) + FiLM global conditioning (Re/AoA/NACA scale+shift per TransolverBlock, DiT/AdaLN-Zero init).

Training: ~30.33 min wall-clock, **truncated at ep57/75** (still improving). Best epoch = 57 = last epoch reached. Peak VRAM 9.89 GB, n_params 252,487.

Per-split val:
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 33.7806 |
| val_geom_camber_rc | 51.6584 |
| val_geom_camber_cruise | 19.6970 |
| val_re_rand | 38.2266 |
| **val_avg** | **35.8406** |

Per-split test:
| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 29.8002 |
| test_geom_camber_rc | 44.5661 |
| test_geom_camber_cruise | 16.0529 |
| test_re_rand | 29.2713 |
| **test_avg** | **29.9226** |

Key observations:
- `val_geom_camber_rc` (51.66) is by far the dominant error split — 1.5× the avg, 2.6× the cruise split. This is the biggest remaining lever for val_avg.
- Best epoch = final reached epoch (wall-clock truncation). Training-dynamics levers (LR horizon, warmup, scheduling) still have headroom.
- T_max=75 single-decay was the primary driver of the −3.33% gain over PR #1175 (warmup+T_max=45). At ep49 alone, PR #1208 already achieved 37.21 vs PR #1175's 37.07 — the LR-horizon decision is what matters, not the longer epoch budget.

## Current Research Focus

The FiLM + Fourier + extended single-decay cosine recipe is now the floor. Active levers being explored:

1. **Training dynamics / horizon composition** — extended training (PR #1226 100ep+T_max=100), warmup composition (PR #1231 askeladd 60ep+T_max=60+warmup=5), Lion LR sweep (PR #1209).
2. **Capacity scaling** — n_hidden=192 with current cosine (PR #1228), older n_hidden=192 trial (PR #1202), n_layers=2 with warmup (PR #1216). Note: PR #1108/#1170 closed prior depth/width sweeps as negative on the pre-FiLM baseline; these in-flight runs re-test on the FiLM frontier.
3. **Input representation** — extended Fourier freqs sweep L∈{5,6,7,8} octaves (PR #1174).
4. **Optimization-batch coupling** — bs sweep {8,16,32} with sqrt-scaled LR (PR #1234 alphonse, fresh assignment).

**Primary metric**: `val_avg/mae_surf_p` (lower is better; 4-split mean over single_in_dist, geom_camber_rc, geom_camber_cruise, re_rand).

## Active Experiments (WIP)

| PR    | Student   | Hypothesis                                                                          |
|-------|-----------|-------------------------------------------------------------------------------------|
| #1226 | frieren   | Extended training 100ep + T_max=100 + warmup=5 on FiLM+Fourier baseline             |
| #1228 | edward    | Width scaling: n_hidden=192 with T_max=75 + warmup=5 on FiLM+Fourier baseline       |
| #1216 | thorfinn  | Depth: n_layers=2 on FiLM+Fourier+warmup baseline                                   |
| #1209 | nezuko    | Lion LR sweep {1e-4,2e-4,3e-4,5e-4} on FiLM+Fourier baseline                        |
| #1202 | fern      | Width scaling: n_hidden=192 on FiLM+Fourier (older config, may overlap #1228)        |
| #1174 | tanjiro   | Extended Fourier freqs sweep L in {5,6,7,8} octaves on (x,z)                        |
| #1231 | askeladd  | Compose warmup=5 + T_max=60 single-decay (60ep) on FiLM+Fourier baseline (NEW)      |
| #1234 | alphonse  | Batch size sweep {8,16,32} with sqrt-scaled LR on FiLM+Fourier+warmup baseline (NEW)|

## Recently Closed (this advisor session)

- PR #1218 (askeladd, SWA late-epoch averaging) — EMA val_avg=36.3455, SWA val_avg=36.7199; neither beats baseline 35.8406. Confirmed under cycling cosine annealing without SWALR, EMA dominates equal-weight SWA. Mechanism analysis valuable; reassigned.
- PR #1167 (alphonse, FiLM+Fourier on best baseline) — superseded chain (now subsumed by PR #1104 / #1175 / #1208 merge sequence). Final result 38.0015 well above current baseline. Reassigned.
- PR #1232 (alphonse, surf_weight sweep {20,32,40,48}) — closed before student picked up; duplicate of PR #1173 nezuko (sw=28-32 already established as optimum on Fourier-only baseline). Reassigned to batch size sweep instead (#1234).

## Potential Next Research Directions

### Short-term (next idle students)

1. **Attack val_geom_camber_rc directly** — this split (51.66) dominates val_avg. Candidates:
   - Per-split surface-pressure normalization (separate stats for tandem-camber regime).
   - One-hot regime label injection into FiLM conditioning vector.
   - Targeted geometric data augmentation (chord scaling, camber jitter) on training cases overlapping the rc regime.
   - Higher mlp_ratio in the FiLM conditioner only (let global physics representation widen without touching backbone).
2. **SWA + SWALR (canonical recipe)** — askeladd's PR #1218 finding suggests SWA fails because cosine LR isn't restarted. The canonical fix: use `SWALR` after `swa_start` to set a flat or restart-cycle LR, then average. Would re-test with proper recipe.
3. **EMA-of-checkpoints** — average EMA snapshots from late epochs (instead of raw weights as SWA does, or instantaneous EMA as baseline does).
4. **Fourier on dsdf shape descriptor (dims 4–11)** — multi-scale shape encoding analogous to (x,z) Fourier; was attempted in PR #1169 but on Fourier-only baseline, may need re-test on FiLM frontier.

### Medium-term (if current sweeps plateau)

5. **Relative positional encoding within Physics_Attention slices** — arc-length-based RPE to explicitly model proximity along airfoil surface; targets val_geom_camber_rc.
6. **Loss reformulation**: Huber/SmoothL1 (robust to outliers in surface pressure) — requires `--loss huber` flag in train.py.
7. **Learnable random Fourier features (Tancik et al. 2020)** — replace fixed octave grid with trainable Gaussian projection, letting the network learn spatial frequencies.
8. **Mixup / CutMix on geometry** — interpolate between airfoil meshes for improved OOD generalization.
9. **Multi-task auxiliary head**: predict boundary-layer thickness or wall-shear from intermediate features as a regularizer.
10. **Test-time augmentation** — average predictions over geometric symmetries (mirror flip).

### Longer-term (if plateau persists)

11. **GNN backbone variant**: Replace Transolver with mesh-aware GNN to explicitly model connectivity.
12. **Ensemble of diverse checkpoints**: Average predictions from k models with different seeds or knob variations.
13. **Physics-informed regularization**: continuity-equation soft constraints or pressure-gradient smoothness terms.
14. **Curriculum learning**: train single-foil cases first, introduce tandem/camber regimes later.
15. **Attention temperature tuning**: learnable per-head temperature with bf16-safe clipping.

## Closed Earlier This Round (Negatives / Superseded)

- PR #1196 (frieren, T_max=50 on Fourier-only) — merged but superseded by FiLM frontier; insight about single-decay cosine adopted.
- PR #1181 (frieren, wider hidden + FiLM combined) — DECISIVE NEGATIVE: catastrophic train-loss spike at ep31; rules out wider hidden direction.
- PR #1173 (nezuko, surf_weight {28,32,40,50}) — DEAD END; sw=28-32 confirmed optimal on Fourier baseline.
- PR #1170 (fern, depth sweep n_layers {2,3} on Fourier-only) — DECISIVE NEGATIVE; depth hurts at this dataset size.
- PR #1108 (tanjiro, n_hidden width sweep on compound) — DECISIVE NEGATIVE.
- PR #1107 (nezuko, EMA decay sweep) — within noise; no clear winner.
- PR #1109 (thorfinn, log(Re×|saf|+ε) BL feature) — NEGATIVE.
- PR #1105 (fern, W_p per-channel pressure weighting) — NEGATIVE.
- PR #1103 (askeladd, slice_num sweep) — slice_num=64 Pareto-optimal on compound baseline.
