# TandemFoilSet Baseline

**Advisor track:** `kagent_v_students`
**Research tag:** `kagent-v-students-20260423-2055`
**W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current best

**PR #20 — fern: Fourier σ=1.0 + SwiGLU feedforward**
- **val_avg/mae_surf_p: 73.660** (lower is better)
- W&B run: `eg6i88yf` (`fern/sigma1-swiglu`)
- Best epoch: 17 (timeout-bounded at ~31 min)
- test_avg/mae_surf_p: **63.983**

### Per-split val surface-p MAE (best checkpoint, epoch 17)

| Split | mae_surf_p | vs PR #7 |
|-------|-----------|----------|
| val_single_in_dist | 81.39 | −21.7% |
| val_geom_camber_rc | 92.45 | −1.7% |
| val_geom_camber_cruise | 50.31 | −18.3% |
| val_re_rand | 70.50 | −11.2% |
| **val_avg** | **73.660** | **−13.1%** |

### Per-split test surface-p MAE (best checkpoint)

| Split | mae_surf_p | vs PR #7 |
|-------|-----------|----------|
| test_single_in_dist | 73.20 | −19.2% |
| test_geom_camber_rc | 76.82 | −7.9% |
| test_geom_camber_cruise | 44.04 | −19.0% |
| test_re_rand | 61.87 | −14.8% |
| **test_avg** | **63.983** | **−15.0%** |

### Current default config (post-merge)

| Param | Value |
|-------|-------|
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| grad_accum | 4 (effective bs=16) |
| amp | True (bf16 autocast) |
| surf_weight | 1.0 |
| epochs | 50 |
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR(T_max=total_optimizer_steps) |
| loss | L1 (abs, vol + surf_weight × surf) in normalized space |
| fourier_features | fixed |
| fourier_m | 160 |
| fourier_sigma | 1.0 |
| **ffn** | **SwiGLU** (replaces standard MLP in TransolverBlock) |

Reproduce:
```bash
cd target && python train.py \
    --agent <student> \
    --loss_type l1 \
    --surf_weight 1 \
    --amp true \
    --grad_accum 4 \
    --batch_size 4 \
    --fourier_features fixed \
    --fourier_m 160 \
    --fourier_sigma 1.0 \
    --use_swiglu \
    --wandb_name "<student>/<experiment>"
```

---

## Baseline history

### 2026-04-24 — PR #20: fern SwiGLU feedforward + Fourier σ=1.0 (σ fine-sweep)

- **val_avg/mae_surf_p: 73.660** (previous: 84.737, PR #7)
- W&B run: `eg6i88yf` (group: `fern/fourier-sigma-swiglu`)
- Change: Replaced standard GELU MLP in every TransolverBlock with SwiGLU (SiLU-gated, three projections, 2/3 hidden-width to match param count). σ fine-sweep confirmed σ=1.0 remains the optimum among completed runs; per-coordinate anisotropic σ regresses.
- Delta: **−13.1% val / −15.0% test** vs PR #7 baseline (84.737 / 75.244).
- Wins uniformly across all 4 val splits and all 4 test splits. Biggest lift on `val_single_in_dist` (−21.7%) and `val_geom_camber_cruise` (−18.3%).
- Peak VRAM: 37.8 GB (+2.9 GB vs PR #7), well within 96 GB headroom.
- Best epoch 17 vs 18 baseline — negligible wall-clock cost.
- Note: student's compound claim σ=0.7+SwiGLU at 71.49 was based on crashed W&B runs. Only the verified σ=1.0+SwiGLU result was merged; the σ×SwiGLU interaction requires a verified re-run.

### 2026-04-23 — PR #7: alphonse Fourier PE on (x,z) — fixed σ=1 m=160

- **val_avg/mae_surf_p: 84.737** (previous: 88.268, PR #12)
- W&B run: `91z1948k` (group: `alphonse/fourier-sw1`)
- Change: Random Fourier Features on (x,z) coordinates: `γ(p) = [sin(2πBp), cos(2πBp)]` with B∈R^{m×2} from N(0,σ²=1), m=160 frequencies.
- Delta: −4.0% val / −5.6% test vs PR #12.
- Note (from later PR #19 multi-seed follow-up): the m=160 seed distribution has σ ≈ 8 pts; this 84.737 baseline sits ~1σ below the config's seed-mean. Still a valid pinned-seed reference.

### 2026-04-23 — PR #12: fern throughput scaling (AMP bf16 + grad_accum=4)

- **val_avg/mae_surf_p: 88.268** (previous: 93.127, PR #11)
- W&B run: `n68w9q7o` (group: `fern/throughput-amp-sw1`)
- Change: AMP (bf16) + grad_accum=4 (eff_bs=16). +28% throughput, +5 epochs in 30-min budget.
- Delta: −5.2% val / −13.2% test.

### 2026-04-23 — PR #11: frieren fine surf_weight sweep on L1 (surf_weight=1)

- **val_avg/mae_surf_p: 93.127** (previous: 103.036, PR #3)
- W&B run: `yt7eup38` (group: `frieren/l1-surf-weight-sweep`)
- Change: surf_weight reduced from 10 → 1 under L1 loss.
- Delta: −9.62%.

### 2026-04-23 21:40 — PR #3: frieren Huber/L1 loss reformulation

- **val_avg/mae_surf_p: 103.036** (previous: no baseline on this track)
- W&B run: `w2jsabii` (group: `frieren/loss-reformulation-v2`)
- Change: L1 loss in normalized space instead of MSE.
- Delta: −21.9% vs MSE baseline.

---

## Primary metric

- **Validation (checkpoint selection):** `val_avg/mae_surf_p` — equal-weight mean across four validation splits. Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` — same quantity, computed from the best-val checkpoint on the four held-out test splits. Patched scoring.py excludes the one non-finite sample in `test_geom_camber_cruise`.

## Update protocol

When a PR's best `val_avg/mae_surf_p` is lower than the current entry here, the advisor:

1. Squash-merges the winning PR into `kagent_v_students`.
2. Updates this file with the new metric, PR number, and W&B run link.
3. Commits the update on the advisor branch.
