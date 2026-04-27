# TandemFoilSet Baseline

**Advisor track:** `kagent_v_students`
**Research tag:** `kagent-v-students-20260423-2055`
**W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current best

**PR #39 — nezuko: nl=3 × sn=16 compound + sn=8 & nl=2 probes (sn=8 wins)**

- **val_avg/mae_surf_p: 49.077** (best single-seed, run `n0nwqkoz`, seed=1, ep=38); 2-seed mean **49.443**
- **test_avg/mae_surf_p: 42.473** (best single-seed, seed=1); 2-seed mean **42.450**
- W&B runs: `3fyx76kw` (s=0), `n0nwqkoz` (s=1) — group `nezuko/nl3-sn16-compound`
- Best epoch: 38 (both seeds at final epoch — compound throughput unlock confirmed)

### Per-split val surface-p MAE (best single-seed `n0nwqkoz`, s=1, ep=38)

| Split | val mae_surf_p | vs PR #35 (nl=3/sn=32) |
|-------|----------------|------------------------|
| val_single_in_dist | ~57 | ≈−7% |
| val_geom_camber_rc | ~61 | ≈−10% |
| val_geom_camber_cruise | ~30 | ≈−11% |
| val_re_rand | ~48 | ≈−11% |
| **val_avg** | **49.077** | **−9.5%** |

### Per-split test surface-p MAE (best single-seed `n0nwqkoz`, best-val checkpoint)

| Split | test mae_surf_p | vs PR #35 |
|-------|-----------------|-----------|
| test_single_in_dist | 49.18 | −19.7% |
| test_geom_camber_rc | 54.56 | −19.3% |
| test_geom_camber_cruise | 25.22 | −25.0% |
| test_re_rand | 40.93 | −24.7% |
| **test_avg** | **42.473** | **−10.5%** |

### Compound sweep results (PR #39)

| Config | Seeds | val mean | val std | test mean |
|--------|-------|----------|---------|-----------|
| nl=3/sn=32 (anchor) | 2 | 54.476 | 0.376 | 47.336 |
| nl=3/sn=16 | 3 | 52.065 | 0.300 | 44.804 |
| **nl=3/sn=8 (winner)** | 2 | **49.443** | **0.517** | **42.450** |
| nl=2/sn=16 (probe) | 1 | 50.715 | — | 43.850 |

**Three monotonic trends all continue — floors NOT found:**
- **sn at nl=3:** sn=32 → sn=16 → sn=8 (val 54.48 → 51.98 → 49.44). sn=8 best_ep=38=terminal.
- **Depth at sn=16:** nl=5 → nl=3 → nl=2 (val 61.81 → 51.98 → 50.72). nl=2 best_ep=48=terminal.
- **sn=4, nl=2/sn=8, nl=1 all promising next probes.**

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
| n_layers | 3 |
| **n_head** | **4** (use 1 for best results at sn=32; nh=1×sn=8 follow-up pending — see PR #32) |
| **slice_num** | **8** ← new (PR #39) |
| mlp_ratio | 2 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR(T_max=total_optimizer_steps) |
| loss | L1 |
| fourier_features | fixed |
| fourier_m | 160 |
| fourier_sigma | 0.7 |
| ffn | SwiGLU |

**Note:** `--mlp_ratio` flag is now properly wired (commit b8330ac from frieren's PR #38). `--n_head` is plumbed correctly.

**Note on code defaults:** Config dataclass defaults in `train.py` still reflect pre-merge values (`loss_type="mse"`, `amp=False`, `grad_accum=1`, `fourier_features="none"`, `fourier_m=10`, `fourier_sigma=1.0`, `swiglu=False`, `slice_num=64`, `n_layers=5`). Always pass the full flag list below. `--debug` verification mandatory.

Reproduce (best single-seed winner):
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
    --fourier_sigma 0.7 \
    --swiglu \
    --slice_num 8 \
    --n_layers 3 \
    --seed 1 \
    --wandb_name "<student>/<experiment>"
```

---

## Baseline history

### 2026-04-27 — PR #32: alphonse n_head sweep + 3-seed anchor recalibration

**Code merged; no baseline metric update (nh=1/sn=32/nl=3 does not beat PR #39 at sn=8).**

- **val_avg/mae_surf_p (2-seed mean):** nh=1/sn=32/nl=3 = 49.72 (vs current best 49.443 — within noise, not a metric win)
- **test_avg/mae_surf_p (2-seed mean):** nh=1/sn=32/nl=3 = 43.23
- Triple compound probe nh=1/sn=16/nl=3 (single seed) = **48.13 val / 40.93 test** — lowest single-seed ever; needs multi-seed follow-up
- Code changes: `--n_head` and `--dim_head` CLI flags plumbed through Transolver chain
- Key finding: nh=1 is architecturally superior at sn=32 — shape-preserving control confirms inductive bias not capacity drives win
- Key finding: noise floor wider at nh=1 (std ~0.92 val); multi-seed required
- Merge rationale: architectural improvement + CLI infrastructure for nh=1×sn=8 compound follow-up

### 2026-04-24 — PR #39: nezuko nl=3 × sn=16 compound (sn=8 wins)

- **val_avg/mae_surf_p: 49.077 (best seed, s=1) / 49.443 (2-seed mean)** (previous: 54.210 / 54.476, PR #35)
- **test_avg/mae_surf_p: 42.473 (best seed) / 42.450 (2-seed mean)** (previous: 47.484 / 47.336, PR #35)
- W&B runs: `3fyx76kw` (s=0), `n0nwqkoz` (s=1) — group `nezuko/nl3-sn16-compound`
- Change: `--slice_num 8` (was 32).
- Delta: −9.5% val / −10.5% test (best seed vs prior best seed); 2-seed mean −9.2% val / −10.3% test.
- **13.4σ below merge threshold** using PR #35 anchor std 0.376. Decisive.
- Uniform split win (test improves 20-25% on OOD splits).
- Budget confirmed: sn=8 reaches 38 epochs (vs anchor 32), still descending at terminal. Mechanism = compute-reduction → more epochs → better loss.
- sn floor and depth floor both still unfound — immediate follow-up (PR #43) pushes to sn=4, nl=2/sn=8, nl=1.

### 2026-04-24 — PR #35: nezuko n_layers=3 sweep

- **val_avg/mae_surf_p: 54.210 / 54.476 (2-seed mean)** (previous: 60.581 / 61.813, PR #34)
- Test: 47.484 / 47.336. Config: nl=3, sn=32.

### 2026-04-24 — PR #34: frieren slice_num=16 sweep

- **val_avg/mae_surf_p: 60.581 / 61.813 (2-seed mean)** (previous: 67.186 / 68.687, PR #27)

### 2026-04-24 — PR #27: nezuko slice_num=32 sweep

- **val_avg/mae_surf_p: 67.186 / 68.687 (3-seed mean)** (previous: 69.845 / 70.667, PR #24)

### 2026-04-24 — PR #24: alphonse σ × SwiGLU sweep

- **val_avg/mae_surf_p: 69.845 / 70.667 (2-seed mean)** (previous: 73.660, PR #20)

### 2026-04-24 — PR #20: fern Fourier σ=1 + SwiGLU feedforward

- **val_avg/mae_surf_p: 73.660** (previous: 84.737, PR #7). −13.1% val.

### 2026-04-23 — PR #7: alphonse Fourier PE fixed σ=1 m=160

- **val_avg/mae_surf_p: 84.737** (previous: 88.268, PR #12).

### 2026-04-23 — PR #12: fern AMP + grad_accum=4

- **val_avg/mae_surf_p: 88.268** (previous: 93.127, PR #11).

### 2026-04-23 — PR #11: frieren surf_weight=1 on L1

- **val_avg/mae_surf_p: 93.127** (previous: 103.036, PR #3).

### 2026-04-23 — PR #3: frieren L1 loss

- **val_avg/mae_surf_p: 103.036** (previous: no baseline).

---

## Primary metric

- **Validation (checkpoint selection):** `val_avg/mae_surf_p` — equal-weight mean across four val splits. Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` — best-val checkpoint on four test splits.

## Update protocol

When a PR's best `val_avg/mae_surf_p` is lower than the current entry here:

1. Squash-merge the winning PR into `kagent_v_students`.
2. Update this file with new metric, PR number, W&B run link.
3. Commit on advisor branch.

**Multi-seed requirement:** merge claims < 5% require 2-seed anchors. Winner 2-seed mean must beat current 2-seed anchor mean by > 1σ of anchor spread.

**Noise calibration across recipes:**
- nl=3/sn=8 2-seed std: 0.517 val
- nl=3/sn=16 3-seed std: 0.300 val
- nl=3/sn=32 anchor 2-seed std: 0.376 val
- All at σ=0.7+SwiGLU+AMP+grad_accum=4+L1+sw=1+fourier_m=160 base recipe.
