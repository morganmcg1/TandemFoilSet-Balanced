# Baseline — `icml-appendix-charlie-pai2i-48h-r1`

This is the fresh-track baseline for the Charlie local-metrics arm (research tag
`charlie-pai2i-48h-r1`, advisor branch `icml-appendix-charlie-pai2i-48h-r1`,
target base `icml-appendix-charlie`).

## Current best configuration (merged as of 2026-05-17)

| Group | Value |
|-------|-------|
| Model | Transolver, `n_hidden=128`, `n_layers=5`, `n_head=4`, **`slice_num=8` (PR #4107)**, **`mlp_ratio=2` (PR #4282)**, `unified_pos=False`, **FiLM head on [log_Re, AoA0, AoA1]**, **GEGLU FFN (PR #4105)** |
| FFN width | **`mlp_ratio=2` effective** (PR #4282 fixed dead-code bug: `GEGLUBlock(hidden_dim, hidden_dim, hidden_dim=int(hidden_dim * mlp_ratio))`) — inner GEGLU projection now 256-d instead of 128-d |
| Compile | **`torch.compile(model, dynamic=True, mode="default")`** (PR #4069) — fuses FiLM affine + GEGLU gate + QKV projections; `dynamic=True` required for pad_collate variable-length batches |
| Optim | **Schedule-Free AdamW** `schedulefree.AdamWScheduleFree(lr=5e-4, weight_decay=1e-4, warmup_steps=200)` — PR #4071; NO LR scheduler |
| **Grad clip** | **`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`** after `loss.backward()`, before `optimizer.step()` — PR #4398; clip activates ~100% of steps (typical pre-clip norms 20–250) |
| Loss  | **SmoothL1 (Huber, beta=0.25)** in normalized space, `surf_weight=10.0` (PR #3400) |
| EMA   | **Polyak averaging, decay=0.997**, evaluated at val/test time (PR #3783); EMA built before compile so `ema_model.module` is uncompiled |
| Dropout | **dropout=0.1** in PhysicsAttention (attn + to_out) — PR #3402 |
| Precision | **bf16 autocast** (`torch.autocast(device_type='cuda', dtype=torch.bfloat16)`) — PR #4064 |
| FFN | **GEGLU gating** in Transolver block MLP (`FFN(x) = W2(GELU(W1a(x)) * W1b(x))`) with **inner_dim=256** (mlp_ratio=2) — PR #4105 + PR #4282 |
| Scoring | NaN-safe accumulators (PR #3279) — `torch.where` instead of `mask * err` |
| Sampler | `WeightedRandomSampler` over 3 domain groups |
| Caps  | `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MIN=30.0` (hard per-run wall clock) |
| Test  | Best-val EMA checkpoint evaluated on 4 test splits at end of run; use `load_target = getattr(model, "_orig_mod", model)` to load state dict after compile |

## Current best metrics (PR #4398, gradient clipping max_norm=1.0, single-seed, best epoch 36)

**Beat this to be a winner.**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` **(primary)** | **33.6757** |
| `test_avg/mae_surf_p` | **29.6535** |
| `test/test_single_in_dist/mae_surf_p` | 32.69 |
| `test/test_geom_camber_rc/mae_surf_p` | 43.66 |
| `test/test_geom_camber_cruise/mae_surf_p` | 14.47 |
| `test/test_re_rand/mae_surf_p` | 27.79 |

Per-split val surface-p MAE at best checkpoint (single seed, epoch 36):

| Split | mae_surf_p | Δ vs prev (36.13) |
|-------|------------|-----------|
| `val_single_in_dist`     |  31.858 | **-13.1%** |
| `val_geom_camber_rc`     |  48.254 | +0.2% (unchanged — structural bottleneck) |
| `val_geom_camber_cruise` |  17.771 | **-16.8%** |
| `val_re_rand`            |  36.820 | **-4.0%** |
| **avg** | **33.6757** | **-6.8%** |

Artifact: `models/model-charliepai2i48h1-frieren-grad-clip-1p0-20260517-055140/metrics.jsonl`

**Gradient clipping impact:**
- Pre-clip gradient norms: mean 25–70, p50 20–64, p99 100–177, max up to 262 — **across ALL epochs, ALL steps**
- Clip activates on 374–375/375 steps every epoch (near-100% activation rate)
- Without clipping, SF AdamW's second-moment EMA was being destabilized by O(100) gradient outliers on every step
- All non-rc val splits improved strongly: single_in_dist −4.81, cruise −3.60, re_rand −1.52
- rc-split essentially unchanged (+0.10): confirmed structural bottleneck, not a gradient-noise problem
- Wall-clock unchanged: ~49s/epoch (clip_grad_norm_ is negligible overhead)
- Peak VRAM unchanged: 22.6 GB

**IMPORTANT — 3-seed variance update (PR #4342 askeladd):**
The "true" 3-seed mean of PR #4282 (old baseline) was **37.20 ± 0.62 (1σ)**. PR #4282's val=36.13 was a 1.7σ favorable seed — NOT the typical draw. Tightened variance estimate:
- **Single-seed 1σ noise: ≈ 0.62 pts** (NOT ±5-10 pts as previously stated)
- **2σ "clear improvement" threshold: ~1.2 pts below baseline**
- For the new baseline val=33.68: a single-seed win requires val ≤ 32.5 to be clear of noise; anything in [32.5, 34.9] is "within noise" — needs 3-seed confirmation
- Note: 3-seed std was measured on the OLD stack; std on new stack (with grad clipping) may differ but is expected to be equal or smaller (more stable training)

**Key implementation note — gradient clipping placement:**
```python
# In the training loop, after loss.backward() and BEFORE optimizer.step():
optimizer.train()
# ... forward pass, loss computation ...
scaler.scale(loss).backward()   # if using GradScaler; otherwise just loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
optimizer.step()
optimizer.zero_grad()

# CLI flag:
parser.add_argument('--max_grad_norm', type=float, default=1.0,
                    help='Max gradient norm for clipping (0 to disable)')
```

**Key implementation notes (unchanged from PR #4282):**
```python
# After EMA wrap, before training loop:
try:
    model = torch.compile(model, dynamic=True, mode="default")
except Exception as e:
    print(f"torch.compile: FAILED, falling back to eager — {e}")

# On checkpoint reload for test eval:
load_target = getattr(model, "_orig_mod", model)
load_target.load_state_dict(torch.load(model_path, ...))

# SF mode switches still required:
optimizer.train()   # before each training step
optimizer.eval()    # before val/test (including after checkpoint load)
```

Reproduce:
```bash
cd target/
pip install schedulefree
python train.py --experiment_name grad-clip-1p0-repro --agent <name> --max_grad_norm 1.0
```

### Note on val variance

**Updated from 3-seed confirmation (PR #4342):** Single-seed 1σ noise is **≈ 0.62 pts** on `val_avg/mae_surf_p` (NOT ±5-10 pts). The 2σ clear-regression threshold is ~1.2 pts above baseline. The old ±5-10pt estimate was wrong and led to premature closures.

Decision thresholds for **this** baseline (val=33.68):
- **Clear win**: val ≤ 32.5 (≥1.2 pts below baseline — ≥2σ)
- **Within noise / soft win**: val in [32.5, 34.9] — single-seed tie, needs 3-seed confirmation to distinguish from noise
- **Clear regression**: val ≥ 34.9 (≥1.2 pts above baseline — ≥2σ)

**Total improvement from calibration baseline:** 143.52 → 33.68 = **-76.5%**

### Calibration-only baseline (PR #3107, default config MSE)

For reference — this is the un-improved baseline, not the current winner:

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | 143.52 (epoch 11, 14 epochs run) |
| `test_avg/mae_surf_p` (NaN-safe recompute) | 130.34 |

Split: `val_single=181.35, val_rc=163.47, val_cruise=105.77, val_re_rand=123.49`

## Primary ranking metric

`val_avg/mae_surf_p` for checkpoint selection; `test_avg/mae_surf_p` for
paper-facing ranking. Lower is better. Equal-weight mean of surface-pressure
MAE across the four val/test splits in physical (denormalized) units.

## How this file is updated

After every merged winner, the advisor:
1. Replaces the "Current best" block with the new PR's `val_avg/mae_surf_p`
   and `test_avg/mae_surf_p` (and the per-split surface-p MAE table).
2. Appends a one-line entry under "History" with PR #, hypothesis tag, and the
   new score.

## History

| Date | PR | Hypothesis | val_avg/mae_surf_p | Δ |
|------|----|------------|--------------------|---|
| 2026-05-15 | #3107 | baseline (MSE, default config) | 143.52 | — (calibration) |
| 2026-05-15 | #3111 | SmoothL1 loss (Huber beta=1.0) | 115.17 | -19.7% |
| 2026-05-15 | #3279 | NaN-safe scoring (infra, also re-rolls val seed) | 108.47 | -5.8% (stochastic) |
| 2026-05-15 | #3285 | EMA model weights, decay=0.999 | 104.52 | -3.6% |
| 2026-05-15 | #3280 | SmoothL1 beta=0.5 (tuned from 1.0) | 98.45 | -5.81% |
| 2026-05-15 | #3400 | SmoothL1 beta=0.25 (2-seed mean; beta lever saturated) | 97.15 | -1.32% |
| 2026-05-15 | #3402 | dropout=0.1 in PhysicsAttention (8/8 split consistency) | 96.17 | -1.01% |
| 2026-05-16 | #3533 | slice_num=64→32 (halve slice-attention cost, +2 epochs, implicit reg) | 90.58 | -5.81% |
| 2026-05-16 | #3602 | slice_num=32→16 (continue halving, +2 epochs to 18, still compute-bound) | 84.44 | -6.78% |
| 2026-05-16 | #3601 | EMA decay 0.999→0.998 (tighter window, confirmed on slice_num=16 base) | 81.16 | -3.88% |
| 2026-05-16 | #3783 | EMA decay 0.998→0.997 (probe looser; diminishing returns) | 80.88 | -0.34% |
| 2026-05-16 | #3950 | slice_num 16→12 (triangulate; tie within noise) | 80.60 | -0.34% |
| 2026-05-16 | #3982 | mlp_ratio 2→1 (halve FFN width, +1 epoch from compute saving) | 79.05 | -1.92% |
| 2026-05-16 | #4004 | FiLM-on-Re: condition each Transolver block on log(Re) scalar | 71.46 | -9.6% |
| 2026-05-16 | #4018 | FiLM-Re+AoA: expand conditioning to [log_Re, AoA0, AoA1] | 68.80 | -3.7% |
| 2026-05-16 | #4064 | bf16 autocast: -27% sec/epoch, 18→25 epochs in 30-min cap | 59.08 | -14.1% |
| 2026-05-16 | #4105 | GEGLU FFN: gating projection replaces vanilla MLP, all 4+4 splits improved 9-19% | 50.57 | -14.4% |
| 2026-05-16 | #4071 | Schedule-Free AdamW: eliminates cosine T_max fragility, all 8 splits improved 7-18% | 45.07 | -10.9% |
| 2026-05-16 | #4107 | slice_num 12→8 on bf16+GEGLU+SF: -8.7% sec/epoch → +2 epochs, 3/4 splits improved, rc-split flip from regress to win | 43.82 | -2.78% |
| 2026-05-17 | #4069 | torch.compile(dynamic=True) on bf16+GEGLU+SF+slice=8: -41.3% sec/epoch (72→42s), 25→42 epochs, all 8 splits improved | 37.31 | -14.9% |
| 2026-05-17 | #4282 | mlp_ratio=2 fix (dead-code bug: GEGLUBlock now uses int(hidden_dim*mlp_ratio)); +33.6% params, 983k, all 4 splits improved | 36.13 | -3.2% |
| 2026-05-17 | #4342 | 3-seed baseline confirmation: true mean 37.20 ± 0.62; PR #4282's 36.13 was 1.7σ favorable seed; updates noise model from ±5-10pt to ±0.62pt | N/A (analysis) | N/A |
| 2026-05-17 | #4398 | Gradient clipping max_norm=1.0: pre-clip norms 20–250 every step, ~100% activation rate; single_in_dist −13%, cruise −17%, re_rand −4%; rc unchanged (structural bottleneck confirmed) | **33.68** | **-6.8%** |
