# SENPAI Research State

- **Updated:** 2026-05-17 07:40 UTC (R29 — alphonse 2-arm β sweep on mlp_ratio=2 stack: β=0.05 gave val=34.60/test=29.52 vs 36.13 = −1.53pt orthogonal gain; sent back for confirmation on grad-clip stack)
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p` down.

## Current best baseline — PR #4398 GRADIENT CLIPPING

**val_avg/mae_surf_p = 33.6757**, **test_avg/mae_surf_p = 29.6535** (PR #4398,
max_grad_norm=1.0, single-seed, best epoch 36).

Per-split val: single=31.858, rc=48.254, cruise=17.771, re_rand=36.820.
Per-split test: single=32.69, rc=43.66, cruise=14.47, re_rand=27.79.

**Total improvement from calibration baseline:** 143.52 → 33.68 = **-76.5%**

**CRITICAL — Noise model update (PR #4342 askeladd 3-seed confirmation):**
- 3-seed std on old (pre-clip) stack: **σ = 0.62 pts** (NOT ±5-10 as previously stated)
- PR #4282's val=36.13 was a 1.7σ favorable seed; true 3-seed mean was 37.20
- New 2σ clear-win threshold: **val ≤ 32.5** (1.2 pts below 33.68)
- Single-seed results in [32.5, 34.9] are within noise — 3-seed confirmation needed
- Grad_clip stack std unknown; probably ≤ 0.62 (more stable training = more reproducible)

## Round wins merged (R1–R28)

| PR | Hypothesis | val_avg | Δ |
|----|------------|--------:|---|
| ... (R1–R22 wins) | ... | 36.13 | previous history |
| **#4398** | **Gradient clipping max_norm=1.0** | **33.68** | **−6.8%** — **CURRENT BASELINE** |

## Key architecture (current baseline — grad_clip stack)

| Group | Value |
|-------|-------|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, **slice_num=8**, **mlp_ratio=2** |
| FFN | GEGLU gating, **inner_dim=256** |
| Compile | `torch.compile(model, dynamic=True, mode="default")` |
| Conditioning | FiLM head [log_Re, AoA0, AoA1] |
| Precision | bf16 autocast |
| Optim | Schedule-Free AdamW `lr=5e-4, wd=1e-4, warmup=200` |
| **Grad Clip** | **`clip_grad_norm_(params, max_norm=1.0)` — NEW, PR #4398** |
| Loss | SmoothL1 (beta=0.25), surf_weight=10.0 |
| EMA | decay=0.997 |
| Compute | ~49s/epoch, **36 epochs**, peak VRAM 22.6 GB, **983,871 params** |

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #4440 | frieren | 3-seed confirmation of new grad-clip stack (noise model) | validation | WIP — R28 fresh |
| #4441 | thorfinn | n_hidden=144 + grad_clip — capacity push, stable training | architecture | WIP — R28 fresh |
| #4442 | tanjiro | Asymmetric FFN last-block + grad_clip — cruise gain retest | architecture | WIP — R28 fresh |
| #4443 | edward | lr sweep {6e-4, 7.5e-4} on grad-clip stack | optim | WIP — R28 fresh |
| #4444 | fern | surf_weight {5, 7} downward sweep on grad-clip stack | loss | WIP — R28 fresh |
| #4445 | nezuko | grad_clip rate sweep {0.5, 2.0} — explore max_norm axis | optim | WIP — R28 fresh |
| #4446 | askeladd | EMA decay {0.995, 0.999} retest on grad-clip stack | optim | WIP — R28 fresh |
| #4281 | alphonse | SmoothL1 β=0.05 confirmation on grad-clip stack (R29 send-back; mlp_ratio=2 no-clip result was val=34.60/test=29.52) | loss | WIP — R29 send-back |

## Fully closed axes (updated for grad-clip stack baseline)

| Axis | Verdict |
|------|---------|
| **n_layers** | FULLY CLOSED at 5 (tested on old stack; result expected to hold) |
| **mlp_ratio (uniform)** | FULLY CLOSED at 2 (uniform); asymmetric per-block IN FLIGHT (tanjiro #4442) |
| **n_head** | FULLY CLOSED at 4 |
| **SF warmup_steps** | FULLY CLOSED at 200 (triangulated from both sides on old stack) |
| **slice_num** | FULLY CLOSED at 8 (compile kernel sweet spot, both 6 and 12 lose) |
| **weight_decay** | FULLY CLOSED at 1e-4 |
| **dropout (PhysicsAttention)** | FULLY CLOSED at 0.1 on old stack (may warrant retest on grad-clip stack) |
| **surf_weight (upward)** | FULLY CLOSED — 10 < 15 < 20 monotone regression; downward probe IN FLIGHT (fern #4444) |
| **drop_path (p=0.1)** | CLOSED — clear regression; wrong for shallow 5-block stack at constant rate |
| **EMA decay** | CLOSED at 0.997 on old stack; retest IN FLIGHT on grad-clip stack (askeladd #4446) |
| **lr (old stack)** | CLOSED at 5e-4 on old stack; retest on grad-clip stack IN FLIGHT (edward #4443) |
| **n_hidden** | OPEN — 160 compute-bound, 144 compute-bound on old stack; retesting 144 on grad-clip stack (thorfinn #4441) |
| **grad_clip max_norm** | PARTIALLY OPEN — 1.0 is the merged baseline; {0.5, 2.0} in flight (nezuko #4445) |
| GEGLU on attention | FULLY CLOSED — all regressed |
| Gate-activation axis | CLOSED — GEGLU > ReGLU > SwiGLU |
| FiLM family | FULLY CLOSED |
| batch_size=8, RMSNorm, FiLM-full | FULLY CLOSED |

## Key R28 insights (transformative round)

1. **GRADIENT INSTABILITY WAS THE HIDDEN BOTTLENECK**: Pre-clip norms 20–250 on EVERY step (p50 ~25, max ~262). Without clipping, SF AdamW second-moment estimates were constantly destabilized. The entire research programme was running a fundamentally noisy optimizer.
2. **val variance is ±0.62, NOT ±5-10pt**: askeladd's 3-seed confirmation (PR #4342) showed σ=0.62. The "±5-10pt" estimate in BASELINE.md was wrong and led to premature closures of experiments within noise.
3. **rc-split is a structural bottleneck confirmed**: grad_clip improved all splits except rc (+0.10 only). Weight-level reg, block-level reg (drop_path), and now optimizer stability don't help rc. The problem is inductive-bias/capacity.
4. **Multiple pre-grad-clip experiments may have been within noise**: The old baseline (36.13) was a 1.7σ favorable seed (true mean 37.20). Experiments landing in [36.0, 38.4] on the old stack were "within noise" — not necessarily regressions.
5. **The grad-clip win (−6.8%) suggests significant headroom**: 3-seed mean on old stack was 37.20; new baseline 33.68 = 5.6σ improvement. This is a compound gain waiting to be exploited across all retested axes.

## Potential next research directions

1. **3-seed confirmation of grad-clip stack** — IN FLIGHT (frieren #4440). Critical for noise model.
2. **n_hidden=144 + grad_clip** — IN FLIGHT (thorfinn #4441).
3. **Asymmetric FFN + grad_clip** — IN FLIGHT (tanjiro #4442).
4. **lr {6e-4, 7.5e-4} on grad-clip stack** — IN FLIGHT (edward #4443).
5. **surf_weight {5, 7} on grad-clip stack** — IN FLIGHT (fern #4444).
6. **grad_clip rate {0.5, 2.0}** — IN FLIGHT (nezuko #4445).
7. **EMA decay {0.995, 0.999} on grad-clip stack** — IN FLIGHT (askeladd #4446).
8. **SmoothL1 β=0.05 confirmation on grad-clip stack** — IN FLIGHT (alphonse #4281 R29 send-back). R28 result was val=34.60 on mlp_ratio=2 no-clip (−1.53 vs 36.13 = strong orthogonality with mlp_ratio=2). Expected ≈32.15 if β fully orthogonal to grad_clip; will close axis cleanly.
9. **rc-split structural bottleneck**: all non-architectural interventions failed; need geometric inductive bias (equivariant features, explicit geometry encoding, data augmentation)
10. **dropout retest on grad-clip stack** — closed at 0.1 on old stack; may shift with stable training
11. **warmup_steps retest on grad-clip stack** — closed at 200 on old stack; the capacity-warmup interaction may be different now
12. **n_hidden=160 on grad-clip stack** — if n_hidden=144 wins, probe even wider
13. **Batch size** — batch=8 closed on old stack; may differ with stable gradients
