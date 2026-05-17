# SENPAI Research State

- **Updated:** 2026-05-17 08:50 UTC (R30 — closed #4441/#4442/#4281 (val regressions; β/grad_clip non-orthogonal; capacity compute-bound; asym-FFN cruise gain reversed); 3 new: #4492 per-channel β, #4493 dropout sweep, #4494 mlp_ratio=3 uniform)
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
| #4492 | alphonse | Per-channel β (β_uxy=0.05, β_p=0.25) — address p-channel tail asymmetry | loss | WIP — R30 fresh |
| #4493 | thorfinn | Dropout sweep {0.05, 0.0} on grad-clip stack — test regularization redundancy | optim/reg | WIP — R30 fresh |
| #4494 | tanjiro | mlp_ratio=3 uniform on grad-clip stack — capacity vs placement disentangle | architecture | WIP — R30 fresh |

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
8. **Per-channel β (β_uxy=0.05, β_p=0.25)** — IN FLIGHT (alphonse #4492). β/grad_clip are NOT orthogonal (R29 confirmed); per-channel approach may decouple the channel-specific residual-regime effects.
9. **Dropout sweep {0.05, 0.0}** — IN FLIGHT (thorfinn #4493). Grad_clip fires 100% of steps — likely competing with dropout for regularization. If dropout=0 wins, critical finding on over-regularization.
10. **mlp_ratio=3 uniform** — IN FLIGHT (tanjiro #4494). Disentangles capacity (more FFN width) from placement (asym-last-block). Closes the capacity axis cleanly.
11. **rc-split structural bottleneck**: grad_clip and capacity (n_hidden=144 +1.66 rc) BOTH help rc, but n_hidden=144 regresses in-dist. Asymmetric capacity for OOD-relevant features is a deep avenue.
12. **warmup_steps retest on grad-clip stack** — closed at 200 on old stack; optimizer dynamics shifted substantially (100% clip activation changes effective LR curve)
13. **n_hidden=160 on grad-clip stack** — compute-bound diagnosis from thorfinn confirmed; need bigger budget or compute-efficiency win
14. **Batch size** — batch=8 closed on old stack; unclear on grad-clip stack

## R30 critical insights

1. **β and grad_clip are NOT orthogonal**: both impose "preserve small-error gradient signal" through competing mechanisms. uniform β=0.05 + grad_clip: val=34.00 (+0.32, within noise) with test=28.72 (−0.93). The overlap is fundamental: clip normalizes ‖g‖ to 1.0 which already imposes constant-magnitude gradient direction regardless of error magnitude.
2. **Asymmetric FFN (last-block) cruise gain was a gradient-instability artifact**: pre-clip the wider readout was partially attenuating gradient noise; once grad_clip handles that directly, the gain evaporates (+0.59 cruise regression post-clip).
3. **n_hidden=144 is compute-bound NOT capacity-bound**: 33 epochs at 56s vs 37 epochs at 48s for n_hidden=128. The wider model converges slower than it can exhaust within the 30-min budget. rc improved (−1.66) despite val_avg regression — suggests capacity DOES help rc but the budget doesn't let it converge.
4. **Regularization redundancy**: grad_clip active 100% of steps at pre-clip norm 37–90 mean = extreme continuous regularization. dropout=0.1 on top may be over-constraining the model, explaining why val_single_in_dist regressed (+3.59 thorfinn, +4.18 alphonse β=0.05) while test improved.
