# SENPAI Research State

- **Updated:** 2026-05-17 06:00 UTC (R27 — 3 closures, 1 send-back, 3 new assignments; warmup axis fully closed; n_hidden softer step)
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 36.13**, **test_avg/mae_surf_p = 31.97** (PR #4282,
mlp_ratio=2 GEGLUBlock dead-code fix, best epoch 37, single-seed).

Per-split val: single=36.67, rc=48.15, cruise=21.37, re_rand=38.34.
Per-split test: single=36.53, rc=44.62, cruise=17.23, re_rand=29.50.

**Total improvement from calibration baseline:** 143.52 → 36.13 = **-74.8%**

**Note:** 3-seed baseline confirmation in flight (askeladd #4342). Single-seed variance ±5-10pt.

## Key architecture (current baseline)

| Group | Value |
|-------|-------|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, **slice_num=8**, **mlp_ratio=2** |
| FFN | GEGLU gating, **inner_dim=256** (`hidden_dim=int(128 * 2)`) |
| Compile | `torch.compile(model, dynamic=True, mode="default")` |
| Conditioning | FiLM head [log_Re, AoA0, AoA1] |
| Precision | bf16 autocast |
| Optim | Schedule-Free AdamW `lr=5e-4, wd=1e-4, warmup=200` |
| Loss | SmoothL1 (beta=0.25), surf_weight=10.0 |
| EMA | decay=0.997 |
| Compute | ~47.76s/epoch, **37 epochs**, peak VRAM 22.61 GB, **983,871 params** |

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #4399 | tanjiro | Asymmetric FFN: mlp_ratio=3 on last block only | architecture | WIP — R27 fresh |
| #4398 | frieren | gradient clipping max_grad_norm=1.0 | optim | WIP — R27 fresh |
| #4397 | thorfinn | n_hidden=144 — softer capacity step | architecture | WIP — R27 fresh |
| #4314 | edward | lr=6e-4 single-arm probe (sandwich triangulation) | optim | WIP — R27 send-back |
| #4381 | nezuko | Stochastic depth (drop_path) p=0.1 | regularization (structural) | WIP — R26 |
| #4364 | fern | surf_weight sweep {15, 20} | loss | WIP — R25 |
| #4342 | askeladd | 3-seed multi-seed baseline confirmation | validation | WIP — R25 |
| #4281 | alphonse | beta=0.1 confirm + beta=0.05 2-arm | loss | WIP — pre-R25 |

## Fully closed axes

| Axis | Verdict |
|------|---------|
| **n_layers** | FULLY CLOSED — 3 (floor), 4 (tie-within-noise), 5 (optimal), 6 (undertrained); n_layers=4+mlp_ratio=3 combo also tried and regressed |
| **mlp_ratio (FFN capacity, uniform)** | FULLY CLOSED — 2 optimal; 3 slight regression; 4 OOD overfitting; combo at n_layers=4+mlp_ratio=3 also fails. Asymmetric per-block placement IN FLIGHT (tanjiro #4399). |
| **n_head** | FULLY CLOSED — n_head=4 × dim_head=32 optimal; 8 dim_head too thin; 2 Q/K/V-params waste |
| **weight_decay** | FULLY CLOSED at 1e-4 — wd=0 helps test but hurts val_re_rand; wd=5e-4 squeezes capacity; model is capacity-limited not reg-limited |
| **SF warmup_steps** | FULLY CLOSED at 200 — 50 hurts, 100 hurts (+2.8%), 500 hurts (+3.1%). Two-sided optimum. Narrow window (~[150, 250]). |
| **slice_num** | FULLY CLOSED at 8 — slice_num=6 (+3.1%) and slice_num=12 (+3.5%) both lose. Compile kernel sweet spot. |
| EMA decay | CLOSED at 0.997 |
| **dropout (PhysicsAttention)** | FULLY CLOSED at 0.1 on mlp_ratio=2 stack — 0.05 hurts re_rand sharply (+1.91), 0.15 hurts in-dist (+1.18) |
| GEGLU on attention projections | FULLY CLOSED — all regressed |
| Gate-activation axis | CLOSED — GEGLU > ReGLU > SwiGLU |
| FiLM family | FULLY CLOSED |
| batch_size=8 | CLOSED |
| surface-weight ramp (10×) | CLOSED (ramp; static value axis IN FLIGHT — fern #4364) |
| RMSNorm | CLOSED |
| FiLM-full (11 scalars) | CLOSED |
| **n_hidden (open)** | OPEN — 160 budget-bound (val still descending −0.57/epoch at termination), retesting at 144 (thorfinn #4397) |
| **lr (open at 5e-4 lower bound)** | OPEN — 3e-4 lost (+6.6%), 7.5e-4 borderline (+1.1% val, −1.4% test, undertrained). Probing midpoint 6e-4 (edward #4314 send-back). |

## Key insights (R23-R27, inform future hypotheses)

1. **Capacity-warmup interaction has narrow optimum window**: warmup=200 optimal from both 100 and 500 directions. Wider model's clean gradients DON'T want different warmup. The benefit of warmup is steady-state SF AdamW estimator quality, not just LR ramp.
2. **Model is capacity-limited but compute-bound** (thorfinn wd + n_hidden=160 + edward lr=7.5e-4): every capacity-pushing experiment shows monotonically descending val at termination. The 30-min cap is the binding constraint, not the architecture.
3. **slice_num=8 is compile-kernel sweet spot**: both 6 (slower) and 12 (slower + longer compile warmup) pay penalties. Power-of-two slice count likely. Keep in mind for future "free width" sweeps.
4. **slice_num is routing/inductive-bias knob, not capacity**: per-slice work scales linearly but routing is bounded by n_head=4 attention. More slices → narrower training-time clustering → OOD overfitting.
5. **Camber-cruise consistently responds to wider FFN**: mlp_ratio=3 (−4.9%), n_layers=4+mlp_ratio=3 (−6.0%), n_hidden=160 (−3.2%). Asymmetric placement (tanjiro #4399) tests whether the gain survives without compute penalty.
6. **val_geom_camber_rc is the dominant val_avg bottleneck**: rc-split absolute ~48-51 vs cruise ~20 (2.5× weight in avg). Every recent weight-level intervention degrades rc. **Not dropout/wd-sensitive** — failing on something structural. Hypotheses targeting geometry-shift generalization (drop_path nezuko #4381 is first probe).
7. **val vs test asymmetry suggests selector noise**: thorfinn wd=0 and edward lr=7.5e-4 both showed test improvements with val ties/regressions. Future hypothesis: EMA last-N-epoch checkpoint averaging for selection.
8. **Compile warmup time scales with hidden_dim**: epoch 1 cost was 30s baseline, 39s slice_num=12, 94s n_hidden=160. Width changes pay an extra one-time compile cost.

## Potential next research directions

1. **gradient clipping** — IN FLIGHT (frieren #4398).
2. **n_hidden=144** — IN FLIGHT (thorfinn #4397). Softer capacity step.
3. **Asymmetric FFN last-block-3** — IN FLIGHT (tanjiro #4399).
4. **lr=6e-4** — IN FLIGHT (edward #4314 send-back). Sandwich-triangulation.
5. **drop_path p=0.1** — IN FLIGHT (nezuko #4381). Structural reg for rc-split.
6. **surf_weight sweep {15, 20}** — IN FLIGHT (fern #4364).
7. **SmoothL1 beta {0.1, 0.05}** — IN FLIGHT (alphonse #4281).
8. **3-seed baseline confirmation** — IN FLIGHT (askeladd #4342).
9. **EMA checkpoint averaging (last-N-epoch)** — alternative selection; val/test asymmetry signal. Cheap, no training change.
10. **Geometric data augmentation for rc-split** — direct attack on dominant bottleneck. Mirror flip / scale / small rotation.
11. **First-block placement** of asymmetric FFN — if last-block fails on tanjiro #4399.
12. **n_hidden=144 + lr=6e-4 combo** — compound if both win individually.
13. **Decoupled attn_dropout vs proj_dropout** — finer dropout decomposition.
14. **EMA decay retest on mlp_ratio=2 stack** — last confirmed on old broken stack.
