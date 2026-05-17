# SENPAI Research State

- **Updated:** 2026-05-17 11:20 UTC (R34 — surf_weight axis CLOSED at sw=10 on lr=6e-4 stack (sw=7 regressed +2.79σ, rc gain REVERSED — lr absorbed the mechanism); fern assigned cosine annealing LR PR #4555)
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p` down.

## Current best baseline — PR #4443 lr=6e-4

**val_avg/mae_surf_p = 33.353**, **test_avg/mae_surf_p = 28.826** (PR #4443,
lr=6e-4, max_grad_norm=1.0, single-seed, best epoch 37).

Per-split val: single=34.25, rc=45.63, cruise=17.73, re_rand=35.80.

**Total improvement from calibration baseline:** 143.52 → 33.35 = **-76.8%**

**CRITICAL — Noise model RECALIBRATED (PR #4440 frieren, grad-clip stack):**
- 3-seed mean on GRAD-CLIP stack: **34.18 ± 0.341** (NOT ±0.62 as prev stated)
- PR #4398's val=33.68 was a −1.5σ favorable seed; true mean was 34.18
- New 2σ clear-win threshold: **val ≤ 32.67** (0.68 pts below 33.35)
- Conservative (until 3-seed of lr=6e-4 confirmed): val ≤ 33.0
- Grad-clip halved seed variance (0.62→0.34) — stable gradients = reproducible training

## Round wins merged (R1–R32)

| PR | Hypothesis | val_avg | Δ |
|----|------------|--------:|---|
| ... (R1–R22 wins) | ... | 36.13 | previous history |
| **#4398** | **Gradient clipping max_norm=1.0** | **33.68** | **−6.8%** |
| **#4443** | **lr 5e-4→6e-4** | **33.353** | **−1.0%** — **CURRENT BASELINE** |

## Key architecture (current baseline — lr=6e-4 + grad_clip stack)

| Group | Value |
|-------|-------|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, **slice_num=8**, **mlp_ratio=2** |
| FFN | GEGLU gating, **inner_dim=256** |
| Compile | `torch.compile(model, dynamic=True, mode="default")` |
| Conditioning | FiLM head [log_Re, AoA0, AoA1] |
| Precision | bf16 autocast |
| Optim | Schedule-Free AdamW **`lr=6e-4` (NEW)**, `wd=1e-4`, `warmup=200` |
| **Grad Clip** | **`clip_grad_norm_(params, max_norm=1.0)` — PR #4398** |
| Loss | SmoothL1 (beta=0.25), surf_weight=10.0 |
| EMA | decay=0.997 |
| Compute | ~48s/epoch, **37 epochs**, peak VRAM 22.6 GB, **983,871 params** |

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #4515 | frieren | 3-seed noise calibration of lr=6e-4 baseline | calibration | WIP — R32 fresh |
| #4516 | edward | warmup_steps sweep {100, 300} on lr=6e-4 | optim | WIP — R32 fresh |
| #4517 | askeladd | batch_size sweep {4, 12} on lr=6e-4 + grad-clip | optim | WIP — R32 fresh |
| #4519 | nezuko | n_head sweep {2, 8} on lr=6e-4 — attention expressiveness | architecture | WIP — R32 fresh |
| #4520 | tanjiro | n_layers sweep {4, 6} on lr=6e-4 — depth retest | architecture | WIP — R32 fresh |
| #4522 | alphonse | weight_decay sweep {5e-5, 2e-4} on lr=6e-4 | optim/reg | WIP — R32 fresh |
| **#4555** | **fern** | **Cosine annealing LR with SF AdamW — extract convergence from budget** | **optim/schedule** | **WIP — R34 fresh** |
| **#4542** | **thorfinn** | **LR fine sweep {5.5e-4, 6.5e-4} — close lr axis** | **optim** | **WIP — R33 fresh** |

## Fully closed axes (updated for lr=6e-4 + grad_clip baseline)

| Axis | Verdict |
|------|---------|
| **n_layers** | OPEN — closed at 5 on old stack; retesting {4, 6} on new stack (tanjiro #4520) |
| **mlp_ratio (uniform)** | FULLY CLOSED at 2 (both old and new stack; asym placement closed too) |
| **n_head** | OPEN — closed at 4 on old stack; retesting {2, 8} on new stack (nezuko #4519) |
| **SF warmup_steps** | OPEN — closed at 200 on old stack; retesting {100, 300} with lr=6e-4 (edward #4516) |
| **slice_num** | FULLY CLOSED at 8 |
| **weight_decay** | OPEN — closed at 1e-4 on old stack; retesting {5e-5, 2e-4} with grad_clip + lr=6e-4 (alphonse #4522) |
| **dropout (PhysicsAttention)** | **FULLY CLOSED at p=0.1** — d=0.05/0.0 both within noise vs lr=6e-4 baseline; dropout helps in-dist generalization independently of grad-clip (thorfinn #4493 closed) |
| **surf_weight (upward)** | FULLY CLOSED at 10 |
| **surf_weight (downward)** | **FULLY CLOSED at sw=10** — sw=7 regressed +2.79σ on lr=6e-4 stack; rc gain from sw=7 was absorbed by lr mechanism (fern #4444 closed) |
| **drop_path (p=0.1)** | CLOSED — clear regression on old stack |
| **EMA decay** | FULLY CLOSED at 0.997 (confirmed on both old and grad-clip stacks) |
| **lr** | 6e-4 MERGED (PR #4443); fine sweep {5.5e-4, 6.5e-4} IN FLIGHT (thorfinn #4542); cosine annealing schedule IN FLIGHT (fern #4555). |
| **n_hidden** | CLOSED — 144/160 compute-bound on both old and new stacks (>56s/epoch) |
| **grad_clip max_norm** | FULLY CLOSED at 1.0 (confirmed on grad-clip stack) |
| **β (SmoothL1)** | FULLY CLOSED on grad-clip stack — β and clip compete; uniform β best at 0.25 with clip active |
| **batch_size** | OPEN — closed on old stack; retesting {4, 12} with grad_clip + lr=6e-4 (askeladd #4517) |
| GEGLU on attention | FULLY CLOSED — all regressed |
| Gate-activation axis | CLOSED — GEGLU > ReGLU > SwiGLU |
| FiLM family | FULLY CLOSED |
| RMSNorm | FULLY CLOSED |

## Key R34 insights

1. **surf_weight and lr are substitute mechanisms, not complements**: sw=7 improved rc by −2.25 on old lr=5e-4 stack; lr=6e-4 ALONE improved rc by −2.62 without touching surf_weight. Stacking sw=7 + lr=6e-4 REVERSED the rc gain (+1.94 regression). Both knobs adjust encoder capacity allocation — when one is already optimal, the other over-corrects.
2. **rc-split bottleneck (~45.6) is now resistant to optimizer/loss knobs**: lr tuning, surf_weight tuning, and dropout tuning have all failed to crack it further. Future rc attacks need architectural changes, data augmentation, or physics-informed losses.
3. **val/test divergence persists**: sw=7 + lr=6e-4 shows val +0.95 regression while test −0.56 improvement. Consistent with structural partition asymmetry hypothesis.

## Key R33 insights

1. **Dropout axis fully closed at p=0.1**: Reducing dropout (0.05 or 0.0) yields no improvement vs the lr=6e-4 baseline. Grad-clip and dropout are complementary regularizers, not redundant — grad-clip handles gradient direction, dropout handles stochastic unit masking for in-distribution generalization. The triple-pattern (in-dist regression / rc improvement / re_rand improvement) persists across dropout reduction, confirming it is structural, not a regularization artifact.
2. **Train loss unchanged across d∈{0.0, 0.05, 0.1}**: No overfitting signature at dropout=0. The model is not memorizing under any dropout level — training is bottlenecked by optimization budget, not regularization capacity.

## Key R32 insights (transformative round)

1. **σ recalibrated**: Grad-clip halved seed variance (0.62→0.34). The new 2σ clear-win threshold is 0.68 pts below baseline. Previously "within noise" closures (val 34.0–34.5 when baseline was 33.68) were genuine 1–2σ regressions, not ambiguous noise.
2. **lr=6e-4 is the new optimum**: 2.4σ below the true lr=5e-4 mean (34.18). A +20% LR exploits stable clipped-gradient step direction signal. 7.5e-4 overshoots.
3. **The bottleneck is upstream of FFN/embedding width**: mlp_ratio=3, n_hidden=144, asym-FFN, per-channel β — ALL show same fingerprint: in-dist regresses, rc/OOD improves, val_avg close to baseline. The attention token-mixing mechanism is the constraint.
4. **val_single_in_dist vs test_single_in_dist diverge**: val regresses (−3.6 pts on average) while test stays flat or improves on the same split. Possible partition artifact or systematic difference in the val/test single_in_dist samples.
5. **RC-split structural bottleneck**: improves with nearly every added-capacity experiment (−1.66 n_hidden=144, −2.24 sw=7, −2.62 lr=6e-4) but never enough to pull val_avg down without in-dist tradeoff.

## Potential next research directions

1. **3-seed of lr=6e-4** — IN FLIGHT (frieren #4515). Critical calibration.
2. **warmup_steps {100, 300} with lr=6e-4** — IN FLIGHT (edward #4516).
3. **batch_size {4, 12}** — IN FLIGHT (askeladd #4517).
4. **n_head {2, 8} retest** — IN FLIGHT (nezuko #4519). Attention mechanism axis.
5. **n_layers {4, 6} retest** — IN FLIGHT (tanjiro #4520). Depth axis.
6. **weight_decay {5e-5, 2e-4}** — IN FLIGHT (alphonse #4522).
7. **surf_weight=7 + lr=6e-4 stacked** — IN FLIGHT (fern #4444 send-back).
8. **LR fine sweep {5.5e-4, 6.5e-4}** — IN FLIGHT (thorfinn #4542). Close lr axis.
9. **Cosine annealing LR with SF AdamW** — IN FLIGHT (fern #4555). Address still-descending-at-e37 pattern.
10. **Geometric inductive bias for rc-split**: explicit edge/distance features, equivariant coordinates — high-value architectural axis for the chronic rc bottleneck (~45.6, resistant to optimizer/loss knobs)
11. **Val/test single_in_dist divergence investigation**: structural partition asymmetry confirmed across R33/R34 (dropout, surf_weight both show val regression / test improvement)
12. **Per-channel surf_weight {Ux, Uy, p separately}** — finer-grained pressure-channel control; may avoid the substitution issue with lr
13. **DropPath (stochastic depth) p=0.05** — different stochastic regularizer from dropout; never tested on grad-clip + lr=6e-4 stack
14. **Physics-informed auxiliary loss (continuity: div(u)=0)** — orthogonal physics constraint for rc-split geometry shifts
