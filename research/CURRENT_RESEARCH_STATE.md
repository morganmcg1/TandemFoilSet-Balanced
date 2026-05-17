# SENPAI Research State

- **Updated:** 2026-05-17 12:15 UTC (R36 — 3 closures (warmup/batch_size/cosine all confirmed obsolete vs new n_layers=4 baseline); 3 new R36 assignments targeting new compute budget: n_hidden=144 retest, long warmup + higher peak LR, slice_num=16)
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p` down.

## Current best baseline — PR #4520 n_layers=4

**val_avg/mae_surf_p = 32.859**, **test_avg/mae_surf_p = 28.283** (PR #4520,
n_layers=4, lr=6e-4, max_grad_norm=1.0, single-seed, best epoch 45).

Per-split val: single=33.389, rc=45.420, cruise=17.257, re_rand=35.371.

**Total improvement from calibration baseline:** 143.52 → 32.859 = **-77.1%**

**CRITICAL — Noise model (PR #4440 frieren, grad-clip stack):**
- σ=0.341 on grad-clip stack; 2σ clear-win threshold: **val ≤ 32.18**
- Conservative threshold: val ≤ **32.52** (1σ below 32.859)
- Within noise: val in [32.18, 33.54]; Clear regression: val ≥ 33.54

## Round wins merged (R1–R35)

| PR | Hypothesis | val_avg | Δ |
|----|------------|--------:|---|
| ... (R1–R22 wins) | ... | 36.13 | previous history |
| **#4398** | **Gradient clipping max_norm=1.0** | **33.68** | **−6.8%** |
| **#4443** | **lr 5e-4→6e-4** | **33.353** | **−1.0%** |
| **#4520** | **n_layers=4 (compute savings)** | **32.859** | **−1.5%** — **CURRENT BASELINE** |

## Key architecture (current baseline — n_layers=4 + lr=6e-4 + grad_clip stack)

| Group | Value |
|-------|-------|
| Model | Transolver, n_hidden=128, **n_layers=4** (PR #4520), n_head=4, **slice_num=8**, **mlp_ratio=2** |
| FFN | GEGLU gating, **inner_dim=256** |
| Compile | `torch.compile(model, dynamic=True, mode="default")` |
| Conditioning | FiLM head [log_Re, AoA0, AoA1] |
| Precision | bf16 autocast |
| Optim | Schedule-Free AdamW **`lr=6e-4`**, `wd=1e-4`, `warmup=200` |
| **Grad Clip** | **`clip_grad_norm_(params, max_norm=1.0)` — PR #4398** |
| Loss | SmoothL1 (beta=0.25), surf_weight=10.0 |
| EMA | decay=0.997 |
| Compute | **~40s/epoch, 45 epochs**, peak VRAM **18.6 GB**, **798,515 params** |

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #4515 | frieren | 3-seed noise calibration of lr=6e-4 baseline (n_layers=5) | calibration | WIP — R32 (baseline changed; partial info on old config) |
| #4519 | nezuko | n_head sweep {2, 8} on lr=6e-4 (n_layers=5) | architecture | WIP — R32 (baseline changed) |
| #4522 | alphonse | weight_decay sweep {5e-5, 2e-4} on lr=6e-4 (n_layers=5) | optim/reg | WIP — R32 (baseline changed) |
| #4578 | tanjiro | n_layers=3 depth probe | architecture | WIP — R35 fresh |
| #4542 | thorfinn | LR fine sweep {5.5e-4, 6.5e-4} on n_layers=5 | optim | WIP — R33 (baseline changed) |
| **#4591** | **fern** | **n_hidden=144 retest on n_layers=4 — capacity-for-depth** | **architecture** | **WIP — R36 fresh** |
| **#4592** | **edward** | **warmup=1500 + lr=7.5e-4 on n_layers=4 — long warmup unlock higher peak LR?** | **optim** | **WIP — R36 fresh** |
| **#4593** | **askeladd** | **slice_num=16 on n_layers=4 — break rc-split structural bottleneck** | **architecture** | **WIP — R36 fresh** |

## Fully closed axes (updated for lr=6e-4 + grad_clip baseline)

| Axis | Verdict |
|------|---------|
| **n_layers** | **OPEN (partial)** — n_layers=4 MERGED (+1.5σ win, PR #4520, compute savings); n_layers=3 probe IN FLIGHT (tanjiro #4578). n_layers=6 clearly regressed. n_layers≤4 may continue trend. |
| **mlp_ratio (uniform)** | FULLY CLOSED at 2 (both old and new stack; asym placement closed too) |
| **n_head** | OPEN — closed at 4 on old stack; retesting {2, 8} on new stack (nezuko #4519) |
| **SF warmup_steps** | **FULLY CLOSED at 200** — {100, 300} both regressed on n_layers=5+lr=6e-4 stack (edward #4516 closed). Retesting interaction with HIGHER peak LR: warmup=1500 + lr=7.5e-4 on n_layers=4 (edward #4592). |
| **slice_num** | OPEN — closed at 8 on old stack (pre-clip); retesting slice_num=16 on n_layers=4 + grad-clip (askeladd #4593) |
| **weight_decay** | OPEN — closed at 1e-4 on old stack; retesting {5e-5, 2e-4} with grad_clip + lr=6e-4 (alphonse #4522) |
| **dropout (PhysicsAttention)** | **FULLY CLOSED at p=0.1** — d=0.05/0.0 both within noise vs lr=6e-4 baseline; dropout helps in-dist generalization independently of grad-clip (thorfinn #4493 closed) |
| **surf_weight (upward)** | FULLY CLOSED at 10 |
| **surf_weight (downward)** | **FULLY CLOSED at sw=10** — sw=7 regressed +2.79σ on lr=6e-4 stack; rc gain from sw=7 was absorbed by lr mechanism (fern #4444 closed) |
| **drop_path (p=0.1)** | CLOSED — clear regression on old stack |
| **EMA decay** | FULLY CLOSED at 0.997 (confirmed on both old and grad-clip stacks) |
| **lr** | 6e-4 MERGED (PR #4443); fine sweep {5.5e-4, 6.5e-4} IN FLIGHT (thorfinn #4542); cosine annealing schedule IN FLIGHT (fern #4555). |
| **n_hidden** | OPEN — was compute-bound on n_layers=5 (>56s/epoch); retesting n_hidden=144 on n_layers=4 (now ~48s/epoch, feasible) (fern #4591) |
| **grad_clip max_norm** | FULLY CLOSED at 1.0 (confirmed on grad-clip stack) |
| **β (SmoothL1)** | FULLY CLOSED on grad-clip stack — β and clip compete; uniform β best at 0.25 with clip active |
| **batch_size** | **FULLY CLOSED at 4** — batch=12 catastrophic regression on n_layers=5+lr=6e-4 (askeladd #4517). Mechanism: step count drives convergence under grad-clip, larger batches structurally disadvantaged by wall-clock budget. |
| GEGLU on attention | FULLY CLOSED — all regressed |
| Gate-activation axis | CLOSED — GEGLU > ReGLU > SwiGLU |
| FiLM family | FULLY CLOSED |
| RMSNorm | FULLY CLOSED |

## Key R36 insights

1. **Triple noise repeat at n_layers=5+lr=6e-4**: 33.353, 33.037, 33.683 — mean 33.36 ± 0.32. Independent confirmation of σ=0.34 calibration.
2. **The dominant frame is now "useful optimizer steps within budget"**: warmup/batch/cosine closures all confirm this. With grad-clip on 100% of steps, schedule shape matters less than total time at peak LR. The n_layers=4 win mechanism (compute savings) is the model for future axes.
3. **Cosine annealing on SF AdamW is structurally wrong** (fern's insight): the model is undertrained, needs MORE peak-lr time, not less. SF AdamW already provides large-early/small-late effective steps via internal averaging. Outer cosine is redundant AND harmful (truncates the productive plateau).
4. **Long warmup + higher peak LR is the next test** (edward's follow-up): if 1500-step warmup unlocks lr=7.5e-4 (previously failed at warmup=200), that confirms the SF moment-calibration mechanism behind warmup matters for peak-LR access.

## Key R35 insights

1. **Compute-savings is the dominant mechanism for depth reduction**: n_layers=4 wins not because 4 attention blocks is the right inductive bias, but because fewer layers = more epochs within 30-min budget (40s vs 48s/epoch → 45 vs 37 epochs). n_layers=6 loses for the opposite reason (57s/epoch → 32 epochs, compute-starved).
2. **All 4 val splits improved under n_layers=4**: in-dist −0.86, rc −0.21, cruise −0.47, re_rand −0.43. Unlike most previous wins which traded in-dist for OOD, this improvement is uniform — budget gain helped everywhere.
3. **7 in-flight PRs are now testing on the obsolete n_layers=5 baseline**: #4515, #4516, #4517, #4519, #4522, #4542, #4555. When they complete, compare results against new baseline (32.859). If results are near-baseline (32.859±noise), they may need rerun on n_layers=4.
4. **n_hidden=144 retest opportunity**: with n_layers=4 at 40s/epoch, n_hidden=144 might bring time back to ~48s/epoch — same budget as old baseline but with both n_layers=4 and n_hidden=144. Width-for-depth trade is unexplored at n_layers=4.

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

**In flight (8 students)**:
1. **frieren #4515** — 3-seed calibration at n_layers=5 (results give σ on obsolete config)
2. **nezuko #4519** — n_head {2, 8} on n_layers=5 (compare vs new baseline)
3. **alphonse #4522** — weight_decay {5e-5, 2e-4} on n_layers=5
4. **thorfinn #4542** — lr fine sweep {5.5e-4, 6.5e-4} on n_layers=5
5. **tanjiro #4578** — n_layers=3 depth probe (R35)
6. **fern #4591** — n_hidden=144 on n_layers=4 (R36)
7. **edward #4592** — warmup=1500 + lr=7.5e-4 on n_layers=4 (R36)
8. **askeladd #4593** — slice_num=16 on n_layers=4 (R36, rc-targeted)

**Backlog (not yet assigned)**:
- **mlp_ratio=3 retest on n_layers=4** — previously compute-bound at n_layers=5
- **Per-channel surf_weight {Ux, Uy, p separately}**
- **DropPath p=0.05 on n_layers=4 + lr=6e-4 + grad-clip stack**
- **Geometric data augmentation** (rotation/reflection of input geometry) — rc-targeted
- **Physics-informed auxiliary loss (continuity: div(u)=0)** — orthogonal physics constraint
- **Test-time augmentation (TTA)** — geometric averaging at inference
- **slice_num=4 on n_layers=4** — opposite direction from #4593, more compute saving
- **Variant attention mechanisms** — non-slice attention (full mesh attention, hierarchical)
10. **Geometric inductive bias for rc-split**: explicit edge/distance features, equivariant coordinates — high-value architectural axis for the chronic rc bottleneck (~45.6, resistant to optimizer/loss knobs)
11. **Val/test single_in_dist divergence investigation**: structural partition asymmetry confirmed across R33/R34 (dropout, surf_weight both show val regression / test improvement)
12. **Per-channel surf_weight {Ux, Uy, p separately}** — finer-grained pressure-channel control; may avoid the substitution issue with lr
13. **DropPath (stochastic depth) p=0.05** — different stochastic regularizer from dropout; never tested on grad-clip + lr=6e-4 stack
14. **Physics-informed auxiliary loss (continuity: div(u)=0)** — orthogonal physics constraint for rc-split geometry shifts
