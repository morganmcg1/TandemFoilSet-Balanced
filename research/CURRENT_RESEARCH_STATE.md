# SENPAI Research State

- **Updated:** 2026-05-17 10:20 UTC (R32 — MAJOR: σ recalibrated 0.62→0.34 (grad-clip halves seed variance); lr 5e-4→6e-4 merged (val 33.68→33.353); 5 closures + 6 new R32 assignments)
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
| #4444 | fern | surf_weight=7 confirmation on lr=6e-4 baseline (send-back) | loss | WIP — R32 send-back |
| #4493 | thorfinn | dropout sweep {0.05, 0.0} on grad-clip stack | optim/reg | WIP — R30 ongoing |

## Fully closed axes (updated for lr=6e-4 + grad_clip baseline)

| Axis | Verdict |
|------|---------|
| **n_layers** | OPEN — closed at 5 on old stack; retesting {4, 6} on new stack (tanjiro #4520) |
| **mlp_ratio (uniform)** | FULLY CLOSED at 2 (both old and new stack; asym placement closed too) |
| **n_head** | OPEN — closed at 4 on old stack; retesting {2, 8} on new stack (nezuko #4519) |
| **SF warmup_steps** | OPEN — closed at 200 on old stack; retesting {100, 300} with lr=6e-4 (edward #4516) |
| **slice_num** | FULLY CLOSED at 8 |
| **weight_decay** | OPEN — closed at 1e-4 on old stack; retesting {5e-5, 2e-4} with grad_clip + lr=6e-4 (alphonse #4522) |
| **dropout (PhysicsAttention)** | OPEN — closed at 0.1 on old stack; retesting {0.05, 0.0} with grad_clip (thorfinn #4493) |
| **surf_weight (upward)** | FULLY CLOSED at 10 |
| **surf_weight (downward)** | BORDERLINE — sw=7 val=33.61 (1.7σ below true mean 34.18); needs confirmation on lr=6e-4 stack (fern #4444) |
| **drop_path (p=0.1)** | CLOSED — clear regression on old stack |
| **EMA decay** | FULLY CLOSED at 0.997 (confirmed on both old and grad-clip stacks) |
| **lr** | 6e-4 MERGED (PR #4443); optimum in {5e-4, 6e-4, 7.5e-4}. Fine sweep or schedule options open. |
| **n_hidden** | CLOSED — 144/160 compute-bound on both old and new stacks (>56s/epoch) |
| **grad_clip max_norm** | FULLY CLOSED at 1.0 (confirmed on grad-clip stack) |
| **β (SmoothL1)** | FULLY CLOSED on grad-clip stack — β and clip compete; uniform β best at 0.25 with clip active |
| **batch_size** | OPEN — closed on old stack; retesting {4, 12} with grad_clip + lr=6e-4 (askeladd #4517) |
| GEGLU on attention | FULLY CLOSED — all regressed |
| Gate-activation axis | CLOSED — GEGLU > ReGLU > SwiGLU |
| FiLM family | FULLY CLOSED |
| RMSNorm | FULLY CLOSED |

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
8. **dropout retest** — IN FLIGHT (thorfinn #4493, on old baseline).
9. **Geometric inductive bias for rc-split**: explicit edge/distance features, equivariant coordinates — high-value architectural axis for the chronic rc bottleneck
10. **Val/test single_in_dist divergence investigation**: why val regresses while test improves on same split
11. **LR fine sweep {5.5e-4, 6.5e-4}** — close lr axis; confirm 6e-4 is at the optimum
12. **Cosine annealing LR with SF AdamW** — schedule experiments; risky but unexplored
