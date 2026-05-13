# Research Ideas — 2026-05-13 10:45

Generated after grad_clip_max_norm=5.0 breakthrough (PR #2090, test=68.0957, −15.5%).

## Context

**Current best**: test_avg/mae_surf_p = 68.0957 (val=75.8431)
**Stack**: bf16 + bs=4 + accumulation_steps=2 (eff_bs=8) + Lion lr=1.5e-4 betas=(0.9,0.99) + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 + grad_clip_max_norm=5.0

**In-flight (DO NOT DUPLICATE)**:
- #2141 edward — LayerScale γ=1e-4
- #2118 frieren — Per-axis Fourier Lx=8/Ly=4
- #2117 fern — EMA decay=0.95
- #2115 alphonse — Mesh-node dropout=0.1
- #2088 askeladd — Lion lr=2.1e-4 (sqrt(2))
- #2121 nezuko — slice_num=48
- #2165 tanjiro — Grad-clip sweep (max_norm=2.0, 10.0)
- #2166 thorfinn — Cosine T_max=15 realignment

---

## Hypothesis 1: Depth Revisit — n_layers=6 with clip=5.0

**Priority: 1 (Tier 1 — most mechanistically motivated)**

### What it is
Increase Transolver depth from 5 to 6 layers with grad_clip_max_norm=5.0 in the stack.

### Mechanism
n_layers=6 was tested as PR #1862 (pre-clip) and regressed +14.7%. The diagnostic was clear: gradient instability at the added depth, not capacity saturation. The grad_clip=5.0 breakthrough directly addresses this root cause — it smooths the sign-vote direction variance that caused the instability. With clip=5.0 now the default, the previous failure mode no longer applies. The 30-min budget yields ~127 s/epoch; adding one layer increases per-epoch cost by ~15-20%, meaning ~12-13 epochs within budget. This is sufficient for convergence given the Lion optimizer's fast initial descent.

### Expected gain
−2% to −5% (60–66 range from 68.10 baseline). Higher upside because the architecture is deeper AND clip-stabilized simultaneously for the first time.

### Risk
Moderate. The n_layers=6 experiment did fail before, but under different training conditions (no clip, no grad-accum=2). If it still regresses, it provides strong evidence that depth=5 is a genuine ceiling at this width, not just a gradient-stability artifact.

### Implementation instructions
**REQUIRES DIRECT train.py EDIT** — n_layers is NOT in the Config dataclass; it is hardcoded in the model_config dict.

In `train.py`, change line ~469:
```python
# BEFORE
n_layers=5,
# AFTER
n_layers=6,
```

CLI for the run (no n_layers flag exists):
```bash
python train.py \
  --accumulation_steps 2 \
  --grad_clip_max_norm 5.0 \
  --wandb_group depth-revisit-clip
```

All other hyperparameters remain at baseline defaults (lr=1.5e-4, lion_beta1=0.9, fourier_L=8, slice_num=64).

**Verification**: confirm model param count increases from ~14.2M to ~16.8M (approx); log shows ~127s/epoch baseline → expect ~150s/epoch with n_layers=6; watch for OOM (should be safe at 96 GB VRAM).

---

## Hypothesis 2: Lion beta1=0.95 — Slower Momentum Decay

**Priority: 2 (Tier 1 — direct optimizer lever, never tested with clip)**

### What it is
Increase Lion's beta1 (momentum decay) from 0.9 to 0.95, lengthening the effective momentum half-life from 6.5 steps to 13.5 steps.

### Mechanism
The clip=5.0 breakthrough showed that smoothing the gradient direction signal before the Lion momentum buffer update is strongly beneficial (−15.5%). beta1=0.95 achieves a complementary form of direction smoothing from the inside: slower momentum decay means the buffer integrates more history before committing to a sign vote. Under grad-accum=2 with clip=5.0, the direction entering the buffer is already pre-smoothed; a longer half-life should further stabilize the direction trajectory and reduce sign-flip noise during the LR annealing phase. The EMA half-life of beta1=0.95 (13.5 steps) matches the short-horizon EMA being tested in #2117 (decay=0.95 on weights) — a coincidence that may reflect a natural scale of the problem's gradient correlation length.

### Expected gain
−1% to −3% (66–67 range). Moderate confidence. Optimizer momentum is a clean lever with no interaction with data representation or architecture.

### Risk
Low-to-moderate. Lion with beta1=0.95 may slow convergence, costing epochs within the 30-min budget. Monitor train loss at epoch 5 vs. baseline to catch stalls early.

### Implementation instructions
CLI-only change (lion_beta1 is in Config dataclass):
```bash
python train.py \
  --lion_beta1 0.95 \
  --accumulation_steps 2 \
  --grad_clip_max_norm 5.0 \
  --wandb_group lion-beta1-sweep
```

All other hyperparameters at baseline defaults.

**Falsifier**: if val_avg/mae_surf_p at epoch 8 is worse than 105 (vs. baseline's ~92 at ep 8 from W&B run 0w7kkvb8), abort and close as convergence-stalled.

---

## Hypothesis 3: Lion beta1=0.85 — Faster Momentum Turnover

**Priority: 3 (Tier 1 — completes beta1 bracketing sweep with H2)**

### What it is
Decrease Lion's beta1 from 0.9 to 0.85, shortening the effective momentum half-life from 6.5 steps to 4.3 steps.

### Mechanism
Opposite direction from H2. Under grad-accum=2, each optimizer step represents 2 micro-batch gradient accumulations. A shorter half-life means the sign vote is more responsive to recent gradient direction, which could help navigate the sharp loss landscape near foil surfaces at high Re. The interplay with clip=5.0 is different here: clip already smooths direction at the micro-batch level; beta1=0.85 says "trust the recent smoothed direction more than the historical average." This tests whether the current beta1=0.9 is over-smoothing.

### Expected gain
−0.5% to −2% (66.5–67.5 range). Similar order of magnitude to H2; one will be better than the other, providing a bracket around the optimum.

### Risk
Low. Clean lever. beta1=0.85 should converge faster per epoch, potentially reaching a better minimum within the 30-min budget despite the same epoch count.

### Implementation instructions
```bash
python train.py \
  --lion_beta1 0.85 \
  --accumulation_steps 2 \
  --grad_clip_max_norm 5.0 \
  --wandb_group lion-beta1-sweep
```

Use the same `--wandb_group lion-beta1-sweep` as H2 so both beta1 experiments are grouped in W&B.

---

## Hypothesis 4: Fourier L=10 Isotropic — Frequency Bracket Above Current Best

**Priority: 4 (Tier 2 — independent representation lever)**

### What it is
Increase the isotropic Fourier positional encoding from L=8 to L=10, adding two frequency bands uniformly across both spatial axes.

### Mechanism
Fourier L=8 was the best isotropic setting when it was tested (PR #1387, −6.42% from L=4 baseline). L=16 was closed due to aliasing on the irregular mesh — but that boundary is likely between L=12 and L=16, not at L=8. The per-axis experiment #2118 (Lx=8, Ly=4) tests asymmetric downsampling; L=10 isotropic tests symmetric upsampling from the current optimum. At L=10, space_dim = 2 + 4*10 = 42 (vs current 34 at L=8), adding 8 dimensions to the input projection. This adds marginal parameter cost (~0.3M params in the first linear layer) while potentially improving representation of near-foil pressure gradients where the current basis may underresolve.

### Expected gain
−0.5% to −2%. If L=8 is already near the aliasing knee, gain is smaller; if the knee is around L=12, this is a stepping stone.

### Risk
Low. Pure representation change, no architecture modification. Worst case is aliasing regression similar to L=16 but milder (~+2-5% vs baseline).

### Implementation instructions
CLI-only (fourier_L is in Config dataclass):
```bash
python train.py \
  --fourier_L 10 \
  --accumulation_steps 2 \
  --grad_clip_max_norm 5.0 \
  --wandb_group fourier-L-sweep
```

**Note**: space_dim will automatically be computed as `2 + 4*fourier_L = 42` in train.py's FourierFeatures setup (lines ~60-65). Verify at run start that the model prints space_dim=42.

---

## Hypothesis 5: Fourier L=12 — Aliasing Boundary Probe

**Priority: 5 (Tier 2 — forms bracket with H4)**

### What it is
Increase Fourier positional encoding to L=12, probing the upper edge of the usable frequency range before the L=16 aliasing failure.

### Mechanism
H4 and H5 form a bracketing experiment: L=10 and L=12 bracket the transition between "more frequency bands help" and "aliasing dominates." If L=10 wins and L=12 regresses, the sweet spot is between 8 and 12. If L=12 also wins, a further L=14 probe may be warranted. space_dim at L=12 = 2 + 4*12 = 50 (vs current 34). The added dimensions still fit well within n_hidden=192's input projection capacity.

### Expected gain
−0.3% to −1.5% if aliasing is not dominant; mild regression (+1-3%) if aliasing appears between L=10 and L=12.

### Risk
Low-moderate. The L=16 aliasing failure was clear; L=12 may or may not hit that boundary. Useful diagnostic regardless.

### Implementation instructions
```bash
python train.py \
  --fourier_L 12 \
  --accumulation_steps 2 \
  --grad_clip_max_norm 5.0 \
  --wandb_group fourier-L-sweep
```

Use the same `--wandb_group fourier-L-sweep` as H4 to group both runs. Run H4 first; if L=10 regresses, skip L=12.

---

## Hypothesis 6: Gradient Accumulation=4 with clip=5.0 (Mechanism Changed)

**Priority: 6 (Tier 2 — ruled-out mechanism no longer applies)**

### What it is
Test gradient accumulation=4 (eff_bs=16) now that clip=5.0 is in the stack.

### Mechanism
PR #2009 tested accum=4 WITHOUT clip and regressed +10.4% (vs accum=2 baseline). The diagnosed mechanism was "step starvation" — at eff_bs=16, micro-batch padding variance increases sign disagreement across accumulation sub-steps, corrupting the sign vote. However, that experiment was run BEFORE clip=5.0 was discovered. The clip=5.0 mechanism directly addresses sign-vote variance: by rescaling gradient directions toward unit norm before accumulation, the variance between micro-batch gradients is substantially reduced. With clip active, accum=4 is effectively applying the same direction-smoothing filter across more micro-batches. This changes the expected behavior fundamentally. Effective batch size=16 also reduces per-step variance from padding, and the extended gradient window may smooth across the high-variability Re regime.

### Expected gain
−1% to −3% if the clip-smoothing hypothesis is correct. Note: accum=4 halves optimizer steps per epoch, so this may also exhibit schedule sensitivity — should pair with T_max awareness.

### Risk
Moderate-high. The previous failure was measured and real; the mechanism hypothesis (clip changes accum=4 behavior) is plausible but unconfirmed. This is the most speculative of the 6 hypotheses. If it fails again, it provides very strong evidence that the step starvation cause is NOT gradient direction variance but something else (e.g., schedule misalignment at fewer steps/epoch).

### Implementation instructions
```bash
python train.py \
  --accumulation_steps 4 \
  --batch_size 4 \
  --grad_clip_max_norm 5.0 \
  --wandb_group accum-clip-retest
```

**Critical**: keep `--batch_size 4` (same per-GPU micro-batch as baseline). The accumulation increases effective batch size from 8 to 16, not the micro-batch size. VRAM should be fine (accum=4 does not increase peak VRAM).

**Falsifier**: if val_avg/mae_surf_p at epoch 5 is worse than the accum=2 baseline at epoch 5 from run 0w7kkvb8, close early — the mechanism hypothesis is refuted.

---

## Hypothesis 7 (Bonus): Surface-Weighted Loss Tuning — surf_weight Sweep

**Priority: 7 (Tier 2 — objective alignment lever, never swept)**

### What it is
Sweep surf_weight from the current value to 15.0 or 20.0, increasing the relative loss weight on surface nodes (which are the primary evaluation target) versus volume nodes.

### Mechanism
The primary evaluation metric is `mae_surf_p` — surface pressure MAE. The training loss is a weighted combination of surface and volume MAE in normalized space, with surface nodes upweighted by `surf_weight` (default=10.0 in Config). The current surf_weight=10.0 was set at the start of the programme and has never been swept. With the model now at test=68.10 and 4 test splits showing a 31.5-point gap between cruise (50.71) and rc (82.24), there is likely remaining capacity in objective alignment: increasing surf_weight further would push the model to prioritize surface pressure accuracy more aggressively, at the cost of some volume accuracy that is not measured in the primary metric. The risk is that over-weighting surface nodes makes it harder for the model to learn the flow field structure (which supports accurate surface prediction), but surf_weight=20 is a reasonable moderate increase to test.

### Expected gain
−0.5% to −2% on test_avg/mae_surf_p. May differentially help the rc split (82.24) which lags behind cruise.

### Risk
Low-moderate. Pure loss weight change; no architecture or optimizer modification. The loss landscape changes but the model capacity does not. If surf_weight=20 helps, it suggests the default 10.0 was under-utilizing the optimization signal from the most important nodes.

### Implementation instructions
CLI-only (surf_weight is in Config dataclass):
```bash
python train.py \
  --surf_weight 15.0 \
  --accumulation_steps 2 \
  --grad_clip_max_norm 5.0 \
  --wandb_group surf-weight-sweep
```

Try surf_weight=15.0 first. If it wins, follow up with surf_weight=20.0 in the same wandb_group. If it regresses, close.

---

## Prioritization Summary

| Rank | Hypothesis | Lever | Expected Gain | Risk | Key Mechanism |
|------|-----------|-------|---------------|------|----------------|
| 1 | n_layers=6 + clip=5.0 | Architecture depth | −2% to −5% | Moderate | Previous failure was gradient instability; clip directly removes that cause |
| 2 | Lion beta1=0.95 | Optimizer momentum | −1% to −3% | Low | Direction smoothing from inside buffer; complements clip=5.0 external smoothing |
| 3 | Lion beta1=0.85 | Optimizer momentum | −0.5% to −2% | Low | Tests opposite direction; brackets optimum with H2 |
| 4 | Fourier L=10 | Input representation | −0.5% to −2% | Low | Adds frequency bands before aliasing boundary |
| 5 | Fourier L=12 | Input representation | −0.3% to −1.5% | Low-moderate | Probes aliasing boundary; forms bracket with H4 |
| 6 | Accum=4 + clip=5.0 | Batch size / signal | −1% to −3% | Moderate-high | Previously closed WITHOUT clip; mechanism changed |
| 7 | surf_weight=15.0 | Loss alignment | −0.5% to −2% | Low-moderate | Never swept; objective alignment with primary metric |

## Decision Tree

```
START: Baseline = 68.0957
│
├── H1 (n_layers=6): 
│   ├── WINS (≤66): Merge. Follow up with n_layers=6 + beta1 sweep.
│   └── REGRESSES (>68.5): Confirms depth=5 ceiling. Move to H2/H3 only.
│
├── H2 (beta1=0.95) + H3 (beta1=0.85): Run in parallel.
│   ├── Both improve: Merge better one. Try beta1=0.92 to refine.
│   ├── One improves: Merge winner. Beta1 sweep closed.
│   └── Both regress: beta1 lever closed. Lion at 0.9 is optimal given clip.
│
├── H4 (L=10) + H5 (L=12): Run H4 first.
│   ├── L=10 wins: Run L=12.
│   │   ├── L=12 also wins: Fourier ceiling is above L=12. Try L=14.
│   │   └── L=12 regresses: Ceiling is L=10. Close Fourier sweep.
│   └── L=10 regresses: Close H5 (skip L=12). L=8 is optimal.
│
├── H6 (accum=4 + clip):
│   ├── WINS: Merge. Confirms clip changed accum=4 failure mode.
│   └── REGRESSES: Close. Step starvation is not gradient-direction variance.
│
└── H7 (surf_weight=15):
    ├── WINS: Try surf_weight=20 for further gain.
    └── REGRESSES: Close. 10.0 is optimal for this architecture/optimizer.
```

## Research State Update

**Current best explanation for remaining gap (68.10 vs theoretical floor)**:
The model is capacity-limited at depth=5 AND potentially direction-limited by the current Fourier basis. The clip=5.0 breakthrough suggests gradient direction quality was the primary bottleneck — now that it's addressed, the next gains likely come from (a) increased model expressiveness (depth, H1) and (b) better input representation (Fourier, H4/H5). The 31.5-point OOD gap between cruise and rc splits points to a secondary distribution alignment problem that surf_weight (H7) and mesh-dropout (#2115 in-flight) may partially address.

**Confidence**: Moderate-to-high on H1 mechanism (pre-clip failure was diagnosed as gradient instability; clip directly targets that). Moderate on H2/H3 (clean lever, plausible complementarity). Lower on H6 (speculative reversal of a prior failure).
