# Research Ideas — 2026-05-16 07:45

## Sanity Check: H46 (tanjiro) and H47 (nezuko)

**H46 — n_head=1 (head_dim=128).** Clean extrapolation of the confirmed monotone 8→4→2 MAE trend. The mechanism (fewer heads forces each head to integrate wider spatial context before projection) is consistent, and the next step is logical. Minor risk: at head_dim=128 the single attention map may over-concentrate on a narrow spatial scale and lose multi-scale coverage. Nonetheless, it is the right next test — cheap, discriminating, directly tests whether the trend continues or saturates. Approved.

**H47 — cosine eta_min sweep (1e-5, 5e-5).** Genuinely orthogonal to T_max work. A nonzero LR floor prevents the final epochs from undershooting in normalized space where pressure residuals are small. Concretely: with T_max=20 and eta_min=0 the model spends the last ~4 epochs near a numerically zero LR, which may cause frozen weights after the best checkpoint. Testing eta_min=1e-5 vs 5e-5 directly addresses this. Approved.

---

## H48 — GEGLU FFN Activation (Priority 1)

**Hypothesis.** Replacing GELU with GEGLU in all Transolver FFN blocks improves surface pressure MAE by providing gated gradient flow, which better handles the spatially heterogeneous pressure gradients in overset-mesh CFD fields.

**Mechanism.** GEGLU computes `(xW + b) ⊙ σ(xV + c)` — the sigmoid gate multiplies the linear and nonlinear paths element-wise. For pressure fields where near-wall gradients are orders of magnitude larger than far-field values, the gate learns to suppress or amplify per-neuron, giving the FFN finer spatial selectivity without widening n_hidden. A 2025 Mines Paris paper validated this mechanism specifically for mesh-based CFD surrogate learning.

**Arms.**
- Arm A: GEGLU (replace nn.GELU with GEGLU in FFN, keep n_hidden=128)
- Arm B: SwiGLU (same structure, sigmoid replaced by SiLU gate) as diagnostic variant

Both arms: lr=1e-3, wd=5e-5, T_max=20, n_head=2, clip_grad_norm=1.0, FiLM cond_dim=11, Huber δ_vel=0.5/δ_p=0.25. Match the H38 wd best config exactly, changing only the activation.

**Predicted val_avg.** 64–65.5. Architecture improvements stack with the existing optimizer/schedule wins.

**Risk.** GEGLU doubles FFN weight count (two projection matrices), adding ~15% overhead in FFN layers only. Weight initialization must be symmetric (use standard nn.Linear init for both gates). If Arm A diverges, check init scale.

---

## H49 — Lion Optimizer (Priority 3)

**Hypothesis.** Sign-based gradient updates (Lion) act as implicit per-parameter gradient normalization, potentially better handling the order-of-magnitude Re variation that causes uneven gradient magnitudes across Ux/Uy/p channels.

**Mechanism.** Lion computes `sign(β₁·m + (1-β₁)·g)` and clips each update to {-1, 0, +1}. This discards gradient magnitude information, forcing the optimizer to treat each parameter update identically regardless of loss scale. For mixed-Re batches where high-Re samples produce 10× larger gradients, this prevents high-Re samples from dominating weight updates in early training.

**Arms.**
- Arm A: Lion lr=1e-4, wd=1e-3, β₁=0.9, β₂=0.99 (standard Lion config)
- Arm B: Lion lr=2e-4, wd=5e-4, β₁=0.95, β₂=0.99 (higher momentum variant)

All other hyperparams from H38 best config. Note: Lion requires 3–10× LR reduction vs AdamW.

**Predicted val_avg.** 64–67 depending on LR sensitivity. High variance.

**Risk.** Sign-based updates are known to underperform AdamW on regression tasks vs language modeling. LR tuning is non-trivial. If both arms regress beyond 68, close immediately — regression data may not benefit from sign clipping.

---

## H50 — WSD Trapezoidal Schedule (Priority 2)

**Hypothesis.** A warmup-stable-decay trapezoidal schedule outperforms cosine by spending more training iterations at peak LR, allowing the model to consolidate learning before a sharp linear cooldown.

**Mechanism.** Cosine schedules decay immediately from epoch 1, wasting capacity in early epochs. WSD holds lr=1e-3 constant for a stable phase, then applies a linear cooldown. At 14-epoch budget: 2ep cosine warmup → 8ep stable at lr=1e-3 → 4ep linear decay to 0. Theory from Inria/EPFL 2025 shows WSD converges at the same rate as cosine with identical compute, but the stable phase enables the model to escape saddle points without the immediate LR pressure of cosine. Orthogonal to H47 eta_min work.

**Arms.**
- Arm A: WSD 2/8/4 split (2ep warmup, 8ep stable, 4ep linear cooldown)
- Arm B: WSD 1/9/4 split (shorter warmup, longer stable) — tests whether warmup length matters

Both arms: lr=1e-3, wd=5e-5, n_head=2, all other H38 best config hyperparams.

**Predicted val_avg.** 64–66. Schedule experiments have historically shown 1–2 MAE improvement in this stack.

**Risk.** H43 (warmup WIP) partially overlaps in hypothesis space. If H43 finds that extended warmup helps, WSD Arm B provides a direct comparison. If H43 finds warmup length is neutral, that is evidence for WSD Arm A (shorter warmup, longer stable).

---

## Priority Ranking

1. **H48 (GEGLU)** — Architectural, mechanism validated on mesh-CFD data directly, stackable with existing optimizer/schedule wins, low implementation risk. Highest confidence.
2. **H50 (WSD)** — Schedule-orthogonal, theory-grounded, two arms provide cross-diagnostic value against H43 warmup results. Medium-high confidence.
3. **H49 (Lion)** — Bold swing with plausible mechanism but known regression task risk. Assign after H48/H50 land results, or to a student with a spare slot.

Baseline to beat: val_avg/mae_surf_p = 66.1060 (PR #3629, H37b config with wd=5e-5 from H38).
