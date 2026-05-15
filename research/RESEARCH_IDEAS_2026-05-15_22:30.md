# Research Ideas — 2026-05-15 22:30

Branch: icml-appendix-willow-pai2i-48h-r3
Context: Round-3 active assignments. SOAP merged (+31.7%), EMA (nezuko, PR #3430 active), log-Re sinusoidal (frieren, PR #3415 active), SOAP LR sweep (alphonse, pending assignment). Two idle students: askeladd and tanjiro.

---

## Literature Synthesis

### SOAP / Shampoo Optimizer (2024–2025)

Key papers reviewed:
- **SOAP: Improving and Stabilizing Shampoo using Adam in the Preconditioner's Eigenbasis** (Vyas et al., 2024, arXiv 2409.11321): Original SOAP paper. Core finding: preconditioner recomputation at every step is too expensive; every-10-steps is the validated operating point. Benchmarks show SOAP matches or beats AdamW in convergence quality.
- **Benchmarking Shampoo and SOAP Optimizers** (OpenReview 2025): Empirically confirms that prior Shampoo implementations use every-10-steps recomputation and this frequency is necessary to prevent divergence. Runtime has non-linear (super-linear) dependence on model parameter count due to matrix decomposition cost.
- **Purifying Shampoo: Adaptive Eigenbasis Update Criterion** (OpenReview 2025): Proposes a QR-based criterion to decide *when* the preconditioner eigenbasis has drifted enough to warrant recomputation, rather than recomputing on a fixed schedule. Achieves comparable quality to freq=1 at 3–5x less preconditioner overhead. Key insight: the optimal recomputation frequency varies during training — early training benefits from more frequent updates; later training can amortize safely.

**Implication for our stack**: Our canonical uses `precondition_frequency=10`. The literature suggests freq=5 may give better curvature tracking at a modest cost increase (~17% extra preconditioner calls vs freq=10, negligible vs a 131 s/epoch baseline). Freq=20 tests whether we can amortize further without quality loss. The 30-minute wall-clock budget makes freq=1 risky but freq=5 is feasible.

### Gradient Clipping with Adaptive Optimizers (2024–2025)

Key papers reviewed:
- **Taming the Adamax: Gradient Clipping Corrects AdamW's Implicit ℓ∞ Norm Bias** (arXiv 2404.04454): Shows that clipping before optimizer step mitigates instability from large gradient outliers, especially in settings with heterogeneous loss scales (e.g., multi-task or weighted losses).
- **Adaptive Gradient Regularization (AGR)** (arXiv 2407.16944): Proposes layer-wise adaptive clipping thresholds based on running gradient statistics. Improves generalization on OOD splits in physical simulation benchmarks.
- **Gradient Clipping for SOAP** (community benchmarks, 2025): Empirical reports that SOAP (like AdamW) benefits from moderate gradient clipping when the loss surface has heterogeneous curvature. Typical good values: max_norm ∈ {0.5, 1.0, 2.0}.
- **No PEFT without Proper Gradient Scaling** (DeepMind, 2024): In fine-tuning settings with frozen+unfrozen parameter mixes (analogous to our surf_weight-weighted loss), gradient clipping before the optimizer step is consistently beneficial.

**Implication for our stack**: We have `surf_weight=10.0` creating a 10:1 weighting between surface and volume loss gradients. Variable mesh sizes (74K–242K) mean gradient magnitudes vary substantially by sample. No `clip_grad_norm_` is called anywhere in `train.py`. This is a concrete, identified gap.

### CFD Surrogate Landscape (2024–2025)

Papers surveyed: HiPPO, UPT (Universal Physics Transformer), FNO++, GNO variants, GINO (Geo-FNO), BENO, ONO (Ortho-Normalized), Transolver follow-ups.

Key finding: The Transolver remains competitive on irregular mesh benchmarks as of mid-2025. The main improvements over vanilla Transolver in recent CFD literature come from:
1. Better training protocols (learning rate schedule, loss weighting) — already explored
2. Geometry-aware tokenization (GINO-style adaptive mesh refinement tokens) — not yet tried
3. Multi-scale aggregation (hierarchical attention) — not yet tried
4. Optimizer improvements (SOAP over AdamW) — already merged

No paper has definitively beaten Transolver on a tandem-foil or ground-effect setting specifically.

---

## Experiment Spec 1: SOAP Preconditioner Frequency Sweep

**Assigned to**: askeladd
**Hypothesis slug**: soap-precond-freq-sweep
**Research mode**: Frontier refinement
**Mechanistic grounding**: The SOAP preconditioner recomputes the Shampoo eigenbasis at fixed intervals. Too-frequent recomputation wastes wall-clock budget; too-infrequent recomputation means the optimizer uses a stale curvature estimate that doesn't reflect current gradient geometry. Our canonical setting (freq=10) was adopted from the SOAP paper defaults without sweeping. Given the 30-minute wall-clock constraint and our 131 s/epoch baseline, freq=5 adds ~17% extra preconditioner overhead (feasible) while freq=20 tests whether amortization hurts convergence quality on our mesh distribution.

**What it is**: A controlled sweep of `precondition_frequency` ∈ {5, 10, 20} on the canonical SOAP stack, all else held fixed.

**Why it might help**: freq=10 was validated on NLP/image tasks with different gradient geometry. Our mesh distribution (overset zones, surface-vs-volume gradient weighting, variable N from 74K–242K) may have faster-evolving curvature that benefits from freq=5. The Purifying Shampoo paper shows early training needs more frequent updates — a fixed schedule may be leaving value on the table.

**Falsifying result**: If freq=5 does not improve `val_avg/mae_surf_p` relative to freq=10 control, and freq=20 regresses, then freq=10 is well-tuned and preconditioner frequency is not the bottleneck. Close and move on.

**Stop condition**: All 3 arms complete without OOM or divergence. If freq=5 arm diverges (loss > 2x control at epoch 5), report and stop that arm only.

**Implementation (changes to `train.py` only)**:

1. Add `precond_freq: int = 10` to the `Config` dataclass.
2. Pass `precondition_frequency=cfg.precond_freq` in the SOAP instantiation (replacing the hardcoded `10`).
3. Log `cfg.precond_freq` to W&B config.
4. No other changes.

**Three runs** (sequential or parallel if VRAM allows):
```
# Arm A — freq=5 (experimental, tighter curvature tracking)
python train.py \
  --optimizer soap \
  --lr 1e-3 \
  --precond_freq 5 \
  --surf_weight 10.0 \
  --wandb_group soap-precond-freq-sweep

# Arm B — freq=10 (canonical control, should reproduce baseline ~75.70)
python train.py \
  --optimizer soap \
  --lr 1e-3 \
  --precond_freq 10 \
  --surf_weight 10.0 \
  --wandb_group soap-precond-freq-sweep

# Arm C — freq=20 (more amortization, faster wall-clock)
python train.py \
  --optimizer soap \
  --lr 1e-3 \
  --precond_freq 20 \
  --surf_weight 10.0 \
  --wandb_group soap-precond-freq-sweep
```

**Expected diagnostic**: If freq=5 shows lower val loss at early epochs (3–8) vs freq=10, that supports the "stale curvature" hypothesis. If freq=20 matches freq=10, we can use higher frequency for free speed.

**Taste rubric**:
- Mechanistic grounding: 3 — targets a specific optimizer lever (preconditioner recompute rate) with direct literature support.
- Research-state value: 3 — distinguishes whether our SOAP config is under-adapting or well-tuned; useful either way.
- Execution value: 3 — trivial 1-line code change, negligible wall-clock overhead for freq=5 vs freq=10.

**Current baseline to beat**: `val_avg/mae_surf_p = 75.70` (SOAP canonical, PR #3283)

---

## Experiment Spec 2: Gradient Clipping Sweep

**Assigned to**: tanjiro
**Hypothesis slug**: grad-clip-sweep
**Research mode**: Diagnostic + frontier refinement
**Mechanistic grounding**: No gradient clipping is applied anywhere in the current `train.py`. With `surf_weight=10.0`, surface and volume gradients flow at a 10:1 ratio into the same parameter update. Variable mesh sizes (74K–242K nodes) mean gradient magnitudes vary by sample — high-Re samples with extreme pressure values (p up to 2692 in raceCar single) produce large gradient signals. This combination creates conditions where gradient spikes can degrade SOAP's preconditioner accumulation. `clip_grad_norm_` is a single line with essentially zero wall-clock cost and also produces a diagnostic `grad_norm` trace in W&B that reveals the gradient scale regime.

**What it is**: A sweep of `max_norm` ∈ {0.5, 1.0, 5.0} for `torch.nn.utils.clip_grad_norm_` inserted before `optimizer.step()`, on the canonical SOAP stack.

**Why it might help**: SOAP's Shampoo preconditioner conditions the gradient before the Adam step, but the preconditioner itself is computed from the raw (unclipped) gradient outer products. If rare large-gradient batches corrupt the preconditioner accumulation, clipping before the outer product update stabilizes the curvature estimate. The AGR paper (arXiv 2407.16944) shows this pattern specifically improves OOD generalization on physics simulation benchmarks — relevant for our `val_geom_camber_rc` and `val_re_rand` OOD splits.

**Falsifying result**: If all three clipping values produce `val_avg/mae_surf_p` within ±0.5% of the no-clip control (arm D), then gradient spikes are not a limiting factor and this direction should be closed.

**Stop condition**: All 4 arms (3 clipping + 1 no-clip control) complete. Include a no-clip control arm to measure the baseline on this exact stack.

**Implementation (changes to `train.py` only)**:

1. Add `grad_clip: float = 0.0` to the `Config` dataclass (0.0 = no clipping, preserving backward compatibility).
2. After `loss.backward()` and before `optimizer.step()`, insert:
   ```python
   if cfg.grad_clip > 0.0:
       grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
   else:
       grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
   wandb.log({"train/grad_norm": grad_norm.item()}, step=global_step)
   ```
3. Log `cfg.grad_clip` to W&B config.
4. No other changes.

**IMPORTANT**: Always use `mask` when computing loss (already done in baseline). The clipping is applied to the aggregated gradient, not inside the loss computation, so no mask changes needed.

**Four runs** (sequential, one GPU):
```
# Arm A — max_norm=0.5 (aggressive clipping)
python train.py \
  --optimizer soap \
  --lr 1e-3 \
  --grad_clip 0.5 \
  --surf_weight 10.0 \
  --wandb_group grad-clip-sweep

# Arm B — max_norm=1.0 (moderate clipping)
python train.py \
  --optimizer soap \
  --lr 1e-3 \
  --grad_clip 1.0 \
  --surf_weight 10.0 \
  --wandb_group grad-clip-sweep

# Arm C — max_norm=5.0 (light clipping)
python train.py \
  --optimizer soap \
  --lr 1e-3 \
  --grad_clip 5.0 \
  --surf_weight 10.0 \
  --wandb_group grad-clip-sweep

# Arm D — no clipping (control, should reproduce ~75.70)
python train.py \
  --optimizer soap \
  --lr 1e-3 \
  --grad_clip 0.0 \
  --surf_weight 10.0 \
  --wandb_group grad-clip-sweep
```

**Expected diagnostic**: The `train/grad_norm` trace will reveal gradient scale. If typical grad_norm > 5.0, this confirms gradient spikes are present and clipping at 1.0 is likely beneficial. If grad_norm is consistently < 0.5, then clipping at 0.5 is too aggressive and the 5.0 arm is the right operating point (or clipping doesn't help).

**Taste rubric**:
- Mechanistic grounding: 4 — identifies a specific untested gap (no clipping despite 10:1 surf/vol weighting + variable mesh sizes), backed by two papers directly applicable to physics simulation OOD settings.
- Research-state value: 4 — grad_norm logging is a diagnostic that updates our understanding regardless of whether val metric improves; OOD metric movement is directly relevant to `val_geom_camber_rc` and `val_re_rand`.
- Execution value: 4 — single-line implementation, zero wall-clock overhead, produces both a metric result and a diagnostic trace that informs all future optimizer experiments.

**Current baseline to beat**: `val_avg/mae_surf_p = 75.70` (SOAP canonical, PR #3283)

---

## Currently Active Assignments (for reference)

| Student | PR | Hypothesis |
|---------|-----|------------|
| nezuko  | #3430 | EMA weights (swa_utils) |
| frieren | #3415 | Log-Re sinusoidal embedding on SOAP stack |
| alphonse| pending | SOAP LR sweep {5e-4, 1e-3, 2e-3} |
| askeladd| this spec | SOAP preconditioner frequency sweep {5, 10, 20} |
| tanjiro | this spec | Gradient clipping sweep {0.5, 1.0, 5.0} |

---

## Research State Update

**Current best explanation for what limits progress**: The SOAP optimizer has been the single largest gain. The model architecture (Transolver with learnable temperature, 5 layers, 128 hidden) is likely not the primary bottleneck at this stage — optimizer and training dynamics are still the highest-leverage lever. The `surf_weight=10.0` creates asymmetric gradient flows that have not been characterized diagnostically.

**Ruled-out paths**:
- Fixed-temperature sweep (PhysicsAttention temperature is already learnable at 0.5)
- AoA augmentation (PR #3322 closed)
- Entropy regularization (PR #3323 closed)

**Open uncertainties**:
1. Whether SOAP's canonical precondition_frequency=10 is well-calibrated for our mesh geometry and gradient distribution
2. Whether gradient spikes from extreme high-Re samples (p up to 29,136 in raceCar single) are corrupting SOAP preconditioner accumulation
3. Whether log-Re sinusoidal (frieren) and EMA (nezuko) compound positively with SOAP

**Next discriminating experiment**: The gradient clipping sweep (tanjiro) is the highest information-per-compute experiment because it both tests a hypothesis AND produces a grad_norm diagnostic that updates the research map regardless of outcome.

**Stop condition for this direction**: If both askeladd and tanjiro show no improvement over the SOAP canonical (±0.5% threshold), the optimizer/training-dynamics neighborhood is exhausted and the next round should pivot to architecture-level changes (hierarchical attention, AMR tokenization, geometry-aware encoding).
