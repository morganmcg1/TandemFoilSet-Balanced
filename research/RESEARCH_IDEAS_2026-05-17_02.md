# SENPAI Research Ideas — 2026-05-17 (Cycle 35, Advisor Round 2)

Generated after reviewing all prior PRs (#3440–#4292), the full experiment log,
`program.md`, `CURRENT_RESEARCH_STATE.md`, and `train.py` code.

Current best: H95 Arm A — val_avg/mae_surf_p = 40.5066 / test = 39.0160 (PR #4215)
Noise floor: 2σ = 1.67 pts. Pass threshold (conclusive win): val < 38.84.

---

## Idea 1: Per-Sample Loss Normalization by GT Pressure Std

**Title:** Dynamic per-sample loss scaling by ground-truth surf_p std

**Hypothesis:** Dividing each sample's pressure Huber loss by its per-sample GT pressure std before averaging eliminates the 40x dynamic-range problem (low-Re samples ~50 Pa std, high-Re ~2077 Pa std), giving equal gradient contribution per sample regardless of Reynolds number.

**Why (mechanism):** The current loss aggregates Huber residuals over all surface nodes in a batch, raw. At high Re, GT surf_p std is 10-40x higher than low-Re samples, so a single high-Re sample can contribute 100-1600x more gradient signal. This biases the model toward accuracy on high-Re cases at the expense of the primary metric across all four splits — including `val_geom_camber_rc` (54.51) and `val_re_rand` (42.43), both of which are likely Re-diverse. H8 attempted this in R1 under AdamW on a much less-tuned base (before Lion, before slice=96, before bf16). The mechanism is orthogonal to all current locked levers.

**Why not already ruled out:** H8 was WIP/partial in R1 (pre-Lion, pre-GEGLU, pre-slice=96) and was never retested on the current stack. Lion's sign-update ignores loss magnitude for the step direction anyway, but the loss scale still affects the gradient before the sign is taken — normalization may sharpen the gradient signal per batch.

**Estimated implementation complexity:** S (student adds a ~5-line block in the per-sample loss computation inside the training loop, scaling each sample's surf_loss by `1.0 / y[b][is_surface[b], 2].std().clamp(min=1.0)` before summing over the batch).

**Expected risk/reward:** Medium risk (Lion's sign-update already partially mitigates scale variance; the gain may be small), high reward if the 40x dynamic range is a real bottleneck for low-Re split generalization.

**Concrete first-arm spec:**

```bash
# Arm A: per-sample surf_p std normalization ON
python train.py --epochs 50 \
  --experiment_name h104-arm-a-sample-norm \
  --agent <student> \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.997 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --use_bf16 \
  --per_sample_surf_norm  # new flag — student adds this
```

- Expected wall time: ~85.6 s/epoch * 21 epochs = ~30 min (same as H95 baseline)
- Pass criterion: val_avg/mae_surf_p < 40.51 (any improvement over baseline)
- Conclusive win: val < 38.84 (Δ > 1.67 pts, outside noise floor)
- Falsifying result: val ≥ 40.51 (regression or within noise)

---

## Idea 2: SWA (Stochastic Weight Averaging) over Last K Epochs

**Title:** Stochastic Weight Averaging over last 5–7 bf16 epochs

**Hypothesis:** Averaging model weights uniformly over the final 5–7 epochs of the 21-epoch bf16 training run finds a flatter loss basin with better OOD generalization than any single checkpoint.

**Why (mechanism):** EMA (H24/H65) failed because the model was still rapidly improving in early-training — EMA's lagged shadow was strictly worse than the live model. SWA targets the opposite regime: once the cosine LR schedule is near its trough (epochs 15–21), the model oscillates around a local minimum. Weight-averaging inside this oscillation finds the basin center, which is known to generalize better (Izmailov et al. 2018, Cha et al. 2021). PyTorch's `torch.optim.swa_utils.AveragedModel` is a built-in utility. H95 best epoch was 17 — with T_max=21 (from H99), epochs 17–21 are in the low-LR regime where SWA should work. There are no conflicts with any current locked lever.

**Why not already ruled out:** EMA was closed as wrong-regime; SWA was proposed as a follow-up (H28 in R1 logs) but the experiment history shows no confirmed SWA result under the current Lion+bf16+slice=96+β₂=0.997 stack. The mechanism is distinct from EMA.

**Estimated implementation complexity:** S (student calls `swa_model = AveragedModel(model)` and calls `swa_model.update_parameters(model)` each epoch after epoch `start_swa_epoch`; BN update not needed since model has LayerNorm; evaluate `swa_model` for the final val/test pass).

**Expected risk/reward:** Low risk (the run costs nothing extra — SWA runs alongside normal training and averages the weights at the end), moderate-to-good reward (basin center is reliably better for OOD splits in the literature).

**Concrete first-arm spec:**

```bash
# Arm A: SWA over last 7 epochs (epochs 15–21 of a 21-epoch run)
python train.py --epochs 50 \
  --experiment_name h105-arm-a-swa-last7 \
  --agent <student> \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.997 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --use_bf16 \
  --swa_start_epoch 15 \
  --T_max 21  # assuming H99 is merged by then; if not, add --T_max 21 manually
```

- Expected wall time: ~30 min (21 epochs at 85.6 s/epoch, same as H95)
- Pass criterion: val_avg/mae_surf_p < 40.51
- Conclusive win: val < 38.84
- Falsifying result: SWA-averaged model ≥ best single-checkpoint (regression vs H95)

---

## Idea 3: Fourier Positional Encoding for Node (x,z) Coordinates

**Title:** Random Fourier Features for mesh node position encoding in preprocess MLP

**Hypothesis:** Replacing the raw normalized (x,z) 2D coordinates fed into the preprocess MLP with random Fourier features `[sin(Bx), cos(Bx)]` (dim ~32–64) gives the model richer high-frequency spatial inductive bias, improving accuracy on fine surface-pressure gradients near foil leading/trailing edges.

**Why (mechanism):** The preprocess MLP `MLP(fun_dim + space_dim=2, n_hidden*2, n_hidden)` currently receives raw (x,z). Neural networks with ReLU/GELU activations are known to have spectral bias toward low frequencies — they struggle to represent fine-scale spatial variation (Tancik et al., Fourier Features, NeurIPS 2020). In CFD, surface pressure has strong high-frequency variation near stagnation points, separation regions, and leading/trailing edges. Fourier features expand (x,z) to e.g. 64-dim sinusoidal encodings at random frequencies (sampled from N(0,σ²) with σ~1–10), directly feeding the spatial information needed. This technique has been validated in NeRF, Physics-Informed Neural Networks (Sitzmann et al., Wang et al.), and mesh-based PDE surrogates. It is orthogonal to all optimizer/architecture locks.

**Why not already ruled out:** No Fourier feature experiment appears anywhere in the experiment log. The preprocess MLP code shows `space_dim=2` raw coordinates as direct input. This is an unexplored technique on this model.

**Estimated implementation complexity:** S-M (student adds a `FourierEmbedding(in_dim=2, out_dim=32, sigma=5.0)` module before the preprocess MLP, samples B matrix once at init with `torch.randn`, updates `space_dim` from 2 to 64, passes through sin/cos). The key question is whether to keep (x,z) alongside the Fourier features or replace them entirely — include both for safety.

**Expected risk/reward:** Medium risk (the model already has the spatial coordinates and may learn equivalent representations), high reward (Fourier features are known to materially accelerate convergence and improve fine-scale accuracy in mesh/PDE settings at no extra compute).

**Concrete first-arm spec:**

```bash
# Arm A: Fourier features sigma=5.0, out_dim=32 (total space_dim 2 + 64 = 66 → preprocess MLP input 88)
python train.py --epochs 50 \
  --experiment_name h106-arm-a-fourier-pe-sigma5 \
  --agent <student> \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.997 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --use_bf16 \
  --fourier_pe_dim 32 --fourier_pe_sigma 5.0  # new flags

# Arm B: sigma=2.0 (tighter frequency spectrum, more suitable for mesh-scale variation)
python train.py --epochs 50 \
  --experiment_name h106-arm-b-fourier-pe-sigma2 \
  --agent <student> \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.997 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --use_bf16 \
  --fourier_pe_dim 32 --fourier_pe_sigma 2.0
```

- Expected wall time: ~30 min per arm (21 epochs; parameter count increase negligible for n_hidden=128)
- Pass criterion: val_avg/mae_surf_p < 40.51 on at least one arm
- Conclusive win: val < 38.84
- Falsifying result: both arms ≥ 40.51

---

## Idea 4: Auxiliary Regression Task — Reynolds Number Prediction

**Title:** Auxiliary Re-prediction head as a physics-regularized training signal

**Hypothesis:** Adding a lightweight auxiliary MLP head that predicts log(Re) from the volume (non-surface) token pool acts as a physics-regularization signal, forcing the model to maintain Re-specific latent representations and improving generalization on the Re-diverse val_re_rand split.

**Why (mechanism):** The model conditions on log(Re) via FiLM (cond_dim includes it), but FiLM is an affine modulation — it cannot force the node embeddings themselves to encode Re information. An auxiliary Re-regression head attached to the last TransolverBlock's volume-node outputs (mean-pooled over non-surface nodes) adds a gradient signal that must pass through the full attention stack and preprocess MLP. This is analogous to multi-task learning in protein structure prediction (AlphaFold2's pLDDT head) and has been used in weather forecasting surrogates to improve generalization across forcing regimes. val_re_rand is currently our second-weakest split at 42.43 — it tests exactly the Re-generalization axis this auxiliary task targets.

**Why not already ruled out:** No auxiliary task experiment appears in the log. The mechanism is distinct from the main task and targets an identified weak split.

**Estimated implementation complexity:** S-M (student adds `AuxHead = nn.Linear(n_hidden, 1)` at the end of the forward pass, mean-pools over `fx[~is_surface]`, computes MSE against `log_re`, adds `aux_weight * aux_loss` to main loss. The key hyperparameter is `aux_weight` — suggest 0.01 to avoid dominating the main loss).

**Expected risk/reward:** Medium risk (auxiliary tasks can hurt if the Re signal from FiLM is already sufficient and the auxiliary gradient interferes), good reward if Re generalization is a real bottleneck — val_re_rand is 4.37 pts above the best split, suggesting Re coverage is a real issue.

**Concrete first-arm spec:**

```bash
# Arm A: aux_weight=0.01, Re regression on volume node pool
python train.py --epochs 50 \
  --experiment_name h107-arm-a-aux-re-0p01 \
  --agent <student> \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.997 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --use_bf16 \
  --aux_re_weight 0.01  # new flag — student adds this

# Arm B: aux_weight=0.05 (stronger regularization)
python train.py --epochs 50 \
  --experiment_name h107-arm-b-aux-re-0p05 \
  --agent <student> \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.997 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --use_bf16 \
  --aux_re_weight 0.05
```

- Expected wall time: ~30 min per arm (minimal overhead from auxiliary head)
- Pass criterion: val_avg/mae_surf_p < 40.51 AND val_re_rand improves
- Conclusive win: val < 38.84
- Falsifying result: val ≥ 40.51 or val_re_rand does not improve while val_avg improves (would suggest auxiliary task hurts Re generalization despite main task gains)

---

## Idea 5: Geometry-Aware Data Augmentation (AoA Jitter + Re Jitter)

**Title:** Training-time input perturbation — AoA ±0.5° and log(Re) ±0.05 Gaussian noise

**Hypothesis:** Adding small Gaussian noise to the AoA and log(Re) input features (dims 13, 14, 18 of x) during training acts as a data augmentation that forces the model to represent smoother, more robust mappings along the Re and AoA axes, improving OOD generalization on `val_geom_camber_rc` (hardest split, 54.51) and `val_re_rand` (42.43).

**Why (mechanism):** The training corpus has only 1499 samples across 3 domains. For Re-axis generalization, the model sees a finite set of Re values and must interpolate. Adding small noise (σ_logRe = 0.05, σ_AoA_rad = 0.009 ≈ 0.5°) during training regularizes the model along these continuous axes without changing the geometry or topology of the mesh. This is analogous to `SpecAugment` in speech or `MixupRe` in weather forecasting surrogates — augmenting along the continuous conditioning axes that generalization splits probe. Importantly, the augmentation is applied only to the conditioning features (13, 14, 18), not to the positional features (0–3, 4–11) or the binary flags (12), keeping the mesh geometry intact. The ground truth labels (Ux, Uy, p) remain unchanged — this is a feature-level augmentation, not a label augmentation.

**Why not already ruled out:** Mixup (H55) was closed, but that was label-space mixup of full samples — a fundamentally different mechanism. Feature-level perturbation along continuous conditioning axes is unexplored and closely targets the identified OOD axes.

**Estimated implementation complexity:** XS-S (student adds ~10 lines in the training loop: `if training: x[:, :, [13, 14, 18]] += torch.randn_like(x[:, :, [13, 14, 18]]) * sigma`, where sigma is set in the normalized feature space. Key detail: the noise amplitude must be in normalized units — student must read stats.json and scale sigma accordingly).

**Expected risk/reward:** Low risk (perturbation is small, ground truth unchanged, reversible), good reward if Re/AoA axis interpolation is the OOD bottleneck for the hardest splits.

**Concrete first-arm spec:**

```bash
# Arm A: AoA jitter σ=0.5° + log(Re) jitter σ=0.05 (in physical units; normalize for actual training)
python train.py --epochs 50 \
  --experiment_name h108-arm-a-aoa-re-jitter \
  --agent <student> \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.997 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --use_bf16 \
  --aoa_jitter_deg 0.5 --re_jitter_sigma 0.05  # new flags
```

- Expected wall time: ~30 min (negligible overhead)
- Pass criterion: val_avg/mae_surf_p < 40.51, with val_geom_camber_rc and/or val_re_rand improving
- Conclusive win: val < 38.84
- Falsifying result: val ≥ 40.51 or val_geom_camber_rc/val_re_rand do not improve despite val_single_in_dist possibly improving (would suggest augmentation disrupts in-distribution accuracy without helping OOD)

---

## Idea 6: Separate Surface / Volume Normalization Statistics

**Title:** Split normalization stats: compute y_mean/y_std separately for surface vs volume nodes

**Hypothesis:** Computing global y_mean and y_std from all mesh nodes mixes surface (high-pressure-gradient, small population) and volume statistics, under-representing the surface distribution that the primary metric evaluates; using separate normalization for surface vs volume nodes lets the loss weighting and Huber δ thresholds operate in well-scaled spaces.

**Why (mechanism):** Surface nodes are a minority (surface nodes are masked and separately reported). The global y_std is dominated by volume nodes, which likely have different distribution shapes than surface nodes (where separation bubbles, stagnation, and leading-edge suction create extreme local pressure). If global y_std over-estimates the typical surface pressure magnitude, then the loss computed in normalized space over-compresses surface pressure residuals. Per-subset statistics would ensure the 10× surf_weight and Huber δ_p=0.25 operate in correctly-calibrated normalized units. This is a data representation fix, not an architecture change.

**Why not already ruled out:** No normalization-split experiment appears in the log. The mechanism is distinct from surf_weight (which re-weights loss contributions) and Huber δ (which clips tails) — this fixes the scale of the normalized space itself.

**Estimated implementation complexity:** M (student must precompute surface/volume statistics from the training data, modifying how y_norm is computed — surface nodes normalized by surface stats, volume nodes by volume stats. The scoring.py contract uses global stats for denormalization and is read-only, so the student must ensure the model outputs unnormalized (physical space) predictions or re-compose at eval time. This requires careful attention to the scoring contract).

**Expected risk/reward:** Medium risk (implementation requires careful handling of the scoring.py denormalization contract which is read-only; bug risk is higher), high reward if surface/volume distribution mismatch is a real calibration problem.

**Concrete first-arm spec:**

```bash
# Arm A: separate surface/volume normalization stats
python train.py --epochs 50 \
  --experiment_name h109-arm-a-surf-vol-norm \
  --agent <student> \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.997 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --use_bf16 \
  --split_normalization  # new flag; student must handle scoring.py contract carefully
```

- Expected wall time: ~30 min
- Pass criterion: val_avg/mae_surf_p < 40.51
- Conclusive win: val < 38.84
- Falsifying result: val ≥ 40.51 or NaN in metrics (scoring contract broken)
- Key diagnostic: compare `{split}/surf_loss` vs `{split}/vol_loss` between H95 and H109 — if surface_loss drops more than vol_loss, the mechanism is confirmed

---

## Idea 7: Higher-Capacity Width Probe at bf16 (n_hidden=256)

**Title:** Width capacity probe: n_hidden=256 (double current) under bf16

**Hypothesis:** The current model with n_hidden=128 (662K params) is width-starved; doubling to n_hidden=256 increases model capacity by ~4x (attention and FFN scale quadratically with n_hidden), and with bf16's 30% throughput gain the run reaches ~14 epochs before wall-cut — enough to determine whether the quality gap closes.

**Why (mechanism):** H100 already probes n_hidden=192 (active, PR #4276). This idea targets n_hidden=256 as a bolder bet — the model has 96 GB VRAM available, uses only 30.46 GB currently, and VRAM scales quadratically with n_hidden (dominant term is QKV projections over N_max=242K nodes per layer). At slice=96 and n_layers=4, the attention computation is over 96 slice tokens, not N nodes, so the VRAM hit is manageable. The question is whether the learning bottleneck is representational capacity or data — n_hidden=256 provides a clean discriminating experiment.

**Why not already ruled out:** H100 tests 192; H86 tested 192 at fp32 and was wall-cut. n_hidden=256 has not been probed. H100 and H101 are both active (H100=192, H101=n_layers=5) — this experiment extends the capacity sweep to a point beyond both.

**Note:** H100 (n_hidden=192) must be resolved before this is assigned; if H100 shows diminishing returns at 192, this provides a strong data point; if H100 wins, n_hidden=256 is the natural next step.

**Estimated implementation complexity:** XS (student adds `--n_hidden` as a CLI arg to Config and passes it into model_config; hardcoded `n_hidden=128` at line 621 must be replaced with `cfg.n_hidden`. H100 student may have done this already — reuse their implementation).

**Expected risk/reward:** Medium risk (could OOM if VRAM estimate is off; students should add a dry-run VRAM check first), high reward (capacity improvements are often the cleanest wins once compute budget allows).

**Concrete first-arm spec:**

```bash
# Arm A: n_hidden=256, all else at H95 baseline
python train.py --epochs 50 \
  --experiment_name h110-arm-a-nhidden256 \
  --agent <student> \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.997 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --use_bf16 \
  --n_hidden 256  # requires H100's CLI arg implementation
```

- Expected wall time: ~30 min (fewer epochs than 192 due to larger model — VRAM check needed)
- Pass criterion: val_avg/mae_surf_p < 40.51 AND no OOM
- Conclusive win: val < 38.84
- Falsifying result: val ≥ 40.51 or OOM (VRAM ceiling hit)
- Pre-condition: H100 (n_hidden=192) result available to gate decision

---

## Idea 8: Surface-Node Pressure Residual Head (H16 Revival)

**Title:** Dedicated pressure residual output for surface nodes only, on top of current p output

**Hypothesis:** Adding a thin 2-layer MLP head that takes the last TransolverBlock's surface-node embeddings and predicts a pressure correction (residual added to the main `p` output) gives the model a specialised computation path for surface pressure, directly targeting the primary metric.

**Why (mechanism):** The current architecture uses a single `mlp2(ln3(fx))` output head in the last TransolverBlock for all nodes and all 3 channels. Surface pressure is the primary metric but receives no architectural specialization. H13 (R1) attempted a surface dual-head without FiLM and failed — but that was before Lion, GEGLU, slice=96, bf16, and the full FiLM conditioning. H16 (R1) was a follow-up WIP that added the surface head on top of the FiLM-conditioned stack, but its final result is not visible in the experiment log (was WIP at R1 end). In the current regime, FiLM already conditions the full token set; a thin surface residual head adds capacity for the specific pressure-on-surface computation without disrupting the volume field. Analogous to the auxiliary surface-specific heads in aerodynamic DNN literature (Li et al., ICLR 2024 FlowBench).

**Why not already ruled out:** H13 failed without FiLM; H16 was WIP with FiLM but its result is not recorded (likely not merged). The current config now has Lion+bf16+β₂=0.997+slice=96 — a materially different optimization regime that may allow this specialized head to train stably.

**Estimated implementation complexity:** S-M (student adds `SurfacePressureHead = nn.Sequential(nn.Linear(n_hidden, n_hidden//2), nn.GELU(), nn.Linear(n_hidden//2, 1))` applied to surface node embeddings from the final layer's `fx`, computing `p_residual = surf_head(fx_surface)`, then `p_final = p_main + p_residual` for surface nodes only).

**Expected risk/reward:** Medium risk (architectural addition may require careful weight init to ensure the residual starts near zero — student should zero-init the final linear layer), good reward (direct architectural support for the primary metric is theoretically motivated).

**Concrete first-arm spec:**

```bash
# Arm A: surface pressure residual head, zero-init output layer, surf_head_weight=1.0 (no extra weighting)
python train.py --epochs 50 \
  --experiment_name h111-arm-a-surf-p-residual \
  --agent <student> \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.997 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --use_bf16 \
  --surf_p_residual_head  # new flag; student adds SurfacePressureHead with zero-init last layer
```

- Expected wall time: ~30 min (negligible overhead vs H95)
- Pass criterion: val_avg/mae_surf_p < 40.51
- Conclusive win: val < 38.84
- Falsifying result: val ≥ 40.51 (regression vs H95)
- Key diagnostic: Check if `val_geom_camber_rc/mae_surf_p` improves specifically — that is the hardest OOD split and the one where a surface-specialized path should matter most

---

## Priority Ranking

| Rank | Idea | Rationale | Complexity | Risk |
|------|------|-----------|------------|------|
| 1 | H105: SWA over last 7 epochs | Near-zero cost, well-grounded theory, no interaction with locked levers | S | Low |
| 2 | H108: AoA+Re input jitter | XS cost, directly targets identified OOD axes (val_geom_camber_rc=54.51, val_re_rand=42.43) | XS-S | Low |
| 3 | H106: Fourier PE for (x,z) | Well-validated in PDE/mesh literature; no locked-lever conflicts; directly addresses spectral bias | S-M | Medium |
| 4 | H104: Per-sample surf_p std norm | H8 was unresolved under Lion; mechanism is orthogonal; 40x dynamic range is a real diagnostic | S | Medium |
| 5 | H107: Auxiliary Re-prediction | Targets val_re_rand directly; zero compute cost; one new head | S-M | Medium |
| 6 | H111: Surface-P residual head | H13 failed but conditions have changed; direct metric targeting | S-M | Medium |
| 7 | H110: n_hidden=256 | Bold capacity bet; gate on H100 result first | XS (CLI) | Medium-High |
| 8 | H109: Surface/volume norm split | Theoretically motivated but scoring.py contract makes this complex | M | Medium-High |

---

## Research State Update (Cycle 35)

**Current bottleneck:** The local hyperparameter neighborhood (optimizer axes, LR, BS, WD, bf16) is near-saturated at val=40.51. The main open levers are: (a) schedule fix (H99), (b) capacity under bf16 (H100/H101/H102/H103), and (c) untested mechanisms: data representation, auxiliary tasks, test-time ensembling.

**Frontier directions by category:**
- **Training dynamics:** SWA (H105, this doc)
- **Data representation / augmentation:** Per-sample norm (H104), AoA+Re jitter (H108), Fourier PE (H106), surface/volume norm split (H109)
- **Architecture:** Surface-P residual head (H111), n_hidden=256 (H110)
- **Multi-task regularization:** Auxiliary Re-prediction (H107)

**What should not be repeated:** WSD schedule (closed 3 times — requires longer budget), EMA (wrong regime at 4875 steps), Mixup (wrong inductive bias — full-sample label mixing), warmup LR (regresses at slice=96), n_head=4 (regresses), η_min > 0 (H87 negative), DropPath.

**Most discriminating next experiment:** H105 (SWA) — lowest cost, most interpretable result, no interaction with locked levers. H108 (AoA+Re jitter) is equally compelling if OOD splits are the binding constraint.
