# SENPAI Research Results

## 2026-04-29 19:35 — PR #1114 (CLOSED): Curriculum surf_weight ramp 1→20 over training

- charliepai2f4-frieren/surf-weight-curriculum
- Hypothesis: A linear curriculum surf_weight 1.0→20.0 over training would let the model establish a coherent global flow field first, then increasingly focus on surface pressure (easy-to-hard). Across 5 rounds the hypothesis evolved into a full surf_weight landscape mapping.
- Final round: sw=4 vs sw=10 control on the AMP + n_hidden=160 + lr=1e-3 + T_max=15 recipe (frieren manually ported the recipe to r4 since the codebase still had n_hidden=128/no AMP/Re-adaptive loss at r4 HEAD).

| Run | val_avg/mae_surf_p | Best epoch | Δ vs sw=10 control |
|---|---:|---:|---:|
| sw=4 + AMP + n_hidden=160 | 92.544 | 16 | +2.81 (worse) |
| sw=10 control (same port) | 89.734 | 16 | — |

Per-split (sw=4 vs sw=10): val_single_in_dist 108.03/101.70, val_geom_camber_rc 105.96/102.61, val_geom_camber_cruise 70.13/67.64, val_re_rand 86.05/86.98. sw=4 loses on 3/4 val splits and 3/3 clean test splits (test 3-split avg gap = 2.96, excluding the corrupt cruise sample).

Earlier rounds (old recipe lr=5e-4 / T_max=50) had mapped a non-monotonic optimum at sw=4-5: sw=3→134.91, sw=4→126.49, sw=5→126.93, sw=6→132.90, sw=10 (canonical)→124.727, curriculum 1→20→130.68. That optimum did NOT transfer to the new recipe.

- CLOSED as a clean negative result. Mechanistic interpretation (frieren): the old sw=4-5 optimum was an artifact of slow LR cooling — with lr=5e-4 and T_max=50, the model spent more epochs with effective LR floor and benefitted from lower surface gradient strength early on. Under the new recipe (lr=1e-3, T_max=15, AMP) the cosine schedule fully completes inside the 30-min budget and n_hidden=160 has more capacity, so volume converges fast enough that high surface weight no longer destabilizes training. Val trajectory crossover is at epoch 7-10: sw=4 leads early (volume-stabilization mechanism), sw=10 catches up by epoch 10 and pulls away.
- Useful side effect: frieren's manual port (AMP autocast + GradScaler, n_hidden=128→160, removal of per-sample Re-adaptive loss) hit 89.734 — confirms the recipe works on r4 codebase, but is now superseded by PR #1243 (n_hidden=192 → 88.421).
- Architectural gap with r1 (75.750) localized: r1 has RFF (PR #1138) + SwiGLU (PR #1160); r4 has neither. The 14-point residual gap is the cost of those missing features. RFF is in flight (tanjiro PR #1193), SwiGLU is in flight (fern PR #1271).
- Pre-existing test_geom_camber_cruise NaN (sample 000020.pt has 761 +Inf values) flagged 4× by frieren; needs a separate evaluate_split predictions-side guard PR.

---

## 2026-04-29 10:00 — PR #1112: Attention dropout=0.1 for OOD slice regularization

- charliepai2f4-edward/attention-dropout
- Hypothesis: Applying dropout=0.1 to PhysicsAttention slices improves OOD generalization on geom_camber and re_rand splits by preventing over-reliance on specific attention patterns.
- Metric summary: `models/model-attention-dropout-20260429-100507/metrics.yaml`

| Split | mae_surf_p |
|-------|------------|
| val_single_in_dist | 159.429 |
| val_geom_camber_rc | 155.559 |
| val_geom_camber_cruise | 92.955 |
| val_re_rand | 110.181 |
| **val_avg/mae_surf_p** | **129.531** |
| test_single_in_dist | 139.912 |
| test_geom_camber_rc | 138.778 |
| test_re_rand | 110.986 |
| test_avg/mae_surf_p | NaN (data bug: test_geom_camber_cruise/000020.pt has 761 inf values) |

- MERGED. Establishes baseline: val_avg/mae_surf_p=129.531 at best epoch 13/50. Training timed out at ~14 epochs (~133s/epoch). VRAM peaked at 42.73 GB of 96 GB available — substantial headroom for scaling. Cruise split (92.955) dramatically easier than raceCar splits (~155-159), suggesting multi-domain difficulty imbalance.

---

## 2026-04-29 14:00 — PR #1116: OneCycleLR (max_lr=2e-3, pct_start=0.3) vs cosine annealing

- charliepai2f4-tanjiro/onecycle-lr
- Hypothesis: OneCycleLR with high peak LR (2e-3) and warmup phase (pct_start=0.3) achieves better convergence than cosine annealing within the ~14-epoch training budget.
- Metric summary: `models/model-charliepai2f4-tanjiro-onecycle-lr-v2-20260429-112417/metrics.yaml`

| Split | mae_surf_p |
|-------|------------|
| val_single_in_dist | 192.578 |
| val_geom_camber_rc | 186.574 |
| val_geom_camber_cruise | 125.872 |
| val_re_rand | 146.952 |
| **val_avg/mae_surf_p** | **162.994** |
| test_single_in_dist | 174.447 |
| test_geom_camber_rc | 168.262 |
| test_re_rand | 145.871 |

- CLOSED. +25.8% worse than baseline (162.994 vs 129.531). Root cause: budget mismatch. With epochs_configured=50 and 375 steps/epoch, total_steps=18,750. Only ~5,250 steps ran before the ~30-minute timeout (28% of the full cycle). LR was still in the ascending warmup phase throughout all training — epoch 1: ~1.01e-4, epoch 5: ~5.60e-4, epoch 14: ~1.98e-3 (confirmed from metrics.jsonl trajectory). The LR never peaked or annealed, so the model trained entirely in high-gradient-variance warmup regime. OneCycleLR is fundamentally incompatible with the timeout-limited training budget when configured over 50 epochs.

---

## 2026-04-29 15:00 — PR #1114: Curriculum surf_weight ramp 1→20 over training

- charliepai2f4-frieren/surf-weight-curriculum
- Hypothesis: Ramping surf_weight linearly from 1→20 over training lets the model learn global flow structure first before specializing on surface pressure, improving generalization.
- Metric summary: `models/model-charliepai2f4-frieren-surf-weight-curriculum-20260429-..../metrics.yaml`

| Split | mae_surf_p |
|-------|------------|
| val_single_in_dist | 172.92 |
| val_geom_camber_rc | 135.23 |
| val_geom_camber_cruise | 94.48 |
| val_re_rand | 120.09 |
| **val_avg/mae_surf_p** | **130.677** |
| test_single_in_dist | 141.01 |
| test_geom_camber_rc | 125.30 |
| test_re_rand | 112.96 |

- SENT BACK for revision. 130.677 vs baseline 129.531 (+0.9%, marginally worse). Key insight: at best epoch 13, the ramp formula `1.0 + (epoch-1)*(20-1)/49` gives surf_weight ≈ 5.65 — only 28% through the configured ramp. The model never experienced surf_weight > ~7 during the entire run. The improvement over frieren's own fixed-weight=10 baseline (137.93) may be explained by the *lower effective* surf_weight (~5.65) rather than the ramp mechanism itself. Next: ablate flat surf_weight=5 and flat surf_weight=6 to isolate whether the ramp shape or the effective magnitude drives the result.

---

## 2026-04-29 18:30 — PR #1113: Learnable 3-way domain embedding for multi-domain routing

- charliepai2f4-fern/domain-tag-embedding
- Hypothesis: A learnable nn.Embedding(3, n_hidden) injected after the preprocess MLP routes each sample through domain-aware representations, reducing multi-domain confusion in PhysicsAttention.
- Two trials run:
  - **Trial 1 (noisy heuristic domain IDs, ~68% accurate):** val_avg/mae_surf_p=130.238 — worse than baseline
  - **Trial 2 (clean manifest-derived domain IDs, 100% accurate):** val_avg/mae_surf_p=135.125 — even worse

| Split | Trial 1 (noisy) mae_surf_p | Trial 2 (clean) mae_surf_p |
|-------|--------------------------|--------------------------|
| val_single_in_dist | 167.3 | 174.2 |
| val_geom_camber_rc | 140.6 | 149.8 |
| val_geom_camber_cruise | 98.7 | 104.3 |
| val_re_rand | 114.3 | 112.1 |
| **val_avg/mae_surf_p** | **130.238** | **135.125** |

- CLOSED. Both trials failed to beat baseline 124.727 (or even the prior 129.531). Surprisingly, clean domain IDs were worse than noisy ones — suggesting domain routing is already implicitly encoded in the 24-dim geometry features (foil NACA params, gap, stagger, AoA) and Transolver's PhysicsAttention. Injecting explicit domain tags adds a conflicting shortcut that hurts generalization. Student offered to create bug-fix PR for the Inf*0=NaN scoring issue in test_geom_camber_cruise/000020.pt; noted for future investigation.

---

## 2026-04-29 18:45 — PR #1114: Flat surf_weight ablation sw=5,6 (frieren, sent back for sw=3,4 sweep)

- charliepai2f4-frieren/surf-weight-curriculum
- Hypothesis: Linear ramp 1→20 over training; sent back after finding ramp only reached sw≈5.65 at epoch 13. Ablation of flat values:
  - sw=5: val_avg/mae_surf_p=126.93
  - sw=6: val_avg/mae_surf_p=132.90
  - sw=10 (baseline): ~137.93 (frieren's own run)
  - ramp 1→20 (effective sw~5.65 at epoch 13): 130.677

| surf_weight | val_avg/mae_surf_p | Notes |
|------------|-------------------|-------|
| 5 (flat) | 126.93 | Best in this round |
| 6 (flat) | 132.90 | |
| 10 (flat) | ~137.93 | frieren's own baseline |
| ramp 1→20 | 130.677 | Effective sw≈5.65 at epoch 13/50 |

- SENT BACK for revision (2nd time). Clear apparent monotonic trend: lower surf_weight is better. sw=5 gives 126.93 but doesn't beat current best 124.727. Directed to ablate sw ∈ {3, 4}.

---

## 2026-04-29 19:30 — PR #1114: Flat surf_weight ablation sw=3,4 (frieren, sent back for sw=4+gradclip+LR8e-4)

- charliepai2f4-frieren/surf-weight-curriculum
- Hypothesis: Continue surf_weight sweep at lower values {3, 4} to see if trend continues below 124.727 baseline.

### Complete surf_weight sweep results

| surf_weight | val_avg/mae_surf_p | best epoch |
|------------|-------------------|------------|
| 3 | 134.91 | 12 |
| **4** | **126.49** | **14** |
| 5 | 126.93 | 12 |
| 6 | 132.90 | 13 |
| 10 (default) | 137.93 | 14 |
| ramp 1→20 | 130.677 | 13 (effective sw≈5.65) |

### Per-split breakdown — best run (sw=4)

| Split | sw=3 | sw=4 | sw=5 | sw=10 |
|-------|------|------|------|-------|
| val_single_in_dist | 172.77 | **156.03** | 157.21 | 165.81 |
| val_geom_camber_rc | 140.18 | **132.79** | 137.25 | 145.42 |
| val_geom_camber_cruise | 111.84 | 99.37 | **98.80** | 114.09 |
| val_re_rand | 114.84 | 117.75 | **114.47** | 126.40 |
| **val_avg/mae_surf_p** | **134.91** | **126.49** | **126.93** | **137.93** |

- SENT BACK for revision (3rd time). Key finding: the trend is **non-monotonic** — sw=3 is decisively worse than sw=4 (134.91 vs 126.49), inverting the apparent pattern. The optimum is at sw=4–5. Neither sw=3 nor sw=4 beats the canonical baseline of 124.727.

  **Key insight from best result (PR #1187, 102.080):** gradient clipping + LR=8e-4 was the dominant factor in that 18% improvement. The surf_weight sweep was done against the old LR=5e-4 baseline. Combining sw=4 with the winning recipe is the logical next step.

  New direction: `surf_weight=4 + grad_clip_max_norm=1.0 + lr=8e-4`. Target: beat val_avg/mae_surf_p < 102.080.

---

---

## 2026-04-29 16:15 — PR #1147: Double slice_num 64→128 for richer PhysicsAttention physics tokens

- charliepai2f4-tanjiro/slice-num-128
- Hypothesis: Doubling slice_num from 64 to 128 in PhysicsAttention gives the model twice as many learnable physics tokens to represent distinct flow regimes, improving surface pressure prediction.
- Metric summary: `target/models/model-charliepai2f4-tanjiro-slice-num-128-20260429-121610/metrics.yaml`

| Split | mae_surf_Ux | mae_surf_Uy | mae_surf_p | Δ vs baseline (p) |
|-------|-------------|-------------|------------|-------------------|
| val_single_in_dist | 1.926 | 0.913 | 160.451 | +1.022 |
| val_geom_camber_rc | 3.097 | 1.180 | 143.959 | -11.600 |
| val_geom_camber_cruise | 1.393 | 0.630 | 105.724 | +12.769 |
| val_re_rand | 2.198 | 0.880 | 126.117 | +15.936 |
| **val_avg/mae_surf_p** | | | **134.063** | **+4.532 (worse)** |
| test_single_in_dist | 1.880 | 0.892 | 144.374 | |
| test_geom_camber_rc | 3.179 | 1.111 | 138.791 | |
| test_geom_camber_cruise | 1.275 | 0.589 | NaN (corrupt baseline sample) | |
| test_re_rand | 2.098 | 0.883 | 127.308 | |

Compute: n_params=672,919 (+1.6%), peak VRAM=55.00 GB (+28.7%), sec/epoch=174s (+30.8%), epochs completed=11 (vs 14 for baseline at same wall-clock budget).

- CLOSED. 3.5% worse than baseline (134.063 vs 129.531). The +30% per-epoch slowdown reduced the epoch budget from ~14 to 11 within the 30-min wall-clock limit, while the wider softmax over 128 bins operates on the same 32-d head projection (n_hidden=128, n_head=4 → dim_head=32), diluting per-slice signal. Two failure modes: (1) compute regression from wider einsum eating into the epoch budget; (2) "more clusters than data supports" — doubling slice_num at fixed n_hidden packs twice as many tokens into the same projection capacity. Interesting split-level signal: val_geom_camber_rc improved -11.6 mae, suggesting some benefit for raceCar-tandem geometry. Next experiment: try increasing n_hidden capacity rather than slice count.

---

## 2026-04-29 14:01 — PR #1177: Auxiliary surface-pressure MLP head with aux_surf_weight=20

- charliepai2f4-tanjiro/aux-surface-pressure-head
- Hypothesis: A 2-layer auxiliary MLP head applied to the Transolver's pre-final hidden state, trained with aux_surf_weight=20 on surface pressure only, will improve val_avg/mae_surf_p by providing a focused gradient signal without competing with velocity channels.
- Metrics summary: `models/model-charliepai2f4-tanjiro-aux-surface-pressure-head-20260429-132459/metrics.yaml`

| Split | Aux head | Baseline (PR #1128) | Δ |
|-------|----------|---------------------|---|
| val_single_in_dist | 168.931 | 153.200 | +15.731 |
| val_geom_camber_rc | 139.045 | 133.070 | +5.975 |
| val_geom_camber_cruise | 97.895 | 96.830 | +1.065 |
| val_re_rand | 118.011 | 115.800 | +2.211 |
| **val_avg/mae_surf_p** | **130.971** | **124.727** | **+6.244 (worse)** |
| test_single_in_dist | 152.505 | 138.800 | |
| test_geom_camber_rc | 128.898 | 124.400 | |
| test_re_rand | 115.097 | 114.690 | |

- CLOSED. 130.971 vs baseline 124.727 (+4.9% worse). The aux_surf_weight=20 dominated the gradient budget (aux_contrib ≈ 4.4 > main_loss ≈ 3.8 at epoch 13), over-specializing the backbone toward training surface structure. Notably, val_geom_camber_rc improved vs PR #1112 prior baseline (155.56→139.05, -10.6%) but was +5.975 worse than current baseline (133.07). The mechanism is real — a focused aux head does shift the representation toward surface pressure — but weight=20 is too aggressive. Student suggested sweeping AUX_SURF_WEIGHT to 1, 5, 10 and detaching hidden before aux head. Not pursuing this direction further as other higher-impact experiments are in flight. Aux head params: 16.6K (negligible). Peak VRAM: 43.7 GB.

---

## 2026-04-29 14:38 — PR #1187: Gradient clipping + raised LR 8e-4 for faster convergence

- charliepai2f4-fern/gradient-clip-lr8e4
- Hypothesis: Adding gradient clipping (max_norm=1.0) stabilizes training enough to safely raise LR from 5e-4 to 8e-4, allowing faster convergence within the ~14-epoch timeout-limited budget.
- Metric summary: `models/model-gradient-clip-lr8e4-20260429-135924/metrics.yaml`

| Split | mae_surf_p | Δ vs baseline (PR #1128) |
|-------|------------|--------------------------|
| val_single_in_dist | 124.600 | -18.7% |
| val_geom_camber_rc | 105.819 | -20.5% |
| val_geom_camber_cruise | 82.739 | -14.6% |
| val_re_rand | 95.161 | -17.8% |
| **val_avg/mae_surf_p** | **102.080** | **-18.2%** |
| test_single_in_dist | 106.286 | |
| test_geom_camber_rc | 97.570 | |
| test_geom_camber_cruise | NaN (pre-existing corrupt sample) | |
| test_re_rand | 90.364 | |

- MERGED. Outstanding 18.2% improvement across all splits — the single largest jump seen in this research programme. The hypothesis was perfectly calibrated: the CosineAnnealingLR schedule was barely active at epoch 14/50, so the LR was essentially flat the entire run. The 60% LR increase (5e-4 → 8e-4) with gradient clipping (max_norm=1.0) gave the optimizer real per-step progress without instability. Val loss curve went 221.6 → 102.1 over 14 epochs with is_best epochs at {1,2,3,4,5,6,8,9,11,14}, confirming steady improvement throughout. Peak VRAM unchanged at 42.75 GB. New baseline: val_avg/mae_surf_p = **102.080**. Student's follow-up suggestions are excellent: push LR higher (1e-3+), match T_max to actual epoch budget (~14), or switch to OneCycleLR sized to actual training budget.

---

## 2026-04-29 20:00 — PR #1186: Combine surf_weight=5 with per-sample Re-adaptive loss (SENT BACK)

- charliepai2f4-edward/sw5-adaptive-loss
- Hypothesis: surf_weight=5 (validated by frieren's ablation as best single-val-weight) combined with per-sample Re-adaptive 1/σ loss weighting (validated in PR #1128) are orthogonal improvements that should stack.
- Metric summary: `target/models/model-sw5-adaptive-loss-20260429-150814/metrics.jsonl`
- Recipe: n_hidden=128, no AMP, lr=5e-4, T_max=50, surf_weight=5 + per-sample 1/σ adaptive loss. 14 epochs completed.

| Split | mae_surf_p (best seed, epoch 14) |
|-------|----------------------------------|
| val_single_in_dist | 153.708 |
| val_geom_camber_rc | 130.242 |
| val_geom_camber_cruise | 92.701 |
| val_re_rand | 106.557 |
| **val_avg/mae_surf_p** | **120.802** |
| test_single_in_dist | 138.614 |
| test_geom_camber_rc | 117.909 |
| test_geom_camber_cruise | NaN (corrupt 000020.pt) |
| test_re_rand | 105.253 |

Seed variance across 3 runs: 127.844 → 125.175 → 120.802 (range ~7, mean ~124.6)
Weight diagnostics (epoch 14): weight_min=0.080, weight_max=3.116, surf_p_std_mean=0.850 — healthy bounded range.

- SENT BACK. The PR ran against the **old recipe** (n_hidden=128, lr=5e-4, no AMP, T_max=50) and the old baseline (124.727). It achieves 120.802 on the best seed, but the **current best baseline is 75.750** (PR #1197, AMP bfloat16 + n_hidden=160 + lr=1e-3 + T_max=15). The direction itself is validated: sw=5 + adaptive 1/σ loss do combine meaningfully against the pre-AMP recipe. Sent back with instructions to rebase onto the current best recipe and re-test:
  - n_hidden=160, AMP bfloat16, lr=1e-3, CosineAnnealingLR(T_max=15, eta_min=1e-6), grad_clip max_norm=1.0
  - Keep: surf_weight=5 + per-sample Re-adaptive 1/σ loss (with clamp=0.2, mean-1 batch norm)
  - Target: val_avg/mae_surf_p < 75.750
  - CLI: `python train.py --experiment_name sw5-adaptive-loss-amp --n_hidden 160 --n_layers 5 --n_head 4 --slice_num 64 --mlp_ratio 2 --dropout 0.1 --lr 1e-3 --surf_weight 5 --batch_size 4 --amp --amp_dtype bfloat16`

---

## 2026-04-29 16:39 — PR #1213: Batch size 8 + linear LR scale (2e-3) for gradient quality (CLOSED)

- charliepai2f4-fern/batch-size-8-linear-lr
- Hypothesis: Doubling batch_size 4→8 with linear LR scaling 1e-3→2e-3 (Goyal et al. 2017) improves gradient quality and convergence given VRAM headroom (~42 GB of 96 GB at bs=4).
- Recipe: ported PR #1197 code to r4 first (AMP bfloat16, n_hidden=160, plain masked MSE — replacing per-sample Re-adaptive loss), then applied bs=4→8, lr=1e-3→2e-3.
- Metric summary: `models/model-charliepai2f4-fern-batch-size-8-lr2e3-20260429-160107/metrics.yaml`

| Split | mae_surf_p | Baseline (PR #1197) | Δ |
|-------|------------|---------------------|---|
| val_single_in_dist | 154.028 | 78.755 | +75.273 |
| val_geom_camber_rc | 126.584 | 88.578 | +38.006 |
| val_geom_camber_cruise | 85.887 | 61.344 | +24.543 |
| val_re_rand | 105.894 | 74.322 | +31.572 |
| **val_avg/mae_surf_p** | **118.098** | **75.750** | **+42.348 (+56% worse)** |
| test_single_in_dist | 131.169 | 67.414 | |
| test_geom_camber_rc | 113.244 | 72.814 | |
| test_geom_camber_cruise | NaN (bf16 pressure overflow) | 50.498 | |
| test_re_rand | 103.756 | 69.206 | |
| **test_avg/mae_surf_p** | **NaN** | **64.983** | |

- Throughput: epoch ~121s (basically unchanged from baseline 124s); peak VRAM 77.5 GB (1.83x baseline 42.29 GB); 15 epochs realised.
- Val trajectory: 265.7 → 225.5 → 238.5 → 197.7 → 172.5 → 160.7 → 154.8 → 168.2 → 142.6 → 134.6 → 129.5 → 132.0 → 118.1 → 123.1 → 118.2 — far slower convergence than baseline (which reaches 75.75 in same wall-clock).
- CLOSED. Clear regression (+56%). Diagnosis (validated by student):
  1. **lr=2e-3 is too aggressive at this scale.** Linear LR scaling (Goyal et al.) holds in the large-batch regime (bs in 100s); at bs=4→8 with ~188 steps/epoch, gradient noise scale dominates and 2e-3 overshoots even with grad_clip=1.0.
  2. **Cosine T_max=15 + bs doubling halves effective parameter updates.** Schedule was tuned for bs=4 step density; bs=8 collapses LR (1.92e-4 by epoch 13, 2.28e-5 by epoch 15) before convergence.
  3. **bf16 numerical headroom marginal at lr=2e-3** — likely cause of test_geom_camber_cruise pressure NaN (Ux/Uy clean on same split; val_geom_camber_cruise also clean).

- **Key takeaway:** linear LR scaling fails in the small-batch regime here; sqrt-scaling (lr ∝ √k → 1.41e-3) or no scaling (lr=1e-3) would be more conservative. Batch-size scaling direction not closed — gradient accumulation at bs=4 (effective bs=8, lr=1e-3, no VRAM cost) is the highest-value follow-up to isolate the pure batch-size effect.

- **Side-benefit:** student ported PR #1197 code (AMP bfloat16 + n_hidden=160 + plain MSE) to r4 in this PR. The recipe is now validated as the right port pattern; remaining r4 students still need to do the same port in their PRs (already communicated via send-back comments).

---

## 2026-04-29 21:00 — PR #1230: Gradient accumulation (bs=4, accum_steps=2, effective bs=8) (CLOSED)

- charliepai2f4-fern/grad-accum-bs8-emulation
- Hypothesis: Emulate bs=8 gradient quality via accum_steps=2 at bs=4, lr=1e-3 unchanged (no VRAM cost); isolates pure batch-size gradient signal from PR #1213's confounds (lr=2e-3 + VRAM pressure).
- Metric summary: `target/models/model-charliepai2f4-fern-grad-accum-bs8-emulation-20260429-165542/metrics.yaml`

| Split | mae_surf_p | Baseline (PR #1197) | Δ |
|-------|------------|---------------------|---|
| val_single_in_dist | ~223.9 | 78.755 | +145.1 |
| val_geom_camber_rc | ~91.7 | 88.578 | +3.1 |
| val_geom_camber_cruise | ~66.9 | 61.344 | +5.6 |
| val_re_rand | ~81.8 | 74.322 | +7.5 |
| **val_avg/mae_surf_p** | **101.013** | **75.750** | **+25.263 (+33.4% worse)** |

- Throughput: epoch ~111-114s; VRAM 38.8 GB (lower than baseline 42.29 GB due to half micro-batch count); 17 epochs / 3,179 optimizer steps realized vs ~5,625 for baseline.
- Micro-steps/epoch=375, optim-steps/epoch=187 (2:1 ratio confirmed correct).
- Val trajectory: epoch 1: val=221.28 → epoch 15 (cosine T_max reached): val=101.85 → epoch 17 (timeout): val=101.01 — still far from convergence at schedule end.

- CLOSED. Clear regression (+33.4%). Root cause: **optimizer-step starvation**. At accum_steps=2 the optimizer fires only ~187 steps/epoch vs ~375 in baseline. Within the 30-min / T_max=15 cosine budget, only ~2,800 optimizer steps execute vs ~5,625 for baseline. The gradient-quality improvement from averaging 2 micro-batches cannot compensate for training to only 50% of the optimizer-step budget. surf_loss=0.23 at epoch 17 while baseline has decayed the LR fully and converged. Student diagnosis was correct on all counts.
- **Key takeaway:** This experiment closes the batch-size-scaling research direction. The pair of experiments PR #1213 (bs=8, lr=2e-3 — +56% regression) and PR #1230 (bs=4 grad-accum, lr=1e-3 — +33% regression) conclusively shows that at this wall-clock budget (30 min / 15 epochs), batch size is not the lever. Moving on to capacity, loss formulation, and architectural directions.


## 2026-04-29 18:30 — PR #1186 (CLOSED): sw=5 + per-sample Re-adaptive loss (rerun on new baseline)
- charliepai2f4-edward/sw5-adaptive-loss
- Hypothesis: surf_weight=5 + per-sample 1/σ loss (which beat the OLD baseline of val=124.727 by reaching val=120.802) would also help on the new PR #1197 baseline (val=75.750).
- Results:

| Metric | Seed 1 | Seed 2 | Mean | Baseline (PR #1197) | Δ vs baseline |
|--------|--------|--------|------|--------------------|---------------|
| val_avg/mae_surf_p | 87.924 | 90.97 | ~89.4 | 75.750 | +16.1% / +19.5% |

- CLOSED. Robust two-seed regression. Per-sample 1/σ + sw=5 worked on the OLD recipe (n_hidden=128, lr=5e-4) by remedying under-fitting on high-Re tails; on the new recipe (n_hidden=160 + AMP + lr=1e-3 + grad clip + T_max=15), the model fits the tails directly and equalizing per-sample magnitudes flattens gradient signal on the very samples that drive the primary metric.
- **Key takeaway:** Per-sample inverse-std weighting is a remedy for under-capacity / under-trained regimes only. Do not stack with the n_hidden=160/AMP recipe in future experiments.

## 2026-04-29 18:30 — PR #1235 (CLOSED): Deeper FiLMNet 3-layer residual conditioner [r1 track]
- charliepai2f1-thorfinn/film-depth-3layers
- Hypothesis: deepening the FiLM conditioner from 2-layer to 3-layer residual (Linear(11→512)→GELU→Linear(512→512)→GELU→Linear(512→out_dim) + skip) would let the model exploit more conditioning capacity.
- Results: monotonically descending loss but slower convergence at every epoch vs 2-layer baseline; only 13 epochs fit in 30-min budget (vs ~15 baseline). val_single_in_dist improves slightly (+3.1) but all three OOD splits regress; val_avg does not beat baseline.
- CLOSED. Student's own conclusion was "don't merge — baseline wins."
- **Key takeaway:** At the 30-min/15-epoch budget on this dataset, **width > depth** for the FiLM conditioner. Identity/zero-init residuals make the deeper network equivalent to the shallower one at step 0 — the depth advantage only materializes after many gradient steps which the budget doesn't grant. Future capacity experiments on the conditioner should widen the existing 2-layer MLP (e.g. 11→768→out_dim or 11→1024→out_dim) before adding more layers.

## 2026-04-29 18:35 — PR #1249 (ASSIGNED): Curvature-weighted surface loss
- charliepai2f4-edward/curvature-weighted-surf-loss
- Hypothesis: up-weight surface nodes near the leading and trailing edges (regions of steepest pressure gradient) using the raw signed-arc-length feature `saf` (input dims 2-3, pre-normalization). Weight formula: `w_i = 1 + alpha * exp(-beta * min(|saf_i|, 1-|saf_i|))` with alpha=5, beta=20, mean-1 normalized per-sample. Orthogonal to per-sample 1/σ loss already in baseline.
- Reference: AirfRANS / Bonnet et al. 2022 — boundary-layer-weighted losses consistently improve surface pressure prediction by 5-15% in CFD-ML.
- r4 baseline target: val_avg/mae_surf_p < 102.080 (PR #1187, gradient clip + lr=8e-4 — current actual r4 merged baseline). Note: BASELINE.md mentions PR #1197 (75.750) but that is on a separate r1 track and has not been merged into r4 advisor branch.
