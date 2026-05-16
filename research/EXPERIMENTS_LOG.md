# SENPAI Research Results — charlie-pai2i-24h-r4

Per-PR results log. Earliest at the bottom; latest at the top.

## 2026-05-16 12:24 — PR #3867: H41 Domain-type indicator embedding (thorfinn) — **CLOSED**

- Branch: `charliepai2i24h4-thorfinn/domain-type-indicator`
- Result: val_avg=78.13 (+15.5%), val_single=96.39 (+20.0% on target split — opposite direction from hypothesis). best_epoch=11/15 (one fewer than baseline's 12).
- Key diagnostic: embedding norms grew monotonically from 0 → ~0.10 (5× init scale) — mechanism active. Both is_tandem and dim-22-only discriminators agreed 1.000.
- **Critical refutation**: dims 18-23 are EXACTLY zero for single-foil (not noisy, not fuzzy). The Transolver's preprocess MLP can learn this AND-of-zero with a single hidden unit. The PR's premise that "Transformer learns AND patterns slowly" applies to FUZZY ANDs; here the AND is tight and the model already handles it.
- **Implementation finding**: forward() receives normalized x_norm where dims 18-23 map to constants (-mean/std). The student correctly computed is_tandem from raw x at the loop level, confirmed via Step 0 diagnostic.
- **Updates val_single bottleneck story**: NOT a routing problem. Must be rarity, variance, output calibration, or shape distribution. The "highest researcher priority" hypothesis from 2026-05-16_09:30 is refuted.
- **Next**: thorfinn reassigned to H47 Focal regression loss (PR #3948) — attacks the rarity hypothesis via per-sample hard-example weighting.

## 2026-05-16 12:24 — PR #3948: H47 Focal regression loss (thorfinn) — **assigned**

- Branch: `charliepai2i24h4-thorfinn/focal-regression-loss`
- Hypothesis: per-sample focal weighting on the existing MSE-in-log1p loss. weights = (loss_i.detach() / mean).clamp(0.1, 5.0) ** gamma, normalized to preserve total magnitude. Attacks the rarity hypothesis from H41 closure: if single-foil samples are getting under-trained because they're a small fraction of balanced batches, focal weighting will up-weight them via their high relative error. Two arms gamma ∈ {1.0, 2.0}. Predicted -1 to -4 from 67.64.
- Replaces: #3867 (H41 domain-indicator closed +15.5%)

## 2026-05-16 11:40 — Round closures and reassignments (4 PRs)

### PR #3742: H33 OneCycleLR pct_start sweep (tanjiro) — **CLOSED**
- 3 arms (pct_start ∈ {0.10, 0.15, 0.20}). Best val=77.77 (+15%). All regress. best_epoch stays at 10-11; compression shifts LR lower sooner without moving the optimum. val_single hit hardest (+17 to +21). Together with H37, schedule axis is exhausted.
- **Next**: reassigned to H44 foil-relative coords (PR #3930).

### PR #3788: H36 channel-weighted surf_loss (frieren) — **CLOSED**
- 3 arms (surf_p_weight ∈ {2, 3, 5}). Best val=76.73 (+13.4%, w=2). All regress monotonically.
- **Critical insight**: Ux/Uy surface supervision is auxiliary regularization for the shared backbone. Cutting velocity weight corrupts the multi-task representation that p depends on. val_single (most p-dominated) hit hardest, +17 to +31. Rules out output-loss reweighting axis.
- **Next**: reassigned to H45 pressure-head split (PR #3928) — architectural rather than loss-side intervention for the same channel-imbalance question.

### PR #3539: H23 slice_num=32 retest on H24 baseline (alphonse) — **CLOSED**
- 4 seeds, mean val=72.22 (range 71.16-73.80, σ=1.1), +6.8% regression. test=62.80 within noise. Original SwiGLU-baseline win (-22%) does NOT compose with OneCycleLR.
- **Insight**: slice_num=32 and OneCycleLR are anti-synergistic — both regularize the same overfitting axis. Compositionality lesson for the project.
- **Next**: reassigned to H43 flow-condition MixUp (PR #3929) — first data-augmentation experiment, attacks a different axis.

### PR #3850: H39 BF16 AMP mixed precision (askeladd) — **CLOSED**
- Single arm. val=69.69 (+3.0%), test=60.92 (-1.9% improved). Schedule completed cleanly (LR=2.1e-9, best_epoch=15, monotone descent through all 15 epochs).
- val regression is structural (val_single +10.3% dominates), not BF16 numerical (slog1p stats clean, no NaN). Same single-foil bottleneck thorfinn H41 targets. Speedup only 5-25%, not 30-40% — attention dominates compute here, not matmul.
- **Insight**: AMP works mechanistically but speedup is too small for this compute regime. Stays available for future hypotheses needing more wall-clock budget.
- **Next**: reassigned to H46 gradient accumulation 2x (PR #3931) — variance-reducing optimizer steps from a different angle.

## 2026-05-16 11:40 — PR #3928: H45 Pressure-specific output head (frieren) — **assigned**
- Branch: `charliepai2i24h4-frieren/pressure-head-split`
- Hypothesis: Split final block output into `vel_head: Linear(n_hidden, 2)` for [Ux, Uy] and `p_head: SwiGLUMLP(n_hidden, n_hidden//2, 1)` for p. Keeps full Ux/Uy supervision intact while giving p dedicated capacity. Architectural counterpart to closed H36 (which reweighted output losses and corrupted multi-task representations). Predicted -2 to -5 from 67.64.
- Replaces: #3788 (H36 channel-weighted loss closed +13.4%)

## 2026-05-16 11:40 — PR #3929: H43 Flow-condition MixUp (alphonse) — **assigned**
- Branch: `charliepai2i24h4-alphonse/flow-condition-mixup`
- Hypothesis: MixUp on flow-condition subspace (dims 13-23: log_Re, AoA, NACA params, gap, stagger) and target y; keep node positions (dims 0-1) and shape descriptors (dims 2-11) from primary sample. Creates synthetic (geometry, flow condition) combinations without physically inconsistent node interpolation. Two arms α ∈ {0.4, 0.2}. First data-augmentation experiment in the project. Predicted -2 to -6, concentrated on val_geom_camber splits.
- Replaces: #3539 (H23 slice_num=32 retest closed +6.8%)

## 2026-05-16 11:40 — PR #3930: H44 Foil-relative coordinate encoding (tanjiro) — **assigned**
- Branch: `charliepai2i24h4-tanjiro/foil-relative-coords`
- Hypothesis: Augment input with per-node `log(1 + dist_to_surface_centroid)`. Computed from existing surface flag (dim 12) + position dims. Gives the model an explicit boundary-layer physics prior; dsdf features (dims 4-11) may already encode this, in which case redundant. First input-feature engineering since H5 RFF was merged. Predicted -1 to -4.
- Replaces: #3742 (H33 pct_start sweep closed +15%)

## 2026-05-16 11:40 — PR #3931: H46 Gradient accumulation 2x (askeladd) — **assigned**
- Branch: `charliepai2i24h4-askeladd/grad-accum-2x`
- Hypothesis: accum_steps=2 doubles effective batch from 4 to 8 without VRAM increase. Reduces gradient variance during OneCycleLR's high-LR warmup phase. Scheduler step interval tied to optimizer step (every 2 micro-batches) to keep schedule length constant. Chen et al. ICLR 2021 establishes the technique for small-batch fine-tuning. Predicted -1 to -4.
- Replaces: #3850 (H39 AMP closed +3.0% on val despite test improvement and clean schedule completion)

## 2026-05-16 11:22 — PR #3900: H42 Auxiliary Re/AoA prediction head (nezuko) — **assigned (recreates #3889)**

- Branch: `charliepai2i24h4-nezuko/aux-re-aoa-head-v2`
- Hypothesis: Lightweight auxiliary head predicts `log(Re)` and `AoA_foil1` from global masked mean-pooled latent after the final TransolverBlock. Auxiliary MSE loss with weight sweep {0.1, 0.05}. Forces the encoder to maintain Re/AoA-aligned representations via direct gradient signal. Caruana 1997 multi-task learning; Kendall, Gal, Cipolla CVPR 2018. Attacks val_re_rand=63.96 — only split with no previous targeted intervention. Predicted -2 to -6, concentrated on val_re.
- Replaces: #3687 (H30 grad-clip — closed), #3889 (advisor-side accidental auto-merge of the assignment branch when the advisor docs commit landed on the same branch then got merged into advisor; recreated cleanly as #3900). No student work lost — #3889 was never executed.

## 2026-05-16 10:34 — PR #3687: H30 Gradient clipping max_norm=1.0 (nezuko) — **CLOSED**

- Branch: `charliepai2i24h4-nezuko/grad-clip-1p0`
- Result: val_avg=71.93 mean of 4 seeds (+6.3% regression vs baseline 67.64). grad_clip=5.0 single seed: val_avg=71.53 (+5.8%). No seed beats baseline.
- Key diagnostic: `max_norm=1.0` fires on **100% of batches** throughout epochs 1-11. Natural gradient norm = mean 4-30, max up to 150 — clip fires on every step, not just spikes. AdamW steps become sign-of-gradient pseudo-steps, destroying first/second moment estimates. `max_norm=5.0` reduces fire rate to 25.9% by epoch 11 but early high-LR phase (≥60% fire rate epochs 1-7) still over-clips and regresses.
- Per-split: val_single worst affected (80.32 → 87.45 mean), consistent across all 4 seeds — largest absolute gradient signal under surf_weight=10 is dampened most.
- Lesson: OneCycleLR + surf_weight=10 + signed_log1p produces natural gradient norms 4-30. Any fixed max_norm below ~20 is not spike-filtering but constant damping. H24's clean monotonic descent confirms the optimizer is already stable; there is no instability problem to clip.
- **Closed**: negative result. Hypothesis disconfirmed under current OneCycleLR + surf_weight=10 + log1p stack.

## 2026-05-16 10:25 — PR #3867: H41 Domain-type indicator embedding is_tandem (thorfinn) — **assigned**

- Branch: `charliepai2i24h4-thorfinn/domain-type-indicator`
- Hypothesis: val_single=80.32 is the worst split despite being in-distribution. Root cause: single-foil samples have dims 18-23 all zero (gap, stagger, foil2 params) — structurally indistinguishable from degenerate tandem at zero gap/stagger. `nn.Embedding(2, n_hidden)` indexed by `is_tandem = (x[:, 0, 18:24].abs() > 1e-6).any(dim=-1)`, zero-init (no-op initially), added as bias to preprocess output. BERT token type IDs / segment embeddings. Predicted -3 to -8 (highest priority from researcher-agent survey).

## 2026-05-16 10:25 — PR #3762: H34 RFF n_freq sweep {16,64} (thorfinn) — **CLOSED**

- Branch: `charliepai2i24h4-thorfinn/rff-nfreq-sweep`
- Result: n_freq=16: val=76.52 (+13.1%), n_freq=64: val=81.97 (+21.2%). Both arms regress uniformly across all 4 splits. No split where richer/sparser RFF helps.
- Key insight: n_freq=32 (frieren H5 merged choice) is empirically near-optimal. Asymmetry (doubling hurts more than halving): larger input dilutes per-input-dim weight magnitude — preprocess bottleneck stays at n_hidden=256 regardless of input size. CFD pressure fields are spatially smoother than 3D NeRF textures; 32-mode basis with σ=1 captures relevant spatial frequencies.
- Truncation caveat: 1-epoch gap (n_freq=16 got 11 vs baseline's 12), but trend robust — n_freq=64 would need unrealistically fast convergence to catch baseline.
- Student suggested sigma sweep (σ ∈ {0.5, 1.0, 2.0} at fixed n_freq=32) as follow-up. Also correctly identified val_single anomaly as next target — exactly what H41 tests.
- **Closed**: definitive negative. RFF frequency axis exhausted at n_freq=32.

## 2026-05-16 09:40 — PR #3852: H40 SWA tail-averaging K={3,5} (edward) — **assigned**

- Branch: `charliepai2i24h4-edward/swa-tail-averaging`
- Hypothesis: After completing standard 15-epoch training, average the model state dicts from the last K epochs to create a pseudo-ensemble. Different from EMA (H8v3 failed — β=0.999 never catches up to converged region under truncated OneCycleLR): SWA explicitly averages the final converged tail, not a running estimate. Addresses schedule-truncation variance without losing gradient steps. Izmailov et al. 2018 (UAI). K sweep {3, 5}. Predicted -0.5% to -3%.

## 2026-05-16 09:40 — PR #3850: H39 BF16 mixed precision AMP (askeladd) — **assigned**

- Branch: `charliepai2i24h4-askeladd/amp-bf16-mixed-precision`
- Hypothesis: BF16 autocast (torch.amp) on forward+loss reduces matmul time by ~30-40%, unlocking 18+ realized epochs in the 30-min budget vs current ~12-13 epochs. Direct attack on the schedule-truncation structural bottleneck that closed H37, H25, H31, H27. BF16 (not FP16) avoids gradient scaler complexity, same dynamic range as FP32. Kalamkar et al. 2019; Wortsman et al. 2023. Predicted -2% to -6%.

## 2026-05-16 09:35 — PR #3792: H37 OneCycleLR epochs=12 schedule-fit (edward) — **CLOSED**

- Branch: `charliepai2i24h4-edward/onecycle-epochs12`
- Result: val_avg=77.30 vs baseline 67.64 → **+14.3% regression**. test_avg=68.81 (+10.8%). Per-split: val_single +24.3%, val_rc +3.5%, val_cruise +21.5%, val_re +10.5%. best_epoch=11 (last epoch, still descending at -4.2 val units/epoch). end_LR=1.72e-5 (91.6% of schedule completed). Per-epoch time: 166s (vs advisor's 130s estimate, +27% node variance). Realized epochs: 10.8 (not 12 as planned).
- Key insight: Compressing epochs to 12 reduced gradient steps by ~20%; this dominated the schedule-completion benefit. Per-epoch wall-time drifted +27% from different GPU node SKU, so schedule was STILL truncated (at epoch 10.8 vs planned 12). Schedule truncation cannot be fixed by epoch-count alone under variable wall-time — need step-count targets or schedule rearrangement.
- Student analysis sharp: correct mechanism, wrong knob. The H33 pct_start direction (rearrange within realized budget without losing steps) is the right path.
- **Closed**: structural negative. Filed insight: per-epoch wall-time variance from GPU node SKU is a confound for schedule-fitting hypotheses.

## 2026-05-16 09:35 — PR #3686: H31 SAM ρ=0.05 OneCycleLR (askeladd) — **CLOSED**

- Branch: `charliepai2i24h4-askeladd/sam-optimizer`
- Result: Run 1 (epochs=15) killed at ep 2 (80 min budget). Run 2 (contingency, epochs=10) truncated at ep 6 with val_avg=124.68. Run 3 (contingency, epochs=5) completed val_avg=131.58, test_avg=120.50. All arms 84-95% worse than baseline.
- SAM per-step time: 5.3 min/epoch (2.1× baseline, ~5.3 min/ep). With 30-min cap: only 5-6 SAM epochs fit. OneCycleLR schedules designed for 5-10 epochs cannot deliver the fine-tune tail that makes OneCycleLR's gain.
- No SAM-specific instabilities (loss monotone, no NaN, scheduler sync correctly handled by student). Negative result reflects compute-budget collision, NOT a SAM property failure.
- Student diagnostic excellent — correctly identified structural constraint, ran three contingency arms to confirm.
- **Closed**: SAM non-viable under current 30-min cap. AMP (H39) assigned as next step to unlock longer effective training.

## 2026-05-16 09:00 — PR #3824: H38 input Gaussian noise injection sweep {0.01,0.03,0.10} (fern) — **assigned**

- Branch: `charliepai2i24h4-fern/input-noise`
- Hypothesis: 3rd regularization axis untested. Bishop 1995: input noise σ on x_norm during training is mathematically equivalent to Tikhonov regularization on the input Jacobian — `σ² * E[||∇_x f(x)||²]`. Orthogonal to FFN dropout (H12, internal acts) and weight decay (params). Small training set (1499) overfits readily — visible in val-test gap (67.64 vs 62.12). 3 arms σ ∈ {0.01, 0.03, 0.10}. Noise fires ONLY in train forward, not eval. Predicted -0.5% to -2%.

## 2026-05-16 09:00 — PR #3760: H35 AdamW no-decay param groups (fern) — **CLOSED**

- Branch: `charliepai2i24h4-fern/no-decay-groups`
- Result: val_avg=76.64 vs baseline 67.64 → **+13.3% regression** on all 4 splits. Implementation correct (no_decay routing: biases + LN + LS gains + 1D scalars = 1.20% of params).
- Key insight: predicted mechanism empirically absent. ls2 norms FLAT at ~0.42 under no_decay vs baseline's monotone 0.43→0.51 — deep-block ls2 actually grew LESS. At wd=1e-4 × lr ≤5e-4 × LS≈0.5, integrated shrinkage over 4125 steps is ~1e-4 relative — numerically negligible as PR-hypothesis math had warned. Combined with H26 (wd>1e-4 regresses on OneCycleLR), the no-decay carve-out has no headroom.
- Confound: 30-min cap truncated at epoch 11 (vs baseline's 12) due to ~10% unexplained per-epoch slowdown.
- **Closed**: insight added to state — no-decay needs higher WD to matter, which OneCycleLR disallows.

## 2026-05-16 07:50 — PR #3792: H37 OneCycleLR epochs=12 schedule-fit (edward) — **assigned**

- Branch: `charliepai2i24h4-edward/onecycle-epochs12`
- Hypothesis: H24 baseline uses epochs=15 but wall-clock cap gives ~13.8 realized epochs (130s/epoch). Best epoch=12 at LR=9.4e-5; schedule end (epoch 15, LR=0) never reached. H25 diagnostic revealed this truncation mechanism. Single arm: `--epochs 12` fits the schedule to realized budget. `pct_start=0.30` preserved; peak at epoch 3.6, LR=~0 by epoch 12. Complementary to tanjiro H33 (pct_start side) and edward H25 (depth side). Predicted -1% to -3%.

## 2026-05-16 07:50 — PR #3788: H36 channel-weighted surf_loss, surf_p_weight sweep {2,3,5} (frieren) — **assigned**

- Branch: `charliepai2i24h4-frieren/channel-weighted-surf-loss`
- Hypothesis: Primary metric is `mae_surf_p` (pressure only), but `surf_loss` is averaged uniformly across all 3 channels (Ux, Uy, p). p gets 1/3 of gradient signal. Upweighting p in surf_loss redirects gradient where eval cares. Three arms: `surf_p_weight ∈ {2.0, 3.0, 5.0}`. Division by `ch_weights.mean()` preserves surf_loss scale so `surf_weight=10.0` requires no re-tuning. Frieren's own suggestion from H32 analysis. Predicted -1% to -4%.

## 2026-05-16 07:45 — PR #3705: H32 robust loss L1/smooth_l1 (frieren) — **CLOSED**

- Branch: `charliepai2i24h4-frieren/robust-loss`
- Result: L1 arm val_avg=72.88 (+5.24, +7.7%), smooth_l1 β=0.1 arm val_avg=75.72 (+8.08, +11.9%) vs baseline 67.64. Both arms truncated at epoch 11.
- Key insight: MSE in signed_log1p space is already well-matched. signed_log1p compresses the heavy Re tail so log-domain residuals are O(0.1-1); MSE's quadratic penalty doesn't chase outliers. L1's constant-magnitude gradient is noisier near zero, not steadier. Smooth-L1 β=0.1 is 5× MSE curvature inside |err|<0.1, biasing toward easy samples. MAE-eval ↔ MSE-train mismatch is NOT the dominant factor when log1p does the same job.
- **Closed**: negative result, mechanism validated. MSE confirmed optimal for this dataset.

## 2026-05-16 07:45 — PR #3559: H25 n_layers=6 OneCycleLR retest (edward) — **CLOSED**

- Branch: `charliepai2i24h4-edward/n-layers-6`
- Result: Seed1 val_avg=89.67 (+32.6%), Seed2 val_avg=85.57 (+26.5%) vs baseline 67.64. Spread 4.10 (not seed-luck). Test_avg 76.17 (+22.6%).
- Key insight: Deeper model (199s/epoch) gets only 10 epochs in 30-min budget vs baseline's 12 epochs. At epoch 10, LR=2.3e-4 (still in mid-descent). The baseline's gain came from epochs 11-12 at LR=9.4e-5 — a fine-tune tail the deeper model never reaches. Architecture is not broken (H18 n_layers=6 worked at -2.7% on cosine baseline); the OneCycleLR wall-clock interaction is structural. Depth + OneCycleLR is non-orthogonal under 30-min cap.
- **Closed**: structural incompatibility with 30-min budget. H37 (epochs=12) assigned to fix the schedule-budget fit directly.

## 2026-05-16 07:45 — PR #3762: H34 RFF n_freq sweep {16,64} (thorfinn) — **assigned**

- Branch: `charliepai2i24h4-thorfinn/rff-nfreq-sweep`
- Hypothesis: H5 merged RFF at n_freq=32 (+3.9%). H28 closed (preprocess width not the bottleneck). Back to input encoding: does more/fewer Fourier frequencies help? Arm1: n_freq=16 (32-dim), Arm2: n_freq=64 (128-dim, preprocess input changes 86→150). Predicted: n_freq=64 wins for richer spatial basis.

## 2026-05-16 07:45 — PR #3760: H35 AdamW no-decay param groups (fern) — **assigned**

- Branch: `charliepai2i24h4-fern/no-decay-groups`
- Hypothesis: Current AdamW applies WD uniformly including LayerScale gains, biases, LN. Canonical split (DeiT-III / ConvNeXt / CaIT): only Linear weights get decayed; biases, LN, LayerScale go to no_decay group. Single arm, wd=1e-4 (current default). Predicted -0.5% to -2%.

## 2026-05-16 07:30 — PR #3627: H28 widen preprocess MLP w512+dropout=0.1 (thorfinn) — **CLOSED**

- Branch: `charliepai2i24h4-thorfinn/preprocess-mlp`
- Result: val_avg=81.97 vs baseline 67.64 → **+21% regression**. Two changes stacked (width 256→512 AND dropout 0.0→0.1); can't disentangle. Per-epoch slowdown 13% (170s vs 150s) ate budget. val_single degressed MOST (+33%) — opposite of prediction. Current 256-wide preprocess is NOT the bottleneck.
- **Closed**: clear regression.

## 2026-05-16 07:30 — PR #3583: H26 wd=0.001 retest on OneCycleLR (fern) — **CLOSED**

- Branch: `charliepai2i24h4-fern/weight-decay-sweep`
- Result: val_avg=75.97 vs baseline 67.64 → **+12.3% regression**. All splits worse.
- Key insight: WD and OneCycleLR are NOT orthogonal. The integrated lr×wd AdamW shrinkage under OneCycleLR's long low-LR tail over-shrinks during fine-tuning. Same mechanism as H27 (max_lr), different lever: schedule's effective WD depends on the LR trajectory.
- **Closed**: weight_decay=1e-4 (current default) confirmed for OneCycleLR baseline.

## 2026-05-16 07:30 — PR #3742: H33 OneCycleLR pct_start sweep {0.10,0.15,0.20} (tanjiro) — **assigned**

- Branch: `charliepai2i24h4-tanjiro/pct-start-sweep`
- Hypothesis: H27 proved max_lr>5e-4 regresses because the fine-tune tail never starts in 30-min budget. pct_start compression gives more budget to the descent phase. Arms: {0.10, 0.15, 0.20} vs baseline 0.30. Predicted -1% to -4%.

## 2026-05-16 07:30 — PR #3625: H27 OneCycleLR max_lr sweep (tanjiro) — **CLOSED**

- Branch: `charliepai2i24h4-tanjiro/max-lr-sweep`
- Results: max_lr=1e-3 seed1 val=73.22, seed2 val=83.26; max_lr=2e-3 val=70.98. All worse than H24 baseline (67.64).
- Mechanism: At 30-min cap, ~11/15 epochs complete. Higher max_lr "burns" budget in high-LR exploration — LR at ep11 is 6.3e-4 for 2e-3 arm (higher than H24's peak!), so fine-tune tail never starts. High seed variance at 1e-3 (73 vs 83).
- **Closed**: clear regression, mechanistically understood. max_lr=5e-4 is near-optimal for the 30-min budget.

## 2026-05-16 05:30 — PR #3705: H32 robust regression loss L1 vs smooth_l1 (frieren) — **assigned**

- Branch: `charliepai2i24h4-frieren/robust-loss`
- Hypothesis: Replace MSE on signed_log1p targets with L1 (arm 1) and smooth_l1 beta=0.1 (arm 2). Motivation: train loss (MSE) mismatches eval metric (MAE). L1 gives constant gradient magnitude — robust to heavy-tailed outliers on high-Re, OOD, and cam_rc splits. Orthogonal to all merged mechanisms.

## 2026-05-16 05:25 — PR #3517: H19 DropPath=0.20 + LayerScale + OneCycleLR (frieren) — **CLOSED**

- Branch: `charliepai2i24h4-frieren/droppath-rebase`
- Result: val_avg=89.08 vs baseline 67.64 → **+31.7% regression** on full stack. Three retests (LayerScale seed1, seed2, OneCycleLR arm) all negative.
- Mechanism: LayerScale's ls2 gamma reversed depth pattern from monotone-growth to monotone-decay under DropPath's linear depth schedule. DropPath (deeper blocks suppressed more) conflicts with LayerScale (deeper FFN blocks normally activated more). Combined with geom_gate failure (~0.05 vs baseline 0.13) and OneCycleLR's tight convergence budget, DropPath destabilized the entire stack.
- Note: H19 was a real win (-8.3%) on H15 SwiGLU baseline — the win was stack-specific. DropPath does NOT compose with LayerScale + OneCycleLR.
- **Closed**: clear regression, mechanistically understood.

## 2026-05-16 04:45 — PR #3687: H30 gradient clipping max_norm=1.0 (nezuko) — **assigned**

- Branch: `charliepai2i24h4-nezuko/grad-clip`
- Hypothesis: 2-line change adds `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before optimizer.step(). Targets OneCycleLR high-LR phase stability (epochs 5-6, lr ≈ 5e-4). Standard transformer practice (DeiT max_norm=1.0). Conservative, no params, no overhead.

## 2026-05-16 04:45 — PR #3686: H31 SAM ρ=0.05 (askeladd) — **assigned**

- Branch: `charliepai2i24h4-askeladd/sam-optimizer`
- Hypothesis: Inline SAM optimizer wrapper (Foret 2020). Two forward+backward passes per step, weight perturbation toward sharp direction. Replaces EMA direction (which closed). Predicted -3% to -8% from 67.64 — works on gradient side, orthogonal to all merged mechanisms. Step time ~1.8× baseline.

## 2026-05-16 04:30 — PR #3197: H8v3 EMA v4 on OneCycleLR (askeladd) — **CLOSED**

- Branch: `charliepai2i24h4-askeladd/ema`
- Result: val_avg=87.22, test_avg=78.20 vs baseline 67.64 → **+29% regression**. Live val 77.28 also above baseline.
- Mechanism: EMA β=0.999 needs ~1000 steps of stable low-LR to catch up to live. OneCycleLR's late phase drops LR fast (epoch 11 lr=1.58e-4 vs cosine equivalent ~2.5e-4). Combined with 11-epoch budget (one fewer than baseline due to dual val pass), EMA shadow can't converge.
- Student diagnosis correct: EMA only beats live at epoch 10 (one epoch) before crossing back. Confirms EMA + OneCycleLR + 30-min cap is fundamentally incompatible.
- **Closed**: useful negative (EMA + cosine v3 was -7.5% on its baseline; EMA + OneCycleLR fails).

## 2026-05-16 04:30 — PR #3628: H29 per-block geom_proj (nezuko) — **CLOSED**

- Branch: `charliepai2i24h4-nezuko/per-block-geom-proj`
- Result: val_avg=81.10 vs baseline 67.64 → **+20% regression**. All splits worse, including cam_rc target (+15.77).
- Mechanism: 5× duplication of geom_proj MLPs (+144K params) caused gradient interference. geom_gates ended with alternating signs [+0.048, -0.033, +0.034, -0.039, +0.062] — optimizer never settled. Per-block specialization didn't emerge in 15 epochs.
- Per-block projection norms 5.05-6.49 — all similar magnitude, no specialization.
- **Closed**: useful negative (depth-specialized geom conditioning needs longer training or pre-training shared projection first).

## 2026-05-16 04:30 — PR #3559: H25 n_layers=6 (edward) — **SENT BACK FOR ONECYCLE RETEST**

- Branch: `charliepai2i24h4-edward/n-layers-6`
- Result on H18 baseline (79.52): val_avg=77.39 (-2.7%), test_avg=68.81 (~tied), best_epoch=10.
- **Key finding**: Cam_rc recovered from 93.29 → 85.81 (-8.0%), undoing H18's regression on that split.
- Diagnostics: New block 5 ls2_norm=0.465 (highest of all blocks, despite gamma init=1e-6). geom_gates monotone growth with depth [0.044→0.056].
- Status: rebased onto OneCycleLR baseline 67.64; sent back for single-arm retest with `--n_layers 6 --use_onecycle`. Predicted val_avg 62-68 if compose works.

## 2026-05-16 04:30 — PR #3583: H26 weight_decay sweep {0.001, 0.01, 0.05} (fern) — **SENT BACK FOR ONECYCLE RETEST**

- Branch: `charliepai2i24h4-fern/weight-decay-sweep`
- Result on H18 baseline (79.52):
  - **wd=0.001: val=73.92 (-7.04%), test=64.98 (-5.76%) — winner**
  - wd=0.01: val=76.91 (-3.28%)
  - wd=0.05: val=77.30 (-2.79%)
- Inverted U-shape: smaller wd is better. Canonical DeiT-III (0.05) underperforms vs modest 10× bump from current 1e-4 to 0.001.
- Status: sent back for single-arm wd=0.001 retest on OneCycleLR baseline. Predicted val_avg 62-67.

## 2026-05-16 03:30 — PR #3539: H23 slice_num sweep {32, 64, 128} (alphonse) — **MASSIVE WIN, SENT BACK FOR REBASE+RETEST**

- Branch: `charliepai2i24h4-alphonse/slice-num-sweep`
- Results on SwiGLU baseline (80.21):
  - **slice_num=32: val=62.63 (-22%), test=55.41 (-24%)** — best, all 4 splits monotonically better
  - slice_num=64 verify: val=71.15 (vs claimed 80.21 — ~10% seed variance evidence)
  - slice_num=128: val=75.47
- Monotone trend: smaller slice_num is uniformly better. Mechanism: fewer slices = simpler PhysicsAttention routing on 1499-sample dataset.
- VRAM: slice_num=32 used 48.8 GB / 96 GB — well within budget.
- **Sent back for single-arm rebase + retest at slice_num=32 on current OneCycleLR baseline (67.64).** Predicted: val_avg ≤ 60 if effects compose orthogonally.

## 2026-05-16 03:30 — PR #3627: H28 deeper preprocess MLP (thorfinn) — **REVISED, student found premise was wrong**

- Branch: `charliepai2i24h4-thorfinn/preprocess-mlp`
- Student finding: current preprocess IS already 2-layer (86→256→GELU→Dropout(0.0)→128 via MLP class with n_layers=0, 55,168 params). Original PR proposed identical structure.
- **Revised to option 3 (widen)**: change to width=512 (86→512→GELU→Dropout(0.1)→128, ~109K params, 2× wider bottleneck). Predicted -1% to -3% from 67.64, primarily helping val_single_in_dist (80.32, the current bottleneck split).

## 2026-05-16 02:15 — PR #3628: H29 per-block geom_proj (nezuko) — **assigned**

- Branch: `charliepai2i24h4-nezuko/per-block-geom-proj`
- Hypothesis: Replace shared geom_proj MLP (all blocks share one projection) with per-block independent projections (each of 5 blocks gets its own MLP(11,256,128)). +144K params. Target cam_rc recovery — that split needs block-specific geometric representation.

## 2026-05-16 02:15 — PR #3627: H28 deeper preprocess MLP (thorfinn) — **assigned**

- Branch: `charliepai2i24h4-thorfinn/preprocess-mlp`
- Hypothesis: Replace single linear preprocess layer (86→128) with 2-layer MLP (86→256→128, GELU, dropout=0.1). +43K params. May help with heterogeneous input features (RFF + velocity + geometry + Re).

## 2026-05-16 02:15 — PR #3625: H27 OneCycleLR max_lr sweep {1e-3, 2e-3} (tanjiro) — **assigned**

- Branch: `charliepai2i24h4-tanjiro/max-lr-sweep`
- Hypothesis: H24 used max_lr=5e-4 (same as AdamW lr, not true super-convergence). Test {1e-3, 2e-3} — literature suggests 3-10× base lr for saddle-escape super-convergence. H24 was also truncated at epoch 12/15.

## 2026-05-16 02:00 — PR #3540: H24 OneCycleLR super-convergence (tanjiro) — **MERGED, new best**

- Branch: `charliepai2i24h4-tanjiro/onecycle`
- Result: val_avg=67.64, test_avg=62.12. Epoch 12/15 (truncated, trajectory still descending). **-15.7% vs 80.21 SwiGLU baseline** — biggest single-PR win of the round.

| Split | H18 baseline | H24 OneCycleLR | Delta |
|---|---:|---:|---:|
| val_single_in_dist | 104.62 | 80.32 | -23.1% |
| val_geom_camber_rc | 93.29 | 81.81 | -12.3% |
| val_geom_camber_cruise | 50.00 | 44.46 | -11.1% |
| val_re_rand | 70.14 | 63.96 | -8.8% |
| **val_avg** | **79.52** | **67.64** | **-14.9%** |
| test_avg | 68.95 | 62.12 | -9.9% |

- Implementation notes: student correctly set `--epochs 15` (not default 50) for schedule sizing; set `cycle_momentum=False` (AdamW-safe).
- LR curve confirms classic OneCycleLR: rise epochs 1-5 to peak 5e-4, cosine fall 5-12. Best epoch=12 (still descending at cutoff → lower bound).
- Follow-up: tanjiro H27 now tests max_lr={1e-3, 2e-3} for true super-convergence.
- **New baseline: val_avg=67.64, test_avg=62.12.**

## 2026-05-16 02:00 — PR #3538: H22 LR warmup + cosine eta_min (thorfinn) — **CLOSED as redundant**

- Branch: `charliepai2i24h4-thorfinn/lr-warmup`
- Result: val_avg=69.01 (-14.0% vs baseline), test_avg=60.97 (-16.7%). Every split improved. Strong result.
- **Closed as redundant**: OneCycleLR (#3540) integrates warmup (pct_start=0.3 = built-in linear ramp) and achieved lower val (67.64 < 69.01). The two schedulers are mutually exclusive. Warmup independently validated warmup direction.

## 2026-05-16 02:00 — PR #3421: H14 cosine T_max sweep (nezuko) — **CLOSED as obsolete**

- Branch: `charliepai2i24h4-nezuko/cosine-tmax`
- Result (pre-GALE baseline): arm T_max=14 val=88.43, arm T_max=20 val=90.21. Both beat T_max=50 (92.80) on stale baseline.
- **Closed as obsolete**: OneCycleLR merged, cosine scheduler replaced entirely. T_max tuning irrelevant.

## 2026-05-16 02:00 — PR #3197: H8v3 EMA decay=0.999 (askeladd) — **SENT BACK FOR RETEST on OneCycleLR baseline**

- Branch: `charliepai2i24h4-askeladd/ema`
- Prior result (v3 on SwiGLU+cosine): val_avg=74.18, test_avg=66.62. Rebased onto H18 LayerScale + OneCycleLR (CLEAN).
- Status: sent back for retest because schedule change (cosine→OneCycleLR) fundamentally changes EMA dynamics. EMA+OneCycleLR's low-LR tail (where EMA is most powerful) may give a larger or different gain. Required: single-arm retest with `--use_onecycle --onecycle_pct_start 0.3 --epochs 15`.

## 2026-05-16 01:30 — PR #3467: H17 attention dropout {0.05, 0.10} (fern) — **CLOSED**

- Branch: `charliepai2i24h4-fern/attention-dropout`
- Result vs H13 GALE reference (85.16, stale at time of assignment): arm 0.05 val=85.98 (+0.97%), arm 0.10 val=87.31 (+2.52%). Both regress.
- Against current baseline (H18 LayerScale, 79.52): both arms regress severely (+8.1% / +9.6%).
- Arm 0.05 had interesting test-side improvement (-1.3% test, driven by test_single_in_dist -4.98%) but val ranking metric regressed.
- **Decision**: Close. Attention routing softmax (slice_num=64, 4 heads) doesn't benefit from dropout noise on weights — softmax is already low-entropy, so dropout corrupts more signal than overfitting it prevents.
- Useful contrast: confirms H19 DropPath (residual-side) >> H17 attn dropout (weight-side) for this architecture.

## 2026-05-16 01:30 — PR #3517: H19 DropPath {0.10, 0.20} on SwiGLU baseline (frieren) — **SENT BACK FOR REBASE**

- Branch: `charliepai2i24h4-frieren/droppath`
- Result vs H15 SwiGLU baseline (80.21): arm 0.10 val=75.65 (-5.7%), **arm 0.20 val=73.55 (-8.3%), test=67.05 (-8.4%)** — best raw metric this round.
- best_epoch=11 with trajectory still descending sharply at timeout (sched T_max=15). Per-split: 0.20 dominates with -12.4% on val_single_in_dist (the dominant in-dist bottleneck).
- Status: CONFLICTING (built on SwiGLU, advisor branch has H18 LayerScale). **Sent back for rebase + single-arm retest at max_drop_path=0.20 to confirm composition with LayerScale.**
- Predicted compose value: val_avg ≤ 75 (likely 70-73). If confirmed, becomes the winner over askeladd EMA (74.18). Both target orthogonal axes (residual stochastic depth vs weight averaging) so could potentially compose.

## 2026-05-16 01:45 — PR #3583: H26 AdamW weight_decay sweep {0.001, 0.01, 0.05} (fern) — **assigned**

- Branch: `charliepai2i24h4-fern/weight-decay-sweep`
- Hypothesis: Current `weight_decay=1e-4` is two orders of magnitude below standard transformer range (0.01–0.1). In the small-data, regularization-bound regime (1499 samples, 880K params), this should compound with FFN dropout, DropPath, and LayerScale. Three arms: {0.001, 0.01, 0.05}. Predicted: -1% to -4% val_avg.

## 2026-05-16 00:40 — PR #3559: H25 n_layers=6 deeper Transolver (edward) — **assigned**

- Branch: `charliepai2i24h4-edward/n-layers-6`
- Hypothesis: Add a 6th TransolverBlock to the existing 5-block stack. With LayerScale init=1e-6, the new block starts as near-identity and activates only if useful. H18 diagnostics showed block 4 is most active (ls2=0.51 vs 0.43 at block 0) — suggests the model wants more depth. ~880K → ~1.04M params. 
- Single arm. Target: val_avg < 79.52 (new H18 baseline). Also beat 74.18 (askeladd EMA pending).

## 2026-05-16 00:30 — PR #3514: H18 LayerScale residual scaling (edward) — **MERGED, new best**

- Branch: `charliepai2i24h4-edward/layerscale`
- Result: val_avg=79.52, test_avg=68.95. Terminated at 30-min cap epoch 11.

| Split | H15 baseline | H18 LayerScale | Delta |
|---|---:|---:|---:|
| val_single_in_dist | 104.46 | 104.62 | +0.15% (flat) |
| val_geom_camber_rc | 88.50 | 93.29 | **+5.42% (worse)** |
| val_geom_camber_cruise | 53.88 | 50.00 | **-7.21%** |
| val_re_rand | 74.00 | 70.14 | **-5.22%** |
| **val_avg** | **80.21** | **79.52** | **-0.86%** |
| test_avg | 73.20 | 68.95 | **-5.80%** |

- LayerScale gamma diagnostics: ls2 (FFN) monotone 0.43→0.51 (textbook depth-activation). ls1 (attn) U-shaped 0.19/0.13/0.14/0.17/0.21 (attention suppressed mid-stack). All gammas escaped init=1e-6 (grew ~5 orders of magnitude).
- **Analysis**: Val gain (-0.86%) is within seed variance, but test_avg gain (-5.80%) is a clear generalization win. LayerScale's FFN path shows textbook behavior; attention U-shape is novel — mid-stack blocks suppress attention more. Cam_rc regression (+5.4%) is concerning — same pattern as FiLM. May be related to mid-block attention suppression for OOD geometry.
- **Decision**: Merged. Small val gain but clear test_avg improvement. +1,280 params overhead negligible.
- New baseline: val_avg=79.52, test_avg=68.95.

## 2026-05-16 00:30 — PR #3540: H24 OneCycleLR super-convergence (tanjiro) — **assigned**

- Branch: `charliepai2i24h4-tanjiro/onecycle-lr`
- Hypothesis: OneCycleLR (Smith & Topin 2019) replaces CosineAnnealingLR with a phased super-convergence schedule: rising LR (warmup-like phase), rapid fall (steeper than cosine), then very-low LR fine-tune phase. `max_lr=5e-4, div_factor=25, final_div_factor=1e4, pct_start=0.3`. Different paradigm from cosine — single structured cycle with rapid final fall.
- Single arm with per-batch stepping. Target: val_avg < 80.21.
- References: Smith & Topin arXiv:1708.07120.

## 2026-05-16 00:30 — PR #3539: H23 slice_num sweep {32, 64, 128} (alphonse) — **assigned**

- Branch: `charliepai2i24h4-alphonse/slice-num-sweep`
- Hypothesis: PhysicsAttention has `slice_num=64` (Transolver default). Test {32, 64, 128} to find optimum on 1499-sample dataset. Lower may regularize attention routing; higher may capture finer boundary-layer features but overfit.
- 3 arms. Target: val_avg < 80.21.

## 2026-05-16 00:30 — PR #3538: H22 LR warmup + cosine eta_min=1e-5 (thorfinn) — **assigned**

- Branch: `charliepai2i24h4-thorfinn/warmup-cosine`
- Hypothesis: Linear LR warmup over 2 epochs (1e-5 → 5e-4) before cosine anneal to eta_min=1e-5. Addresses early-training noise observed in all PR curves (epochs 1-3 typically show large val_avg drops). Synergistic with H18 LayerScale (edward, in flight) and SwiGLU's gate init (random).
- Single arm. Target: val_avg < 80.21.
- References: He et al. "Bag of Tricks" CVPR 2019, Goyal et al. arXiv:1706.02677.

## 2026-05-16 00:25 — PR #3461: H16 FiLM Geometry Conditioning (tanjiro) — **CLOSED**

- Branch: `charliepai2i24h4-tanjiro/film-geom-cond`
- Result: val_avg=84.36 (on H13 baseline 85.16), test_avg=77.74. -0.94% vs H13 GALE baseline but +5.2% vs current 80.21 baseline (run was on pre-SwiGLU code).

| Split | H13 baseline | H16 FiLM | Delta |
|---|---:|---:|---:|
| val_single_in_dist | 106.16 | 106.44 | +0.28 (tied) |
| val_geom_camber_rc | 92.10 | 95.28 | **+3.18 (worse)** |
| val_geom_camber_cruise | 61.36 | 57.47 | -3.89 (better) |
| val_re_rand | 81.01 | 78.26 | -2.74 (better) |
| **val_avg** | **85.16** | **84.36** | **-0.94%** |
| test_avg | 77.61 | 77.74 | +0.13 |

- FiLM gates: scale gates dominated shift gates by ~10× (0.22-0.25 vs 0.01-0.04). Scale gate highest at block 0 (not monotone with depth).
- **Analysis**: Multiplicative path overfits in-distribution feature importance and structurally regresses on the hardest OOD split (camber_rc). H13 GALE's additive shift was already optimal for OOD geometry. Multiplicative scaling reverses GALE's gain. SwiGLU's gates already implement per-dimension multiplicative feature modulation; FiLM's scale path is redundant + antagonistic to GALE additive injection.
- **Decision**: Closed. Camber_rc regression is mechanistic, not seed variance. Rerunning on post-SwiGLU stack wouldn't change the fundamental tradeoff. Tanjiro reassigned to H24 OneCycleLR.

## 2026-05-16 00:25 — PR #3184: H1 LinearNO ablation (alphonse) — **CLOSED, diagnostic**

- Branch: `charliepai2i24h4-alphonse/linearno-no-interslice-qkv`
- v1 result (pre-RFF code): val_avg=144.93, test_avg=131.04
- v2 result (post-RFF + dropout + log1p baseline): val_avg=93.02, test_avg=84.96

| Configuration | val_avg | delta vs current 80.21 |
|---|---:|---:|
| Baseline (with PhysicsAttention QKV) | 80.21 | — |
| **LinearNO (no inter-slice QKV)** | **93.02** | **+16.0%** |

- **Analysis**: Diagnostic ablation. Inter-slice QKV contributes ~13 absolute val_avg points. Largest single architectural contributor measured. Removing it costs more than the cumulative loss-side gains (log1p + dropout = ~30 points each). Confirms attention is essential and the baseline architecture is doing real work.
- **Decision**: Closed (expected diagnostic regression). Alphonse reassigned to H23 slice_num sweep.

## 2026-05-16 00:25 — PR #3417: H11b log1p alpha sweep {0.5, 1.0, 2.0} (thorfinn) — **CLOSED, α=1.0 optimal**

- Branch: `charliepai2i24h4-thorfinn/log1p-alpha-sweep`
- 3-arm sweep on pre-GALE pre-SwiGLU code:

| α | val_avg | test_avg | delta vs α=1.0 |
|---|---:|---:|---:|
| 0.5 | 103.22 | 94.30 | +14.9% |
| **1.0 (current default)** | **89.85** | **80.01** | — |
| 2.0 | 95.99 | 88.31 | +6.8% |

- **Analysis**: α=1.0 wins on every val and test split. No mixed result. α=0.5 underfits high-|y| tail; α=2.0 over-compresses and flattens loss landscape. The optimum is firmly at α=1.0 (already baked into baseline).
- **Decision**: Closed (α=1.0 confirmed, no follow-up needed — α is loss-side and orthogonal to architectural changes, so optimum won't shift under SwiGLU/GALE). Thorfinn reassigned to H22 LR warmup.

## 2026-05-16 00:20 — PR #3197: H8v3 EMA on combined baseline (askeladd) — **SENT BACK FOR REBASE — WINNER**

- Branch: `charliepai2i24h4-askeladd/ema-weights-decay-0p999`
- Result: **val_avg=74.178, test_avg=66.62** — **NEW BEST candidate, -7.5% vs current 80.21!**
- v3 ran on FULL combined post-H15 stack (rebased onto advisor at 22:41 UTC, after H15 SwiGLU merged).

| Split | Baseline (H15) | EMA v3 | Delta |
|---|---:|---:|---:|
| val_single_in_dist | 104.46 | 98.18 | -6.28 |
| val_geom_camber_rc | 88.50 | 81.38 | -7.12 |
| val_geom_camber_cruise | 53.88 | 49.79 | -4.09 |
| val_re_rand | 74.00 | 67.37 | -6.63 |
| **val_avg** | **80.21** | **74.178** | **-7.5%** |
| test_avg | 73.20 | 66.62 | -9.0% |

- Best epoch 11 (EMA). EMA wins on 3/4 splits over live; live wins on single_in_dist (98.18 EMA vs 92.80 live) but EMA wins on average.
- **Status**: GitHub mergeStateStatus=DIRTY (CONFLICTING) — needs rebase. Sent back with detailed merge conflict resolution guidance (the SwiGLU MLP class change is the expected conflict in train.py). Result is already on post-H15 stack; no rerun needed. Will merge after force-push.

## 2026-05-15 23:00 — PR #3517: H19 DropPath stochastic depth (frieren) — **assigned**

- Branch: `charliepai2i24h4-frieren/droppath`
- Hypothesis: DropPath (stochastic depth) randomly drops entire block residual contributions per-sample. Targets block co-adaptation: after H15 SwiGLU, each block is more expressive — DropPath forces independent learning. Linear drop-rate schedule from 0.0 (block 0) to max_drop_prob (block 4). Complementary to FFN dropout=0.1 (within-block) and LayerScale (H18, edward, block-output scaling).
- Two arms: max_drop_prob ∈ {0.10, 0.20}. DeiT survival_prob=0.9 (0.10 drop) as reference.
- Target to beat: val_avg/mae_surf_p < 80.21 (current best, H15 SwiGLU).
- References: Huang et al. ECCV 2016, Touvron et al. DeiT ICML 2021.

## 2026-05-15 23:00 — PR #3514: H18 LayerScale residual scaling (edward) — **assigned**

- Branch: `charliepai2i24h4-edward/layerscale`
- Hypothesis: LayerScale (CaIT, Touvron 2021) adds learnable per-channel diagonal scaling on each residual connection: `x += gamma * Block(norm(x))`, where gamma is init at 1e-6. Starts as near-identity, gradually activates depth. Synergistic with SwiGLU: each block now has higher capacity (gated FFN), and LayerScale's controlled depth activation should help the optimizer benefit from that capacity.
- Single arm: SwiGLU + LayerScale (1280 extra scalars, ~843K → ~844K params).
- Diagnostic: log gamma norms per block to verify monotone growth (deeper = larger gamma).
- Target to beat: val_avg/mae_surf_p < 80.21 (current best, H15 SwiGLU).
- References: Touvron et al. CaIT ICCV 2021 (arXiv:2103.17239), DeiT III (2022).

## 2026-05-15 22:50 — PR #3318: H6v2 Grad clip + SGDR (frieren) — **CLOSED, SGDR doesn't compose**

- Branch: `charliepai2i24h4-frieren/gradclip-sgdr`
- H6v1 result on old baseline: val_avg=99.85 (-18.7% vs RFF baseline 122.81). Grad clip eliminated oscillation; SGDR drove best to cosine bottom (epoch 10). Strong mechanism.
- H6v2 result on H13 baseline (post-GALE): val_avg=86.21 vs current baseline 80.21.

| Split | H13 baseline (85.16) | H6v2 | Delta |
|---|---:|---:|---:|
| val_single_in_dist | — | 103.22 | — |
| val_geom_camber_rc | — | 97.23 | — |
| val_geom_camber_cruise | — | 62.58 | — |
| val_re_rand | — | 81.81 | — |
| **val_avg** | **85.16** | **86.21** | **+1.2%** |
| test_avg | — | 76.42 | — |

- Note: H15 SwiGLU merged while v2 was running; current baseline is now 80.21, making H6v2 (86.21) +7.5% above target.
- **Analysis**: SGDR's mechanism requires ≥2 LR restart cycles; only 1 fires in 13 realized epochs. Log1p + dropout absorb grad-clip's stability benefit. Sub-additive composition. Student's honest analysis confirmed the mechanism: SGDR restart fired cleanly (lr jump at epoch 11), but the combined baseline already reaches a lower minimum via architectural mechanisms. Closing the SGDR direction.
- **Decision**: Closed (+7.5% regression vs current 80.21 baseline). Student reassigned to H19 DropPath.

## 2026-05-15 22:45 — PR #3421: H14 Cosine T_max sweep (nezuko) — **sent back for v2 (single-arm retest)**

- Branch: `charliepai2i24h4-nezuko/cosine-tmax-alignment`
- Hypothesis: Align cosine T_max to realized epoch budget (~14 epochs under 30-min cap). Two arms: T_max=14 (full anneal) and T_max=20 (partial anneal) vs T_max=50 baseline.
- Both arms ran on PRE-GALE codebase (before T_max=15 was baked in by H13 merge).

| T_max | val_avg | test_avg | best_epoch | delta vs old 92.80 |
|---|---:|---:|---:|---:|
| 50 (old base) | 92.80 | 84.11 | 14 | — |
| **14 (arm 1)** | **88.43** | **80.24** | **14** | **-4.7%** |
| 20 (arm 2) | 90.21 | 78.59 | 14 | -2.8% |
| 15 (new baked baseline) | 85.16 | — | 14 | — |

- Metrics: `models/model-charliepai2i24h4-nezuko-cosine-t14-etamin1e5-20260515-202657/metrics.jsonl`, `models/model-charliepai2i24h4-nezuko-cosine-t20-etamin1e5-20260515-212210/metrics.jsonl`
- **Analysis**: Direction validated (T_max=14 > T_max=20 > T_max=50). However, baseline has shifted twice (92.80 → 85.16 → 80.21) since assignment. Arms ran on pre-GALE pre-SwiGLU code and cannot be directly compared. Student correctly identified that T_max=15 is now hardcoded, making T_max=14 + eta_min=1e-5 the cheapest next test on the current code.
- **Decision**: Sent back for single-arm v2 retest: T_max=14 + eta_min=1e-5 on current post-H15 train.py. Target: val_avg < 80.21. Expected delta modest (sub-percent, given T_max=14 vs T_max=15 is marginal change); confirm vs seed variance.

## 2026-05-15 21:30 — PR #3224: H13 Geom-cond GALE (tanjiro) — **MERGED, new best**

- Branch: `charliepai2i24h4-tanjiro/geom-cond-additive`
- Hypothesis: Persistent additive geometry conditioning at every TransolverBlock, GALE-style. Global dims 13-23 (Re, AoA, NACA params, gap, stagger) extracted once per sample and projected via MLP. Learnable per-block scalar gates init at 0 (identity start). Predicted -3% to -8%, camber splits expected to benefit most.
- Round 2 results (full combined baseline stack + T_max=15 cosine alignment):

| Split | Baseline (92.80) | H13 v2 | Delta |
|---|---:|---:|---:|
| val_single_in_dist | 115.48 | 106.160 | -8.1% |
| val_geom_camber_rc | 105.48 | 92.098 | **-12.7%** ← biggest |
| val_geom_camber_cruise | 63.87 | 61.360 | -3.9% |
| val_re_rand | 86.36 | 81.005 | -6.2% |
| **val_avg** | **92.80** | **85.156** | **-8.2%** |
| test_avg | 84.11 | 77.613 | -7.7% |

- Metrics: `models/model-charliepai2i24h4-tanjiro-geom-cond-v2-restrat-rff-tmax15-20260515-193031/metrics.jsonl`
- Learned gates: `[-0.05, -0.11, -0.13, -0.14, -0.15]` — monotone with depth, all non-zero. Mechanism active at every block.
- **Analysis**: GALE mechanism confirmed working — camber_rc split benefited most (-12.7%) as predicted (OOD geometry interpolation). T_max=15 alignment was critical: round 1 (T_max=50) showed oscillating val_avg late-training; round 2 with T_max=15 showed monotone descent to epoch 14 best. New baseline: 85.156.
- **Note on T_max**: tanjiro's merge baked T_max=15 into train.py. Nezuko's H14 (CLI --cosine_t_max) needs to handle this correctly on rebase.

## 2026-05-15 22:35 — PR #3423: H15 SwiGLU MLP (edward) — **MERGED, new best (-5.8%)**

- Branch: `charliepai2i24h4-edward/swiglu-mlp`
- Hypothesis: Replace GELU FFN with SwiGLU gated FFN: `linear_in → silu(gate)*value → dropout(0.1) → linear_out`. Gate-modulated multiplication allows per-dimension feature attenuation. ~+165K params (678K→843K for full model with geom-cond).
- Two runs committed (both beat baseline):

| Run | val_avg | test_avg | best_epoch |
|---|---:|---:|---:|
| 20260515-202620 (run 1) | 89.48 | 79.71 | 11 |
| **20260515-212619 (run 2, primary)** | **80.21** | **73.20** | **10** |
| Baseline (H13) | 85.16 | 77.61 | 14 |

- Per-split (run 2): single=104.46, rc=88.50, cruise=53.88, re=74.00
- **Analysis**: OOD splits gained 1.5–1.7× more than in-dist (rc −16.1%, re −14.3% vs single −9.5%). Gate modulation reduces co-adaptation similarly to dropout but structurally. Best epoch at 10 (29% faster convergence). ~10% seed variance between runs — notable for future reference.
- New baseline: 80.21.

## 2026-05-15 21:36 — PR #3467: H17 Attention dropout sweep 0.05/0.10 (fern) — **assigned (post H12b close)**

- Branch: `charliepai2i24h4-fern/attention-dropout`
- Hypothesis: PhysicsAttention has dropout=0.0 (fully unregularized). Sweep attn_dropout ∈ {0.05, 0.10} while keeping FFN dropout=0.1 fixed. Attention dropout regularizes the slice-token routing (different axis from FFN dropout). Predicted -2-5%.
- Two arms: attn_dropout=0.05 and attn_dropout=0.10 via new `--attn_dropout` CLI arg wired into model_config.
- Target to beat: val_avg/mae_surf_p < 85.16.

## 2026-05-15 21:30 — PR #3375: H12b dropout rate sweep (fern) — **CLOSED, 0.10 confirmed optimal**

- Branch: `charliepai2i24h4-fern/fern/dropout-sweep`
- All 3 arms completed on OLD baseline (112.49): dropout ∈ {0.05, 0.15, 0.20}

| dropout | val_avg | delta vs 0.10 |
|---|---:|---:|
| 0.05 | 116.77 | +3.8% |
| **0.10 (baseline)** | **112.49** | — |
| 0.15 | 118.47 | +5.3% |
| 0.20 | 120.05 | +6.7% |

- **Analysis**: Clear U-shape minimum at 0.10. All alternatives regress. Basin is narrow — even 0.05 regresses. FFN dropout=0.10 confirmed as optimal. Closed without rerun on new baseline because the result is mechanistically clear (orthogonal to log1p and geom-cond) and not worth 30 min GPU time to repeat.

## 2026-05-15 21:30 — PR #3461: H16 FiLM geom-cond (tanjiro) — **assigned (post H13 merge)**

- Branch: `charliepai2i24h4-tanjiro/film-geom-cond`
- Hypothesis: Extend H13 additive geom-cond to FiLM: `fx ← fx ⊙ (1 + γ_i(ctx)) + β_i(ctx)`. Shared film_proj MLP(11, 256, 256) outputs 2×n_hidden (split into gamma/beta). Per-block scale/shift gates init at 0 (identity start). ~+33K params vs current baseline.
- Single arm: beat val_avg < 85.16.
- Predicted delta: -2-6% on top of additive baseline.

## 2026-05-15 19:35 — PR #3423: H15 SwiGLU MLP (edward) — **assigned (idle slot fill)**

- Branch: `charliepai2i24h4-edward/swiglu-mlp`
- Hypothesis: Replace standard `linear → GELU → linear` FFN in `TransolverBlock.mlp` with SwiGLU gated `linear → silu(gate) * value → linear`. H12 (dropout) showed FFN is high-leverage; SwiGLU targets the same sub-layer structurally. ~50% more MLP params (~165K total over 5 blocks).
- Single arm. Keep mlp_ratio=2, dropout=0.1, no other change.
- Target to beat: val_avg/mae_surf_p < 92.80.
- Predicted delta: -2% to -5%. Composes with log1p (loss-side), Re-strat (sampler), RFF (input). Orthogonal mechanism.

## 2026-05-15 19:35 — PR #3421: H14 Cosine T_max + eta_min alignment (nezuko) — **assigned (idle slot fill, fresh hypothesis post-H9 close)**

- Branch: `charliepai2i24h4-nezuko/cosine-tmax-alignment`
- Hypothesis: With 30-min cap → ~14 epochs realized, T_max=50 means cosine barely anneals. Late-stage low-LR is where cosine gains accrue. Sweep T_max ∈ {14, 20} with eta_min=1e-5.
- Two arms. T_max=14 (full anneal) and T_max=20 (moderate anneal).
- Target to beat: val_avg/mae_surf_p < 92.80.
- Predicted delta: -2% to -6%. High-confidence direction (multiple students flagged independently). Orthogonal to model/loss.

## 2026-05-15 19:30 — PR #3417: H11b log1p alpha sweep (thorfinn) — **assigned (verify combined baseline + find optimal alpha)**

- Branch: `charliepai2i24h4-thorfinn/thorfinn/log1p-alpha-sweep`
- Hypothesis: Parameterize signed-log1p as `sign(y) * log1p(α|y|) / α`. Sweep α ∈ {0.5, 1.0, 2.0}. α=1 arm verifies true combined val_avg under H11+H12 stack.
- Three arms. α=0.5 (less compression), α=1.0 (verify, current default), α=2.0 (more compression).
- Target to beat: val_avg/mae_surf_p < 92.80 (or establish true combined number from α=1 arm).
- Critical: arm 2 (α=1) IS the verification of the current 92.80 baseline under combined code.

## 2026-05-15 19:30 — PR #3222: H9 Cautious AdamW v2 (nezuko) — **CLOSED, did not compose with dropout**

- Branch: `charliepai2i24h4-nezuko/cautious-adamw`
- v2 hypothesis: Cautious AdamW + H12 dropout on RFF+Re-strat baseline. Test orthogonal composition.

| Metric | v2 Value | vs H12 (112.49) |
|---|---|---|
| `val_avg/mae_surf_p` (best, epoch 12) | 113.60 | +1.0% (worse) |
| `val_single_in_dist/mae_surf_p` | 126.23 | -7.7% |
| `val_geom_camber_rc/mae_surf_p` | 133.57 | +13.0% (worse) |
| `val_geom_camber_cruise/mae_surf_p` | 85.51 | -2.1% |
| `val_re_rand/mae_surf_p` | 109.11 | +1.4% |
| `test_avg/mae_surf_p` | 100.68 | -4.0% (better) |
| mean_mask | 0.61 ± 0.01 | (mechanism active, not collapsing) |

- Metric artifact: `models/model-charliepai2i24h4-nezuko-cautious-adamw-v2-20260515-183110/metrics.jsonl`
- Diagnostic: Cautious mask mechanism is healthy (mean_mask stable at 0.61, ~39% of update positions zeroed each step). But the val/test divergence is uncomfortable (val +1.0%, test -4.0%). Val_geom_camber_rc badly regressed despite test_geom_camber_rc only +4.9%. The student's analysis is correct: Cautious AdamW + FFN dropout don't compose strictly additively because they fight the same overfitting mechanism. The new combined baseline (92.80) is 22% better than this result and unrecoverable through optimizer tweaks alone.
- Decision: **Close** — mechanism doesn't compose with current best stack. Nezuko reassigned to H14 (T_max alignment).

## 2026-05-15 19:30 — PR #3201: H3 channel-loss v2 milder p=1.5 (edward) — **CLOSED, direction exhausted**

- Branch: `charliepai2i24h4-edward/channel-weighted-surf-loss-p3x` (was reused for v2)
- v2 hypothesis: Milder channel weighting [1, 1, 1.5] to test if reduced over-emphasis preserves the velocity-pressure coupling.

| Metric | v2 (p=1.5) | vs H12 (112.49) | vs v1 (p=3.0) |
|---|---|---|---|
| `val_avg/mae_surf_p` (best, epoch 11) | 135.50 | +20.5% (worse) | -2.1% |
| `val_single_in_dist/mae_surf_p` | 177.37 | +29.6% (worse) | -0.8% |
| `val_geom_camber_rc/mae_surf_p` | 143.98 | +21.8% (worse) | -4.7% |
| `val_geom_camber_cruise/mae_surf_p` | 98.36 | +12.6% (worse) | -1.0% |
| `val_re_rand/mae_surf_p` | 122.27 | +13.7% (worse) | -1.8% |
| `test_avg/mae_surf_p` | 127.37 | +21.5% (worse) | — |

- Metric artifact: `models/model-charliepai2i24h4-edward-channel-loss-p1p5-20260515-183105/metrics.jsonl`
- Diagnostic: Halving the pressure emphasis (18× → 12.9× effective) bought ~2% on val_avg but in-dist barely moved (-0.8% v1→v2 single_in_dist). The student's own analysis is correct: *any* explicit pressure overweighting disrupts the velocity-pressure coupling the model needs for in-distribution prediction. Smoothly interpolating magnitude doesn't interpolate harm. Direction is closed.
- Decision: **Close** — direction exhausted, both p=3.0 and p=1.5 regress severely on the in-dist split. Edward reassigned to H15 (SwiGLU MLP). Good empirical work on the variance analysis and NaN root cause find — those were genuinely useful.

## 2026-05-15 19:30 — PR #3184: H1 LinearNO ablation (alphonse) — **stale_wip, nudged**

- Branch: `charliepai2i24h4-alphonse/linearno-no-interslice-qkv`
- Hypothesis: Remove inter-slice QKV attention from `PhysicsAttention` (set `out_slice = slice_token`). LinearNO paper (Hao et al. 2025) showed this works across NS2d/Elasticity/Plasticity/Weather.
- Status: Pod GPU was at 100% from 18:38–19:02 UTC (~24 min, consistent with hitting the 30-min wall-clock cap), but no metrics committed and no PR comment with results. Branch HEAD is still at assignment commit. Looks like training completed but the student-Claude didn't finalize.
- Advisor action: Posted directive comment instructing student to check models/ artifacts, commit, post SENPAI-RESULT marker. If no artifacts, rerun. Reminded that baseline is now 92.80.

## 2026-05-15 19:00 — PR #3345: H11 signed-log1p target transform (thorfinn) — **MERGED, new baseline**

- Branch: `charliepai2i24h4-thorfinn/thorfinn/log1p-targets`
- Hypothesis: Apply `signed_log1p(y) = sign(y)*log1p(|y|)` to both pred and y before loss computation. Compress the 13× per-sample y-std dynamic range (164→2077) to ~3×. Eval path untouched.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 14) | **92.80** |
| `val_single_in_dist/mae_surf_p` | 115.48 |
| `val_geom_camber_rc/mae_surf_p` | 105.48 |
| `val_geom_camber_cruise/mae_surf_p` | 63.87 |
| `val_re_rand/mae_surf_p` | 86.36 |
| `test_avg/mae_surf_p` | **84.11** |
| `test_single_in_dist/mae_surf_p` | 108.91 |
| `test_geom_camber_rc/mae_surf_p` | 91.72 |
| `test_geom_camber_cruise/mae_surf_p` | 56.73 |
| `test_re_rand/mae_surf_p` | 79.06 |
| Wall clock | 30 min cap; epoch 14 of 50 |

- Metric artifact: `models/model-thorfinn-log1p-targets-20260515-173623/metrics.jsonl`
- Diagnostic: Massive win across all splits. geom_camber_cruise -37.1%, re_rand -27.4%, single_in_dist -20.2%. The student's slog1p diagnostic confirmed 3.5× std compression (y_norm std 1.51 → slog1p std 0.60). OOD splits improved more than in-dist, suggesting that the high-Re gradient domination was most severely distorting OOD learning. Effect exceeded prediction (-24% vs -3 to -7%).
- Caveat: run was on pre-dropout baseline (122.81). After squash merge the code has dropout + log1p; combined val_avg needs verification.
- Decision: **Merge** — largest single-PR improvement in this round by far. Fundamental improvement to optimization landscape.

## 2026-05-15 19:00 — PR #3318: H6 grad clip + SGDR warm restarts (frieren) — **sent back, v2 on combined baseline**

- Branch: `charliepai2i24h4-frieren/gradclip-sgdr`
- Hypothesis: Add `clip_grad_norm_(max_norm=1.0)` + `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)` per-batch with fractional epoch stepping. Targets noisy training curve.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 10) | 99.85 |
| `val_single_in_dist/mae_surf_p` | 113.87 |
| `val_geom_camber_rc/mae_surf_p` | 107.71 |
| `val_geom_camber_cruise/mae_surf_p` | 77.55 |
| `val_re_rand/mae_surf_p` | 100.28 |
| `test_avg/mae_surf_p` | 89.75 |
| Wall clock | 30 min cap; epoch 14 of 50; best at epoch 10 |

- Diagnostic: SGDR fired perfectly (LR sawtooth at epoch 11, eta_min at epoch 10). Grad clip eliminated oscillation — monotone descent vs prior sawtoothed curve. Both mechanisms confirmed. val_avg -18.7% vs RFF baseline. Run was on pre-dropout, pre-log1p baseline (122.81). Against new combined baseline (92.80), the 99.85 is a regression.
- Decision: **Sent back** — mechanism validated, needs rerun on combined baseline (dropout + log1p + SGDR + gradclip).

## 2026-05-15 19:00 — PR #3197: H8 EMA v2 (askeladd) — **sent back, v3 on combined baseline**

- Branch: `charliepai2i24h4-askeladd/ema-weights-decay-0p999`
- Hypothesis: Shadow EMA (decay=0.999) of model weights, evaluate on EMA model. v2 rebased on RFF+Re-strat baseline.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best EMA, epoch 13) | 114.89 |
| `val_single_in_dist/mae_surf_p` | 140.88 |
| `val_geom_camber_rc/mae_surf_p` | 118.24 |
| `val_geom_camber_cruise/mae_surf_p` | 90.13 |
| `val_re_rand/mae_surf_p` | 110.32 |
| `test_avg/mae_surf_p` | 105.64 |
| EMA gain over live | ~7.4% (best live 116.82 → best EMA 114.89) |

- Diagnostic: EMA gain confirmed and widening in second half (EMA wins 6/7 epochs 7-13). Clean implementation. Run on RFF+Re-strat baseline (122.81). Against new combined baseline (92.80), 114.89 is a regression. EMA mechanism is orthogonal — should compose with dropout and log1p.
- Decision: **Sent back** — v3 rerun on combined baseline (dropout + log1p + EMA).

## 2026-05-15 18:20 — PR #3326: H12 MLP dropout=0.1 (fern) — **MERGED, new baseline**

- Branch: `charliepai2i24h4-fern/mlp-dropout`
- Hypothesis: Add `dropout=0.1` to `MLP` class and apply in `TransolverBlock.mlp` FFN sub-layers. `PhysicsAttention`, preprocess MLP, and final head remain at `dropout=0.0`. With only 1499 training samples, the FFN path was memorizing training-distribution feature correlations.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 13) | **112.49** |
| `val_single_in_dist/mae_surf_p` | 136.83 |
| `val_geom_camber_rc/mae_surf_p` | 118.25 |
| `val_geom_camber_cruise/mae_surf_p` | 87.31 |
| `val_re_rand/mae_surf_p` | 107.55 |
| `test_avg/mae_surf_p` | **104.83** |
| `test_single_in_dist/mae_surf_p` | 126.77 |
| `test_geom_camber_rc/mae_surf_p` | 112.01 |
| `test_geom_camber_cruise/mae_surf_p` | 75.35 |
| `test_re_rand/mae_surf_p` | 105.20 |
| Wall clock | 30 min cap; epoch 14 of 50; best at epoch 13 |

- Metric artifact: `models/model-fern-mlp-dropout-0p1-20260515-163433/metrics.jsonl`
- Diagnostic: Clean OOD-dominant signature: geom_camber_cruise -14.1%, re_rand -9.6%, geom_camber_rc -6.1%, single_in_dist -5.4% on val. Test single_in_dist slightly regressed (+2.3%) — classic regularizer tradeoff near the sweet spot. The regularizer reduces memorization of training-set feature correlations, which pays dividends most where the test distribution differs from train.
- Follow-up: fern assigned H12b dropout rate sweep {0.05, 0.15, 0.20} to find optimal beyond 0.1.
- Decision: **Merge** — -8.4% val_avg, strongest single-PR improvement so far. Clean implementation, OOD signature exactly as hypothesized.

## 2026-05-15 18:20 — PR #3222: H9 Cautious AdamW (nezuko) — **sent back, v2 rerun needed**

- Branch: `charliepai2i24h4-nezuko/cautious-adamw`
- Hypothesis: Replace AdamW with from-scratch CautiousAdamW that masks update components where sign(m_t) ≠ sign(g_t).

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 13) | **118.91** |
| `val_single_in_dist/mae_surf_p` | 143.16 |
| `val_geom_camber_rc/mae_surf_p` | 128.82 |
| `val_geom_camber_cruise/mae_surf_p` | 94.85 |
| `val_re_rand/mae_surf_p` | 108.80 |
| `test_avg/mae_surf_p` | 109.23 |
| Wall clock | 30 min cap; epoch 14 of 50; best at epoch 13 |

- Metric artifact: `models/model-charliepai2i24h4-nezuko-cautious-adamw-20260515-163837/metrics.jsonl`
- Diagnostic: v1 beat old baseline 122.81 → 118.91 (-3.2%). But fern's dropout merged during review, raising the bar to 112.49. v1 is above the new target. CONFLICTING (needs rebase, had duplicate NaN workaround). Sent back for v2 rerun on top of new baseline (including H12 dropout) to test composition.
- Decision: **Sent back** — mechanism is proven; v2 tests composition with dropout.

## 2026-05-15 18:20 — PR #3201: H3 channel-weighted surface loss p=3 (edward) — **sent back, p=1.5 rerun**

- Branch: `charliepai2i24h4-edward/channel-weighted-surf-loss-p3x`
- Hypothesis: Weight pressure channel 3× in surface loss (ch_weights=[1,1,3] / mean), keeping surf_weight=10.0.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best run, epoch 14) | 138.39 |
| `val_single_in_dist/mae_surf_p` | 178.75 (178.75 vs 144.70 baseline = +23.5%) |
| `val_geom_camber_cruise/mae_surf_p` | 99.36 (-2.2% slight improvement) |
| `test_avg/mae_surf_p` (3-split partial) | 138.95 |
| Seeds tested | 4 runs (137.64 to 149.19 range, ~8% spread) |

- Diagnostic: Over-emphasis on pressure at the cost of velocity coupling. Single_in_dist badly regressed. Only cruise improved marginally. v2 (p=1.5, milder) sent back to test: if still regresses, channel-reweighting direction closed.
- Decision: **Sent back** — milder ratio one more try; if p=1.5 also regresses, hypothesis closed.

## 2026-05-15 17:00 — PR #3291: H7 two-branch output head (thorfinn) — **CLOSED, regressed**

- Branch: `charliepai2i24h4-thorfinn/two-branch-head`
- Hypothesis: Replace the shared output MLP with two separate decoders — a wider `surf_head` (n_layers=2) and a narrower `vol_head` (n_layers=1) — to let surface and volume predictions specialize their feature pathways.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 15) | **135.74** |
| `val_single_in_dist/mae_surf_p` | 166.35 |
| `val_geom_camber_rc/mae_surf_p` | 139.93 |
| `val_geom_camber_cruise/mae_surf_p` | 111.16 |
| `val_re_rand/mae_surf_p` | 125.53 |
| `test_avg/mae_surf_p` | NaN (run before frieren NaN fix) |
| Param count | 670,810 (~0.67M) |
| Wall clock | 30 min cap; epoch 15 of 50 |

- Metric artifact: `models/model-charliepai2i24h4-thorfinn-two-branch-head-*/metrics.jsonl`
- Diagnostic: 3 of 4 splits regressed vs current baseline 122.81 (+10.5% worse val_avg). Only geom_camber_rc nominally improved (139.93 vs 125.95 — but still well above baseline). Student correctly self-assessed: "hypothesis did NOT pan out."
- Root cause analysis: The shared-decoder's cross-channel feature sharing likely provides implicit regularization that benefits generalization. Separating surface/volume decoders loses this shared representation and adds little value with only ~0.67M params. The in-dist split (166.35) regressed the most, suggesting the two-branch design does not help the bottleneck split.
- Decision: **Closed** — unambiguous regression on all splits including the target split (single_in_dist). Thorfinn reassigned to H11 (log1p target normalization, PR #3345).

## 2026-05-15 16:25 — PR #3217: H5 RFF coord encoding + NaN fix (frieren) — **MERGED, new baseline**

- Branch: `charliepai2i24h4-frieren/rff-coord-nfreq32-sigma1`
- Hypothesis: Replace raw (x,z) positional dims 0-1 with a fixed 64-dim Random Fourier Feature expansion (n_freq=32, sigma=1.0) to lift spectral bandwidth for boundary-layer gradients. Bonus: added a `y_finite_sample` mask + `nan_to_num` in `evaluate_split` to resolve the branch-wide `test_avg` NaN.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 12) | **122.81** |
| `val_single_in_dist/mae_surf_p` | 144.70 |
| `val_geom_camber_rc/mae_surf_p` | 125.95 |
| `val_geom_camber_cruise/mae_surf_p` | 101.61 |
| `val_re_rand/mae_surf_p` | 119.00 |
| `test_avg/mae_surf_p` (now finite!) | **111.16** |
| `test_single_in_dist/mae_surf_p` | 123.91 |
| `test_geom_camber_rc/mae_surf_p` | 114.82 |
| `test_geom_camber_cruise/mae_surf_p` | 88.14 |
| `test_re_rand/mae_surf_p` | 117.78 |
| Wall clock | 30 min cap; epoch 12 of 50 |

- Metric artifact: `models/model-frieren-rff-nfreq32-sigma1-20260515-140556/metrics.jsonl`
- Diagnostic: Training curve showed pronounced noise until cosine annealing began to bite at epoch 11, with a sharp ~30-point drop then plateau (156→125→122→122). The RFF expansion gave the preprocess MLP a higher-frequency vocabulary for boundary-layer features; the gain shows up most in `val_geom_camber_rc` (-22.7 vs Re-strat baseline) and `val_single_in_dist` (-15.4).
- NaN fix: `evaluate_split` now masks and zero-fills samples where `isfinite(y).all(dim=-1)` is False before computing `err = (pred - y).abs()`. This resolves the IEEE 754 `NaN * 0 = NaN` propagation through `surf_mask` for test_geom_camber_cruise sample 20.
- Decision: **Merge** — -4.3% val_avg improvement, clean RFF implementation with buffer-registered B matrix (non-trainable), plus a valuable branch-wide bug fix now baked in.

## 2026-05-15 14:35 — PR #3226: H10 Re-stratified sampler (thorfinn) — **MERGED**

- Branch: `charliepai2i24h4-thorfinn/re-strat-high2x`
- Hypothesis: Upweight Re>1e6 samples by 2x in the `WeightedRandomSampler`, on top of the existing domain-balanced weights. Targets the higher-error high-Re regime which dominates `val_avg/mae_surf_p` via the per-sample y std (164→2077 across the dataset).

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 14) | **127.84** |
| `val_single_in_dist/mae_surf_p` | 160.10 |
| `val_geom_camber_rc/mae_surf_p` | 148.67 |
| `val_geom_camber_cruise/mae_surf_p` | 91.50 |
| `val_re_rand/mae_surf_p` | 111.08 |
| `test_avg/mae_surf_p` | NaN (data quirk; 3 finite splits) |
| Re>1e6 samples upweighted | 1303 / 1499 (≈87%) |
| Wall clock | 30 min cap; epoch 14 of 50 |

- Metric artifact: `models/<experiment>/metrics.jsonl` on the student branch.
- Diagnostic: `val_re_rand` (111.08) and `val_geom_camber_cruise` (91.50) — the OOD-Re-stratified splits — were the two lowest, consistent with the high-Re upweight paying off where the test boundary stresses Re generalization.
- Implementation note: student verified that stored `x[..., 13]` is **already** `log(Re)` (not normalized), so used `log_re = float(x_i[0, 13].item())` with `threshold = log(1e6)` instead of denormalizing via stats. Mathematically equivalent to the PR-body recipe.
- Decision: **Merge** — clear winner, simple sampler change, +Re-strat now part of the baseline.

## 2026-05-15 14:30 — PR #3197: H8 EMA model weights (askeladd) — sent back

- Branch: `charliepai2i24h4-askeladd/ema-weights-decay-0p999`
- Hypothesis: Maintain shadow EMA (decay=0.999) of model weights and evaluate val/test against the EMA model, not the live model. Reduces step-to-step variance and tends to improve OOD generalization.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best EMA, epoch 13) | 132.17 |
| `test_avg/mae_surf_p` | NaN (data quirk) |
| Live val_avg (best) | similar/slightly worse than EMA |
| Peak memory | 42.11 GB |

- Diagnostic: EMA worked as designed — clean dual live/EMA tracking, EMA consistently beat live across epochs. But the absolute number is now above the new baseline (127.84 from H10).
- Decision: **Send back** — Re-run EMA on top of the merged Re-strat baseline. Mechanism is orthogonal and should stack.

## 2026-05-15 14:30 — PR #3224: H13 Persistent geometry conditioning (tanjiro) — sent back

- Branch: `charliepai2i24h4-tanjiro/geom-cond-additive`
- Hypothesis: Inject the per-sample global geometry context (dims 13-23: Re, AoA, NACA params, gap, stagger) at every Transolver block via a gated additive projection, GALE-style.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 10) | 134.31 |
| `val_geom_camber_cruise/mae_surf_p` | 103.45 (lowest of 4 splits) |
| `val_geom_camber_rc/mae_surf_p` | 141.89 |
| `test_avg/mae_surf_p` | NaN (data quirk) |
| Param count | 698K (+36K vs baseline) |
| Final geom_gates | [0.05, 0.13, 0.16, 0.17, 0.17] |
| Wall clock | 30 min cap; epoch 14 of 50 |

- Diagnostic: Gates started at 0 (identity init) and learned to non-zero values monotonically — the model used the conditioning. But training was cap-bound, cosine LR barely decayed (ratio 0.83 at e14), val_avg still oscillating.
- Decision: **Send back** — Re-run on merged baseline, fix cap issue (reduce slice_num=32 or T_max=15) so cosine actually anneals within the time budget.

## 2026-05-15 14:25 — PR #3210: H2 Scale Transolver to ~4M params (fern) — sent back

- Branch: `charliepai2i24h4-fern/scale-256x6x8-lr3e4`
- Hypothesis: Scale n_hidden=128→256, n_layers=5→6, n_head=4→8 (≈4M params).

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 6) | 158.40 |
| Param count | 3.01M |
| Peak memory | 94 GB at bs=4 → fell back to bs=2 |
| Wall clock | 30 min cap; epoch 6 of 7 |

- Diagnostic: Training very noisy (val_avg 158 → 239 → 158 over 7 epochs). Cap-bound, bs=2 after OOM fallback. Capacity vs. epochs tradeoff is paying too much for epochs.
- Decision: **Send back** — Add `clip_grad_norm_(1.0)`, drop lr to 2e-4, try mid-size variant (n_hidden=192, n_layers=6, n_head=6) to fit ~12-15 epochs.

## Known branch-wide quirk

`data/scoring.py` (read-only per program.md) accumulates MAE via `(pred - y).abs() * surf_mask`. Sample 20 in `test_geom_camber_cruise` has 761 `inf` values in `y[..., 2]` (p channel). `NaN * 0 = NaN` (IEEE 754) propagates through the multiplication, contaminating `test_avg/mae_surf_p` for every experiment. Both askeladd and tanjiro independently identified this. **Workaround:** `val_avg/mae_surf_p` is the canonical ranking metric. `test_avg/mae_surf_p` should be reported as NaN-aware (the 3 finite test splits averaged, or skip).
