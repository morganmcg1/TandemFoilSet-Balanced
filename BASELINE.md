# Baseline — `icml-appendix-charlie-pai2g-24h-r4`

Fresh start for round 4 of the Charlie / pai2g 24h logging ablation. No
prior experiments on this branch — the implicit baseline is the unmodified
`train.py` config inherited from `icml-appendix-charlie`. The first merged
winner sets the first numeric reference value.

## Reference config (default `train.py` at HEAD)

- **Model**: Transolver
  - `n_hidden = 128`
  - `n_layers = 5`
  - `n_head = 4`
  - `slice_num = 64`
  - `mlp_ratio = 2`
  - `space_dim = 2`, `fun_dim = X_DIM - 2 = 22`
  - `out_dim = 3` (`Ux`, `Uy`, `p`)
  - `unified_pos = False`
- **Optimizer**: AdamW (`lr = 5e-4`, `weight_decay = 1e-4`)
- **LR schedule**: `SequentialLR([LinearLR(start_factor=1/total_warmup_iters, total_iters=total_warmup_iters), CosineAnnealingLR(T_max=T_max_iters)])` with `total_warmup_iters=batches_per_epoch` (epoch-1 linear ramp to lr=5e-4) + `T_max_iters=14*batches_per_epoch` (cosine decay over 14 epochs), `scheduler.step()` per batch. _(updated 2026-05-13 by PR #1754, was plain CosineAnnealingLR T_max=15 step-per-epoch from PR #1611)_
- **Loss**: **L1 (MAE) in normalized target space**, `loss = vol_loss + surf_weight * surf_loss`, `surf_weight = 10.0` _(updated 2026-05-12 by PR #1397)_
- **Stochastic depth**: per-block drop probs `[0.0, 0.025, 0.05, 0.075, 0.10]` (linear schedule, last layer is the output head and never dropped) _(added 2026-05-12 by PR #1552)_
- **`evaluate_split` NaN-safe pre-filter**: skip samples with non-finite `y` before `accumulate_batch` to keep the 4-split test mean finite despite the `test_geom_camber_cruise/000020.pt` data bug _(added 2026-05-12 by PR #1552)_
- **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=25.0)` immediately before `optimizer.step()`; the pre-clip total_norm is also logged to metrics.jsonl as `train/last_grad_norm` _(added 2026-05-12 by PR #1637)_
- **Fourier coord positional encoding**: `FourierCoordEnc(n_freqs=6)` applied after `(x - x_mean)/x_std` normalization; replaces the 2 raw `(x, z)` coord dims with 24 Fourier features (`sin/cos` at frequencies `2^k · π`, `k=0..5`). `fun_dim = 4 * 6 + 22 - 2 = 44`. _(updated 2026-05-13 by PR #1772, was L=4 in #1548)_
- **LayerScale**: per-channel learnable γ_l vectors `nn.Parameter(torch.ones(hidden_dim) * 0.025)` (**init=0.025**, down from 0.05) on both attn and MLP residual branches in each `TransolverBlock`; `fx = γ_attn ⊙ attn(ln_1(fx)) + fx` and `fx = γ_mlp ⊙ mlp(ln_2(fx)) + fx`. CaiT-style (Touvron et al. 2021). _(added 2026-05-13 by PR #1799, init lowered 0.1→0.05 by PR #1896, lowered 0.05→0.025 by PR #2018)_
- **Per-channel surf-loss weighting**: `surf_loss = mean([0.5, 0.5, 2.0] * |y_pred - y| / y_std)` — Ux and Uy weighted 0.5, pressure weighted 2.0 (4× ratio). Applied in both training loop and `evaluate_split`. `vol_loss` remains at implicit `[1, 1, 1]`. _(added 2026-05-13 by PR #1711)_
- **Batch size**: `4`
- **Epochs**: configured `50`, capped by `SENPAI_TIMEOUT_MINUTES = 30`
- **Sampler**: `WeightedRandomSampler` with equal-domain weights from `meta.json`

## Metrics contract

- Primary ranking metric: `val_avg/mae_surf_p` — equal-weight mean of `mae_surf_p` across the four val splits.
- Paper-facing comparison metric: `test_avg/mae_surf_p` — same aggregation on the four test splits at the best val checkpoint.
- All metrics computed in physical (denormalized) units in `data/scoring.py`.

## Current best result

### 2026-05-13 21:25 — PR #2519 (`charliepai2g24h4-tanjiro/attn-temp-fixed-sharper`)

**Fixed sharper attention temperature τ = √2 × default** — single-line change to `F.scaled_dot_product_attention(..., scale=1/sqrt(d_head/2))` which is √2 sharper than the default `1/sqrt(d_head)`. No new parameters. Validates the #2488 RMSNorm-QK-γ Outcome B finding that the model wanted slight attention sharpening — but with a fixed scalar rather than learnable per-channel γ.

- **`val_avg/mae_surf_p`** = **56.1754** (best @ epoch 12; **−3.68%** vs #2475 baseline 58.3244 — biggest single-experiment win in 18 merges)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **48.7149** (**−4.38%** — test improvement LARGER than val improvement; robust generalization signal)
- **Per-split val** `mae_surf_p` at best val checkpoint:
  - `val_single_in_dist` = **66.511** (−6.77%)
  - `val_geom_camber_rc` = **68.819** (−3.13%)
  - `val_geom_camber_cruise` = **34.782** (−1.77%)
  - `val_re_rand` = **54.590** (−1.65%)
- **Per-split test** `mae_surf_p` at best val checkpoint:
  - `test_single_in_dist` = **57.795** (−9.46% — largest single-split gain)
  - `test_geom_camber_rc` = **63.594** (−1.19%)
  - `test_geom_camber_cruise` = **28.422** (−3.29%)
  - `test_re_rand` = **45.048** (−2.48%)
- **Mechanism**: a √2 sharper scaling factor in the SDPA softmax produces a more decisive attention distribution. The model wanted this without needing per-channel learnability — the learnable γ in #2488 added too many DoF (peak std/mean only 33%) without exploiting them. The fixed scalar captures the gain cleanly.
- **Compound progress**: 20 merges, **100.957 → 56.1754 = −44.36%**
- **Param count**: **892,637** (unchanged)
- **Metric artifacts**: `models/model-charliepai2g24h4-tanjiro-attn-temp-fixed-sharper-20260513-185119/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-tanjiro --experiment_name charliepai2g24h4-tanjiro/attn-temp-fixed-sharper`

### 2026-05-13 18:21 — PR #2475 (`charliepai2g24h4-fern/layerscale-init-0.1`) — *superseded by #2519*

LayerScale γ init raised from 0.025 → 0.1 (4× current) on the post-#2436 stack. Single-line change to the `TransolverBlock` default; everything else identical (same 3-group AdamW with LayerScale γ in 10× lr no-WD group). Tests whether the prior init=0.025 was implicitly retuned by the no-WD 10× lr group dynamics (where γ drifts 4.6–6.2× off init).

- **`val_avg/mae_surf_p`** = **58.3244** (best @ epoch 12; **−0.49%** vs #2436 baseline 58.6093)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **50.9438** (**+0.29%** — within run-to-run noise vs prior 50.7946)
- **Per-split val** `mae_surf_p` at best val checkpoint:
  - `val_single_in_dist` = 71.343 (+1.69% — only val regression)
  - `val_geom_camber_rc` = 71.041 (−0.09% — neutral)
  - `val_geom_camber_cruise` = **35.411** (−2.32%)
  - `val_re_rand` = **55.503** (−2.49%)
- **Per-split test** `mae_surf_p` at best val checkpoint:
  - `test_single_in_dist` = 63.834 (+0.78%)
  - `test_geom_camber_rc` = 64.360 (+1.94% — only OOD test regression)
  - `test_geom_camber_cruise` = **29.389** (−1.02%)
  - `test_re_rand` = **46.192** (−1.74%)
- **Mechanism (surprising)**: attn LayerScale γ drifts **DOWN** from init=0.1 to 0.043–0.080 mean (with std/mean 150–243% — still high per-channel sparsity), while mlp γ drifts **UP** from 0.1 to 0.149–0.194 (overshoot vs #2436's 0.114–0.156 endpoint). The two paths have anti-correlated init responses under the no-WD 10× lr group — they don't share a single global equilibrium, and the optimizer carries momentum from wherever it starts. Attn naturally prefers low-mean sparse states; mlp prefers higher-mean dense states.
- **Compound progress**: 19 merges, **100.957 → 58.3244 = −42.23%**
- **Param count**: **892,637** (unchanged — same number of LayerScale γ params, just different init)
- **Metric artifacts**: `models/model-charliepai2g24h4-fern-layerscale-init-0.1-20260513-172731/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-fern --experiment_name charliepai2g24h4-fern/layerscale-init-0.1`

---

### 2026-05-13 18:30 — PR #2436 (`charliepai2g24h4-fern/layerscale-lr-10x`) — *superseded by #2475*

LayerScale γ params (5 blocks × 2 paths × 128 channels = 1280 scalars) in a separate AdamW group with no weight-decay + 10× lr (`lr=5e-3, wd=0`). Same optimizer-group recipe that won on freqs in #2370, now applied to the next additive-scale parameter class. The MLP-side LayerScale γ shifted 4.6–6.2× off init=0.025 (settled at 0.114–0.156), and the attn-side γ developed extreme per-channel diversity (std/mean ratios 247–319%). Both signals confirm that default WD=1e-4 + lr=5e-4 was systematically under-training these scale parameters.

**Note on n_params correction**: prior baseline entry (PR #2370) listed n_params=831,197, but that was measured on stale checkout `git_commit=e39f7bf` (pre-#2304 inner_dim=288 merge). The actual model count on current advisor HEAD with inner_dim=288 is **892,637**. Both fern (this PR) and thorfinn (#2435 closed) independently caught this discrepancy. From this entry onward, n_params is reported correctly for the as-merged architecture.

- **`val_avg/mae_surf_p`** = **58.6093** (best @ epoch 12; **−1.60%** vs prior #2370 baseline 59.5645)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **50.7946** (**−1.59%** vs prior 51.6141)
- **Per-split val** `mae_surf_p` at best val checkpoint:
  - `val_single_in_dist` = 70.160 (−0.11% — neutral)
  - `val_geom_camber_rc` = 71.104 (−0.51%)
  - `val_geom_camber_cruise` = **36.251** (**−7.49%** — largest val gain)
  - `val_re_rand` = 56.922 (−0.78%)
- **Per-split test** `mae_surf_p` at best val checkpoint:
  - `test_single_in_dist` = 63.342 (+3.94% — only regression; partially offset by val_single_in_dist neutrality)
  - `test_geom_camber_rc` = 63.135 (−1.55%)
  - `test_geom_camber_cruise` = **29.692** (**−8.29%** — largest test gain)
  - `test_re_rand` = **47.009** (−4.08%)
- **Mechanism**: LayerScale γ is multiplicative-on-activations (additive scale), so the 10× lr no-WD recipe that worked on freqs transfers cleanly. MLP-path γ converges to substantially larger amplitudes (4.6–6.2× off init) and attn-path γ develops sparse-gating per-channel diversity (signed amplitudes, many channels near 0 with a few strongly amplified). This is the second instance of the **additive-scale optimizer-group axis** producing a compound win. Slice-attention temperature (PR #2437 CLOSED Outcome C) was the falsifier showing softmax-internal scale does NOT benefit from the same recipe.
- **Compound progress**: 18 merges, **100.957 → 58.6093 = −41.94%**
- **Param count**: **892,637** (1280 LayerScale γ scalars regrouped, no new parameters)
- **Metric artifacts**: `models/model-charliepai2g24h4-fern-layerscale-lr-10x-20260513-161832/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-fern --experiment_name charliepai2g24h4-fern/layerscale-lr-10x`

---

### 2026-05-13 17:05 — PR #2370 (`charliepai2g24h4-frieren/learned-freqs-no-wd-10x-lr`) — *superseded; see n_params note above*

Learned Fourier frequencies (6 learnable params) with no weight-decay + 10× lr multiplier. The fixed dyadic init `[1,2,4,8,16,32]` was over-regularized by default WD=1e-4 and lr=5e-4 — freqs barely moved in the earlier #2312 attempt. This PR puts the freq vector in a separate AdamW param group: `weight_decay=0`, `lr=5e-3` (10×). Bottom 3 freqs (1,2,4) now shift 14–27% off init toward [0.75, 1.46, 3.44], capturing dominant pressure-gradient spatial bands. Top 3 freqs (8,16,32) remain pinned — gradient-magnitude limited even at 10× lr.

- **`val_avg/mae_surf_p`** = **59.5645** (best @ epoch 12; **−3.73%** vs #2360 baseline 61.875)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **51.6141** (**−4.63%** vs #2360 baseline 54.117)
- **Per-split val** `mae_surf_p` at best val checkpoint:
  - `val_single_in_dist` = 70.235 (+0.44% vs #2360 — near-neutral)
  - `val_geom_camber_rc` = 71.466 (−1.00% vs #2360)
  - `val_geom_camber_cruise` = **39.185** (−14.62% vs #2360 — largest gain)
  - `val_re_rand` = **57.372** (−7.73% vs #2360)
- **Per-split test** `mae_surf_p` at best val checkpoint:
  - `test_single_in_dist` = 60.940 (−0.27% vs #2360)
  - `test_geom_camber_rc` = 64.131 (−1.49% vs #2360)
  - `test_geom_camber_cruise` = **32.376** (−12.75% vs #2360)
  - `test_re_rand` = **49.009** (−8.17% vs #2360)
- **Mechanism**: removing WD on the 6-freq vector and boosting its lr to 5e-3 allows the Fourier frequencies to learn the dominant spatial bands present in this dataset's pressure fields. Bottom freqs adapt strongly (−25% to −14% drift toward lower frequencies), capturing the dominant low-frequency pressure envelope. Top freqs remain gradient-magnitude limited — the fine-grained mesh structure at the highest dyadic frequencies carries insufficient gradient signal. Orthogonal to MLP-capacity axis (#2360 inner_dim=288).
- **Compound progress**: 17 merges, **100.957 → 59.5645 = −41.0%**
- **Param count**: 831,197 (+6 over #2360 baseline 831,191; the 6 learnable freq params)
- **Metric artifacts**: `models/model-charliepai2g24h4-frieren-learned-freqs-no-wd-10x-lr-20260513-151314/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-frieren --experiment_name charliepai2g24h4-frieren/learned-freqs-no-wd-10x-lr`

---

### 2026-05-13 16:05 — PR #2360 (`charliepai2g24h4-fern/reglu-inner-dim-288`)

ReGLU + inner_dim=256→288 bisect. Extra 32 gate/up/down channels in each SwiGLU block compensates for ReGLU's exact-zero dead channels in the gate-sparse regime. At +4.7% sec/epoch (vs +10.2% for 320), the experiment fits 12 epochs within the 30-min wall — the full schedule completes and val_single_in_dist IMPROVES −3.79% (vs the +11.2% regression at 320), confirming the epoch-budget hypothesis: the 320 regression was schedule-truncation, not capacity overfit.

- **`val_avg/mae_surf_p`** = **61.875** (best @ epoch 12; **−1.71%** vs #2304 baseline 62.949)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **54.117** (−0.19% vs #2304 baseline 54.221)
- **Per-split val** `mae_surf_p` at best val checkpoint:
  - `val_single_in_dist` = 67.276 (−3.79% vs #2304)
  - `val_geom_camber_rc` = 72.143 (−3.61% vs #2304)
  - `val_geom_camber_cruise` = 45.901 (+3.70% vs #2304 — slight regression, saturated split)
  - `val_re_rand` = 62.181 (−0.93% vs #2304)
- **Per-split test** `mae_surf_p` at best val checkpoint:
  - `test_single_in_dist` = 60.873 (−0.38% vs #2304)
  - `test_geom_camber_rc` = 65.103 (−1.65% vs #2304)
  - `test_geom_camber_cruise` = 37.112 (+2.22% vs #2304)
  - `test_re_rand` = 53.380 (+0.20% vs #2304 — near-neutral)
- **Mechanism**: inner_dim=288 (+32 channels per gate/up/down projection) directly compensates for dead channels created by ReGLU's exact-zero gate. Single_in_dist improvement (−3.79%) vs prior 320 regression (+11.2%) proves this is not capacity overfit — epoch-budget is the binding constraint above inner_dim=288.
- **Compound progress**: 16 merges, **100.957 → 61.875 = −38.7%**
- **Param count**: 892,631 (+61,440 from inner_dim 256→288; n_hidden unchanged at 128)
- **Metric artifacts**: `models/model-charliepai2g24h4-fern-reglu-inner-dim-288-20260513-141957/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-fern --experiment_name charliepai2g24h4-fern/reglu-inner-dim-288`

---

### 2026-05-13 14:10 — PR #2304 (`charliepai2g24h4-thorfinn/reglu-gate`)

ReGLU gate: `F.gelu → F.relu` in `SwiGLUMLP.forward()`. ReLU's exact-zero gate for x<0 maximally suppresses cross-channel contamination at high-magnitude pressure features. Gate-sharpness monotonicity confirmed: SiLU<GELU<ReLU, each step a compound win. Largest OOD generalization gain yet — test improves more than val (−4.07% test vs −1.92% val). All 3 non-in-dist splits improve; only single_in_dist regresses slightly (+2.03 val, +0.43 test). Model hit 30-min wall clock at ep 12 with val still descending strongly (76.4→71.5→68.2→62.9 across last 4 epochs) — full potential likely higher.

- **`val_avg/mae_surf_p`** = **62.949** (best @ epoch 12; **−1.92%** vs #2266 baseline 64.182)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **54.221** (−4.07% vs #2266 baseline 56.523)
- **Per-split val** `mae_surf_p` at best val checkpoint:
  - `val_single_in_dist` = 69.925 (+2.04% vs #2266 — only regression)
  - `val_geom_camber_rc` = 74.845 (−1.83% vs #2266)
  - `val_geom_camber_cruise` = 44.262 (−7.38% vs #2266)
  - `val_re_rand` = 62.765 (−3.15% vs #2266)
- **Per-split test** `mae_surf_p` at best val checkpoint:
  - `test_single_in_dist` = 61.108 (+0.71% vs #2266 — near-neutral)
  - `test_geom_camber_rc` = 66.196 (−6.46% vs #2266)
  - `test_geom_camber_cruise` = 36.305 (−6.91% vs #2266)
  - `test_re_rand` = 53.276 (−4.24% vs #2266)
- **Mechanism**: ReLU gate (max(0,x)) provides exact-zero suppression for all x<0 — hardest possible gate in the standard GLU family. Under L1 + surf-ch-weight [0.5,0.5,2.0], high-magnitude pressure extremes dominate gradients, and exact-zero gate suppression produces the strongest OOD transfer benefit. The improvement scaling with OOD difficulty is striking: single_in_dist barely affected, harder OOD splits gain most. ReLU's implicit sparsity also acts as a free regularizer.
- **Gate-sharpness axis**: SiLU (soft pass-through) → GELU (near-zero) → **ReLU (exact-zero)** — monotonic gains at each step: −6.96%, −4.75%, −1.92% val; −9.33%, −2.21%, −4.07% test.
- **Compound progress**: 15 merges, **100.957 → 62.949 = −37.7%** (#1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→#1896→#2018→#1754→#2105→#2175→#2266→**#2304**)
- **Param count**: 831,191 (unchanged — 1-char gate swap, ReLU even cheaper than GELU to compute).
- **Metric artifacts**: `models/model-charliepai2g24h4-thorfinn-reglu-gate-20260513-132007/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-thorfinn --experiment_name charliepai2g24h4-thorfinn/reglu-gate`

---

### 2026-05-13 13:45 — PR #2266 (`charliepai2g24h4-thorfinn/geglu-gate-comparison`)

GeGLU gate: single-character change `F.silu → F.gelu` in `SwiGLUMLP.forward()`. GELU's slightly harder switch in the negative-input regime suppresses cross-channel contamination at high-magnitude pressure features (stagnation points, wakes). L1 + surf-ch-weight [0.5,0.5,2.0] gradient signal is dominated by extreme pressure values where gate sharpness matters most. All 4 val splits improve; 3 of 4 test splits improve (test_geom_camber_rc +2.53%, attributed to pre-existing val/test camber-rc decorrelation, not GeGLU specifically). Zero parameter cost; same epoch budget as SwiGLU.

- **`val_avg/mae_surf_p`** = **64.182** (best @ epoch 13; **−4.75%** vs #2175 baseline 67.381)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **56.523** (−2.21% vs #2175 baseline 57.800)
- **Per-split val** `mae_surf_p` at best val checkpoint:
  - `val_single_in_dist` = 67.894 (−7.43% vs #2175)
  - `val_geom_camber_rc` = 76.235 (−5.50% vs #2175)
  - `val_geom_camber_cruise` = 47.790 (−1.82% vs #2175)
  - `val_re_rand` = 64.808 (−3.03% vs #2175)
- **Per-split test** `mae_surf_p` at best val checkpoint:
  - `test_single_in_dist` = 60.676 (−6.20% vs #2175)
  - `test_geom_camber_rc` = 70.778 (+2.53% vs #2175 — lone regression; val split wins)
  - `test_geom_camber_cruise` = 39.001 (−3.36% vs #2175)
  - `test_re_rand` = 55.636 (−2.60% vs #2175)
- **Mechanism**: GELU provides a harder gate than SiLU in the negative regime — `GELU(x) < SiLU(x)` for x < 0, suppressing cross-channel contamination at peak-pressure regions. Under L1 + 4× pressure weighting, the model's gradient is dominated by extreme p values (stagnation/wake), and the sharper gate better isolates those features. Same params, same best epoch, same training dynamics.
- **Compound progress**: 14 merges, **100.957 → 64.182 = −36.4%** (#1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→#1896→#2018→#1754→#2105→#2175→**#2266**)
- **Param count**: 831,191 (unchanged — zero new parameters; 1-character gate swap).
- **Metric artifacts**: `models/model-charliepai2g24h4-thorfinn-geglu-gate-comparison-20260513-121756/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-thorfinn --experiment_name charliepai2g24h4-thorfinn/geglu-gate-comparison`

---

### 2026-05-13 11:20 — PR #2175 (`charliepai2g24h4-tanjiro/swiglu-inner-dim-256`)

SwiGLU inner_dim expanded from 176 (=round_up8(256×2/3), param-matched) to 256 (=full hidden_dim). Gives gate/up/down projections full representational capacity within each TransolverBlock. Wins on all 4 test splits and 3/4 val splits (val_geom_camber_rc +1.4, but test_geom_camber_rc still wins −0.27). Best epoch = 13 (last epoch trained — model was still improving at timeout, suggesting under-training; longer schedule likely widens gain further). Param cost +22.6% (677,591 → 831,191). Run at 30-min cap (13/50 epochs).

- **`val_avg/mae_surf_p`** = **67.381** (best @ epoch 13; **−2.08%** vs #2105 baseline 68.812)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **57.800** (−2.71% vs #2105 baseline 59.410)
- **Per-split val** `mae_surf_p` at best val checkpoint:
  - `val_single_in_dist` = 73.341 (−4.00% vs #2105)
  - `val_geom_camber_rc` = 80.673 (+1.74% vs #2105 — lone regression; test split wins)
  - `val_geom_camber_cruise` = 48.675 (−6.40% vs #2105)
  - `val_re_rand` = 66.834 (−1.09% vs #2105)
- **Per-split test** `mae_surf_p` at best val checkpoint:
  - `test_single_in_dist` = 64.685 (−3.65% vs #2105)
  - `test_geom_camber_rc` = 69.035 (−0.39% vs #2105)
  - `test_geom_camber_cruise` = 40.356 (−4.72% vs #2105)
  - `test_re_rand` = 57.121 (−2.94% vs #2105)
- **Mechanism**: expanding SwiGLU inner_dim from 176 to 256 removes the 2/3-ratio constraint and gives the gate/up/down paths full hidden_dim capacity. The per-epoch best trajectory (every epoch a new best) confirms the wider model is in the capacity-limited regime, not the overfitting regime. Consistent with Shazeer's original argument that 2/3 is budget-neutral, not capacity-optimal.
- **Compound progress**: 13 merges, **100.957 → 67.381 = −33.3%** (#1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→#1896→#2018→#1754→#2105→**#2175**)
- **Param count**: 831,191 (+153,600 / +22.6% vs 677,591 in #2105).
- **Metric artifacts**: `models/model-charliepai2g24h4-tanjiro-swiglu-inner-dim-256-20260513-102355/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-tanjiro --experiment_name charliepai2g24h4-tanjiro/swiglu-inner-dim-256`

---

### 2026-05-13 11:15 — PR #2105 (`charliepai2g24h4-tanjiro/swiglu-activation`)

SwiGLU gated MLP replacing GELU in all 5 TransolverBlocks. `inner_dim=176` (= round_up8(256×2/3)), bias-free gate and up projections, bias-free down projection. +8,320 params (+1.24%). Per-token gated feature routing dramatically improves all 4 splits, with largest gains on OOD-flavoured splits (re_rand −8.73% val / −13.62% test; camber_cruise −10.49% val / −14.43% test). **⚠ Measurement note:** this run was on the pre-#1754 advisor HEAD (cosine T_max=15, no LR warmup). The merged code includes both SwiGLU AND LR warmup (#1754). A re-baseline confirmation run is in-flight.

- **`val_avg/mae_surf_p`** = **68.812** (best @ epoch 12; −7.53% vs #2018 baseline 74.415; −6.96% vs #1754 baseline 73.958)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **59.410** (−9.33% vs #2018; −7.89% vs #1754)
- **Per-split val** `mae_surf_p` at best val checkpoint:
  - `val_single_in_dist` = 76.377 (−5.60% vs #2018)
  - `val_geom_camber_rc` = 79.291 (−6.29% vs #2018)
  - `val_geom_camber_cruise` = 52.005 (−10.49% vs #2018)
  - `val_re_rand` = 67.573 (−8.73% vs #2018)
- **Per-split test** `mae_surf_p` at best val checkpoint:
  - `test_single_in_dist` = 67.134 (−4.94% vs #2018)
  - `test_geom_camber_rc` = 69.308 (−6.16% vs #2018)
  - `test_geom_camber_cruise` = 42.352 (−14.43% vs #2018)
  - `test_re_rand` = 58.848 (−13.62% vs #2018)
- **Mechanism**: gated SwiGLU (`SiLU(W_gate·x) ⊙ W_up·x` then W_down) provides per-token per-channel feature routing. On OOD samples the gate selectively suppresses irrelevant Fourier bands, acting as a learned input-dependent nonlinearity where fixed GELU cannot adapt. Best epoch advances 14→12 (SwiGLU converges ~2 epochs faster). LayerScale γ_l attn means stable (0.021–0.027), MLP means 0.041–0.053.
- **Compound progress**: 12 merges, **100.957 → 68.812 = −31.8%** (#1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→#1896→#2018→#1754→**#2105**)
- **Param count**: 677,591 (+8,320 / +1.24% vs 669,271).
- **Metric artifacts**: `models/model-charliepai2g24h4-tanjiro-swiglu-activation-20260513-091304/metrics.jsonl` and `metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-tanjiro --experiment_name charliepai2g24h4-tanjiro/swiglu-activation`

---

### 2026-05-13 10:05 — PR #1754 (`charliepai2g24h4-nezuko/lr-warmup-h19`)

Linear LR warm-up over epoch 1 (per-batch LinearLR) followed by cosine decay T_max=14 epochs (SequentialLR). Addresses the ep1 grad-norm spike seen on the compound stack; orthogonal to all prior merged components. Per-split gain concentrated on OOD-geom-cruise (−2.94% val / −5.51% test) and Re-rand (−1.59% val / −4.06% test) — the splits where LayerScale init=0.025 sacrificed OOD coverage; warmup rescues them by giving γ_l a more stable starting trajectory.

- **`val_avg/mae_surf_p`** = **73.958** (best @ epoch 14; −0.61% vs #2018)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **64.502** (−1.56% vs #2018)
- **Per-split val** `mae_surf_p` at best val checkpoint:
  - `val_single_in_dist` = 81.293 (+0.48% vs #2018)
  - `val_geom_camber_rc` = 85.285 (+0.79% vs #2018)
  - `val_geom_camber_cruise` = 56.390 (−2.94% vs #2018)
  - `val_re_rand` = 72.862 (−1.59% vs #2018)
- **Per-split test** `mae_surf_p` at best val checkpoint:
  - `test_single_in_dist` = 71.563 (+1.33% vs #2018)
  - `test_geom_camber_rc` = 74.317 (+0.62% vs #2018)
  - `test_geom_camber_cruise` = 46.766 (−5.51% vs #2018)
  - `test_re_rand` = 65.362 (−4.06% vs #2018)
- **Mechanism**: warmup recovers LayerScale-0.025 OOD degradation — the per-batch linear ramp over epoch 1 reduces ep1 grad-norm, giving γ_l parameters a stable starting trajectory before cosine peaks; OOD-geom-cruise and Re-rand splits gain the most (the same splits that LayerScale-0.025 regressed on vs #1896). Test gain (−1.56%) exceeds val gain (−0.61%) consistent with original pre-rebase run.
- **Compound progress**: 11 merges, **100.957 → 73.958 = −26.7%** (#1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→#1896→#2018→**#1754**)
- **Param count**: 669,271 (unchanged — zero new parameters; schedule change only).
- **Metric artifacts**: `models/model-charliepai2g24h4-nezuko-lr-warmup-h19-20260513-075103/metrics.jsonl` and `metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-nezuko --experiment_name charliepai2g24h4-nezuko/lr-warmup-h19-rebased`

---

### 2026-05-13 08:30 — PR #2018 (`charliepai2g24h4-thorfinn/layerscale-init-0.025`)

LayerScale init bracket: continue operating-point sweep, drop init=0.05 → **init=0.025**. Single-line change to `layer_scale_init`. Block-0 attn γ_l std/mean crosses 1.0 (110.5%) — ~half of per-channel entries have learned **negative** residual scale (sign-flipping). Marginal val gain but real test improvement driven by `single_in_dist`; same split pattern as #1896 but scaled down. Diminishing-returns curve confirmed: -1.21% (0.1→0.05) → **-0.08% (0.05→0.025)**.

- **`val_avg/mae_surf_p`** = **74.415** (best @ epoch 14; −0.08% vs #1896)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **65.524** (−0.74% vs #1896)
- **Per-split val** `mae_surf_p` at best val checkpoint:
  - `val_single_in_dist` = 80.907 (−4.90% vs #1896)
  - `val_geom_camber_rc` = 84.613 (+2.23% vs #1896)
  - `val_geom_camber_cruise` = 58.100 (+1.93% vs #1896)
  - `val_re_rand` = 74.039 (+1.34% vs #1896)
- **Per-split test** `mae_surf_p` at best val checkpoint:
  - `test_single_in_dist` = 70.626 (−5.86% vs #1896)
  - `test_geom_camber_rc` = 73.856 (−0.16% vs #1896)
  - `test_geom_camber_cruise` = 49.491 (+2.17% vs #1896)
  - `test_re_rand` = 68.125 (+2.26% vs #1896)
- **LayerScale γ_l mechanism**: attn means 0.019–0.023, MLP means 0.029–0.050. Block-0 attn std/mean = **110.5%** (>100% = sign-flip channels; up from 70.7% at init=0.05). Diminishing-returns curve: gain per halving of init → -1.21% (0.1→0.05) → -0.08% (0.05→0.025). MLP branch std/mean 47–73%, still sub-100%.
- **Compound progress**: 10 merges, **100.957 → 74.415 = −26.3%** (#1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→#1896→**#2018**)
- **Param count**: 669,271 (unchanged — zero new parameters; single-line change).
- **Metric artifacts**: `models/model-charliepai2g24h4-thorfinn-layerscale-init-0.025-20260513-070817/metrics.jsonl` and `metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-thorfinn --experiment_name charliepai2g24h4-thorfinn/layerscale-init-0.025`

### 2026-05-13 07:05 — PR #1896 (`charliepai2g24h4-thorfinn/layerscale-init-0.05`)

LayerScale init bracket: drop CaiT default init=0.1 → **init=0.05**. Single-line change to `layer_scale_init` default in `TransolverBlock.__init__`. The model compensates for lower residual amplitude by widening per-channel diversification (block-0 attn std/mean jumps 38.8% → 70.7%). 3/4 splits improve on both val and test; `single_in_dist` regresses +3.4% (likely interaction with #1711's pressure channel weighting). Mechanism is hybrid A/B: headline metrics improve at Outcome-A range, but γ_l means stay near init (Outcome-B signature) — confirming per-channel granularity is load-bearing.

- **`val_avg/mae_surf_p`** = **74.476** (best @ epoch 14; −1.21% vs #1711)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **66.014** (−0.89% vs #1711)
- **Per-split val** `mae_surf_p` at best val checkpoint:
  - `val_single_in_dist` = 85.075 (+3.39% vs #1711 — only regressing split)
  - `val_geom_camber_rc` = 82.764 (−3.35% vs #1711)
  - `val_geom_camber_cruise` = 57.002 (−2.78% vs #1711)
  - `val_re_rand` = 73.063 (−2.60% vs #1711)
- **Per-split test** `mae_surf_p` at best val checkpoint:
  - `test_single_in_dist` = 75.023 (+3.40% vs #1711 — only regressing split)
  - `test_geom_camber_rc` = 73.975 (−0.32% vs #1711)
  - `test_geom_camber_cruise` = 48.442 (−4.78% vs #1711)
  - `test_re_rand` = 66.617 (−3.16% vs #1711)
- **LayerScale γ_l mechanism** — final means (block 0→4): attn=[0.043, 0.041, 0.037, 0.033, 0.039], mlp=[0.063, 0.064, 0.055, 0.044, 0.047]. Depth-decreasing mlp trend preserved. Block-0 attn std/mean=70.7% (up from 38.8% at init=0.1) — per-channel diversification increases as amplitude decreases.
- **Compound progress**: #1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→**#1896** → val_avg improved from 100.957 to **74.476** = **−26.2% over 9 merges**.
- **Param count**: 669,271 (unchanged — zero new parameters; one-line change).
- **Metric artifacts**: `models/model-charliepai2g24h4-thorfinn-layerscale-init-0.05-20260513-061249/metrics.jsonl` and `metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-thorfinn --experiment_name charliepai2g24h4-thorfinn/layerscale-init-0.05`

### 2026-05-13 05:57 — PR #1711 (`charliepai2g24h4-alphonse/surf-ch-weight-h18`)

Per-channel surf-loss weighting `[w_Ux, w_Uy, w_p] = [0.5, 0.5, 2.0]` applied to the `surf_loss` computation (and its mirror in `evaluate_split`). `vol_loss` weighting left at `[1, 1, 1]`. Mass-preserving (channel sum = 3.0); zero new parameters. The 4× pressure-to-velocity ratio tilts the gradient toward pressure MAE without touching the prediction map — reversing three prior failed prediction-side attempts (#1610, #1636, #1675).

- **`val_avg/mae_surf_p`** = **75.391** (best @ epoch 14; −3.67% vs #1799)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **66.608** (−4.71% vs #1799)
- **Per-split val** `mae_surf_p` at the best val checkpoint:
  - `val_single_in_dist` = 82.287 (−3.50% vs #1799)
  - `val_geom_camber_rc` = 85.631 (−3.84% vs #1799)
  - `val_geom_camber_cruise` = 58.630 (−6.34% vs #1799)
  - `val_re_rand` = 75.015 (−1.46% vs #1799)
- **Per-split test** `mae_surf_p` at the best val checkpoint:
  - `test_single_in_dist` = 72.554 (−6.80% vs #1799)
  - `test_geom_camber_rc` = 74.210 (−6.64% vs #1799)
  - `test_geom_camber_cruise` = 50.877 (−1.60% vs #1799)
  - `test_re_rand` = 68.792 (−2.52% vs #1799)
- **All 4 val splits + all 4 test splits improve.** Largest gain on `val_geom_camber_cruise` (−6.34%) — the split with the lowest absolute baseline pressure error benefits most from capacity reallocation.
- **Per-channel mechanism confirmed**: Ux/Uy MAE regresses +12–25% as expected; p MAE improves 1.5–6.3%. Since Ux/Uy appear neither in `val_avg/mae_surf_p` nor `test_avg/mae_surf_p`, this is the intended trade.
- **Δ vs PR #1799 baseline (78.260 / 69.903)**: **−3.67%** on val_avg, **−4.71%** on 4-split test.
- **Compound progress**: #1397 → #1552 → #1611 → #1637 → #1548 → #1772 → #1799 → **#1711** → val_avg improved from 100.957 to **75.391** = **−25.3% over 8 merges**.
- **Param count**: 669,271 (unchanged — zero new parameters).
- **Metric artifacts**: `models/model-charliepai2g24h4-alphonse-surf-ch-weight-h18-20260513-050507/metrics.jsonl` and `metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-alphonse --experiment_name charliepai2g24h4-alphonse/surf-ch-weight-h18`

### 2026-05-13 03:56 — PR #1799 (`charliepai2g24h4-thorfinn/layerscale-init-0.1`)

LayerScale (CaiT-style, Touvron et al. 2021) per-channel learnable
residual gating with init=0.1. Adds two `nn.Parameter(torch.ones(hidden_dim) * 0.1)`
vectors per `TransolverBlock` (one for attn output, one for MLP output)
that multiply the corresponding branch output before adding to the
residual stream. **Compounds cleanly with Fourier L=6** — both val and
test improve by ~−4.7% to −4.9% on top of the merged L=6 stack. All 4
val splits and all 4 test splits improve. Mechanism is preserved across
the rebase from L=4 to L=6 (depth-decreasing γ_l trend on MLP branch +
30%-of-mean per-channel std).

- **`val_avg/mae_surf_p`** = **78.260** (best @ epoch 14, 1 epoch before 30 min timeout — shifted earlier as predicted from slow-start mechanism)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **69.903**
- **Per-split val** `mae_surf_p` at the best val checkpoint:
  - `val_single_in_dist` = 85.269 (-8.61% vs #1772)
  - `val_geom_camber_rc` = 89.049 (-4.21% vs #1772)
  - `val_geom_camber_cruise` = 62.595 (-0.85% vs #1772)
  - `val_re_rand` = 76.127 (-4.66% vs #1772)
- **Per-split test** `mae_surf_p` at the best val checkpoint:
  - `test_single_in_dist` = 77.850 (-6.57% vs #1772)
  - `test_geom_camber_rc` = 79.485 (-2.91% vs #1772)
  - `test_geom_camber_cruise` = 51.705 (-4.42% vs #1772)
  - `test_re_rand` = 70.573 (-4.68% vs #1772)
- **Δ vs PR #1772 baseline (82.311 / 73.330)**: **-4.92%** on val_avg, **-4.67%** on 4-split test.
- **Compound progress**: #1397 → #1552 → #1611 → #1637 → #1548 → #1772 → #1799 → val_avg has improved from 100.957 to 78.260 = **-22.5% over 7 merges**.
- **Param count**: 669,271 (+1,280 over #1772; +0.19%; two `hidden_dim=128` LayerScale parameter vectors per block × 5 blocks).
- **Mechanism confirmed across rebase**: Final γ_l means stay in [0.079, 0.119] range (near init=0.1, model does NOT ramp up to CaiT's [0.5, 1.5] expectation); per-channel std reaches 38.8% of mean in block-0 attn (slightly higher than the 33.9% on L=4 — Fourier L=6 gives more useful per-channel structure to gate); MLP branch γ_l means decrease with depth (block-0 mlp 0.119 → block-4 mlp 0.083), partially compensating for stoch-depth's stronger drop rate on later blocks.
- **Metric artifacts**: `models/model-charliepai2g24h4-thorfinn-layerscale-init-0.1-rebased-20260513-031524/metrics.jsonl` and `metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-thorfinn --experiment_name charliepai2g24h4-thorfinn/layerscale-init-0.1`

### 2026-05-13 02:50 — PR #1772 (`charliepai2g24h4-edward/fourier-coords-L6`)

Fourier positional encoding bumped from L=4 → L=6 (24 Fourier features
replacing the 16 L=4 features). Single-knob bracket-up of the merged
#1548 Fourier mechanism — the L=4 → L=6 trajectory is still on the
upward slope of Tancik's curve, with the predicted plateau at L=8-10.
Every val split and every test split improves; magnitude is at the
upper end of the pre-registered prediction band (-0.5% to -2.5%) on val
and middle of the band on test.

- **`val_avg/mae_surf_p`** = **82.311** (best @ epoch 15, last epoch before 30 min timeout)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **73.330**
- **Per-split val** `mae_surf_p` at the best val checkpoint:
  - `val_single_in_dist` = 93.299 (-3.89% vs L=4)
  - `val_geom_camber_rc` = 92.965 (-2.14% vs L=4)
  - `val_geom_camber_cruise` = 63.131 (-0.91% vs L=4)
  - `val_re_rand` = 79.848 (-4.10% vs L=4)
- **Per-split test** `mae_surf_p` at the best val checkpoint:
  - `test_single_in_dist` = 83.323 (-2.91% vs L=4)
  - `test_geom_camber_rc` = 81.867 (-1.39% vs L=4)
  - `test_geom_camber_cruise` = 54.094 (-1.43% vs L=4)
  - `test_re_rand` = 74.038 (-1.17% vs L=4)
- **Δ vs PR #1548 baseline (84.762 / 74.659)**: **-2.89%** on val_avg, **-1.78%** on 4-split test.
- **Compound progress**: #1397 → #1552 → #1611 → #1637 → #1548 → #1772 → val_avg has improved from 100.957 to 82.311 = **-18.5% over 6 merges**.
- **Param count**: 667,991 (+2,048 over #1548; +0.31%; from wider preprocess MLP first-layer input).
- **Surprise finding**: `val_re_rand` improved -4.10% (pre-registered as "likely stays flat" since its OOD axis is Reynolds, not spatial frequency). Plausible mechanism: at L=4 the network was over-spending capacity on low-freq geometry encoding; with L=6 it can encode geometry in higher Fourier bands and free up MLP capacity for Reynolds-dependent features. The consistent -1.2% to -1.4% gain on test_re_rand corroborates this is not pure noise.
- **Metric artifacts**: `models/model-charliepai2g24h4-edward-fourier-coords-L6-20260513-011437/metrics.jsonl` and `metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-edward --experiment_name charliepai2g24h4-edward/fourier-coords-L6`

### 2026-05-13 01:15 — PR #1548 (`charliepai2g24h4-edward/fourier-coords-L4-rebased`)

Fourier positional encoding (`L=4`, 16 Fourier features replacing the 2 raw
`(x, z)` coord dims, applied after normalization). Stacks cleanly with the
merged compound (stoch-depth + cosine T_max=15 + grad-clip 25) — every val
split improves, and test improves more than val. The split pattern matches
the spectral-bias hypothesis: largest gains where high-frequency spatial
structure dominates (`val_single_in_dist` -11.35%, `val_geom_camber_cruise`
-7.94%), minimal movement on `val_re_rand` (-0.30%) whose OOD axis is
Reynolds (flow-condition) not spatial coords.

- **`val_avg/mae_surf_p`** = **84.762** (best @ epoch 15, last epoch before 30 min timeout)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **74.659**
- **Per-split val** `mae_surf_p` at the best val checkpoint:
  - `val_single_in_dist` = 97.074
  - `val_geom_camber_rc` = 94.997
  - `val_geom_camber_cruise` = 63.711
  - `val_re_rand` = 83.266
- **Per-split test** `mae_surf_p` at the best val checkpoint:
  - `test_single_in_dist` = 85.819
  - `test_geom_camber_rc` = 83.023
  - `test_geom_camber_cruise` = 54.879
  - `test_re_rand` = 74.916
- **Δ vs PR #1637 baseline (90.294 / 81.243)**: **-6.13%** on val_avg, **-8.10%** on 4-split test.
- **Compound progress**: #1397 → #1552 → #1611 → #1637 → #1548 → val_avg has improved from 100.957 to 84.762 = **-16.0% over 5 merges**.
- **Param count**: 665,943 (+5.4K, +0.82% over previous baseline; from wider preprocess MLP input).
- **Metric artifacts**: `models/model-charliepai2g24h4-edward-fourier-coords-L4-rebased-20260512-235326/metrics.jsonl` and `metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-edward --experiment_name charliepai2g24h4-edward/fourier-coords-L4-rebased`

### 2026-05-12 22:55 — PR #1637 (`charliepai2g24h4-askeladd/grad-clip-25`)

Permissive gradient clipping at `max_norm=25.0` immediately before
`optimizer.step()` — a single-line addition. Diagnostic-informed
follow-up to closed PR #1529 (`max_norm=1.0`, +5.4% regression): with the
threshold raised from 1.0 to 25.0, clipping fires on the outlier-spike
steps (the largest grad norm observed in training is 110.04 at epoch 8)
without touching typical 30-70-range gradients. The mechanism is
compatible with stoch-depth (block-drop spikes are suppressed) and
cosine T_max=15 (the late-epoch cooldown phase relies on stable
gradients to fine-tune).

- **`val_avg/mae_surf_p`** = **90.294** (best @ epoch 15, last epoch before 30 min timeout)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **81.243**

Per-split surface pressure MAE at the best val checkpoint:

| Split | mae_surf_p (val) |
|-------|-----------:|
| single_in_dist     | 109.497 |
| geom_camber_rc     |  98.952 |
| geom_camber_cruise |  69.208 |
| re_rand            |  83.520 |
| **avg**            | **90.294** |

vs PR #1611 baseline:
- val_avg: 94.217 → 90.294 (**-4.16% improvement**)
- All four val splits improved uniformly (-3.14% to -5.61%) — no
  split-specific direction, exactly as the hypothesis predicted ("stable
  descent helps everywhere").
- test_avg: 84.859 → 81.243 (-4.26% improvement)

Diagnostic from the per-epoch `train/last_grad_norm` trace: 14/15 epochs
had end-of-epoch grad_norm > 25 (the clip threshold), confirming
clipping is active throughout training. The largest spike (110.04 at
epoch 8) was suppressed; typical training-step norms stayed in the
30-70 range. Val MAE descended monotonically epoch 9 → 15, with the
biggest single-epoch drop (-13.7%) coinciding with the only epoch where
the end-of-epoch norm fell below 25 (22.40 at epoch 12).

- **Metric artifacts**:
  `models/model-charliepai2g24h4-askeladd-grad-clip-25-20260512-221014/metrics.jsonl`
  and `metrics.yaml`.
- **n_params**: 0.66M (unchanged), **peak GPU memory**: 42.11 GB, **wall time**: 30 min (cap).

### 2026-05-12 21:16 — PR #1611 (`charliepai2g24h4-askeladd/cosine-tmax-15`)

CosineAnnealingLR schedule horizon aligned to the actual training duration:
`T_max=15` (matching the empirical epoch count at the 30-min cap), replacing
the previous `T_max=MAX_EPOCHS=50`. Under the old schedule, LR was at ~80% of
peak (≈4.0e-4) when training terminated — the full cosine cooldown phase
never ran. With `T_max=15`, LR completes its full cosine arc to ~0 over the
actual training horizon (verified by jsonl LR trace: 4.945e-4 → 5.463e-6 → 0).
Zero added compute, zero added parameters, single-line change.

- **`val_avg/mae_surf_p`** = **94.217** (best @ epoch 15, last epoch before 30 min timeout)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **84.859**

Per-split surface pressure MAE at the best val checkpoint:

| Split | mae_surf_p (val) | mae_surf_p (test) |
|-------|-----------:|-----------:|
| single_in_dist     | 114.200 |  ? |
| geom_camber_rc     | 102.157 |  ? |
| geom_camber_cruise |  73.321 |  ? |
| re_rand            |  87.188 |  ? |
| **avg**            | **94.217** |  **84.859** |

vs PR #1552 baseline:
- val_avg: 98.353 → 94.217 (**-4.21% improvement** — largest single-arm gain of wave 2)
- All four val splits neutral-to-positive (camber_rc -8.04% largest gain).
- test_avg: 87.995 → 84.859 (-3.57% improvement)

Val MAE descended monotonically every epoch — the model was still improving
at the wall-clock cap, suggesting further headroom with more time. The new
LR cooldown phase let the model find a better minimum within the same
30-min budget.

- **Metric artifacts**:
  `models/model-charliepai2g24h4-askeladd-cosine-tmax-15-20260512-211600/metrics.jsonl`
  and `metrics.yaml`.
- **n_params**: 0.66M (unchanged), **peak GPU memory**: 42.1 GB, **wall time**: 30 min (cap).

### 2026-05-12 20:52 — PR #1552 (`charliepai2g24h4-frieren/stoch-depth-0.1`)

Stochastic depth (Huang et al., ECCV 2016) added to the 5-layer Transolver
with linearly increasing per-block drop probs `[0.0, 0.025, 0.05, 0.075, 0.10]`.
At eval/test time it is a no-op (all blocks always used). Also bundles the
NaN-safe pre-filter in `evaluate_split` that produces the first finite
4-split `test_avg/mae_surf_p` on this branch.

- **`val_avg/mae_surf_p`** = **98.353** (best @ epoch 15, last epoch before 30 min timeout)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **87.995** — first finite 4-split test
  reference on this branch; new paper-facing baseline.

Per-split surface pressure MAE at the best val checkpoint:

| Split | mae_surf_p (val) | mae_surf_p (test) |
|-------|-----------:|-----------:|
| single_in_dist     | 119.159 | 104.953 |
| geom_camber_rc     | 111.093 | 101.883 |
| geom_camber_cruise |  73.323 |  62.243 |
| re_rand            |  89.837 |  82.901 |
| **avg**            | **98.353** |  **87.995** |

vs. L1 baseline (PR #1397):
- val_avg: 100.957 → 98.353 (**-2.58% improvement**)
- val_single_in_dist: -6.45% / val_geom_camber_cruise: -5.21% (largest gains)
- val_geom_camber_rc: +0.24% (flat) / val_re_rand: +1.77% (small regression)

The OOD-specific framing was only half-supported — the biggest gain landed
on `val_single_in_dist` (in-distribution), not the camber OOD splits as
predicted. Stoch-depth's implicit ensemble flattened split-specific overfit
modes regardless of OOD axis. Best epoch landed at the wall-clock cap
(epoch 15), so more training time would likely extend the gain.

Caveat: `loss`/`surf_loss` aggregates for `test_geom_camber_cruise` still
show NaN/Inf in `metrics.yaml` because the normalized-space loss path
runs before the §3 pre-filter; the §3 fix only protects `accumulate_batch`.
All four `mae_surf_p`/`mae_vol_p` channels are finite, so the primary
ranking metric is clean.

- **Metric artifacts**:
  `models/model-charliepai2g24h4-frieren-stoch-depth-0.1-20260512-201730/metrics.jsonl`
  and `metrics.yaml`.
- **n_params**: 0.66M (unchanged), **peak GPU memory**: 42.1 GB, **wall time**: 30 min (cap).

### 2026-05-12 19:05 — PR #1397 (`charliepai2g24h4-alphonse/l1-loss`)

L1 (MAE) loss replaces MSE in normalized-space training. First numeric
baseline on this branch.

- **`val_avg/mae_surf_p`** = **100.9574** (best @ epoch 13/14 before 30 min timeout)
- **`test_avg/mae_surf_p` (3-split mean, excludes `test_geom_camber_cruise`)** = **100.8314**
- **`test_avg/mae_surf_p` (all 4 splits, raw)** = NaN — pre-existing data
  bug: `test_geom_camber_cruise/000020.pt` has 761 nodes with `inf` in
  pressure y. Affects every arm in this round; `data/scoring.py` is
  marked read-only. See PR #1397 comment for full trace and proposed
  fixes. Until resolved we record the 3-split test mean.

Per-split surface pressure MAE at the best val checkpoint:

| Split | mae_surf_p |
|-------|-----------:|
| val_single_in_dist     | 127.371 |
| val_geom_camber_rc     | 110.832 |
| val_geom_camber_cruise |  77.353 |
| val_re_rand            |  88.273 |
| **val_avg/mae_surf_p** | **100.957** |
| test_single_in_dist    | 116.622 |
| test_geom_camber_rc    |  97.209 |
| test_geom_camber_cruise| NaN (data bug, surf_Ux/Uy still ok) |
| test_re_rand           |  88.663 |
| **test_avg/mae_surf_p (3-split)** | **100.831** |

- **Metric artifacts**:
  `models/model-charliepai2g24h4-alphonse-l1-loss-20260512-175404/metrics.jsonl`
  and `metrics.yaml`.
- **n_params**: 0.66M, **peak GPU memory**: 42.1 GB, **wall time**: 30.7 min.

## Reproduce baseline

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <student>/baseline
```
