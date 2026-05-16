# SENPAI Research Results — willow-pai2i-24h-r4

## 2026-05-16 10:35 — PR #3891: LayerNorm on FiLM conditioning input — ASSIGNED to thorfinn

- **Student/branch:** willowpai2i24h4-thorfinn / `willowpai2i24h4-thorfinn/layernorm_film_cond`
- **Hypothesis:** Add LayerNorm on the 11-dim FiLM cond vector before the first Linear. Raw features have badly mixed scales (log_Re ~13 vs AoA ~0.1 vs gap ~0.5 vs NACA ~0.05). FiLM head has to learn implicit scale invariance with only ~5K steps. LayerNorm gives well-conditioned signal, frees FiLM capacity for actual feature interactions. +22 params, zero FLOP increase — free-lunch optimization in budget-constrained regime.
- **Status:** WIP, smoke test mandatory first

---

## 2026-05-16 10:35 — PR #3890: p_channel_weight sweep {4, 5} — ASSIGNED to frieren

- **Student/branch:** willowpai2i24h4-frieren / `willowpai2i24h4-frieren/p_channel_weight_sweep`
- **Hypothesis:** Following frieren's own #3826 finding (surf_weight reduction = uniform regression, FiLM and surf_weight are complementary not redundant), test the opposite direction: increase p_channel_weight from 3 to {4, 5}. Increasing p emphasis redistributes within-surface gradient toward the primary metric (mae_surf_p) without reducing total surface signal. Zero compute overhead — pure CLI flag. Primary arm pcw=4, gated arm pcw=5 if pcw=4 beats baseline by ≥0.5 MAE.
- **Status:** WIP

---

## 2026-05-16 10:30 — PR #3826: surf_weight sweep {7, 5} — CLOSED (regression, mechanistically informative)

- **Student/branch:** willowpai2i24h4-frieren / `willowpai2i24h4-frieren/surf_weight_sweep`
- **Hypothesis:** Test whether 30× implicit surface bias (10×3) is over-tuned now that FiLM(cond_dim=11) provides explicit geometry conditioning.
- **W&B run:** `45jsh9p1` (sw=7, sw=5 gated and skipped per protocol)

| Metric | Baseline #3504 (sw=10) | sw=7 | Δ |
|--------|------:|-------:|---|
| val_avg/mae_surf_p | 67.30 | 68.75 | +2.16% |
| **test_avg/mae_surf_p** | **59.29** | **60.89** | **+2.69% (regression)** |
| test_single_in_dist | 69.69 | 71.00 | +1.31 |
| test_geom_camber_rc | 74.16 | 77.07 | +2.91 |
| test_geom_camber_cruise | 36.63 | 38.49 | +1.86 |
| test_re_rand | 56.69 | 57.00 | +0.32 |

- **Results commentary:** Mechanistic falsification — FiLM conditioning and surf_weight are **complementary, not redundant**. FiLM injects geometry priors into the forward pass (affine modulation); surf_weight controls which targets the optimizer prioritizes during backprop. Importantly, the camber-OOD splits (rc +2.91, cruise +1.86) regressed MOST — these are exactly the splits we'd expect to benefit if FiLM substituted for surf_weight. Instead, they're the most sensitive to weakening the surface gradient. Clean falsification.
- **Decision:** CLOSED — regression. Student correctly stopped at one arm per gating criterion. Reassigned to p_channel_weight sweep (PR #3890) following their own suggested follow-up #2.

---

## 2026-05-16 10:30 — PR #3761: slice_num sweep {96, 128} — CLOSED (budget-limited regression, capacity vs compute lost)

- **Student/branch:** willowpai2i24h4-thorfinn / `willowpai2i24h4-thorfinn/slice_num_cap`
- **Hypothesis:** slice_num=64 is partition-limited on single_in_dist (78.65, largest absolute error on #3258 baseline). Test {96, 128} to expand attention partition diversity.
- **W&B runs:** `zktfti2r` (smoke), `um7pnde4` (s=96, 12 epochs), `m4imamc2` (s=128, 11 epochs)

| Metric | Baseline #3258 (s=64, t=14) | Arm A: s=96, t=12 | Arm B: s=128, t=10 |
|--------|------:|-------:|-------:|
| n_params | ~687K | 692,599 (+5.3K) | 697,879 (+10.6K) |
| Epochs achieved | 14 | 12 | 11 |
| val_avg/mae_surf_p | 77.65 | 86.56 | 91.80 |
| **test_avg/mae_surf_p** | **66.87** | **75.26 (+12.6%)** | **83.27 (+24.5%)** |

- **Results commentary:** Capacity-vs-compute trade-off **lost by budget**. Both arms hit LR=0 (cosine schedule terminus) while val is still descending steeply. s=96 dropped 88.99→86.56 in the FINAL epoch (still on steep curve); s=128 dropped 96.73→91.80 in epoch 10 (LR=0 reached early). Param delta was 5× smaller than predicted (5.3K/10.6K vs 33K/65K) — slice projection is much cheaper than back-of-envelope, but per-step time grew 15-25%, forcing 2-3 epoch loss. single_in_dist hit hardest (consistent with capacity-hungry but confounded with global undertraining). PR ran against then-current #3258 baseline (test=66.87); both arms also worse vs current #3504 baseline (test=59.29).
- **Key insight:** Capacity hypothesis NEITHER confirmed NOR refuted at this compute envelope. The right falsification path is slice-softmax entropy logging (student's suggested follow-up #4), not raw slice_num bumps.
- **Decision:** CLOSED — clear regression. Reassigned to LayerNorm-on-FiLM-cond (PR #3891) — free-lunch optimization in same budget-constrained regime.

---

## 2026-05-16 08:50 — PR #3826: surf_weight sweep {7, 5} — ASSIGNED to frieren

- **Student/branch:** willowpai2i24h4-frieren / `willowpai2i24h4-frieren/surf_weight_sweep`
- **Hypothesis:** Test whether the 10× surf_weight (giving 30× implicit surface-p bias with p_channel_weight=3) is still optimal now that FiLM(cond_dim=11, mid=128) provides explicit AoA/NACA/gap/stagger conditioning. Prior surf_weight=5 attempt (#3406) failed on old cond_dim=1 base; the new geometry-aware FiLM changes the hypothesis. Primary arm sw=7 (21× effective bias), optional sw=5 gated on sw=7 winning by >1 MAE unit.
- **Status:** WIP

---

## 2026-05-16 08:50 — PR #3825: Per-block FiLM (cond_dim=11, 5 heads) — ASSIGNED to nezuko

- **Student/branch:** willowpai2i24h4-nezuko / `willowpai2i24h4-nezuko/per_block_film`
- **Hypothesis:** Replace single block-0 FiLM gate with 5 per-block FiLM heads (one before each Transolver block), each conditioned on the same 11 dims (log_Re, AoA_1, NACA_1×3, AoA_2, NACA_2×3, gap, stagger). Deeper blocks lose geometry/regime signal through attention transforms — per-block re-injection should compound cruise+re_rand gains. +71K params (+10.1%), no attention FLOPs added. Zero-init preserved on each head.
- **Status:** WIP

---

## 2026-05-16 08:30 — PR #3504: Richer FiLM conditioning (cond_dim=11, film_mid=128) — MERGED (R3 winner #2)

- **Student/branch:** willowpai2i24h4-frieren / `willowpai2i24h4-frieren/richer_film`
- **Hypothesis:** Extend FiLM conditioning from log_Re only (cond_dim=1) to all 11 per-sample-global scalars (log_Re, AoA_1, NACA_1×3, AoA_2, NACA_2×3, gap, stagger). Also widen film_mid from 64→128.
- **W&B run (winner):** `v38snhoe` (film_mid=128); also ran `6lpw8m00` (film_mid=64)

| Metric | Baseline (#3258) | mid64 arm | mid128 arm | Δ (mid128 vs baseline) |
|--------|------:|-------:|-------:|---|
| val_avg/mae_surf_p | 77.65 | 70.32 | **67.30** | **−13.31%** |
| **test_avg/mae_surf_p** | **66.87** | **61.03** | **59.29** | **−11.34%** |
| test_single_in_dist | 78.65 | 72.57 | 69.69 | −11.39% |
| test_geom_camber_rc | 77.66 | 72.42 | 74.16 | −4.51% |
| test_geom_camber_cruise | 46.17 | 39.66 | **36.63** | **−20.66%** |
| test_re_rand | 65.00 | 59.47 | **56.69** | **−12.79%** |

- **Results commentary:** Decisive mechanism win. AoA/NACA/gap/stagger conditioning orthogonal to RFF and grad-clip/warmup. Cruise (tandem-only) gains most (−20.66%) via gap/stagger features. re_rand gains −12.79%. The single_in_dist regression seen in R1 on old base is gone — RFF's spatial structure gives the model enough redundancy that FiLM's degenerate cond manifold on single-foil samples (6 zero features) doesn't hurt. mid128 > mid64 by ~3% test. Peak VRAM only 44.5 GiB. Rebased cleanly onto #3258.
- **New baseline:** test_avg/mae_surf_p = **59.29** (cumulative −44.2% from vanilla in 6 PRs).
- **Decision:** MERGED — decisive test win (−11.34%), +17,792 params (+2.5%), minimal VRAM.

---

## 2026-05-16 08:30 — PR #3618: Surface-only decoder head (parallel zero-init residual) — CLOSED (wash + regression)

- **Student/branch:** willowpai2i24h4-nezuko / `willowpai2i24h4-nezuko/surf_head`
- **Hypothesis:** Add parallel 128→128→3 residual head on surface nodes only, zero-init so model starts identical to baseline. Rationale: trunk compromised between surface (30× loss weight) and volume physics.
- **W&B run:** `x8jwcej7`

| Metric | Baseline (#3258) | PR #3618 | Δ |
|--------|------:|-------:|---|
| val_avg/mae_surf_p | 77.65 | 79.94 | +2.95% |
| **test_avg/mae_surf_p** | **66.87** | **69.62** | **+4.11%** (regression) |
| test_single_in_dist | 78.65 | 77.30 | −1.77% ✓ |
| test_geom_camber_rc | 77.66 | 79.27 | +2.07% |
| test_geom_camber_cruise | 46.17 | 50.53 | +9.43% ✗ |
| test_re_rand | 65.00 | 71.38 | +9.81% ✗ |

- **Results commentary:** Classic added-capacity-without-regularization on OOD splits (two ID splits improved, two OOD splits regressed). Net wash on old base (+0.50% vs prior #3262), clear regression (+4.11%) on new #3258 baseline. Zero-init epoch-1 parity confirmed (0.26%). The assumption was wrong: with surf_weight=10 × p_channel_weight=3 = 30× per-sample weighting, the shared trunk is already surface-biased by the loss. A dedicated head adds no new capacity — it trains on the same skewed gradient signal. Output-head specialization is not a missing degree of freedom on this baseline.
- **Decision:** CLOSED — regression on current baseline.

---

## 2026-05-16 07:22 — PR #3781: Re-conditioned loss reweighting α=1.0 — ASSIGNED to tanjiro

- **Student/branch:** willowpai2i24h4-tanjiro / `willowpai2i24h4-tanjiro/re_loss_weighting`
- **Hypothesis:** `test_single_in_dist` (78.65 — largest residual error) spans Re 104K-5M (widest range). Training distribution under-samples Re tails. Per-sample loss weight `1 + α*(log_Re_z)²` up-weights tail Re samples (2σ from median gets 5× weight at α=1.0). Weights normalized to preserve batch loss scale. Zero compute overhead — purely a loss modification.
- **Status:** WIP, sanity test at α=0.0 mandatory first

---

## 2026-05-16 07:30 — PR #3658: Transolver depth n_layers=6 — CLOSED (budget-limited)

- **Student/branch:** willowpai2i24h4-tanjiro / `willowpai2i24h4-tanjiro/depth`
- **Hypothesis:** Add one Transolver block (n_layers 5→6) to increase capacity.
- **Best W&B run:** `4zhloinn` (cosine_tmax=12, 12 epochs — best-calibrated of 3 full attempts)

| Metric | Baseline (#3258) | PR #3658 | Δ |
|--------|-------------:|-------:|---|
| val_avg/mae_surf_p | 77.65 | 85.40 | +10.0% |
| **test_avg/mae_surf_p** | **66.87** | **75.31** | **+12.6%** |
| test_single_in_dist | 78.65 | 87.10 | +10.7% |
| test_geom_camber_rc | 77.66 | 84.55 | +8.9% |
| test_geom_camber_cruise | 46.17 | 54.62 | +18.3% |
| test_re_rand | 65.00 | 74.98 | +15.4% |

- **Result commentary:** Budget-limited regression. +17% per-step compute → only 12 epochs vs baseline 14. Two lost epochs ≈ 7% budget-regression (matches ~−3.5%/epoch from cumulative path). Uniform regression across all splits confirms "lost 2 epochs everywhere" story — no split-specific benefit from depth. 5-layer is not depth-limited within 30-min cap; it's epoch-limited.
- **Key insight for future:** Within 30-min cap, capacity additions that increase per-step compute must deliver gains exceeding ~7% per 2 epoch-loss. Depth=6 didn't. Reassigned to Re-loss reweighting (zero compute overhead, targets Re-tail capacity).
- **Decision:** CLOSED — clear regression across all splits per own success criteria.

---

## 2026-05-16 07:03 — PR #3761: Slice_num capacity sweep {96, 128} — ASSIGNED to thorfinn

- **Student/branch:** willowpai2i24h4-thorfinn / `willowpai2i24h4-thorfinn/slice_num_cap`
- **Hypothesis:** `test_single_in_dist` (78.65 on current baseline) is the dominant bottleneck and may be partition-limited. slice_num=64 creates 64 physics partitions per block; bumping to 96/128 gives the model more granularity to separate laminar, turbulent, and near-foil flow regimes, especially on the widest Re range (104K–5M). Orthogonal to all in-flight optimizer/loss/depth experiments.
- **Motivation:** Drawn from thorfinn's #3468 analysis: "the mechanism overlap pattern suggests we're approaching the capacity ceiling of the current 5-block × 128-hidden trunk on single_in_dist." Slice_num is the most direct architectural attack on this ceiling.
- **Status:** WIP, smoke test mandatory before full run

---

## 2026-05-16 07:03 — PR #3468: Per-block FiLM v2 on RFF+cosine+FiLM stack — CLOSED

- **Student/branch:** willowpai2i24h4-thorfinn / `willowpai2i24h4-thorfinn/per_block_film`
- **Hypothesis:** Per-block FiLM heads (all 5 Transolver blocks) stacks with RFF+cosine+FiLM base.
- **W&B run:** `g7f1m73s` (name: `per-block-film-v2-on-rff-base`, group: `per-block-film`)

| Metric | Baseline #3258 | PR #3468 | Δ |
|--------|-------------:|-------:|---|
| val_avg/mae_surf_p | 77.65 | 79.66 | +2.57% |
| **test_avg/mae_surf_p** | **66.87** | **69.87** | **+4.49%** |
| test_single_in_dist | 78.65 | 80.60 | +2.43% |
| test_geom_camber_rc | 77.66 | 79.77 | +2.72% |
| test_geom_camber_cruise | 46.17 | 49.21 | +6.58% |
| test_re_rand | 65.00 | 69.91 | +7.56% |

*(Thorfinn compared to #3262 RFF baseline 69.27; re-stated above vs current merged baseline 66.87)*

- **Result commentary:** Negative result — full mechanism overlap with RFF on `test_single_in_dist`. The per-block FiLM mechanism that gave −6.73% on the #3263-only base was attacking the same capacity bottleneck that RFF + cosine now exploit. +67K params (+9.7%) without orthogonal mechanism = small regularization tax. Thorfinn's self-analysis was precise: "the whole regression comes from test_single_in_dist (+2.43%)" with other splits flat (±0.1-0.4%). Classic stacking failure due to mechanism overlap.
- **Decision:** CLOSED — clear regression vs both old and new baselines. Thorfinn reassigned to slice_num capacity bump (#3761).

---

## 2026-05-16 06:40 — PR #3746: Grad-clip cap sweep {10.0, 100.0} — ASSIGNED to fern

- **Student/branch:** willowpai2i24h4-fern / `willowpai2i24h4-fern/clip_cap_sweep`
- **Hypothesis:** clip=1.0 clips 100% of steps (min pre-clip norm=14.8 >> 1.0). Test whether a higher cap targeting only outlier batches outperforms the winning clip=1.0. Arm A clip=10.0 (still clips 100% of steps, but 10× larger effective step). Arm B clip=100.0 (~30% clipped — outlier-only targeting). No code changes needed.
- **Motivation:** fern's prior run (clip=1.0) showed slight in-dist val regression (+4.1%) while OOD gains were strong (cruise −6.10%, re_rand −6.68%). Outlier-only clipping may recover in-dist without sacrificing OOD protection.
- **Status:** WIP

---

## 2026-05-16 06:30 — PR #3258: Grad-clip 1.0 + warmup-5 — MERGED (R3 winner #1)

- **Student/branch:** willowpai2i24h4-fern / `willowpai2i24h4-fern/grad-clip-warmup`
- **Hypothesis:** Gradient clipping (max_norm=1.0) + 5-epoch linear LR warmup on the fully-stacked RFF+cosine+FiLM+surf-MAE base.
- **W&B run:** `vrnb926l` (name: `clip1.0-wu5-on-rff-base`, group: `grad-clip-warmup`)

| Metric | Baseline (#3262) | PR #3258 | Δ |
|--------|----------------:|-------:|---|
| val_avg/mae_surf_p | 79.28 | **77.65** | −2.06% |
| **test_avg/mae_surf_p** | 69.27 | **66.87** | **−3.47%** |
| test_single_in_dist | 78.69 | 78.65 | −0.06% |
| test_geom_camber_rc | 79.59 | 77.66 | −2.43% |
| test_geom_camber_cruise | 49.16 | 46.17 | −6.10% |
| test_re_rand | 69.65 | 65.00 | −6.68% |

- **Grad-norm trace:** median=70 (pre-clip), max=432, clips 100% of steps. Cosine base compressed max from 1110 → 432 (2.6×) but typical-step norm unchanged.
- **Result commentary:** Clear winner — OOD-concentrated gain pattern (re_rand −6.68%, cruise −6.10%) confirms gradient clipping is a variance-reduction intervention targeting high-Re and sharp-geometry outlier batches. The warmup + cosine schedule realized exactly as designed (warmup epochs 1–5 to 5e-4, then cosine T_max=9 to 0). The mechanism is orthogonal to all 5 prior wins and composes cleanly. Slight in-dist val regression (+4.1%) is within noise, and test_single_in_dist was essentially flat (−0.06%).
- **Decision:** MERGED — new baseline test=66.87.

---

## 2026-05-16 04:41 — PR #3693: Peak LR sweep {1e-3, 2.5e-4} — ASSIGNED to alphonse

- **Student/branch:** willowpai2i24h4-alphonse / `willowpai2i24h4-alphonse/lr_sweep`
- **Hypothesis:** lr=5e-4 has never been swept in this track — all 4 stacked wins trained at the original Transolver value. With cosine T_max=14 annealing to 0, peak LR controls average effective step size. Two-arm bracket: Arm A lr=1e-3 (primary, predicted best — doubles average effective step, canonical ~1M-param transformer value), Arm B lr=2.5e-4 (control — halves effective step, tests if 5e-4 was already aggressive). No code changes needed — `lr` is a CLI flag.
- **Arms:**
  - Arm A: `python train.py --wandb_group lr-sweep --wandb_name lr-1e-3-on-rff-base --lr 1e-3` (predicted test ~66–68)
  - Arm B: `python train.py --wandb_group lr-sweep --wandb_name lr-2.5e-4-on-rff-base --lr 2.5e-4` (control, predicted slight regression)
- **Motivation:** Prior #3565 AdamW sweep (wd=0.05, beta2=0.95) ran on PRE-RFF base (confirmed via W&B config absence of fourier_n_freqs/fourier_sigma). Peak LR is the cleaner next optimizer lever — has never been tested on the full FiLM+cosine+RFF stack.
- **Status:** WIP, both arms sequential (~64 min total)

---

## 2026-05-16 04:25 — PR #3565: AdamW betas + weight_decay sweep — CLOSED (pre-RFF base, invalid comparison)

- **Student/branch:** willowpai2i24h4-alphonse / `willowpai2i24h4-alphonse/adamw_sweep`
- **Hypothesis:** AdamW (beta1=0.9, beta2=0.95 or 0.999) + weight_decay sweep (0.05 or 0.1) on full stack.
- **W&B run:** `inuxz240` (arm B: beta2=0.999, wd=0.05; only arm that ran)
- **Result:** val_avg=82.08, test_avg=72.88 — **+5.2% above 69.27 baseline** — but W&B config confirms run was on PRE-RFF base (`fourier_n_freqs`/`fourier_sigma` both absent from logged config). Invalid comparison; experiment measures AdamW tuning on an older, weaker base.
- **Key observation:** WD=0.05 arm regressed even vs its own base. Confirms model is **capacity-limited, not noise-limited** — weight decay reduces effective parameter motion on an under-parameterized model.
- **Decision:** CLOSED — invalid base; negative signal on high WD still informative. LR sweep is the clean follow-up (assigned as #3693).

---

## 2026-05-16 03:35 — PR #3658: Transolver depth n_layers=6 test — ASSIGNED to tanjiro

- **Student/branch:** willowpai2i24h4-tanjiro / `willowpai2i24h4-tanjiro/depth`
- **Hypothesis:** Add one Transolver block (5 → 6) to test depth-limited capacity on the FiLM+RFF-conditioned input. First architecture-side experiment in this track (4 prior wins were all loss/input/schedule).
- **Predicted impact:** Conservative 1–3% (test ~67–68), hopeful 5% (test ~65–66). +114K params (+17%), +17% per-step compute → student should drop cosine_tmax to ~11-12 to match achievable epochs under 30-min cap.
- **Status:** WIP, smoke test mandatory before full run

---

## 2026-05-16 03:30 — PR #3406: surf_weight sweep — CLOSED (re-run regressed on new base)

- **Student/branch:** willowpai2i24h4-tanjiro / `willowpai2i24h4-tanjiro/surf-weight-sweep`
- **Hypothesis:** sw5 sweep arm winner from R1 (frieren-only base, val=98.30/test=88.80) re-run on full FiLM+RFF stack.
- **W&B run:** `2kulmdv6` (sw5-on-film-base, surf_weight=5, all other configs at merged defaults)
- **Result:** val_avg=81.22, test_avg=72.17 — **+4.17% above 69.27 baseline**, all 4 splits regressed (+0.4% / +1.1% / +11.1% / +7.1%). Largest hits on `test_geom_camber_cruise` (+11.1%) and `test_re_rand` (+7.1%) — the two splits FiLM+RFF were already strongest on.
- **Decision:** CLOSED — student's analysis correctly identified the mechanism: loss rebalancing was unlocking headroom on the simpler base that FiLM+RFF now provide structurally. Reducing surf_weight on the new base muffles the dominant gradient signal without freeing useful budget elsewhere. **Loss reweighting lever is exhausted on this stack.**

---

## 2026-05-16 02:00 — PR #3618: Surface-only decoder head (parallel zero-init residual) — ASSIGNED to nezuko

- **Student/branch:** willowpai2i24h4-nezuko / `willowpai2i24h4-nezuko/surf_head`
- **Hypothesis:** Add a parallel 128→128→3 surface-only decoder head on `h = ln_3(fx)` after block 5, zero-init final layer, gated by `is_surface`. Architectural surface specialization — first output-head split in this track. Orthogonal to all 4 prior wins (loss / input conditioning / schedule / input encoding).
- **Predicted impact:** Conservative 2–4% (test ~66–68), hopeful 5–7% (~64–66). Pessimistic wash. +16,899 params (+2.5%). Zero-init keeps initial loss within ~1% of baseline.
- **Status:** WIP, awaiting first run

---

## 2026-05-16 01:55 — PR #3550: Volume MAE reformulation (unify L1 across surface + volume) — CLOSED

- **Student/branch:** willowpai2i24h4-nezuko / `willowpai2i24h4-nezuko/volume_mae`
- **Hypothesis:** Replace MSE volume loss with MAE for L1 consistency surface↔volume.
- **W&B run:** result run on OLD pre-RFF base
- **Result:** test_avg/mae_surf_p ≈ 83.85 (+4.7% vs #3358 baseline 80.08 on old base; +21% above current 69.27 baseline)
- **Decision:** CLOSED — the volume MSE→MAE swap regressed even before factoring in the RFF baseline shift. The asymmetry (MSE volume, MAE surface) appears load-bearing: MSE on the volume provides rich gradient signal across the dense mesh, while MAE on the surface aligns with the ranking metric. Unifying loses information.

---

## 2026-05-16 01:27 — PR #3262: Random Fourier Features σ=1.0 — MERGED (R2 winner #2)

- **Student/branch:** willowpai2i24h4-edward / `willowpai2i24h4-edward/fourier-pos-enc`
- **Hypothesis:** Replace raw (x, z) coordinate inputs with 32-dim RFF basis (16 sin+cos pairs, σ=1.0). Resolves higher-frequency surface pressure patterns.
- **W&B run:** `tnna02ob` (run on fully-stacked #3358+#3263+#3257 base)

### Result vs #3358 baseline (val=90.44, test=80.08)

| Metric | Baseline `b9qv36aq` | RFF `tnna02ob` | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 90.44 | **79.28** | **−12.3%** |
| `test_avg/mae_surf_p` | 80.08 | **69.27** | **−13.5%** |
| `test_single_in_dist` | 96.49 | 78.69 | **−18.4%** |
| `test_geom_camber_rc` | 90.24 | 79.59 | −11.8% |
| `test_geom_camber_cruise` | 55.95 | 49.16 | −12.1% |
| `test_re_rand` | 77.65 | 69.65 | −10.3% |

Best epoch: 14/50. Wall-clock: 32.3 min. Params: 687,319 (+8K over #3358 base).

### Decision: MERGED as R2 winner #2

**Analysis:** Largest single gain in this track (−13.5% test). Gain SCALES UP on the better base (old MSE gain: −9.8% val → new stacked base: −12.3% val). Biggest gain on hardest split (single_in_dist: −18.4%). All four mechanisms (loss, FiLM, schedule, input encoding) are orthogonal and composing constructively. Cumulative path: vanilla 106.23 → #3257 94.35 → #3263 90.06 → #3358 80.08 → #3262 **69.27** (−34.8% from vanilla).

---

## 2026-05-16 01:30 — PR #3504: Richer FiLM conditioning (cond_dim 1→11) — SENT BACK (rebase needed)

- **Student/branch:** willowpai2i24h4-frieren / `willowpai2i24h4-frieren/richer_film`
- **Hypothesis:** Extend FiLM conditioning cond_dim=1 (log_Re) → 11 (log_Re + AoA_1 + NACA_1 + AoA_2 + NACA_2 + gap + stagger).
- **W&B runs:** v1 `tkxld39k` (film_mid=64), mid128-v1 `y4bphsbs` (film_mid=128) — both on OLD #3263 base

### Result vs #3263 baseline (val=100.24, test=90.06) — OLD base, pre-#3358 and pre-#3262

| Arm | test_avg | vs #3263 | vs CURRENT (69.27) |
|-----|--------:|--------:|---------:|
| v1 film_mid=64 (`tkxld39k`) | 83.61 | −7.16% | +20.7% above |
| mid128 (`y4bphsbs`) | **81.07** | **−9.98%** | +17% above |

### Decision: SENT BACK for rebase+rerun on full RFF-stacked base

**Analysis:** Mechanism strong (−9.98% on ref base). Timing complication: #3358 + #3262 merged while running, raising the target. VRAM concern: mid128 at 94.0 GiB; start with film_mid=64. Predicted on new stack: hopeful test ~64.4 (−7%), conservative ~65.8–67.2.

---

## 2026-05-16 00:25 — PR #3429: Multi-scale slice tokens — CLOSED

- **Student/branch:** willowpai2i24h4-nezuko / `willowpai2i24h4-nezuko/multi-scale-slice`
- **Hypothesis:** Parallel coarse-global (slice_num=64) + fine-surface (slice_num_surf=32) groups, with a separate routing softmax for the surface group.
- **W&B runs:** primary `n7qecihj`, control (multi-scale OFF) `8pogqq8d`

### Result vs OLD #3263 baseline (val=100.24, test=90.06)

| Arm | val_avg | test_avg | s/epoch | epochs in 30min | best epoch |
|-----|--------:|---------:|--------:|----------------:|-----------:|
| ms-slice-32 (primary) | 115.21 | **104.60** (+16.7%) | 163 | 11 | 11 |
| ms-slice-control | 99.93 | 89.62 | 131 | 14 | 13 |

### Equal-epoch comparison (at epoch 11)
| Arm | val_avg/mae_surf_p |
|-----|---:|
| ms-slice-32 | 115.21 |
| ms-slice-control | 114.87 |

Within 0.3% — the multi-scale routing added NO architectural quality. The headline regression came entirely from the +24% per-epoch wall-clock cost stealing 3 epochs from the timeout-capped training.

### Decision: CLOSED (hypothesis failed)

### Analysis
- **Slice routing is not the right lever for surface specialization.** 64 global slices already cover surface structure adequately; adding 32 routing-disjoint slices creates redundant tokens.
- **`is_surface` is already in `x[..., 12]`** — features carry surface info pre-routing. Routing-layer specialization is too shallow.
- **Architectural changes that slow per-epoch wall-clock get punished by the 30-min cap.** ms-slice-32 used +35% peak VRAM (57.1 GB vs 42.1 GB control) AND +24% s/epoch.
- **Future surface specialization should go deeper:** separate Q/K/V heads, surface-only decoder branching after the trunk.
- **Useful side observation:** the control vs published baseline (89.62 vs 90.06 test) suggests single-seed run-to-run variance is ~±5% on this metric — worth recording for future borderline calls.

---

## 2026-05-16 00:24 — PR #3358: Cosine LR T_max=14 (matched to wall-clock cap) — MERGED (R2 WINNER #1)

- **Student/branch:** willowpai2i24h4-alphonse / `willowpai2i24h4-alphonse/cosine-tmax-fix`
- **Hypothesis:** Align cosine `T_max` to the wall-clock 14-epoch cap (was set to 50 epochs). Mechanically orthogonal to loss + architecture.
- **W&B run:** `b9qv36aq`

### Result vs prior #3263 baseline (val=100.24, test=90.06)

| Metric | Baseline #3263 | **#3358** | Δ |
|--------|---------------:|----------:|---:|
| `val_avg/mae_surf_p` | 100.24 | **90.4369** | **−9.78%** |
| `test_avg/mae_surf_p` | 90.06 | **80.0794** | **−11.08%** |
| `test_single_in_dist/mae_surf_p` | 119.11 | 96.49 | **−19.0%** |
| `test_geom_camber_rc/mae_surf_p` | 100.27 | 90.24 | −10.0% |
| `test_geom_camber_cruise/mae_surf_p` | 58.62 | 55.95 | −4.6% |
| `test_re_rand/mae_surf_p` | 82.27 | 77.65 | −5.6% |

### Decision: **MERGED as R2 winner #1**

Wins all 4 val splits and all 4 test splits. The new baseline.

### Analysis
- **Mechanism cleanly verified.** LR trace decays to 0 at epoch 14 (was at 4.09e-04 = 82% of peak at termination under T_max=50).
- **Larger gain than predicted hopeful case** (val 92–96 / test 81–87 predicted; actual val 90.44 / test 80.08).
- **Stacks additively with R1 winners.** Cumulative: vanilla 106.23 → #3257 frieren loss 94.35 → #3263 single FiLM 90.06 → #3358 cosine T_max=14 80.08 (−24.6% from vanilla in 3 PRs).
- **The single_in_dist reversal is the strongest evidence of mechanism.** On the old MSE base this was the only split where T_max=14 *lost* vs T_max=50. On the new FiLM+MAE base it's the biggest winner. The schedule fix interacts constructively with the merged base.

---

## 2026-05-16 00:24 — PR #3468: Per-block FiLM heads — SENT BACK (rebase)

- **Student/branch:** willowpai2i24h4-thorfinn / `willowpai2i24h4-thorfinn/per_block_film`
- **Hypothesis:** Extend single FiLM (#3263) to per-block heads (4 heads gating blocks 0–3 outputs; block 4 is last_layer so output is 3-dim).
- **W&B runs:** v1 post-block `1pggtz8o`, pre-block `nogqo7co`

### Result vs OLD #3263 baseline (val=100.24, test=90.06)

| Arm | val_avg | test_avg | Δ test |
|-----|--------:|---------:|-------:|
| **v1 post-block (4 heads)** | **94.011** | **84.001** | **−6.73%** |
| pre-v1 pre-block (5 heads) | 101.700 | 91.371 | +1.46% |

### Per-split (v1 vs OLD #3263)
| split | baseline | v1 | Δ |
|-------|---------:|---:|---:|
| `test_single_in_dist`     | 119.11 |  99.75 | **−16.25%** |
| `test_geom_camber_rc`     | 100.27 |  99.01 | −1.25% |
| `test_geom_camber_cruise` |  58.62 |  56.94 | −2.87% |
| `test_re_rand`            |  82.27 |  80.30 | −2.39% |

### Decision: SENT BACK for rebase + rerun on new #3358 baseline

Clean win on the OLD base (−6.73%) but new baseline is now test=80.08 (#3358 cosine fix merged). Per-block FiLM mechanism is orthogonal to cosine schedule, but biggest per-split gains (single_in_dist) OVERLAP — both per-block FiLM (−16.25%) and cosine fix (−19.0%) target the same split.

Student to rebase + rerun v1 only on new baseline. Predicted on rebased base:
- Hopeful (full stacking): test ~73–77 (massive merge)
- Conservative (partial overlap on single_in_dist): test ~76–80 (strong merge)
- Pessimistic (full overlap): test ~78–82 (marginal)

### Strong implementation notes
- Student self-resolved the `last_layer=True` issue (block 4 outputs 3-dim, not 128-dim).
- Pre-block placement loss (+1.46%) is a useful negative result — confirms post-block is the right design.
- Zero-init identity verified vs un-FiLM'd Transolver with shared weights (max abs diff = 0.0).

---

## 2026-05-15 23:21 — PR #3386: Re-stratified loss reweighting (1/per_sample_y_std) — CLOSED

- **Student/branch:** willowpai2i24h4-frieren / `willowpai2i24h4-frieren/restratified-loss`
- **Hypothesis:** Per-sample inv-std reweighting on surface MAE — equalize gradient contribution across samples whose `y_norm` p-channel has wildly different dynamic range (cruise small, single_in_dist large).
- **W&B runs:** v2 (clip+normalize) `1d39kc2y`, clip_only `d9r8l5um`, primary (no clip/no norm) `axeuhzhy`

### Result (vs frieren-only #3257 baseline, val=106.67/test=94.35 — NOT against current FiLM+frieren base)

| Arm | val_avg | test_avg | Δ test |
|-----|--------:|---------:|--------|
| baseline #3257 | 106.67 | 94.35 | — |
| v2 (clip + normalize) | 106.53 | **95.98** | **+1.7%** |
| clip_only | 151.56 | 140.21 | +48.6% |
| primary (no clip/no norm) | 261.24 | 245.83 | +160.6% |

### Per-split test (v2 vs baseline)
| split | baseline | v2 | Δ | predicted |
|-------|---------:|---:|---:|-----------|
| `single_in_dist`     | 122.34 | 125.06 | +2.2% | flatten/slight regress ✓ |
| `geom_camber_rc`     | 106.31 | 101.55 | −4.5% | (not predicted) |
| `geom_camber_cruise` |  62.47 |  65.64 | **+5.1%** | predicted −5–10% ✗ |
| `re_rand`            |  86.28 |  91.66 | **+6.2%** | predicted −5–10% ✗ |
| **avg**              |  94.35 |  95.98 | **+1.7%** | predicted −5–8% ✗ |

### Decision: CLOSED (hypothesis failed)

The mechanism worked exactly as designed (`train/sample_inv_std_*` traces confirm) — cruise samples got higher gradient weight, single_in_dist got lower. But the predicted per-split improvements inverted: both cruise (+5.1%) and re_rand (+6.2%) regressed. Student recommended dropping the hypothesis.

### Analysis
- The merged loss (MAE + p_weight=3) already captures the right per-sample gradient structure for the four-split objective.
- Re-stratification is zero-sum: boosting cruise's gradient share reduces single_in_dist's; single_in_dist is the harder split, so the model loses signal there faster than it gains on cruise.
- The hypothesis would have been even less likely to help on the current FiLM+frieren baseline, where FiLM provides a structural Re-conditioned route that already addresses cross-regime dynamic-range variation.
- Strong negative result for the writeup.

---

## 2026-05-15 23:21 — PR #3406: surf_weight sweep {5,10,20} — SENT BACK (rebase)

- **Student/branch:** willowpai2i24h4-tanjiro / `willowpai2i24h4-tanjiro/surf-weight-sweep`
- **Hypothesis:** Sweep `surf_weight ∈ {5, 10, 20}` on the merged surf-MAE+p_weight=3 base. With p_channel_weight=3 baked in, sw=10 → 30× effective weighting on pressure, potentially over-rotating loss.
- **W&B runs (on OLD frieren-only base):** sw5 `h3cfjdwu`, sw10 `acxfxpnj`, sw20 `pajdxlc1`

### Result (vs frieren-only #3257 baseline, val=106.67/test=94.35 — NOT against current FiLM+frieren base)

| Arm | val_avg | test_avg | Δ test |
|-----|--------:|---------:|-------:|
| **sw5** ⭐ | **98.30** | **88.80** | **−5.88%** |
| sw10 (control) | 104.02 | 91.75 | −2.76% |
| sw20 | 102.04 | 91.36 | −3.17% |

sw5 wins decisively on the OLD base. But the experiment was on frieren-only base (#3257), not the current FiLM+frieren merged base (#3263). Cannot conclude sw5 still wins after FiLM is added without re-running.

### Per-split test (sw5 vs OLD baseline)
| split | baseline | sw5 | Δ |
|-------|---------:|----:|---:|
| `single_in_dist`     | 122.34 | 104.96 | **−14.2%** |
| `geom_camber_rc`     | 106.31 | 101.83 | −4.2% |
| `geom_camber_cruise` |  62.47 |  61.64 | −1.3% |
| `re_rand`            |  86.28 |  86.76 | +0.6% |
| **avg**              |  94.35 |  88.80 | **−5.88%** |

### Decision: SENT BACK for rebase + rerun on new merged base

The mechanism is well-supported on the old base. But sw5's numerical 88.80 vs current baseline 90.06 (−1.4%) is **not paired** — we need a paired run on the FiLM+frieren base before we can merge. Student instructed to:
1. Rebase onto current advisor head (gets FiLM merged)
2. Change `surf_weight` default to 5.0 in Config (codifies the sweep winner)
3. Re-run sw5 only on the new base. The merged baseline #3263 (W&B `69jp9tvt`) is the paired control.

Predicted on rebased base: hopeful val ~92–96 / test ~83–87 (sw5+FiLM stacks); conservative val ~96–100 / test ~86–89 (smaller gain because FiLM already captured some); pessimistic val ~99–102 / test ~89–92 (wash).

---

## 2026-05-15 14:10 — PR #3257: Surface MAE loss + pressure-channel weight 3×

- **Student/branch:** willowpai2i24h4-frieren / `willowpai2i24h4-frieren/surf-mae-p-weight`
- **Hypothesis:** Switch surface loss from MSE to MAE and weight the p channel 3× to align the training signal with `test_avg/mae_surf_p`. Volume loss kept as MSE.
- **W&B run:** `zz2r70lt` (https://wandb.ai/wandb-applied-ai-team/senpai-v1/runs/zz2r70lt)

### Result (best checkpoint at epoch 13 of 14; timeout cap)

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------:|----------------:|
| `single_in_dist`        | 115.42 | 106.18 |
| `geom_camber_rc`        | 119.27 | 104.56 |
| `geom_camber_cruise`    |  73.02 | **NaN** (non-finite p preds) |
| `re_rand`               |  91.93 |  86.33 |
| **avg**                 | **99.91** | **NaN** |

Peak GPU: 42.1 GB / 96 GB. Wall time: 30.8 min (hit cap). Val curve still descending at termination (ep 1 → 228.01, ep 13 → 99.91, ep 14 → 122.76).

### Decision: send back to student (#3257-comment-4460628326)

`test_avg/mae_surf_p = NaN` is disqualifying per advisor protocol — the primary ranking metric for the paper-facing comparison must be finite.

### Root cause

`data/scoring.py:accumulate_batch` skips samples with non-finite **ground truth** but does not guard against non-finite **predictions** — one runaway pred poisons the running sum. `data/scoring.py` is read-only, so the fix has to live in `train.py:evaluate_split`. Sent back with explicit `torch.nan_to_num(pred_orig, nan=0.0, posinf=1e6, neginf=-1e6)` patch + `n_nonfinite_pred` per-split diagnostics, plus instructions to rerun the same arm.

### Analysis

- Per-split val shape matches the hypothesis prediction (cruise/re_rand carry the gain), suggesting the loss change is doing what we wanted — we just can't confirm the test-side number until the rerun.
- Training was wall-clock-capped at epoch 14/50 — the cosine schedule was set for `T_max=50` but only ~14 epochs run. The model never saw the low-LR end of the schedule. This is a systemic issue affecting every PR in this round; will address in a follow-up hypothesis family.
- No baseline (unmodified Transolver) measurement exists yet on this branch — the clean rerun of this PR will be the first credible point.

## 2026-05-15 17:35 — PR #3261: Wider-shallower Transolver (n_hidden=256, n_layers=3, n_head=8)

- **Student/branch:** willowpai2i24h4-alphonse / `willowpai2i24h4-alphonse/wider-shallower-256d`
- **Hypothesis:** Wider-shallower configuration — 2.45× more params, 3/5 depth — should improve in-distribution capacity with better per-layer width.
- **W&B runs:** baseline `xfayvdk2` (with NaN guard), wider `qjzx09k6`

### Result (best checkpoint, timeout at epoch 10–14 / 50)

| | Baseline `xfayvdk2` | Wider `qjzx09k6` | Δ |
|---|---:|---:|---:|
| n_params | 0.66M | 1.61M (+145%) | — |
| Epochs completed | 14 (best@13) | 11 (best@10) | wider → fewer steps |
| `val_avg/mae_surf_p` | **117.89** | 146.26 | **+24.1% WORSE** |
| `val_geom_camber_rc` | 125.09 | 176.33 | +41% WORSE (largest gap) |
| `test_avg/mae_surf_p` | **106.23** | 133.34 | **+25.5% WORSE** |
| `test_geom_camber_cruise` | **78.72** | 96.07 | finite (NaN guard applied) |

### Decision: closed (>5% regression on primary metrics)

### Analysis

- **Hypothesis disconfirmed.** Wider-shallower is substantially worse across all splits. The largest degradation is on `val_geom_camber_rc` (+41%) — the hardest OOD split — consistent with depth being critical for compositional generalization.
- **Params not matched.** The PR description implied "roughly matched budget" but actual ratio is ×2.45 (0.66M → 1.61M). Depth halving (5→3) doesn't compensate width doubling (128→256) because attention and MLP both scale as O(d²).
- **Fewer epochs under wall-clock cap.** Per-epoch wall time: 132s → 165s (+25%). 11 epochs at best vs 14 for baseline — introduces compounding unfairness in a still-converging regime.
- **Depth is doing real compositional work.** The depth-5 pattern is empirically validated. Will not revisit width-vs-depth ablations at matched budget in the near term.
- **MOST VALUABLE CONTRIBUTION: alphonse's vanilla baseline `xfayvdk2`.** This is the first run with NaN guard applied, giving us the **first finite 4-split test_avg reference**: val_avg=117.89, test_avg=106.23. Added to BASELINE.md.

## 2026-05-15 16:45 — PR #3264: Dropout p=0.1 in Transolver MLP and attention

- **Student/branch:** willowpai2i24h4-askeladd / `willowpai2i24h4-askeladd/dropout-0.1`
- **Hypothesis:** Enable dropout=0.1 in both PhysicsAttention and MLP pathways to reduce overfitting on ~1500-sample training set and improve OOD camber generalization.
- **W&B runs:** baseline `j4y20e31`, dropout=0.1 `chzqcfyz`

### Result (best checkpoint, both runs hit 30-min timeout at epoch 13 / 50)

| Split | Baseline (d=0) | **dropout=0.1** | Δ |
|-------|------:|------:|---:|
| `val_single_in_dist/mae_surf_p`     | 153.89 | 170.57 | +10.8% WORSE |
| `val_geom_camber_rc/mae_surf_p`     | 139.60 | 146.70 | +5.1% WORSE |
| `val_geom_camber_cruise/mae_surf_p` | 106.15 | 115.73 | +9.0% WORSE |
| `val_re_rand/mae_surf_p`            | 130.71 | 129.28 | −1.1% (noise) |
| **`val_avg/mae_surf_p`**            | **132.59** | **140.57** | **+6.0% WORSE** |
| `test_avg/mae_surf_p` (3 valid) | 131.53 | 136.27 | +3.6% WORSE |

### Decision: closed (>5% regression on primary val metric)

Dropout=0.1 degrades performance across 3 of 4 val splits including both OOD camber splits it was meant to help. Closed per CLAUDE.md protocol (>5% regression).

### Analysis

- **Mechanism correctly diagnosed by askeladd:** model is in UNDERFIT regime at the 30-min cap (val curve still descending at epoch 13, training loss still falling from 1.09 to 0.27). Dropout slows convergence (partially-masked subnetworks reduce effective gradient signal per step). Slower convergence × same wall-clock = strictly worse best-checkpoint on a still-descending loss curve. This is not an overfitting regime.
- **The single split where dropout helped (`val_re_rand`, −1.1%)** is the Re-stratified holdout — plausibly the axis most prone to co-adaptation memorization, but the effect is noise-level (1.43 MAE on 130-MAE base).
- **Dropout may not be worth retrying** at this compute budget. Would only be worth testing once models consistently train past 25+ epochs and val curves start to plateau. Marked as low-priority for future rounds.

### Critical diagnostic contribution: correct NaN root cause

Askeladd's bug-report comment (16:28 UTC) independently found the same root cause that frieren found (#3257, 16:32 UTC): `+inf` in `test_geom_camber_cruise_gt/000020.pt` y-channel for p at ~761 nodes. The actual IEEE 754 bug: `(pred - inf).abs() = inf`, then `inf * 0 = NaN` in `(err * mask).sum()` — poisons the running sum even though `accumulate_batch` would otherwise skip the sample via `y_finite`. **My original patch (sanitize predictions) was wrong.** The correct fix (from frieren's #3257 commit `34600cf`): zero the mask for non-finite-y samples AND `nan_to_num` y before `accumulate_batch`. Sent corrected patches to #3257, #3258, #3262.

## 2026-05-15 15:30 — PR #3262: Random Fourier Features positional encoding

- **Student/branch:** willowpai2i24h4-edward / `willowpai2i24h4-edward/fourier-pos-enc`
- **Hypothesis:** Augment input with Random Fourier Features (RFF; Tancik et al. 2020) of unnormalized (x, z) coordinates. n_freqs=16, swept σ ∈ {1.0, 4.0}. Baseline measured in parallel (no RFF).
- **W&B runs:** baseline `17fia1vd`, σ=1.0 `vlv1b0ab`, σ=4.0 `q9vkl63z`

### Result (best checkpoint, all runs hit 30-min timeout cap at epoch 13–14 / 50)

| Split | Baseline | σ=1.0 | σ=4.0 | σ=1.0 Δ |
|-------|---------:|------:|------:|--------:|
| `val_single_in_dist/mae_surf_p`     | 155.71 | **140.41** | 157.36 | −9.8% |
| `val_geom_camber_rc/mae_surf_p`     | 136.10 | **120.10** | 146.07 | −11.8% |
| `val_geom_camber_cruise/mae_surf_p` | 103.19 | **92.11**  | 101.16 | −10.7% |
| `val_re_rand/mae_surf_p`            | 118.38 | **110.49** | 118.94 | −6.7% |
| **`val_avg/mae_surf_p`**            | **128.34** | **115.78** | 130.88 | **−9.8%** |
| `test_single_in_dist/mae_surf_p`    | 135.28 | 119.89 | 139.12 | −11.4% |
| `test_geom_camber_rc/mae_surf_p`    | 128.51 | 108.99 | 132.54 | −15.2% |
| `test_geom_camber_cruise/mae_surf_p`| **NaN** | **NaN** | **NaN** | — |
| `test_re_rand/mae_surf_p`           | 118.07 | 104.24 | 114.87 | −11.7% |
| `test_avg/mae_surf_p` (4-split mean) | NaN | NaN | NaN | — |
| `test_avg/mae_surf_p` (3 valid splits, edward's report) | 127.29 | 111.04 | 128.84 | −12.8% |

Peak GPU σ=1.0: 42.5 GB / 96 GB. n_params σ=1.0: 670,551 (vs baseline 662,359, +1.2%).

### Decision: send back to student (#3262-comment-4461135244)

`test_avg/mae_surf_p = NaN` (formal 4-split mean) is disqualifying per advisor protocol — same `data/scoring.py` non-finite-prediction bug surfaced via #3257. The val win is strong (−9.8% across all splits) and consistent. Sent back with the same `torch.nan_to_num` patch for `train.py:evaluate_split` and instruction to rerun only the σ=1.0 arm (skip σ=4.0 confirmed loser, skip baseline rerun).

### Analysis

- **Hypothesis worked, large effect size.** RFF σ=1.0 reduces `val_avg/mae_surf_p` by 9.8% (within and beyond the 3–8% predicted envelope) with consistent per-split gains. Largest improvements on OOD geometry splits (`val_geom_camber_rc` −11.8%, `test_geom_camber_rc` −15.2%) — suggests RFF helps spatial generalization more than in-distribution fitting, consistent with the spectral-bias literature interpretation.
- **σ=4.0 confirms scale.** Slight regression at σ=4.0 (+2.0% val_avg vs baseline) brackets the useful range as σ ∈ [0.5, 2.0] (Tancik 2020 alias-warning regime).
- **Volume pressure also improves** (1.5–7.4% per split), so RFF benefit is not surface-only.
- **NaN on `test_geom_camber_cruise` is pre-existing**, present in vanilla baseline too — confirms it's the systemic `accumulate_batch` bug, not RFF-induced divergence.
- **First credible baseline measurement on this branch:** edward's paired vanilla `17fia1vd` gives `val_avg/mae_surf_p = 128.34` and 3-split `test_avg/mae_surf_p = 127.29`. Once the σ=1.0 rerun lands with finite 4-split test_avg, this becomes BASELINE.md.
- **R2 follow-up queue (from edward's suggestions, ranked):** (1) finer σ sweep ∈ {0.5, 1.0, 2.0}; (2) bump n_freqs to 32 or 64 (literature norm 128–256); (3) anisotropic σ (different x vs z); (4) stack RFF with arc-length encoding via `saf` features.

## 2026-05-15 16:30 — PR #3258: Gradient clip 1.0 + 5-epoch LR warmup

- **Student/branch:** willowpai2i24h4-fern / `willowpai2i24h4-fern/grad-clip-warmup`
- **Hypothesis:** Add gradient clipping (max_norm=1.0) and 5-epoch linear LR warmup before cosine annealing. Bound catastrophic-batch updates and let slice routing stabilize before peak LR.
- **W&B runs:** baseline `nylo2tvd`, clip1.0-wu5 `69np1sbe`, clip0.5-wu3 `4yg5bhtc`

### Result (best checkpoint, all runs hit 30-min timeout cap at epoch 11–14 / 50)

| Split | Baseline `nylo2tvd` | **clip1.0-wu5 `69np1sbe`** | clip0.5-wu3 `4yg5bhtc` (best ckpt) |
|-------|-----:|-----:|-----:|
| `val_single_in_dist/mae_surf_p`     | 172.20 | **142.37** | 148.51 |
| `val_geom_camber_rc/mae_surf_p`     | 161.19 | **124.69** | 152.15 |
| `val_geom_camber_cruise/mae_surf_p` | 109.57 | **90.45**  | 95.82  |
| `val_re_rand/mae_surf_p`            | 124.80 | **105.42** | 113.35 |
| **`val_avg/mae_surf_p` (best ckpt)**| **141.94** | **115.73 (−18.5%)** | 127.46 (−10.2%) |
| `val_avg/mae_surf_p` (terminal) | 141.94 | — | 117.90 (student reported this value as if best-ckpt — mismatch) |
| `test_avg/mae_surf_p` (4-split mean) | NaN | NaN | NaN |
| `test_avg/mae_surf_p` (3 valid splits, fern's report) | 139.34 | 115.46 (−17.1%) | 118.47 (−15.0%) |

Gradient norm trace from clip1.0-wu5: median 56.0, mean 87.2, p99 445, **max 1110**, clipping binds on 100% of steps.

### Decision: send back to student (#3258-comment-4461464504)

Same NaN-poisoning blocker as #3257 and #3262 — `test_avg/mae_surf_p = NaN` (4-split formal) is disqualifying. Sent back with the same `torch.nan_to_num` patch for `train.py:evaluate_split` and instructed to rerun only the `clip1.0-wu5` arm (skip clip0.5-wu3 which is the smaller win, skip baseline rerun).

### Analysis

- **Hypothesis confirmed, large effect size.** clip1.0-wu5 reduces `val_avg/mae_surf_p` by 18.5% (vs predicted 2–5%), with consistent per-split gains and largest effect on `val_geom_camber_rc` (−22.6%) and `val_re_rand` (−15.5%).
- **Mechanism is real and important.** Pre-clip gradient norms median 56, peak 1100 (50–1000× the clip cap). This is unusual for a 1.5M-param model. Without clipping, a few outlier batches per epoch take steps ~1000× the median, swinging the PhysicsAttention slice-routing softmax into bad basins. Baseline `nylo2tvd` shows characteristic catastrophic-batch behavior: val_avg goes 142 → 192 between epochs 11 and 12 before partial recovery. Clipped runs never regress.
- **clip0.5-wu3 best-ckpt reporting issue.** Fern reported 117.90 for clip0.5-wu3, but actual W&B best checkpoint is 127.46. Fern's table mixed terminal-value vs best-checkpoint reporting across runs. The clip1.0-wu5 winner is correctly reported at 115.73 (best=terminal for that run). The rerun will be measured consistently.
- **Run-to-run variance is large (~13pt on val_avg).** Edward's vanilla baseline `17fia1vd` reported 128.34, fern's vanilla baseline `nylo2tvd` reported 141.94 — same code, same config, ~9–13pt gap. This is purely stochastic; with median grad norm ~56 and peaks >1000, an unclipped baseline naturally lands in different local minima per run. A consequence: **a "win" of <10% may be partial regression-to-the-mean** from a bad-luck baseline draw. The clip1.0-wu5 win at 115.73 beats *either* baseline by 9.8–18.5%, so the gain is real.
- **#3258 and #3262 produce nearly identical val_avg (115.73 vs 115.78).** When both reruns land cleanly, we should test whether grad-clip+warmup and RFF compound or are redundant. Merge fern first (foundational training fix), then re-evaluate edward's RFF on the new baseline.
- **R2 follow-up queue (from fern's diagnostics):** (1) investigate gradient-norm sources (PhysicsAttention softmax temperature 0.5, surf_weight=10); (2) longer warmup (10 epochs) + lower clip (0.5); (3) log post-clip grad_norm and clip ratio.

## 2026-05-15 18:25 — PR #3257 (MERGED): Surface MAE loss + p-weight 3× — rerun with canonical NaN guard

- **Student/branch:** willowpai2i24h4-frieren / `willowpai2i24h4-frieren/surf-mae-p-weight`
- **Hypothesis:** Switch surface loss from MSE to MAE with per-channel weight [1, 1, 3] on (Ux, Uy, p). Volume loss stays MSE. Frieren also independently traced and fixed the cruise-NaN root cause (commit `34600cf`).
- **W&B run:** `szru1ogx` (https://wandb.ai/wandb-applied-ai-team/senpai-v1/runs/szru1ogx)

### Result (best checkpoint at epoch 13/14; 30-min timeout cap)

| Split | val mae_surf_p | test mae_surf_p | n_skipped_y_samples |
|-------|---------------:|----------------:|--------------------:|
| `single_in_dist`       | 124.31 | 122.34 | 0 |
| `geom_camber_rc`       | 131.21 | 106.31 | 0 |
| `geom_camber_cruise`   |  78.23 |  62.47 | 1 |
| `re_rand`              |  92.92 |  86.28 | 0 |
| **avg (4-split)**      | **106.67** | **94.35** | — |

Beats prior baseline `xfayvdk2` (val_avg=117.89, test_avg=106.23) by **−9.5% val / −11.2% test**.

### Decision: MERGED as R1 winner #1 (commit `a059a65`)

Per CLAUDE.md, this beats the baseline on the primary ranking metric (`test_avg/mae_surf_p`) with a finite 4-split mean and a terminal SENPAI-RESULT marker. New BASELINE.md anchor.

### Analysis

- **The cruise split swung the most.** Test cruise = 62.47 (vs prior 78.72, −20.6%). This is the split where pressure dynamic range is largest relative to other test splits, so the p-weight=3 had the biggest leverage exactly where MAE-vs-MSE matters most.
- **Per-split shape mirrors the hypothesis.** Frieren predicted cruise/re_rand would lead the gain on val, and they did on test too (cruise leads at −20.6%, re_rand at −18.6% vs prior baseline). Single_in_dist gained least (−3.4%), consistent with MAE-on-p being most valuable where the p distribution is heavy-tailed.
- **Canonical NaN guard works as designed.** `n_skipped_y_samples=1` on cruise, 0 everywhere else — confirms exactly one bad-GT sample (`000020.pt`) is skipped and the masking is precise.
- **Cosine T_max=50 mismatch is still unaddressed for this run.** Frieren's run uses the same cosine schedule the rest of R1 used (T_max=50 nominal, ~14 epochs actual). Implicit under-annealing across the board — alphonse's #3358 will address.
- **Run-to-run variance still applies.** Frieren's win margin vs the alphonse baseline (also NaN-guarded) is ~11pt val / ~12pt test. Edward's and fern's unclipped baselines were 128/142 on val — frieren's improvement is robust against any of these reference points.

### Two PRs sent back due to base change (rebase required)

- **#3258 (fern, grad-clip+warmup)** — already reran with the corrected NaN guard, returned `val_avg=117.31, test_avg=105.70` on the OLD MSE base. Beats the old baseline by 0.5% on test_avg but regresses against the new merged baseline by 12%. Sent back for rebase onto frieren's loss + rerun. Mechanism (clipping median-56 gradients with peak >1000) is orthogonal to loss reformulation, so should compose.
- **#3262 (edward, RFF σ=1.0)** — never got to corrected-patch rerun before frieren merged. Posted note: rebase onto new base + apply NaN guard + rerun RFF σ=1.0 (skip σ=4.0 which already lost).

## 2026-05-15 19:23 — PR #3263: FiLM log(Re) conditioning (send-back, needs rebase)

- **Student/branch:** willowpai2i24h4-thorfinn / `willowpai2i24h4-thorfinn/film-re-cond`
- **Hypothesis:** Inject a FiLM (Feature-wise Linear Modulation) module conditioned on log(Re) after the Transolver preprocess layer, before the attention blocks. FiLM adds a learned affine gate `(γ, β) = MLP(log(Re))` applied to the preprocess hidden state, giving the model a direct low-rank route to modulate all 128 channels by global Re.
- **W&B runs:** `zjogv9vn` (film-re-v1), `rlildyv4` (film-re-v2), `joszk2jg` (film-re-v3 best), `vsuqhyt5` (fresh baseline)

### Result (vs own fresh baseline `vsuqhyt5`, all runs hit 30-min timeout cap at epoch 14)

| Split | Baseline `vsuqhyt5` | FiLM v3 `joszk2jg` | Δ (rel) |
|-------|--------------------:|-------------------:|--------:|
| `val_single_in_dist`      | 161.82 | 142.03 | **−12.2%** |
| `val_geom_camber_rc`      | 137.18 | 125.03 | **−8.9%** |
| `val_geom_camber_cruise`  | 121.72 |  95.99 | **−21.1%** |
| `val_re_rand`             | 129.41 | 111.15 | **−14.1%** |
| **`val_avg/mae_surf_p`**  | **137.53** | **118.55 (−13.8%)** | — |
| `test_single_in_dist`     | 137.69 | 126.24 | −8.3% |
| `test_geom_camber_rc`     | 126.39 | 120.84 | −4.4% |
| `test_re_rand`            | 127.87 | 110.98 | −13.2% |
| `test_geom_camber_cruise` | NaN    | NaN    | pre-existing (run pre-fix) |
| **3-split test mean**     | 130.65 | 119.35 | **−8.6%** |

Seed spread across 3 FiLM runs: 133.34 / 127.89 / 118.55 (best). 3-seed mean: 126.59. All seeds beat baseline on val_avg.

### Decision: sent back for rebase + rerun on frieren-base

Thorfinn's fresh baseline was `vsuqhyt5` (val=137.53) — a high-variance unclipped run. While thorfinn was running, frieren's #3257 merged (val=106.67, test=94.35). FiLM v3's val=118.55 does NOT beat the new merged baseline (106.67). Mechanism is orthogonal (Re conditioning vs. loss reformulation), so should compose. Sent back to rebase and rerun v3 on the new base.

### Analysis

- **Directional signal is strong.** −21.1% on val_cruise (the split with widest Re range 122K–5M) and −14.1% on val_re_rand exactly matches the prediction that FiLM leverages Re most where Re varies most. The mechanism is working as intended.
- **Seed variance is large (133→127→118).** This is the unclipped baseline variance. The rebased rerun should reduce variance if clip+warmup lands (orthogonal merge candidate via #3258).
- **The test_3split=119.35 vs merged baseline test_4split=94.35 gap is 27%** — but these are against different loss configurations. The rebased run on frieren's loss may close most of this gap.
- **NaN guard was NOT applied** (run pre-merge). The rebased run will automatically inherit the canonical guard from the merged head — test_avg will be 4-split finite.

### R2 follow-ups (from thorfinn's suggestions, saved)

1. Per-block FiLM heads (standard recipe in conditional generation) — natural R2 extension after rebase win
2. Richer conditioning vector `(log_Re, AoA_1, AoA_2, gap, stagger)` — all global scalars, same mechanism
3. ~~NaN fix~~ — RESOLVED in #3257 merge

## 2026-05-15 19:25 — PR #3256 (CLOSED): Huber loss delta=0.5 (redundant with #3257 merge)

- **Student/branch:** willowpai2i24h4-tanjiro / `willowpai2i24h4-tanjiro/huber-loss`
- **Hypothesis:** Replace MSE with Huber loss (δ=0.5) for outlier-robust loss metric alignment.
- **Outcome:** No results produced. Closed as redundant — frieren's #3257 empirically validated the L1-style robustness direction, and pure MAE+p-weight=3 is a superset of the Huber approach. Pod was blocked by GitHub API rate-limit cycling for 6+ hours with no commits pushed.
- **Reassignment:** #3406 surf_weight sweep {5,10,20} on merged baseline.

## 2026-05-15 20:05 — PR #3260 (CLOSED): Surface-biased slice routing in PhysicsAttention

- **Student/branch:** willowpai2i24h4-nezuko / `willowpai2i24h4-nezuko/surf-biased-slice`
- **Hypothesis:** Concat `is_surface` (per-node bool) into the slice assignment projection, threading through `Transolver → TransolverBlock → PhysicsAttention`. The routing should learn to specialize slices for surface vs volume regimes.
- **W&B runs:** `j9cz0wkn` (baseline-control), `kmmcoz9f` (surf-bias-v1), `a9tcw7t2` (seed-A), `hg44hunu` (seed-C best), `q8bmjltf` (baseline-v2)

### Result (vs paired baseline `j9cz0wkn`, run pre-merge against OLD MSE loss)

| | baseline-control `j9cz0wkn` | surf-bias-v1 `kmmcoz9f` | Δ |
|---|---:|---:|---:|
| best_val_avg/mae_surf_p | 131.42 | 131.36 | −0.05% (essentially unchanged) |
| best_epoch | 12 | 13 | — |

3-seed mean of surf-bias = 129.20 vs baseline 2-seed mean = 140.12. But baseline-vs-baseline range is 131.4–148.8 = 13%, so seed variance dominates the observed effect.

| split (test) | baseline | surf-bias-v1 | Δ |
|---|---:|---:|---:|
| `test_single_in_dist` | 133.92 | 136.95 | +2.3% (worse) |
| `test_geom_camber_rc` | 128.55 | 130.79 | +1.7% (worse) |
| `test_geom_camber_cruise` | NaN | NaN | (pre-existing bug, run pre-merge) |
| `test_re_rand` | 123.52 | 114.75 | **−7.1% (better)** |
| **3-split mean (excl cruise)** | 128.66 | 127.50 | −0.9% |

### Decision: CLOSED

Paired comparison shows −0.05% on the primary metric — effectively no change. The 3-seed mean appears favorable but is dominated by baseline seed variance. Per-split test eval is mixed: only `test_re_rand` clearly improves (−7.1%), while in-distribution and geom-camber are slightly worse. Best seed val=126.08 doesn't approach the new merged baseline val=106.67 either. Compared against merged #3257 baseline (test=94.35), nezuko's result is ~+33% above target on 3-split test mean.

The mechanism (binary surface flag at slice routing layer) doesn't add specialization capacity — it just informs existing slices. R2 path with more potential: dedicated parameters per regime (multi-scale slice tokens).

### NaN diagnosis note

Nezuko traced cruise-NaN to non-finite **predictions** — same incorrect diagnosis I initially had. Actual root cause (per askeladd, frieren) is `+inf` in **ground truth** at `test_geom_camber_cruise_gt/000020.pt`. Nezuko's run was pre-merge so the canonical fix wasn't inherited.

### Reassignment

Reassigned nezuko to **#3429 Multi-scale slice tokens** (parallel coarse-global + fine-surface slice groups). Different architectural lever — adds dedicated slice capacity per regime instead of informing the routing.

## 2026-05-15 20:35 — PR #3351: EMA weights β=0.999 (old-base result)

- **Student/branch:** willowpai2i24h4-askeladd / `willowpai2i24h4-askeladd/ema-weights-beta-0.999`
- **Hypothesis:** Apply Polyak/EMA averaging on model weights with β=0.999; evaluate with EMA-swapped weights to recover 2–4% from averaging late-training oscillations.
- **W&B run:** `4v7imfa8`
- **Ran against:** OLD pre-#3257 baseline (MSE loss), NOT the merged frieren base.

### Result (best checkpoint at epoch 14/14, 30-min cap)

| Metric | EMA-β0.999 run | Reference (old MSE base) |
|---|---:|---:|
| `val_avg/mae_surf_p` | **131.37** | edward 128.34 / fern 141.94 (variance band) |
| `test_avg/mae_surf_p` (4-split, finite) | **118.65** | — (NaN guard helped; askeladd was diagnoser) |
| `ema/avg_diff_norm` | 0.30 | EMA materially diverged from live weights ✓ |
| `n_skipped_y_samples` (cruise) | 1 | canonical NaN guard inherited ✓ |

### Decision: send back for rebase + retry with β=0.99

Mechanism (EMA shadow weights, `update_ema`, `eval_with_ema_swap`) is correctly implemented — `ema/avg_diff_norm=0.30` confirms divergence from live. But result is on OLD MSE base, +25.7% above new merged target 94.35. Within seed variance of old-base anchors (128.34 / 141.94), so β=0.999 didn't deliver on the old base either.

### Askeladd's own analysis (correct)

β=0.999 effective horizon ≈ 1000 steps ≈ 2.7 epochs of 14-epoch training — too long, averages over high-LR pre-converged weights. β=0.99 effective horizon ≈ 100 steps ≈ 0.3 epoch — averages over only recent near-converged low-LR weights. This is the right diagnosis.

### Send-back instructions

Rebase onto new advisor head (merged frieren surf-MAE + p-weight=3 + canonical NaN guard). Conflicts in `train.py` loss + eval regions — keep frieren's content. EMA logic is separate. Re-run with `--ema_beta 0.99`; skip 0.999 and 0.9995. Predicted val ~103–106, test ~90–94 (−1–4% gain typical for EMA on well-tuned baselines).

## 2026-05-15 21:55 — PR #3263: FiLM(log_Re) conditioning on Transolver hidden state — **R1 WINNER #2, MERGED**

- **Student/branch:** willowpai2i24h4-thorfinn / `willowpai2i24h4-thorfinn/film-re-cond` (rebased)
- **Hypothesis:** Add a FiLM (Feature-wise Linear Modulation) gate after the preprocess MLP, conditioned on log(Re). Zero-init the FiLM head so training starts as identity. Gives the model an explicit, low-rank affine route from the global Re scalar into every channel of the trunk's hidden state.
- **W&B run (rebased on frieren base):** `69jp9tvt`
- **First-run W&B runs (old MSE base, pre-rebase):** `joszk2jg` (v3), `rlildyv4` (v2), `zjogv9vn` (v1), `vsuqhyt5` (baseline)

### Result (best checkpoint at epoch 14/50, 31.8 min wall-clock)

| Metric | Frieren base | FiLM-on-frieren | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 106.67 | **100.24** | **−6.03%** |
| `test_avg/mae_surf_p` (4-split, finite) | 94.35 | **90.06** | **−4.55%** |

### Per-split test (all finite, n_skipped_y_samples=1 on cruise as expected)

| Split | Frieren base | FiLM-on-frieren | Δ |
|---|---:|---:|---:|
| `test_single_in_dist` | 122.34 | 119.11 | −2.6% |
| `test_geom_camber_rc` | 106.31 | 100.27 | −5.7% |
| `test_geom_camber_cruise` | 62.47 | **58.62** | **−6.2%** |
| `test_re_rand` | 86.28 | 82.27 | −4.6% |
| **`test_avg`** | **94.35** | **90.06** | **−4.55%** |

### Decision: MERGED (R1 winner #2)

Clean merge winner. All 4 test splits improve. Mechanism is mechanically orthogonal to frieren's loss reformulation — FiLM adds a structural affine route from log_Re into the hidden state, while the MAE+p_weight=3 loss reweights per-channel gradient contributions. Both target Re-dependent pressure physics through different mechanisms; they compose with diminishing returns (old base FiLM gain was −13.8% val; new base FiLM gain is −6.0% val — confirming partial overlap, but the structural route still adds robust value beyond loss reweighting).

### Per-split shape carries

Hypothesis predicted "biggest impact on splits with widest Re spread (cruise, re_rand)". Confirmed: cruise (−6.2%) is largest absolute reduction on the new base, re_rand (−4.6%) and rc (−5.7%) also gain meaningfully. Single (−2.6%) gains the least, consistent with its narrower test distribution.

### Thorfinn's own bug-diagnosis credit

In his first comment (before frieren's #3257 merged), thorfinn independently traced the cruise-NaN to `y` having `+inf` values and proposed `torch.nan_to_num` on y before the masked sum. Same root cause askeladd and frieren independently identified — three students converged on the diagnosis simultaneously. The canonical fix is frieren's; thorfinn inherited it via rebase.

### Suggested follow-ups (rerouted to R2)

1. Per-block FiLM heads (each Transolver block gets own FiLM head) — natural next step ← **next assignment for thorfinn**
2. Richer conditioning vector `(log_Re, AoA_1, AoA_2, gap, stagger)` — composes with #1
3. Combine FiLM + Re-stratified augmentation — extra Re coverage during training
4. Variance reduction via fixed `--seed` flag — separate instrumentation patch

## 2026-05-15 21:55 — PR #3358: Cosine LR T_max=14 (old-base result, sent back for rebase)

- **Student/branch:** willowpai2i24h4-alphonse / `willowpai2i24h4-alphonse/cosine-tmax-fix`
- **Hypothesis:** Cosine schedule `T_max=50` mismatches actual ~14-epoch wall-clock cap. LR stays at 82% of peak when training ends. Setting `T_max=14` gives proper annealing tail.
- **W&B runs (old MSE base):** `8d2gpkjn` (tmax14 primary), `9i62w0t4` (tmax50 paired baseline), `qrzzn3mw` (tmax10 aggressive)

### Result (paired comparison on old base)

| Arm | val_avg | test_avg | W&B |
|---|---:|---:|---|
| `cosine-tmax14` | **117.67** | **104.95** | `8d2gpkjn` |
| `cosine-tmax50-baseline` | 125.67 | 115.62 | `9i62w0t4` |
| `cosine-tmax10` | 129.79 | 117.45 | `qrzzn3mw` |

**Paired Δ:** val −6.37%, test −9.22% (clean mechanism win on old base).

### LR trace confirms mechanism

| Epoch | tmax14 LR | tmax50 LR |
|---|---:|---:|
| 7 | 2.50e-04 | 4.76e-04 |
| 13 | 6.27e-06 | 4.21e-04 (82% of peak!) |
| 14 | 0.0 | 4.09e-04 |

tmax10 fails because `CosineAnnealingLR` continues into the next half-period after T_max, so LR rebounds (0→1.03e-04→1.73e-04 in last 4 epochs) — useful negative result.

### Decision: send back for rebase + retry on new merged base

Result test=104.95 is on OLD MSE base, +16.5% above new target 90.06. The schedule fix is mechanically orthogonal to both frieren's loss and thorfinn's FiLM (both are merged on the new base). Sending back with rebase + rerun-primary-arm-only instructions. Predicted val ~92–99, test ~81–89 on new base.
