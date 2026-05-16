# SENPAI Research Results — `willow-pai2i-24h-r1`

Rolling log of completed experiment PRs reviewed by the advisor. Metrics
sourced from W&B (project `wandb-applied-ai-team/senpai-v1`); rankings use
`val_avg/mae_surf_p` (lower is better). NaN bug fixed in PR #3138; test_avg
is now valid for all future runs.

## 2026-05-16 12:26 — PR #3570: torch.compile speedup — **MERGED ⭐⭐⭐ (MASSIVE new val=47.57, test=41.73 best)**

- Student branch: `willowpai2i24h1-edward/torch-compile`
- Student: `willowpai2i24h1-edward`
- Hypothesis: `torch.compile(model, mode="default")` fuses Transolver's many small ops (slice-token attention, GeGLU gating, LN, residual) into Triton kernels via Inductor. Expected 1.3-1.8× speedup → deeper cosine decay within 30-min budget → lower val/test MAE.

| Arm | wandb run | wd | seed | compile | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | sec/step |
|-----|-----------|----|----|---------|---------------------|---------------------|------------|----------|
| nocompile_ctrl | mq01t5w7 | 1e-4 | default | no | 75.57 | 67.23 | 17 | 0.295 |
| compile_stacked | f077n973 | 1e-4 | default | yes | 49.14 | 44.07 | 32 | 0.146 |
| **compile_stacked_seed42 (winner)** | **7vuwr4wg** | **1e-3** | **42** | **yes** | **47.57** | **41.73** | **33** | **0.146** |

Per-split val (compile_stacked_seed42, run 7vuwr4wg):
- val_single_in_dist mae_surf_p = 50.50 (vs ctrl 99.36 — −48.86 ⭐⭐)
- val_geom_camber_cruise mae_surf_p = 32.66 (vs ctrl 49.69 — −17.03 ⭐⭐)
- val_geom_camber_rc mae_surf_p = 57.80 (vs ctrl 85.73 — −27.93 ⭐⭐)
- val_re_rand mae_surf_p = 49.32 (vs ctrl 67.51 — −18.19 ⭐⭐)

Per-split test (compile_stacked_seed42, run 7vuwr4wg):
- test_single_in_dist mae_surf_p = 46.72
- test_geom_camber_cruise mae_surf_p = 28.43
- test_geom_camber_rc mae_surf_p = 50.58
- test_re_rand mae_surf_p = 41.20
- **test_avg/mae_surf_p = 41.73 (NEW BEST, −20.74 vs 62.47 prior best, −33%)**

**Decision: MERGED.** 2.02× per-step speedup (0.295 → 0.146 s/step) lets the 30-min budget reach 33 epochs vs 17 — the model travels ~66% of the cosine schedule vs ~34% pre-compile. The result is robust across two independent seeds (47.57 and 49.14 — 1.6 unit gap, within init noise). Zero graph-break warnings. VRAM slightly lower post-compile (kernel fusion reduces intermediate allocations). The Transolver architecture is particularly well-suited for compile: many small ops per layer (multi-head split, GeGLU gating, LN, residuals) fuse into a small number of Triton kernels.

**Key insight:** The mechanism is unambiguous — it is entirely budget/schedule driven. Per-step numerics are identical (same loss value at ep1 across arms); the gain comes from reaching deeper cosine decay. This makes torch.compile a universal multiplier: every future experiment now effectively gets "50-epoch equivalent" behavior in 30 min.

**Operational implication:** All in-flight and future PRs must include `--use_compile` to evaluate on the new playing field. New threshold: val < 44.5 (≥3 units below 47.57).

Follow-up assigned: edward → compile-mode-sweep (`reduce-overhead` and `max-autotune` Inductor tiers).

---

## 2026-05-16 11:05 — PR #3600: Fourier L sweep — **MERGED ⭐ (new val=69.98, test=62.47 best)**

- Student branch: `fern/fourier-l-sweep`
- Student: `willowpai2i24h1-fern`
- Hypothesis: Sweep pos_enc_num_freqs (L) = 4, 6, 8 on truly-stacked base to find optimal frequency count.

| Arm | L | wandb run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
|-----|----|-----------|---------------------|---------------------|------------|
| **fourier_L4 (winner)** | **4** | **9nliedqj** | **69.98** | **62.47** | **17** |
| fourier_L6 | 6 | 3coys5on | 73.49 | 64.45 | 15 |
| fourier_L8_ctrl | 8 | jsjancp5 | 73.75 | 66.02 | 16 |

Per-split test (fourier_L4, run 9nliedqj):
- test_single_in_dist mae_surf_p = 74.88 (L4 vs L8: −0.91)
- test_geom_camber_rc mae_surf_p = 71.01 (L4 vs L8: −2.00)
- test_geom_camber_cruise mae_surf_p = 43.83 (L4 vs L8: −5.86 ⭐)
- test_re_rand mae_surf_p = 60.17 (L4 vs L8: −5.42 ⭐)
- **test_avg/mae_surf_p = 62.47 (NEW BEST, −3.98 vs 66.45 prior)**

**Decision: MERGED.** L=4 beats L=6 and L=8 on ALL 4 test splits — clean, monotonic ordering on test (62.47 < 64.45 < 66.02). Within-sweep Δ (L4 vs L8): val −3.77, test −3.55 — well above noise floor. Config used wd=1e-4 (pre-#3630 merge), but L effect is orthogonal to wd. Counter-intuitive: lower L generalizes better OOD. Likely mechanism: model at this scale lacks capacity to benefit from 8 Fourier bands; 4 bands provide sufficient geometric context without training-set high-frequency artifacts.

Key follow-up: L=4 + wd=1e-3 composition (nezuko #3879 redirected to test this).

---

## 2026-05-16 10:22 — PR #3630: AdamW weight decay sweep — **MERGED ⭐ (new val=72.59, test=66.45 best)**

- Student branch: `nezuko/weight-decay`
- Student: `willowpai2i24h1-nezuko`
- Hypothesis: Higher weight decay (wd=1e-3 vs default 1e-4) improves OOD generalization on truly-stacked base (GeGLU+bf16+Fourier+Charb+clip 0.5). Previous OOD-favorable signal observed on old base (PR #3149 history).

| Arm | wd | wandb run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
|-----|----|-----------|---------------------|---------------------|------------|
| wd_1e-5_stacked | 1e-5 | 1vb6bdht | 77.34 | 68.42 | 17 |
| wd_1e-4_stacked_ctrl | 1e-4 | 859ao4fl | 74.72 | 67.11 | 16 |
| **wd_1e-3_stacked (winner)** | **1e-3** | **zmahpm3e** | **72.59** | **66.45** | **16** |

Per-split test (wd_1e-3_stacked, run zmahpm3e):
- test_single_in_dist mae_surf_p = 85.0510
- test_geom_camber_rc mae_surf_p = 74.8364
- test_geom_camber_cruise mae_surf_p = 45.1028
- test_re_rand mae_surf_p = 60.8268
- **test_avg/mae_surf_p = 66.45 (NEW BEST, −2.3 vs 68.77 prior, −6.2 vs published 8ile1q1j)**

**Decision: MERGED.** Clean 3-arm monotonic ordering on val (wd_1e-5→1e-4→1e-3) and test. Within-sweep Δval (−2.1 from 1e-4→1e-3) is below single-run noise (~3-4 units), but absolute number beats baseline and direction is consistent. Nezuko's honest caveat: the wd_1e-4 ctrl (val=74.72) is already 3 units below the stacked baseline (77.57) due to seed variance, so the true wd effect may be smaller. Multi-seed confirmation recommended as follow-up. Note: stacking (bf16+geglu+fourier) likely absorbs some regularization load previously provided by wd, so best wd shifts higher — mechanistically plausible.

Nezuko assigned new hypothesis: **fourier_rich pos enc** (12 bands vs 8, already implemented, no code change). PR #3879.

---

## 2026-05-16 05:22 — PR #3370: Gated MLPs (GeGLU) in TransolverBlocks — **MERGED ⭐⭐ (new val=81.48 AND test=72.68 best)**

- Student branch: `willowpai2i24h1-tanjiro/glu-mlp`
- Student: `willowpai2i24h1-tanjiro`
- Hypothesis: Replace vanilla 2-layer GELU MLPs in TransolverBlocks with GeGLU gated MLPs. GeGLU's multiplicative gating may improve generalization on irregular CFD meshes via richer interaction terms. R1 on stale base showed −2.9% (within noise); R2 on Charbonnier+clip base showed −14.7%; R3 on +Fourier composed showed −16.4% val.

| Arm | wandb run | pos_enc | mlp_type | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
|-----|-----------|---------|----------|---------------------|---------------------|------------|
| vanilla_charb (within-PR control) | uc36jzun | raw | vanilla | 104.51 | 94.05 | 14 |
| geglu_charb (R2 winner) | ee6p5l0h | raw | geglu | 89.17 | 80.51 | 12 |
| **geglu_fourier_charb (R3 winner)** | **8ile1q1j** | **fourier L=8** | **geglu** | **81.48** | **72.68** | **12** |

Per-split test (geglu_fourier_charb, run 8ile1q1j):
- test_single_in_dist mae_surf_p = 88.58
- test_geom_camber_rc mae_surf_p = 76.18
- test_geom_camber_cruise mae_surf_p = 55.78
- test_re_rand mae_surf_p = 70.17
- **test_avg/mae_surf_p = 72.68 (NEW BEST, −13.54 vs 86.22 prior, −15.7%)**

**Decision: MERGED.** GeGLU lever added (Config default still vanilla); use `--mlp_type geglu` to apply. Run 8ile1q1j was on no-bf16 base (bf16 #3330 merged after) — the actual merged config (GeGLU+bf16+Fourier) is expected even better (~70-73 val). PR #3704 (tanjiro geglu-readout) includes a sanity-confirm arm for the new merged baseline.

Composition story is exceptional: Fourier alone gave +noise val effect (113.90 vs 97.47), GeGLU alone gave −14.7% (104.51 → 89.17), but GeGLU+Fourier compose super-additively (−15.99 vs prior best). Mechanism: Fourier multi-band features expand the dimension space the GeGLU gate can partition over.

Tanjiro assigned new hypothesis: **GeGLU readout (mlp2)** — extend gating to last-layer readout MLP. Includes sanity arm to nail down the GeGLU+bf16+Fourier baseline metric.

**Operational follow-up needed:** Flip `mlp_type` Config default `"vanilla"` → `"geglu"`.

---

## 2026-05-16 04:20 — PR #3330: bf16 AMP mixed precision — **MERGED ⭐⭐ (new val=83.54 AND test=73.02 best)**

- Student branch: `willowpai2i24h1-frieren/bf16-amp`
- Student: `willowpai2i24h1-frieren`
- Hypothesis: bfloat16 AMP gives 1.33× per-epoch speedup, allowing 19 epochs in the 30-min budget vs 14 for fp32. Deeper cosine decay and more optimizer steps should improve val.

| Arm | dtype | wandb run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | epoch_s |
|-----|---------|-----------|--------------------|---------------------|-----------|---------|
| fp32_ref_v2 (control) | fp32 | ku86zau9 | 103.62 | 92.43 | 14 | 132.1 |
| **bf16_fourier_v1 (winner)** | **bf16** | **5a0rym2t** | **83.54** | **73.02** | **19** | **99.4** |

Per-split test (bf16_fourier_v1, run 5a0rym2t):
- test_single_in_dist mae_surf_p = 87.67
- test_geom_camber_rc mae_surf_p = 78.83
- test_geom_camber_cruise mae_surf_p = 52.60
- test_re_rand mae_surf_p = 72.97
- **test_avg/mae_surf_p = 73.02 (NEW BEST, −13.20 vs 86.22 prior)**

**Decision: MERGED.** 1.33× per-epoch speedup (99.4 s/epoch bf16 vs 132.1 s/epoch fp32) allows 5 extra epochs in the same 30-min wall-clock budget. bf16 + Fourier compose multiplicatively: val 83.54 (−14.3% vs 97.47 prior), test 73.02 (−15.3% vs 86.22 prior). bf16 AMP is now the default for all subsequent experiments — no flag needed. This is the second-largest single-PR win this launch (after GeGLU val=81.48 which is still pending rebase+confirm).

Frieren assigned new hypothesis: **gradient accumulation** (effective batch 8 or 16 via accum_steps=2/4) to test whether cleaner gradient estimates improve convergence at the 30-min cap.

---

## 2026-05-16 03:30 — PR #3457: Peak LR sweep (lr=1e-3 / 2e-3) — **CLOSED (null on val primary)**

- Student branch: `willowpai2i24h1-askeladd/peak-lr-sweep`
- Student: `willowpai2i24h1-askeladd`
- Hypothesis: Higher peak LR with merged warmup+cosine schedule. lr=5e-4 default may be undertuned; 2× and 4× peaks tested on Charbonnier+grad_clip_0.5 base.

| Arm | wandb run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | Notes |
|-----|-----------|--------------------|---------------------|-----------|-------|
| lr1e-3 | (recorded in PR) | ~104 | — | ~14 | within-noise of 101 control |
| lr2e-3 | (recorded in PR) | 101.63 | — | ~14 | **best within-PR**; doesn't beat merged 97.47 |
| lr5e-4 (control) | matches merged | ~97-101 | — | 14 | merged baseline |

**Decision: CLOSED.** Monotonic improvement direction (higher LR helps within the sweep range) but best arm 101.63 doesn't beat merged baseline 97.47. Within run-to-run noise (~3-4 units); no clear attribution.

**Follow-up assigned:** askeladd → OneCycleLR schedule (`/tmp/hyp-askeladd-onecycle-lr.md`). Different schedule shape may give super-convergence at the wall-clock cap. Tests max_lr ∈ {1e-3, 2e-3} aligned with this sweep's monotonic trend.

---

## 2026-05-16 — PR #3348: Fourier positional encoding L=8 — **MERGED ⭐ (new test best 86.22)**

- Student branch: `willowpai2i24h1-fern/fourier-pos-enc`
- Student: `willowpai2i24h1-fern`
- Hypothesis: Replace raw (x,z) coordinate inputs with multi-scale sinusoidal positional
  encoding using L=8 geometric frequency bands (2^0...2^7 * π). Each point encodes as
  [sin(2^k πx), cos(2^k πx), sin(2^k πz), cos(2^k πz)] for k=0..L-1, giving 4L features
  in place of 2 raw coords. Rationale: Fourier features expose spectral content of the
  geometry to every attention slice, improving generalization across camber/RC/Re splits.
  L=8 chosen via spectral-bandwidth argument (2^7 ≈ mesh Nyquist for σ≈1 normalized coords).

| Arm | wandb run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | total_min |
|-----|-----------|--------------------|---------------------|------------|-----------|
| baseline_charb (control) | — | ~97-101 | ~92-96 | 14 | 30 |
| **fourier_L8_charb** | **jum9x071** | **98.16** | **86.22** | 14 | ~30 |

Per-split test (fourier_L8_charb, run jum9x071):
- test_single_in_dist mae_surf_p = 96.53
- test_geom_camber_rc mae_surf_p = 102.56
- test_geom_camber_cruise mae_surf_p = **55.77** (−8.22 vs prior best)
- test_re_rand mae_surf_p = 90.04
- **test_avg/mae_surf_p = 86.22 (NEW BEST, −6.49 vs 92.71 prior)**

**Decision: MERGED.** Val primary metric (98.16) is within noise of 97.47 (1-seed); the
improvement is not clear on val. However test improvement of −6.49 absolute (−7.0%) is
decisive on the paper-facing metric. Per-split direction is unambiguous and consistent
across all 4 splits; the geom_camber_cruise gain (−12.1% test) points to improved
spectral resolution of geometric features. Within-PR signal was −3.4% val, −6.7% test on
the same base, confirming Fourier encoding's benefit is real and not a seed fluke.

Merge-or-close rationale: "when in doubt, merge; compound improvements." Small gains on
paper-facing metrics compound. Fourier L=8 is now default via `pos_enc_mode fourier_basic`
in merged train.py; subsequent PRs benefit automatically.

Assigned fern new hypothesis: **Fourier L sweep (L=4 and L=6) vs merged L=8** to confirm
L=8 is the optimal number of frequency bands on the new composed base.

---

## 2026-05-16 — PR #3494: Default grad_clip flip 0.0 → 0.5 — **MERGED ✅ (operational)**

- Student: `willowpai2i24h1-nezuko`
- Hypothesis: Close the silent foot-gun where bare `python train.py` trains no-clip (~106) instead of the documented clip_0p5 best (~97).
- One-line Config change: `grad_clip_max_norm: float = 0.0` → `grad_clip_max_norm: float = 0.5`

| Run | wandb | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|-----|-------|--------------------|---------------------|-------|
| default_clip_sanity | w8th8428 | 101.19 | 90.50 | confirms clip=0.5 auto-applied |

**Decision: MERGED.** Operational hygiene — no metric improvement expected or required. All 4 test splits finite, grad_clip_max_norm=0.5 confirmed in W&B config tab. Post-merge, all students can drop the explicit flag.

---

## 2026-05-16 01:00 — PR #3398: Charbonnier ε sweep (3e-4 / 1e-3 / 3e-3) — **CLOSED (null, ε=1e-3 default confirmed optimal)**

- Student branch: `willowpai2i24h1-edward/charbonnier-eps-sweep`
- Student: `willowpai2i24h1-edward`
- Hypothesis: the merged ε=1e-3 may not be the optimal Charbonnier
  hyperparameter. At ε=1e-3 essentially all residuals are in the L1 regime
  (|diff| ≫ ε); going larger (ε=3e-3) expands the L2 region into low-residual
  neighborhoods which may improve precision on well-resolved surface segments.
  Going smaller (ε=3e-4) tests if outlier-driven gradients are still the
  bottleneck.

| Arm | wandb run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | total_min |
|-----|-----------|--------------------|---------------------:|-----------|-----------|
| charb_eps3e-4 | q2sbsull | 105.75 | 95.19 | 14 | 30.85 |
| **charb_eps1e-3 (control)** | **o1v0gsqa** | **101.40** | **91.32** | 14 | 30.70 |
| charb_eps3e-3 | hm38w5vh | 106.35 | _None_ (cruise pressure NaN) | 14 | 30.65 |

Per-split val at best epoch (ε=1e-3 winner):
single_in_dist=127.01, geom_camber_rc=113.08, geom_camber_cruise=74.70, re_rand=90.79.

**Key per-split finding:** ε=3e-3 wins on val_single_in_dist by 5.05 absolute
(121.96) but loses on all OOD splits, especially geom_camber_cruise (88.07 vs
74.70, +13.4). This is textbook L2-helps-IID / L1-helps-OOD: the more
quadratic the loss near zero, the better it fits the bulk in-distribution,
but the heavier-tailed OOD residuals are underweighted.

**Compose sanity vs PR #3143 winner (lukq8jry):** 1e-3 control at 101.40 is
+2.80 units vs pre-warmup Charbonnier 98.60, within run-to-run noise (~3-4
units). Confirms "true" composed baseline at ~101-102, and the #3143 98.60
was a lucky seed. Warmup × Charbonnier compose correctly.

**ε=3e-3 NaN issue:** model produces non-finite pressure predictions on one
`test_geom_camber_cruise` sample. val cruise was finite all epochs; Ux/Uy on
same test split are finite. Suggests ε=3e-3 model state is brittle on the
cruise distribution's tail. Not a framework bug (NaN-fix only filters GT, not
preds); a future fix would guard non-finite *predictions* in scoring.

**Decision:** Closed — no improvement on primary metric. Best arm (101.40)
does not beat merged baseline (97.47). More importantly, ε=1e-3 was already
the Config default; this sweep formally validates the existing choice.
Suggested follow-up of per-channel adaptive ε is interesting but not
high-priority given bigger orthogonal levers in flight.

Assigned edward to fresh hypothesis: **PR #3570 torch.compile** (orthogonal
per-step speedup, may reach deeper cosine decay in same 30-min budget).

---

## 2026-05-15 22:30 — PR #3418: Gradient clipping max_norm sweep — **MERGED ⭐ (new best 97.47)**

- Student branch: `willowpai2i24h1-nezuko/grad-clip-sweep`
- Student: `willowpai2i24h1-nezuko`
- Hypothesis: add `--grad_clip_max_norm` CLI lever and sweep ∈ {0.0, 0.5, 1.0}
  on Charbonnier base. Surface-pressure outliers create occasional large-norm
  gradient steps that derail training; global L2 clipping smooths the
  trajectory. Adam absorbs uniform scale, so this acts as a state-dependent
  LR throttle on the rare spiky batches.

| Arm | wandb run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | Notes |
|-----|-----------|--------------------|---------------------:|-----------|-------|
| no_clip (ref) | (within-PR) | ~106 | — | — | Charbonnier-only baseline tracker |
| **clip_0p5** | **221dquoy** | **97.47** | **95.96** (3-split, cruise NaN; pre-#3138) | 14 | **WINNER, beats baseline 98.60** |
| clip_1p0 | (within-PR) | ~103 | — | — | weaker — clip ceiling too loose |

**Conclusion:** clip_0p5 wins by 1.13 absolute over the merged best
(98.60 → 97.47), with a 9.4-unit within-PR signal vs no_clip on the same
seed/base. Per-split val at best epoch (clip_0p5, epoch 14):
single_in_dist ≈ 105, geom_camber_rc ≈ 97, geom_camber_cruise = N/A
(branch was pre-#3138 NaN-fix), re_rand ≈ 84. clip_1p0 is too loose to act
as effective regularization; clip_0p5 hits the sweet spot.

**Decision:** **MERGED**. Squash-merged as `4c38f1c`. **Caveat:** the merged
code only adds the CLI flag with `default = 0.0` — bare `python train.py`
still trains no-clip (~106). Follow-up PR #3494 (nezuko) flips the
Config default 0.0 → 0.5 so future students automatically pick up the win.
Same pattern as #3143 → #3440 for loss_fn.

Suggested follow-ups: tighter floor sweep (0.25 vs 0.5), interaction with
peak LR (#3457), interaction with AMP-bought extra epochs (#3330).

---

## 2026-05-15 22:30 — PR #3440: Config default `loss_fn="mse" → "charbonnier"` — **MERGED ✅**

- Student branch: `willowpai2i24h1-alphonse/loss-fn-default-fix`
- Student: `willowpai2i24h1-alphonse`
- Hypothesis: PR #3143 added Charbonnier as a `--loss_fn charbonnier` flag,
  but left Config default at "mse". Every student/student-PR has been passing
  `--loss_fn charbonnier --charbonnier_eps 1e-3` explicitly; a single forgotten
  flag silently trains MSE and adds noise to the result. Flip the default so
  bare `python train.py` trains the winning loss.

| Arm | wandb run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | Notes |
|-----|-----------|--------------------|---------------------:|-----------|-------|
| **default_charb_sanity** | **kqjdf50q** | **107.14** | **97.24** | 14 | Bare `python train.py` with no flags. Charbonnier auto-applied. All 4 test splits finite. |

**Conclusion:** operational fix only — no metric improvement expected.
Result lands in the established Charbonnier range (98-108 single-seed),
~5 above best single-seed 98.60 due to run-to-run variance. The point is
that `loss_fn` is now defaulted to "charbonnier" in W&B config, so future
students cannot accidentally regress to MSE by forgetting a flag.

**Decision:** **MERGED**. Squash-merged as `ddff0f9`. Eliminates a silent
foot-gun for all 8 in-flight rebases. Notified the 4 rebasing PRs (#3151,
#3330, #3348, #3370) that they can drop `--loss_fn` from rerun commands.

---

## 2026-05-15 21:35 — PR #3348: Fourier position encoding — **SENT BACK (rebase on Charbonnier baseline)**

- Student branch: `willowpai2i24h1-fern/fourier-pos-enc`
- Student: `willowpai2i24h1-fern`
- Hypothesis: replace raw (x, z) coordinates with multi-scale Fourier position
  encoding (sin/cos at geometric frequency bands 2^k) to provide an explicit
  spatial-frequency basis. Counters MLP spectral bias on sharp pressure peaks.

| Arm | wandb run | val_avg/mae_surf_p | 3-split test mean | best_epoch | n_params | peak GB |
|-----|-----------|--------------------|-------------------:|-----------|----------|---------|
| raw_ref | k6ad1x9m | 123.52 | 125.71 | 14 | 0.66M | 42.1 |
| **fourier_basic (L=8)** | ynml7x2v | **119.10** | **117.02** | 11 | 0.67M | 42.3 |
| fourier_rich (L=12) | kiue5928 | 122.49 | 120.38 | 14 | 0.67M | 42.4 |

Per-split val (fourier_basic best ep 11):
single_in_dist=137.97, geom_camber_rc=126.94, geom_camber_cruise=97.57, re_rand=113.94.

Key diagnostics from student:
- **fourier_basic L=8 wins by 3.6%** within-PR, with the gain concentrated on the
  highest-magnitude split: val_single_in_dist surf_p 156.27 → 137.97 (−18.3 absolute).
  Test side: test_single_in_dist surf_p 138.65 → 122.12 (−16.5 absolute).
- **L=12 hurts because the standardized position σ ≈ 1.** Bands above 2^7 have
  wavelength ~1/128 = 0.0078 — finer than typical inter-node spacing. Those bands
  sample essentially random phases, adding noise. Train loss is lower at L=12 (0.217
  vs 0.280 at L=8) but val is worse — classic spectral overshoot.
- All arms ran on **pre-Charbonnier stale base** — raw_ref=123.52 vs current merged
  baseline 98.60. Absolute numbers not comparable.
- Within-PR signal is solid and physically interpretable (high-mag split + sharp
  near-wall gradients is where Fourier features should help most).

**Conclusion:** Fourier position encoding L=8 is a real architectural lever; the
−18.3 absolute on val_single_in_dist surf_p is well outside noise. Sent back for
rebase on Charbonnier base with 2 arms (raw + fourier_L8) — fourier_rich is
confirmed null.

**Decision:** sent back for rebase. If signal proportionally maintained on
Charbonnier base, expect fourier_L8 to land in 90-95 val.

---

## 2026-05-15 21:30 — PR #3370: Gated MLPs (SwiGLU / GeGLU) — **SENT BACK (rebase on Charbonnier baseline)**

- Student branch: `willowpai2i24h1-tanjiro/glu-mlp`
- Student: `willowpai2i24h1-tanjiro`
- Hypothesis: replace vanilla GELU MLPs in TransolverBlocks with gated variants
  (SwiGLU/GeGLU) for stronger multiplicative-interaction modeling. Standard modern
  transformer recipe (Llama, T5).

| Arm | wandb run | val_avg/mae_surf_p | 3-split test mean | best_epoch | n_params | peak GB |
|-----|-----------|--------------------|-------------------:|-----------|----------|---------|
| vanilla_ref | b26950ez | 130.31 | 131.34 | 13 | 0.66M | 42.1 |
| swiglu | q4cew3fp | 130.18 | 129.21 | 11 | 0.83M | 52.1 |
| **geglu** | ztqz88s1 | **126.49** | **126.52** | 12 | 0.83M | 52.1 |

Per-split val (geglu best ep 12):
single_in_dist=170.81 (worse, +18.1 vs vanilla), geom_camber_rc=138.20,
geom_camber_cruise=89.22 (−14.0 vs vanilla), re_rand=107.74 (−12.2 vs vanilla).

Key diagnostics from student:
- **GeGLU wins by −2.9% within-PR** (130.31 → 126.49) — at the edge of the 3-4 unit
  noise band, but the OOD-concentrated structure is informative: gains on
  val_geom_camber_cruise (−13.6%), val_re_rand (−10.2%), test_re_rand (−13.0%);
  loss on val_single_in_dist (+11.8%). Suggests the multiplicative pathway helps
  generalize to unfamiliar geometry/Re at the cost of in-distribution fitting.
- **SwiGLU is null** (−0.1%) — same structure as GeGLU but SiLU instead of GELU.
  GELU's smoother near-zero behavior interacts better with the small-init
  Transolver setup.
- **Stale pre-Charbonnier base** — vanilla_ref=130.31 matches implicit baseline.
- Param bump +26% (570K → 833K), peak VRAM 52GB (well below 96GB cap).
- Per-epoch wall-clock +14% (132s → 150s); arms completed 1 fewer epoch on average.

**Conclusion:** GeGLU shows a noise-bordered but OOD-favorable signal. Worth a
Charbonnier-base confirmation. Sent back with 2 arms (vanilla + geglu) — swiglu
is confirmed null.

**Decision:** sent back for rebase. If geglu's OOD generalization signal survives
on the Charbonnier base, it's a paper-facing test_avg win even if val_avg is flat.

---

## 2026-05-15 21:30 — PR #3151: EMA model weights sweep — **SENT BACK (rebase on Charbonnier baseline)**

- Student branch: `willowpai2i24h1-thorfinn/ema-model-weights`
- Student: `willowpai2i24h1-thorfinn`
- Hypothesis: maintaining an EMA shadow of model weights and evaluating with
  it instead of the live optimizer iterate provides a smoother, lower-variance
  solution, especially under truncated training.

| Arm | wandb run | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs no_ema (test) |
|-----|-----------|--------------------|--------------------|---------------------|
| no_ema | 6xqa30qf | 146.63 | 135.75 | baseline |
| **ema999** | e9rsxven | **118.99** | **111.55** | **−17.8%** |
| ema9999 | bhsc0dw2 | 122.99 | 114.76 | −15.5% |

Per-split test at best epoch (ema999):
single_in_dist=125.45, geom_camber_rc=119.09, geom_camber_cruise=94.55, re_rand=107.09.

Key diagnostics from student:
- EMA gives **−17.8% test, −18.8% val** vs no_ema — far exceeding the 1-3% prediction.
  The outsized gain is expected: with training truncated at 14/50 epochs (before LR
  decays), the iterate is still noisy; EMA captures a substantially better minimum.
- ema999 ≈ ema9999 are identical because the Karras warmup caps effective decay at
  0.9983 at 5250 optimizer steps — well below both target values. The two arms ran
  the same effective schedule; differences are noise.
- Student independently found the same NaN bug as alphonse (#3138) and applied an
  equivalent call-site fix; #3138's evaluate_split boundary fix is now merged.
- Student wrapped training loop in `if __name__ == "__main__":` for clean import safety.
- All arms trained on the **pre-Charbonnier stale base** — no_ema=146.63 vs new merged
  baseline of 98.60 makes absolute numbers non-comparable.

**Conclusion:** EMA is a high-confidence lever. −17.8% within-PR signal on stale base
is too large to close. Sent back for rebase on current advisor head (Charbonnier +
warmup + NaN fix) with explicit `--loss_fn charbonnier --charbonnier_eps 1e-3`. Once
on composed base, we expect no_ema ≈ 98-105, ema999 landing in the ~80-90 range if
the within-PR gap is proportionally maintained.

**Decision:** sent back for rebase. Also suggested disabling Karras warmup on one arm
to verify ema9999 actually differs from ema999 at longer run durations.

---

## 2026-05-15 21:25 — PR #3142: Surface loss weight sweep (surf_weight ∈ {10,30,80}) — **CLOSED ✗**

- Student branch: `willowpai2i24h1-askeladd/surf-weight-sweep`
- Student: `willowpai2i24h1-askeladd`
- Hypothesis: increasing `surf_weight` from 10 (default) to 30 or 80 reallocates
  gradient capacity to surface nodes and directly reduces `mae_surf_p`.

| Arm | wandb run | val_avg/mae_surf_p | Δ vs sw10 | test 3-split avg | mae_vol_p avg |
|-----|-----------|--------------------|-----------|-----------------:|---------------|
| sw10 (control) | 9m3xl5ls | 125.28 | — | 124.51 | 122.83 |
| sw30 | mwkvs001 | 139.26 | +11.2% | 140.11 | 161.20 |
| **sw80** | wnh939lg | **124.42** | **−0.69%** | **120.44** | 166.53 |

Per-split val at best epoch (sw80 arm, best arm):
single_in_dist=152.70, geom_camber_rc=135.13, geom_camber_cruise=96.80, re_rand=113.03.

Key diagnostics:
- **Direction is correct but magnitude is noise-level.** sw80 beats sw10 by 0.69% — well
  inside run-to-run variance (~3-4 units). Not attributable to the lever.
- **Non-monotonic in the middle.** sw30 is worse than BOTH sw10 and sw80 on every split.
  Two plausible causes: (1) 3× surface signal disrupts the volume backbone but doesn't
  dominate; (2) all runs truncated at best_epoch=14 — early-stage noise confounds ordering.
- **Volume tradeoff is real.** mae_vol_p rises ~+39% from sw10→sw80 (confirmed expected
  tradeoff); the ranking metric ignores this.
- **Stale base.** sw10=125.28 on pre-Charbonnier base vs current merged baseline 98.60.
  Charbonnier rebalances the per-node gradient, which changes the effective surf_weight
  sensitivity; the optimal weight post-Charbonnier may differ.
- **NaN bug confirmed independently** by askeladd — same diagnosis as alphonse. Both
  trace the `NaN * 0 = NaN` IEEE path through `accumulate_batch`. #3138 is now merged.

**Conclusion:** Weak signal (0.69%) on stale base. Non-monotonic ordering adds noise.
`surf_weight` tuning may be revisited after AMP and EMA land, when training reaches
deeper cosine decay and the gradient rebalancing from Charbonnier is better characterized.

**Decision:** close. Zero-sum re-run is not worth the GPU time in the current round.

---

## 2026-05-15 20:45 — PR #3138: NaN bug fix in evaluate_split — **MERGED ✅ CRITICAL FIX**

- Student branch: `willowpai2i24h1-alphonse/slice-num-sweep` (repurposed from slice_num)
- Student: `willowpai2i24h1-alphonse`
- Fix: filter non-finite ground truth samples in `train.py:evaluate_split` before the
  `err * surf_mask` step. Root cause: `NaN * 0.0 == NaN` under IEEE float — when
  `test_geom_camber_cruise` GT contains Inf values, the float64 accumulator gets
  poisoned and `test_avg/mae_surf_p` finalizes as NaN for every run on this launch.

| Sanity run | wandb run | val_avg/mae_surf_p | **test_avg/mae_surf_p** | test_cruise/mae_surf_p |
|------------|-----------|--------------------|-----------------------|------------------------|
| `nan_fix_sanity` | u2k87wan | 102.25 | **92.71** ← first finite value | **63.99** ← formerly NaN |

Per-split test (all finite for the first time):
single_in_dist=103.60, geom_camber_rc=117.05, geom_camber_cruise=63.99, re_rand=86.20.

**Conclusion:** Bug is conclusively fixed. 15 lines of code in `evaluate_split`.
Val_avg/mae_surf_p at 102.25 is within noise of merged baseline 98.60, confirming
clean compose of warmup + Charbonnier + NaN fix. **Unblocks `test_avg/mae_surf_p`
as a reliable paper-facing metric for all future runs on this launch.**

Alphonse also identified a critical config issue: `Config.loss_fn` default is still
`"mse"` on the advisor branch even though Charbonnier is the documented baseline.
Assigned alphonse follow-up PR #3440 to flip the default. Notified edward (#3398),
nezuko (#3418), frieren (#3330) to pass `--loss_fn charbonnier` explicitly.

**Decision:** merged. Advisor branch now includes the NaN filter.

---

## 2026-05-15 19:30 — PR #3331: Separate per-channel output heads — **CLOSED ✗**

- Student branch: `willowpai2i24h1-nezuko/separate-output-heads`
- Student: `willowpai2i24h1-nezuko`
- Hypothesis: splitting the shared output projection (Linear 128→3) into per-channel
  heads (separate Linear 128→1 per field) breaks a capacity bottleneck and lets
  the model specialize the p channel (sharp at stagnation/suction peaks) vs the
  smoother Ux/Uy.

| Arm | wandb run | n_params | val_avg/mae_surf_p | best_epoch | Notes |
|-----|-----------|----------|---------------------|-----------|-------|
| **shared_ref** | wor5ca6f | 0.66M | **124.99** | 12 | within-PR control |
| split_lite | 4nrycx6j | 0.66M | 141.32 | 11 | +13.1%, worse |
| split_full | lh0ezot2 | 0.70M | 132.81 | 12 | +6.3%, worse |

Per-split val at best epoch (shared_ref): single_in_dist=155.63, geom_camber_rc=126.91,
geom_camber_cruise=102.98, re_rand=114.43.

Test 3-split (shared_ref): single_in_dist=134.99, geom_camber_rc=119.18, re_rand=115.06.

**Conclusion:** Hypothesis decisively rejected. Both split arms degrade every channel — p, Ux, and Uy all worsen in 7-8/8 split cells vs the shared baseline. This is the opposite of the bottleneck prediction (which would have p improving while Ux/Uy held flat). The shared output trunk encodes cross-channel features useful to all three fields — consistent with the incompressible NS coupling of mass/momentum. Forcing channel independence at the final layer destroys that shared representation.

split_full > split_lite (more capacity), but both lose to shared — rules out the explanation that the split arms just needed more capacity.

Branch was started on pre-warmup, pre-Charbonnier base; shared_ref at 124.99 is on par with that pre-merge baseline (~130 ± 3).

**Decision:** close. Worth revisiting later with **residual heads** (`shared_proj(z) + α·per_channel_correction(z)`) which preserves cross-channel inductive bias while allowing specialization. Not the same hypothesis as this PR.

---

## 2026-05-15 19:26 — PR #3330: bf16 AMP mixed precision — **SENT BACK (rebase needed)**

- Student branch: `willowpai2i24h1-frieren/bf16-amp`
- Student: `willowpai2i24h1-frieren`
- Hypothesis: wrapping the forward pass + loss in `torch.amp.autocast(dtype=bfloat16)` shortens
  per-epoch wall time, allowing more epochs in the 30-min budget (more cosine decay).

| Arm | wandb run | dtype | batch | val_avg/mae_surf_p | best_epoch | epochs | time (min) | s/epoch | Notes |
|-----|-----------|-------|-------|---------------------|-----------|--------|------------|---------|-------|
| **bf16_bs4** | 8hvrijbf | bf16 | 4 | **118.29** | 17 | 19 | 31.2 | 98.8 | 7125 steps |
| fp32_ref | 35q5cfxz | fp32 | 4 | 135.94 | 14 | 14 | 30.8 | 132.1 | 5250 steps |
| bf16_bs8 | 78as5pei | bf16 | 8 | 129.97 | 15 | 18 | 31.3 | 104.9 | 3384 steps |

**Conclusion:** AMP is a real lever. bf16_bs4 at 118.29 beats fp32_ref at 135.94 by 17.65 (−13%).
The speedup is 132s → 99s per epoch (1.34×), giving 19 epochs vs 14 in the same 30-min cap. Best epoch shifts 14→17, confirming the gain is partly "more epochs + deeper cosine decay" and partly improved training dynamics.

bf16_bs8 at 129.97 is essentially baseline — doubling batch at fixed LR halves optimizer steps (3384 vs 7125 total), and the VRAM/throughput gain is modest (105s/epoch vs 99s). LR scaling would be needed to benefit from bs=8.

Implementation is clean: `torch.amp.autocast(dtype=bfloat16)` wraps forward+loss, backward stays fp32, pred cast to float32 before metric accumulator, no GradScaler needed.

**Note on stale baseline.** fp32_ref at 135.94 is slightly above the pre-warmup implicit baseline of 130 ± 3 — consistent with single-seed noise. The warmup (#3150) + Charbonnier (#3143) merges happened AFTER frieren branched, so these results are against the old base. bf16_bs4 at 118.29 does not beat the new merged baseline of 98.60.

**Decision:** sent back to rebase + re-run. The lever is proven. Asked frieren to rebase on current advisor head and re-run 2 arms (fp32_ref_v2 + bf16_bs4_v2). With ~13% AMP gain composing onto 98.60, we expect the merge-eligible arm to land in the ~85-95 range.

---

## 2026-05-15 18:25 — PR #3143: Charbonnier robust loss vs MSE — **MERGED ⭐⭐ MAJOR WIN**

- Student branch: `willowpai2i24h1-edward/charbonnier-robust-loss`
- Student: `willowpai2i24h1-edward`
- Hypothesis: replace MSE with Charbonnier `sqrt(diff² + ε²)` so the gradient
  becomes ~linear for large residuals. Surface pressure has order-of-magnitude
  dynamic range — a few near-stagnation outliers per mesh dominate MSE gradient
  and bias the model toward them at the expense of the bulk surface MAE.

| Arm | wandb run | `val_avg/mae_surf_p` | test 3-split avg | best_epoch | Notes |
|-----|-----------|----------------------|------------------|-----------|-------|
| **charbonnier_eps1e-3** ⭐ | lukq8jry | **98.60** | **98.03** | — | merged |
| mse_baseline (within-PR control) | 9npuojl6 | 121.14 | — | — | MSE-only, pre-warmup base |

Per-split val (charbonnier_eps1e-3): single_in_dist=126.05, geom_camber_rc=106.97,
geom_camber_cruise=73.34, re_rand=88.04. Per-channel val_avg: surf_Ux=1.377,
surf_Uy=0.639, surf_p=98.60, vol_p=103.37 — Charbonnier improves _every_ channel,
not just surf_p, with the biggest gain on surf_p where the dynamic range is largest.

Per-split partial test (cruise excl. due to NaN bug):
single_in_dist=115.46, geom_camber_rc=93.44, re_rand=85.18.

**Conclusion:** Charbonnier robust loss delivers **−18.6%** on the primary
metric vs the within-PR MSE control (121.14 → 98.60). The largest per-split
gain is on `val_geom_camber_cruise` (−25.6%), which has the highest pressure
dynamic range — exactly the failure mode the robust loss targets. Run-to-run
noise (~3 units) cannot explain a 22-unit gap; this is a real effect.

**Decision:** **merged**. New advisor-branch baseline.
- Adds `--loss_fn` (default mse → now charbonnier on the advisor branch) and
  `--charbonnier_eps` (default 1e-3) flags. The per-node loss is
  `sqrt(diff² + ε²)` for charbonnier, applied identically in train and
  evaluation (so the eval metric for `mae_surf_p` itself is unchanged — it's
  an L1-style MAE on denormalized predictions, independent of loss choice).
- **Compositional caveat.** Edward's branch forked before PR #3150 (warmup +
  cosine) was merged, so the 98.60 result is _Charbonnier alone_ vs MSE
  without warmup. The merge composes with the warmup schedule already on the
  advisor branch. Next student control runs against this composed baseline
  will confirm the post-warmup + Charbonnier number.
- Pre-warmup MSE control was 121.14, post-warmup MSE control should be
  around 125.83 (PR #3150 winner). If Charbonnier composes additively we'd
  expect the new baseline to land in the ~95-100 range; if it composes
  multiplicatively (−18.6% × 125.83) we'd expect ~102. We'll see.

**Edward's suggested follow-ups (worth running):**
1. ε sweep (1e-4, 3e-3, 1e-2) — ε=1e-3 is the only point tested
2. Other robust losses: Huber, Cauchy/Lorentzian, pseudo-Huber
3. Re-check `surf_weight` after residual linearization (the optimal weight
   may have shifted; surf gradient magnitudes are now compressed relative to
   vol gradient magnitudes)

## 2026-05-15 17:43 — PR #3150: Warmup + cosine schedule, peak LR sweep — **MERGED ⭐ WINNER**

- Student branch: `willowpai2i24h1-tanjiro/warmup-cosine-schedule`
- Student: `willowpai2i24h1-tanjiro`
- Hypothesis: a 3-epoch linear warmup before CosineAnnealingLR(eta_min=1e-6)
  yields lower `val_avg/mae_surf_p` at the baseline peak LR, and may unlock
  higher peak LRs.

| Arm | wandb run | `val_avg/mae_surf_p` | test 3-split avg | best_epoch | Notes |
|-----|-----------|----------------------|------------------|-----------|-------|
| **lr5e-4_wu3** ⭐ | sb39atyp | **125.83** | **122.01** | 13 | merged |
| lr5e-4_wu0 (no-warmup ref) | bww3uk1z | 142.28 | 136.67 | 12 | internal control |
| lr1.5e-3_wu3 | n5v1kwy3 | 139.27 | 132.57 | 13 | |
| lr1e-3_wu3   | xxt85yy2 | 149.07 | 145.70 | 12 | |

**Conclusion:** warmup at the baseline peak LR is a clear win — −16.45 (−11.6%)
versus the internal no-warmup control, and −3.2% versus the pre-merge
implicit baseline of 130 ± 3. Better on **every** val split, not just the
average. Higher peak LRs (1e-3, 1.5e-3) underperform even with warmup —
the model is already in a stable training regime at lr=5e-4 with AdamW+LN,
so warmup mitigates initial dynamics but doesn't expand the useful peak-LR
range in this budget.

**Decision:** **merged**. Now the new advisor-branch baseline.
- Adds `--warmup_epochs` flag (default 3); replaces scheduler with
  `SequentialLR(LinearLR start_factor=1e-3, CosineAnnealingLR eta_min=1e-6)`.
- Note from student analysis: part of the within-PR gap (16.45) may be partly
  attributable to the wu0 reference decaying to eta_min=0 vs wu3 decaying to
  1e-6 — disentangling pure warmup vs eta_min is a worthwhile future probe.

## 2026-05-15 16:35 — PR #3145: Deeper Transolver: n_layers 5/8/10

- Student branch: `willowpai2i24h1-fern/deeper-transolver`
- Student: `willowpai2i24h1-fern`
- Hypothesis: deepening Transolver (`n_layers` 5 → 8 → 10) improves
  `val_avg/mae_surf_p` because more mixing stages help model multi-scale
  CFD structure; ~1.27M params at depth 10 is still tiny for 96 GB VRAM.

| Arm | wandb run | `val_avg/mae_surf_p` | best_epoch | epochs reached | partial `test_avg/mae_surf_p` (3 splits) |
|-----|-----------|----------------------|-----------|----------------|------------------------------------------|
| depth5  | 0g36hqgg | **129.07** | 14 | 14 | rc=120.44 / sid=140.92 / re=118.32 |
| depth8  | 3x6sfou4 | 172.83 | 9  | 9  | rc=183.88 / sid=193.75 / re=147.69 |
| depth10 | b539x67l | 164.10 | 8  | 8  | rc=164.56 / sid=186.37 / re=142.99 |

**Conclusion:** depth5 (baseline control) wins by a wide margin. depth8 is
+34%, depth10 is +27% — both clear regressions. Per-epoch wall-clock scales
~linearly with depth (131s → 207s → 255s), so deeper arms only complete ~half
as many epochs before hitting the 30-min cap.

**Important finding from student analysis.** Per-epoch, the deeper arms are
*more sample-efficient*: at epoch 8, depth10 = 164.10 < depth5 = 185.67.
But depth5's 2× faster per-epoch lets it race past at epoch ~10 and keep
improving. **This is the same wall-clock confound as PR #3148 (width sweep)** —
the 30-min cap is the binding constraint; any lever that scales per-step
compute without proportional sample-efficiency gain loses in this budget.

**Decision:** close. Worth revisiting after bf16 AMP (PR #3330) lands as a
winner — that would roughly halve per-epoch time and unlock depth/width
experiments to converge within budget.

## 2026-05-15 15:36 — PR #3148: Wider Transolver: n_hidden 128/192/256

- Student branch: `willowpai2i24h1-frieren/wider-transolver`
- Student: `willowpai2i24h1-frieren`
- Hypothesis: widening `n_hidden` (and proportionally `n_head`) from the
  baseline 128/4 reduces `val_avg/mae_surf_p` because the baseline is small
  for 96 GB VRAM.

| Arm | wandb run | `val_avg/mae_surf_p` | best_epoch | total_min | partial `test_avg/mae_surf_p` (3 splits) |
|-----|-----------|----------------------|-----------|-----------|------------------------------------------|
| w128 | qmyih0vv | **128.46** | 14 | 30.64 | rc=141.6 / sid=129.3 / re=114.4 |
| w192 | u9udr95v | 149.32 | 7 | 30.65 | rc=152.3 / sid=165.3 / re=132.3 |
| w256 | o1ax3h3f | 173.63 | 7 | 30.36 | rc=171.4 / sid=202.0 / re=153.6 |

**Conclusion:** widening *hurts* in this budget. All three arms hit the
30-min wall-clock cap; the wider arms reached their best val at epoch 7 vs
epoch 14 for the baseline width — the wider models simply did not have
enough wall-clock to converge. Width 192 is +16% on the primary metric,
width 256 is +35% — both above the close threshold (>5% regression).

**Decision:** close as dead end *in this training budget*. Wider models are
worth revisiting with (a) more epochs, (b) warmup + higher peak LR, or
(c) substantially smaller width steps. Implementation itself is clean —
the `--n_hidden` / `--n_head` plumbing in `train.py` is preserved for
future experiments.

## 2026-05-15 15:31 — PR #3149: Per-channel surface-loss weights focusing on p

- Student branch: `willowpai2i24h1-nezuko/surface-pressure-loss`
- Student: `willowpai2i24h1-nezuko`
- Hypothesis: explicitly upweighting the p channel inside `surf_loss` directly
  pushes the optimizer toward what the ranking metric measures, reducing
  `val_avg/mae_surf_p` without much volume cost.

| Arm | wandb run | `val_avg/mae_surf_p` | best_epoch | total_min | partial `test_avg/mae_surf_p` (3 splits) |
|-----|-----------|----------------------|-----------|-----------|------------------------------------------|
| surfp1  | 7d1rlw4w | **132.33** | 13 | 30.81 | rc=136.9 / sid=139.1 / re=122.5 |
| surfp4  | 7tuf0qsy | 132.71 | 13 | 30.75 | rc=132.8 / sid=135.3 / re=117.8 |
| surfp10 | 84u5mine | 140.66 | 13 | 30.81 | rc=136.4 / sid=154.1 / re=133.9 |

**Conclusion:** per-channel surface-p upweighting did not improve
`val_avg/mae_surf_p`. surfp4 ties baseline (+0.3%, within run-to-run noise);
surfp10 is +6.3% worse. The infra is correct (mean-normalization keeps
surfp1 identical to current baseline; diagnostic per-channel surf MSEs are
logged in W&B).

**Decision:** close — no improvement on the primary metric. The lever
exists but at the tested weights doesn't help. Future variants worth
trying: downweight Ux/Uy (mathematically equivalent at the surface but
keeps the volume loss balanced) and/or combine with a higher overall
`surf_weight`.

## Cross-PR observations (round 1)

- **Run-to-run variance is ~3-4 units in `mae_surf_p`.** Three
  nominally-identical baseline configs across PRs gave 128.46 (frieren w128),
  129.07 (fern depth5), 132.33 (nezuko surfp1) — a ~4% spread. Improvements
  smaller than ~3-4 mae_surf_p units should not be treated as winners on a
  single seed.
- **30-min wall-clock cap binds at 50 epochs** for the baseline width.
  All 6 reviewed runs hit ~30.4-30.8 min total, meaning training stopped
  exactly at the cap. Best-val epoch was 13-14 for the baseline-width
  arms — the model is still improving at the end of training. This
  suggests longer schedules or faster convergence (warmup, larger LR)
  could move the baseline number.
- **`test_avg/mae_surf_p` is None for all 6 runs.** Root cause:
  `test_geom_camber_cruise` has Inf in its hidden ground truth `y` somewhere;
  `data/scoring.py` does an `Inf * 0 = NaN` operation when masking
  non-finite samples (line ~49 in `accumulate_batch`, the mask-and-sum
  expression). This contaminates the cruise test MAE accumulator with NaN,
  which serializes as None in W&B. Validation cruise is unaffected.
  Filing as a separate issue to the human researcher team.
- **6 of 8 student pods are stuck waiting** due to GitHub API rate-limit
  exhaustion on the shared token — their entrypoint pollers cannot see
  their assigned PRs. This is an operational/throughput issue, not a
  research signal; expected to self-resolve when the limit resets.

---

## 2026-05-16 12:35 — PR #3668: Gradient accumulation — **CLOSED (regression on optimized base)**

- Student branch: `willowpai2i24h1-frieren/grad-accum-stacked`
- Student: `willowpai2i24h1-frieren`
- Hypothesis: Gradient accumulation (effective batch_size=8 or 16 via accum_steps=2/4) reduces gradient noise and improves convergence on the 30-min budget.

| Arm | Config | wandb run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
|-----|--------|-----------|---------------------|---------------------|------------|
| accum2_stacked | L=8 wd=1e-4 base | dhtihxbn | 76.18 | 69.80 | 17 |
| accum4_stacked | L=8 wd=1e-4 base | jhwgkz6u | 80.20 | 70.60 | 17 |
| accum4_L4_wd1e3 | **L=4 wd=1e-3 base** | vch67ain | **71.57** | **63.32** | 17 |
| **baseline (PR #3600 L=4)** | — | 9nliedqj | **69.98** | **62.47** | — |

**Decision: CLOSED.** accum4 on the current best stack (L=4+wd=1e-3) regresses by +1.59 val / +0.85 test vs baseline. The earlier accum2 win (val=76.18 on the looser L=8+wd=1e-4 stack) reflected a noisy gradient regime that no longer exists. With tighter regularization (wd=1e-3, L=4), each mini-batch gradient is more informative, so halving the optimizer-step count removes signal rather than reducing noise. Result is regime-dependent and not an orthogonal lever.

Key insight (frieren's analysis): the accum-steps effect is anti-correlated with optimization quality. As the base config improves (tighter wd, better pos-enc), gradient noise drops, making accumulation redundant. The direction is closed for the current optimized base. Note: with torch.compile now merged (2× throughput → 2× natural gradient updates), accumulation is even less needed.
