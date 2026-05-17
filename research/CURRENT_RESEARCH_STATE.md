# SENPAI Research State

- **Last updated:** 2026-05-17 11:45 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Primary metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## 🎯 NEW BASELINE — PR #4550 (plateau broken)

**PR #4550** — Per-Foil Chord-Relative Coords + foil_id (edward), MERGED 2026-05-17 11:32 UTC
- **val_avg/mae_surf_p = 44.2736** (W&B `a46jhvdo`, −2.72 vs prior #4270)
- **test_avg/mae_surf_p = 38.1696** (−2.31 vs prior)
- Per-split test: single_in_dist=42.74, **geom_camber_rc=53.02 (flat — new plateau target)**, geom_camber_cruise=21.50 (−4.33!), re_rand=35.42 (−4.70!)
- Wall: ~30.4 min / 14 ep, Peak VRAM: 44.6 GB
- Reproduce: `cd "target/" && python train.py --n_hidden 176 --epochs 14 --use_bf16 --use_lion --lion_lr 1e-4 --lion_wd 1e-3 --use_qk_norm --use_per_foil_coords`

**STANDARD STACK NOW INCLUDES:** `--use_qk_norm --use_per_foil_coords`. New experiments must include both unless specifically testing their removal.

**Plateau broken:** First val improvement after 28 consecutive non-improvements since #4270 merged at 05:30. Plateau counter RESET to 0.

## Most recent research direction from human researcher team

No GitHub Issues open for this track as of 2026-05-17 11:45 UTC. Proceeding from program contract.

## Current research focus & themes (post-plateau-break)

**Active plateau target:** `geom_camber_rc` (53.02 in new baseline, flat vs #4270 53.79). The per-foil-coords win came from cruise/re_rand splits which have stronger AoA variation. geom_camber_rc tests racecar-camber-OOD where the foil shapes are heavily extrapolated — this remains the hardest split.

**Working mechanism axes after #4550 win:**
1. **Per-foil local frame extensions** — translation gave us −2.72 val. Natural next axes: AoA-rotation, chord-scale (camber-relative), per-foil readout.
2. **Slice-collapse architectural fix** — tanjiro's #4532 diagnostic exposed a real pathology in Transolver: slice_weights are softmax-per-node → all slice centroids collapse → slice_attention has nothing to specialize on. Two complementary fixes assigned: per-node RoPE before aggregation (#4584), and load-balancing aux loss (#4586).
3. **Pair-free OOD augmentation** — #4567 camber-jitter is the only pair-free augmentation feasible given unique mesh topology per sample. Still in flight.
4. **Physics-aware auxiliary objectives** — #4551 Stokes div(u) loss still in flight.

**Closed/dead axes (cumulative since round-10):** Loss-shaping (Huber, Focal-L1 two-sided null), eta_min, skip-residual variance, LR-schedule, batch-size both sides, surf_weight, FFN width, num-layers/n-hidden trade, LayerScale γ, EMA, SWA, Post-LN, AGC, n_head=2, β1=0.95, Lion lr=2e-4, wd-only sweeps, V-norm, TTA, constant-LR, GeoMix-style mesh pairing, variance loss, zonal TE+LE loss, LLRD, 2D RoPE on slice-Q/K.

## Current in-flight experiments (8 active, zero idle)

| PR | Student | Axis being tested | Round | Branch baseline |
|---|---|---|---|---|
| **#4535** | thorfinn | LinearNO drop-in linear attention (Wu 2024) | R-13 | old (pre-#4550) |
| **#4548** | askeladd | LE-emphasis-only loss (w=3 on x_norm<0.1) | R-14 | old |
| **#4551** | nezuko | Stokes incompressibility aux (λ=0.01) | R-14 | old |
| **#4567** | fern | Camber-M jittering (σ=0.5 on x[:, 15]) | R-14 | old |
| **#4568** | frieren | Adaptive surface focal-loss (γ=0.5 mean-norm) | R-14 | old |
| **#4583** | edward | **Per-Foil AoA Rotation** (chord-aligned local frame) | R-15 | NEW (#4550) |
| **#4584** | tanjiro | **Per-Node 2D RoPE on fx_mid BEFORE slice agg** (fixes #4532 collapse) | R-15 | NEW |
| **#4586** | alphonse | **Slice-Routing Diversity Loss** (Switch-style λ-sweep {0.001, 0.01, 0.1}) | R-15 | NEW |

**Zero idle students. Zero idle GPUs.** 3 fresh Round-15 PRs (compound on new baseline) + 5 Round-13/14 PRs running on old baseline.

### Round-14 / Round-13 PRs running on OLD baseline (post-merge reconciliation needed)

The 5 R-13/R-14 in-flight PRs (#4535, #4548, #4551, #4567, #4568) all started before #4550 merged at 11:32. When they return results, evaluation needs to be against BOTH:
- **Old baseline (#4270, val=46.99):** for the original hypothesis test as-designed
- **New baseline (#4550, val=44.27):** to determine if the result merits merging now

Decision rules for R-13/R-14 returns:
- If val < 44.27 → MERGE (improves on new baseline, despite being on old branch — the mechanism is independent of per-foil-coords)
- If val ∈ [44.27, 46.99] → request rebase onto advisor branch + retrain with `--use_per_foil_coords` (compounding test)
- If val > 46.99 → CLOSE per original rubric

### Round-13 / Round-14 design logic

- **Loss-shaping axis confirmed dead** (Huber #4410 + Focal-L1 #4489 two-sided null): no further loss-shaping PRs beyond what's currently in flight.
- **Eta_min axis confirmed dead** (#4478 non-monotone): no further LR-schedule PRs.
- **Skip-residual variance axis dead** (#4474 closed via W&B).
- **Live axes:** loss-zonal (LE-only #4548), feature engineering (per-foil coords #4550), physics regularizer (Stokes #4551), optimization geometry (LLRD #4549), plus Round-13 axes (variance loss, GeoMix, RoPE-2D, LinearNO).
- **Hot follow-up pair:** #4548 (LE-only loss) and #4550 (per-foil coords) both stem from askeladd's #4511 diagnostic. They test complementary interpretations: LE-emphasis fixes the dominant error zone; per-foil-coords fixes the structural feature gap that confused her zone-loss formula.

### CRITICAL NOISE-FLOOR RECALIBRATION (#4411 seed-2 result, 2026-05-17 09:34)

- **Inter-seed spread on this stack:** val=2.31, test=1.14 (from #4411 seed-1 vs seed-2 at noise=0.005).
- **Old "close-tie" zone (val 46.6-47.5):** TOO TIGHT given the new spread. A single seed in this band is NOT a reliable signal.
- **Decision rubric update:** For close-tie zones, require either:
  - val < 46.5 (1σ below baseline)
  - OR ≥3 seeds with mean val ≤ 46.99
  - OR test_avg < 40.0 AND test_geom_camber_rc < 51.0 (per-split structural signal)
- **In-flight #4478 eta_min=0.05 send-back:** treat its N=1 result with skepticism. If interesting, immediately request 2 more seeds before any merge.

## Round-10/11 dead-ends (cumulative, 16 closures since #4270 merged)

| PR | Student | Axis | W&B verdict |
|---|---|---|---|
| #4280 | frieren | Lion+nh=192+ep12: 3 seeds val 49.6-50.9 | CLOSED |
| #4285 | nezuko | Lion lr=2e-4: 2 seeds val 49.2-49.7 | CLOSED |
| #4233 | tanjiro | AGC clip=0.03: val=57.37 (+22% catastrophic) | CLOSED |
| #4354 | alphonse | Lion n_head=2: 2 seeds val 48.82-49.17 | CLOSED |
| #4382 | edward | V-norm: val=72.52 (+54.4% CATASTROPHIC) | CLOSED |
| #4366 | fern | Lookahead k=3/5: val=50.03 (axis dead) | CLOSED |
| #4324/#4413 | askeladd | wd=5e-4+QK-norm: mechanism overlap | CLOSED |
| #4178 | thorfinn | EMA (decay=0.999): no signal | CLOSED (prior) |
| #4409 | frieren | mlp_ratio=3: val=50.76 (+8%) — FFN width axis | CLOSED |
| #4410 | nezuko | loss_type=huber: val=54.27 (+15.5%) — tail-suppress wrong direction | CLOSED |
| #4412 | alphonse | batch_size=2: val=50.54 (+7.6%) | CLOSED |
| #4416 | edward | LayerScale γ=1e-4 AND γ=1.0 both regress | CLOSED (axis exhausted) |
| #4417 | fern | SWA ep11-14: val=48.53 (+3.3%) | CLOSED (3rd time-avg failure) |
| #4418 | askeladd | Lion β1=0.95: val=54.40 (+15.8% severe) | CLOSED |
| #4476 | frieren | n_layers=6 nh=128: val=50.16 (+6.8%) | CLOSED (param-budget exhausted) |
| #4383 | thorfinn | surf_weight {5,15} 3 arms close-tie/regress | CLOSED (axis exhausted) |
| #4485 | askeladd | Constant LR after warmup: val=59.50 (+26.6% catastrophic, oscillation) | CLOSED |
| #4488 | frieren | Post-LN: val=50.71 (+7.9%) — converges but slow, every split worse | CLOSED (norm-placement axis exhausted) |
| #4486 | fern | TTA K=8 coord-noise: test+0.032, all splits worse — Jensen's bias | CLOSED (TTA axis closed) |
| #4483 | thorfinn | bs=8+ep18: val=55.95 (+19%) — Lion update-count-limited, not gradient-variance-limited | CLOSED (bs axis closed both sides) |
| #4411 | tanjiro | coord_noise=0.005 seed-1 fluke; 2-seed mean val=48.19 worse, OOD splits flip direction | CLOSED (noise-floor calibration learning) |
| **#4474** | alphonse | Skip-scale 1/√2: val=48.12 (+1.13), test=40.85 (+0.37) | CLOSED via W&B (student pod stuck in rate-limit loop) |
| **#4489** | edward | Focal-L1 α=0.5: val=49.56 (+5.5%), geom_camber_rc=55.98 (+6.0%) — TWO-SIDED NULL with #4410 Huber | CLOSED (loss-shaping axis dead) |
| **#4511** | askeladd | Zonal w=3 TE + w=2 LE: val=48.21 (+2.60%); **valuable diagnostic: LE-MAE 2× larger than TE-MAE**, inverting Kutta hypothesis | CLOSED — diagnostic motivates #4548 LE-only follow-up |
| **#4478** | nezuko | eta_min=0.05 rerun: val=47.92 (+0.93), test=41.68 (+1.20), all splits regress; NON-MONOTONE vs 0.10 floor | CLOSED (eta_min axis dead) |
| **#4530** | fern | GeoMix p=0.15: val=48.15. **DATASET-STRUCTURAL FINDING:** unique mesh topology per sample (0/457 racecar_tandem pairs feasible). Rules out per-node pairing methods. | CLOSED (axis untestable as-stated) |
| **#4510** | frieren | Variance+Mean α=0.8: val=48.81 (+1.82), geom_camber_rc=53.20 (+0.41 wrong dir). Spatial uniformity regularizer pulls capacity FROM physically-real spikes. | CLOSED (variance-loss axis dead) |
| **#4532** | tanjiro | 2D RoPE on slice-Q/K (post-aggregation): val=53.09 (+13%). **Diagnostic: slice centroids collapse to ~same point (std=0.017 rad) → rotation is no-op.** | CLOSED — diagnostic motivates #4584 per-node RoPE pre-agg follow-up |
| **#4549** | alphonse | Lion LLRD α=0.7: val=48.83 (+1.84), rubric falsified. LLRD is fine-tune protocol; we train from scratch → early blocks need full LR. | CLOSED (optimizer axis well-explored) |

## PLATEAU BROKEN at 2026-05-17 11:32 UTC (PR #4550 merged, val=44.27)

**Plateau counter RESET to 0** after #4550 edward per-foil-coords delivered the first val improvement (−2.72) since #4270. Plateau lasted 28 experiments (16 Round-10/11 closures + 12 Round-12/13 closures).

**Mechanism of the breakthrough:** Two raw features appended to inputs — `x_chord` (world-x minus stagger for foil-2) and `foil_id` (0/1 indicator). Gives attention/slice-routing a per-foil translation-invariant frame. Massive gains on geom_camber_cruise (−4.33) and re_rand (−4.70). geom_camber_rc flat (53.02) — NEW plateau target.

**Round-15 follow-ups** (3 PRs assigned, all compound on #4550 with `--use_per_foil_coords`):
- #4583 edward — per-foil AoA rotation (translation + rotation completion of local-frame story)
- #4584 tanjiro — per-node 2D RoPE BEFORE slice aggregation (fixes the slice-collapse pathology #4532 surfaced)
- #4586 alphonse — slice-routing diversity loss (Switch-style aux, independent fix for slice-collapse)

**Historical PLATEAU PROTOCOL note (2026-05-17 11:00 UTC):**

**Two valuable diagnostic findings this cycle:**
1. **Dataset-structural (fern #4530):** TandemFoilSet has unique mesh topology per sample. Rules out per-node pairing methods (MixUp, GeoMix as-stated, paired distillation, sample interpolation). The only pair-based augmentation possible without infra work is at INPUT scalar level (NACA params) — implemented in fern's follow-up #4567 camber-M jittering.
2. **Loss-mechanism (frieren #4510):** Spatial-uniformity regularization (variance-loss) is the WRONG direction. Stagnation/TE error spikes are physically real, not artifacts. The right direction is error-MAGNITUDE-aware concentration (frieren's #4568 follow-up), NOT uniformity.

**6 of 8 in-flight are Round-14 PRs testing complementary mechanisms** triggered by askeladd's LE-dominance finding and frieren's variance diagnosis. All target geom_camber_rc plateau from different angles.

### Most promising signal — but technically not a win

- **tanjiro #4411 arm A (coord_noise_std=0.005):** val=47.03 (+0.09% regress) but **test=40.35 (-0.32% IMPROVEMENT)**. Just outside noise floor on val. Arm B (std=0.02) still running — if both arms close-tie, the coord-noise axis may be a noise floor under QK-norm. WAIT for arm B.
- **thorfinn #4383 arm A (surf_weight=5, run-B):** val=47.36 (+0.79% regress), test=40.64. Run-A `1grt9rc9` val=48.21 (+2.6%), test=40.22 (BEATS by 0.26). 0.85 spread on same config = noise floor ~0.5-1.0 val. sw=15 still running.

### Round-12 wave COMPLETE. Round-13 wave (5 fresh axes) in flight.

**Round-12 wave final tally (5 of 5):**
- thorfinn #4483 bs=8+ep18 — **CLOSED** (val=55.95, update-count-limited)
- askeladd #4485 constant LR — **CLOSED** (val=59.50 catastrophic)
- fern #4486 TTA K=8 — **CLOSED** (test +0.032, Jensen's bias)
- frieren #4488 Post-LN — **CLOSED** (val=50.71, every split worse)
- edward #4489 focal-L1 α=0.5 — still running

**Round-13 wave (5 mechanism surfaces, just assigned):**
1. **Loss-formulation tier:**
   - #4510 frieren Variance+Mean Loss — L = 0.8·mean(|e|) + 0.2·std(|e|) (Hanna 2024)
   - #4511 askeladd Zonal/Wake-Emphasis — 3x w on TE, 2x on LE (PINN-zonal Sep 2025)
2. **Data tier:**
   - #4530 fern GeoMix — λ-mix training samples at p_mix=0.15 (Chen 2024 ICML)
3. **Position-encoding tier:**
   - #4532 tanjiro 2D RoPE — geometry-relative attention bias, d_rope=32 (EVA-02)
4. **Attention-compute tier:**
   - #4535 thorfinn LinearNO — drop-in elu+1 linear attention (Wu 2024 NeurIPS)

**Plus 3 carryovers/send-backs:**
- alphonse #4474 skip-1/√2 (residual scaling — Round-11 carryover)
- nezuko #4478 eta_min_fraction=0.05 send-back (LR floor sweet-spot — critical falsifying test for OOD signal)
- edward #4489 focal-L1 α=0.5 (Round-12 still running)

**Total 8 in-flight on 5 orthogonal mechanism surfaces.** Round-13 attacks the plateau from data, loss, position, and attention angles simultaneously. If ANY breaks the plateau, several may compound.

## Key learnings (Round-10 to date)

1. **QK-norm + Lion is the new stack baseline.** ALL new experiments must build on both. Students who were assigned pre-#4270 had their results re-evaluated against new baseline; most failed.
2. **Baseline shift mid-round (from #4252 to #4270):** When #4270 merged mid-cycle (val 49.26→46.99), several in-flight PRs that beat the OLD baseline failed against the NEW one. Decision rule: send back for QK-norm stack retest if the mechanism is orthogonal AND per-split shows meaningful signal (e.g., geom_camber_rc improvement). Close if mechanism is redundant or all splits regress uniformly.
3. **Students stuck in Claude loop without SENPAI-RESULT:** PRs #4280, #4285, #4233, #4354 all had multiple finished W&B runs but no posted terminal marker. Closed via advisor W&B-data verdict. Students need to be reminded to post results promptly.
4. **nh=192 width axis exhausted under Lion.** 3 seeds at val 49.6-50.9 confirm nh=176 is the width sweet spot at ep12-14 cap. Cap-bound (12ep) can't fully converge nh=192.
5. **Lion LR=1e-4 is the confirmed local optimum.** lr=2e-4 (2× up) regresses; lr=5e-5 (2× down) not needed.
6. **AGC is redundant with Lion.** Lion's sign-update provides gradient-direction stability; AGC-on-top catastrophically regresses. AGC axis closed.
7. **n_head=2 (d_head=88) doesn't help without QK-norm.** Uniform all-split regression confirms wider heads need normalization to unlock; could be retested with QK-norm but per-split showed no asymmetric signal.

## Round-13 backlog (researcher-agent ideas, 2026-05-17 08:35)

See `research/RESEARCH_IDEAS_2026-05-17_0820.md` for full reasoning + literature citations. Top 5 ranked:

1. **Variance+Mean Composite Loss** — `L = 0.8·mean(|e|) + 0.2·std(|e|)`. 2-line change. Penalizes localized error spikes (geom_camber_rc dominant pattern). Backed by Hanna et al. arXiv:2412.13993.
2. **GeoMix Geometry Augmentation** — Interpolate input/target pairs from nearby-camber training cases (λ ~ Beta(2,2)). Bridges to OOD M=6-8 from training M=3-5. Medium effort. Highest expected single-PR gain.
3. **2D Rotary Position Encoding (RoPE-2D)** — Encode (x,y) as rotary frequencies in Q,K. Geometry-relative attention prior. Apply d_rope = n_hidden//4 = 48.
4. **Zonal / Wake-Emphasis Loss** — 3× weight on TE/wake nodes (x_norm > 0.6). Domain knowledge: camber shift moves Kutta condition → most strongly affects TE. 5-line change.
5. **GFocal Dual-Path Attention** — Parallel Nyström global path (m=64 landmarks) + slice local path, learnable gate fusion. Hardest impl. Save for plateau extension if loss/data ideas (#1, #2, #4) all fail.

**Stop condition (per researcher):** If Variance+Mean loss AND GeoMix both fail to improve `geom_camber_rc` by >1%, the OOD gap is architectural — escalate to GFocal.

---

## Plateau-break next tier (Round-12 backlog if Round-11 wave regresses)

If alphonse skip-1/√2, frieren L6, nezuko eta_min all regress (and prior 5 plateau-breaks), escalate to:
1. **Physics-informed divergence-free loss** — penalty term λ × E[||∂Ux/∂x + ∂Uy/∂y||²] computed via finite-diff along foil contour. Big swing, ICML-worthy. Implementation needs contour-ordered surface points (verify dataset has them).
2. **Gradient-based input features** — append ∂p/∂x, ∂p/∂y to input encoding via local finite-diff.
3. **Multi-scale slice_num** — heterogeneous slice_num across layers (e.g., layers 0-1 → 128, layers 2-3 → 64, layer 4 → 32). Captures hierarchical physics scales.
4. **bs=8 with VRAM headroom** — alphonse bs2 regressed; opposite direction may help (Lion needs variance reduction).
5. **Wider heads n_head=8 with QK-norm** — d_head=22 was bad pre-QK-norm; QK-norm may stabilize narrow heads.
6. **Constant lr after warmup** — if nezuko eta_min works, this is the natural extension (full constant LR, no cosine).
7. **Tail-emphasizing loss** — focal-style weighting or |∂p/∂x|-region upweight (opposite of huber's tail-suppress).

## Round-11 in-flight summary (8 plateau-break axes)

All 8 in-flight experiments target *different* mechanisms — orthogonal axes per CLAUDE.md "one hypothesis per PR":

| Axis | PR | Mechanism |
|---|---|---|
| Augmentation strength | #4411 (tanjiro) | coord_noise sweep — close-tie arm A, arm B pending |
| Loss reweighting | #4383 (thorfinn) | surf_weight — close-tie sw=5, sw=15 pending |
| Residual gating (learnable) | #4416 (edward) | LayerScale γ=1.0 retest |
| Weight averaging (eval-time) | #4417 (fern) | SWA over ep11-14 |
| Optimizer momentum window | #4418 (askeladd) | Lion β1=0.95 |
| Residual scaling (fixed) | #4474 (alphonse) | Skip-connection 1/√2 |
| Depth-width tradeoff | #4476 (frieren) | n_layers=6 at nh=128 |
| LR schedule floor | #4478 (nezuko) | eta_min=1e-5 in cosine |

If ANY of these breaks plateau, the others may compound — Lion β1 + eta_min + SWA could all combine.

## Cross-cutting findings (apply to all in-flight PRs)

1. **SwiGLU FFN is default** (#3814 merged).
2. **L1 loss is default** (`Config.loss_type = "l1"`) — nezuko #4410 testing huber.
3. **Lion is default optimizer** (#4252 merged).
4. **QK-norm is NOW STANDARD** (#4270 merged) — `--use_qk_norm` required on all new experiments.
5. **bf16 autocast is default** (#3981 merged).
6. **Fourier PE num_freq=4 is default** (#3372 merged, "4 won vs 8" confirmed in code comment).
7. **coord_noise_std=0.01 is default** (#3632 merged) — tanjiro #4411 sweeping {0.005, 0.02}.
8. **Grad clip max_norm=1.0**, warmup 2 epochs, batch=4.
9. **n_hidden=176, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2** are current default architecture.
10. **`geom_camber_rc` is the structural hard split** — QK-norm has moved it twice (54.75→52.79), still the hardest test split.

## Confirmed exhausted (do not retry on this stack)

- AdamW optimizer variants (any betas / wd / lr) — Lion supersedes
- Surface loss reweighting by target magnitude (pmag-weight, val +4.5% regress)
- Surface loss reweighting by DSDF proxy (curvature, val +12.2% regress)
- slice_num=48 (U-shape), 96 (monotonic worse from 64), 128
- n_head=8 (d_head=22 CUDA fragmentation)
- n_head=2 WITHOUT QK-norm (d_head=88, uniform regression) — see #4354
- n_layers=6 (cap-bound under-trained)
- RMSNorm (slower kernel + slice-attention breakage)
- Multi-scale Fourier PE wide (absorbed by width)
- DropPath, mlp2 gate, attn_dropout, asinh input transform
- AdaBelief optimizer, OneCycleLR, curriculum learning
- DSDF clip thresholds (no-op confirmed via dataset analysis)
- Camber flip augmentation (NACA-M asymmetry unflippable)
- nh=208 (cap-bound) and nh=192 (width saturates at Lion+ep12)
- AGC (Adaptive Gradient Clipping) — redundant with Lion sign-update, #4233
- EMA weight averaging — no signal under monotonic cosine-descent, #4178
- Lion lr=2e-4 (regresses vs lr=1e-4 optimum), lr=5e-5 (inferred from lr landscape)

## Pod environment notes

- All Round-10 student pods enforce **`SENPAI_TIMEOUT_MINUTES=30`** hard cap.
- Per-epoch walls (bf16): nh=176 ≈ 131 s/ep (Lion+QK-norm), nh=192 ≈ 131-145 s/ep.
- VRAM peaks (bf16): nh=176+Lion+QK-norm ≈ 44.6 GB. H100 has 96 GB — ample headroom.

## Baseline progression (val_avg/mae_surf_p)

- #3091 baseline: 109.42 → ... → #3814 SwiGLU: 64.24 → ... → #3981 bf16+ep18: 53.82 → #4082 nh=176: 50.90 → #4106 nh=192+ep20: 48.84 → #4252 Lion+nh=176+ep14: 49.26 → **#4270 QK-norm+Lion+nh=176+ep14: 46.99**

Total improvement from #3091: **−57.0% val, −55.8% test.** QK-norm + Lion + bf16 + width-sweep at 1.23M params (Transolver SwiGLU at nh=176).
