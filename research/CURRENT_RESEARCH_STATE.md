# SENPAI Research State

- **Last updated:** 2026-05-17 06:30 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Primary metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline (QK-norm + Lion)

**PR #4270** — QK-norm (LayerNorm on Q,K) + Lion + n_hidden=176 + bf16 + epochs=14 (edward), merged 2026-05-17 ~05:30 UTC
- **val_avg/mae_surf_p = 46.9886** (W&B `oospddft`)
- **test_avg/mae_surf_p = 40.4803**
- Per-split test: single_in_dist=43.18, geom_camber_rc=52.79, geom_camber_cruise=25.83, re_rand=40.12
- Wall: ~30.6 min / 14 ep (~131 s/ep), Peak VRAM: 44.6 GB
- Reproduce: `cd "target/" && python train.py --n_hidden 176 --epochs 14 --use_bf16 --use_lion --lion_lr 1e-4 --lion_wd 1e-3 --use_qk_norm`

**QK-NORM IS NOW STANDARD.** All new experiments must include `--use_qk_norm` unless specifically testing its removal.

## Most recent research direction from human researcher team

No GitHub Issues open for this track as of 2026-05-17 06:30 UTC. Proceeding from program contract.

## Current in-flight experiments (8 active)

| PR | Student | Axis being tested | Status |
|---|---|---|---|
| **#4382** | edward | QKV-norm: extend QK-norm to also normalize V projections | WIP — running |
| **#4383** | thorfinn | surf_weight sweep {5, 15} vs default 10 on QK-norm+Lion | WIP — running |
| **#4366** | fern | Lookahead(k=3, α=0.5) on QK-norm+Lion (k=5 was too aggressive, retest) | WIP — sent back, rebase+rerun in progress |
| **#4324** | askeladd | lion_wd=5e-4 + QK-norm stack retest (wd=5e-4 won vs old baseline) | WIP — sent back, rebase+rerun in progress |
| **#4409** | frieren | mlp_ratio=3 + QK-norm + Lion at nh=176+ep12 (FFN capacity) | WIP — just assigned |
| **#4410** | nezuko | loss_type=huber + QK-norm + Lion at nh=176+ep14 (outlier-robust loss) | WIP — just assigned |
| **#4411** | tanjiro | coord_noise_std {0.005, 0.02} sweep + QK-norm + Lion (aug strength) | WIP — just assigned |
| **#4412** | alphonse | batch_size=2 + QK-norm + Lion at nh=176+ep14 (finer gradient steps) | WIP — just assigned |

**Zero idle students. Zero idle GPUs.**

## Round-10 dead-ends (closed this cycle)

| PR | Student | Axis | W&B verdict |
|---|---|---|---|
| #4280 | frieren | Lion+nh=192+ep12: 3 seeds at val 49.6-50.9, +5.5% vs new baseline | CLOSED |
| #4285 | nezuko | Lion lr=2e-4: 2 seeds at val 49.2-49.7, +4.8% vs new baseline | CLOSED |
| #4233 | tanjiro | AGC clip=0.03: val=57.37, +22% catastrophic regress | CLOSED |
| #4354 | alphonse | Lion n_head=2: 2 seeds at val 48.82-49.17, all 4 splits regress | CLOSED |

## Key learnings (Round-10 to date)

1. **QK-norm + Lion is the new stack baseline.** ALL new experiments must build on both. Students who were assigned pre-#4270 had their results re-evaluated against new baseline; most failed.
2. **Baseline shift mid-round (from #4252 to #4270):** When #4270 merged mid-cycle (val 49.26→46.99), several in-flight PRs that beat the OLD baseline failed against the NEW one. Decision rule: send back for QK-norm stack retest if the mechanism is orthogonal AND per-split shows meaningful signal (e.g., geom_camber_rc improvement). Close if mechanism is redundant or all splits regress uniformly.
3. **Students stuck in Claude loop without SENPAI-RESULT:** PRs #4280, #4285, #4233, #4354 all had multiple finished W&B runs but no posted terminal marker. Closed via advisor W&B-data verdict. Students need to be reminded to post results promptly.
4. **nh=192 width axis exhausted under Lion.** 3 seeds at val 49.6-50.9 confirm nh=176 is the width sweet spot at ep12-14 cap. Cap-bound (12ep) can't fully converge nh=192.
5. **Lion LR=1e-4 is the confirmed local optimum.** lr=2e-4 (2× up) regresses; lr=5e-5 (2× down) not needed.
6. **AGC is redundant with Lion.** Lion's sign-update provides gradient-direction stability; AGC-on-top catastrophically regresses. AGC axis closed.
7. **n_head=2 (d_head=88) doesn't help without QK-norm.** Uniform all-split regression confirms wider heads need normalization to unlock; could be retested with QK-norm but per-split showed no asymmetric signal.

## Merge decisions on send-back PRs

### #4366 fern Lookahead(k=3) retest
- **Merge if:** val < 46.99 AND test < 40.48 (strict win on both)
- **Close if:** either metric regresses — Lookahead axis closed

### #4324 askeladd wd=5e-4 + QK-norm retest
- **Merge if:** val < 46.99 AND test < 40.48
- **Close if:** either metric regresses — wd axis needs further refinement (possibly 4e-4 or QK-norm shifts optimal wd)

## Round-10 backlog (post-current-round candidates)

### Tier 1 — CLI sweeps (zero code change, on QK-norm+Lion+nh=176+ep14 baseline)
- **surf_weight sweep** (5, 15): in flight (#4383 thorfinn)
- **mlp_ratio=3**: in flight (#4409 frieren)
- **loss_type=huber**: in flight (#4410 nezuko)
- **coord_noise_std {0.005, 0.02}**: in flight (#4411 tanjiro)
- **batch_size=2**: in flight (#4412 alphonse)
- **mlp_ratio=2 + more epochs (ep16)**: cap-sensitive; may be too long (~35 min at 131 s/ep) but worth checking if frieren opens

### Tier 2 — Small code edits (QK-norm+Lion baseline)
- **Lion eta_min > 0 in cosine schedule**: eta_min=1e-5 (10% of peak_lr), addresses sign-update oscillation at lr→0. ~10 lines in scheduler init.
- **Lion betas sweep** (unhardcode `train.py:586`): try (0.95, 0.99) per recent vision-task literature
- **V-norm (QKV-norm complete)**: in flight (#4382 edward) — extend QK-norm to V projections
- **SWA at end-of-training**: cheap poly-average over last 3 epochs; ~10 lines, zero extra VRAM

### Tier 3 — Architecture changes
- **n_layers=6 at nh=128 or 144** (depth vs width tradeoff, deeper+narrower → same VRAM)
- **DataAugMix geometry** for geom_camber splits (the structural hard split)

### Tier 4 — Bold swings (if plateau continues)
- **Physics-informed loss** — divergence-free penalty (∇·u=0) on velocity
- **Cross-attention with geometric invariance** for pressure-field decoding

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
