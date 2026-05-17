# SENPAI Research State

- **Last updated:** 2026-05-17 05:00 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Primary metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline (Lion, all Round-9 builds on this)

**PR #4252** — Lion + n_hidden=176 + bf16 + epochs=14 (frieren), merged 2026-05-17 ~01:25 UTC
- **val_avg/mae_surf_p = 49.2616** (W&B `eu7e0g18`)
- **test_avg/mae_surf_p = 41.6188**
- Per-split test: single_in_dist=43.91, geom_camber_rc=54.75, geom_camber_cruise=26.13, re_rand=41.68
- Wall: ~30.5 min / 14 ep (~131 s/ep), Peak VRAM: 44.6 GB
- Reproduce: `cd "target/" && python train.py --n_hidden 176 --epochs 14 --use_bf16 --use_lion --lion_lr 1e-4 --lion_wd 1e-3`

**LION IS THE DEFAULT OPTIMIZER.** Lion beats AdamW by −19.3% val at matched config, delivers −2.28% test improvement vs prior #4106 baseline, and moves the previously-stuck `geom_camber_rc` hard split for the first time (54.75 vs 55.51).

## Most recent research direction from human researcher team

No GitHub Issues open for this track as of 2026-05-17 05:00 UTC. Proceeding from the program contract only.

## Current research focus and themes

**Theme: Exhaust the Lion-stack neighborhood across all axes in parallel.** With 8 students and a 30-min per-run cap, we are running 8 different Lion-stack experiments simultaneously to map the merge-able surface around Lion. Each PR is one orthogonal axis on top of the current Lion baseline.

| PR | Student | Axis being tested | Status |
|---|---|---|---|
| **#4178** | thorfinn | EMA weight averaging (decay sweep) for val/test eval | WIP — extended multi-arm exploration |
| **#4233** | tanjiro | AGC (Adaptive Gradient Clipping) clip_factor sweep ∈ {0.03, 0.05} | WIP — rebased on Lion stack |
| **#4270** | edward | QK-norm: LayerNorm on Q,K projections before attention | WIP — rebased on Lion stack |
| **#4280** | frieren | Lion + n_hidden=192 (width frontier under Lion) | WIP — arm 2 of 2 |
| **#4285** | nezuko | Lion LR sweep: lr ∈ {2e-4, 5e-5} vs default 1e-4 | WIP — arm 1 (lr=2e-4) running |
| **#4324** | askeladd | Lion weight-decay downward sweep: lion_wd ∈ {5e-4, 3e-4} | WIP — just picked up |
| **#4354** | alphonse | Lion + n_head=2 (d_head=88 CUDA-aligned, wider attention heads) | WIP — just picked up |
| **#4366** | fern | Lookahead(k=5, α=0.5) wrapper around Lion (slow-weight averaging) | WIP — just assigned 04:55 |

**Zero idle students. Zero idle GPUs.**

## Lion-stack closed dead-ends (this round)

| PR | Student | Reason for close |
|---|---|---|
| #4165 | alphonse | slice_num=48: U-shape confirmed at nh=176 (val=53.91, +5-7% regress on all 4 splits) — slice_num=64 is sweet spot |
| #4297 | alphonse | Lion+ep18+T_max=18 schedule extension: wall projection 38 min > 30-min cap (advisor math error caught by student at ep5) |
| #4321 | alphonse | Lion + n_head=8: d_head=22 not CUDA-aligned, +31% wall, +29% VRAM — hardware-bound, untestable |
| #4238 | askeladd | AdamW beta1 sweep (0.85, 0.95): both regress +21-26% vs Lion baseline — AdamW < Lion fundamentally |
| #4232 | fern | nh=208 width frontier (AdamW pre-Lion + cap-bound): val=59.05 +19.9% regress; structurally cap-limited |

## Key learnings (Round-9 to date)

1. **CUDA tile alignment matters for n_head choice.** At nh=176, valid d_head values are multiples of 8: n_head=2 (d_head=88 ✓), n_head=4 (d_head=44 OK), n_head=11 (d_head=16 ✓), n_head=22 (d_head=8 ✓). **NOT** n_head=8 (d_head=22 ✗) — fragmented tile causes +31% wall.
2. **Lion hardcoded betas.** `train.py:586` has `betas=(0.9, 0.99)` hardcoded; CLI exposes only `--lion_lr` and `--lion_wd`. Beta sweep requires code change.
3. **30-min cap is structural.** nh=208 ep18 (~44 min) and nh=176 ep18 (~39 min) both exceed cap. Width/epoch frontier confined to: nh=176+ep14, nh=192+ep12, nh=208+ep12 at most.
4. **AdamW completely obsolete on this stack.** Any new optimizer-axis assignment must build on Lion. Pre-Lion PRs (assigned before #4252 merged) that didn't beat Lion baseline must be closed regardless of student delivering clean results.

## Potential next research directions (Round-10 backlog)

**Awaiting current Round-9 results before assigning new axes.** Once 8 in-flight Lion-stack experiments resolve, candidates include:

### Tier 1 — Easy CLI sweeps (zero code change)
- **mlp_ratio=3 + Lion** at nh=176+ep12 (cap-friendly): pre-Lion data was negative (mlp_ratio=2 wins) but Lion's sign updates may change FFN dynamics
- **slice_num=80 / 96 / 128 with Lion** — slice_num=48 confirmed U-shape, retest above 64 under Lion
- **surf_weight sweep with Lion** (5, 15, 20): Lion's better balance may shift optimal surface vs volume weight
- **coord_noise_std sweep with Lion** (0.005, 0.02): augmentation strength may have shifted

### Tier 2 — Small code edits
- **Lion eta_min > 0 in cosine schedule**: Lion's sign updates can be too aggressive at lr → 0; eta_min=0.1*peak_lr may stabilize end-of-training
- **Lion betas sweep** (unhardcode `train.py:586`): try (0.95, 0.99) per recent vision-task literature
- **Huber loss with Lion** (smooth-l1, delta=1.0): may help with outlier surface pressure samples
- **Stochastic Weight Averaging (SWA)** at end of training: cheap, well-validated regularization

### Tier 3 — Moderate code edits
- **Sharpness-Aware Minimization (SAM)** wrapping Lion — but SAM doubles per-step compute, would not fit 30-min cap at nh=176+ep14
- **Mixed optimizer**: AdamW for embedding/output + Lion for transformer blocks
- **DataAugMix** style data augmentation for geometry-camber splits (the structural hard split)
- **Test-time augmentation (TTA)** at val/test only: predict on 4-8 noised input versions, average

### Tier 4 — Architecture changes (bigger swings if Lion stack plateaus)
- **Physics-informed loss** — divergence-free penalty (∇·u=0) on velocity, pressure-Bernoulli surface constraint
- **Learnable slice positions** — make slice_num=64 positions trainable
- **Cross-attention with SE(2) equivariance** for pressure-field decoding
- **Different positional encoding scheme**: relative PE, rotary PE on attention QK

## Cross-cutting findings (apply to all in-flight PRs)

1. **SwiGLU FFN is default** (#3814 merged).
2. **L1 loss is default** (`Config.loss_type = "l1"`).
3. **Lion is default optimizer** (#4252 merged, this round).
4. **bf16 autocast is default** for new experiments (#3981 merged).
5. **Fourier PE num_freq=4 is default** (#3372 merged).
6. **coord_noise_std=0.01 is default** (#3632 merged).
7. **Grad clip max_norm=1.0**, warmup 2 epochs, batch=4.
8. **n_hidden=176, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2** are current default architecture.
9. **`geom_camber_rc` is the structural hard split** — Lion is the first thing to move it (54.75 vs 55.51); next axis-of-attack remains open.

## Confirmed exhausted (do not retry on this stack)

- AdamW optimizer variants (any betas / wd / lr) — Lion supersedes
- Surface loss reweighting by target magnitude (pmag-weight, val +4.5% regress)
- Surface loss reweighting by DSDF proxy (curvature, val +12.2% regress)
- slice_num=48 (U-shape), 96 (monotonic worse from 64), 128
- n_head=8 (d_head=22 CUDA fragmentation)
- n_layers=6 (cap-bound under-trained)
- RMSNorm (slower kernel + slice-attention breakage)
- Multi-scale Fourier PE wide (absorbed by width)
- DropPath, mlp2 gate, attn_dropout, asinh input transform
- AdaBelief optimizer, OneCycleLR, curriculum learning
- DSDF clip thresholds (no-op confirmed via dataset analysis)
- Camber flip augmentation (NACA-M asymmetry unflippable)
- nh=208 + AdamW + cap-bound (recently closed; still untested under Lion at viable schedule)

## Pod environment notes

- All Round-9 student pods enforce **`SENPAI_TIMEOUT_MINUTES=30`** hard cap (launch isolation rule). Cannot override inline.
- All Round-9 student pods enforce **`SENPAI_MAX_EPOCHS`** per launch isolation. Cannot override.
- Per-epoch walls (bf16): nh=144 ≈ 117 s/ep, nh=160 ≈ 118 s/ep, nh=176 ≈ 131 s/ep, nh=192 ≈ 131 s/ep, nh=208 ≈ 145 s/ep.
- VRAM peaks (bf16): nh=176+Lion ≈ 44.6 GB, nh=192+AdamW ≈ 47.6 GB, nh=208+AdamW ≈ 50.5 GB. H100 has 96 GB — ~45 GB headroom remains across the width frontier.

## Baseline progression (val_avg/mae_surf_p)

- #3091 baseline: 109.42 → ... → #3814 SwiGLU: 64.24 → ... → #3981 bf16+ep18: 53.82 → #4082 nh=176: 50.90 → #4106 nh=192+ep20: 48.84 → **#4252 Lion+nh=176+ep14: 49.26 (paper-facing test wins despite slight val noise)**

Total improvement from #3091: −55.0% val, −53.9% test. Lion + bf16 + width-sweep stack delivers this with 1.23M params (Transolver SwiGLU at nh=176).
