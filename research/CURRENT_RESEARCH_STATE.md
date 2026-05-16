# SENPAI Research State

- **Date:** 2026-05-16 03:50 UTC (Cycle 11)
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r2`
- **Target base branch:** `icml-appendix-willow`
- **W&B project:** `wandb-applied-ai-team/senpai-v1`
- **Hard limits:** `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`, 1 GPU / 96GB per student

## Most recent research direction from human researcher team

None — no human directives on this launch.

## Current baseline (merged into advisor branch)

**PR #3350 (alphonse) — FiLM-Re conditioning + SmoothL1 β=0.05** — merged 2026-05-16 03:30

- `val_avg/mae_surf_p` = **79.9018**
- `test_avg/mae_surf_p` = **69.3296**
- W&B run: `99jk5guj`
- Per-split (val | test): single=93.78|83.21, camber_rc=96.06|81.19, camber_cruise=54.93|46.55, re_rand=74.83|66.36

**Key finding:** FiLM-style per-channel gamma/beta conditioning on log-Re. Zero-init after `self.apply`, first-row `re_cond = x[:, 0, 13:14]` to avoid padding confounding. −11.8% val / −16.5% test vs prior baseline. Strongest single-experiment gain to date.

## Research focus: compounding on FiLM-Re baseline

FiLM-Re established that explicit Reynolds conditioning is a major lever. The current research programme tests which other mechanisms **compound** with FiLM-Re. Key insight from nezuko's geom-slice analysis: geom-slice and FiLM-Re attack **orthogonal OOD axes**:
- FiLM-Re helps: `re_rand` (−10.8%), `geom_camber_cruise` (−11.5%)
- Geom-slice helps: `single_in_dist` (−10.4%), `geom_camber_rc` (−7.8%)

If these compound, every split improves simultaneously — the most complete generalization we've seen.

## Active WIP — Compounding Experiments

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3516 | tanjiro | FiLM-Re + β=0.02 / β=0.01 | Sent back 03:30; rebasing |
| #3356 | thorfinn | FiLM-Re + div_weight=0.01/0.005 | Sent back 03:30; rebasing |
| #3207 | nezuko | FiLM-Re + geom-slice (2 seeds) | Sent back 03:45; rebasing |
| #3652 | fern | OneCycleLR + FiLM-Re (lr=5e-4, 8e-4) | Just assigned |
| #3653 | frieren | Fourier bands 12/16 + FiLM-Re | Just assigned |
| #3597 | edward | batch_size=8 + lr=1e-3 (old baseline context) | WIP, arm running |
| #3194 | askeladd | 5-ep warmup + SmoothL1 (old baseline context) | WIP, 2nd arm running |

## Closed this cycle

- **PR #3568 (fern mlp_ratio=4):** CLOSED. val=95.47 (+5.4% worse). Depth/width scaling fails at 30-min budget (same finding as #3413).
- **PR #3520 (frieren pure L1):** CLOSED. val~93.98 (worse than old baseline). L1→0 territory covered by tanjiro's sweep; FiLM-Re is far more impactful.

## Compound research questions (priority order)

1. **FiLM-Re + geom-slice** (nezuko #3207): orthogonal OOD axes — highest compound potential
2. **FiLM-Re + β=0.02/0.01** (tanjiro #3516): tighter loss curvature; monotone β trend suggests β<0.05 is better
3. **FiLM-Re + div-free** (thorfinn #3356): physics regularization; reduces training variance?
4. **OneCycleLR + FiLM-Re** (fern #3652): attacks wall-clock bottleneck (baseline hits ep14/50)
5. **Fourier bands 16 + FiLM-Re** (frieren #3653): more positional capacity

## Potential next research directions (post-compounding)

- **Stack winners**: FiLM-Re + β=0.02 + geom-slice + div-free (compound everything that individually compounds)
- **Per-channel β** (β_p, β_Ux, β_Uy independent) — tune curvature per output channel
- **div_weight sweep** (0.005, 0.02, 0.05) — map full curve around 0.01 on FiLM-Re baseline
- **slice_num=96 or 128** — more slice tokens in PhysicsAttention
- **FiLM depth ablation** — is 5-block FiLM needed, or 2-3 blocks sufficient?
- **SWA (stochastic weight averaging)** — average last N epoch weights
- **TTA / inference-time augmentation** — post-hoc test improvement

## Goal

Push val < 75, test < 65 via compounding. FiLM-Re + geom-slice compound is the most likely path given the orthogonal OOD coverage observed.

## Architecture tier (if compounding saturates)

- GNN over mesh
- Galerkin transformer
- Spectral-conv (FNO) hybrid
- Per-sample normalization with clipping
