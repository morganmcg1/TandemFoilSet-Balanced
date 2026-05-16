# SENPAI Research State

- **Date:** 2026-05-16 05:25 UTC (Cycle 15 — PLATEAU)
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

## PLATEAU STATUS — 10 compound experiments, ZERO beat baseline

The FiLM-Re baseline (val=79.90, test=69.33) has proven to be a strong attractor. None of the 10 finished compound experiments has beaten it. The closest miss is thorfinn FiLM-Re+div-free at val=80.86 (1.2% off), but it does not pass the threshold.

**Cycle 11-12 compound experiment results (all on FiLM-Re baseline):**

| Run | Student | Config | val_avg | test_avg | Δ val |
|---|---|---|---|---|---|
| `hukaxiix` | thorfinn | div_w=0.01 (1 of 2 arms) | **80.86** | 71.58 | +1.2% |
| `snjlp7xq` | alphonse | multi-signal cond_dim=5 | 82.46 | 72.73 | +3.2% |
| `m3u0225j` | tanjiro | β=0.02 | 83.99 | 78.71 | +5.1% |
| `hw2aksew` | nezuko | geom-slice seed 1 | 84.41 | 77.99 | +5.6% |
| `4bw2hrdu` | tanjiro | β=0.01 | 86.16 | 74.56 | +7.8% |
| `t60xj83c` | askeladd | surf_weight=5 | 86.78 | 77.00 | +8.6% |
| `4p8o19be` | fern | OneCycleLR lr=5e-4 | 88.76 | 82.61 | +11.1% |
| `pftv6no3` | thorfinn | div-free arm 2 | 95.46 | 76.40 | +19.5% |
| `e55dm25a` | frieren | Fourier bands=16 | 96.09 | 78.46 | +20.3% |
| `hpw0veo8` | edward | SWA (50% start) | 102.24 | 79.75 | +27.9% |

**Mechanistic conclusions:**
- **β decrease alone DOES NOT compound with FiLM-Re**: standalone β=0.02 gave val=88.11; β=0.02 on FiLM-Re gave val=83.99 — improvement is smaller and FiLM-Re's role dominates.
- **div-free physics loss is FRAGILE on FiLM-Re**: one seed val=80.86 (close), other val=95.46. High variance suggests interaction with FiLM-Re training dynamics.
- **geom-slice DOES NOT compound additively with FiLM-Re**: standalone val=85.60, with FiLM-Re val=84.41. Marginal improvement, much less than the orthogonal OOD analysis suggested.
- **Multi-signal FiLM is promising as REPLACEMENT, not addition**: cond_dim=5 at val=82.46 — second-best, may indicate the FiLM mechanism itself can be improved further.
- **SWA, OneCycleLR, more Fourier bands ALL HURT** at the 30-min wall-clock budget. These techniques need longer training to pay off.

## Plateau Protocol — Researcher Agent Dispatched (2026-05-16 05:25)

Per CLAUDE.md plateau protocol (10 failed experiments triggers escalation), the researcher-agent has been dispatched to explore:
1. Architecture-tier replacements (GNN, FNO, Galerkin, U-Net)
2. Loss formulation paradigms (percentile-weighted, focal, residual learning)
3. Data representation (per-domain normalization, log-space targets)
4. Self-supervised pretrain (masked reconstruction)
5. Inference-time techniques (TTA, ensemble)
6. Hyperparameter regimes not yet explored

Output expected at `/workspace/senpai/target/research/RESEARCH_IDEAS_2026-05-16_05:25.md`.

## Active WIP — Compounding Experiments

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3657 | alphonse | Multi-signal FiLM: cond_dim 1→5/9 (Re + geometry scalars) | Just assigned |
| #3516 | tanjiro | FiLM-Re + β=0.02 / β=0.01 | Sent back 03:30; rebasing |
| #3356 | thorfinn | FiLM-Re + div_weight=0.01/0.005 | Sent back 03:30; rebasing |
| #3207 | nezuko | FiLM-Re + geom-slice (2 seeds) | Sent back 03:45; rebasing |
| #3652 | fern | OneCycleLR + FiLM-Re (lr=5e-4, 8e-4) | Just assigned |
| #3653 | frieren | Fourier bands 12/16 + FiLM-Re | Just assigned |
| #3669 | edward | SWA on FiLM-Re (stochastic weight averaging) | Just assigned |
| #3670 | askeladd | surf_weight sweep {5,15,20} on FiLM-Re | Just assigned |

## Closed this cycle

- **PR #3597 (edward bs=8 + lr=1e-3):** CLOSED. val=94.08 (+3.8% vs old baseline). Batch/LR scaling fails at 30-min budget.
- **PR #3194 (askeladd warmup+cosine):** CLOSED. Best arm val=91.90 (fails old baseline by 1.4%). Warmup eats wall-clock budget without benefit.
- **PR #3568 (fern mlp_ratio=4):** CLOSED. val=95.47 (+5.4% worse). Depth/width scaling fails at 30-min budget.
- **PR #3520 (frieren pure L1):** CLOSED. val~93.98 (worse than old baseline). L1→0 territory covered by tanjiro's sweep.

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
