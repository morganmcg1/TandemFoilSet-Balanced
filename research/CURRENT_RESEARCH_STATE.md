<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State — `icml-appendix-willow-pai2g-24h-r2`

- **Date / time:** 2026-05-12 20:00
- **Advisor branch:** `icml-appendix-willow-pai2g-24h-r2`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r2`
- **Most recent direction from human researcher team:** none (no inbound issues at this cycle).

## Current research focus

Round 2 of the 24h Charlie-vs-Willow logging ablation on TandemFoilSet, with a hard 30-min per-training-run wall-clock cap. Primary metric `val_avg/mae_surf_p`; the paper-facing comparator `test_avg/mae_surf_p` is structurally NaN on this branch (cruise test split overflows on every run including baseline) — decisions use val_avg + the three finite per-test-split metrics.

## Baseline (measured)

Two stock-config runs by alphonse (W&B `hqj9bt84`, `89653mip`): `val_avg/mae_surf_p` = **131.79 / 132.73** (~132, ±~0.5%). Single-run noise floor ~0.5–1%; a hypothesis needs to clear that to be a clean winner. See `BASELINE.md` for the full per-split breakdown.

## Live observations (W&B; not all PRs formally submitted)

Multiple students have arms finished in W&B but haven't all marked PRs as ready for review. Forward-looking signal:

| Student | Best W&B run | val_avg/mae_surf_p | Δ vs baseline | PR status |
|---|---|---|---|---|
| thorfinn (bf16+accum) | `opxuv6bq` | **118.17** | **-10.3%** | WIP — strong winner candidate |
| fern (lr=2e-3+clip) | `z8ub3am9` | 123.53 | -6.3% | WIP — winner candidate |
| askeladd (surf_weight=30) | `1uu93jzg` | 124.43 | -5.6% | WIP — winner candidate |
| frieren (p_weight=3) | `ph14bsim` | 130.98 | -0.9% | **sent back** — re-running with p_weight=2 + clip |
| tanjiro (per-field heads) | `bexja50y` | 131.74 | ~tie | WIP — likely no-op |
| alphonse (baseline) | `hqj9bt84` | 131.79 | (reference) | WIP — establishing reference |
| nezuko (wider 256/8h) | `cvi8ju7g` | 159.35 | +21% (worse) | WIP — capacity hurt under 30-min cap |
| edward (Huber persample) | `ct3iuldn` | 287.78 | +118% (worse) | WIP — major regression |

These are W&B observations; formal PR adjudication happens only when each student posts a terminal `SENPAI-RESULT` and marks ready for review.

## Themes that are working

- **Throughput hypothesis (thorfinn, bf16+accum)** is the current leader. More epochs in the 30-min budget directly translates into better val_avg. Implication: anything that reduces per-step wall time is potentially compoundable.
- **Higher LR with grad clip (fern)** also works strongly. Suggests the baseline LR is under-tuned for the 30-min regime.
- **Region prioritization (askeladd, surf_weight=30)** also pays off. The baseline `surf_weight=10` is genuinely too low for a 3%-surface-fraction mesh.

## Themes that are not working

- **Per-sample Huber normalization (edward)** is catastrophic (val_avg ~3× worse). The per-sample rescaling appears to destroy the learned scale relationships across Re. Will likely need to close this PR when submitted.
- **Wider model (nezuko, 256/8h)** is hurt by the 30-min cap — wider epochs take longer, fewer epochs fit, undertrained.
- **Per-field output heads (tanjiro)** is a wash. Architectural decoupling at the head doesn't help in this regime.
- **Intra-channel pressure upweight (frieren, p_weight=3)** is marginal and numerically risky.

## Next directions (round 3 candidates, conditional)

Held back from `RESEARCH_IDEAS_2026-05-12_initial.md`, prioritized by what we've now learned:

1. **Stack the throughput + LR + region winners.** thorfinn-style bf16+accum *with* fern-style lr=2e-3+clip *with* askeladd-style surf_weight=30 — three independent levers that should compose. High-value compound experiment for r3.
2. **Sweep `surf_weight`** more finely (20, 30, 40, 50) now that we know 30 helps. Find the saturation point.
3. **Higher LR + larger effective batch** combo (lr=1e-3 or 2e-3, accum=4, bf16). Tests whether the LR win is bottlenecked by step noise.
4. **Diagnose the cruise-test NaN at the source** (model output sanitization in `train.py` before `accumulate_batch`). One student in r3 should attempt to make every finished run report a finite `test_avg`. This unlocks paper-facing numbers.
5. **OneCycleLR with high peak (1e-3)** as a schedule alternative to bare cosine.
6. **Surface-aware sampling** — oversample batches with high surface-node fraction.
7. **EMA of weights** as a near-free regularizer.
8. **slice_num sweep** (32, 64, 128) — physics-attention granularity.

Held back as lower priority (post-r2):
- H7 (deeper + stochastic depth), H8 (SiLU), H10 (log1p dsdf), H13 (Re-Fourier position).

## Operational notes

- All 8 students currently have active WIP PRs (frieren just returned to WIP via send-back).
- Plateau protocol: not active — strong differentiated signal across hypotheses.
- Next polling priorities: thorfinn / fern / askeladd PRs marked ready for review (any of these is a likely merge); edward likely close; nezuko likely send-back to drop to default width but use a different angle.
