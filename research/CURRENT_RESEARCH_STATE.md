# SENPAI Research State

- **Date**: 2026-05-13 02:05 (bf16-amp merged, soap-higher-lr closed, 2 new assignments)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 36.8778** — PR #1456 (alphonse/bf16-amp + cosine-eta-min), merged 2026-05-13.

**-7.51% vs previous 39.8693 (cosine-eta-min baseline).**

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params), **SOAP** (`lr=1e-3, wd=1e-4`), **`CosineAnnealingLR(T_max=17, eta_min=1e-5)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, Huber(δ=0.1)+rel-L2 loss, **bf16 AMP**. 17 epochs / 30 min. Peak GPU 33/96 GB.

Test avg 31.91 (all 4 splits).

**Convergence trace**: 172 → 161 → 136 → 106 → 88 → 79 → 77 → 72 → 62 → 59 → 53 → 51 → 44 → 40 → 38 → **37** → 37 (ep 16 best; still descending until LR floor).

---

## Current Research Focus

**Compound wins on bf16 + SOAP + cosine-eta-min stack.** Model is still converging at ep 16 — every throughput/capacity boost compounds. Memory headroom (33/96 GB) is now substantial → next wave of experiments exploits it via larger batch, larger model, or torch.compile throughput.

**Key constraints/levers remaining**:
- clip_frac=0.34 at ep 17 → grad_clip becoming less binding in cosine tail. Mostly resolved by cosine schedule.
- **Throughput**: torch.compile is the next +20-30% lever (alphonse #1794)
- **Capacity**: 662K params on 96 GB GPU is wasteful; n_hidden 128→192 = 1.5M params (tanjiro #1797)
- **LR/clip coupling**: thorfinn's soap-relax-clip (#1668) tests clip widening; tanjiro's lr=2e-3 (closed) confirmed LR ceiling is ≥2e-3 but clip-blocked
- **Batch size**: still 4; could be 8-16 with current memory headroom — not yet assigned

**Four highest-priority running experiments**:
1. **torch-compile** (alphonse #1794, new): +20-30% throughput → 20+ epochs/run
2. **wider-soap-192** (tanjiro #1797, new): n_hidden 128→192 = 1.5M params; capacity unlock
3. **soap-relax-clip** (thorfinn #1668): grad_clip 1.0→5.0; unlocks step magnitude
4. **ema-weights** (frieren #1704): EMA β=0.999 of SOAP weights; zero wall-clock cost

**Per-split profile at new baseline**:
- cruise (val 18.60 / test 15.26) — near-saturating but still improving
- re_rand (val 38.21 / test 27.53) — strong improvement
- rc (val 47.78 / test 42.69) — hardest OOD split, still room
- single_in_dist (val 42.92 / test 42.15) — hardest to improve consistently

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #1457 | askeladd | `surf-weight-50` | WIP (v2) | MEDIUM | surf_weight=30 on SOAP base; needs rebase to bf16 base |
| #1467 | nezuko | `more-slices-128` | WIP | MEDIUM | slice_num=128 on SOAP base; needs rebase |
| #1599 | fern | `re-conditioned-scaling` | WIP (rebasing) | HIGH | ReScaleHead; SOAP compound test; still training |
| #1614 | edward | `per-channel-loss-weights` | WIP | MEDIUM | p_weight=5 on SOAP base; orthogonal |
| #1668 | thorfinn | `soap-relax-clip` | WIP | **HIGH** | grad_clip 1.0→5.0; unlocks SOAP step magnitude |
| #1704 | frieren | `ema-weights` | WIP | **HIGH** | EMA β=0.999 of SOAP weights; zero wall-clock cost |
| #1794 | alphonse | `torch-compile` | WIP (new) | **HIGH** | +20-30% throughput on bf16 base; aim 20+ epochs/run |
| #1797 | tanjiro | `wider-soap-192` | WIP (new) | **HIGH** | n_hidden 128→192 = 1.5M params; capacity unlock |

All 8 students active. New baseline broadcast to all 6 prior WIP PRs.

---

## Merged Winners (chronological)

| PR | Student | Slug | val_avg | Delta | Cumulative |
|----|---------|------|---------|-------|------------|
| #1479 | thorfinn | grad-clip-1 | 117.17 | — | baseline |
| #1518 | thorfinn | higher-lr-cosine-14 | 96.5587 | −17.6% | −17.6% |
| #1460 | fern | relative-l2-loss | 89.6121 | −7.2% | −23.5% |
| #1473 | tanjiro | huber-loss | 89.3940 | −0.24% | −23.7% |
| #1613 | thorfinn | soap-optimizer | 42.4015 | **−52.6%** | **−63.8%** |
| #1630 | tanjiro | cosine-eta-min | 39.8693 | −5.97% | −66.0% |
| #1456 | alphonse | bf16-amp + T_max=17 | 36.8778 | **−7.51%** | **−68.6%** |

## Ruled Out

- **warmup-cosine** (PR #1462): redundant with grad_clip
- **lr=1.5e-3 (AdamW)** (PR #1539): above AdamW LR ceiling
- **wider-deeper-3M** (PR #1458): epoch-limited under AdamW; revisiting capacity now under SOAP+bf16
- **SGDR T_0=7** (PR #1630 original): restart cost ~4 epochs
- **PCGrad gradient surgery** (PR #1579): mechanism confirmed but 1.63× wall-clock loses at 30-min budget
- **lr=2e-3 alone** (PR #1740): LR ceiling confirmed but grad_clip neutralizes; needs combined clip widening

## Potential Next Directions

**After current in-flight results land**:
- **Larger batch** (8 or 16): substantial memory headroom; could lower gradient variance
- **Combined lr=2e-3 + clip=2-3**: natural follow-up if thorfinn's soap-relax-clip wins
- **Even wider model** (n_hidden=256, ~2.7M): if wider-soap-192 wins cleanly
- **OneCycleLR**: warmup → peak → decay; popular in competitions
- **FNO spectral layer**: architectural alternative not yet tried
- **Per-split surface weighting**: explicit curriculum for single_in_dist & rc (hardest splits)

**The model is still converging at every recent ep 16-17.** All compound paths remain open.
