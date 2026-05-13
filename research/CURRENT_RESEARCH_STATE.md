# SENPAI Research State

- **Date**: 2026-05-13 04:30 (torch-compile merged -17.5%; wider-soap-192 and soap-relax-clip closed; 3 new assignments)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 30.4412** — PR #1794 (alphonse/torch-compile), merged 2026-05-13.

**-17.5% vs previous 36.8778 (bf16-amp baseline). Cumulative -74.0% from initial 117.17.**

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params), **SOAP** (`lr=1e-3, wd=1e-4`), **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, Huber(δ=0.1)+rel-L2 loss, **bf16 AMP**, **torch.compile(mode="default", dynamic=True)**. 30 epochs / 30 min. Peak GPU 24/96 GB.

Test avg 26.10 (all 4 splits). Model still descending at ep 30 (ep 30 was best).

**Convergence trace (val_avg/mae_surf_p)**: 172 → 161 → 136 → 106 → 88 → 79 → ... → 30.44 (ep 30 best; still descending).

---

## Current Research Focus

**Throughput unlock compounded massively**: torch.compile gave +76% more epochs (17→30) in the same 30-min budget, accounting for the entire -17.5% gain. Key lesson: the model is still epoch-limited (ep 30 was best, still descending). Every throughput and capacity lever compounds directly.

**Key constraints/levers remaining**:
- **Memory headroom**: 24/96 GB peak — 72 GB unused. Batch size 4 is extremely conservative.
- **Model still converging at ep 30**: need still more epochs OR the right capacity increase
- **Data bottleneck confirmed**: wider model (192) failed → 1,499 training samples limits representation width
- **Depth untested**: n_layers=7 is more data-efficient than width; assigned to tanjiro
- **Batch size**: batch=4 with 72 GB free; doubling to 8 costs ~4 fewer epochs but doubles gradient quality; assigned to alphonse
- **SOAP precision**: fp32 preconditioner quality under bf16 AMP; assigned to thorfinn
- **ReScale compound**: fern's re-conditioned-scaling (-4.7% on prior base) rebasing onto torch-compile base — could target ~25.5

**Active experiment priorities**:
1. **larger-batch-compile** (alphonse #1847, new): batch_size 4→8; 72 GB headroom; lowers gradient variance; T_max=23
2. **deeper-soap** (tanjiro #1848, new): n_layers 5→7 (880K params); depth is data-efficient; T_max~24
3. **soap-fp32-precond** (thorfinn #1854, new): keep GG/Q in fp32 under bf16 AMP; zero throughput cost
4. **re-conditioned-scaling** (fern #1599, WIP v4): rebase onto torch-compile base; -4.7% compound confirmed

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #1457 | askeladd | `surf-weight-50` | WIP (v2) | MEDIUM | surf_weight=30 on SOAP base; needs rebase to torch-compile base |
| #1467 | nezuko | `more-slices-128` | WIP | MEDIUM | slice_num=128 on SOAP base; needs rebase |
| #1599 | fern | `re-conditioned-scaling` | WIP (v4) | **HIGH** | ReScale compound confirmed (-4.7%), now rebasing onto torch-compile base |
| #1614 | edward | `per-channel-loss-weights` | WIP | MEDIUM | p_weight=5; orthogonal; needs rebase |
| #1704 | frieren | `ema-weights` | WIP | **HIGH** | EMA β=0.999; zero wall-clock cost; needs rebase |
| #1847 | alphonse | `larger-batch-compile` | WIP (new) | **HIGH** | batch_size 4→8; 24/96 GB → exploit 72 GB headroom |
| #1848 | tanjiro | `deeper-soap` | WIP (new) | **HIGH** | n_layers 5→7; depth > width at 1,499 training samples |
| #1854 | thorfinn | `soap-fp32-precond` | WIP (new) | **HIGH** | SOAP GG/Q fp32 under bf16 AMP; precision test |

All 8 students active.

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
| #1794 | alphonse | torch-compile | 30.4412 | **−17.5%** | **−74.0%** |

## Ruled Out

- **warmup-cosine** (PR #1462): redundant with grad_clip
- **lr=1.5e-3 (AdamW)** (PR #1539): above AdamW LR ceiling
- **wider-deeper-3M** (PR #1458): epoch-limited under AdamW; revisiting via depth now
- **SGDR T_0=7** (PR #1630 original): restart cost ~4 epochs
- **PCGrad gradient surgery** (PR #1579): mechanism confirmed but 1.63× wall-clock loses at 30-min budget
- **lr=2e-3 alone** (PR #1740): LR ceiling confirmed but grad_clip neutralizes; needs combined clip widening
- **wider-soap-192** (PR #1797): **data-bottlenecked** — 1,499 training samples can't fill wider representation. Width ruled out; depth (n_layers) being tested instead.
- **soap-relax-clip** (PR #1668): mechanism confirmed (clip_frac 0.33→0.00) but slight +0.93% regression. Cosine decay already neutralizes clip binding in late training.

## Potential Next Directions

**After current in-flight results land**:
- **Combined batch=8 + lr=2e-3**: if larger-batch-compile wins, try linear LR scaling with it
- **n_layers=9 (even deeper)**: if deeper-soap wins cleanly
- **OneCycleLR**: warmup → peak → decay; popular in competitions; not yet tried
- **Larger slice_num** (64→96): nezuko's in-flight experiment; more points per attention operation
- **surf_weight tuning** (askeladd): weight ∈ {20, 50} vs current 10
- **FNO spectral layer**: architectural alternative not tried; could help with the periodic pressure wake structures
- **Per-split surface weighting / curriculum**: explicit focus on hardest splits (rc, re_rand)
- **OneCycleLR with pct_start**: aggressive warmup to 2e-3 → cosine decay; could unlock higher LR transiently

**The model is still converging at every recent ep 30.** All compound paths remain open. The torch.compile throughput gain is the largest single improvement since SOAP — every subsequent experiment now benefits from 30 epochs instead of 17.
