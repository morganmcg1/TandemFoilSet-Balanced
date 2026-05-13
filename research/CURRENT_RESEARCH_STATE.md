# SENPAI Research State

- **Date**: 2026-05-13 05:15 (re-conditioned-scaling merged -1.95%; new baseline 29.8463; convergence-limited diagnosis active)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 29.8463** — PR #1599 (fern/re-conditioned-scaling), merged 2026-05-13.

**-1.95% vs previous 30.4412 (torch-compile baseline). Cumulative -74.5% from initial 117.17.**

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params) **+ ReScaleHead** (163-param Re→scale head), **SOAP** (`lr=1e-3, wd=1e-4`), **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, Huber(δ=0.1)+rel-L2 loss, **bf16 AMP**, **torch.compile(mode="default", dynamic=True)**. 29 epochs / 30 min. Peak GPU 24/96 GB.

Per-split val: single_in_dist=30.20, rc=43.11, cruise=14.54, re_rand=31.54. Test avg 26.1005.

---

## Current Research Focus — Convergence/Budget-Limited Programme

**Diagnosis**: Model is **convergence/budget-limited at the 30-min compute floor**, NOT regularization-limited.
- Two regularization experiments (stochastic-depth +8.48%, attention-dropout +0.47%) both failed.
- Loss curves monotonically descend to the 30-min cutoff in every recent run — we're undertrained, not overfitted.
- Wider/deeper/sharper-precond OOD regressions reflect compute-budget loss, not overfitting.

**ReScaleHead is now the default baseline**: Separates Re-scale learning from shape learning. Mechanism confirmed in 3 runs: Uy/p channels show strong Re-correlation (0.86–0.94), Ux barely moves. Strong single_in_dist gain (-4.07) with mild OOD-rc regression.

**Active themes**:
1. **Faster convergence**: OneCycleLR with higher peak LR (#1884 alphonse, in flight)
2. **Weight averaging**: EMA-only val (#1917 frieren, in flight), SWA last-5-epochs (#1933 thorfinn, in flight)
3. **Loss-domain rebalancing**: surf_weight 10→7 (#1936 tanjiro, in flight)
4. **Compound win stacks**: rebases pending — surf-weight-30 (#1457 askeladd), more-slices (#1467 nezuko), per-channel-loss-weights (#1614 edward)
5. **ReScaleHead refinement**: 2-channel head (drop Ux; scale_std≈0.058 is nearly identity) — to be assigned to fern

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #1457 | askeladd | `surf-weight-50` | WIP | MEDIUM | surf_weight=30; rebasing onto new 29.8463 base |
| #1467 | nezuko | `more-slices-128` | WIP | MEDIUM | slice_num=128; rebasing onto new 29.8463 base |
| #1614 | edward | `per-channel-loss-weights` | WIP | MEDIUM | p_weight=5; rebasing onto new 29.8463 base (ReScaleHead included) |
| #1884 | alphonse | `onecycle-lr` | WIP | **HIGH** | OneCycleLR(max_lr=2e-3, pct_start=0.1); convergence pivot |
| #1917 | frieren | `ema-weights-v2` | WIP | **HIGH** | EMA β=0.999 with EMA-only val (fixes v1 protocol bug) |
| #1933 | thorfinn | `swa-last-k` | WIP (new) | **HIGH** | SWA over last 5 cosine-floor epochs |
| #1936 | tanjiro | `surf-weight-7` | WIP (new) | **HIGH** | Direct test of OOD-rc loss-rebalance hypothesis |
| TBD | fern | `rescale-head-2ch` | NEW | **HIGH** | Drop Ux channel from ReScaleHead (scale_std≈0.058 ≈ identity) |

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
| #1599 | fern | re-conditioned-scaling | 29.8463 | **−1.95%** | **−74.5%** |

## Ruled Out

- **warmup-cosine** (PR #1462): redundant with grad_clip
- **lr=1.5e-3 (AdamW)** (PR #1539): above AdamW LR ceiling
- **wider-deeper-3M** (PR #1458): epoch-limited under AdamW
- **SGDR T_0=7** (PR #1630 original): restart cost ~4 epochs
- **PCGrad gradient surgery** (PR #1579): mechanism confirmed but 1.63× wall-clock loses at 30-min budget
- **lr=2e-3 alone** (PR #1740): LR ceiling confirmed but grad_clip neutralizes
- **wider-soap-192** (PR #1797): data-bottlenecked; OOD regression
- **soap-relax-clip** (PR #1668): mechanism confirmed but slight regression; cosine already neutralizes clip binding
- **torch-compile reduce-overhead**: variable pad_collate shapes cause recompilation storms; mode="default" + dynamic=True required
- **larger-batch-compile** (PR #1847): training NOT compute-bound; half the optimizer steps at same wall-clock; regression +21.3%
- **soap-fp32-precond** (PR #1854): bf16 Q acts as implicit regularization; fp32 Q hurts OOD +4.3%
- **deeper-soap** (PR #1848): compute-budget loss at 30 min (21 vs 30 epochs); regression +11.6%
- **stochastic-depth** (PR #1897): +8.48% on ALL splits. Refutes regularization-limited diagnosis.
- **attention-dropout** (PR #1900): +0.47% (within noise); loss still descending at ep 29. Confirms convergence-limited diagnosis.
- **ema-weights v1** (PR #1704): +5.9% — dual-val overhead cost 4 epochs; v2 #1917 fixes protocol.

## Potential Next Directions

**After current in-flight convergence experiments land**:
- **SWA + EMA compound**: if either wins independently, combine them
- **OneCycleLR + ReScaleHead-2ch compound**: schedule + architecture refinement
- **FiLM-style Re conditioning**: inject log(Re) into PhysicsAttention slice weighting directly (instead of output rescaling)
- **Input feature augmentation**: ±5% Re noise during training → generalizes Re-specific scale head
- **Coordinate jitter**: ±std=0.001-0.01 on input spatial dims during training (data-domain augmentation)
- **T_max sweep** (T_max=21, 35): probe whether 28 is optimal floor-reach epoch given ReScaleHead inclusion
- **Gradient accumulation** at effective batch=2: more optimizer steps per epoch without memory increase
- **surf_weight sensitivity curve**: compound surf-weight-7 (tanjiro) + surf-weight-30 (askeladd) results to find optimal

**The model is still converging at ep 29-30.** ReScaleHead now in baseline — single_in_dist improved strongly. OOD-rc is the hardest remaining target (val 43.11, test 39.41). Focus: convergence speed + weight averaging + loss-domain rebalancing.
