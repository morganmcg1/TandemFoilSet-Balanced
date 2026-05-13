# SENPAI Research State

- **Date**: 2026-05-13 05:05 (stochastic-depth + attention-dropout closed; diagnosis revised — convergence/budget-limited, NOT regularization-limited)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 30.4412** — PR #1794 (alphonse/torch-compile), merged 2026-05-13.

**-17.5% vs previous 36.8778 (bf16-amp baseline). Cumulative -74.0% from initial 117.17.**

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params), **SOAP** (`lr=1e-3, wd=1e-4`), **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, Huber(δ=0.1)+rel-L2 loss, **bf16 AMP**, **torch.compile(mode="default", dynamic=True)**. 30 epochs / 30 min. Peak GPU 24/96 GB.

Test avg 26.10 (all 4 splits). Model still descending at ep 30 (ep 30 was best).

---

## Current Research Focus — DIAGNOSIS REVISED

**OLD diagnosis (3 experiments)**: Model is regularization-limited, not capacity-limited.
**NEW diagnosis (5 experiments)**: Model is **convergence/budget-limited at the 30-min compute floor**, NOT regularization-limited.

**Refutation of the regularization-limited diagnosis**:
- **stochastic-depth** (#1897, +8.48%): EVERY split regressed, in-dist WORST. DropPath was net-harmful.
- **attention-dropout** (#1900, +0.47%): in-dist actually improved; OOD outcome mixed. Loss still descending at ep 29.

The student observation that crystallized it (#1900 tanjiro):
> "Loss curve was still trending down at epoch 29 — itself evidence the model is not regularization-limited — there was no train/val gap to close."

**Revised interpretation**: The wider/deeper/sharper-precond OOD regressions reflect **optimization fragility + compute-budget loss** (each ate epochs through extra per-step cost), NOT underfit regularization. With 1,499 training samples + monotone descending loss at cutoff, we are undertrained, not overfitted.

---

## Strategic Path Forward — Convergence-Limited Programme

**Active themes** (all aligned with "still descending at cutoff" diagnosis):
1. **Faster convergence**: OneCycleLR with higher peak LR (#1884 alphonse, in flight)
2. **Weight averaging**: EMA (#1917 frieren, in flight) — averages over the descending trajectory
3. **Loss-domain rebalancing**: lower surf_weight (tanjiro next)
4. **Stochastic weight averaging (SWA)**: averages last K epochs of cosine floor (thorfinn next)
5. **Compound win stacks**: rebases pending — surf-weight-30 (#1457), more-slices (#1467), re-conditioned-scaling (#1599), per-channel-loss-weights (#1614)

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #1457 | askeladd | `surf-weight-50` | WIP (v2) | MEDIUM | surf_weight=30; needs rebase to torch-compile base |
| #1467 | nezuko | `more-slices-128` | WIP | MEDIUM | slice_num=128; needs rebase |
| #1599 | fern | `re-conditioned-scaling` | WIP (v4) | **HIGH** | ReScale compound; re-rand-specific signal in dropout test supports this |
| #1614 | edward | `per-channel-loss-weights` | WIP | MEDIUM | p_weight=5; needs rebase |
| #1884 | alphonse | `onecycle-lr` | WIP | **HIGH** | OneCycleLR(max_lr=2e-3, pct_start=0.1); convergence pivot |
| #1917 | frieren | `ema-weights-v2` | WIP | **HIGH** | EMA β=0.999 with EMA-only val (fixes v1's +13% wall-clock penalty) |
| TBD | thorfinn | `swa-last-k` | NEW | **HIGH** | SWA over last 5 epochs at cosine floor |
| TBD | tanjiro | `surf-weight-7` | NEW | **HIGH** | Direct test of OOD-rc loss-rebalance hypothesis |

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
- **stochastic-depth** (PR #1897): +8.48% on ALL splits, in-dist WORST. Refutes regularization-limited diagnosis.
- **attention-dropout** (PR #1900): +0.47% (within noise); loss still descending at ep 29 → not regularization-limited. Confirms revised diagnosis.
- **ema-weights v1** (PR #1704): +5.9% — dual-val overhead cost 4 epochs. Mid-run signal confirmed (Δ=-11.7 at ep14). v2 #1917 fixes protocol.

## Potential Next Directions

**After current in-flight convergence experiments land**:
- **SWA + EMA compound** (if either wins): average over last K cosine-floor checkpoints
- **OneCycleLR + lower-surf-weight compound**: if both schedule and loss rebalance win
- **Re-conditioned input embedding** (fern's #1599): re_rand was the only positive outlier in dropout test → Re-specific regularization may be the right knob
- **Mixup/CutMix on input features**: data-domain augmentation (not model-domain regularization)
- **T_max sweep** (T_max=21, 35): probe whether 28 is optimal floor-reach epoch
- **Pseudo-replay / curriculum**: train on easier in-dist first, then OOD-augmented later
- **Coordinate-aware Re scaling**: inject log(Re) into PhysicsAttention slice weighting
- **Gradient accumulation** at effective batch=2 (more optimizer steps per epoch)

**The model is still converging at ep 30.** The torch.compile throughput gain is the most important lever found — every future experiment benefits from 30 epochs. Convergence-aware experiments are the current priority.
