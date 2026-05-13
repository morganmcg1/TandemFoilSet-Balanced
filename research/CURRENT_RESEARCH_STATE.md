# SENPAI Research State

- **Date**: 2026-05-13 04:45 (soap-fp32-precond + deeper-soap closed; regularization sweep launched)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 30.4412** — PR #1794 (alphonse/torch-compile), merged 2026-05-13.

**-17.5% vs previous 36.8778 (bf16-amp baseline). Cumulative -74.0% from initial 117.17.**

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params), **SOAP** (`lr=1e-3, wd=1e-4`), **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, Huber(δ=0.1)+rel-L2 loss, **bf16 AMP**, **torch.compile(mode="default", dynamic=True)**. 30 epochs / 30 min. Peak GPU 24/96 GB.

Test avg 26.10 (all 4 splits). Model still descending at ep 30 (ep 30 was best).

---

## Current Research Focus

**CRITICAL FINDING: Model is regularization-limited, not capacity-limited or optimization-limited.**

Three consecutive experiments confirm this with the same OOD-degradation signature:
1. **wider-soap-192** (+33%): OOD splits hurt worst by extra width  
2. **soap-fp32-precond** (+4.3%): sharper fp32 preconditioner → OOD worse, in-dist better  
3. **deeper-soap** (+11.6%): deeper model undertrains at 30-min budget (uniform regression)

**ALSO CONFIRMED** (batch-bottleneck):
4. **larger-batch-compile** (+21.3%): halving optimizer steps hurts; training NOT compute-bound at batch=4

**The interpretation**: 1,499 training samples create two constraints simultaneously:
- **Data bottleneck**: Model memorizes training set if given extra capacity (width, depth) or extra precision (fp32 Q)
- **Compute bottleneck**: Larger models cost more per epoch and get fewer epochs in 30 min

The 662K / 5-layer model is in the **sweet spot** — it gets 30 epochs in 30 min AND doesn't overfit aggressively. But it still overfits: the OOD gap is real (single_in_dist val~34 but rc val~41, re_rand val~32 with test~22).

**Current research focus: regularization sweep at current capacity.**

**OneCycleLR** (alphonse, #1884) simultaneously addresses the schedule question: can a warmup → 2e-3 peak → cosine decay unlock a higher effective LR without cold-start instability?

**Stochastic Depth** (thorfinn, #1897): DropPath drop_path_max=0.1 across 5 layers. Structural regularization — randomly skip blocks, forcing each layer to work independently. Should especially help OOD splits.

**Attention Dropout** (tanjiro, #1900): Enable existing dropout=0.1 in PhysicsAttention output projection (already wired, currently 0.0). Cheapest regularization test.

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #1457 | askeladd | `surf-weight-50` | WIP (v2) | MEDIUM | surf_weight=30; needs rebase to torch-compile base |
| #1467 | nezuko | `more-slices-128` | WIP | MEDIUM | slice_num=128; needs rebase |
| #1599 | fern | `re-conditioned-scaling` | WIP (v4) | **HIGH** | ReScale compound confirmed (-4.7%), rebasing onto torch-compile base |
| #1614 | edward | `per-channel-loss-weights` | WIP | MEDIUM | p_weight=5; needs rebase |
| #1704 | frieren | `ema-weights` | WIP | **HIGH** | EMA β=0.999; zero wall-clock cost; needs rebase |
| #1884 | alphonse | `onecycle-lr` | WIP (new) | **HIGH** | OneCycleLR(max_lr=2e-3, pct_start=0.1); per-batch scheduler.step() |
| #1897 | thorfinn | `stochastic-depth` | WIP (new) | **HIGH** | DropPath drop_path_max=0.1 across 5 layers |
| #1900 | tanjiro | `attention-dropout` | WIP (new) | **HIGH** | dropout=0.1 in PhysicsAttention output |

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

## Potential Next Directions

**After current in-flight regularization results land**:
- **Compound dropout + stochastic-depth**: if either wins independently, combine them
- **Label smoothing on surf_weight**: smooth the surface vs volume weight boundary
- **OneCycleLR + dropout**: combine schedule improvement with regularization if both win
- **FiLM-style Re conditioning** (from fern's analysis): inject log(Re) into Transolver preprocess MLP as learned bias instead of output rescaling
- **Input feature augmentation**: ±5% Re noise during training, or geometry mirroring for symmetric foil cases
- **SWA (Stochastic Weight Averaging)**: average checkpoints from last K epochs; complements EMA-weights

**The model is still converging at ep 30.** The torch.compile throughput gain is the most important lever found — every future experiment benefits from 30 epochs. Regularization sweep is the current priority.
