# SENPAI Research State — charlie-pai2g-48h-r5

- **As of:** 2026-05-13 05:00 (round-14: closed #1788 dropout loss, #1741 mlp_ratio loss; sent back #1774 alphonse lr=7.5e-4 for L1 rebase; assigned #1845 edward AdamW betas=(0.9,0.95) + #1846 frieren slice_num=32 on L1 baseline. Capacity axis triangulated CLOSED.)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r5` (advisor) — Charlie no-W&B logging ablation, round 5
- **Most recent human-team direction:** None on this branch; instructions scoped to the launch (no W&B, `SENPAI_TIMEOUT_MINUTES=30` cap per training execution).

## Current research focus

**5 merged winners → baseline 59.5354 L1 (-46% from 110.76 at round-1 start).** All PRs now compete against val_avg < 59.5354.

**Primary focus areas:**
1. **Optimizer levers on L1** — betas tuning (#1845 edward), LR peak retest (#1774 alphonse rebase).
2. **Known wins from β=0.5 rebasing to L1** — grad-clip (#1653 askeladd), WD=5e-5 (#1775 fern).
3. **Architecture inductive-bias** — slice_num=32 (#1846 frieren), schedule floor (#1826 thorfinn).
4. **Data/sampler** — nezuko #1619 sampler 2× (still WIP, may need L1 rebase).
5. **Loss weighting** — tanjiro #1789 surf_weight=15 (WIP, may need L1 rebase).

**Closed axes after round-14:**
- **Capacity (uniform width):** #1688 n_hidden=160 (+5.49%). CLOSED.
- **Capacity (FFN-only):** #1741 mlp_ratio=3 (+7.6%). CLOSED.
- **Attention dropout (per-weight):** #1788 p=0.1 (+2.75%). CLOSED at 30-min cap.
- **All other closed axes:** depth, batch=8, WD-up, warmup, per-channel weights, AdamW β2=0.95 (on β=0.5).

## Merged winners

| PR | Student | Hypothesis | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|---|---|
| #1700 ✓ | thorfinn | Pure L1 loss | **59.54** | **51.47** |
| #1633 ✓ | thorfinn | Huber β=0.5 | 64.07 | 55.50 |
| #1568 ✓ | thorfinn | torch.compile + bf16 | 69.83 | 61.87 |
| #1532 ✓ | thorfinn | bf16 AMP + scoring-NaN fix | 101.12 | 91.50 |
| #1444 ✓ | thorfinn | MSE → Huber β=1.0 | 110.76 | NaN |

**Current baseline: val_avg/mae_surf_p = 59.5354, test_avg/mae_surf_p = 51.4666 (PR #1700 L1)**

> Advisor config: Pure L1 + bf16 AMP + torch.compile(dynamic=True) + scoring-NaN workaround.
> ~37 epochs in 30 min at baseline. Peak GPU: ~24 GB. 96 GB available.

## In-flight (WIP)

| PR | Student | Hypothesis | Notes |
|---|---|---|---|
| #1653 | askeladd | Grad clip max_norm=1.0 on L1 — rebase from β=0.5 | β=0.5 result: −6.94% vs 64.07; need L1 rebase to compare vs 59.54 |
| #1775 | fern | WD=5e-5 on L1 — rebase from β=0.5 | β=0.5 result: −4.43% vs 64.07; likely stacks |
| #1826 | thorfinn | CosineAnnealingLR eta_min=5e-5 (LR floor) | New on L1 baseline |
| #1789 | tanjiro | surf_weight 10 → 15 | β=0.5-era assignment; may need L1 rebase if result >59.54 |
| #1619 | nezuko | Sampler 2× boost single_in_dist | β=0.5-era assignment; may need L1 rebase if result >59.54 |
| #1774 | alphonse | lr 5e-4 → 7.5e-4 — rebase to L1 | β=0.5 result: noise-bound (-1.2% val/+1.5% test); L1 landscape may respond differently |
| #1845 | edward | AdamW betas=(0.9, 0.95) on L1 | **New round-14 assignment** |
| #1846 | frieren | slice_num 64 → 32 on L1 | **New round-14 assignment** |

## Portfolio assessment

**High-confidence stacking candidates** (proven lever, just needs L1 rebase):
- Grad-clip (#1653 askeladd) — β=0.5 showed −6.94% val; if grad norms stay large under L1, likely stacks.
- WD=5e-5 (#1775 fern) — β=0.5 showed −4.43% val; parameter-level, orthogonal to loss shape.

**Medium-confidence new levers on L1:**
- AdamW betas=(0.9, 0.95) (#1845 edward) — L1 sign gradients change what preconditioner memory is optimal.
- slice_num=32 (#1846 frieren) — CFD spatial regimes argue 32 may be the natural bottleneck width.
- CosineAnnealingLR eta_min (#1826 thorfinn) — prevents LR collapse at epoch 37; should help budget-bound runs.

**Unknown outcome (need L1 rebase):**
- lr=7.5e-4 (#1774 alphonse) — noise-bound on β=0.5; L1 unit gradients may change this.
- surf_weight=15 (#1789 tanjiro) — reasonable if p-channel dominates residual budget.
- Sampler 2× (#1619 nezuko) — clear mechanism; performance on L1 unknown.

## Open questions / next experiments if slots open

1. **n_head sweep (4 → 8):** last untested architecture axis. Halves dim_head (16→8 per head) but doubles heads — same attention compute, different mixing pattern. Could expose more spatial modes.
2. **Lookahead optimizer wrapper:** slow-weights buffer that averages fast-weight trajectories. Known to help on noisy loss surfaces; L1's sign gradient is high-variance.
3. **Per-channel p-weighting in surf_loss:** currently uniform across Ux/Uy/p within the surface. Primary metric is p — could weight p column 2× or 3× within `sq_err * surf_mask`.
4. **Block-level DropPath (Stochastic Depth):** frieren's #1788 follow-up suggestion. Different from per-weight dropout — skips entire residual blocks with low prob; less impact on convergence speed.
5. **eps sweep (AdamW eps 1e-8 → 1e-7):** tiny change; on L1 with bounded gradients, denominator stability may interact differently with second-moment EMA.

## Constraints

- **Epoch budget:** ~37 epochs in 30 min at baseline config.
- **Memory:** ~24 GB peak; abundant headroom on 96 GB.
- **Compute:** All runs still budget-bound (best_epoch = terminal in recent experiments).
- **No W&B logging:** local JSONL only, enforce on every assignment.
