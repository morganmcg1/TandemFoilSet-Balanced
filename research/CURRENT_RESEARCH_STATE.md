# SENPAI Research State

- **Last updated:** 2026-05-16 02:35 (closed 4 round-6/5 negatives: #3579, #3569, #3564, #3535; round-8 assignments queued for 4 idle students)
- **Most recent research direction from human researcher team:** none (no open issues — verified 02:30Z).
- **Current best (merged):** `val_avg/mae_surf_p` = **95.808** (PR #3314 weight_decay=3e-4)
- **GH rate-limit status:** 719/5000 remaining (reset 02:19Z passed; healthy).

## Active PRs after this triage

| PR | Student | Hypothesis | State | Notes |
|----|---------|-----------|-------|-------|
| #3536 | tanjiro | eta_min=1e-5 (compound retest with wd=3e-4) | WIP (rebase pending) | val=95.835 in isolation, sent back to compound |
| #3606 | nezuko | RFF σ=3 (lower freq variant) | WIP | Round-7 |
| #3607 | thorfinn | dropout p=0.1 FFN | WIP | Round-7 |
| #3608 | frieren | SwiGLU activation (param-matched) | WIP | Round-7 |
| **(queued)** | alphonse | EMA / Polyak weight averaging | pending PR creation | lr=1e-3 closed; pivoting to optimizer-level axis |
| **(queued)** | askeladd | n_head=2 (head_dim=48) | pending PR creation | n_head=8 closed; opposite-direction probe |
| **(queued)** | edward | surf_weight=10→5 (loss rebalance) | pending PR creation | n_layers=4 closed; capacity exhausted, loss axis next |
| **(queued)** | fern | stochastic depth / drop-path p=0.1 | pending PR creation | wd=5e-4 closed; orthogonal stochastic regularization |

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`. Local JSONL metrics only.

## Established baseline stack (merged to HEAD)
1. **PR #3208** (Huber loss) — `val_avg/mae_surf_p` 116.61
2. **PR #3276** (grad-clip + AdamW selective decay + NaN guard) — 109.68
3. **PR #3294** (warmup+cosine 14ep, lr=7e-4) — 100.811
4. **PR #3399** (slice_num=64→96) — 97.757
5. **PR #3377** (n_hidden=128→96) — 96.667
6. **PR #3314** (weight_decay=1e-4→3e-4) — **95.808** (current baseline)

Key config: SmoothL1 (Huber β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (wd=3e-4) + NaN guard + SequentialLR (LinearLR 2ep warmup + CosineAnnealingLR T_max=12, eta_min=0) + lr=7e-4 + epochs=14 + slice_num=96 + n_hidden=96 + n_head=4 + mlp_ratio=2 + n_layers=5 + GELU + dropout=0.0 + surf_weight=10.

## Round-6/5 closure summary (this batch)

| PR | Student | Hypothesis | val Δ | Verdict |
|----|---------|-----------|------|---------|
| #3579 | alphonse | lr 7e-4 → 1e-3 | +2.47% | CLOSED — peak-LR upper-edge confirmed |
| #3569 | fern | wd 3e-4 → 5e-4 | +1.30% | CLOSED — WD ladder bracketed; OOD/in-dist trade |
| #3564 | edward | n_layers 5 → 4 | +2.71% | CLOSED — depth=5 sweet spot from below |
| #3535 | askeladd | n_head 4 → 8 (head_dim=12) | +22.9% | CLOSED — head_dim collapse, Pareto loss |

## Plateau watch — ESCALATED

Best is **95.808 (PR #3314)**. Total improvement: −17.9% from original 116.61 Huber baseline.

**Negative streak: 0 winners + 8 negatives + 1 marginal sent-back across rounds 5-6.** Classical hyper-param sweep axes are now exhausted:

| Axis | Status | Evidence |
|------|--------|----------|
| Width | DEAD | 64 (+9.2%) / 96 (sweet) / 128 (+1.1%) / 192 (+8.5%) |
| Depth | DEAD | 4 (+2.71%) / 5 (sweet) / 6 (+15.0%) / 8 (+1.5%) |
| mlp_ratio | DEAD | 2 (sweet) / 4 (+5.0%) |
| n_head | DEAD upward; n_head=2 last probe | 4 (sweet) / 8 (+22.9%, head_dim collapse) |
| Peak LR | DEAD | 7e-4 (sweet) / 1e-3 (+2.47%) |
| WD | DEAD as global knob | 1e-4 / **3e-4 (sweet)** / 5e-4 (+1.30%) |
| T_max | DEAD | 10 (+3.55%) / 12 (sweet) |
| RFF σ=10 | DEAD; σ=3 last probe | (+2.89%, mesh-overfit) |
| Per-channel loss [1,1,2] | DEAD | (+2.85%) |

**Surviving frontier (the only axes still untested or with positive momentum):**
- **Schedule tail (eta_min)** — tanjiro #3536 compound retest pending; in isolation already val 95.835.
- **Stochastic regularization** — thorfinn #3607 (FFN dropout) in flight; fern (drop-path) queued.
- **Activation function** — frieren #3608 (SwiGLU) in flight.
- **Positional encoding (low-freq)** — nezuko #3606 (RFF σ=3) in flight.
- **Optimizer-level tricks** — alphonse (EMA/Polyak) queued; later: Lookahead, SAM.
- **Loss rebalance** — edward (surf_weight=5) queued; later: log-pressure target.
- **Head allocation downward** — askeladd (n_head=2) queued.

## Confirmed design insights

### Single_in_dist is the bottleneck
- Largest absolute val (110.886 baseline). Every regularization-up experiment regresses single_in_dist (wd=5e-4: +8.53%) while improving cruise/re_rand.
- **Implication:** single_in_dist needs more capacity-to-memorize, not more regularization. surf_weight=5 (edward) and EMA (alphonse) directly attack this.

### Regularization mechanism is asymmetric
- WD helps OOD splits, hurts in-dist memorization. The "right" regularization for this dataset must respect this asymmetry: either per-layer-class WD, or stochastic methods (dropout / drop-path) that target activations rather than weight magnitudes.

### Schedule tail (eta_min > 0) is real but already half-tested
- tanjiro #3536 in isolation: val 95.835 (−0.86% vs OLD baseline). Compound retest on new stack pending — if it lands at 95.7-95.5, it's a clean win.

## Open questions
- Does eta_min=1e-5 compound with wd=3e-4? (tanjiro rebase pending)
- Does RFF σ=3 reverse the σ=10 failure? (nezuko #3606)
- Does FFN dropout regularize productively? (thorfinn #3607)
- Does SwiGLU improve over GELU? (frieren #3608)
- Does EMA / Polyak averaging smooth the late cosine tail? (alphonse queued)
- Does n_head=2 (head_dim=48) reverse the n_head=8 collapse? (askeladd queued)
- Does surf_weight=5 let single_in_dist memorize more? (edward queued)
- Does drop-path p=0.1 regularize at the block level? (fern queued)

## Closed/regressed (cumulative)
- #3579 alphonse lr=1e-3: +2.47% (peak-LR upper edge)
- #3569 fern wd=5e-4: +1.30% (WD ladder reversed)
- #3564 edward n_layers=4: +2.71% (depth sweet spot from below)
- #3535 askeladd n_head=8: +22.9% (head_dim collapse)
- #3502 alphonse n_hidden=64: +9.20%
- #3503 edward mlp_ratio=4: +5.02%
- #3505 frieren per-channel pressure [1,1,2]: +2.85%
- #3506 thorfinn n_layers=6: +15.0%
- #3534 nezuko RFF σ=10: +2.89%
- #3453 edward T_max=10: +3.55%
- #3301 alphonse width-192 (rebased): +8.53%
- #3304 frieren surf_weight=20 (rebased): +4.12%
- #3302 askeladd depth-8: +1.53%
- #3223 thorfinn BF16+batch=8: +34%
- #3295 edward slice_num=128: +20%
- Round-1: #3205, #3179, #3183, #3214, #3216, #3220

## Next research directions (post round-7/8)

If round-7 (nezuko/thorfinn/frieren) and round-8 (alphonse/askeladd/edward/fern) all fail to beat 95.808, escalate to plateau-protocol tier-3:

1. **Log-pressure target** — handles dynamic range across splits; reformulate target space.
2. **Best-checkpoint test evaluation** — paper-facing improvement, low risk.
3. **Per-layer-class weight decay** — decouple OOD/in-dist trade revealed by fern's wd=5e-4 closure.
4. **Attention dropout** — separate from FFN dropout, attacks attention overfit.
5. **WSD (Warmup-Stable-Decay)** — schedule reformulation if eta_min compound fails.
6. **Data augmentation** — physics-aware flip or geometric perturbation.
7. **Lookahead wrapping AdamW** — orthogonal optimizer-level technique.
8. **SAM (Sharpness-Aware Minimization)** — explicit flatness penalty.
9. **Alternative architectures** — completely new models, not architecture tweaks (per plateau protocol).
