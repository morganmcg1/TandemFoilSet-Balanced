# SENPAI Research State

- **Last updated:** 2026-05-16 01:45 (round-5 review: 1 sent-back compound test + 3 closures; round-7 assignments queued pending GH rate-limit reset at 02:19Z)
- **Most recent research direction from human researcher team:** none (no open issues — verified 00:45Z).
- **Current best (merged):** `val_avg/mae_surf_p` = **95.808** (PR #3314 weight_decay=3e-4)
- **GH rate-limit status:** Fully exhausted at 01:38Z (0/5000 remaining); resets at ~02:19Z. Local doc updates and hypothesis prep complete; PR creation paused.

## Active PRs after this triage

| PR | Student | Hypothesis | State | Notes |
|----|---------|-----------|-------|-------|
| #3579 | alphonse | lr=1e-3 higher peak LR | WIP | Round-6 |
| #3564 | edward | n_layers=4 depth bracket | WIP | Round-6 |
| #3569 | fern | weight_decay=5e-4 WD ladder | WIP | Round-6 |
| #3535 | askeladd | n_head=8 | WIP | Round-5 still in-flight |
| #3536 | tanjiro | eta_min=1e-5 (compound retest) | WIP (rebase pending) | val=95.835, sent back for wd=3e-4+eta_min compound test |
| **(queued)** | nezuko | RFF σ=3 (lower freq variant) | pending PR creation | RFF σ=10 closed |
| **(queued)** | thorfinn | dropout p=0.1 FFN | pending PR creation | n_layers=6 closed; capacity axis exhausted |
| **(queued)** | frieren | SwiGLU activation (param-matched) | pending PR creation | per-channel loss closed |

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`. Local JSONL metrics only.

## Established baseline stack (merged to HEAD)
1. **PR #3208** (Huber loss) — `val_avg/mae_surf_p` 116.61
2. **PR #3276** (grad-clip + AdamW selective decay + NaN guard) — 109.68
3. **PR #3294** (warmup+cosine 14ep, lr=7e-4) — 100.811
4. **PR #3399** (slice_num=64→96) — 97.757
5. **PR #3377** (n_hidden=128→96) — 96.667
6. **PR #3314** (weight_decay=1e-4→3e-4) — **95.808** (current baseline)

Key config: SmoothL1 (Huber β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (wd=3e-4) + NaN guard + SequentialLR (LinearLR 2ep warmup + CosineAnnealingLR T_max=12, eta_min=0) + lr=7e-4 + epochs=14 + slice_num=96 + n_hidden=96 + n_head=4 + mlp_ratio=2 + n_layers=5 + GELU + dropout=0.0.

## Round-5 result summary

| PR | Student | Hypothesis | val Δ | Verdict |
|----|---------|-----------|------|---------|
| #3536 | tanjiro | eta_min=1e-5 | −0.86% vs OLD; +0.03% vs NEW | SENT BACK (compound retest with wd=3e-4) |
| #3534 | nezuko | RFF σ=10 | +2.89% | CLOSED — σ=10 mesh-overfit; σ=3 follow-up |
| #3506 | thorfinn | n_layers=6 | +15.0% | CLOSED — capacity axis decisively dead |
| #3505 | frieren | per-channel loss [1,1,2] | +2.85% | CLOSED — physics-coupling + LR confound |
| #3502 | alphonse | n_hidden=64 | +9.20% | CLOSED — width-96 sweet spot confirmed |

## Confirmed design insights

### Schedule is the dominant lever
- Budget-matched warmup+cosine (14ep, lr=7e-4) → −8.08% (PR #3294)
- T_max smaller than budget hurts (PR #3453 closed — annealing too aggressive)
- Eta_min raise WORKS (#3536 val −0.86% vs OLD baseline) — compound retest pending
- lr axis untested on new stack — alphonse #3579 testing lr=1e-3

### Capacity axis — DECISIVELY EXHAUSTED
- Width ladder: 64 (+9.2%) → 96 (sweet) ← 128 (+1.1%) ← 192 (+8.5%): NON-MONOTONE around 96
- Depth: 5 (sweet) → 6 (+15.0%) — depth axis dead above 5
- mlp_ratio: 2 (sweet) → 4 (+5.0%) — FFN expansion dead
- edward #3564 (n_layers=4) is the last open capacity probe; expectation is depth=5 also sweet from below

### Regularization — productive axis
- weight_decay=3e-4 > 1e-4: −0.89% (merged, single_in_dist −4.96% concentrated)
- WD ladder pending: fern #3569 tests 5e-4
- **dropout untested** — queued for thorfinn
- Net pattern: more regularization helps single_in_dist (hardest), regresses cruise (easiest)

### Architecture axes — orthogonal probes
- n_head=4→8 (askeladd #3535) still in-flight
- **Activation function untested** — queued for frieren (SwiGLU)
- **Positional encoding axis partially probed** — RFF σ=10 failed; σ=3 queued for nezuko

## Open questions
- Does lr=1e-3 outperform 7e-4 on new stack? (alphonse #3579)
- Does eta_min=1e-5 compound with wd=3e-4? (tanjiro rebase pending)
- Does WD=5e-4 continue the ladder? (fern #3569)
- Does depth=4 bracket confirm depth=5? (edward #3564)
- Does n_head=8 help? (askeladd #3535)
- Does RFF σ=3 reverse the σ=10 failure? (nezuko queued)
- Does FFN dropout regularize productively? (thorfinn queued)
- Does SwiGLU improve over GELU? (frieren queued)

## Closed/regressed (complete list)
- #3502 alphonse n_hidden=64: +9.20% (width-96 confirmed sweet spot)
- #3503 edward mlp_ratio=4: +5.02% (capacity overfit)
- #3505 frieren per-channel pressure [1,1,2]: +2.85% (physics coupling)
- #3506 thorfinn n_layers=6: +15.0% (depth axis dead)
- #3534 nezuko RFF σ=10: +2.89% (high-freq mesh overfit)
- #3453 edward T_max=10: +3.55% (annealing too aggressive)
- #3301 alphonse width-192 (rebased): +8.53% (over-budget)
- #3304 frieren surf_weight=20 (rebased): +4.12% (lever absorbed)
- #3302 askeladd depth-8: +1.53% (budget-bound on old stack)
- #3223 thorfinn BF16+batch=8: +34% (padding overhead)
- #3295 edward slice_num=128: +20% (budget-mismatch)
- Round-1: #3205, #3179, #3183, #3214, #3216, #3220

## Plateau watch
Best is 95.808 (PR #3314, merged). Total improvement: −17.9% from original 116.61 Huber baseline.
- **5 negatives + 1 marginal in this round-5 review batch**. Pattern: classical hyper-param sweep axes exhausted.
- **The most productive remaining levers are non-capacity**: regularization (dropout), activation (SwiGLU), schedule tail (eta_min compound), positional encoding (RFF σ-sweep).
- If round-6/7 also fails to beat 95.808, escalate to plateau protocol — consider loss reformulation (log-pressure), data augmentation (physics-aware), or alternative architectures.

## Next research directions (post round-6/7)
1. **Lookahead optimizer wrapping AdamW** — orthogonal optimizer technique
2. **Log-pressure target** — handles dynamic range across splits
3. **Best-checkpoint test evaluation** — paper-facing improvement, low risk
4. **Attention dropout** — separate from FFN dropout, attacks attention overfit
5. **Per-layer-class weight decay** — decouple cruise/single trade-off
6. **Data augmentation** — physics-aware flip or geometric perturbation
