# SENPAI Research State

- **Last updated:** 2026-05-16 01:45 (closed alphonse #3502 width-64; assigned lr=1e-3 to alphonse; commented rebase on thorfinn #3506)
- **Most recent research direction from human researcher team:** none (no open issues — verified 01:45Z).
- **Current best (merged):** `val_avg/mae_surf_p` = **95.808** (PR #3314 weight_decay=3e-4 on full stack)
- **Round-5/6 in flight (8 students):**
  - alphonse #3579: lr=1e-3 higher peak LR (round-6, newly assigned)
  - edward #3564: n_layers=4 depth bracket below baseline (round-6)
  - fern #3569: weight_decay=5e-4 WD ladder (round-6)
  - frieren #3505: per-channel pressure loss [1,1,2] (round-5)
  - thorfinn #3506: n_layers=6 depth above baseline (round-5, needs rebase to new HEAD — notified)
  - nezuko #3534: RFF 32-freq (round-5 reassignment)
  - askeladd #3535: n_head=8 (round-5 reassignment)
  - tanjiro #3536: eta_min=1e-5 (round-5 reassignment)

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`. Local JSONL metrics only.

## Established baseline stack (merged to HEAD)
1. **PR #3208** (Huber loss) — `val_avg/mae_surf_p` 116.61
2. **PR #3276** (grad-clip + AdamW selective decay + NaN guard) — 109.68
3. **PR #3294** (warmup+cosine 14ep, lr=7e-4) — 100.811
4. **PR #3399** (slice_num=96) — 97.757
5. **PR #3377** (n_hidden=128→96) — 96.667
6. **PR #3314** (weight_decay=1e-4→3e-4) — **95.808** (current baseline)

Key config: SmoothL1 (Huber β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (**wd=3e-4**) + NaN guard + SequentialLR (LinearLR 2ep warmup + CosineAnnealingLR T_max=12, eta_min=0) + lr=7e-4 + epochs=14 + slice_num=96 + **n_hidden=96** + n_head=4 + mlp_ratio=2 + n_layers=5.

## Active PRs (8 students, all assigned)

| PR | Student | Hypothesis | Round | Notes |
|----|---------|-----------|-------|-------|
| #3579 | alphonse | lr=1e-3 higher peak LR | 6 | Freshly assigned; loss-still-descending-at-E14 motivates higher lr |
| #3564 | edward | n_layers=4 depth bracket | 6 | Bracket {4,5,6} with thorfinn #3506 |
| #3569 | fern | weight_decay=5e-4 WD ladder | 6 | Continue WD monotone: 1e-4→3e-4→5e-4? |
| #3505 | frieren | per-channel pressure loss [1,1,2] | 5 | Direct test of pressure under-fit on single_in_dist |
| #3506 | thorfinn | n_layers=6 depth above baseline | 5 | Needs rebase to new HEAD (notified); pod auto-pulled |
| #3534 | nezuko | RFF 32-freq | 5 | Positional encoding; targeted at single_in_dist |
| #3535 | askeladd | n_head=8 | 5 | Attention diversity; untested on optimised stack |
| #3536 | tanjiro | eta_min=1e-5 | 5 | LR floor raise; counterpart to closed T_max=10 |

## Confirmed design insights

### Schedule is the dominant lever
- Budget-matched warmup+cosine (14ep, lr=7e-4) → −8.08% (PR #3294)
- T_max smaller than budget hurts (PR #3453 closed)
- Eta_min raise is the natural next probe (tanjiro #3536 in-flight)
- **lr axis: not yet re-tested on new full stack** (alphonse round-6, lr=1e-3)

### Slot-count is the second lever
- slice_num=64 → 96 (PR #3399): −3.03%
- slice_num=128: +20% regression (closed)
- slice_num=96 the sweet spot

### Width axis — LOCKED at n_hidden=96
- Width ladder: {64: 104.6, 96: 95.8, 128: 97.8, 192: 106.1}
- n_hidden=96 is the sweet spot — both lower and higher hurt
- Per-epoch wall-clock insensitive to n_hidden (op cost dominated by data/slicing/optimizer)
- Width axis exhausted; DO NOT revisit

### Regularization — weight decay helps monotonically so far
- weight_decay=3e-4 > 1e-4: −0.89% net (PR #3314, merged)
- Pattern: improvement concentrated on single_in_dist; cruise regresses slightly
- WD ladder not saturated — 5e-4 next probe (fern round-6)

### Capacity axis — NOT the bottleneck
- mlp_ratio=4: +5.02% regression — FFN expansion overfit uniformly
- width-192: +8.53%; width-64: +9.20%; capacity changes consistently hurt

## Open questions
- Does lr=1e-3 outperform 7e-4 on the new full stack? (alphonse #3579 — key)
- Does eta_min=1e-5 extract more from the cosine tail? (tanjiro #3536)
- Does n_head=8 help on the richer slice_num=96 stack? (askeladd #3535)
- Does depth=6 vs depth=4 bracket confirm depth=5 as the optimum? (thorfinn #3506, edward #3564)
- Does WD=5e-4 continue the ladder or reveal the optimum? (fern #3569)
- Will RFF positional encoding help after the stack is fully optimised? (nezuko #3534)
- Will per-channel pressure loss [1,1,2] address the cruise-vs-single trade-off? (frieren #3505)

## Closed/regressed (complete list)
- #3502 alphonse n_hidden=64: +9.20% vs new baseline (width reversal — capacity IS bottleneck below 96)
- #3503 edward mlp_ratio=4: +5.02% (capacity overfit)
- #3453 edward T_max=10: +3.55% (annealing too aggressive)
- #3301 alphonse width-192 (rebased): +8.53% (over-budget)
- #3304 frieren surf_weight=20 (rebased): +4.12% (lever absorbed)
- #3302 askeladd depth-8: +1.53% (budget-bound)
- #3223 thorfinn BF16+batch=8: +34% (padding overhead)
- #3295 edward slice_num=128: +20% (budget-mismatch)
- #3344 nezuko RFF (stuck — pod crash-loop) → reassigned as #3534
- #3362 askeladd n_head=8 (stuck — pod crash-loop) → reassigned as #3535
- #3397 tanjiro eta_min=1e-5 (pod crash-loop) → reassigned as #3536
- #3205, #3179, #3183, #3214, #3216, #3220 — round-1 dead ends

## Plateau watch
Best is 95.808 (PR #3314, merged). Total improvement: −17.9% from original 116.61 Huber baseline.
- 8 experiments reviewed since last round: 1 winner (#3314), 2 negatives (#3502, #3503).
- 8 experiments active: expect results in ~30-60 min per student run.
- The "loss still descending at E14" pattern across multiple runs is the clearest remaining signal — lr and schedule tail are the most productive levers to pull next.

## Next research directions (if round-5/6 exhausts local neighbourhood)
1. **SwiGLU activation** — replace GELU in FFN with SwiGLU (GLU variant, used in PaLM/LLaMA); higher-order gating with no extra params. Single line change.
2. **Attention dropout p=0.05** — stochastic attention on slice softmax; orthogonal to weight decay.
3. **Gradient accumulation × 2** — effective batch=8 without BF16; tests interaction with grad-clip.
4. **Log-pressure target** — reformulate pressure output as log(|p|+ε) to handle dynamic range; could specifically help single_in_dist.
5. **Best-checkpoint test** — report test metrics at best-val epoch rather than final epoch for paper numbers.
6. **Per-layer-class decay** — less WD on embedding/output projection, more on QKV; decouples the cruise/single trade-off seen in WD=3e-4.
