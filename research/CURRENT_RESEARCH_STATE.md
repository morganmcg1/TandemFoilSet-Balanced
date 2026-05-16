# SENPAI Research State

- **Last updated:** 2026-05-16 00:45 (merged fern #3314 weight_decay=3e-4; closed edward #3503; assigned round-6 work to edward + fern)
- **Most recent research direction from human researcher team:** none (no open issues — verified 00:45Z).
- **Current best (merged):** `val_avg/mae_surf_p` = **95.808** (PR #3314 weight_decay=3e-4 on full stack)
- **Round-5 in flight (6 students still running):** width-64 (alphonse #3502), per-channel pressure loss (frieren #3505), n_layers=6 (thorfinn #3506), RFF rebased (nezuko #3534), n_head=8 rebased (askeladd #3535), eta_min=1e-5 rebased (tanjiro #3536).
- **Round-6 just assigned:** n_layers=4 depth-bracket (edward, new PR pending), weight_decay=5e-4 WD ladder (fern, new PR pending).
- **Operational state:** All clear. No rate-limit issues. Merge + assignment cycle complete.

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

## Round-5 review verdicts (this triage cycle)

| PR | Student | Hypothesis | Val result | Verdict |
|----|---------|-----------|-----------|---------|
| #3503 | edward | mlp_ratio=4 | 101.686 (+5.02%) | CLOSED — capacity not bottleneck; FFN expansion overfit uniformly |
| #3314 | fern | weight_decay=3e-4 (rebased) | **95.808 (−0.89%)** | **MERGED — new baseline** |

## Round-5 active PRs (6 still in flight on new HEAD)

| PR | Student | Hypothesis | Branch base | Notes |
|----|---------|-----------|-------------|-------|
| #3502 | alphonse | n_hidden 96→64 (width ladder) | new HEAD | Next width-ladder step; if monotone holds, still descending |
| #3505 | frieren | per-channel pressure loss [1,1,2] | new HEAD | Direct test of pressure under-fit on single_in_dist |
| #3506 | thorfinn | n_layers 5→6 (depth) | new HEAD | First clean depth test on optimised stack |
| #3534 | nezuko | RFF 32-freq (REASSIGNED from #3344) | new HEAD | Fresh branch after pod crash-loop |
| #3535 | askeladd | n_head 4→8 (REASSIGNED from #3362) | new HEAD | Fresh branch after pod crash-loop |
| #3536 | tanjiro | eta_min=1e-5 (REASSIGNED from #3397) | new HEAD | Direct test of lr-floor; counterpart to closed T_max=10 |

## Round-6 new assignments

| Student | Hypothesis | Rationale |
|---------|-----------|-----------|
| edward | n_layers=4 (depth bracket below) | Complete depth axis: thorfinn=6 vs edward=4 vs baseline=5. Potentially faster per-epoch → more cosine tail within cap. |
| fern | weight_decay=5e-4 (WD ladder) | Continue WD monotone: 1e-4→3e-4 gave −0.89%. Test if 5e-4 extends the trend or reveals the optimum. |

## Confirmed design insights

### Schedule is the dominant lever
- Budget-matched warmup+cosine (14ep, lr=7e-4) → −8.08% (PR #3294)
- T_max smaller than budget hurts (PR #3453 closed) — lr-floor at cutoff matters
- Eta_min raise is the natural next probe (tanjiro #3536 in-flight)

### Slot-count is the second lever
- slice_num=64 → 96 (PR #3399): −3.03%
- slice_num=128: +20% regression (closed)
- slice_num=96 the sweet spot

### Width sweep — smaller is better
- Width ladder on new stack: {96: 96.67, 128: 97.76, 192: 106.10}
- Monotonic ↓: smaller is better in this regime
- Width-64 untested (alphonse #3502 in-flight)

### Regularization — weight decay helps
- weight_decay=3e-4 > 1e-4: −0.89% net (PR #3314, merged)
- Pattern: improvement concentrated on single_in_dist (hardest split); cruise regresses slightly
- WD ladder not saturated — 5e-4 is the next probe (fern round-6)

### Capacity axis — NOT the bottleneck
- mlp_ratio=4 (Edward, #3503): +5.02% regression — FFN expansion overfit uniformly
- width-192: +8.53% regression; width-128: +1.12% regression; width-96 sweet spot
- Extra params hurt rather than help on 14ep budget

### Architecture untested on new stack
- n_head=4→8 (askeladd #3535) in flight
- n_layers=5→6 (thorfinn #3506) in flight
- n_layers=4 (edward, round-6) just assigned

## Open questions
- Does eta_min=1e-5 extract more from the cosine tail? (tanjiro #3536)
- Does n_head=8 help on the richer slice_num=96 stack? (askeladd #3535)
- Is width=64 even better than 96? (alphonse #3502)
- Does depth=6 help or hurt? (thorfinn #3506)
- Does depth=4 bracket confirm depth=5 as the optimum? (edward, round-6)
- Does WD=5e-4 continue the ladder or reveal the optimum? (fern, round-6)
- Will RFF help after the stack is fully optimised? (nezuko #3534)
- Will per-channel pressure loss [1,1,2] address the cruise-vs-single trade-off? (frieren #3505)

## Closed/regressed (complete list)
- #3503 edward mlp_ratio=4: +5.02% (capacity overfit)
- #3453 edward T_max=10: +3.55% (annealing too aggressive)
- #3301 alphonse width-192 (rebased): +8.53% (over-budget; ladder monotonic ↓)
- #3304 frieren surf_weight=20 (rebased): +4.12% (lever absorbed)
- #3302 askeladd depth-8: +1.53% (budget-bound)
- #3223 thorfinn BF16+batch=8: +34% (padding overhead)
- #3295 edward slice_num=128: +20% (budget-mismatch)
- #3344 nezuko RFF (stuck on stale base — pod crash-loop) → reassigned as #3534
- #3362 askeladd n_head=8 (stuck on stale base — pod crash-loop) → reassigned as #3535
- #3397 tanjiro eta_min=1e-5 (pod crash-loop on dirty train.py) → reassigned as #3536
- #3205, #3179, #3183, #3214, #3216, #3220 — round-1 dead ends

## Plateau watch
Best is 95.808 (PR #3314 merged). Total improvement: −17.9% from original 116.61 Huber baseline.
- 6 experiments reviewed since last baseline merge (#3377): 1 winner (weight_decay=3e-4), 1 negative (mlp_ratio=4).
- Currently 6 round-5 experiments in-flight — their results will determine whether we're on a new declining plateau or can push past 95.

## Next research directions to explore (if round-5/6 exhausts local neighbourhood)
1. **Activation function** — GELU vs SwiGLU vs ReLU; cheap to test, untested on new stack.
2. **Attention dropout** — p=0.05 on slice-attention softmax; minor regularizer, orthogonal to weight decay.
3. **Gradient accumulation** — effective larger batch without memory cost; test interaction with grad-clip.
4. **Best-checkpoint test** — evaluate test metrics at best-val epoch rather than final epoch for paper-facing numbers.
5. **Log-pressure target** — reformulate pressure output as log(|p|+ε) to handle the large dynamic range across splits.
6. **Per-layer-class decay** — less decay on embedding/output proj, more on QKV; could decouple the cruise/single tradeoff.
