# SENPAI Research State

- **Last updated:** 2026-05-15 23:30 (post-rate-limit reset; all 4 idle students assigned; 8 students now active)
- **Most recent research direction from human researcher team:** none (no open issues — last verified 21:30Z).
- **Current best (merged):** `val_avg/mae_surf_p` = **96.667** (PR #3377 n_hidden=96, all 14 epochs; test 85.454)
- **Operational state:** All students assigned. Rate-limit storm resolved at 23:19Z reset. 8 experiments in flight across orthogonal axes.

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`. Local JSONL metrics only.

## Established baseline stack (merged to HEAD)
1. **PR #3208** (Huber loss) — `val_avg/mae_surf_p` 116.61
2. **PR #3276** (grad-clip + AdamW selective decay + NaN guard) — 109.68
3. **PR #3294** (warmup+cosine 14ep, lr=7e-4) — 100.811
4. **PR #3399** (slice_num=96) — 97.757
5. **PR #3377** (n_hidden=96) — **96.667** (current baseline)

Key config: SmoothL1 (Huber β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (wd=1e-4) + NaN guard + SequentialLR (LinearLR 2ep warmup + CosineAnnealingLR T_max=12) + lr=7e-4 + epochs=14 + slice_num=96 + **n_hidden=96** + n_layers=5 + n_head=4 + mlp_ratio=2.

## Active PRs (in flight)

| PR | Student | Hypothesis | Status | Notes |
|----|---------|-----------|--------|-------|
| #3314 | fern | weight_decay=3e-4 (RETEST) | WIP (stale base) | Pod running; verify config from committed metrics.yaml |
| #3344 | nezuko | RFF 32-freq (RETEST) | WIP (stale base) | Pod running |
| #3362 | askeladd | n_head 4→8 | WIP | Pod running |
| #3397 | tanjiro | eta_min=1e-5 in cosine | WIP (correct base) | Direct successor to #3453 — picks up where T_max=10 failed |
| #3502 | alphonse | n_hidden 96→64 (width ladder continuation) | WIP | Continue monotone ↓ ladder; smaller may = more epochs |
| #3503 | edward | mlp_ratio 2→4 (FFN expansion) | WIP | Untested orthogonal axis on new stack |
| #3505 | frieren | Per-channel loss [1,1,2] (pressure 2x) | WIP | Distinct from surf_weight — targets gradient channel balance |
| #3506 | thorfinn | n_layers 5→6 (first clean depth test) | WIP | Prior depth-8 (#3302) was budget-confounded; depth-6 is budget-safe |

## Confirmed design insights

### Schedule is the dominant lever
- Budget-matched warmup+cosine (14ep, lr=7e-4) → −8.08% (PR #3294)
- T_max smaller than budget hurts (PR #3453 closed) — lr-floor at cutoff matters
- Eta_min raise is the natural next probe (tanjiro #3397 in-flight)

### Slot-count is the second lever
- slice_num=64 → 96 (PR #3399): −3.03%
- slice_num=128: +20% regression (closed)
- slice_num=96 the sweet spot

### Width sweep — confirmed monotone decreasing
- Width ladder on new stack: {96: 96.67, 128: 97.76, 192: 106.10}
- Smaller n_hidden = better in this regime (capacity not the bottleneck — extra params steal wall-clock)
- Width-96 merged as new baseline
- Width-64 in flight (alphonse #3502) — next point on the ladder

### Regularization axes — old wins did NOT compound on new stack
- surf_weight=20: was −5.49% vs old; rebased +4.12% → CLOSED (absorbed)
- weight_decay=3e-4: was −3.69% vs old → fern in-flight retest pending
- RFF 32-freq: was −5.28% vs old → nezuko in-flight retest pending
- Pattern: levers compensating for under-fitting `single_in_dist` lose value once schedule + slot fixes the residual

### Architecture — mostly untested on new stack
- n_head=4→8 (askeladd #3362) in flight
- Depth-6 (thorfinn #3506) — first clean depth test; prior depth-8 was budget-confounded
- mlp_ratio=4 (edward #3503) — typical transformer FFN is 4x; current 2x is unusually small
- Per-channel loss reweighting (frieren #3505) — pressure gets 2x gradient vs Ux/Uy

## Open questions
- Will weight_decay=3e-4 still help with the new full stack? (fern #3314 in-flight, stale-base — needs careful interpretation)
- Will corrected RFF compound with schedule + slot change? (nezuko #3344 in-flight)
- Does eta_min=1e-5 extract more from the cosine tail? (tanjiro #3397)
- Does n_head=8 help? (askeladd #3362)
- Does n_hidden=64 continue the monotone width ladder down? (alphonse #3502)
- Does mlp_ratio=4 improve fitting without budget penalty? (edward #3503)
- Does pressure channel upweighting directly improve mae_surf_p? (frieren #3505)
- Is depth-6 budget-safe and beneficial on the new stack? (thorfinn #3506)

## Closed/regressed
- #3302 askeladd depth-8: +1.53% (budget-bound — 9ep only under old 50ep cosine; NOT a true signal about depth)
- #3223 thorfinn BF16+batch=8: +34% (padding overhead)
- #3295 edward slice_num=128: +20% (budget-mismatch)
- #3453 edward T_max=10: +3.55% (annealing too aggressive at cutoff)
- #3301 alphonse width-192 (rebased): +8.53% (over-budget; ladder monotonic ↓)
- #3304 frieren surf_weight=20 (rebased): +4.12% (lever absorbed)
- #3205, #3179, #3183, #3214, #3216, #3220 — round-1 dead ends

## Potential next research directions (post-round-5)
1. **Width-64** — if alphonse's #3502 wins, continue to width-48
2. **SwiGLU activation** — GELU vs SwiGLU; cheap to test, potentially significant
3. **Gradient accumulation** — effective larger batch (2–4x) without VRAM cost; interaction with grad-clip
4. **Best-checkpoint test** — currently test reported at final epoch; paper-facing numbers need test @ best-val epoch
5. **Log-pressure target** — loss reformulation; pressure has large dynamic range across Re regimes
6. **Per-channel loss exploration** — if frieren's 2x wins, try asymmetric combos or learned channel weights
7. **Attention dropout** — minor regularizer; Transolver's physics attention may benefit
8. **Depth ladder** — if depth-6 (thorfinn) wins, try depth-7; build the full ladder
9. **Focal-pressure loss** — emphasize high-Re samples that drive surface extremes
10. **Cosine restart (SGDR)** — if eta_min experiment closes, test warm restarts as alternative to monotone cosine
