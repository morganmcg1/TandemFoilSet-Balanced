# SENPAI Research State

- **Last updated:** 2026-05-15 22:40 (round-4 reviews complete; 3 closures, 1 held winner; GH rate-limit storm — waiting for 23:19Z reset)
- **Most recent research direction from human researcher team:** none (no open issues — last verified 21:30Z).
- **Current best (merged):** `val_avg/mae_surf_p` = **97.757** (PR #3399 slice_num=96 on warmup+cosine baseline)
- **Pending merge:** PR #3377 (n_hidden=96) — val 96.667 (−1.12%), test 85.454; HELD pending student push of rebased commits
- **Operational state:** GH rate-limit hit at 22:34Z, resets at 23:19Z. 5 student pods still in-flight on assigned axes; 3 students newly idle (alphonse, edward, frieren). Assignments deferred to wakeup at 23:25Z.

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`. Local JSONL metrics only.

## Established baseline stack (merged to HEAD)
1. **PR #3208** (Huber loss) — `val_avg/mae_surf_p` 116.61
2. **PR #3276** (grad-clip + AdamW selective decay + NaN guard) — 109.68
3. **PR #3294** (warmup+cosine 14ep, lr=7e-4) — 100.811
4. **PR #3399** (slice_num=96) — **97.757** (current baseline)

Key config: SmoothL1 (Huber β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (wd=1e-4) + NaN guard + SequentialLR (LinearLR 2ep warmup + CosineAnnealingLR T_max=12) + lr=7e-4 + epochs=14 + **slice_num=96**.

## Round-4 review verdicts (this triage cycle)

| PR | Student | Hypothesis | Val result | Verdict |
|----|---------|-----------|-----------|---------|
| #3453 | edward | T_max=10 | 101.225 (+3.55%) | CLOSED — annealing too aggressive at cutoff |
| #3301 | alphonse | width-192 (rebased) | 106.099 (+8.53%) | CLOSED — over-budget; width ladder confirms smaller is better |
| #3304 | frieren | surf_weight=20 (rebased) | 101.782 (+4.12%) | CLOSED — lever absorbed by new baseline |
| #3377 | thorfinn | n_hidden=96 (rebased) | **96.667 (−1.12%)** | **HELD — winner; awaiting student push of rebased commits** |

## Active PRs (in flight)

| PR | Student | Hypothesis | Status | Notes |
|----|---------|-----------|--------|-------|
| #3314 | fern | weight_decay=3e-4 (RETEST) | WIP (stale base) | Pod running; verify config from committed metrics.yaml |
| #3344 | nezuko | RFF 32-freq (RETEST) | WIP (stale base) | Pod running |
| #3362 | askeladd | n_head 4→8 | WIP | Pod running |
| #3397 | tanjiro | eta_min=1e-5 in cosine | WIP (correct base) | Direct successor to #3453 — picks up where T_max=10 failed |
| #3377 | thorfinn | n_hidden=96 (rebased push) | WIP — held for branch update | Must push rebased commits, then re-mark ready |

## Idle students awaiting new assignments (post-rate-limit reset)
- **alphonse** — width-192 dead-end closed. Suggested next direction: depth or attention-pattern axis (NOT width — that ladder is now monotonic ↓).
- **edward** — T_max=10 dead-end closed. Suggested next direction: opposite end — raise eta_min, or try ReLR after cosine bottoms out. May overlap with tanjiro #3397 — pick a non-overlapping axis.
- **frieren** — surf_weight=20 closed. Suggested next direction: try the OPPOSITE direction (downweight surface to free velocity), or a different loss reformulation (e.g. log-magnitude, gradient-of-pressure penalty, frequency-domain loss).
- **thorfinn** — held on #3377 (not idle; will re-run after pushing).

## Confirmed design insights

### Schedule is the dominant lever
- Budget-matched warmup+cosine (14ep, lr=7e-4) → −8.08% (PR #3294)
- T_max smaller than budget hurts (PR #3453 closed) — lr-floor at cutoff matters
- Eta_min raise is the natural next probe (tanjiro #3397 in-flight)

### Slot-count is the second lever
- slice_num=64 → 96 (PR #3399): −3.03%
- slice_num=128: +20% regression (closed)
- slice_num=96 the sweet spot

### Width sweep — newly closed
- Width ladder on new stack: {96: 96.67, 128: 97.76, 192: 106.10}
- Monotonic ↓: smaller is better in this regime
- Capacity not the bottleneck — extra params steal wall-clock from cosine
- Width-96 (PR #3377) — pending merge as new baseline
- Width-64 untested but predicted next probe direction

### Regularization axes — old wins did NOT compound on new stack
- surf_weight=20: was −5.49% vs old; rebased +4.12% → CLOSED (absorbed)
- weight_decay=3e-4: was −3.69% vs old → fern in-flight retest pending
- RFF 32-freq: was −5.28% vs old → nezuko in-flight retest pending
- Pattern: levers compensating for under-fitting `single_in_dist` lose value once schedule + slot fixes the residual

### Architecture untested on new stack
- n_head=4→8 (askeladd #3362) in flight
- Depth-8: +1.5% regression on old stack (closed #3302); untested on new

## Open questions
- Will weight_decay=3e-4 still help with the new full stack? (fern #3314 in-flight, stale-base — needs careful interpretation)
- Will corrected RFF compound with schedule + slot change? (nezuko #3344 in-flight)
- Does eta_min=1e-5 extract more from the cosine tail? (tanjiro #3397)
- Does n_head=8 help? (askeladd #3362)
- Once width=96 is the new baseline, is width=64 better still? (next probe)
- Are there architecture-level levers untouched? (depth, MLP ratio, dropout, attention pattern)
- Loss formulation: log-pressure, gradient-of-pressure penalty, frequency-domain — completely untested

## Stale-base lessons (cascading round-3 / round-4 issue)
Most round-3/4 students were assigned before downstream merges (#3294, #3399). All required rebase+retest. Two of three rebased retests showed the lever was absorbed (surf_weight, width-192). Width-96 was the lone exception. **Operational improvement to enforce: future round-N students should commit + push the rebased metrics commit at the same time as posting SENPAI-RESULT, not after.** Three of four round-4 students (frieren, alphonse, thorfinn) failed to push rebased artifacts.

## Plateau watch
Best is 96.667 (pending merge from #3377). Total improvement: −16.4% from original 116.61 Huber baseline.
- 5 retest experiments completed since last baseline merge (#3399).
- 1 winner (width-96, pending push); 4 negatives.
- This is NOT a plateau yet — width-96 will likely merge and reset the bar. But the regularization-axis exhaustion pattern (surf_weight, possibly weight_decay, possibly RFF if it follows) signals we should think about NEW levers, not retesting old wins.

## Next research directions to seed (post-rate-limit, alphonse/edward/frieren)
1. **Width-64** — natural follow-up to width-96 win; if monotonic continues, width is still descending.
2. **Eta_min variant** — tanjiro #3397 covers eta_min=1e-5; could try eta_min=5e-5 or holding-LR final epochs as alternative to T_max change.
3. **Loss reformulation** — log-pressure target, or per-channel loss balancing toward velocity (opposite of surf_weight) to test the "velocity-starved" hypothesis.
4. **Mlp_ratio** — currently 2; try 4 (more FFN expansion, holding everything else equal). Compounds well with smaller width.
5. **Attention dropout / weight noise** — minor regularizer not yet tested.
6. **Activation function** — GELU vs SwiGLU vs ReLU; cheap to test.
7. **Gradient accumulation** — could enable effective larger batch w/o memory cost; test interaction with grad-clip.
8. **Best-checkpoint test** — currently test is reported at final epoch; for paper-facing numbers we want test evaluated at best-val epoch.

## Closed/regressed
- #3302 askeladd depth-8: +1.53% (budget-bound)
- #3223 thorfinn BF16+batch=8: +34% (padding overhead)
- #3295 edward slice_num=128: +20% (budget-mismatch)
- #3453 edward T_max=10: +3.55% (annealing too aggressive)
- #3301 alphonse width-192 (rebased): +8.53% (over-budget; ladder monotonic ↓)
- #3304 frieren surf_weight=20 (rebased): +4.12% (lever absorbed)
- #3205, #3179, #3183, #3214, #3216, #3220 — round-1 dead ends
