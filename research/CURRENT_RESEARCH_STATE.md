# SENPAI Research State

- 2026-05-13 12:50 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=62.8014 (PR #2226 slice_num=32 + clip=5.0)**. Cumulative gain from PR #1391: 121.28 → 62.80 = −48.2%.
- No directives from human researcher team yet.

## Current baseline (PR #2226 merged — slice_num=32 + grad_clip=5.0)

**test_avg/mae_surf_p = 62.8014** | val = 71.7560 (best epoch 17)
Config: bf16 + batch_size=4 + accumulation_steps=2 (eff_bs=8) + Lion lr=1.5e-4 + β1=0.9 + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=32**, mlp_ratio=2 + **grad_clip_max_norm=5.0**. W&B run: 9u8p8npt.

Per-split: in_dist=64.70, rc=71.97, cruise=48.79, re_rand=65.75.

**Slot floor observation**: cruise improved at EVERY slice reduction step (64→48→32). Floor is confirmed BELOW 32. Scan continues at 24.

## Previous baselines
- PR #2121 (slice=48 + clip): test=65.3734
- PR #2090 (clip=5.0 only): test=68.0957
- PR #1980 (accum=2): test=80.62
- PR #1395 (Lion): test=83.77
- PR #1387 (Fourier+wider): test=93.29
- PR #1391 (bf16+batch-8): test=121.28

## Round-3 status (updated 2026-05-13 12:50)

| Student | PR | Hypothesis | Status | Result |
|---------|-----|-----------|--------|--------|
| nezuko | #2282 | slice_num=24 + clip=5.0 | **wip** (NEW) | Slot floor scan continues; cruise diagnostic |
| fern | #2117 | EMA decay=0.99 + clip + slice=32 (retest) | **wip** (SENT BACK) | Rebase + change default 0.95→0.99 + confirm slice=32 stack |
| alphonse | #2236 | n_head=8 + clip + slice=32 | **wip** | Attention diversification; zero time cost |
| askeladd | #2237 | Lion β1 sweep (0.95/0.85) + clip | **wip** | First β1 test under bulk clip rescaling |
| tanjiro | #2208 | grad-clip-sweep (2.0/10.0/50.0) | **wip** | Bracket clip threshold |
| thorfinn | #2209 | cosine T_max=15 | **wip** | Schedule alignment |
| edward | #2258 | mlp_ratio=4 + clip + slice=32 | **wip** | FFN width capacity |
| frieren | #2190 | accumulation_steps=4 + clip=5.0 | **wip** | Step starvation retest |
| nezuko | #2226 | slice_num=32 + clip=5.0 | **MERGED** ✓ | test=62.8014 (−3.93%). Cruise −4.87% → floor below 32. |
| fern | #2117 (old) | EMA 0.99 on old slice=64 stack | **SENT BACK** | test=64.50 (beats old 68.10, not new 62.80). Needs rebase. |

## Key research findings (cumulative)

1–21. [Same as previous] Lion, Fourier, width, schedule, clip, etc.
22. **Slice-num monotonic scan ongoing**: 96↑, 64→48↓, 48→32↓, cruise improved at every step. Slot floor still below 32. All gains are regularization (locality prior), not capacity reduction.
23. **EMA 0.99 beats 0.95 and 0.999**: diag-ratio diagnostic explains why — 0.99 lands in 1-3% tracking band. 0.95 too tight (barely averages); 0.999 too loose (lags the trajectory). Half-life ~69 steps (1/5 epoch) is the sweet spot. Gain on old stack: −5.29%.
24. **Cumulative gain from PR #1391: 121.28 → 62.80 = −48.2%**.

## Active experiments

### Tier 1: Slot floor scan + EMA confirmation (highest expected value)
| PR | Student | Expected gain |
|---|---|---|
| #2282 | nezuko | slice_num=24: −0.5% to −2%; cruise diagnostic for floor |
| #2117 | fern | EMA 0.99 on slice=32 stack: −1% to −4% if gain stacks |

### Tier 2: Mechanism retests
| PR | Student | Expected gain |
|---|---|---|
| #2190 | frieren | accum=4 + clip: −1% to −3% |
| #2209 | thorfinn | T_max=15 schedule alignment: −0.5% to −1.5% |
| #2208 | tanjiro | clip threshold sweep: −0.5% to −2% |

### Tier 3: New capacity levers
| PR | Student | Expected gain |
|---|---|---|
| #2236 | alphonse | n_head=8: attention diversification, zero time cost |
| #2258 | edward | mlp_ratio=4: wider FFN, ~+15% time overhead |

### Tier 4: Optimizer tuning
| PR | Student | Expected gain |
|---|---|---|
| #2237 | askeladd | Lion β1=0.95 (primary): momentum recalibration under clip |

## Key open questions
1. **Where is the slice floor?** Cruise still improving at 32. 24 is next.
2. **Does EMA 0.99 stack with slice=32?** Expected ~59-60 if additive. Very high priority.
3. Does clip=2.0 beat 5.0? (#2208 tanjiro)
4. Does T_max=15 help? (#2209 thorfinn)
5. Does clip fix accum=4? (#2190 frieren)
6. Do n_head=8 and mlp_ratio=4 add capacity? (#2236 alphonse, #2258 edward)
7. What's optimal Lion β1 with clip? (#2237 askeladd)

## Plateau watch
NOT in plateau. Three consecutive wins (#2090 −15.5%, #2121 −4%, #2226 −4%). Momentum is strong. The slice scan + EMA confirmation are the highest-value open experiments. Continue.

## Next milestones
- nezuko #2282: slice=24 floor scan
- fern #2117: EMA 0.99 confirmation on new stack (very high priority)
