# SENPAI Research State

- 2026-05-13 11:40 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=65.3734 (PR #2121 slice_num=48 + clip=5.0)**. Cumulative gain from PR #1391: 121.28 → 65.37 = −46.1%.
- No directives from human researcher team yet.
- **Assignment routing bug fixed**: PRs #2165 and #2166 were created on branches `tanjiro/...` and `thorfinn/...` (missing `willowpai2g48h1-` prefix), so student pods never polled them. Closed and reassigned as #2208 (tanjiro) and #2209 (thorfinn) on correctly-prefixed branches.

## Current baseline (PR #2121 merged — slice_num=48 + grad_clip=5.0)

**test_avg/mae_surf_p = 65.3734** | val = 71.9613 (best epoch 15)
Config: bf16 + batch_size=4 + accumulation_steps=2 (eff_bs=8) + Lion lr=1.5e-4 + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=48**, mlp_ratio=2 + **grad_clip_max_norm=5.0**. W&B run: vyjph01c.

Per-split: in_dist=67.70, rc=74.63, cruise=51.29, re_rand=67.87.

**Mechanism**: slice_num=48 is a regularization gain (not capacity bottleneck). Leaner slot partitioning imposes stronger locality bias on Transolver's physics attention, helping OOD splits most (rc −9.25%). cruise held flat confirming slot floor is below 48. Super-additive with clip=5.0: observed −18.9% combined vs predicted −16.8% sum-of-marginals.

## Previous baselines
- PR #2090 (grad_clip=5.0): test=68.0957 | clip bulk direction rescaler on Lion
- PR #1980 (gradient accumulation accum=2): test=80.62 | val=90.82
- PR #1395 (Lion optimizer): test=83.77 | Lion lr=1.5e-4, no accumulation
- PR #1387 (Fourier+wider): test=93.29 | AdamW lr=7e-4, space_dim=34, n_hidden=192
- PR #1361 (wider-192): test=99.69 (3-seed) | n_hidden=192, AdamW lr=7e-4
- PR #1591 (cosine-aligned): test=111.98 | n_hidden=128
- PR #1391 (bf16+batch-8): test=121.28 | n_hidden=128

## Round-3 status (updated 2026-05-13 11:40)

| Student | PR | Hypothesis | Status | Result |
|---------|-----|-----------|--------|--------|
| nezuko | #2226 | slice_num=32 + clip=5.0 — find slot floor | **wip** (NEW) | Continue capacity scan; cruise split is key diagnostic |
| tanjiro | #2208 | grad-clip-sweep (max_norm=2.0/10.0/50.0) | **wip** (REASSIGNED) | Bracket optimal clip threshold |
| thorfinn | #2209 | cosine-realign-epochs15 | **wip** (REASSIGNED) | T_max=15 to match actual 30-min budget |
| edward | #2141 | layerscale-1e-4 | **wip** | CaiT-style per-channel residual scaling, γ_init=1e-4 |
| fern | #2117 | ema-decay-095 + clip=5.0 | **wip** (restarted) | Run ckmhwg39 in flight — previous runs reverted clip by accident, fixed |
| frieren | #2190 | accumulation_steps=4 + clip=5.0 | **wip** | Mechanism-changed retest: step starvation resolved by clip? |
| alphonse | #2191 | n_layers=6 + clip=5.0 | **wip** | Mechanism-changed depth retest: gradient instability was the failure mode |
| askeladd | #2088 | lion-lr-2.1e-4-sqrt2 | **wip** (stale) | 2 finished arms regressed (test=85.22, 89.87). lr=1.8e-4 arm still running (efvjddip, val=113.71 at ep8/18). Likely closing soon. |
| nezuko | #2121 | slice_num=48 + clip=5.0 retest | **MERGED** ✓ | **test=65.3734** (−3.99% vs 68.10 baseline). New best. |
| alphonse | #2115 | mesh-node-dropout=0.1 | **CLOSED** ✗ | +84.7% catastrophic. |
| frieren | #2118 | fourier-per-axis-L (Lx=8, Ly=4) | **CLOSED** ✗ | +4.71%. |

## Key research findings (cumulative)

1. **Throughput at 30-min budget**: bf16+batch-8 → 17 epochs → round-1 win.
2. **Schedule alignment**: T_max=actual epochs → −7.67%. T_max=14 over-corrects.
3. **Width × schedule compounds**: n_hidden=192 → −10.97%.
4. **Fourier × width compounds**: NeRF L=8 → −6.42%. High-freq basis helps near-foil.
5. **Lion optimizer biggest lever (before clip)**: sign-momentum → −15.97% (83.77 from 99.69).
6. **Lion opens memory budget**: AdamW ~94 GB → Lion ~43 GB at n_hidden=192 bs=4.
7. **Depth dead at all tested widths**: horizon-vs-depth tradeoff. CLOSED permanently (pre-clip — being retested in #2191).
8. **Gradient accumulation (accum=2) wins**: −3.77%, free. Tighter micro-batch padding reduces sign-vote noise.
9. **Width lever CLOSED**: empirical O(n_hidden^2.43) per-epoch cost; n_hidden>192 infeasible at 30-min.
10. **EMA decay=0.999 CLOSED**: Rapid descent regime ≠ stationary trajectory. Short-horizon (0.95+clip) being retested in #2117.
11. **Slice-num=96 CLOSED, 48 WINS**: monotonic trend — smaller=better. Regularization not capacity. Scan continues at 32.
12. **Weight-decay lever CLOSED across both axes**: magnitude+structure both null.
13. **Fourier lever permanently CLOSED at L=8 uniform**: L=16 aliases, Lx=8/Ly=4 also closed (+4.71%). Three evidence points confirm L=8.
14. **grad-accum=4 CLOSED (pre-clip)**: Step starvation at eff_bs=16. Being retested with clip in #2190.
15. **DropPath CLOSED**: Underfitting regime, depth=5 too shallow.
16. **Activation-swap CLOSED (SiLU, +14.3%)**: GELU near-optimal at depth=5 width=192.
17. **grad_clip=5.0 MASSIVE WIN**: −15.5% (68.10 from 80.62). Bulk Lion direction rescaler. New best (now superseded by slice stacking).
18. **Mesh-node-dropout CLOSED at p=0.1**: Dense physics attention incompatible with PointNet-style dropout.
19. **slice_num=48 + clip COMPOUND WIN**: −3.99% on top of clip baseline. Super-additive (−18.9% total vs predicted −16.8%). rc split shows biggest OOD benefit. New best test=65.3734.
20. **Lion lr scaling FAILING**: sqrt(2) rule (lr=2.1e-4) regresses all arms tested (test=85-90 vs new 65.37). clip=5.0 likely changes the effective LR sensitivity. lr lever appears closed.

## Active hypotheses — priority order

### Tier 1: Direct clip/slice stacking follow-ups
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #2226 | nezuko | slice_num=32 + clip=5.0 | −0.5% to −2% if regularization continues; cruise is key diagnostic |
| #2208 | tanjiro | grad_clip sweep (2.0, 10.0, 50.0) | −0.5% to −2% if tighter clip is better |
| #2209 | thorfinn | cosine T_max=15 realign | −0.5% to −1.5% low-LR refinement gain |

### Tier 2: Mechanism-changed retests (discriminating)
| PR | Student | Hypothesis | Expected gain | Discriminating? |
|---|---|---|---|---|
| #2190 | frieren | accumulation_steps=4 + clip=5.0 | −1% to −3% if clip resolves step starvation | YES — validates clip mechanism |
| #2191 | alphonse | n_layers=6 + clip=5.0 | −2% to −5% if depth ceiling was clip-conditional | YES — tests architectural ceiling |

### Tier 3: Independent levers
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #2141 | edward | LayerScale γ=1e-4 | −0.3% to −1.5% |
| #2117 | fern | EMA decay=0.95 + clip=5.0 (restarted) | −0.3% to −1.5% |
| #2088 | askeladd | Lion lr=2.1e-4 (likely closing) | Currently regressing; lr=1.8e-4 arm finishing |

## Key open questions (updated)
1. **Is slice_num=32 better than 48?** (#2226) — is the slot floor below 32, or does cruise degrade?
2. **Is 2.0 better or worse than 5.0 for max_norm?** (#2208) — characterizes whether bulk rescaling benefits from being tighter.
3. **Does fully-annealed cosine (T_max=15) help with clip+slice stack?** (#2209)
4. **Does clip resolve accum=4 step starvation?** (#2190) — discriminating mechanism test.
5. **Is depth=5 a genuine architectural ceiling or clip-conditional?** (#2191) — discriminating depth test.
6. **Is there a deeper OOD gap between cruise (51.29) and rc (74.63) that can be closed?** 23-point gap still significant.
7. **Why is Lion lr scaling failing?** clip=5.0 likely changes effective LR sensitivity; sqrt(2) rule may not apply with bulk gradient rescaling.

## Plateau watch
NOT in plateau. Two consecutive wins (#2090 −15.5%, #2121 −3.99%) with 6 further discriminating experiments in flight. New strategy: mining the clip×capacity compound direction. Continue.

## Next milestones
- Nezuko #2226: slice_num=32 + clip=5.0 — continue capacity scan
- Tanjiro #2208: clip threshold sweep — bracket optimal max_norm
- Thorfinn #2209: cosine T_max=15 — schedule alignment on new stack
- Frieren #2190 and Alphonse #2191: discriminating mechanism retests
- Close #2088 askeladd when efvjddip finishes (lr=1.8e-4, epoch 8/18 as of 11:30)
