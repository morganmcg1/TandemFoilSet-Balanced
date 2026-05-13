# SENPAI Research State

- **Date:** 2026-05-13 19:35
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Current baseline (MERGED — 16-compound stack; 16th compound winner)

**PR #2219 — alphonse n_hidden=160** (merged 2026-05-13 19:35):
- `val_avg/mae_surf_p = 45.9186` (↓ from 46.6788, **−1.63%**, −0.76 absolute)
- `test_avg/mae_surf_p = 39.0381` (↓ from 39.7696, **−1.84%**, −0.73 absolute)
- Per-split test: in_dist 42.23 (−4.12%), camber_rc 53.94 (+1.57% slight regr.), camber_cruise 23.44 (−2.93%), re_rand 36.54 (−3.26%)
- Config: 15-compound stack + **n_hidden=160**, batch_size=2, n_layers=3, huber_beta=0.25, grad-clip=2.5, T_max=50, weight_decay=1e-4
- Mechanism: narrower net → 47.4s/epoch (vs 53s at n=192) → 38/50 epochs reached → cosine LR at 14% of base (vs ~26% at n=192) → more late-phase low-LR refinement. Win is specifically a bs=2 × n_hidden=160 interaction — 14-stack (no bs=2) was wash/slight loss.
- **W&B run:** `741bdhcl`
- **Reproduce:** `cd target/ && python train.py --agent <student> --wandb_name "<name>" --n_hidden 160 --n_layers 3 --batch_size 2 --epochs 50`

**Previous baseline — PR #2247** (batch_size=2, 15-compound): val=46.6788, test=39.7696.

**TRUE DIRECT MEASUREMENT THRESHOLD:** val < 45.9186, test < 39.0381.
**CRITICAL: All experiments must now specify** `--n_hidden 160 --n_layers 3 --batch_size 2 --epochs 50`. train.py defaults batch_size=4 (old stack) and n_hidden=128.

## Cumulative compounding (16 merges)

| Baseline | val | test | Key change |
|----------|-----|------|------------|
| Stock (MSE, fp32) | ~160+ | ~130+ | — |
| PR #1419 alphonse bf16 | 109.29 | 97.67 | bf16 autocast |
| PR #1436 fern Huber β=1.0 | 96.49 | 86.33 | Smooth L1 → loss-shape MAE alignment |
| PR #1606 fern EMA | 92.35 | 81.63 | Weight averaging → reduces noise |
| PR #1689 fern Huber β=0.5 | 85.92 | 76.55 | Tighter MAE alignment |
| PR #1672 nezuko warmup 1ep | 85.09 | 75.52 | LR warmup |
| PR #1763 edward torch.compile | 71.44 | 62.59 | 44% speedup → more epochs |
| PR #1875 frieren n_layers=3 | 69.45 | 61.19 | Depth reduction + speedup |
| PR #1784 tanjiro grad-clip=10 | 65.98 | 57.07 | Soft-scaling gradient damping |
| PR #1899 alphonse n_hidden=192 | 63.72 | 55.64 | Width reinvestment |
| PR #1930 tanjiro grad-clip=5 | 63.48 | 54.98 | Tighter threshold scan |
| PR #1953 alphonse T_max=50 | 55.76 | 48.10 | Schedule fix — MASSIVE (−12%) |
| PR #1982 tanjiro grad-clip=2.5 | 52.64 | 44.98 | Clip threshold scan step 3 |
| **PR #2142 fern Huber β=0.25** | **50.38** | **43.72** | Loss-shape tighter MAE alignment (14th) |
| **PR #2247 frieren batch_size=2** | **46.68** | **39.77** | Opt-step density 2× — FIRST saturation easing win (15th) |
| **PR #2219 alphonse n_hidden=160** | **45.92** | **39.04** | Width-narrowing × bs=2 interaction (16th) |

## Active experiments

| Student | PR | Hypothesis | Lever | Status | Note |
|---------|----|-----------|-------|------|-----|
| alphonse | #2395 | n_hidden=128 (push width-floor below n=160 optimum) | Architecture (width, below) | WIP | Just assigned. n=160 won −1.63%/−1.84%. Testing [?,128,160(OPT),192(old-OPT),224+FAIL]. Param count ~0.42M (−36% vs n=160). Risk: capacity floor. camber_rc already at +1.57% at n=160. |
| askeladd | #2328 | grad-clip max_norm 2.5 → 3.0 (+20% threshold raise) | Optimization (clip threshold) | WIP | DIRECT TEST of clip-saturation hypothesis. 7 axes now confirmed blocked by Pattern v3. If 3.0 wins, ALL 7 saturated axes re-open for retesting — massive leverage. Alerted to use `--batch_size 2`. |
| edward | #2024 | EMA decay 0.999 → 0.998 compound retest at bs=2 | Optimization (EMA) | WIP-COMPOUND-RETEST | Won at 14-stack: val 49.59 (−1.56%). Sent back for bs=2 retest. Additive prediction: val ≈ 45.89, test ≈ 39.17. Must use `--batch_size 2`. |
| fern | #2299 | Huber β=0.1 compound retest at bs=2 | Loss shape | WIP-COMPOUND-RETEST | Won at 14-stack: val 48.90 (−2.94%). Sent back for bs=2 retest. Additive prediction: val ≈ 45.35, test ≈ 38.29. Must use `--batch_size 2`. |
| frieren | #2398 | mlp_ratio 2 → 4 at n=160+bs=2 stack | Architecture (FFN width) | WIP | Just assigned. Hypothesis: alphonse's n=160 freed 0.28M params — reinvest in deeper FFN. mlp_ratio 2→4 raises FFN dim 320→640, param count ~0.96M. Key question: is FFN the binding capacity constraint at n=160? Note: mlp_ratio=4 failed at n=192/n_layers=5 stack (PR #1544). This is n=160/n_layers=3 stack — different capacity regime. |
| nezuko | #2329 | AdamW eps 1e-8 → 1e-6 denominator-stability test | Optimization (optimizer denominator) | WIP | Untested optimizer-internal axis. 100× eps floor raise. Alerted to use `--batch_size 2`. |
| tanjiro | #2305 | wd=3e-4 compound retest: n=160 + wd=3e-4 | Regularization (parameter scale) | WIP-COMPOUND-RETEST | Won at 15-stack (val 46.08, −1.28%). Sent back because #2219 merged. Now testing n=160+wd=3e-4+bs=2. Additive prediction: val ≈ 45.33, test ≈ 38.48. Must use `--n_hidden 160 --batch_size 2`. |
| thorfinn | #2276 | n_layers 3 → 2 at n_hidden=192 | Architecture (depth) | WIP | Alerted to use `--batch_size 2`. Note: now measuring n=192+n_layers=2+bs=2 vs new 16-compound baseline (n=160+n_layers=3+bs=2). If it wins, means depth=2 overcomes the n=192 width-excess penalty at bs=2. |

## Axes fully bracketed (no further testing needed)

| Axis | Bracket | Best |
|------|---------|------|
| Width (n_hidden) | [128(TBD), 160 OPT@16, 192 OPT@old, 224 FAIL, 256 FAIL] | 160 at 16-stack |
| Schedule (T_max) | [33 FAIL / 50 OPT / 80 FAIL] | 50 |
| Slice_num | [48 FAIL / 64 OPT / 96 FAIL] | 64 |
| LR (lr) | [3e-4 FAIL / 5e-4 OPT / 7.5e-4 FAIL] | 5e-4 |
| batch_size | [1 FAIL / 2 OPT / 4 FAIL] | 2 |
| n_head | [4 OPT / 8 FAIL] | 4 |
| surf_weight | [5 FAIL / 10 OPT / 30 FAIL] | 10 |
| LR warmup | [1ep OPT / 2ep FAIL] | 1 ep |

## Clip-saturation pattern v3 — confirmed 7 axes

**Pattern:** Any perturbation that increases time-in-high-gradient-regime → more clipping → delayed convergence amplification → failure.

| PR | Lever | clip_rate | val Δ | Verdict |
|---|---|---:|---:|---|
| #2066 tanjiro | n_hidden=224 | 99.31% | +3.22% | FAIL |
| #2000 alphonse | T_max=80 | 99.44% | +4.54% | FAIL |
| #2159 askeladd | lr=7.5e-4 | 99.30% | +7.77% | FAIL |
| #2053 nezuko | mlp_ratio=3 | 99.57% | +6.66% | FAIL |
| #2186 thorfinn | β₂=0.95 | 99.29% | +4.10% | FAIL |
| #2231 askeladd | lr=3e-4 | 99.38% | +4.74% | FAIL |
| **#2355 frieren** | **batch_size=1** | **99.46%** | **+20.59%** | **FAIL** |

**Confirmed bypass axes** (not blocked by clip saturation):
- Huber β tighter (loss-curvature, upstream of gradient) — #2142, #2299
- Weight decay tighter (param shrinkage, downstream of clipped step) — #2305
- EMA decay (post-clip averaging) — #2024
- Width narrowing (throughput/capacity, orthogonal to gradient scale) — #2219
- Opt-step density via bs=2 (more steps, smoother trajectory) — #2247

**Next saturation test:** askeladd #2328 (max_norm 2.5→3.0). **If wins: ALL 7 blocked axes re-open.** This is the most leveraged single experiment in the fleet.

## Closed (cycles 27-38)

- #2267 nezuko slice_num=48 — FAIL +15.10%/+20.27%; slice axis bracketed
- #2231 askeladd lr=3e-4 — FAIL +4.74% vs new; LR axis bracketed; student derived Pattern v3
- #2199 tanjiro epochs=33 — FAIL +9.36% vs new; schedule axis bracketed
- #2186 thorfinn AdamW β₂=0.95 — FAIL +4.10%; first optimizer-internal axis confirmed blocked by saturation
- #2160 frieren wd=1e-5 — FAIL net OOD-negative; wd axis confirmed NOT blocked but direction was wrong
- #2159 askeladd lr=7.5e-4 — FAIL +7.77%; LR amplitude axis blocked by saturation
- #2068 thorfinn n_hidden=256 — FAIL (runtime-induced epoch deficit)
- #2066 tanjiro n_hidden=224 — FAIL +3.22%; width axis bracketed from above
- #2053 nezuko mlp_ratio=3 — FAIL +6.66%; FFN amplitude axis blocked
- #2000 alphonse T_max=80 — FAIL +4.54%; schedule extension blocked by saturation
- **#2355 frieren batch_size=1 — FAIL +20.59%; batch_size axis fully bracketed [1/2/4]; 7th Pattern-v3 confirmation**

## Potential next directions

### Highest priority (in flight)
1. **askeladd #2328 grad-clip=3.0** — most leveraged test: if wins, 7 axes re-open
2. **tanjiro #2305 compound: wd=3e-4 + n=160** — predicted val 45.33/test 38.48; strong additive candidate
3. **fern #2299 compound: Huber β=0.1 + bs=2** — predicted val 45.35/test 38.29; monotone β scan
4. **edward #2024 compound: EMA=0.998 + bs=2** — predicted val 45.89/test 39.17

### Width axis continuation
5. **alphonse #2395 n_hidden=128** — just assigned; width bracket from below
6. If 128 wins → test n_hidden=96 (further throughput push, capacity floor risk)
7. If 128 fails → width axis fully bracketed [128/160/192+/224+]

### Capacity reinvestment at n=160
8. **frieren #2398 mlp_ratio=4 at n=160** — just assigned; reinvest freed params into FFN
9. If mlp_ratio=4 wins → compound with n=128 (if 128 wins)
10. If mlp_ratio=4 fails → FFN is not bottleneck at n=160

### If grad-clip=3.0 wins (re-open saturated axes at threshold=3.0)
11. n_hidden=224 retest at threshold=3.0
12. mlp_ratio=3 retest at threshold=3.0
13. T_max=80 retest at threshold=3.0
14. lr=7.5e-4 retest at threshold=3.0

### Longer horizon
15. **Per-layer weight decay** — apply different wd to FFN vs attention (wd axis shows distinct per-split signatures)
16. **EMA decay scan** — EMA-live gap now +15.7% under wd=3e-4+bs=2; decay=0.998 may be optimal
17. **Re-aware embeddings** — explicit log-Re positional encoding; re_rand still the hardest OOD split
18. **Spectral/Fourier hybrid** — if attention-based approach plateaus
