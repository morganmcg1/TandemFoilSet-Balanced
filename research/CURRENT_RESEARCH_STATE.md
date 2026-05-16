# SENPAI Research State — TandemFoilSet (willow-pai2i-24h-r4)

- **As of:** 2026-05-16 00:42 UTC
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r4`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced`
- **W&B:** `wandb-applied-ai-team/senpai-v1`
- **Most recent human researcher direction:** None recorded. Launch isolation rules are in force.

## Research programme summary

Predict (Ux, Uy, p) at every node of unstructured 2D CFD meshes (74K–242K nodes) of tandem airfoils. The primary ranking metric is `test_avg/mae_surf_p` — equal-weighted surface pressure MAE across four test splits. Per-run budget is 30 min wall clock × 50 epochs hard cap.

## Current best (BASELINE.md)

**PR #3358 (alphonse, merged 2026-05-16 00:24 UTC) — Cosine LR T_max=14 (matched to wall-clock cap)**
- `val_avg/mae_surf_p = 90.44` (W&B `b9qv36aq`)
- `test_avg/mae_surf_p = **80.08**` (4-split finite, wins all 4 val + all 4 test splits)

Cumulative path: vanilla 106.23 → #3257 frieren loss 94.35 → #3263 single FiLM 90.06 → #3358 cosine T_max=14 80.08 (**−24.6% from vanilla in 3 PRs**).

All remaining PRs must beat **test_avg/mae_surf_p < 80.08**.

## Active R2/R3 portfolio (all 8 students busy)

| # | Student | PR | Hypothesis | Status |
|---|---------|----|-----------|--------|
| 1 | nezuko   | #3550 | Volume MAE reformulation (unify L1 loss across surface + volume) | WIP (just assigned 00:30) |
| 2 | frieren  | #3504 | Richer FiLM conditioning (cond_dim 1→11: log_Re+AoA+NACA+gap+stagger) | WIP |
| 3 | thorfinn | #3468 | Per-block FiLM heads — **post-block v1 won OLD base** (test −6.73%); rebase + rerun on new cosine base | WIP (sent back 00:25) |
| 4 | tanjiro  | #3406 | surf_weight sweep — **sw5 winner on OLD base** (test=88.80 vs 94.35); rebase + rerun on new cosine base | WIP (sent back 23:21) |
| 5 | alphonse | #3565 | AdamW betas=(0.9,0.95) + weight_decay=0.05 sweep (3 arms) | WIP (assigned 00:40) |
| 6 | askeladd | #3351 | EMA β=0.99 (shorter horizon) | WIP (CONFLICTING, pod actively training) |
| 7 | edward   | #3262 | RFF σ=1.0 on (x, z) — orthogonal to FiLM | WIP (MERGEABLE) |
| 8 | fern     | #3258 | Grad-clip 1.0 + 5-epoch warmup — MERGEABLE post-FiLM | WIP |

**All 8 students are now active.** Alphonse assigned #3565 (AdamW sweep) at 00:40 UTC.

## R1/R2 closed/merged history

| # | Student | Hypothesis | PR | Status |
|---|---------|-----------|-----|--------|
| 1 | frieren  | Surface MAE + p-weight 3× + NaN guard | #3257 | **MERGED (R1 winner #1)** — test=94.35 |
| 2 | thorfinn | FiLM log(Re) conditioning | #3263 | **MERGED (R1 winner #2)** — val=100.24, test=90.06 (−4.55%) |
| 3 | alphonse | Cosine LR T_max=14 | #3358 | **MERGED (R2 winner #1)** — val=90.44, test=80.08 (−11.08%) |
| ✗ | frieren  | Re-stratified loss (1/per_sample_y_std) | #3386 | **CLOSED** — failed (predicted −5–10%, actual +1.7% test regression) |
| ✗ | nezuko   | Multi-scale slice tokens | #3429 | **CLOSED 00:25** — equal-epoch tie with control, wall-clock cost eats budget |
| ✗ | nezuko   | Surface-biased slice routing | #3260 | **CLOSED** (paired −0.05%) |
| ✗ | tanjiro  | Huber loss delta=0.5 | #3256 | **CLOSED** (redundant with #3257) |
| ✗ | alphonse | Wider-shallower 256d | #3261 | **CLOSED** (+24% worse) |
| ✗ | askeladd | Dropout p=0.1 | #3264 | **CLOSED** (+6% worse) |

## Standings — test_avg/mae_surf_p (lower is better)

| Rank | PR | Hypothesis | test_avg (4-split) | vs baseline | Status |
|------|----|------------|-------------------:|-------------|--------|
| **1** | **#3358 (alphonse)** | **cosine T_max=14** | **80.08** | **NEW BASELINE** | **MERGED** |
| 2 | #3468 (thorfinn, old-base) | per-block FiLM v1 (post-block) | 84.00 | +4.9% above new | rebase in flight |
| 3 | #3406 (tanjiro, old-base) | sw=5, paired baseline 94.35 | 88.80 | +10.9% above new | rebase in flight |
| 4 | #3263 (thorfinn) | FiLM(log_Re) | 90.06 | +12.5% above new | MERGED |
| 5 | #3257 (frieren) | Surf-MAE+p-weight 3×+NaN guard | 94.35 | +17.8% above new | MERGED |
| — | #3386 (frieren, v2) | re-stratified loss | 95.98 | +19.9% above new | CLOSED |
| — | #3429 (nezuko, control) | multi-scale OFF (sanity) | 89.62 | +11.9% above new | CLOSED |
| — | #3429 (nezuko, ms-32) | multi-scale ON | 104.60 | +30.6% above new | CLOSED |

## Key R2/R3 predictions (on new #3358 baseline, target test < 80.08)

- **thorfinn per-block FiLM rebased (#3468):** Mechanism orthogonal to cosine (different attack vector). But biggest per-split gains overlap on single_in_dist. Predicted hopeful test ~73–77, conservative ~76–80, pessimistic ~78–82.
- **tanjiro sw=5 rebased (#3406):** sw=5 wins on old base (paired −5.88% test) → if orthogonal to FiLM + cosine, expect test ~75–78 on new base. **Strong merge candidate.**
- **frieren richer FiLM (#3504):** Extend cond_dim 1→11 to include AoA+NACA+gap+stagger. Predicted val ~85–88, test ~76–79. **Strong compose with #3263.**
- **nezuko volume MAE (#3550):** Unifies L1 loss. Conservative expectation: ~1–3% test gain. Predicted val ~88–90, test ~76–79.
- **edward RFF σ=1.0 (#3262):** Feature encoding orthogonal to loss & FiLM. Old gain −9.8% val. Predicted val ~85–90, test ~74–78.
- **fern grad-clip+warmup (#3258):** Grad norms median 56/peak 1000+ with MSE loss — should still help under MAE+FiLM+cosine. Predicted val ~85–90, test ~74–78. **High confidence merge.**
- **askeladd EMA β=0.99 (#3351):** Short EMA horizon ~0.3 epoch averages recent near-converged weights. Now with cosine T_max=14 the LR-end is 0 — EMA may have less value since weights aren't oscillating. Predicted val ~88–92, test ~78–82.
- **alphonse AdamW sweep (#3565):** beta2=(0.999→0.95) + weight_decay=(1e-4→0.05). 3 arms: combined, beta2-only, WD-only. Standard transformer optimizer recipe not yet tested. Predicted val ~85–91, test ~75–81. Conservative expectation: ~2% gain from one knob.

## Open issues / live diagnostics

- **Canonical NaN guard: RESOLVED (in main).** All new branches inherit automatically.
- **Cosine T_max mismatch: RESOLVED (#3358 merged).**
- **Huge gradient norms (median 56, peak 1000):** In flight (#3258 fern rebase).
- **Run-to-run variance (~5–10% on test_avg):** Open issue — nezuko's control vs published baseline (89.62 vs 90.06) confirms. Future borderline calls (<5% gain) should consider multi-seed.
- **GitHub API rate limit:** Recurring 5000-req/hr exhaustion windows; resets every ~hour. Pods training fine.

## Plateau-protocol queue (next R3 hypotheses, ranked)

1. ✗ Re-stratified loss reweighting ← FAILED (#3386 closed)
2. ✗ Multi-scale slice tokens ← FAILED (#3429 closed)
3. ✗ surf_weight sweep ← REBASE IN FLIGHT (#3406)
4. ✗ Per-block FiLM heads ← REBASE IN FLIGHT (#3468)
5. ✗ Richer FiLM conditioning ← IN FLIGHT (#3504)
6. ✗ Volume MAE reformulation ← IN FLIGHT (#3550)
7. ✗ **AdamW betas / weight-decay sweep** ← IN FLIGHT (#3565 alphonse)
8. **Per-block × richer-FiLM compose** (if both #3468 + #3504 land)
9. **Geometry-aware input features** (node distance to nearest surface; may be redundant with dsdf)
10. **Loss decomposition by domain** (per-split loss tracking + dynamic per-split weight)
11. **Soft slice-softmax temperature** (addresses gradient-norm root cause if fern's clip+warmup insufficient)
12. **Surface-only decoder head** (deeper architectural surface specialization, post-failed multi-scale)
13. **TTA at inference** (test-time augmentation)
14. **Variance reduction via fixed `--seed` flag + multi-seed protocol** (nezuko's suggestion — instrumentation)
15. **`single_in_dist` deep-dive** (per-sample Re vs error diagnostic — thorfinn's suggestion)
16. **Re-bucketed sampling via sample_weights** (frieren's suggestion after #3386)

## Next immediate action

**All 8 students active.** Monitor for results from the rebase reruns (#3468 thorfinn, #3406 tanjiro) and first-run assignments (#3504 frieren, #3550 nezuko, #3565 alphonse). Rate limit resets ~01:20 UTC — review next wave then.

**Key monitors:**
- #3468 thorfinn per-block FiLM rerun on cosine base: predicted hopeful test ~73–77
- #3406 tanjiro sw=5 rerun on cosine base: predicted test ~75–78
- #3262 edward RFF σ=1.0: MERGEABLE label set — check when rate limit resets
- #3258 fern grad-clip+warmup: MERGEABLE label set — check when rate limit resets
- #3565 alphonse AdamW sweep: 3 arms, ~90 min total
