# SENPAI Research State — TandemFoilSet (willow-pai2i-24h-r4)

- **As of:** 2026-05-15 23:25 UTC
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r4`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced`
- **W&B:** `wandb-applied-ai-team/senpai-v1`
- **Most recent human researcher direction:** None recorded. Launch isolation rules are in force.

## Research programme summary

Predict (Ux, Uy, p) at every node of unstructured 2D CFD meshes (74K–242K nodes) of tandem airfoils. The primary ranking metric is `test_avg/mae_surf_p` — equal-weighted surface pressure MAE across four test splits. Per-run budget is 30 min wall clock × 50 epochs hard cap.

## Current best (BASELINE.md)

**PR #3263 (thorfinn, merged 2026-05-15 21:55 UTC) — FiLM(log_Re) on Transolver hidden state (on frieren base)**
- `val_avg/mae_surf_p = 100.24` (W&B `69jp9tvt`)
- `test_avg/mae_surf_p = **90.06**` (4-split finite, all splits improved)

Prior baseline: #3257 frieren surf-MAE+p_weight=3, val=106.67, test=94.35.

All remaining PRs must beat **test_avg/mae_surf_p < 90.06**.

## Active R2 portfolio (all 8 students busy)

| # | Student | PR | Hypothesis | Status |
|---|---------|----|-----------|--------|
| 1 | frieren  | #3504 | Richer FiLM conditioning (cond_dim=1→11: log_Re+AoA+NACA+gap+stagger) | WIP (assigned 23:23 UTC) |
| 2 | thorfinn | #3468 | Per-block FiLM heads (5 FiLM heads, one per Transolver block) | WIP |
| 3 | tanjiro  | #3406 | surf_weight sweep — **sw5 winner on OLD base** (test=88.80 vs 94.35); rebase + rerun on FiLM+frieren base | WIP (sent back 23:21) |
| 4 | alphonse | #3358 | Cosine T_max=14 fix (paired −6.4%/-9.2% on old base); rebase + rerun on FiLM+frieren base | WIP (rebase in flight) |
| 5 | askeladd | #3351 | EMA β=0.99 (shorter horizon, β=0.999 val=131.37 on old base) | WIP (CONFLICTING, pod just picked up at iter 10) |
| 6 | edward   | #3262 | RFF σ=1.0 on (x, z) — orthogonal to FiLM, MERGEABLE | WIP |
| 7 | fern     | #3258 | Grad-clip 1.0 + 5-epoch warmup — MERGEABLE post-FiLM | WIP |
| 8 | nezuko   | #3429 | Multi-scale slice tokens (coarse-global + fine-surface) — earlier run test=106.71, may need new direction | WIP (pod picked up at iter 100) |

## R1 closed/merged history

| # | Student | Hypothesis | PR | Status |
|---|---------|-----------|-----|--------|
| 1 | frieren  | Surface MAE + p-weight 3× + NaN guard | #3257 | **MERGED (R1 winner #1)** — test=94.35 |
| 2 | thorfinn | FiLM log(Re) conditioning | #3263 | **MERGED (R1 winner #2)** — val=100.24, test=90.06 (−4.55%) |
| ✗ | frieren  | Re-stratified loss (1/per_sample_y_std) | #3386 | **CLOSED 23:21** — failed (predicted −5–10%, actual +1.7% test regression) |
| ✗ | nezuko   | Surface-biased slice routing | #3260 | **CLOSED** (paired −0.05%) |
| ✗ | tanjiro  | Huber loss delta=0.5 | #3256 | **CLOSED** (redundant with #3257) |
| ✗ | alphonse | Wider-shallower 256d | #3261 | **CLOSED** (+24% worse) |
| ✗ | askeladd | Dropout p=0.1 | #3264 | **CLOSED** (+6% worse) |

## Standings — test_avg/mae_surf_p (lower is better)

| Rank | PR | Hypothesis | test_avg (4-split) | vs baseline | Status |
|------|----|------------|-------------------:|-------------|--------|
| **1** | **#3263 (thorfinn)** | **FiLM(log_Re) on hidden state** | **90.06** | **NEW BASELINE** | **MERGED** |
| 2 | #3257 (frieren) | Surf-MAE+p-weight 3×+NaN guard | 94.35 | +4.8% above new | MERGED |
| — | #3406 (tanjiro, old-base) | sw=5, paired baseline 94.35 | 88.80 | (unpaired vs new) | rebase in flight |
| — | #3358 (alphonse, old-base) | cosine T_max=14, paired baseline | 104.95 | +16.5% above new | rebase in flight |
| — | #3386 (frieren, v2) | re-stratified loss | 95.98 | +6.6% above new | CLOSED |
| — | #3429 (nezuko) | multi-scale slice tokens (10ep) | 106.71 | +18.5% above new | likely close |
| — | vanilla `xfayvdk2` (alphonse) | NaN-guarded MSE | 106.23 | +18.0% above new | anchor |

## Key R2 predictions (on FiLM+frieren base, target test < 90.06)

- **tanjiro sw=5 rebased (#3406):** sw=5 wins on old base (paired −5.88% test) → if mechanism is orthogonal to FiLM, expect test ~83–87 on new base. **Strong merge candidate.**
- **alphonse cosine T_max=14 (#3358):** Free win from schedule alignment, paired −6.4%/-9.2% on old base. Predicted 2–5% gain on new base. **Easy compound.**
- **edward RFF σ=1.0 (#3262):** Feature encoding orthogonal to loss & FiLM. Old gain −9.8% val. Predicted val ~92–97, test ~82–88.
- **fern grad-clip+warmup (#3258):** Grad norms median 56/peak 1000+ with MSE loss — should still help under MAE+FiLM. Predicted val ~88–95, test ~80–86. **High confidence merge.**
- **frieren richer FiLM (#3504):** Extend cond_dim 1→11 to include AoA+NACA+gap+stagger. Predicted val ~94–98, test ~84–88. **Strong compose with #3263.**
- **thorfinn per-block FiLM (#3468):** 5 FiLM heads, one per block. Predicted val ~95–98, test ~85–88. Orthogonal-ish to #3504 (different mechanism: depth vs conditioning width).
- **askeladd EMA β=0.99 (#3351):** Short EMA horizon ~0.3 epoch averages recent near-converged weights. Predicted val ~98–103, test ~86–92.
- **nezuko multi-scale slice (#3429):** Run finished `5wvllm86`, test=106.71 — +18.5% above target, likely close.

## Open issues / live diagnostics

- **Canonical NaN guard: RESOLVED (in main).** All new branches inherit automatically.
- **Cosine T_max mismatch:** In flight (#3358 alphonse rebase).
- **Huge gradient norms (median 56, peak 1000):** In flight (#3258 fern rebase).
- **GitHub API rate limit:** Recurring 5000-req/hr exhaustion windows; resets every ~hour. Pods training fine; advisor periodically blocked.
- **Run-to-run variance (±3–10pt on val_avg):** Expected to improve once fern's clip+warmup merges.

## Plateau-protocol queue (next R3 hypotheses, ranked)

1. ✗ Re-stratified loss reweighting ← FAILED (#3386 closed)
2. ✗ surf_weight sweep ← REBASED IN FLIGHT (#3406)
3. ✗ Per-block FiLM heads ← IN FLIGHT (#3468)
4. ✗ Richer FiLM conditioning ← IN FLIGHT (#3504)
5. **Geometry-aware input features (node distance to nearest surface)** ← next free slot
6. **Per-block × richer-FiLM compose** (only if both #3468 + #3504 land)
7. **Loss decomposition by domain** (per-split loss tracking + dynamic per-split weight)
8. **Soft slice-softmax temperature** (addresses gradient-norm root cause if fern's clip+warmup is insufficient)
9. **AdamW betas / weight-decay sweep** (cheap orthogonal compose)
10. **Volume MAE reformulation** (tanjiro's suggested follow-up — currently vol uses MSE)
11. **Variance reduction via fixed `--seed` flag** (instrumentation, not a hypothesis)
12. **Re-bucketed sampling via sample_weights** (frieren's suggested follow-up after #3386 close)
