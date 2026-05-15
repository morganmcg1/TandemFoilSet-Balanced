# SENPAI Research State — TandemFoilSet (willow-pai2i-24h-r4)

- **As of:** 2026-05-15 21:55 UTC
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

## Full R1 portfolio status

| # | Student | Hypothesis | PR | Status |
|---|---------|-----------|-----|--------|
| 1 | frieren  | Surface MAE + p-weight 3× + NaN guard | #3257 | **MERGED (R1 winner #1)** — test=94.35 |
| 2 | fern     | Grad-clip 1.0 + 5-epoch warmup | #3258 | **sent back for rebase** (rebased 18:35, re-running on frieren-base) |
| 3 | edward   | RFF σ=1.0 on (x, z) | #3262 | **sent back for rebase** (rebased 18:36, re-running on frieren-base) |
| 4 | thorfinn | FiLM log(Re) conditioning | #3263 | **MERGED (R1 winner #2)** — val=100.24, test=90.06 (−4.55% on primary) |
| 5 | nezuko   | Multi-scale slice tokens (coarse-global + fine-surface groups) | #3429 | WIP (run finished `5wvllm86`, test=106.71 +13% worse — likely close) |
| 6 | alphonse | Cosine T_max fix (T_max=14) | #3358 | **sent back for rebase** (paired −6.4%/-9.2% on old base; rerun on new merged base) |
| 7 | askeladd | EMA weights β=0.999 → β=0.99 on frieren-base | #3351 | **sent back for rebase** (β=0.999 val=131.37 vs old base; rebased + retry with β=0.99 shorter horizon) |
| 8 | frieren  | Re-stratified loss reweighting | #3386 | WIP (just assigned) |
| 9 | tanjiro  | surf_weight sweep {5,10,20} | #3406 | WIP (just assigned, replaces closed #3256) |
| ✗ | nezuko   | Surface-biased slice routing | #3260 | **CLOSED** (paired −0.05%; reassigned to #3429) |
| ✗ | tanjiro  | Huber loss delta=0.5 | #3256 | **CLOSED** (redundant with #3257; no commits in 6h) |
| ✗ | alphonse  | Wider-shallower 256d | #3261 | **CLOSED** (+24% worse) |
| ✗ | askeladd | Dropout p=0.1 | #3264 | **CLOSED** (+6% worse; diagnosed NaN root cause) |

## Standings — test_avg/mae_surf_p (lower is better)

| Rank | PR | Hypothesis | test_avg (4-split) | vs baseline | Status |
|------|----|------------|-------------------:|-------------|--------|
| **1** | **#3263 (thorfinn)** | **FiLM(log_Re) on hidden state** | **90.06** | **NEW BASELINE** | **MERGED** |
| 2 | #3257 (frieren) | Surf-MAE+p-weight 3×+NaN guard | 94.35 | +4.8% above new | MERGED |
| — | #3358 (alphonse, old-base) | cosine T_max=14, paired baseline | 104.95 | +16.5% above new | rebase in flight |
| — | #3429 (nezuko) | multi-scale slice tokens (10ep) | 106.71 | +18.5% above new | likely close |
| — | vanilla `xfayvdk2` (alphonse) | NaN-guarded MSE | 106.23 | +18.0% above new | anchor |

## Key R2 predictions (on FiLM+frieren base, target test < 90.06)

- **fern clip+warmup (#3258 rebase):** Grad norms median 56/peak 1000+ with MSE loss — likely still high under MAE+FiLM. Predicted val ~88–95, test ~80–86. **High confidence merge.**
- **alphonse cosine T_max=14 (#3358 rebase):** Free win from schedule alignment, paired −6.4%/-9.2% confirmed on old base. Predicted 2–5% gain on new base. **Easy compound.**
- **edward RFF σ=1.0 (#3262 rebase):** Feature encoding orthogonal to loss & FiLM. Old gain −9.8% val. Predicted val ~92–97, test ~82–88.
- **frieren re-stratified loss (#3386):** Equalize per-sample gradient contribution by 1/std. Predicted 2–6% val, biggest on cruise.
- **tanjiro surf_weight sweep (#3406):** Effective 30× weighting with p-weight=3 baked in. Predicted sw=5 → val ~95–100, test ~86–90.
- **askeladd EMA β=0.99 (#3351 rebase):** Short EMA horizon ~0.3 epoch averages recent near-converged weights. Predicted val ~98–103, test ~86–92.
- **nezuko multi-scale slice (#3429):** Run finished `5wvllm86`, test=106.71 — +18.5% above new target, likely close.

## Open issues / live diagnostics

- **Canonical NaN guard: RESOLVED (in main).** All new branches inherit automatically.
- **Cosine T_max mismatch:** Still in flight (#3358 alphonse).
- **Huge gradient norms (median 56, peak 1000):** Not yet addressed at root. R2 candidate: soft slice-softmax temperature.
- **GitHub API rate limit:** Intermittent 403s affecting tanjiro/nezuko pods most severely — sessions abort before reaching commit step. Auto-recovers; not blocking but costs time-per-iteration.
- **Run-to-run variance (±5–10pt on val_avg):** Expected to improve once fern's clip+warmup merges.

## Plateau-protocol queue (next R2/R3 hypotheses, ranked)

1. Re-stratified loss reweighting ← **assigned to frieren #3386**
2. surf_weight sweep {5,10,20} ← **assigned to tanjiro #3406**
3. Per-block FiLM heads (natural follow-up to #3263 win) ← **to assign to thorfinn**
4. Richer FiLM conditioning `(log_Re, AoA_1, AoA_2, gap, stagger)`
5. Geometry-aware input features (node distance to nearest surface)
6. Loss decomposition by domain (per-split loss tracking)
7. Soft slice-softmax temperature diagnostic (addresses gradient norm root cause)
8. AdamW betas / weight-decay sweep
9. Variance reduction via fixed `--seed` flag (instrumentation, not a hypothesis)
