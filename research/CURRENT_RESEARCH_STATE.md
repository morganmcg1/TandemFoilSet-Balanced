# SENPAI Research State — TandemFoilSet (willow-pai2i-24h-r4)

- **As of:** 2026-05-15 20:05 UTC
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r4`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced`
- **W&B:** `wandb-applied-ai-team/senpai-v1`
- **Most recent human researcher direction:** None recorded. Launch isolation rules are in force.

## Research programme summary

Predict (Ux, Uy, p) at every node of unstructured 2D CFD meshes (74K–242K nodes) of tandem airfoils. The primary ranking metric is `test_avg/mae_surf_p` — equal-weighted surface pressure MAE across four test splits. Per-run budget is 30 min wall clock × 50 epochs hard cap.

## Current best (BASELINE.md)

**PR #3257 (frieren, merged 2026-05-15 18:25 UTC) — surface MAE + p-weight 3× + canonical NaN guard**
- `val_avg/mae_surf_p = 106.67` (W&B `szru1ogx`)
- `test_avg/mae_surf_p = **94.35**` (4-split finite, new R1 anchor)

All remaining PRs must beat **test_avg/mae_surf_p < 94.35**.

## Full R1 portfolio status

| # | Student | Hypothesis | PR | Status |
|---|---------|-----------|-----|--------|
| 1 | frieren  | Surface MAE + p-weight 3× + NaN guard | #3257 | **MERGED (R1 winner #1)** — test=94.35 |
| 2 | fern     | Grad-clip 1.0 + 5-epoch warmup | #3258 | **sent back for rebase** (rebased 18:35, re-running on frieren-base) |
| 3 | edward   | RFF σ=1.0 on (x, z) | #3262 | **sent back for rebase** (rebased 18:36, re-running on frieren-base) |
| 4 | thorfinn | FiLM log(Re) conditioning | #3263 | **sent back for rebase** (v3 val=118.55 vs old base; rebased, re-running) |
| 5 | nezuko   | Multi-scale slice tokens (coarse-global + fine-surface groups) | #3429 | WIP (replaces closed #3260; #3260 was −0.05% paired) |
| 6 | alphonse | Cosine T_max fix (T_max=14) | #3358 | WIP |
| 7 | askeladd | EMA weights β=0.999 | #3351 | WIP |
| 8 | frieren  | Re-stratified loss reweighting | #3386 | WIP (just assigned) |
| 9 | tanjiro  | surf_weight sweep {5,10,20} | #3406 | WIP (just assigned, replaces closed #3256) |
| ✗ | nezuko   | Surface-biased slice routing | #3260 | **CLOSED** (paired −0.05%; reassigned to #3429) |
| ✗ | tanjiro  | Huber loss delta=0.5 | #3256 | **CLOSED** (redundant with #3257; no commits in 6h) |
| ✗ | alphonse  | Wider-shallower 256d | #3261 | **CLOSED** (+24% worse) |
| ✗ | askeladd | Dropout p=0.1 | #3264 | **CLOSED** (+6% worse; diagnosed NaN root cause) |

## Standings — test_avg/mae_surf_p (lower is better)

| Rank | PR | Hypothesis | test_avg (4-split) | vs baseline | Status |
|------|----|------------|-------------------:|-------------|--------|
| **1** | **#3257 (frieren)** | **Surf-MAE+p-weight 3×+NaN guard** | **94.35** | **NEW BASELINE** | **MERGED** |
| — | vanilla `xfayvdk2` (alphonse) | NaN-guarded MSE | 106.23 | −11.2% above new | anchor |
| — | #3258 (fern, old-base) | clip+warmup, old base | 105.70 | +12.0% above new | rebase in flight |
| — | #3263 (thorfinn, old-base) | FiLM log(Re), 3-split | ~119 | >>> above new | rebase in flight |

## Key R2 predictions (on frieren-base)

- **fern clip+warmup (#3258 rebase):** Grad norms median 56/peak 1000+ with MSE loss — likely still high under MAE loss. Predicted val ~90–100, test ~82–88. **High confidence merge.**
- **alphonse cosine T_max=14 (#3358):** Free win from schedule alignment. Predicted 2–5% gain. **Easy compound.**
- **edward RFF σ=1.0 (#3262 rebase):** Feature encoding orthogonal to loss. Old gain −9.8% val. Predicted val ~96–101, test ~84–91.
- **thorfinn FiLM log(Re) (#3263 rebase):** Re conditioning largest on cruise/re_rand splits. Predicted val ~92–98, test ~83–88.
- **frieren re-stratified loss (#3386):** Equalize per-sample gradient contribution by 1/std. Predicted 2–6% val, biggest on cruise.
- **tanjiro surf_weight sweep (#3406):** Effective 30× weighting with p-weight=3 baked in. Predicted sw=5 → val ~100–104, test ~88–92.
- **askeladd EMA (#3351):** Free 2–4% from weight averaging.
- **nezuko surf-biased slice (#3260):** Architectural, high-variance bet. Needs rebase.

## Open issues / live diagnostics

- **Canonical NaN guard: RESOLVED (in main).** All new branches inherit automatically.
- **Cosine T_max mismatch:** Still in flight (#3358 alphonse).
- **Huge gradient norms (median 56, peak 1000):** Not yet addressed at root. R2 candidate: soft slice-softmax temperature.
- **GitHub API rate limit:** Intermittent 403s affecting tanjiro/nezuko pods most severely — sessions abort before reaching commit step. Auto-recovers; not blocking but costs time-per-iteration.
- **Run-to-run variance (±5–10pt on val_avg):** Expected to improve once fern's clip+warmup merges.

## Plateau-protocol queue (next R2 hypotheses, ranked)

1. Re-stratified loss reweighting ← **assigned to frieren #3386**
2. surf_weight sweep {5,10,20} ← **assigned to tanjiro #3406**
3. Multi-scale slice tokens (coarse global + fine surface-focused)
4. Geometry-aware input features (node distance to nearest surface)
5. Loss decomposition by domain (per-split loss tracking)
6. Soft slice-softmax temperature diagnostic (addresses gradient norm root cause)
7. Per-block FiLM (natural R2 for thorfinn after #3263 win)
8. Richer FiLM conditioning `(log_Re, AoA_1, AoA_2, gap, stagger)`
9. AdamW betas / weight-decay sweep
