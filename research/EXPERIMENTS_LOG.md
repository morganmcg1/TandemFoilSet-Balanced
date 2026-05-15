# SENPAI Research Results

_Branch: icml-appendix-charlie-pai2i-48h-r3_
_Primary metric: val_avg/mae_surf_p (lower is better)_

---

## Round 1 — PRs assigned 2026-05-15

Eight first-round experiments assigned simultaneously to probe the primary bottlenecks in the Transolver baseline.

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3154 | alphonse | H5: n_hidden 128→256 | WIP |
| #3156 | askeladd | H1: p-channel surf loss x3, x5 | WIP |
| #3158 | edward | H2: EMA decay=0.999 | WIP |
| #3160 | fern | H4: Huber loss delta=1.0, 0.5 | WIP |
| #3163 | frieren | H3: Grad clip max_norm=1.0 + 5-ep warmup | WIP |
| #3166 | nezuko | H7: FiLM Re/AoA conditioning | WIP |
| #3168 | tanjiro | H10: slice_num 64→128, 96 | WIP |
| #3170 | thorfinn | H11: n_layers 5→7, 5→8 | WIP |

Results will be appended as PRs complete.

---

## 2026-05-15 14:10 — PR #3166: H7: FiLM Re/AoA conditioning on Transolver blocks

- Branch: `charliepai2i48h3-nezuko/h7-film-re-aoa-conditioning`
- Hypothesis: FiLM conditioning extracts global flow params (log(Re), AoA, NACA, gap, stagger = 11 dims) from input and applies per-layer (γ, β) affine modulation on each Transolver block's post-attention output. Identity-initialized ConditionMLP. Implementation: `cond = x[:, 0, 13:]` (node 0, always real — avoids padding dilution of mean).

| Split | val_avg/mae_surf_p |
|---|---|
| val_single_in_dist | 129.7991 |
| val_geom_camber_rc | 129.0683 |
| val_geom_camber_cruise | 94.8233 |
| val_re_rand | 104.8163 |
| **val_avg** | **114.6268** |
| test_single_in_dist | 111.0206 |
| test_geom_camber_rc | 118.4236 |
| test_geom_camber_cruise | NaN (⚠ scoring bug) |
| test_re_rand | 104.0226 |
| **test_avg (3-split)** | **111.155** |

- Artifacts: `models/model-charliepai2i48h3-nezuko-h7-film-re-aoa-cond-20260515-131301/metrics.jsonl`
- Params: 835K (n_hidden=128, 5 layers, cond_dim=11)
- Peak memory: 13.0 GB / 96 GB H100
- Best epoch: 14 / 50 (cut by 30-min timeout; curve still improving at cutoff)

**Analysis:** Training cut at epoch 14/50 — model not converged. Val was still dropping rapidly (138.7 → 117.3 → 114.6 over last 3 epochs). Result is underestimate of converged FiLM performance. val_re_rand (104.82) is the lowest split, consistent with Re-conditioning hypothesis. val_geom_camber_rc (129.07) hardest split — Re conditioning may not help geometry generalization as much.

**Key findings:**
1. FiLM identity-init works well — no early instability despite the added conditioning.
2. Cosine T_max=50 with only 14 epochs means LR stayed near 4e-4 throughout — no annealing benefit. This affects ALL Round 1 experiments equally.
3. `data/scoring.py` has NaN propagation bug: sample index 20 of test_geom_camber_cruise has non-finite GT, and `nan * 0 = nan` in masked sum. All val metrics unaffected.

**Status: MERGED** — establishes first val baseline at 114.63.

**Follow-up for R2:** Run clean cond_dim=0 baseline to isolate FiLM's true contribution. Fix T_max to match actual training duration for all students.
