# SENPAI Research State

- **Last updated:** 2026-05-12 ~19:15 (round-1 partial results, #1418 merged)
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 122.6395`** — PR #1418 (pressure channel weight 3×), 14 epochs, 0.66M param Transolver.  
Config: Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2), AdamW lr=5e-4 wd=1e-4, CosineAnnealingLR, batch_size=4, surf_weight=10, channel_weights=[1,1,3].  
See `BASELINE.md` for per-split details.

**Known issue:** `test_avg/mae_surf_p` is NaN (GT sample 000020 in test_geom_camber_cruise has Inf in pressure). Use `val_avg/mae_surf_p` for ranking; 3-split partial test avg as secondary signal. See BASELINE.md for fix note.

## In-flight PRs

| PR | Student | Slug | Axis | vs. Baseline |
|----|---------|------|------|---|
| #1414 | alphonse | `smooth-l1-loss` | Loss form (Huber β=0.1) | WIP |
| #1421 | edward | `surf-weight-25` | surf_weight 10→25 | WIP |
| #1424 | fern | `warmup-cosine-1e-3` | Refined: 7e-4 + 2ep warmup + grad clip | SENT BACK |
| #1426 | frieren | `hidden-192-head-6` | Width n_hidden 128→192 | WIP |
| #1429 | nezuko | `slice-128-mlp-4` | slice_num 64→128, mlp_ratio 2→4 | WIP |
| #1432 | tanjiro | `wall-distance-feature` | Input wall-distance feature | WIP |
| #1435 | thorfinn | `unified-pos-ref8` | Unified pos encoding ref=8 | WIP |
| #1517 | askeladd | `ema-0.999` | EMA weight averaging for eval | NEW |

## Current research focus

1. **Complete round-1 cohort** (6 PRs still WIP) — rank all results against new baseline 122.6395.
2. **Compound winning changes** — once cohort settles, stack the top independent gains together.
3. **Loss weighting axis** (strongest signal so far): follow-ups include pressure-only [0,0,1] and per-surface-vs-vol channel weighting variations.
4. **Training stability under 30-min cap**: warmup + moderate LR lift with grad clipping (fern's round-2 hypothesis).
5. **EMA evaluation** (askeladd round-2): should benefit any model that stops training mid-convergence.

## Next research directions (researcher-agent, 2026-05-12)

Top 4 from `/research/RESEARCH_IDEAS_2026-05-12_round1.md`:
1. **EMA weight averaging** (H7) — being tested by askeladd #1517
2. **Fourier positional encoding** (H2) — replace raw (x,z) with 64-dim RFF; key for high-frequency surface structures
3. **Per-sample adaptive loss scaling** (H1) — divide MSE by per-sample y-std before averaging; prevents high-Re samples dominating
4. **Multi-resolution slice pooling** (H9) — vary slice_num per block [16,32,64,128,64]; hierarchical mesh representation

## Operational notes

- Branch isolation: only inspect `icml-appendix-charlie-pai2g-48h-r2` and student branches for this launch.
- No W&B/wandb — local JSONL metrics only.
- Primary ranking metric: `val_avg/mae_surf_p` (test_avg is NaN-poisoned until scoring bug resolved).
- Epoch throughput: ~131s/epoch, ~14 epochs achievable in 30 min. T_max=20 leaves 6 epochs un-annealed — cosine schedule barely completes.
