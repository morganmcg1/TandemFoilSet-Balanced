# SENPAI Research State

- **Date:** 2026-05-15 18:45
- **Launch:** willow-pai2i-48h-r1 (round 2 in progress)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~14 epochs achievable)
- **Latest direction from human team:** None

## Research contract
Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Primary paper-facing metric also includes `test_avg/mae_surf_p` (all 4 splits valid since PR #3309 merged).

## Current best baseline
- **val_avg/mae_surf_p = 91.3319** (PR #3317, cosine T_max=15)
- **test_avg/mae_surf_p = 88.4260** (3-split, cruise NaN — branch predated NaN fix)
- W&B: `kx17n4pn` (T_max=15 winner arm)

Full metrics in `BASELINE.md`.

## Merged PRs
| PR | Hypothesis | val_avg/mae_surf_p | test_avg/mae_surf_p |
|----|-----------|---------------------|---------------------|
| #3159 | Huber loss δ=0.1 | 112.9001 | 115.7589 (3/4 splits) |
| #3309 | NaN fix (cruise test) | 112.8295 | **106.5996** (4/4 valid) |
| #3317 | Cosine T_max=15 | **91.3319** | 88.4260 (3/4, cruise NaN) |

## Round-2 WIP — 8/8 students assigned
| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3305 | alphonse | Huber δ=0.05 rebase on T_max=15 base | WIP — rebase in progress |
| #3395 | askeladd | Peak LR scan: 3e-4 vs 8e-4 on T_max=15 base | WIP — fresh |
| #3359 | edward | Pressure channel-weighted surf loss (p=3×) | WIP — running |
| #3171 | fern | Split pressure head + 3x p weight, rebase on T_max=15 | WIP — rebase in progress |
| #3174 | frieren | L1 surf pressure + surf_weight=50 | WIP — running |
| #3175 | nezuko | Cosine warmup (5-ep linear) | WIP — started 18:33 |
| #3361 | thorfinn | slice_num 64→128 on Huber+NaN base | WIP — running |
| #3363 | tanjiro | AdamW β2=0.95 + grad clip 1.0 | WIP — running |

## Key insight from round 2 so far
**Schedule alignment is the dominant lever.** T_max=50 with 14 achievable epochs means the cosine schedule decays only to 79% of peak LR — essentially no annealing. T_max=15 gave a 19% improvement in one line. This suggests:
- Experiments running with T_max=50 (edward, frieren, thorfinn, tanjiro, nezuko) are handicapped — each losing ~12 MAE points of "lost annealing"
- The true baseline to beat for any hypothesis is now 91.33 on the T_max=15 base
- WIP PRs without T_max=15 will likely need rebase after results come in

## Active hypotheses being tested
1. **Smaller Huber delta on T_max=15 base** (alphonse): δ=0.05 proved -13% on old base; stacking on T_max=15 should compound
2. **Peak LR scan** (askeladd): optimal peak LR may shift now the schedule actually anneals; testing 3e-4 vs 8e-4
3. **Pressure channel weighting** (edward): 3× gradient on scored channel (T_max=50 — rebase needed if promising)
4. **Split pressure head + 3x weight** (fern): OOD gains of -14 to -23 MAE in v2; rebase onto T_max=15 base
5. **L1 surf + surf_weight=50** (frieren): surface-pressure gradient amplification (T_max=50)
6. **Cosine warmup** (nezuko): 5-epoch linear warmup into peak LR
7. **slice_num=128 on Huber+NaN base** (thorfinn): capacity retrial (T_max=50)
8. **AdamW stability** (tanjiro): β2=0.95 + grad clip 1.0 (T_max=50)

## Expected next decisions
- When edward, frieren, thorfinn, tanjiro results land:
  - Beat new baseline (91.33) → merge
  - Beat old baseline (112.90) but not new → send back for rebase on T_max=15
  - Failed even old baseline → close with analysis
- alphonse rebase + fern rebase: likely winners if δ=0.05 and split-head compound with T_max=15
- askeladd LR scan: may give further ~2-5% improvement

## Next research directions after round 2
1. **Combine δ=0.05 + T_max=15** — stack orthogonal improvements (alphonse rebase)
2. **Split head + T_max=15** — confirm OOD improvement carries on new base (fern rebase)
3. **T_max fine-tuning** (14 vs 15 vs 16) — exact-match epoch budget
4. **surf_weight tuning on T_max=15 base** — optimal surface emphasis with proper annealing
5. **Cosine warm restarts (T_0=5)** — multiple cycles in 14-epoch budget
6. **Per-domain normalization** — pressure ranges differ by split
7. **Train-time symmetry augmentation** — horizontal flip (camber-aware)
8. **Unified positional encoding** — unified_pos=True toggle
