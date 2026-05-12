# SENPAI Research State

- **As of:** 2026-05-12 (updated cycle 10)
- **Round:** willow-pai2g-48h-r4 (advisor branch `icml-appendix-willow-pai2g-48h-r4`)
- **Most recent human-team direction:** (none — controlled 24/48 h Charlie-vs-Willow logging ablation, hard cap `SENPAI_TIMEOUT_MINUTES=30`)

## Current baseline

**`val_avg/mae_surf_p = 98.1642`** — PR #1558 (Huber surface loss delta=0.5), merged 2026-05-12 22:00.
**Test 3-split mean: 98.7537** (cruise still NaN, but all other splits now reportable via #1527 guard).

## Improvement trajectory

| Cycle | PR | Change | val_avg/mae_surf_p | Δ |
|-------|-----|--------|-------------------|---|
| 2 | #1502 | BIVW (per-sample IVW) | 126.0751 | baseline |
| 3 | #1528 | BIVW + zero-init surf-head | 119.2987 | −5.37% |
| 5 | #1527 | Test NaN guard (eval only) | (val unchanged) | infra |
| 10 | **#1558** | **Huber surface loss δ=0.5** | **98.1642** | **−17.72%** |

## Current research focus

**Cycle 10.** Huber surface loss is now the dominant improvement with 17.7% gain. The three merged ML improvements (BIVW + surf-head + Huber) each target orthogonal sources of gradient error:
1. BIVW: between-sample variance (different Re)
2. Surf-head: surface vs volume architectural specialisation
3. Huber: within-sample per-node outlier residuals

Key open questions:
1. Is delta=0.5 optimal, or do smaller deltas (0.2, 0.3) give further gain? (thorfinn #1627)
2. Does grad-clip + higher LR stack on top of the new 98.16 baseline? (fern #1499 rebase)
3. Does per-channel BIVW help further? (tanjiro #1580)
4. Does BF16/AMP unlock capacity? (frieren #1572)
5. What do the long-running WIP experiments show? (#1496, #1497, #1498, #1501 — pods recovering from rate limit issues)

## Live PRs

| # | Student | Slug | Status | Notes |
|---|---------|------|--------|-------|
| 1496 | alphonse | pressure-channel-prioritized-loss | WIP (stale) | Rate-limit affected, training now resumed |
| 1497 | askeladd | warmup-cosine-lr | WIP (stale) | Rate-limit affected, training now resumed |
| 1498 | edward | wider-mlp-ratio (2 to 4) | WIP (stale) | Rate-limit affected, resumed iteration 37 |
| 1499 | fern | gradient-clipping-and-higher-lr | WIP (rebase) | Rebasing; need results vs new 98.16 baseline |
| 1501 | nezuko | more-slices (64 to 128) | WIP | Replied: #1527 fix merged, rebase + re-run |
| 1572 | frieren | bf16-mixed-precision | WIP | BF16 arms running |
| 1580 | tanjiro | per-channel-bivw | WIP | New assignment |
| 1627 | thorfinn | huber-delta-sweep | WIP | delta=0.2 and 0.3 arms; new assignment |

## Working hypotheses

1. **BIVW** — confirmed (PR #1502, −5.4%).
2. **BIVW + surf-head** — confirmed (PR #1528, −5.4% additional).
3. **Huber surface loss delta=0.5** — confirmed (PR #1558, **−17.7%**). The largest gain yet.
4. **Smaller Huber delta** — testing (PR #1627). surf_l1_frac high at delta=0.5; smaller delta may push more into L1 regime.
5. **Grad-clip + higher LR** — rebasing on new 98.16 baseline (#1499).
6. **Per-channel BIVW** — testing (#1580).
7. **BF16/AMP** — testing (#1572); primarily for capacity headroom.
8. **Capacity (MLP width, slices, bigger hidden)** — still WIP (#1498, #1501); need comparison vs 98.16.
9. **Warmup schedule** — WIP (#1497); need comparison vs 98.16.
10. **Pressure-channel emphasis** — WIP (#1496); need comparison vs 98.16.

## Closed / rejected hypotheses

- **PR #1503** (standalone surf-head, no BIVW) — 6.2% worse.
- **PR #1500** (n_hidden=256 at FP32) — budget failure; revisiting with BF16.

## Potential next directions

- **Smaller Huber delta** (delta=0.1, 0.2, 0.3) — thorfinn sweeping now (#1627)
- **Huber on volume loss too** — may help volume channels indirectly
- **Per-channel delta** — different delta per channel (p vs Ux/Uy)
- **Compose BIVW + Huber + clip+LR** — if #1499 rebase beats 98.16
- **Compose per-channel BIVW + Huber** — if #1580 beats 98.16
- **Capacity scaling (n_hidden=256) + BF16 + Huber** — full composition
- **Re-bin stratified sampler** — data-level Re balancing
- **LR schedule tuning on Huber recipe** — warmup + cosine on the new optimum
- **surf_weight tuning** — with Huber loss active, optimal surf_weight may have shifted

## Known issues

- ~~**test_avg/mae_surf_p = NaN**~~ — Fixed via PR #1527 (merged). 3-split test mean now reportable.
- **Rate limit impact**: GitHub GraphQL rate limits caused multiple student pods to see "No assigned PRs" and idle for 30–60 min. Pods recovering as rate limit resets.
- **Slice-attention VRAM**: n_hidden=256 needs BF16 (#1572) to be fairly evaluated.
