# SENPAI Research State

- **Last updated:** 2026-05-16 ~11:30 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline

**val_avg/mae_surf_p = 64.2430, test_avg/mae_surf_p = 55.5454** — from PR #3814 (askeladd, SwiGLU FFN), merged 2026-05-16 ~11:30 UTC. See `BASELINE.md` for full details.

Per-split test (dvcj6w25): single_in_dist=64.10, geom_camber_rc=66.03, geom_camber_cruise=37.61, re_rand=54.44.

Baseline progression (val_avg/mae_surf_p):
- #3091: 109.42 (warmup + clip + lr=1e-3, MSE)
- #3089: 100.53 (L1 loss + scoring fix)
- #3507: 96.10 (n_hidden=160 width scaling)
- #3372: 88.24 (Fourier PE 4-freq, lr=1e-3)
- #3632: 83.50 (coord noise augmentation std=0.01, lr=5e-4)
- #3691: 82.50 (--epochs 12 longer training, 3-seed mean val=82.96)
- **#3814: 64.24 (SwiGLU FFN in TransolverBlock, −22% vs prev best) ← CURRENT**

## Winning stack (all additive, all merged)

| Component | PR | val gain | Notes |
|---|---|---|---|
| L1 loss + scoring NaN fix | #3089 | −8.1% | Canonical loss |
| n_hidden=160 | #3507 | −4.4% | Width sweet spot for 30min budget |
| Fourier PE num_freq=4 | #3372 | −8.2% | log-spaced sinusoidal on (x,z), lr=1e-3 |
| Coord noise std=0.01 | #3632 | −5.4% | Spatial augmentation during training, lr=5e-4 |
| --epochs 12 (longer training) | #3691 | −1.2% | Cosine T_max=12; best_epoch=11 in all 3 seeds |
| **SwiGLU FFN** | **#3814** | **−22.1%** | Gated FFN in TransolverBlock; inner_dim=216; mlp2 (output head) left as standard MLP |

**Total improvement from baseline:** 109.42 → 64.24 (−41.3%)

## Most recent research direction from human researcher team

No GitHub Issues open for this track as of last check. Proceeding from the program contract only.

## Cross-cutting findings (apply to ALL in-flight PRs)

1. **SwiGLU FFN is now the default** (merged in #3814). All new experiments build on this.
2. **L1 loss is the default** (Config.loss_type = "l1").
3. **n_hidden=160 is the width sweet spot** — n_hidden=176 and n_hidden=192 both regress (pre-SwiGLU; may be worth retesting on SwiGLU stack).
4. **Fourier PE num_freq=4 is the default** — confirmed sweet spot.
5. **coord_noise_std=0.01 is the default** (merged in #3632).
6. **lr=5e-4 is the default** — lr=1e-3 + coord noise regressed in 3-seed test (#3690).
7. **--epochs 12** is the recommended training budget. SwiGLU run was at --epochs 10, best val still at final epoch — strongly suggests epochs=12 will stack further.
8. **Grad clip max_norm=1.0**, warmup 2 epochs, batch=4.
9. **SwiGLU key detail:** `mlp2` (output head) is left as standard MLP; only `self.mlp` in TransolverBlock is replaced. inner_dim=216 = round_to_mult(160*2*2/3, 8).
10. **DSDF distribution finding (nezuko #3836):** normalized DSDF max abs=2.88 → clip=3.0 is a no-op. Clip=2.0 or 2.5 would actually touch 0.33-1.37% of values.

## Active in-flight PRs (status as of 11:30 UTC)

| # | Student | Hypothesis | State | val_avg/mae_surf_p |
|---|---|---|---|---|
| **#3814** | askeladd | **SwiGLU FFN** | **MERGED 11:30** → new baseline | 64.2430 🏆 |
| **#3838** | alphonse | per-domain output norm | **CLOSED 11:30** (val=89.28 FAIL vs both 82.50 and 64.24) | — |
| **#3741** | fern | eta_min=1e-5 cosine floor | **CLOSED 11:30** (best val=86.21 FAIL × 3 seeds) | — |
| **#3815** | tanjiro | TTA coord noise K=4/K=8 | **stale_wip** — 0 comments yet; checking W&B | pending review |
| **#3833** | thorfinn | OneCycleLR schedule | WIP (assigned 08:35) | awaiting |
| **#3835** | edward | asinh output transform | WIP (assigned 08:35) | awaiting |
| **#3836** | nezuko | DSDF clip pivoted to 2.0/2.5 | WIP — pivoted from clip=3 (no-op) to clip=2.0/2.5 per sanity check | awaiting |
| **#3857** | frieren | attention dropout p=0.1/0.2 | WIP (assigned 09:30, pre-SwiGLU baseline) | awaiting |

## Round-5 new assignments (pending, to be created after rate-limit reset at 11:20 UTC)

| Student | Slug | Hypothesis | Rationale |
|---|---|---|---|
| askeladd | `swiglu-epochs12` | SwiGLU + --epochs 12 stack | Best val still at epoch 10/10 — more training is the clearest next win |
| alphonse | `swiglu-mlp-ratio-3` | mlp_ratio=3 with SwiGLU (inner_dim=320) | Student suggested this; gated FFNs benefit from wider inner dims; mlp_ratio=4 vanilla failed but gating changes the regime |
| fern | `swiglu-attn-dropout` | attn_dropout=0.1 on SwiGLU baseline | Frieren tests attn-dropout WITHOUT SwiGLU; this tests the combination on the new baseline |

## Dataset finding (from nezuko #3836 sanity check, 09:35 UTC)

Normalized DSDF (dims 4-11) across 100 train files / 108M values:
- min = −2.878, max = +0.913, mean ≈ 0.011, std ≈ 0.998
- **max abs = 2.88 — never exceeds 3σ**

| Threshold | Fraction of values above |
|---|---|
| >2.0σ | 1.37% |
| >2.5σ | 0.33% |
| >2.7σ | 0.046% |
| >3.0σ | 0.000% |

**Why:** raw DSDF is hardcoded to [0.0, 5.0] in dataset preprocessing. Clip=3σ is a no-op. Tighter clip values (2.5, 2.0) would touch the surface-side tail near sharp leading/trailing edges.

## Round-5 backlog (unassigned, ordered by priority — all vs new SwiGLU baseline 64.24)

1. **SwiGLU + n_hidden=176** — width scaling failed pre-SwiGLU (val=88.45 at 83.50 baseline); SwiGLU changed the regime; worth one test at the new baseline
2. **bf16 mixed-precision** — 2× throughput → ~20 epochs in 30 min. Previously tested as bf16+batch_size=8 (failed); SwiGLU may interact better with bf16 precision
3. **re-curriculum** — sample-weight schedule by log_Re (MED-HIGH risk; dataset-preprocessing complexity)
4. **camber symmetry augmentation** — (x→x, z→−z) along chord + flip AoA sign; doubles effective training data
5. **slice_num=32** — fewer, more aggregated slice tokens; never tested at SwiGLU baseline
6. **swiglu-mlp2-gate** — gate the output head `mlp2` with SwiGLU(hidden_dim, hidden_dim*mlp_ratio, out_dim); student suggestion #4 from #3814

## Potential next research directions (post-round-5)

### Tier change candidates (high upside, higher risk)
1. **SwiGLU + n_layers=6** — depth scaling failed pre-SwiGLU at epoch 8 (under-converged); with SwiGLU expressivity gain and epochs=12, might now be viable at this training budget
2. **Physics-informed loss** — divergence-free penalty (∇·u=0) on velocity field. High complexity but high upside.
3. **Multi-scale Fourier PE** — frequencies spanning multiple decades (e.g., 4 log-spaced from 1 to 32)
4. **Learnable slice positions** — make slice_num=64 positions trainable rather than fixed attention aggregation

### Confirmed exhausted (do not retry)
- n_hidden=176/192, n_layers=6 (pre-SwiGLU), slice_num=96/128, mlp_ratio=4 (vanilla), n_head=8
- lr=1e-3 + coord noise (3-seed fail), Huber loss, per-channel p-weighting, SGDR, SWA, EMA
- Feature noise aug, AoA jitter, coord_noise std sweep (std=0.03/0.005), eta_min sweep
- Per-domain output norm, per-channel output heads, learnable Fourier freqs
