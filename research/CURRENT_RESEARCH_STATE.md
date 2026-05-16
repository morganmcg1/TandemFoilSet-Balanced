# SENPAI Research State

- **Last updated:** 2026-05-16 ~14:30 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline

**val_avg/mae_surf_p = 59.0038, test_avg/mae_surf_p = 50.7368** — from PR #3908 (alphonse, SwiGLU + mlp_ratio=3), merged 2026-05-16 ~14:25 UTC. See `BASELINE.md` for full details.

Per-split test (4n7z1mwm): single_in_dist=57.19, geom_camber_rc=62.63, geom_camber_cruise=33.66, re_rand=49.46.

Baseline progression (val_avg/mae_surf_p):
- #3091: 109.42 (warmup + clip + lr=1e-3, MSE)
- #3089: 100.53 (L1 loss + scoring fix)
- #3507: 96.10 (n_hidden=160 width scaling)
- #3372: 88.24 (Fourier PE 4-freq, lr=1e-3)
- #3632: 83.50 (coord noise augmentation std=0.01, lr=5e-4)
- #3691: 82.50 (--epochs 12 longer training, 3-seed mean val=82.96)
- #3814: 64.24 (SwiGLU FFN in TransolverBlock, −22% vs prev best)
- #3905: 60.72 (SwiGLU + epochs=12, −5.5% vs #3814)
- **#3908: 59.00 (SwiGLU + mlp_ratio=3, −2.83% vs #3905; inner_dim 216→320) ← CURRENT**

## Winning stack (all additive, all merged)

| Component | PR | val gain | Notes |
|---|---|---|---|
| L1 loss + scoring NaN fix | #3089 | −8.1% | Canonical loss |
| n_hidden=160 | #3507 | −4.4% | Width sweet spot for 30min budget |
| Fourier PE num_freq=4 | #3372 | −8.2% | log-spaced sinusoidal on (x,z), lr=1e-3 |
| Coord noise std=0.01 | #3632 | −5.4% | Spatial augmentation during training, lr=5e-4 |
| --epochs 12 (longer training) | #3691 | −1.2% | Cosine T_max=12; best_epoch=11 in all 3 seeds |
| **SwiGLU FFN** | **#3814** | **−22.1%** | Gated FFN in TransolverBlock; inner_dim=216; mlp2 (output head) left as standard MLP |
| **SwiGLU + epochs=12** | **#3905** | **−5.5%** | Extended budget on still-descending curve; best_epoch=12/12; curve still descending |
| **SwiGLU + mlp_ratio=3** | **#3908** | **−2.83%** | Wider gated FFN inner_dim 216→320 (+0.25M params); best_epoch=12/12; still descending |

**Total improvement from baseline:** 109.42 → 59.00 (−46.1%)

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

## Active in-flight PRs (status as of 14:30 UTC)

| # | Student | Hypothesis | State | val_avg/mae_surf_p |
|---|---|---|---|---|
| **#3908** | alphonse | **SwiGLU + mlp_ratio=3** | **MERGED 14:25** → new baseline | 59.0038 🏆 |
| **#3905** | askeladd | SwiGLU + epochs=12 | MERGED 12:45 (prev baseline) | 60.7195 |
| **#3916** | tanjiro | SwiGLU output head gate (mlp2) | CLOSED 14:25 — val=62.50 (+2.93% regress); dead end | — |
| **#3912** | fern | SwiGLU + attn_dropout=0.1/0.2 | CLOSED 14:05 — val=60.33 small win but test+0.56% regress; mixed | — |
| **#3951** | thorfinn | OneCycleLR + SwiGLU | CLOSED 13:30 — val=61.38 (+1.08% regress vs #3905) | — |
| **#3857** | frieren | attn_dropout (pre-SwiGLU, stale) | CLOSED 13:25 — duplicate of #3912 | — |
| **#3835** | edward | asinh on MLP baseline | CLOSED 12:45 — re-testing on SwiGLU in #3972 | — |
| **#3836** | nezuko | DSDF clip | CLOSED 12:45 — hypothesis dead | — |
| **#3969** | askeladd | SwiGLU + epochs=14 (mlp_ratio=2) | WIP | awaiting |
| **#3972** | edward | SwiGLU+mlp_ratio=3 + asinh (scale=2.0/3.0) | WIP | awaiting |
| **#3974** | nezuko | Re-based curriculum learning | WIP | awaiting |
| **#3979** | frieren | SwiGLU + n_hidden=176 (mlp_ratio=3 base) | WIP | awaiting |
| **#3981** | thorfinn | bf16 mixed-precision + extended epochs | WIP | awaiting |
| **#4000** | fern | attn_dropout=0.2 + epochs=14 | WIP — assigned 14:03 | awaiting |
| **#4001** | tanjiro | slice_num=32 on SwiGLU+mlp_ratio=3 | WIP — assigned 14:26 | awaiting |
| **#4002** | alphonse | mlp_ratio=3 + epochs=14 | WIP — assigned 14:29 | awaiting |
| **#4000** | fern | attn_dropout=0.2 + epochs=14 (follow-up to #3912) | WIP — assigned 14:03 (extend training for regularizer benefit) | awaiting |

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

## Round-6 backlog (unassigned, ordered by priority — all vs new baseline val=59.00/test=50.74)

All 8 students currently have active WIP PRs. Unassigned ideas for next round:

1. **AoA jitter augmentation** — small perturbations to angle of attack (+/− 0.5°–1.0°) to expand training distribution. Low-risk, physics-grounded.
2. **camber symmetry augmentation** — (x→x, z→−z) along chord + flip AoA sign; doubles effective data. Medium risk: needs careful per-sample handling for cambered airfoils.
3. **SwiGLU + n_layers=6** — depth scaling re-test on SwiGLU+mlp_ratio=3 stack; previously failed pre-SwiGLU at epoch 8 (under-converged).
4. **Multi-scale Fourier PE** — frequencies spanning multiple decades (e.g., 4 log-spaced from 1 to 32) instead of current num_freq=4.
5. **mlp_ratio=3 + finer lr tuning** — alphonse suggested lr=3e-4 or 7e-4 for the wider model.

## Potential next research directions (post-round-6)

### Tier change candidates (high upside, higher risk)
1. **Physics-informed loss** — divergence-free penalty (∇·u=0) on velocity field, or pressure-Bernoulli surface constraint. High complexity but high upside.
2. **Learnable slice positions** — make slice_num=64 positions trainable rather than fixed attention aggregation
3. **Surface-aware loss weighting** — weight per-point loss by local surface curvature (high-curvature regions = leading edges = where pressure peaks live)
4. **Equivariance via cross-attention** — replace coord features with positional encoding only, use SE(2)-equivariant cross-attention for pressure-field decoding

### Confirmed exhausted (do not retry on this stack)
- n_hidden=176/192 (pre-SwiGLU only — retesting on SwiGLU+mlp_ratio=3 in #3979)
- slice_num=96/128, mlp_ratio=4 (vanilla pre-SwiGLU), n_head=8
- **SwiGLU output head gate (mlp2)** — closed #3916, consistent 2-3% regress on both arms/seeds
- **SwiGLU attn_dropout p=0.1/0.2 at epochs=12** — mixed signal (#3912); retest at epochs=14 in #4000
- lr=1e-3 + coord noise (3-seed fail), Huber loss, per-channel p-weighting, SGDR, SWA, EMA
- Feature noise aug, AoA jitter, coord_noise std sweep (std=0.03/0.005), eta_min sweep
- Per-domain output norm, per-channel output heads, learnable Fourier freqs
- DSDF clip ±2σ/±2.5σ (no-op confirmed via #3836)
- OneCycleLR + SwiGLU at epochs=12 (regress vs cosine+ep12; #3951 closed)
