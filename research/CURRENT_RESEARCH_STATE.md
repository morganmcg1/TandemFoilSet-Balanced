# SENPAI Research State

- **Last updated:** 2026-05-16 ~20:10 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline

**val_avg/mae_surf_p = 50.9008, test_avg/mae_surf_p = 43.8989** — from PR #4082 (fern, **n_hidden=176 + bf16 + epochs=18**), merged 2026-05-16 ~19:32 UTC. See `BASELINE.md` for full details.

Per-split test (mgu3m5v2): single_in_dist=48.97, geom_camber_rc=55.45, geom_camber_cruise=28.27, re_rand=42.91.

ALL four test splits improve vs #3981. Largest gain: single_in_dist (54.72→48.97, −10.5%). Val curve still descending at ep18 cut (ep17=52.28, ep18=50.90, Δ=−1.38).

**Width frontier finding:** n_hidden=160 → 176 (+18% params) wins +5.4% val / +7.1% test on the bf16+ep18 stack. The earlier n_hidden=176 regress (on mlp_ratio=3+fp32+ep12) was a joint budget+capacity artifact — bf16 unlocked it. Peak VRAM at n_hidden=176 was 44.6 GB; ~50 GB headroom remains.

New reproduce command: `cd target/ && SENPAI_TIMEOUT_MINUTES=45 python train.py --n_hidden 176 --epochs 18 --use_bf16`

Baseline progression (val_avg/mae_surf_p):
- #3091: 109.42 (warmup + clip + lr=1e-3, MSE)
- #3089: 100.53 (L1 loss + scoring fix)
- #3507: 96.10 (n_hidden=160 width scaling)
- #3372: 88.24 (Fourier PE 4-freq, lr=1e-3)
- #3632: 83.50 (coord noise augmentation std=0.01, lr=5e-4)
- #3691: 82.50 (--epochs 12 longer training, 3-seed mean val=82.96)
- #3814: 64.24 (SwiGLU FFN in TransolverBlock, −22% vs prev best)
- #3905: 60.72 (SwiGLU + epochs=12, −5.5% vs #3814)
- #3908: 59.00 (SwiGLU + mlp_ratio=3, −2.83% vs #3905; inner_dim 216→320)
- #4002: 57.35 (SwiGLU + mlp_ratio=3 + epochs=14, −2.80% vs #3908)
- #3969: 56.44 (SwiGLU + mlp_ratio=2 + epochs=14, −1.60% vs #4002; inner_dim=216)
- #3981: 53.82 (bf16 mixed-precision + epochs=18 cut at ep16; ALL 4 test splits improve; 1.47× speedup)
- **#4082: 50.90 (n_hidden=176 + bf16 + epochs=18; +18% params over #3981; ALL 4 splits improve; curve still descending) ← CURRENT**

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
| **SwiGLU + mlp_ratio=3 + epochs=14** | **#4002** | **−2.80%** | Even longer training on wider model; best_epoch=14/14; still descending |
| **SwiGLU + mlp_ratio=2 + epochs=14** | **#3969** | **−1.60%** | Narrower model (inner_dim=216) with longer training beats wider (inner_dim=320) |
| **bf16 + epochs=18 (cut at ep16)** | **#3981** | **−4.64%** | bf16 autocast unlocks 1.47× speedup; same wall clock budget → 50% more epochs; ALL 4 test splits improve |
| **n_hidden=176 + bf16 + epochs=18** | **#4082** | **−5.43%** | +18% params (1.035M→1.23M); ALL 4 splits improve; curve still descending at ep18; 44.6 GB peak VRAM (50 GB headroom); CURRENT BEST |

**Total improvement from baseline:** 109.42 → 50.90 (−53.5%)

## Critical mlp_ratio finding

**mlp_ratio=2 + epochs=14 (val=56.44) beats mlp_ratio=3 + epochs=14 (val=57.35) by 0.91 val units.**

This is a reversal from the epoch=12 result where mlp_ratio=3 was better. Interpretation: the narrower model (mlp_ratio=2, inner_dim=216, ~1.035M params) has better inductive bias or optimization landscape for this budget. The wider model may need more epochs to converge.

**Default base config for round-7 and beyond: `--mlp_ratio 2 --epochs 14`** (mlp_ratio=2 is already the default in code; no flag needed).

## Most recent research direction from human researcher team

No GitHub Issues open for this track as of last check. Proceeding from the program contract only.

## Cross-cutting findings (apply to ALL in-flight PRs)

1. **SwiGLU FFN is now the default** (merged in #3814). All new experiments build on this.
2. **L1 loss is the default** (Config.loss_type = "l1").
3. **n_hidden=160 is the width sweet spot** — n_hidden=176 and n_hidden=192 both regress (pre-SwiGLU only; retesting deferred).
4. **Fourier PE num_freq=4 is the default** — confirmed sweet spot.
5. **coord_noise_std=0.01 is the default** (merged in #3632).
6. **lr=5e-4 is the default** — lr=1e-3 + coord noise regressed in 3-seed test (#3690).
7. **--epochs 18 + --use_bf16 + --n_hidden 176** is the new default training recipe (post-#4082). Val curve still descending at ep18 — more epochs likely helps.
8. **Grad clip max_norm=1.0**, warmup 2 epochs, batch=4.
9. **SwiGLU key detail:** `mlp2` (output head) is left as standard MLP; only `self.mlp` in TransolverBlock is replaced. inner_dim=216 for mlp_ratio=2; 320 for mlp_ratio=3.
10. **mlp_ratio=2 beats mlp_ratio=3 at epochs=14** — narrower model wins; default is mlp_ratio=2.
11. **DSDF distribution finding (nezuko #3836):** normalized DSDF max abs=2.88 → clip=3.0 is a no-op. Clip=2.0 or 2.5 would actually touch 0.33-1.37% of values.

## Active in-flight PRs (status as of ~19:45 UTC)

### Round-8 active (assigned 19:35–20:10 UTC, all on bf16 stack)
| # | Student | Hypothesis | State |
|---|---|---|---|
| **#4106** | fern | Push wider: n_hidden=192 + bf16 + ep18 | WIP |
| **#4108** | alphonse | n_layers=6 retest with bf16 + ep18 (full cosine schedule) | WIP |
| **#4110** | frieren | Curvature loss retest with sharpened proxy on new baseline | WIP |
| **#4111** | tanjiro | Push to epochs=22 on n_hidden=176+bf16 (curve still descending at ep18) | WIP |
| **#4112** | thorfinn | DSDF-norm as input feature (orthogonal to loss-weighting) | WIP |
| **#4129** | askeladd | AdamW beta2 sweep (0.95, 0.98) on n_hidden=176+bf16+ep18 | WIP (assigned 20:10) |

### Round-7 still in-flight (assigned earlier, results pending)
| # | Student | Hypothesis | State |
|---|---|---|---|
| **#4039** | edward | Multi-scale Fourier PE num_freq=8 | WIP |
| **#4043** | nezuko | AdamW weight_decay sweep + eta_min | WIP (redirected 20:08 to new baseline stack) |

### Round-7 results (resolved 19:30–20:10 UTC)
| # | Student | Hypothesis | Outcome |
|---|---|---|---|
| **#4082** | fern | n_hidden=176 + bf16 + ep18 | **MERGED ~19:32** → new baseline val=50.90 / test=43.90 |
| **#4054** | thorfinn | mlp_ratio=3 + bf16 + ep18 | CLOSED — disconfirmed; mlp_ratio=2 wins on bf16 too (val=56.49) |
| **#4047** | tanjiro | ep16/18 fp32 probe | CLOSED — under-trained; moot under bf16 (val=76.04) |
| **#4042** | frieren | Curvature-weighted surface loss | CLOSED — real within-arm signal but absolute regress; retest assigned (#4110) |
| **#4034** | alphonse | n_layers=6 (fp32, ep14) | CLOSED — under-trained (both arms cut at ep9); retest with bf16 assigned (#4108) |
| **#4040** | fern | DropPath stochastic depth | CLOSED ~18:35 (val regress +5.1%/+8.8%) |
| **#4036** | askeladd | Camber flip augmentation | CLOSED ~20:05 (val +20.1% / test +19.6% regress; root cause: NACA-M asymmetry can't be flipped; retest assigned as beta2 sweep #4129) |

### Prior baseline progression
| # | Student | Hypothesis | Outcome |
|---|---|---|---|
| **#3981** | thorfinn | bf16 + epochs=18 (cut at ep16) | MERGED ~16:42 (prev baseline 53.82) |
| **#3969** | askeladd | SwiGLU + mlp_ratio=2 + epochs=14 | MERGED ~15:50 (prev baseline 56.44) |
| **#4002** | alphonse | SwiGLU + mlp_ratio=3 + epochs=14 | MERGED (superseded) |
| **#3908** | alphonse | SwiGLU + mlp_ratio=3 | MERGED |
| **#3905** | askeladd | SwiGLU + epochs=12 | MERGED |

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

## Round-8 assignments (assigned 19:35–20:10 UTC, all build on the new n_hidden=176 + bf16 + ep18 baseline)

| PR | Student | Hypothesis | Key CLI / change |
|---|---|---|---|
| #4106 | fern | Push wider: n_hidden=192 + bf16 + ep18 | `--n_hidden 192 --use_bf16 --epochs 18`, T=50 |
| #4108 | alphonse | n_layers=6 retest with bf16 + ep18 (depth-isolated, n_hidden=160) | `--n_layers 6 --use_bf16 --epochs 18`, T=45 |
| #4110 | frieren | Curvature loss retest with sharpened proxy (squared DSDF-norm) on new baseline | `--n_hidden 176 --use_bf16 --use_curvature_weight --epochs 18`, T=45 (2 arms) |
| #4111 | tanjiro | Push to epochs=22 on n_hidden=176+bf16 (curve still descending) | `--n_hidden 176 --use_bf16 --epochs 22`, T=55 |
| #4112 | thorfinn | DSDF-norm as input feature (orthogonal to frieren's loss-weighting) | `--n_hidden 176 --use_bf16 --use_dsdf_norm_feature --epochs 18`, T=45 |
| #4129 | askeladd | AdamW beta2 sweep (0.95, 0.98) on n_hidden=176+bf16+ep18 | `--adam_beta2 0.95 / 0.98 --n_hidden 176 --use_bf16 --epochs 18`, T=45 (2 arms) |

Deferred to round-9 (backlog):
- SAM optimizer (rho=0.05) — costly; screen at epochs=10 first
- Per-block / per-head learning rate ratios for SwiGLU FFN
- Soft equivariance loss `‖f(x) - flip(f(flip(x)))‖` (askeladd's #4036 follow-up — sidesteps NACA-M asymmetry by operating on predictions)
- Stacking interactions: depth + width, curvature loss + DSDF feature, etc.
- Mixup-style geometry blending across samples
- Stronger curvature proxy variants (gradient of DSDF, learned curvature head)
- EMA of weights, slice_num=96/128 (architecture)

## Potential next research directions (post-round-8)

### Tier change candidates (high upside, higher risk)
1. **Physics-informed loss** — divergence-free penalty (∇·u=0) on velocity field, or pressure-Bernoulli surface constraint. High complexity but high upside.
2. **Learnable slice positions** — make slice_num=64 positions trainable rather than fixed attention aggregation
3. **Equivariance via cross-attention** — replace coord features with positional encoding only, use SE(2)-equivariant cross-attention for pressure-field decoding
4. **Stack the round-8 winners** — if both width=192 (fern) and curvature (frieren) and DSDF feature (thorfinn) win independently, compound them into a single architecture in round-9
5. **Compute-axis pushes** — if ep22 wins (tanjiro), try ep26 or ep30 on the now-confirmed bf16 stack

### Confirmed exhausted (do not retry on this stack)
- n_hidden=176/192 (pre-SwiGLU only — deferred for mlp_ratio=2 stack)
- slice_num=96/128, mlp_ratio=4 (vanilla pre-SwiGLU), n_head=8
- **SwiGLU output head gate (mlp2)** — closed #3916, consistent 2-3% regress
- **SwiGLU attn_dropout p=0.1/0.2 at epochs=12** — mixed signal (#3912); retest at epochs=14 in #4000
- lr=1e-3 + coord noise (3-seed fail), Huber loss, per-channel p-weighting, SGDR, SWA, EMA
- Feature noise aug, AoA jitter, coord_noise std sweep (std=0.03/0.005), eta_min sweep
- Per-domain output norm, per-channel output heads, learnable Fourier freqs
- DSDF clip ±2σ/±2.5σ (no-op confirmed via #3836)
- OneCycleLR + SwiGLU at epochs=12 (regress vs cosine+ep12; #3951 closed)
- **SwiGLU + asinh input transform (scale=2.0/3.0)** — closed #3972, negative result
- **Curriculum learning** — closed #3974, stale (baseline two generations old)
