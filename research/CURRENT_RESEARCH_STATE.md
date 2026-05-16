# SENPAI Research State

- **Last updated:** 2026-05-16 ~18:40 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline

**val_avg/mae_surf_p = 53.8221, test_avg/mae_surf_p = 47.2742** — from PR #3981 (thorfinn, **bf16 + epochs=18** cut at ep16/18), merged 2026-05-16 ~16:42 UTC. See `BASELINE.md` for full details.

Per-split test (b9h4bvnm): single_in_dist=54.72, geom_camber_rc=59.71, geom_camber_cruise=29.13, re_rand=45.53.

ALL four test splits improve vs #3969. Largest gains: geom_camber_cruise (32.02→29.13, −9.0%), re_rand (47.10→45.53, −3.3%). Val curve still descending at ep16 cut.

**Throughput finding:** bf16 autocast gives 1.47× per-epoch speedup vs fp32; 41.9 GB peak VRAM. This effectively converts ~50% more training epochs into the same wall clock budget.

New reproduce command: `cd target/ && SENPAI_TIMEOUT_MINUTES=35 python train.py --epochs 18 --use_bf16`

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
- **#3981: 53.82 (bf16 mixed-precision + epochs=18 cut at ep16; ALL 4 test splits improve; 1.47× speedup) ← CURRENT**

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
| **bf16 + epochs=18 (cut at ep16)** | **#3981** | **−4.64%** | bf16 autocast unlocks 1.47× speedup; same wall clock budget → 50% more epochs; ALL 4 test splits improve; CURRENT BEST |

**Total improvement from baseline:** 109.42 → 53.82 (−50.8%)

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
7. **--epochs 14** is the recommended training budget. Val curve still descending at ep14 — more epochs likely helps.
8. **Grad clip max_norm=1.0**, warmup 2 epochs, batch=4.
9. **SwiGLU key detail:** `mlp2` (output head) is left as standard MLP; only `self.mlp` in TransolverBlock is replaced. inner_dim=216 for mlp_ratio=2; 320 for mlp_ratio=3.
10. **mlp_ratio=2 beats mlp_ratio=3 at epochs=14** — narrower model wins; default is mlp_ratio=2.
11. **DSDF distribution finding (nezuko #3836):** normalized DSDF max abs=2.88 → clip=3.0 is a no-op. Clip=2.0 or 2.5 would actually touch 0.33-1.37% of values.

## Active in-flight PRs (status as of ~16:00 UTC)

| # | Student | Hypothesis | State | val_avg/mae_surf_p |
|---|---|---|---|---|
| **#3969** | askeladd | **SwiGLU + mlp_ratio=2 + epochs=14** | **MERGED ~15:50** → new baseline | 56.4402 |
| **#4002** | alphonse | SwiGLU + mlp_ratio=3 + epochs=14 | MERGED (superseded by #3969) | 57.3537 |
| **#3908** | alphonse | SwiGLU + mlp_ratio=3 | MERGED (prev baseline) | 59.0038 |
| **#3905** | askeladd | SwiGLU + epochs=12 | MERGED (prev baseline) | 60.7195 |
| **#3916** | tanjiro | SwiGLU output head gate (mlp2) | CLOSED — val=62.50 (+2.93% regress) | — |
| **#3912** | fern | SwiGLU + attn_dropout=0.1/0.2 | CLOSED — mixed signal | — |
| **#3951** | thorfinn | OneCycleLR + SwiGLU | CLOSED — val=61.38 regress | — |
| **#3972** | edward | SwiGLU+mlp_ratio=3 + asinh (scale=2.0/3.0) | CLOSED — negative result | — |
| **#3974** | nezuko | Curriculum learning | CLOSED — stale (baseline 2 gens old) | — |
| **#3979** | frieren | SwiGLU + n_hidden=176 (mlp_ratio=3 base) | CLOSED — stale (mlp_ratio=2 now baseline) | — |
| **#4000** | fern | attn_dropout=0.2 + epochs=14 | CLOSED ~16:00 — val=57.02 (+1.02% regress vs #3969); test=50.14 (+2.56% regress); attn_dropout exhausted on this stack | — |
| **#4001** | tanjiro | slice_num=32 on SwiGLU+mlp_ratio=3 | CLOSED ~16:30 — val=61.32 (+8.6% regress); all 4 splits regress; slice direction exhausted | — |
| **#3981** | thorfinn | bf16 mixed-precision (ep16 best) | **MERGED ~16:42** → new baseline | 53.8221 |
| **#4054** | thorfinn | **Round-7:** mlp_ratio=3 + bf16 + epochs=18 (let wider model converge) | WIP — assigned ~16:55 | awaiting |
| **#4047** | tanjiro | **Round-7:** Extended training probe (epochs=16/18 fp32) | WIP — assigned ~16:40 | awaiting |
| **#4034** | alphonse | **Round-7:** n_layers=6 depth scaling | WIP — assigned ~16:00 | awaiting |
| **#4036** | askeladd | **Round-7:** Camber flip augmentation (z-flip + AoA negate) | WIP — assigned ~16:00 | awaiting |
| **#4039** | edward | **Round-7:** Multi-scale Fourier PE (num_freq=8, freq-range sweep) | WIP — assigned ~16:00 | awaiting |
| **#4040** | fern | **Round-7:** DropPath stochastic depth (0.1, 0.15) | CLOSED ~18:35 — val=59.32/61.43 (+5.1%/+8.8% vs #3969); all 4 splits regress | — |
| **#4082** | fern | **Round-7 reassign:** n_hidden=176 + bf16 + epochs=18 (width retest on new stack) | WIP — assigned ~18:35 | awaiting |
| **#4042** | frieren | **Round-7:** Curvature-weighted surface loss (DSDF-norm proxy) | WIP — assigned ~16:00 | awaiting |
| **#4043** | nezuko | **Round-7:** AdamW weight_decay sweep (1e-3, 3e-4) + eta_min | WIP — assigned ~16:00 | awaiting |

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

## Round-7 assignments (assigned ~16:00–16:55 UTC, --epochs 14 + mlp_ratio=2 base unless noted)

| PR | Student | Hypothesis | Key CLI / change |
|---|---|---|---|
| #4034 | alphonse | n_layers=6 depth scaling | `--n_layers 6` (also arm B at epochs=12) |
| #4036 | askeladd | Camber flip augmentation (z-flip + AoA negate) | `--camber_flip_aug` |
| #4039 | edward | Multi-scale Fourier PE num_freq=8 + wider freq range | `--num_freq 8` (+ wider exp range arm B) |
| #4040 | fern | DropPath stochastic depth | CLOSED 18:35 — both arms regress >5% on every split; under-training, not overfit |
| #4082 | fern | Width retest with bf16 budget — n_hidden=176 + bf16 + ep18 | `--n_hidden 176 --use_bf16 --epochs 18` (ASSIGNED 18:35) |
| #4042 | frieren | Curvature-weighted surface loss (DSDF-norm proxy) | `--use_curvature_weight` |
| #4043 | nezuko | AdamW weight_decay sweep + eta_min floor | `--weight_decay 1e-3` (+ eta_min variants) |
| #4047 | tanjiro | Extended training probe (epochs=16/18, fp32 only) | `--epochs 18` |
| #4054 | thorfinn | mlp_ratio=3 + bf16 + epochs=18 (let wider model converge) | `--mlp_ratio 3 --use_bf16 --epochs 18` |

Deferred to round-8 (backlog):
- SAM optimizer (rho=0.05) — costly; screen at epochs=10 first
- DSDF-norm curvature **feature** (separate from #4042 which uses it for loss-weighting)
- AoA jitter augmentation (+/- 0.5-1.0 degrees)
- n_hidden=176 re-test on mlp_ratio=2 stack (was only tested on mlp_ratio=3)
- mlp_ratio=3 + epochs=16/18 (let wider model fully converge)
- Per-block / per-head learning rate ratios for SwiGLU FFN

## Potential next research directions (post-round-7)

### Tier change candidates (high upside, higher risk)
1. **Physics-informed loss** — divergence-free penalty (∇·u=0) on velocity field, or pressure-Bernoulli surface constraint. High complexity but high upside.
2. **Learnable slice positions** — make slice_num=64 positions trainable rather than fixed attention aggregation
3. **Surface-aware loss weighting** — weight per-point loss by local surface curvature (high-curvature regions = leading edges = where pressure peaks live)
4. **Equivariance via cross-attention** — replace coord features with positional encoding only, use SE(2)-equivariant cross-attention for pressure-field decoding
5. **mlp_ratio=3 + epochs=18** — the wider model may need more compute to converge; let it run longer

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
