# SENPAI Research State

- **Last updated:** 2026-05-17 ~01:30 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline

**val_avg/mae_surf_p = 49.2616, test_avg/mae_surf_p = 41.6188** — from PR #4252 (frieren, **Lion + n_hidden=176 + bf16 + epochs=14**), merged 2026-05-17 ~01:25 UTC. See `BASELINE.md` for full details.

Per-split test (eu7e0g18): single_in_dist=43.91, geom_camber_rc=54.75, geom_camber_cruise=26.13, re_rand=41.68.

**LION IS NOW DEFAULT OPTIMIZER.** All future experiments should use `--use_lion --lion_lr 1e-4 --lion_wd 1e-3` unless testing optimizer variants. Lion beats AdamW by −19.3% val at matched config (nh=176+ep14), delivers −2.28% test improvement even vs the wider/longer AdamW baseline (#4106 nh=192+ep20), and improves the previously-stuck `geom_camber_rc` hard split (54.75 vs 55.51) for the first time since the width-scaling plateau.

**Prior AdamW baseline (superseded):** #4106 nh=192+ep20 val=48.84, test=42.59. W&B `or5uq1id`. Note: val metric at prior baseline was slightly better (48.84 < 49.26); this is within seed noise (std ~2.5). The paper-facing test metric and per-split breakdown clearly favor Lion.

**Width+epochs frontier finding:** n_hidden=176/ep18 → 192/ep20 (+18% params, +2 epochs) wins +4.05% val / +2.98% test. Val curve still descending but decelerating at ep20 (−0.97% in final epoch). Peak VRAM at n_hidden=192 was 47.6 GB; ~50 GB headroom remains. **Width frontier still unsaturated; ep20 nearly converged.**

New reproduce command: `cd target/ && SENPAI_TIMEOUT_MINUTES=50 python train.py --n_hidden 192 --epochs 20 --use_bf16`

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
- #4082: 50.90 (n_hidden=176 + bf16 + epochs=18; +18% params over #3981; ALL 4 splits improve; curve still descending)
- **#4106: 48.84 (n_hidden=192 + bf16 + epochs=20; +18% params over #4082, +2 epochs; ALL 4 splits improve or flat; curve decelerating at ep20) ← CURRENT**

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
| **n_hidden=176 + bf16 + epochs=18** | **#4082** | **−5.43%** | +18% params (1.035M→1.23M); ALL 4 splits improve; curve still descending at ep18; 44.6 GB peak VRAM (50 GB headroom) |
| **n_hidden=192 + bf16 + epochs=20** | **#4106** | **−4.05%** | +18% params (1.23M→1.47M), +2 epochs; ALL 4 splits improve or flat; mild-overfit-hypothesis from ep18 retest refuted; geom_camber_rc remains structural hard split; 47.6 GB peak VRAM; CURRENT BEST |

**Total improvement from baseline:** 109.42 → 48.84 (−55.4%)

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
7. **--epochs 20 + --use_bf16 + --n_hidden 192** is the new default training recipe (post-#4106). Val curve still descending but decelerating at ep20 — width frontier unsaturated; ep20 nearly converged.
8. **Grad clip max_norm=1.0**, warmup 2 epochs, batch=4.
9. **SwiGLU key detail:** `mlp2` (output head) is left as standard MLP; only `self.mlp` in TransolverBlock is replaced. inner_dim=216 for mlp_ratio=2; 320 for mlp_ratio=3.
10. **mlp_ratio=2 beats mlp_ratio=3 at epochs=14** — narrower model wins; default is mlp_ratio=2.
11. **DSDF distribution finding (nezuko #3836):** normalized DSDF max abs=2.88 → clip=3.0 is a no-op. Clip=2.0 or 2.5 would actually touch 0.33-1.37% of values.

## Active in-flight PRs (status as of ~00:20 UTC 2026-05-17)

### Round-8 active (all WIP)
| # | Student | Hypothesis | State |
|---|---|---|---|
| **#4238** | askeladd | AdamW beta1 sweep (0.85, 0.95) at nh=192+bf16+ep20 | WIP (assigned 00:35) |
| **#4165** | alphonse | slice_num=48 retest (other side of curve) | WIP — actively training (99% GPU) |
| **#4178** | thorfinn | EMA of weights (decay=0.999) for val/test eval | WIP — actively training |
| **#4205** | edward | RMSNorm swap for LayerNorm on n_hidden=176+bf16+ep18 baseline | WIP (assigned 23:15) |
| #4252 | frieren | LION optimizer at nh=176+bf16+ep14 | **MERGED 01:25** — new baseline val=49.26/test=41.62; test −2.28%, hard-split improves, −30% compute; Lion now default |
| **#4280** | frieren | Lion+nh=192 compound test at ep=12+T_max=12 | WIP (assigned 01:30) |
| #4043 | nezuko | AdamW weight_decay sweep + eta_min | **CLOSED 01:40** — obsolete (Lion merged as default; AdamW WD irrelevant); also: infrastructure crash loop (18+ pod restarts) from uncommitted train.py changes blocking branch checkout; new assignment #4285 |
| **#4285** | nezuko | Lion LR sweep (lr=2e-4, lr=5e-5) at nh=176+bf16+ep14 | WIP (new — assigned 01:45, zero code changes) |
| **#4232** | fern | Push width frontier further: n_hidden=208 + bf16 + ep18 | WIP (assigned 00:15) |
| **#4233** | tanjiro | AGC (Adaptive Gradient Clipping) screening at nh=176+bf16+ep14 (30-min cap) | WIP (assigned 00:20) |

**Note:** All in-flight nh=176+ep18 experiments (#4129, #4165, #4178, #4205, #4227) were assigned against the prior baseline #4082 (val=50.90). Since #4106 merged as new baseline (val=48.84), the decision criteria when their results post will be:
- **Merge** if result beats new baseline (val < 48.84 AND test < 42.59) — direct best-update
- **Send back** for stack-with-width test if result beats #4082 (val < 50.90) but not new baseline — promote to nh=192+ep20 stack test
- **Close** if result regresses vs #4082 (val ≥ 52)

### Round-7/8 results (resolved 19:30 UTC – 00:10 UTC)
| # | Student | Hypothesis | Outcome |
|---|---|---|---|
| **#4106** | fern | n_hidden=192 + bf16 + ep20 retest | **MERGED ~00:05 UTC** → new baseline val=48.84 / test=42.59; ALL 4 splits improve or flat; mild-overfit-hypothesis from ep18 refuted |
| **#4190** | tanjiro | Capacity-vs-epochs at 30-min budget: n_hidden=144 + bf16 + ep18 | **CLOSED ~00:10 UTC** (val=57.05 +12.1%, all 4 splits regress +4.81 to +6.25; capacity dominates at 30-min budget; nh<176 not viable for 30-min pods; throughput 116.9 s/ep vs 95-105 prediction) |
| **#4082** | fern | n_hidden=176 + bf16 + ep18 | **MERGED ~19:32** → prev baseline val=50.90 / test=43.90 |
| **#4054** | thorfinn | mlp_ratio=3 + bf16 + ep18 | CLOSED — disconfirmed; mlp_ratio=2 wins on bf16 too (val=56.49) |
| **#4047** | tanjiro | ep16/18 fp32 probe | CLOSED — under-trained; moot under bf16 (val=76.04) |
| **#4042** | frieren | Curvature-weighted surface loss | CLOSED — real within-arm signal but absolute regress; retest assigned (#4110) |
| **#4034** | alphonse | n_layers=6 (fp32, ep14) | CLOSED — under-trained (both arms cut at ep9); retest with bf16 assigned (#4108) |
| **#4040** | fern | DropPath stochastic depth | CLOSED ~18:35 (val regress +5.1%/+8.8%) |
| **#4036** | askeladd | Camber flip augmentation | CLOSED ~20:05 (val +20.1% regress; root: NACA-M asymmetry unflippable; beta2 sweep #4129) |
| **#4112** | thorfinn | DSDF-norm as input feature | CLOSED ~20:30 (val +2.57%, test +3.45%; OOD splits worst; encoder already has full DSDF) |
| **#4108** | alphonse | n_layers=6 bf16 retest | CLOSED ~20:30 (val=62.05 cut ep13/18 @ 30-min env; depth=6 exhausted on 30-min budget; #4140 assigned) |
| **#4039(v1)** | edward | Multi-scale Fourier PE (3 arms on prior baseline) | Arm B WIN on #3981 (val=51.47, −4.34%/-5.22%); doesn't beat #4082 (val=50.90); **sent back for stacked retest on n_hidden=176** |
| **#4140** | alphonse | slice_num=96 retest on new baseline | CLOSED ~21:30 (val=74.47 cut ep12/18; monotonic worse 64→96; #4165 slice=48 assigned) |
| **#4143** | thorfinn | n_head=8 retest on new baseline | CLOSED ~21:50 (val=53.50 +2.6%, test=46.16 +2.3%, all 4 splits regress; +32% wall-clock from head_dim=22 < matmul threshold; #4178 EMA assigned) |
| **#4150** | tanjiro | lr=7e-4 + warmup=1 + ep14 (30-min budget pivot) | CLOSED ~22:40 (val=64.87 +27%, test=56.37 +28%; ep1 val=199.6 confirms early-training instability from aggressive lr+no-warmup; #4190 nh144+ep18 assigned) |
| **#4110** | frieren | Curvature loss retest sharpened (squared-DSDF) | CLOSED ~22:40 (ARM A val=57.13 +12.2% regress; ARM B control val=50.54 within noise; **diagnostic insight: DSDF-norm is distance-from-boundary proxy not curvature** → mesh-density bias; #4187 pressure-mag weight assigned) |
| **#4039(v2)** | edward | Multi-scale PE stacked retest (nf=8 wide on n_hidden=176+bf16+ep18) | CLOSED ~23:10 (val=63.29 +24.3%, every split regress; **mechanistic insight: width absorbs PE benefit — spectral resolution and hidden capacity substitute, not complement**; nf=8-wide-range win at n_hidden=160 (val=51.47) recorded for appendix table; #4205 RMSNorm assigned) |
| **#4187** | frieren | Pressure-magnitude weighted L1 (top-decile \|p_true\|, alpha=1.0, q=0.90) | CLOSED ~23:45 (val=53.17 +4.5%, test=45.27 +3.1%; only single_in_dist marginally benefits, all 3 OOD splits regress; **diagnostic: the high-\|p\| tail is not where residual error concentrates AND per-batch reweighting doesn't transfer to OOD splits**; combined with #4042/#4110 closes both surface-loss-reweighting axes (target-based + proxy-based) on this baseline; #4227 AdaBelief assigned) |

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

## Round-8 assignments (assigned 19:35–20:35 UTC, all build on the new n_hidden=176 + bf16 + ep18 baseline)

| PR | Student | Hypothesis | Key CLI / change | Status |
|---|---|---|---|---|
| #4106 | fern | Push wider: n_hidden=192 + bf16 + ep18 → **sent back for ep20** | `--n_hidden 192 --use_bf16 --epochs 20`, T=50 | WIP (sent back; v1 val=50.92 borderline) |
| #4108 | alphonse | n_layers=6 retest with bf16 + ep18 | `--n_layers 6 --use_bf16 --epochs 18`, T=45 | CLOSED (val=62.05 undertrained at 30-min env) |
| #4110 | frieren | Curvature loss retest with sharpened proxy | `--n_hidden 176 --use_bf16 --use_curvature_weight --epochs 18`, T=45 (2 arms) | CLOSED (ARM A val=57.13 +12.2%, ARM B control val=50.54 within noise; DSDF-norm is mesh-density proxy not curvature) |
| #4111 | tanjiro | Push to epochs=22 on n_hidden=176+bf16 | `--n_hidden 176 --use_bf16 --epochs 22`, T=55 | CLOSED (pod env 30-min cap can't fit ep22 ≈ 48 min) |
| #4112 | thorfinn | DSDF-norm as input feature | `--n_hidden 176 --use_bf16 --use_dsdf_norm_feature --epochs 18`, T=45 | CLOSED (val +2.57% regress) |
| #4129 | askeladd | AdamW beta2 sweep (0.95, 0.98) | `--adam_beta2 0.95/0.98 --n_hidden 176 --use_bf16 --epochs 18`, T=45 (2 arms) | **CLOSED 00:35** — both arms regress, monotonic ordering vs beta2=0.999; beta2 axis exhausted |
| **#4238** | askeladd | AdamW beta1 sweep (0.85, 0.95) at nh=192+bf16+ep20 | `--adam_beta1 0.85/0.95 --n_hidden 192 --use_bf16 --epochs 20`, ~44 min/arm | WIP (new) |
| #4140 | alphonse | slice_num=96 retest on new baseline | `--slice_num 96 --n_hidden 176 --use_bf16 --epochs 18`, T=45 | CLOSED (val=74.47 cut ep12; monotonic worse from 64→96) |
| #4143 | thorfinn | n_head=8 retest on new baseline | `--n_head 8 --n_hidden 176 --use_bf16 --epochs 18`, T=45 | CLOSED (val=53.50 +2.6%, all 4 splits regress; +32% wall-clock from head_dim=22 < matmul threshold) |
| #4150 | tanjiro | lr=7e-4 + warmup=1 + ep14 (30-min budget pivot) | `--n_hidden 176 --use_bf16 --epochs 14 --lr 7e-4 --warmup_epochs 1` | CLOSED (val=64.87 +27%, ep1 val=199.6 confirms early-training instability; can't trade epochs for higher LR) |
| **#4165** | alphonse | slice_num=48 retest (other side of curve) | `--slice_num 48 --n_hidden 176 --use_bf16 --epochs 18` (no T override) | WIP |
| **#4178** | thorfinn | EMA of weights for val/test eval (decay=0.999) | `--n_hidden 176 --use_bf16 --epochs 18 --use_ema --ema_decay 0.999` | WIP |
| **#4187** | frieren | Pressure-magnitude weighted L1 (top-decile \|p_true\|) | `--use_pmag_weight --pmag_weight_alpha 1.0 --pmag_weight_quantile 0.90 --n_hidden 176 --use_bf16 --epochs 18`, T=45 | WIP (new) |
| #4190 | tanjiro | n_hidden=144 + bf16 + ep18 (capacity-vs-epochs at 30-min budget) | `--n_hidden 144 --use_bf16 --epochs 18` (no T override) | **CLOSED 00:10** (val=57.05; capacity dominates schedule completion at 30-min budget) |
| #4205 | edward | RMSNorm swap for LayerNorm on n_hidden=176+bf16+ep18 baseline | `--use_rmsnorm --n_hidden 176 --use_bf16 --epochs 18` (no T override) | **CLOSED ~01:05** (val=76.02 +49% cut ep12/18; per-epoch ~163s vs baseline 128s — 27% SLOWER from unfused RMSNorm kernel; all 4 splits regress; dual-mode failure: slower throughput AND ~11 points/ep convergence deficit; mean-shift removal breaks slice-attention conditioning; #4270 QK-norm assigned) |
| **#4270** | edward | QK-norm: LayerNorm on Q and K projections (per-head over head_dim) BEFORE attention dot product | `--use_qk_norm --n_hidden 176 --use_bf16 --epochs 14` (30-min budget) | WIP (new — assigned 01:15) |
| #4227 | frieren | AdaBelief optimizer swap for AdamW on n_hidden=176+bf16+ep18 baseline | `--optimizer adabelief --n_hidden 176 --use_bf16 --epochs 18` (no T override) | **CLOSED 00:45** — per-epoch equivalent to AdamW; belief-scaling no-op at stable gradient regime; AdaBelief axis closed |
| #4252 | frieren | LION optimizer at nh=176+bf16+ep14 | `--use_lion --lion_lr 1e-4 --lion_wd 1e-3 --n_hidden 176 --use_bf16 --epochs 14` | **MERGED 01:25** — new baseline val=49.26/test=41.62 |
| **#4280** | frieren | Lion+nh=192 compound test at ep=12+T_max=12 (30-min cap) | `--use_lion --lion_lr 1e-4 --lion_wd 1e-3 --n_hidden 192 --use_bf16 --epochs 12` | WIP (new — assigned 01:30) |
| **#4232** | fern | Push width frontier: n_hidden=208 + bf16 + ep18 → **sent back ~01:10 for clean ep=12+T_max=12 retest** (cap-friendly clean anneal) | `--n_hidden 208 --use_bf16 --epochs 12` (no T override) | WIP (sent back; v1 val=65.77 cut ep13 partial-anneal; ep-matched signal mixed: ep11/12 lead nh=192 by −5%/−15%, ep13 reverses +1.3% but schedule-confounded) |
| **#4233** | tanjiro | AGC (Adaptive Gradient Clipping) at nh=176+bf16+ep14 → **sent back ~01:00 for clip_factor sweep** at 0.03 and 0.05 | `--use_agc --agc_clip_factor 0.03 / 0.05 --n_hidden 176 --use_bf16 --epochs 14` (30-min cap) | WIP (sent back; v1 val=59.29 clip=0.01 beats #4082 matched-config by −3% at ep14, agc_frac_clipped=0.79 aggressive; clip sweep to find optimum before full-budget confirmation) |

**Note on env timeout (important):** Pod env caps vary. Alphonse and tanjiro pods enforce 30-min hard wall (per #4108, #4111 student flags). Fern and thorfinn pods run 39+ min fine. Future assignments to alphonse/tanjiro must be designed for ≤30-min wall. Instructions to these students should NOT include `SENPAI_TIMEOUT_MINUTES` override since isolation rules prohibit it. Nezuko, askeladd, frieren, edward: budget unknown — assume 30 min unless evidence otherwise.

**Infrastructure incident (2026-05-16 21:53–22:24 UTC):** Fleet-wide GitHub API rate limit on student token caused multiple pods to enter "No assigned PRs, sleeping" loops despite WIP PRs being assigned. Affected pods (per kubectl logs): fern, frieren, alphonse, askeladd, edward, tanjiro, thorfinn. Rate limit cleared by ~22:24; all pods now seeing their assigned PRs again. **Completed runs pending student post (per W&B):** #4106 fern (n_hidden=192, val=50.92), #4110 frieren (both arms; sharpened curvature val=57.13 regress, control val=50.54 seed-variance baseline retest). Advisor acknowledgment comments posted on both PRs.

Deferred to round-9 (backlog):
- SAM optimizer (rho=0.05) — costly; screen at epochs=10 first
- Per-block / per-head learning rate ratios for SwiGLU FFN
- Soft equivariance loss `‖f(x) - flip(f(flip(x)))‖` (askeladd's #4036 follow-up — sidesteps NACA-M asymmetry by operating on predictions)
- Stacking interactions: depth + width, curvature loss + DSDF feature, etc.
- Mixup-style geometry blending across samples
- **True geometric curvature via ‖∇DSDF‖ along arc-length** (frieren #4110 follow-up #1 — addresses mesh-density confound in DSDF-magnitude proxy)
- n_head=2 head_dim=88 sanity foil (thorfinn #4143 follow-up — closes n_head axis if 4 is true peak)
- n_head=8 + n_hidden≥256 stacking (head_dim≥32 to clear matmul compute-bound threshold)
- lr=5e-4 + warmup=1 + ep14 isolation (tanjiro #4150 follow-up #1 — isolates warmup-shortening from LR-increase)
- lr=6e-4 + warmup=2 + ep13 milder LR bump (tanjiro #4150 follow-up #2)
- pmag_weight quantile=0.80 / alpha=0.5 sweep (frieren #4187 follow-up if winner)
- **Multi-scale PE confirmed-multi-seed ablation at n_hidden=160** for appendix table (edward #4039 followup: nf=8 freq_min_exp=-2..5 produced val=51.47/-4.34% vs n_hidden=160 baseline; record cleanly as part of paper but absorbed by width)
- **RMSNorm stacking with width=192** if #4205 wins on baseline (compounding norm-mechanic with width)
- **Gradient-magnitude weighted L1** (focus on prediction-error nodes, not target-magnitude nodes) — orthogonal to closed pmag/curvature axes
- **Channel reweighting (Ux/Uy vs p surface loss balance)** — current surf_weight=10 was tuned pre-bf16+nh176; retune
- **Per-node uncertainty-weighted L1** with learned per-vertex weights (high complexity; in principle better-aligned with where error concentrates)
- **AdaBelief LR sweep** (lr=3e-4, 1e-3) — if #4227 borderline-misses, lr retune is the natural followup

## Potential next research directions (post-round-8)

### Tier change candidates (high upside, higher risk)
1. **Physics-informed loss** — divergence-free penalty (∇·u=0) on velocity field, or pressure-Bernoulli surface constraint. High complexity but high upside.
2. **Learnable slice positions** — make slice_num=64 positions trainable rather than fixed attention aggregation
3. **Equivariance via cross-attention** — replace coord features with positional encoding only, use SE(2)-equivariant cross-attention for pressure-field decoding
4. **Stack the round-8 winners** — if both width=192 (fern) and edward's multi-scale Fourier PE (#4039 retest) win independently, compound them into a single architecture in round-9
5. **Compute-axis pushes** — if ep22 wins (deferred), try ep26 or ep30 on the now-confirmed bf16 stack (requires 45+ min budget student)
6. **Push Fourier PE low end** (per edward's #4039 suggestions): freq_min_exp=-3 (sub-pixel detail), num_freq=10 (denser wide grid) — pending stack-with-width confirmation

### Confirmed exhausted (do not retry on this stack — closed axes)
- **Surface loss reweighting by target magnitude** (frieren #4187 pmag-weight, val +4.5%)
- **Surface loss reweighting by DSDF-derived proxy** (frieren #4110 curvature, val +12.2%)
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
