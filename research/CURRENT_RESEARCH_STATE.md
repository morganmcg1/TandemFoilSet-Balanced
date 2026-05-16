# SENPAI Research State — TandemFoilSet (willow-pai2i-24h-r4)

- **As of:** 2026-05-16 04:42 UTC
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r4`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced`
- **W&B:** `wandb-applied-ai-team/senpai-v1`
- **Most recent human researcher direction:** None recorded. Launch isolation rules are in force.
- **Ops issue (ongoing):** GitHub API rate limit recurring across student pods (5000/hr cap, ~3300 remaining at 04:25). Each pod is hitting 403s within ~6 calls per polling iteration; combined with 8 pods polling every 5 min plus advisor activity, the budget gets exhausted in waves. As of 04:23 all pods have 0% GPU utilization — students are alternating between rate-limit blackouts (~10 min each) and short Claude planning sessions (2-4 min) without successfully launching persistent training runs. This is an entrypoint-level issue (out of advisor scope). Alphonse #3565 has had no training in ~4h since assignment.

## Research programme summary

Predict (Ux, Uy, p) at every node of unstructured 2D CFD meshes (74K–242K nodes) of tandem airfoils. The primary ranking metric is `test_avg/mae_surf_p` — equal-weighted surface pressure MAE across four test splits. Per-run budget is 30 min wall clock × 50 epochs hard cap.

## Current best (BASELINE.md)

**PR #3262 (edward, merged 2026-05-16 01:27 UTC) — RFF σ=1.0, n_freqs=16 on (x,z) coords**
- `val_avg/mae_surf_p = 79.2847` (W&B `tnna02ob`)
- `test_avg/mae_surf_p = **69.27**` (4-split finite)
- Per-split test: single=78.69, rc=79.59, cruise=49.16, re_rand=69.65

Cumulative path: vanilla 106.23 → #3257 surf-MAE 94.35 → #3263 FiLM 90.06 → #3358 cosine 80.08 → #3262 RFF **69.27** (**−34.8% from vanilla in 4 PRs**).

All remaining PRs must beat **test_avg/mae_surf_p < 69.27**.

## Active R2/R3 portfolio (all 8 students WIP)

| # | Student | PR | Hypothesis | Status |
|---|---------|----|-----------|--------|
| 1 | nezuko   | #3618 | **Surface-only decoder head (parallel zero-init residual on surface nodes)** | WIP (assigned 02:00; pod picked up 03:20) |
| 2 | frieren  | #3504 | Richer FiLM conditioning (cond_dim 1→11) — SENT BACK for full-stack rebase | WIP (pod picked up 03:21) |
| 3 | thorfinn | #3468 | Per-block FiLM heads — v2; CONFLICTING needs rebase resolution | WIP (pod still rate-limited at 03:19; next iter ~03:24) |
| 4 | tanjiro  | #3658 | **Transolver depth test: n_layers 5 → 6 with matched cosine_tmax** (first architecture experiment in this track) | WIP (assigned 03:35) |
| 5 | alphonse | #3693 | **Peak LR sweep {1e-3, 2.5e-4} on RFF base** (lr=5e-4 never swept in this track) | WIP (assigned 04:41; pod will pick up on next poll) |
| 6 | askeladd | #3351 | EMA β=0.99 (shorter horizon) — CONFLICTING needs rebase resolution | WIP (pod picked up 03:21) |
| 7 | edward   | #3599 | RFF σ sweep {0.5,1.0,2.0} × n_freqs {16,32} — R3 refinement on own win | WIP (pod picked up 03:20) |
| 8 | fern     | #3258 | Grad-clip 1.0 + 5-epoch warmup — MERGEABLE | WIP (pod picked up 03:21) |

**All 8 students active.** First training runs starting now; ETAs vary by complexity (single arm ~30 min; alphonse 3-arm sweep ~90 min if sequential).

## R1/R2 closed/merged history

| # | Student | Hypothesis | PR | Status |
|---|---------|-----------|-----|--------|
| 1 | frieren  | Surface MAE + p-weight 3× + NaN guard | #3257 | **MERGED (R1 winner #1)** — test=94.35 |
| 2 | thorfinn | FiLM log(Re) conditioning | #3263 | **MERGED (R1 winner #2)** — val=100.24, test=90.06 |
| 3 | alphonse | Cosine LR T_max=14 | #3358 | **MERGED (R2 winner #1)** — val=90.44, test=80.08 |
| 4 | edward   | RFF σ=1.0, n_freqs=16 on (x,z) coords | #3262 | **MERGED (R2 winner #2)** — val=79.28, test=69.27 |
| ✗ | frieren  | Re-stratified loss (1/per_sample_y_std) | #3386 | **CLOSED** — failed (+1.7% test regression) |
| ✗ | nezuko   | Multi-scale slice tokens | #3429 | **CLOSED** — equal-epoch tie with control |
| ✗ | nezuko   | Surface-biased slice routing | #3260 | **CLOSED** (paired −0.05%) |
| ✗ | nezuko   | Volume MAE reformulation (L1 on both) | #3550 | **CLOSED** — failed (+4.7% test regression on old base, +21% above 69.27) |
| ✗ | tanjiro  | surf_weight sweep re-run (sw5) on FiLM+RFF base | #3406 | **CLOSED** — failed (+4.17% test on new stack, mechanism absorbed by FiLM+RFF) |
| ✗ | tanjiro  | Huber loss delta=0.5 | #3256 | **CLOSED** (redundant with #3257) |
| ✗ | alphonse | Wider-shallower 256d | #3261 | **CLOSED** (+24% worse) |
| ✗ | askeladd | Dropout p=0.1 | #3264 | **CLOSED** (+6% worse) |

## Standings — test_avg/mae_surf_p (lower is better)

| Rank | PR | Hypothesis | test_avg (4-split) | vs baseline | Status |
|------|----|------------|-------------------:|-------------|--------|
| **1** | **#3262 (edward)** | **RFF σ=1.0, n_freqs=16** | **69.27** | **NEW BASELINE** | **MERGED** |
| 2 | #3358 (alphonse) | cosine T_max=14 | 80.08 | +15.6% above new | MERGED |
| 3 | #3504 (frieren, old-base) | richer FiLM mid128, ref vs #3263 | 81.07 | +17% above new | rebase in flight |
| 4 | #3468 (thorfinn, old-base) | per-block FiLM v1 (post-block) | 84.00 | +21.3% above new | rebase+rerun in flight |
| 5 | #3406 (tanjiro, old-base) | sw=5, paired baseline 94.35 | 88.80 | +28.2% above new | rebase+rerun in flight |
| 6 | #3263 (thorfinn) | FiLM(log_Re) | 90.06 | +30.1% above new | MERGED |
| 7 | #3257 (frieren) | Surf-MAE+p-weight 3×+NaN guard | 94.35 | +36.3% above new | MERGED |

## Key R2/R3 predictions (on new #3262 baseline, target test < 69.27)

- **thorfinn per-block FiLM rebased (#3468):** v1 on old base gave test=84.00 (−6.73% vs #3263). On new RFF base, per-block FiLM is orthogonal (multiple architectural gates vs input encoding). Predicted hopeful test ~64–66, conservative ~66–68, pessimistic ~68–72. New target hard but plausible.
- **tanjiro Transolver depth test (#3658):** n_layers 5 → 6 on full FiLM+RFF stack. First architecture-side experiment in this track (4 prior wins all loss/input/schedule). +114K params (+17%), +17% compute → student matches cosine_tmax to actual achievable epochs (~11-12). Predicted hopeful test ~65–66 (5% gain), conservative ~67–68 (2-3%).
- **frieren richer FiLM rebased (#3504):** cond_dim=11, film_mid=64 first. Old gain −7.16% (film_mid=64 on #3263 ref). On new RFF+cosine+FiLM base, predicted hopeful test ~64.4, conservative ~65.8–67.2. Use film_mid=64 due to VRAM limits (mid128 was 94.0 GiB, RFF adds overhead).
- **nezuko surface-only decoder head (#3618):** Parallel zero-init 128→128→3 head on `h = ln_3(fx)` after block 5, gated by `is_surface`. Orthogonal to all 4 prior wins (output-head specialization vs loss/input/schedule/encoding). +16,899 params (+2.5%). Predicted conservative ~2–4% gain (test ~66–68), hopeful 5–7% (test ~64–66), pessimistic wash.
- **alphonse LR sweep (#3693):** Peak LR {1e-3, 2.5e-4} bracketing untested default 5e-4. No code changes (CLI flag). Arm A lr=1e-3 is primary (canonical value for ~1M-param transformers at batch=4). Predicted Arm A: test ~66–68 (2–4% gain); Arm B lr=2.5e-4: expected slight regression. #3565 AdamW sweep closed — ran on pre-RFF base (invalid comparison), also confirmed high WD hurts capacity-limited models.
- **askeladd EMA β=0.99 (#3351):** β=0.99 (100-step horizon) on new stacked base. Cosine T_max=14 anneals LR→0, which may reduce oscillation benefit. Conservative expectation: small gain (~1–2%) or wash. Rebasing now (branch CONFLICTING).
- **fern grad-clip+warmup (#3258):** clip=1.0, warmup=5 on new stacked base. Grad-norm median was 60, peak 1004. Cosine LR may have already tamed late-epoch norms. Predicted conservative: 3–5% gain, pessimistic: wash. Rebasing now.
- **edward RFF σ/n_freqs sweep (#3599):** σ {0.5, 2.0} × n_freqs {16, 32}, brackets the winning σ=1.0 / n_freqs=16. CLI-only, no code changes. Conservative: σ=1.0 was already optimal → wash. Hopeful: n_freqs=32 doubles spatial resolution → modest 1–2% gain.

## Open issues / live diagnostics

- **Canonical NaN guard: RESOLVED (in main).** All new branches inherit automatically.
- **Cosine T_max mismatch: RESOLVED (#3358 merged).**
- **RFF integration: RESOLVED (#3262 merged).** RFF is now in the base; all rebase-needed PRs will inherit it.
- **Huge gradient norms (median 56, peak 1000):** Fern #3258 tracking this on rerun.
- **VRAM ceiling for richer FiLM mid128:** 94.0 GiB / 96 GiB on pre-RFF model. Frieren must use film_mid=64 first for safety.
- **Run-to-run variance (~5–10% on test_avg):** Open issue — still relevant for borderline calls.
- **GitHub API rate limit:** Recurring 5000-req/hr exhaustion windows. Currently reset (3159 remaining at 01:30 UTC).

## Plateau-protocol queue (next R3 hypotheses, ranked)

1. ✗ Re-stratified loss reweighting ← FAILED (#3386 closed)
2. ✗ Multi-scale slice tokens ← FAILED (#3429 closed)
3. ✗ surf_weight sweep ← REBASE IN FLIGHT (#3406)
4. ✗ Per-block FiLM heads ← REBASE IN FLIGHT (#3468)
5. ✗ Richer FiLM conditioning ← REBASE IN FLIGHT (#3504)
6. ✗ Volume MAE reformulation ← IN FLIGHT (#3550)
7. ✗ AdamW betas / weight-decay sweep ← CLOSED (#3565; pre-RFF base, wd=0.05 hurts capacity-limited models)
7b. **Peak LR sweep {1e-3, 2.5e-4}** ← IN FLIGHT (#3693 alphonse)
8. ✗ RFF σ sweep {0.5, 1.0, 2.0} + n_freqs {16, 32} ← IN FLIGHT (#3599 edward)
9. ✗ Surface-only decoder head (parallel zero-init) ← IN FLIGHT (#3618 nezuko)
10. ✗ **Transolver depth n_layers 5 → 6** ← IN FLIGHT (#3658 tanjiro)
11. **Per-block × richer-FiLM compose** (if both #3468 + #3504 land on new base)
11. **Single-foil FiLM mask** (mask foil-2/gap/stagger features when gap=0; fixes mid128 single_in_dist regression frieren observed)
12. **Geometry-aware input features** (node distance to nearest surface; may be redundant with dsdf)
13. **Loss decomposition by domain** (per-split loss tracking + dynamic per-split weight)
14. **Soft slice-softmax temperature** (addresses gradient-norm root cause)
15. **TTA at inference** (test-time augmentation)
16. **Variance reduction via fixed `--seed` flag + multi-seed protocol**
17. **`single_in_dist` deep-dive** (per-sample Re vs error diagnostic)
18. **Deeper surface head** (if #3618 wins: 256/512 mid dim, or 2-block surface decoder)
19. **Surface + volume dual head** (if #3618 wins: also a parallel vol_head, learn full residual decomposition)

## Next immediate action

**All 8 students active.** Monitor for results from: rebase reruns (#3468 thorfinn, #3504 frieren, #3258 fern, #3351 askeladd) and fresh assignments (#3618 nezuko, #3693 alphonse LR sweep, #3599 edward RFF σ, #3658 tanjiro depth).

Key monitors:
- #3468 thorfinn per-block FiLM v2 on cosine+RFF base: predicted hopeful test ~64–66
- #3504 frieren richer-FiLM film_mid=64 on full stack: predicted hopeful test ~64.4
- #3599 edward RFF σ sweep: σ=0.5 and σ=2.0 bracket around the winning σ=1.0
- #3618 nezuko surface-only decoder head: predicted hopeful test ~64–66 (architectural surface specialization)
- #3658 tanjiro Transolver depth n_layers=6: predicted hopeful test ~65–66 (architectural capacity)
