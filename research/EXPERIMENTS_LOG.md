# SENPAI Research Results

_Track: `icml-appendix-charlie-pai2i-48h-r5` (round 5)._
_New entries appended as each PR is reviewed._

---

## 2026-05-16 16:55 — PR #4009 (charliepai2i48h5-nezuko): Gradient clip sweep {0.5, 1.0} on BF16+LS+n10 — **MERGED (NEW BEST)**

- branch: `charliepai2i48h5-nezuko/bf16-clip-sweep`
- hypothesis: clip=0.25 is double-regularizing with LayerScale's gating; test clip={0.5, 1.0}
- results (both arms beat baseline 67.19/58.05 at clip=0.25, 17 epochs each):

  | arm | clip | val_avg/mae_surf_p | test_avg/mae_surf_p | vs baseline | clip_frac @ep17 |
  |---|---|---|---|---|---|
  | arm-1 | 0.5 | 66.068 | 57.686 | -1.67% / -0.63% ✓ | ~0.997 |
  | **arm-2 (WINNER)** | **1.0** | **65.701** | **57.797** | **-2.22% / -0.44% ✓** | **~0.95** |

- per-split test surf_p (arm-2 clip=1.0): single=65.24 (-3.24% ✓), rc=71.43 (+2.35% ✗), cruise=38.31 (-0.91% ✓), re_rand=56.21 (-0.25% ✓)
- artifacts: `models/model-bf16-layerscale-clip10-20260516-152629/metrics.jsonl` (JSONL-verified)
- commentary: **MERGED** — clear new best. Mechanism: clip acts as effective step-length scale in the fully-clipped regime (clip × grad/‖grad‖). Prior clip=0.25 was under-stepping by 4×. At clip=1.0, ~5% of late-epoch steps are unclipped (first time in this programme). LayerScale γ stable throughout (no blow-up). arm-1 (clip=0.5) edges arm-2 on test_avg (57.69 vs 57.80) but arm-2 wins decisively on val. RC split regresses (+2.35%) — likely due to the increased effective step overshooting fine-grained camber pattern; all other splits improve. **clip=1.0 is now the new default on BF16+LS+n10 stack.**

---

## 2026-05-16 17:00 — PR #3971 (charliepai2i48h5-edward): EMA warm-up ramping on FP32 triple — CLOSED

- branch: `charliepai2i48h5-edward/ema-warmup-ramp`
- hypothesis: ramp EMA start_decay from 0.9 → {0.998, 0.9995} to open the smoothing window faster
- results:

  | arm | schedule | val_avg/mae_surf_p | test_avg/mae_surf_p | vs FP32 baseline (71.20/62.71) |
  |---|---|---|---|---|
  | arm-1 | warmup 100 → 0.998 | 74.115 | 65.433 | +4.1% / +4.3% worse |
  | arm-2 | warmup 200 → 0.9995 | 106.709 | 97.506 | +49.9% / +55.5% — completely divergent |

- artifacts: `target/models/model-triple-ema0998-warmup100-20260516-152746/metrics.jsonl`, `target/models/model-triple-ema09995-warmup200-20260516-142519/metrics.jsonl`
- commentary: CLOSED — mechanism confirmed (warm-up does open EMA faster: epoch 4 vs epoch 11 for static-0.999), but absolute val at 74.11 is worse than static-0.998 FP32 baseline (71.20) and far behind current BF16+LS+n10+clip=1.0 best (65.70). arm-2 fundamentally fails: decay=0.9995 with 4500 total updates → EMA window covers ~2000 steps, EMA never closes its 35% gap to raw-val in 12 epochs. Warm-up is not a route around the 12-epoch budget constraint. EMA experiments on any stack now definitively closed.

---

## 2026-05-16 16:20 — PR #4005 (charliepai2i48h5-tanjiro): BF16+LS+n10+EMA 0.998 missing cell — CLOSED

- branch: `charliepai2i48h5-tanjiro/bf16-layerscale-ema-n10`
- hypothesis: fill the missing cell — does EMA 0.998 help or hurt on the n10 stack at the 17-epoch BF16 horizon?
- results (JSONL-verified, 15 epochs timeout-bound):

  | Metric | This run (EMA 0.998) | Baseline #3527 (no EMA) | Δ |
  |---|---|---|---|
  | val_avg/mae_surf_p | **68.64** | 67.19 | **+2.16% (worse)** |
  | test_avg/mae_surf_p | **59.88** | 58.05 | **+3.15% (worse)** |
  | best_epoch | 15 / 50 | 17 / 50 | -2 epochs from EMA overhead |

- per-split test surf_p: single=67.81 (+0.58%), rc=70.50 (+1.02%), cruise=42.43 (**+9.74%**), re_rand=58.78 (+4.31%) — every split regresses
- artifacts: `models/model-bf16-layerscale-n10-ema0998-20260516-144223/metrics.jsonl`
- commentary: CLOSED — definitive negative result. EMA half-life (~350 steps) vs total training (~850 steps) gives EMA only 41% of training inside its smoothing window — too little to amortize at this horizon. EMA-raw gap collapsed from -12% at epoch 5 to -2.2% at epoch 15. Cost: ~9% per-epoch overhead drops effective budget 17→15 epochs. **All three EMA experiments on BF16+LS now confirm same pattern** (n14+EMA: 68.50, n10+EMA: 68.64, quad-compound also worse than n10 alone). **Drop EMA entirely on the BF16 stack.** Cruise regresses most because it had largest baseline margin (38.66) — most sensitive to losing 2 epochs of fine-tuning. Student's per-epoch EMA-raw delta table was the smoking gun.

---

## 2026-05-16 16:00 — PR #3983 (charliepai2i48h5-askeladd): Huber δ sweep {0.15, 0.5} on FP32 triple compound — CLOSED

- branch: `charliepai2i48h5-askeladd/huber-delta-sweep`
- hypothesis: bracket δ=0.3 with {0.15, 0.5} on FP32 triple compound (LayerScale + n_freqs=14 + EMA) to find optimum at extended budget
- results (both arms regressed vs δ=0.3 baseline 71.20/62.71):

  | arm | Huber δ | val_avg/mae_surf_p | test_avg/mae_surf_p | vs baseline |
  |---|---|---|---|---|
  | arm-1 | 0.15 | 73.35 | 64.68 | +3.0% / +3.1% worse |
  | arm-2 | 0.50 | ~74 | — | worse |

- commentary: CLOSED — δ=0.3 confirmed sweet spot on FP32 triple stack. δ=0.15 too aggressive (over-clips gradient at moderate residuals); δ=0.5 too lenient (loses outlier robustness). Note: with BF16+LS+n10 now the new baseline, this confirms δ=0.3 holds across precision regimes for surf_p prediction error distribution.

---

## 2026-05-16 16:00 — PR #3964 (charliepai2i48h5-alphonse): LayerScale γ-init sweep {0.005, 0.02} on FP32 triple — CLOSED

- branch: `charliepai2i48h5-alphonse/layerscale-gamma-init-sweep`
- hypothesis: bracket γ=0.01 with {0.005, 0.02} on FP32 triple compound to find optimal init magnitude
- results (both arms regressed vs γ=0.01 baseline 71.20/62.71):

  | arm | γ-init | val_avg/mae_surf_p | test_avg/mae_surf_p | vs baseline |
  |---|---|---|---|---|
  | arm-1 | 0.005 | 72.75 | 63.80 | +2.2% / +1.7% worse |
  | arm-2 | 0.020 | ~73 | — | worse |

- commentary: CLOSED — γ=0.01 confirmed optimal. γ=0.005 starts too gated (delayed feature mixing); γ=0.020 starts too unrestricted (residual gating less effective in first epochs). Combined with #3740 (asymmetric γ failed) and #3593 (γ=0.01 was original win), γ=0.01 fully validated as the right LayerScale init.

---

## 2026-05-16 13:55 — PR #3424 (charliepai2i48h5-askeladd): Tighter clip sweep max_norm=0.1 × Huber δ — CLOSED

- branch: `charliepai2i48h5-askeladd/tighter-clip-sweep`
- hypothesis: clip=0.1 + Huber δ ∈ {0.3, 0.1} on triple compound stack
- results (JSONL-verified, both arms 12 epochs, timeout-bound):

  | arm | Huber δ | clip | val_avg/mae_surf_p | test_avg/mae_surf_p | vs baseline (71.20) |
  |---|---|---|---|---|---|
  | arm-1 | 0.3 | 0.1 | 75.11 | 66.11 | +5.5% / +5.4% worse |
  | arm-2 | 0.1 | 0.1 | **74.67** | **65.11** | +4.9% / +3.8% worse |

- per-split test surf_p (arm-2 best): single=74.38, rc=73.75, cruise=46.84, re_rand=65.48
- artifacts: `models/model-huber*-clip01-fullstack-*/metrics.jsonl` (JSONL-verified)
- commentary: CLOSED — both arms regress. Mechanism: LayerScale γ=0.01 already gates effective gradient magnitudes through residual scaling, so clipping to 0.1 creates over-restricted "manifold-projected SGD with fixed step" that starves the optimizer. clip_frac=1.0 throughout. **Notable side-finding: δ=0.1 (74.67) beat δ=0.3 (75.11) even under tight clip** — hint that the optimal δ on the new triple-compound stack may have shifted from 0.3. Assigned follow-up: askeladd #3983 Huber δ sweep {0.15, 0.5} at standard clip=0.25.

---

## 2026-05-16 13:05 — PR #3878 (charliepai2i48h5-edward): EMA decay sweep {0.995, 0.999} — CLOSED

- branch: `charliepai2i48h5-edward/ema-decay-sweep`
- hypothesis: bracket EMA 0.998 with {0.995, 0.999} to find optimal decay at ~600-step budget
- results (JSONL-verified, both arms ran 12 epochs):

  | arm | EMA decay | val_avg/mae_surf_p | test_avg/mae_surf_p | vs baseline (71.20/62.71) |
  |---|---|---|---|---|
  | arm-1 | 0.995 | 71.94 | 63.33 | +1.03% / +0.99% worse |
  | arm-2 | 0.999 | 79.83 | 70.68 | +12.13% / +12.71% worse |

- per-split test surf_p (arm-1 best): single=70.83 (-0.55% ✓), rc=74.71 (+3.42% worse), cruise=45.04 (-0.33% ✓), re_rand=62.74 (+0.88% worse)
- artifacts: `models/model-triple-ema*-fullstack-*/metrics.jsonl` (JSONL-verified to 3 decimal places)
- commentary: CLOSED — clean negative result confirming 0.998 is optimal at current budget. arm-2's per-epoch curves vindicated the EMA half-life analysis: EMA val > raw val through epoch 11 (init-contaminated), only crossing over at epoch 12 — too late in our 12-epoch budget. Student's suggestion 3 (EMA warm-up ramp low→high) is the mechanism that unlocks 0.999's asymptotic regime; assigned as follow-up PR #3971.

---

## 2026-05-16 12:30 — PR #3882 (charliepai2i48h5-alphonse): SAM optimizer (ρ=0.05) — CLOSED

- branch: `charliepai2i48h5-alphonse/sam-optimizer`
- hypothesis: Sharpness-Aware Minimization → flat minima → better OOD generalization
- results (JSONL-verified):

  | metric | value | vs baseline (71.20/62.71) |
  |---|---|---|
  | val_avg/mae_surf_p | 126.72 | +78% worse |
  | test_avg/mae_surf_p | 116.15 | +85% worse |
  | best_epoch | 7 | (vs baseline's 12) |

- per-split test surf_p: single=155.70, rc=115.17, cruise=81.93, re_rand=111.79 — all massively above baseline
- epoch budget: ~283 s/epoch (2.0× overhead); only 7 epochs in 30 min vs baseline's 14
- artifacts: `models/model-layerscale-n14-ema0998-sam005-20260516-112730/metrics.jsonl`
- commentary: CLOSED — structural failure under 30-min budget. SAM trajectory is monotonically descending (algorithm works correctly), but halving available epochs is fatal vs the cosine-T_max=20 schedule. Student's analysis identified the gating criterion violation (val at epoch 6 = 135.63 vs <71.20 target) and noted amortized variants (LookSAM k=5 ~1.2× overhead, periodic-SAM every 4th batch ~1.25× overhead, ESAM) as the path forward. Those need fresh hypothesis PRs.

---

## 2026-05-16 11:35 — PR #3823 (charliepai2i48h5-nezuko): Lookahead optimizer wrapper {k=5, k=10} — CLOSED

- branch: `charliepai2i48h5-nezuko/lookahead-optimizer`
- hypothesis: Lookahead (Zhang et al. 2019) slow-anchor wrapper around AdamW provides variance reduction in our high-clip_frac regime
- results (terminal SENPAI-RESULT, stack on OLD n=10 baseline):

  | arm | k | best_epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | vs current best (71.20) |
  |---|---|---|---|---|---|
  | arm-1 | 5 | 13 | 82.21 | 71.40 | +15.5% / +13.8% worse |
  | arm-2 | 10 | 12 | 85.30 | 75.86 | +19.8% / +20.9% worse |

- per-split test surf_p (arm-1 best): single=83.21, rc=81.43, cruise=50.93, re_rand=70.02 — all worse
- artifacts: committed to student branch; JSONL-verified
- commentary: CLOSED — Lookahead pull-back mechanism interacts badly with LayerScale's per-channel γ gating. The slow weights average across phases where γ is at different positions in its learning trajectory, undoing channel selectivity. Combined with #3708 (β2) and #3782 (eps), the inner-AdamW + meta-optimizer knob family is exhausted: clip=0.25 already detoxifies heavy-tail gradients before they reach optimizer state, so no internal-optimizer modification helps.

---

## 2026-05-16 11:00 — PR #3527 (charliepai2i48h5-tanjiro): BF16 + LayerScale composition — SENT BACK FOR QUAD-COMPOUND ARM

- branch: `charliepai2i48h5-tanjiro/bf16-mixed-precision`
- hypothesis: BF16 mixed precision composes with LayerScale; extra epochs from 1.30× speedup compound the gains
- results (terminal=true SENPAI-RESULT posted, but merge conflict + missing EMA arm):

  | arm | stack | epochs | val_avg/mae_surf_p | test_avg/mae_surf_p | vs current best (71.20/62.71) |
  |---|---|---|---|---|---|
  | arm-1 (best test) | BF16 + LS + n=10, no EMA | 17 | 67.19 | **58.05** | -5.6% val / -7.4% test ✓ |
  | arm-2 | BF16 + LS + n=14, no EMA | 16 | 67.00 | 59.31 | -5.9% val / -5.4% test ✓ |

- per-split test surf_p (arm-1, JSONL-verified): single=67.42, rc=69.79, cruise=38.66, re_rand=56.35
- artifacts: `models/model-bf16-layerscale-fullstack-20260516-082748/metrics.jsonl` (JSONL-verified arm-1)
- commentary: **STRONG WIN** — BF16 enables 17 epochs vs current best's 12, and that extra convergence beats the n14+EMA triple compound on test. Even arm-2 (BF16+LS+n14 WITHOUT EMA) beats the triple compound (val=67.00 vs 71.20), demonstrating that BF16's epoch budget is more valuable than EMA's smoothing in this regime. Per-split: best gains on OOD (cruise -11.8%, re_rand -9.4%) — exactly the splits we most needed to fix. Peak VRAM 36.9 GB. **Sent back** to: (1) rebase onto new advisor HEAD (which has EMA from #3192), (2) run BF16 + LS + n_freqs=14 + EMA 0.998 quad-compound to lock in the new best. If quad-compound holds, we land at sub-60 val. If it regresses, arm-1's 58.05 test is still merge-worthy.

---

## 2026-05-16 11:05 — PR #3740 (charliepai2i48h5-frieren): Asymmetric LayerScale γ-init attn vs MLP — CLOSED

- branch: `charliepai2i48h5-frieren/asymmetric-layerscale`
- hypothesis: γ_attn=0.001, γ_mlp=0.01/0.03 — separate inits to match natural trajectory
- results (terminal SENPAI-RESULT posted, stack on OLD n=10 baseline):

  | arm | γ_attn / γ_mlp | val_avg/mae_surf_p | test_avg/mae_surf_p | vs current best (71.20) |
  |---|---|---|---|---|
  | arm-1 | 0.001 / 0.01 | 80.49 | 70.63 | +13.0% worse |
  | arm-2 | 0.001 / 0.03 | 77.54 | 66.86 | +8.9% worse |

- per-split test surf_p (arm-2): single=77.87, rc=76.52, cruise=47.40, re_rand=65.64 — all worse than current best
- commentary: CLOSED — γ-stats nail the mechanism. Both arms converge to similar resting values regardless of init (γ_attn ~0.007-0.017, γ_mlp ~0.030-0.052). Symmetric γ=0.01 already discovers the right effective asymmetry through learning; manual init steering provides no benefit. clip_frac=1.0 throughout — further constraining what's already optimal. Test was on old n=10 stack but mechanism analysis is conclusive enough that re-running on triple compound wouldn't change the verdict.

---

## 2026-05-16 10:45 — PR #3192 (charliepai2i48h5-edward): EMA decay sweep on LayerScale+n14 stack — MERGED (NEW BEST)

- branch: `charliepai2i48h5-edward/ema-on-layerscale`
- hypothesis: EMA checkpoint averaging (torch.optim.swa_utils.AveragedModel) on the LayerScale + n_freqs=14 stack
- results:

  | arm | EMA decay | n_freqs | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
  |---|---|---|---|---|---|
  | arm-1: EMA on n_freqs=10 | 0.998 | 10 | 73.60 | 66.87 | 13 |
  | arm-2: EMA on n_freqs=10 (ema=0.999) | 0.999 | 10 | — | — | — |
  | arm-3: EMA 0.998 + LayerScale + n_freqs=10 | 0.998 | 10 | ~72 | — | — |
  | **arm-4 (winner): EMA 0.998 + LayerScale + n_freqs=14** | **0.998** | **14** | **71.20** | **62.71** | **12** |
  | arm-5: EMA 0.998 + LayerScale + n_freqs=10 (no n14) | 0.998 | 10 | 72.80 | 65.38 | 13 |

- artifacts: `models/model-layerscale-001-ema-0998-n14-fullstack-20260516-073558/metrics.jsonl`
- per-split test surf_p (arm-4 winner):

  | split | test surf_p | Δ vs prior best (72.77/65.12) |
  |---|---|---|
  | single_in_dist | 71.22 | -9.65% ✓ |
  | geom_camber_rc | 72.24 | -4.72% ✓ |
  | geom_camber_cruise | 45.19 | +3.04% |
  | re_rand | 62.19 | +0.36% |

- commentary: **NEW BEST** — val=71.20 / test=62.71, -2.16% val / -3.71% test vs prior best (72.77/65.12). Triple compound (LayerScale + n_freqs=14 + EMA 0.998) works because EMA checkpoint averaging compensates for the under-convergence of the n14+LayerScale compound in the ~12-epoch budget. The compound without EMA (alphonse #3730) regressed +4-5%; EMA adds enough smoothing to bridge the gap. OOD single and rc splits show biggest gains (-9.65%, -4.72%); cruise slightly regresses. LayerScale γ dynamics remain healthy with EMA. Peak memory 48.1 GB, ~152 s/epoch. Cumulative improvement now -44.7% from round-5 start.

---

## 2026-05-16 10:50 — PR #3730 (charliepai2i48h5-alphonse): LayerScale + n_freqs=14 compound — CLOSED

- branch: `charliepai2i48h5-alphonse/layerscale-n14-compound`
- hypothesis: LayerScale γ=0.01 + n_freqs=14 without EMA — compound two merged wins
- results:

  | arm | val_avg/mae_surf_p | test_avg/mae_surf_p | vs baseline (72.77) |
  |---|---|---|---|
  | arm-1 γ=0.01 (seed 1) | 76.32 | 69.55 | +4.9% worse |
  | arm-1 γ=0.01 (seed 2) | 75.76 | 67.94 | +4.1% worse |
  | arm-2 γ=0.003 | 76.49 | 67.56 | +5.1% worse |

- artifacts: committed to student branch; best arm-1 JSONL verified
- per-split test surf_p (arm-1 primary): single=81.39, rc=83.43, cruise=47.92, re_rand=65.46 — all worse than baseline
- commentary: CLOSED — sub-additive compound under 30-min timeout. LayerScale's channel gating needs more steps to align with the wider Fourier input space (space_dim=58). γ dynamics identical to n=10 (γ_attn stays near 0.01, γ_mlp grows 3-4×) suggesting convergence is the bottleneck, not the mechanism. Formally superseded by edward #3192 which adds EMA 0.998 to resolve the under-convergence issue (val=71.20). BF16 (#3527) remains the path to revisit this compound with a fair budget.

---

## 2026-05-16 10:52 — PR #3782 (charliepai2i48h5-fern): AdamW eps sweep {1e-6, 1e-7} — CLOSED

- branch: `charliepai2i48h5-fern/adam-eps-on-layerscale`
- hypothesis: Raising AdamW eps stabilizes small-v_t parameters (LayerScale γ channels near zero)
- results:

  | arm | adam_eps | val_avg/mae_surf_p | test_avg/mae_surf_p | vs baseline (72.77) |
  |---|---|---|---|---|
  | arm-1 | 1e-6 | 83.39 | 75.05 | +14.6% worse |
  | arm-2 | 1e-7 | 77.17 | 68.97 | +6.0% worse |
  | baseline | 1e-8 | 72.77 | 65.12 | — |

- artifacts: `models/model-layerscale-adam-eps-*/metrics.jsonl`
- per-split test surf_p (arm-2 best): single=78.86, rc=78.01, cruise=49.99, re_rand=69.01 — all worse
- commentary: CLOSED — hypothesis cleanly falsified. Monotone degradation: larger eps is strictly worse. Default eps=1e-8 confirmed optimal. The rationale (spiky v_t destabilizing small-γ channels) is invalidated by grad_clip=0.25 already bounding per-step magnitudes. Together with PR #3708 (β2 sweep falsified), the AdamW denominator knobs are confirmed non-viable. Student conclusion correct: schedule tuning (T_max) is the more promising next lever, now assigned as PR #3883.

---

## 2026-05-15 15:35 — PR #3199 (charliepai2i48h5-fern): Decoupled surface/volume decoder heads — REQUEST CHANGES

- branch: `charliepai2i48h5-fern/dualhead-surface-volume`
- hypothesis: separate LayerNorm+MLP heads for surface vs volume nodes in the final TransolverBlock
- arms:

  | arm | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | secs/epoch |
  |---|---|---|---|---|
  | dualhead (baseline hyperparams) | 122.40 | 110.38 (NaN-safe) | 13/50 | ~135s |
  | dualhead-wd2x (surface head wd=2e-4) | 129.18 | 118.79 (NaN-safe) | 13/50 | ~135s |

- artifacts: `models/model-charliepai2i48h5-fern-dualhead-20260515-131247/`, `models/model-charliepai2i48h5-fern-dualhead-wd2x-20260515-134929/`
- per-split test surf_p (arm-1): single=125.03, rc=117.46, cruise=85.75, re_rand=113.28
- also included: NaN-safe evaluate_split bug fix (pre-dates Huber merge; same fix)
- params: 679K (+17K vs reference)
- verdict: arm-1 solid vs prior round; both arms beaten by Huber merge (103.18). **Sent back** to rebase and combine dualhead with `--huber_delta 0.3`.

---

## 2026-05-15 15:35 — PR #3227 (charliepai2i48h5-thorfinn): Surf-weight curriculum 1→20 — REQUEST CHANGES

- branch: `charliepai2i48h5-thorfinn/surf-weight-anneal-1-20`
- hypothesis: anneal surf_weight linearly 1→20 over first 60% epochs beats constant 10
- arms:

  | arm | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
  |---|---|---|---|
  | surf-anneal-1-20 (schedule) | 130.40 | 117.15 | 14/50 |
  | surf-final-20 (constant 20) | 138.77 | 126.65 | best epoch similar |

- artifacts: `models/model-surf-anneal-1-20-20260515-141406/`, `models/model-surf-final-20-20260515-144850/`
- per-split test surf_p (anneal): single=142.05, rc=131.25, cruise=85.56, re_rand=109.76
- analysis: schedule arm beats constant-20 arm by ~8pts — curriculum effect is real. Both beaten by Huber merge (103.18). Mechanism (loss-space schedule) is orthogonal to Huber.
- verdict: **Sent back** to rebase and combine `--surf_weight 1.0 --surf_weight_final 20.0 --surf_weight_anneal_frac 0.6 --huber_delta 0.3`.

---

## 2026-05-15 15:35 — PR #3225 (charliepai2i48h5-tanjiro): Multi-scale slice attention (32+128) — CLOSED

- branch: `charliepai2i48h5-tanjiro/multiscale-slice-attention`
- hypothesis: (32, 128) dual-scale PhysicsAttention reduces val_avg/mae_surf_p 5-10%
- arms:

  | arm | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | secs/epoch |
  |---|---|---|---|---|
  | multiscale-32-128 | 163.20 | 150.69 (NaN-safe re-eval) | 8/50 | ~248s |
  | multiscale-32-128-bigger (n_hidden=160) | 169.24 | NaN (no re-eval) | 6/50 | ~248s |

- artifacts: `models/model-charliepai2i48h5-tanjiro-multiscale-32-128-20260515-131346/`, `models/model-charliepai2i48h5-tanjiro-multiscale-32-128-bigger-20260515-135926/`
- root cause: attention path ~85% slower per epoch (248s vs 130s baseline), only 8 epochs in 30-min cap vs 14 for single-scale. Both arms ~30% worse than current baseline.
- verdict: **Closed**. Multiscale at (32,128) too slow. Revisit with (32,64) or single-last-layer multiscale.

---

## 2026-05-15 16:28 — PR #3213 (charliepai2i48h5-frieren): Huber loss delta=0.3/1.0 — **MERGED** (new best)

- branch: `charliepai2i48h5-frieren/huber-pressure-loss`
- hypothesis: replacing MSE with Huber loss (switching from quadratic to linear gradient above |residual|>delta) downweights heavy-tail high-Re outliers
- arms:

  | arm | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | secs/epoch |
  |---|---|---|---|---|
  | **huber-0.3 (delta=0.3)** | **103.18** | **92.02** | 13/50 | ~130s |
  | huber-1.0 (delta=1.0) | 118.12 | 106.30 | 14/50 | ~130s |

- artifacts: `models/model-huber-0.3-20260515-140457/`, `models/model-huber-1.0-20260515-130904/`
- per-split test surf_p (huber-0.3): single=111.93, rc=102.85, cruise=62.84, re_rand=90.45
- analysis: tighter delta=0.3 wins by 15pts val / 14pts test over delta=1.0. Smaller delta = more aggressive downweighting of large residuals. Result shows the pressure target distribution is heavy-tailed enough that even L1-like behavior (delta→0) is beneficial. Also included NaN-safe evaluate_split fix.
- verdict: **MERGED** as new round-5 baseline. val_avg=103.18, test_avg=92.02.

---

## 2026-05-15 16:28 — PR #3182 (charliepai2i48h5-askeladd): Gradient clipping (max_norm=0.5/1.0) — REQUEST CHANGES

- branch: `charliepai2i48h5-askeladd/gradient-clipping-heavy-tail`
- hypothesis: gradient clipping at max_norm=1.0 or 0.5 stabilizes heavy-tail gradient signal
- arms (+ in-PR baseline):

  | arm | val_avg/mae_surf_p | 3-split test/mae_surf_p | best_epoch |
  |---|---|---|---|
  | no-clip-baseline | 128.69 | 126.56 | 13/50 |
  | grad-clip-1.0 | 116.02 | 115.90 | 14/50 |
  | **grad-clip-0.5** | **113.90** | **113.64** | 13/50 |

- artifacts: `models/model-charliepai2i48h5-askeladd-no-clip-baseline-20260515-141047/`, `models/model-charliepai2i48h5-askeladd-grad-clip-1.0-20260515-130357/`, `models/model-charliepai2i48h5-askeladd-grad-clip-0.5-20260515-152706/`
- note: test_avg NaN for all 3 arms (no NaN-safe fix applied); used 3-split avg as proxy
- clip_frac=1.0 at both thresholds — every optimizer step was clipped. Pre-clip mean grad norm ~46, max >1000 at high-Re batches. Confirms heavy-tail gradient problem.
- verdict: Both clip arms beat pre-Huber no-clip baseline by 9-11%; **does NOT beat Huber baseline** (113.90 vs 103.18). Sent back to rebase + combine `--huber_delta 0.3 --grad_clip_max_norm 0.5`.

---

## 2026-05-15 19:26 — PR #3182 (charliepai2i48h5-askeladd): Huber-0.3 + gradient clipping (clip=0.25) — **MERGED** (new best)

- branch: `charliepai2i48h5-askeladd/gradient-clipping-heavy-tail`
- hypothesis: Huber-0.3 + grad_clip_max_norm=0.25 are additive — attack heavy-tail gradients at different levels (per-sample residual vs batch-level update magnitude)

- arms:

  | arm | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
  |---|---|---|---|
  | huber-0.3 + clip-0.5 | 102.35 | 94.84 | 12/50 |
  | **huber-0.3 + clip-0.25** | **98.62** | **88.14** | 14/50 |

- artifacts: `models/model-charliepai2i48h5-askeladd-huber-0.3-clip-0.25-20260515-182526/`, `models/model-charliepai2i48h5-askeladd-huber-0.3-clip-0.5-20260515-172602/`
- per-split test surf_p (arm-2/clip-0.25): single=104.75, rc=104.65, cruise=59.24, re_rand=83.90
- n_params: 662K, peak VRAM 42.12GB, clip_frac=1.00 at BOTH thresholds
- gradient diagnostics: mean pre-clip norm ~7 (vs ~46 pre-Huber); max ~25. Huber tames per-sample tail but batch-level heavy-tail variance remains, so clipping is still doing real work on top.
- single regression: val_geom_camber_rc slightly worse with clip-0.25 (119.3 vs 107.2 for clip-0.5 val); tighter clipping discards more of the high-pressure gradient signal on that hardest split.
- verdict: **MERGED** as new round-5 baseline. val_avg=98.62, test_avg=88.14.

---

## 2026-05-15 19:30 — PR #3334 (charliepai2i48h5-tanjiro): Wider Transolver n_hidden=192 — CLOSED

- branch: `charliepai2i48h5-tanjiro/wider-transolver-192`
- hypothesis: n_hidden=192 with Huber-0.3 reduces val_avg by 5–10%
- arms:

  | arm | val_avg/mae_surf_p | test_avg/mae_surf_p | s/epoch | best_epoch |
  |---|---|---|---|---|
  | wider-192-huber03 (n_head=4) | 117.88 | 109.12 | 186 | 9/10 |
  | wider-192-heads6-huber03 (n_head=6) | 123.19 | 112.40 | 204 | 9/9 |

- artifacts: `models/model-wider-192-huber03-20260515-172706/`, `models/model-wider-192-heads6-huber03-20260515-182607/`
- analysis: 5× per-epoch slowdown (36s→186s). Only 9-10 epochs fit in budget vs 14 for baseline. Both arms still improving monotonically at timeout — mechanism is sound but budget is the constraint.
- n_head=4 (dim_head=48) beats n_head=6 (dim_head=32) by 5 val points.
- verdict: **Closed**. Tanjiro reassigned to n_hidden=160 + T_max-aligned schedule (#3419) — same width-scaling direction, smaller step + proper LR cycle within budget.

---

## 2026-05-15 19:30 — PR #3355 (charliepai2i48h5-alphonse): Physics-informed input features — CLOSED

- branch: `charliepai2i48h5-alphonse/physics-features`
- hypothesis: Re_x proxy, gap×log(Re), sin/cos AoA features reduce val_avg 4–8%
- arms:

  | arm | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
  |---|---|---|---|
  | physfeat-all-huber03 (6 features) | 109.80 | 100.88 | 14/50 |
  | physfeat-rephys-huber03 (2 features: Re_x, gap_re) | 105.04 | 93.18 | 14/50 |

- n_params: 663.9K (+1.8K), same s/epoch as baseline (~132s)
- analysis (from student): cos(aoa) std=0.0038 → near-constant input, zero information gain. log_re_x and gap_re are arithmetic combinations of existing features — Transolver MLP synthesizes these in one pass, no inductive bias advantage. 5-layer attention gives global context the MLP would need multiple hops for in a GNN. B-GNN transfer failed because GNNs need explicit feature bridges; attention doesn't.
- arm-2 beats baseline on cruise (75.26 vs 77.24 val) and rc (115.13 vs 116.40) but regresses on single (+7.8) and re_rand (+2.8), netting -1.8% overall.
- verdict: **Closed**. Alphonse reassigned to log-space pressure loss (#3420) — different mechanism targeting the training loss dynamic-range problem at residual level.

---

## Wave-2 new assignments (2026-05-15)

- PR #3333 — charliepai2i48h5-frieren: LR schedule alignment (T_max=14 vs T_max=20 with huber_delta=0.3)
- PR #3334 — charliepai2i48h5-tanjiro: Wider Transolver n_hidden=192 (CLOSED, see above)

## Currently rebasing+combining (sent back)

- PR #3199 (fern): dualhead + Huber-0.3 (WIP)
- PR #3227 (thorfinn): surf-anneal + Huber-0.3 + try terminal weight=30 (WIP)

## Wave-3 new assignments (2026-05-15)

- PR #3419 — charliepai2i48h5-tanjiro: n_hidden=160 + T_max-aligned LR schedule (both huber_delta=0.3 + clip-0.25)
- PR #3420 — charliepai2i48h5-alphonse: log-space pressure loss with sign-preserving log transform

## 2026-05-15 17:10 — PR #3178 (charliepai2i48h5-alphonse): Per-sample pressure-scale normalization — CLOSED

- branch: `charliepai2i48h5-alphonse/per-sample-pressure-scale-norm`
- hypothesis: normalize each sample's targets by that sample's own pressure std before computing loss, forcing the model to learn flow shape independently of scale

- arms:

  | arm | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
  |---|---|---|---|
  | perSampleScale (MSE base) | 122.42 | 111.18 | 14/50 |
  | perSampleScale+huber (delta=1.0) | 127.99 | 116.01 | 11/50 |

- artifacts: `models/model-perSampleScale-20260515-142709/`, `models/model-perSampleScale-huber-20260515-152743/`
- per-split val surf_p (arm-1): single=165.59, rc=134.16, cruise=82.92, re_rand=107.04
- per-split test surf_p (arm-1): single=150.22, rc=121.94, cruise=70.02, re_rand=102.55
- n_params: 662K (unchanged), peak VRAM: 42.16GB, grad clip at max_norm=1.0 required for stability (raw grad norm spikes to ~6.4M in epoch 1 without clip)
- analysis:
  - Arm-1 beats pre-Huber no-clip baseline (128.69) but fails to match Huber-0.3 (103.18)
  - Arm-2 (per-sample scale + Huber delta=1.0) is strictly worse than arm-1 on all splits; student analysis: Huber's robustness to outliers downweights exactly the high-pressure (single/RC) samples that dominate val_avg — double-suppression is anti-synergistic
  - Cruise split was strong for both arms (val 82.9/test 70.0); mechanism helps low-pressure regime
  - Hard splits (single_in_dist, geom_camber_rc) regressed significantly — per-sample normalization removes scale cues that distinguish high/low Re regimes
  - Mechanism is orthogonal to Huber at input level vs loss level, but empirically they interfere
- verdict: **Closed**. Both arms lose to Huber-0.3 baseline. Composition tested and confirmed anti-synergistic. Alphonse reassigned to physics-informed features (PR #3355).

---

## 2026-05-15 19:52 — PR #3221 (charliepai2i48h5-nezuko): Fourier positional features — **MERGED** (new best)

- branch: `charliepai2i48h5-nezuko/fourier-positional-features`
- hypothesis: replacing raw (x,z) coordinates with multi-frequency log-spaced Fourier positional embeddings lifts spatial representation quality, benefiting all splits

- arms:

  | arm | n_freqs | space_dim | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
  |---|---|---|---|---|---|
  | fourier-n6 | 6 | 26 | 98.20 | 88.91 | 12/14 |
  | **fourier-n10** | **10** | **42** | **89.27** | **79.43** | **14/14** |

- artifacts: `models/model-charliepai2i48h5-nezuko-fourier-n6-20260515-183742/metrics.jsonl`, `models/model-charliepai2i48h5-nezuko-fourier-n10-20260515-191358/metrics.jsonl`
- per-split test surf_p (n=10): single=93.65, rc=88.94, cruise=56.92, re_rand=78.20
- n_params: 668K (n=6) / 673K (n=10) — Fourier adds ~4K params, near zero overhead
- peak VRAM: ~46 GB (both arms)
- both runs were wall-clock capped at epoch 14 (not epoch-budget capped) — best checkpoint at last epoch in both cases; model was still improving
- analysis: 6→10 freqs shows continued scaling (val gap 98.20→89.27, 9.1pts). High-freq channels did not hurt despite aliasing concern — slice-attention is learning to filter. Cruise camber split shows largest gain from going 6→10 (cruise test: 69.20→56.92). arm-1 (n=6) has regression on cruise vs Huber baseline (+10.1%), arm-2 fixes this. Hypothesis strongly confirmed.
- verdict: **MERGED** as new round-5 baseline. val_avg=89.27, test_avg=79.43.

---

## 2026-05-15 19:52 — PR #3333 (charliepai2i48h5-frieren): LR T_max alignment — SENT BACK (new baseline test)

- branch: `charliepai2i48h5-frieren/lr-schedule-alignment`
- hypothesis: cosine annealing with T_max=14 or T_max=20 (aligned to wall-clock budget) outperforms T_max=50 (near-constant LR)

- arms:

  | arm | T_max | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | LR at best |
  |---|---|---|---|---|---|
  | huber03-tmax14 | 14 | 95.40 | 86.76 | 14/14 | 6.27e-6 |
  | **huber03-tmax20** | **20** | **94.21** | **86.01** | **14/14** | **1.37e-4** |

- artifacts: `models/model-huber03-tmax14-20260515-172631/metrics.jsonl`, `models/model-huber03-tmax20-20260515-182450/metrics.jsonl`
- per-split test surf_p (T_max=20): single=103.44, rc=95.98, cruise=61.94, re_rand=82.68
- LR trajectory confirmed via per-epoch logging. T_max=50 → near-constant LR at 5e-4 for all 14 epochs. T_max=14 → decays to ~0 by last epoch. T_max=20 → retains 1.37e-4 at epoch 14.
- analysis: T_max=20 beats T_max=14 because model is still in an active learning regime at the 14-epoch cutoff — decaying to ~0 too early (T_max=14) leaves capability on the table. Free-lunch improvement: pure LR schedule change, no architecture/loss changes.
- verdict: Both arms beat old baseline (98.62). But PR #3221 (Fourier n=10, val=89.27) also merged in this review pass, setting a higher bar. T_max=20 result (94.21) does NOT beat new Fourier baseline (89.27). **Sent back** to combine: Fourier n=10 + Huber-0.3 + clip-0.25 + T_max=20.

---

## 2026-05-15 20:00 — PR #3199 (charliepai2i48h5-fern): Dualhead decoder (rebased onto Huber) — CLOSED

- branch: `charliepai2i48h5-fern/dualhead-surface-volume`
- hypothesis (rebased): dualhead architecture (separate LayerNorm+MLP for surface/volume) + Huber-0.3 compound over Huber-only baseline

- arms:

  | arm | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
  |---|---|---|---|
  | dualhead+huber03 (no clip) | 106.62 | 95.69 | 13/14 |
  | dualhead+huber03+gc05 | 102.36 | 93.18 | 13/14 |

- artifacts: `models/model-charliepai2i48h5-fern-dualhead-huber03-20260515-173939/metrics.jsonl`, `models/model-charliepai2i48h5-fern-dualhead-huber03-gc05-20260515-183440/metrics.jsonl`
- per-split test surf_p (arm-2): single=110.54, rc=111.32, cruise=63.53, re_rand=87.34
- analysis: arm-1 (dualhead+Huber only) is WORSE than Huber-only baseline (106.62 vs 103.18). arm-2 (adds gc-0.5) barely beats old Huber baseline (102.36 vs 103.18, -0.8% val) but worse on test (93.18 vs 92.02). Huber and dualhead both address the surface/volume heavy-tail specialization problem — they are redundant. Once Huber stabilizes the surface gradient signal, the single shared head adapts implicitly. Grad-clip only partially compensates for the training instability dualhead introduces.
- new baseline for comparison: Fourier n=10 val=89.27 (merged PR #3221). Both arms are >14% behind.
- verdict: **Closed**. Dualhead not competitive with current stack. Fern reassigned to Gaussian random Fourier features (#3439).

---

## Wave-4 new assignments (2026-05-15 20:00)

- PR #3438 — charliepai2i48h5-nezuko: Fourier freq sweep n_freqs∈{12,14} + grad_clip=0.25 (continuation of Fourier scaling)
- PR #3439 — charliepai2i48h5-fern: Gaussian random Fourier features σ∈{1.0,5.0} (alternative Fourier basis from Tancik 2020)

---

## 2026-05-15 22:40 — PR #3420 (charliepai2i48h5-alphonse): Log-space pressure loss — CLOSED (negative)

- branch: `charliepai2i48h5-alphonse/log-p-loss`
- hypothesis: sign-preserving softlog transform on raw pressure + tunable `log_p_scale` equalizes dynamic range across low/high-pressure splits

- arms:

  | arm | log_p_scale | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
  |---|---|---|---|---|
  | log-p-scale1 | 1.0 | 138.14 (+40%) | 124.54 (+41%) | 13/14 |
  | log-p-scale2 | 2.0 | 141.37 (+43%) | 130.04 (+48%) | 14/14 |

- artifacts: `models/model-log-p-scale1-huber03-clip025-20260515-202719/metrics.jsonl`, `models/model-log-p-scale2-huber03-clip025-20260515-212342/metrics.jsonl`
- per-split test surf_p (arm-1): single=187.80 (+79%), rc=131.46 (+26%), cruise=73.25 (+24%), re_rand=105.65 (+26%)
- scale diagnostic: `logp_to_vel_ratio` started at 5.25× (scale=1.0) and grew to 11× by epoch 14; scale=2.0 started at 7.9×, grew to 21×. Target ratio is ~1.0. Both arms massively miscalibrated.
- Two compounding failure modes identified by student:
  1. Calibration miss: scale=0.2 would be needed to hit ratio~1.0; suggested values (1.0/2.0) far off
  2. Structural mismatch: log-space gradient ∝ 1/(1+|p|) — systematically downweights high-pressure nodes (those with largest absolute MAE contribution). Loss is anti-aligned with primary metric.
- verdict: **Closed**. Structural log-space/MAE mismatch is irrecoverable; even calibrated scale=0.2 would not fix the gradient scaling issue. Alphonse reassigned to stochastic depth (#3509).

---

---

## 2026-05-15 23:29 — PR #3438 (charliepai2i48h5-nezuko): Fourier freq sweep n=12/14 + clip — SENT BACK

- branch: `charliepai2i48h5-nezuko/fourier-nfreqs-sweep`
- hypothesis: n_freqs∈{12,14} + grad_clip=0.25 vs baseline n=10 (no clip)

- arms:

  | arm | n_freqs | clip | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | clip_frac |
  |---|---|---|---|---|---|---|
  | fourier-n12-clip025 | 12 | 0.25 | 100.35 (+12.4%) | 87.33 (+9.9%) | 12/14 | 1.000 |
  | fourier-n14-clip025 | 14 | 0.25 | 92.53 (+3.7%) | 82.78 (+4.2%) | 11/14 | 1.000 |

- baseline (n=10, no clip): val=89.27, test=79.43
- both arms CONFOUNDED: they changed n_freqs AND added clip simultaneously vs baseline (which had no clip)
- student correctly identified the confound: n=14+clip regression is most plausibly from the clip being over-tight (clip_frac=1.0 → >95% of gradient magnitude cut every step), not from higher frequencies
- n=14 > n=12 within this experiment → extra frequencies still productively used despite aggressive clip; both arms peak earlier than n=10 baseline (consistent with clip blocking fine-tuning convergence)
- verdict: **Sent back** to deconfound — test n_freqs=14 on the new full stack baseline (where clip is already included).

---

## 2026-05-15 23:29 — PR #3419 (charliepai2i48h5-tanjiro): n_hidden=160 + T_max aligned — CLOSED

- branch: `charliepai2i48h5-tanjiro/wider-160-tmax-aligned`
- hypothesis: n_hidden=160 + T_max=11 (budget-aligned) beats n_hidden=128 baseline

- arms:

  | arm | n_head | n_params | s/epoch | epochs | val_avg/mae_surf_p | test_avg/mae_surf_p |
  |---|---|---|---|---|---|---|
  | wider-160-tmax-aligned (n_head=4) | 4 | 1.03M | 167 | 11/50 | 99.10 | 88.53 |
  | wider-160-heads6-tmax-aligned (n_head=6) | 6 | 999K | 185 | 10/50 | 100.40 | 89.83 |

- speed wall: 4.6× per-epoch overhead (167s vs 36s). Only 10-11 epochs in budget. Budget constraint dominates.
- interesting signal: both arms beat baseline on geom_camber_rc by ~7% (better geometric OOD generalization from higher capacity)
- verdict: **Closed**. Width scaling infeasible at current speed without AMP. Tanjiro reassigned to bf16 (#3527).

---

## 2026-05-15 23:28 — PR #3333 (charliepai2i48h5-frieren): Full stack Fourier+Huber+T_max=20+clip — **MERGED** (new best)

- branch: `charliepai2i48h5-frieren/lr-schedule-alignment`
- hypothesis (revised): Fourier n=10 + Huber delta=0.3 + T_max=20 + grad_clip=0.25 all compose

- arm:

  | run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | LR at best | clip_frac |
  |---|---|---|---|---|---|
  | fourier-n10-tmax20-clip025 | **84.59** | **73.89** | 14/14 | 1.37e-4 | 1.000 |

- artifacts: `models/model-fourier-n10-tmax20-clip025-20260515-222425/metrics.jsonl`
- per-split test surf_p: single=86.87, rc=86.21, cruise=51.47, re_rand=71.01
- all 4 splits improved vs Fourier-only baseline. Monotone val improvement across all 14 epochs — still learning at timeout.
- cumulative improvement: val ~128.69 → 103.18 → 98.62 → 89.27 → **84.59** (-34% from round-5 start)
- verdict: **MERGED** as new round-5 baseline. val_avg=84.59, test_avg=73.89. Frieren reassigned to grad-clip sweep (#3529).

---

## Wave-5 new assignments (2026-05-15 23:40)

- PR #3509 — charliepai2i48h5-alphonse: Stochastic depth (DropPath) drop_path∈{0.05,0.10} on full stack
- PR #3527 — charliepai2i48h5-tanjiro: Mixed precision BF16 training, n_hidden=128 + n_hidden=160
- PR #3529 — charliepai2i48h5-frieren: Grad-clip sweep max_norm∈{0.5,1.0} on full stack

---

## 2026-05-16 01:33 — PR #3509 (charliepai2i48h5-alphonse): Stochastic depth DropPath — CLOSED (negative)

- branch: `alphonse/stochastic-depth-droppath`
- hypothesis: DropPath (linear schedule across 5 blocks) improves OOD generalization on geom_camber_rc and re_rand

- arms:

  | arm | drop_path | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | vs baseline (test) |
  |---|---|---|---|---|---|
  | baseline (PR #3221) | 0 | 89.27 | 79.43 | 14 | — |
  | arm-1 | 0.05 | 93.85 | 84.06 | 13/13 | +5.8% (worse) |
  | arm-2 | 0.10 | 101.80 | 93.34 | 12/13 | +17.5% (worse) |

- artifacts: `models/model-droppath05-fourier-clip025-20260515-233429/metrics.jsonl`, `models/model-droppath10-fourier-clip025-20260516-000852/metrics.jsonl`
- per-split test (arm-1 dp=0.05): single=93.82 (+0.2%), rc=87.94 (-1.1%✓), cruise=67.75 (+19%), re_rand=86.75 (+10.9%)
- per-split test (arm-2 dp=0.10): single=108.34, rc=111.51, cruise=66.13, re_rand=87.37
- key finding: tiny 1.1% gain on geom_camber_rc at dp=0.05, but all other splits and val_avg regressed. DropPath's implicit ensembling requires deep models (24+ blocks) trained for 100s of epochs; our 5-block Transolver at 13-epoch wall-clock budget is underfit, not overfit — adding noise to the optimization path only slows convergence.
- verdict: **Closed**. Clean negative result. Student flagged LayerScale as better alternative. Alphonse reassigned to LayerScale (#3593).

---

## 2026-05-16 01:33 — PR #3439 (charliepai2i48h5-fern): Gaussian random Fourier features σ sweep — SENT BACK (refine)

- branch: `fern/gaussian-random-fourier-features`
- hypothesis: Gaussian random Fourier features with σ tuned to target spatial frequency beat log-spaced deterministic features (PR #3221)

- arms:

  | arm | σ | n_freqs | space_dim | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
  |---|---|---|---|---|---|---|
  | baseline (log-spaced) | — | 10 | 42 | 89.27 | 79.43 | 14 |
  | arm-1 | 1.0 | 10 | 22 | 100.33 | 91.75 | 12 |
  | arm-2 | 5.0 | 10 | 22 | 90.18 | 81.02 | 13 |

- artifacts: `target/models/model-rff-sigma1-n10-clip025-20260515-233147/metrics.jsonl`, `target/models/model-rff-sigma5-n10-clip025-20260516-002524/metrics.jsonl`
- per-split test (σ=5.0): single=94.31 (+0.7%), rc=97.03 (+9.1%⚠), cruise=55.00 (-3.4%✓), re_rand=77.75 (-0.6%✓)
- key finding: σ=5.0 beats baseline on 2/4 test splits (cruise -3.4%, re_rand -0.6%). geom_camber_rc regression (+9.1%) pulls average above baseline. Log-spaced covers 10 octaves simultaneously (broader spectral coverage); RFF concentrates around one frequency band. Missing --lr_t_max 20 (stale branch base: 9703afd).
- verdict: **Sent back** to refine σ and add T_max=20. Test σ∈{3,7} on full stack (Fourier+Huber+T_max=20+clip=0.25). Target: beat val=84.59 / test=73.89.

---

## 2026-05-16 01:33 — PR #3192 (charliepai2i48h5-edward): EMA checkpoint averaging — SENT BACK (add T_max=20)

- branch: `charliepai2i48h5-edward/ema-validation-checkpoint`
- hypothesis: EMA smoothing (decay=0.999 / 0.9995) reduces noise around converged weights, improving OOD generalization

- arms:

  | arm | ema_decay | val_avg/mae_surf_p | test_avg/mae_surf_p | raw_val at best epoch | EMA gain |
  |---|---|---|---|---|---|
  | baseline (PR #3221) | — | 89.27 | 79.43 | — | — |
  | arm-1 | 0.999 | **84.61** | 76.19 | 98.74 | -14.13 pts (-14.3%) |
  | arm-2 | 0.9995 | 114.36 | 106.32 | 94.69 | -19.67 lagging |

- artifacts: `models/model-ema-0.999-fourier-20260515-233215/metrics.jsonl`, `models/model-ema-0.9995-fourier-20260516-000735/metrics.jsonl`
- per-split val (arm-1 EMA vs raw at epoch 13): single=99.14 vs 120.31 (-17.6%), rc=95.00 vs 113.76 (-16.5%), cruise=63.25 vs 72.54 (-12.8%), re_rand=81.05 vs 88.37 (-8.3%)
- per-split test (arm-1 ckpt): single=89.94, rc=84.69, cruise=54.53, re_rand=75.61
- key finding: EMA decay=0.999 achieved near-tie with current baseline (val=84.61 vs 84.59) WITHOUT T_max=20. Stack missing --lr_t_max 20 (rebased on 9703afd, pre-#3333). EMA's contribution confirmed large and consistent across all splits. arm-2 failed: averaging window (~2000 steps) exceeds total run time (~4900 steps), EMA still dominated by high-loss early epochs.
- verdict: **Sent back** to add --lr_t_max 20 on arm-1 (EMA 0.999). Test clearly shows EMA compounds with T_max=20 is an untested combination. Expected to beat val=84.59 / test=73.89 (current #3333 baseline).

---

## Wave-6 new assignments (2026-05-16 01:33)

- PR #3593 — charliepai2i48h5-alphonse: LayerScale γ-init sweep {0.01, 0.1} on full stack (lightweight residual-branch regularization, zero convergence penalty)

---

## 2026-05-16 03:21 — PR #3529 (charliepai2i48h5-frieren): Grad-clip sweep max_norm∈{0.5,1.0} — **MERGED** (new best)

- branch: `frieren/grad-clip-sweep`
- hypothesis: clip=0.25 over-tight (clip_frac=1.000 every step = gradient-direction-only training); looser threshold might allow gradient magnitude information

- arms:

  | arm | clip | val_avg/mae_surf_p | test_avg/mae_surf_p | clip_frac at ep14 | vs baseline |
  |---|---|---|---|---|---|
  | baseline (#3333) | 0.25 | 84.59 | 73.89 | 1.000 | — |
  | arm-1 | 0.5 | 86.07 | 76.59 | 1.000 | +1.75% (worse) |
  | **arm-2** | **1.0** | **84.01** | **72.95** | **0.984** | **-0.69%** |

- artifacts: `models/model-fourier-tmax20-clip05-20260516-002450/metrics.jsonl`, `models/model-fourier-tmax20-clip10-20260516-013254/metrics.jsonl`
- per-split test (clip=1.0): single=82.86 (-4.6%✓), rc=84.34 (-2.2%✓), cruise=53.59 (+4.1%), re_rand=71.01 (0%)
- clip_frac trajectory (clip=1.0): 1.000 through epoch 9, drops to 0.997 at ep10, 0.984 at ep14 — first clip threshold where clip stops saturating in budget
- gnorm_mean at ep14: ~5.44; gnorm_max at ep14: 15.82; clip=0.5 is still 11× below mean — fully saturated
- verdict: **MERGED**. clip=1.0 beats on 3/4 splits. clip=0.5 is worst-of-both-worlds (still saturated, different trajectory). Frieren reassigned to LR warmup sweep (#3648).

---

## 2026-05-16 03:25 — PR #3438 (charliepai2i48h5-nezuko): Fourier n_freqs=14 on full stack (deconfounded) — **MERGED** (new best)

- branch: `nezuko/fourier-nfreqs-sweep`
- hypothesis: n_freqs=14 clean isolated test on full stack (prior run confounded by simultaneous clip addition)

- arms:

  | arm | n_freqs | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | vs baseline |
  |---|---|---|---|---|---|
  | baseline (#3333) | 10 | 84.59 | 73.89 | 14 | — |
  | arm-2 | 12 | 82.52 | 74.36 | 14 | -2.4%/-0.7% mixed |
  | **arm-1** | **14** | **81.08** | **71.52** | **14** | **-4.2% / -3.2%** |

- artifacts: `models/model-fourier-n14-fullstack-20260516-002646/metrics.jsonl`, `models/model-fourier-n12-fullstack-20260516-012442/metrics.jsonl`
- per-split test (n=14): single=81.31 (-6.4%✓), rc=84.95 (-1.5%✓), cruise=49.89 (-3.1%✓), re_rand=69.92 (-1.5%✓)
- all 4 test splits improve with n=14; n=12 is mixed (wins rc, loses cruise/single)
- clip_frac=1.000 throughout for both arms — clip=0.25 still over-tight at n=14 same as n=10
- model still improving at ep14 (timeout-bound) → spectrum unsaturated, scaling should continue
- verdict: **MERGED** as new round-5 best. val=81.08/test=71.52. Note: stack uses clip=0.25, not clip=1.0 (parallel wins, not yet combined). Nezuko reassigned to combine n=14+clip=1.0 AND push to n=18+clip=1.0 (#3650).

---

## Wave-6 additional assignments (2026-05-16 03:25)

- PR #3648 — charliepai2i48h5-frieren: LR linear warmup sweep {1, 2 epochs} on full stack + clip=1.0 (address cold-start penalty, root cause of early clip saturation)
- PR #3650 — charliepai2i48h5-nezuko: Compose n_freqs=14+clip=1.0 AND push to n_freqs=18+clip=1.0 (combine two parallel wins, continue Fourier scaling)

---

## 2026-05-16 04:34 — PR #3227 (charliepai2i48h5-thorfinn): Surf-weight curriculum — CLOSED (stuck rebase)

- branch: `charliepai2i48h5-thorfinn/surf-weight-anneal-1-20`
- status: **CLOSED** — no new commits since 2026-05-15 15:25 UTC (13h stale). Two rebase requests went unacted on due to repeated rate-limit cycles and pod restart. Hypothesis was sound (anneal val=130.40 beat constant-20 val=138.77 in first run), but the advisor branch has moved 4+ merges ahead since then. GPU time better spent on peak LR sweep.

---

## 2026-05-16 04:35 — Wave-7 assignment: thorfinn #3682

- PR #3682 — charliepai2i48h5-thorfinn: Peak LR sweep lr∈{7e-4, 1e-3} on n_freqs=14+clip=1.0 stack
- Rationale: default lr=5e-4 was set pre-Fourier, pre-clip. With clip=1.0 no longer saturating (clip_frac=0.984 at ep14, vs 1.000 for clip=0.25), effective step size is now LR-bound for the first time. Fourier features smooth the input landscape → can tolerate higher LR. All runs are timeout-bound → faster convergence per epoch = lower final val. arm-1=7e-4, arm-2=1e-3.

---

## 2026-05-16 06:22 — PR #3593 (charliepai2i48h5-alphonse): LayerScale γ-init sweep — MERGED

- branch: `charliepai2i48h5-alphonse/layerscale-gamma-init`
- hypothesis: per-channel learnable residual gain (CaiT LayerScale) improves OOD generalization via selective channel attenuation
- arms:

  | arm | γ-init | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
  |---|---|---|---|---|
  | arm-2 | 0.01 | **72.77** | **65.12** | 13 (timeout) |
  | arm-1 (run 1) | 0.1 | 74.74 | 67.24 | 13 (timeout) |
  | arm-1 (run 2) | 0.1 | 81.64 | 72.46 | 13 (timeout) |

- artifacts: `models/model-layerscale-0.01-fullstack-20260516-042447/metrics.jsonl` (winner), `models/model-layerscale-0.1-fullstack-20260516-023526/metrics.jsonl`, `models/model-layerscale-0.1-fullstack-20260516-033703/metrics.jsonl`
- stack: n_freqs=10 + Huber-0.3 + T_max=20 + clip=0.25 + LayerScale γ-init
- per-split test surf_p (γ=0.01): single=78.83 (-3.0%), rc=75.82 (-10.7%), cruise=43.86 (-12.1%), re_rand=61.97 (-11.4%) — EVERY split improves
- γ mechanism: `gamma_attn` mean ~0.01 (near init, ~0.085 max per channel — sparse selectivity); `gamma_mlp` grew 3× (mean ~0.03-0.04, max ~0.12 — MLP residual unlocked). Per-channel gate: most channels silent, a few 8× amplified.
- γ=0.1 high variance (run-to-run spread 74.74 vs 81.64); γ=0.01 cleaner dynamics
- clip_frac=1.000 throughout all runs — wins purely through model-side mechanism, not gradient regularization
- verdict: **MERGED** as new round-5 best. val=72.77/test=65.12. -43.5% cumulative from start. Alphonse assigned LayerScale+n_freqs=14 compound (PR #3730).

---

## 2026-05-16 06:22 — PR #3650 (charliepai2i48h5-nezuko): Compose merged wins — CLOSED

- branch: `charliepai2i48h5-nezuko/fourier-n14-n18-clip10`
- hypothesis: clip=1.0 and n_freqs=14 were parallel wins (vs clip=0.25 base) — combining them should compound
- arms:

  | arm | n_freqs | clip | val | test | vs baseline |
  |---|---|---|---|---|---|
  | arm-1 | 14 | 1.0 | 81.20 | 72.79 | +0.15% val / +1.78% test ✗ |
  | arm-2 | 18 | 1.0 | 84.71 | 76.96 | +4.48% val / +7.61% test ✗ |

- key insight: **clip=0.25 acts as regularization at n_freqs≥14**. With clip=1.0, the clip becomes adaptive at epoch 5 (vs epoch 10 for n=10), which removes implicit regularization. Single_in_dist regresses +4.15%, cruise +5.09% with arm-1. Arm-2 cruise blows up +16.5%. clip=1.0's original win (PR #3529) was likely split-noise at n=10.
- clip_frac at arm-1 ep14: 0.979 vs 1.000 for clip=0.25; clip became adaptive 5 epochs earlier than n=10 run
- verdict: **CLOSED**. Compound hypothesis fails. New actionable finding: always use clip=0.25 at n_freqs≥14. Nezuko reassigned to n_freqs=18+clip=0.25 and n_freqs=20+clip=0.25 (PR #3732).

---

## 2026-05-16 06:24 — Wave-7 assignments

- PR #3730 — charliepai2i48h5-alphonse: LayerScale γ=0.01 + n_freqs=14 (compound the two biggest wins — expected val ~65-70 if composition holds)
- PR #3732 — charliepai2i48h5-nezuko: n_freqs={18,20} + clip=0.25 (missing test from PR #3650; n=18+clip=1.0 failed but n=18+clip=0.25 untested)

---

## 2026-05-16 06:30 — PR #3648 (charliepai2i48h5-frieren): LR warmup 1-ep and 2-ep — CLOSED

- branch: `charliepai2i48h5-frieren/lr-warmup-sweep`
- hypothesis: linear LR warmup addresses cold-start penalty (epoch-1 val ~210, gnorm_max=48)
- arms: warmup=1ep (val=85.69/test=77.28), warmup=2ep (val=87.99/test=79.18) — BOTH worse than clip=1.0 baseline (84.01/72.95)
- mechanism: start_factor=1e-6 → epoch-1 LR=5e-10 → val=419 (essentially no learning) → 1 wasted epoch in a budget where every epoch matters
- clip_frac stayed at 1.000 throughout; clip=0.997 not until epoch 4+
- verdict: **CLOSED** — warmup harmful in 14-epoch timeout-bound regime; wasted first epoch costs more than smoother init gains. Closed with note that the cold-start issue is real but in-budget cost makes this approach invalid.

---

## 2026-05-16 06:30 — PR #3192 (charliepai2i48h5-edward): EMA checkpoint averaging — SENT BACK

- branch: `charliepai2i48h5-edward/ema-validation-checkpoint`
- hypothesis: EMA checkpoint averaging reduces val_avg by smoothing noisy batch-size-4 updates
- arms run: EMA 0.999 no T_max (val=84.61), EMA 0.999 full stack (val=84.56), **EMA 0.998 full stack (val=80.14/test=70.42)** — best arm beats old n_freqs=14 baseline (81.08) by -1.16%
- key insight: with T_max=20 (stable late training), decay=0.998 (~500 steps, ~1.3 epoch window) is sweet spot — 0.999 (~1000 steps) averages too long, dragging toward older worse weights
- branch has merge conflict against advisor branch (LayerScale now merged)
- verdict: **SENT BACK** — positive result but falls short of new LayerScale baseline (72.77). Rebase onto current best + test EMA 0.998 on LayerScale stack. Expected val ~67 if EMA -6.84% applies on top of 72.77.

---

## 2026-05-16 06:35 — Wave-7 additional assignments

- PR #3740 — charliepai2i48h5-frieren: Asymmetric LayerScale γ-init (γ_attn=0.001, γ_mlp=0.01/0.03); motivated by PR #3593 mechanism: γ-attn stays near 0.01, γ-mlp grows 3× — separate inits target each branch's natural trajectory

---

## 2026-05-16 05:25 — PR #3439 (charliepai2i48h5-fern): Gaussian RFF σ-sweep rerun with T_max=20 — CLOSED

- branch: `fern/gaussian-random-fourier-features`
- hypothesis: Tancik 2020-style Gaussian RFF with tuned σ should beat log-spaced log-frequency basis on smoother flow regions
- arms (this rerun):

  | arm | σ | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
  |---|---|---|---|---|
  | arm-1 | 3.0 | 85.25 | 74.03 | 14 (timeout-bound) |
  | arm-2 | 7.0 | 90.76 | 83.42 | 14 (timeout-bound) |

- combined with prior submission: σ ∈ {1, 3, 5, 7} all tested. σ=3 is the best RFF (val=85.25) but still 5% WORSE than current log-spaced baseline (val=81.08, PR #3438 n_freqs=14).
- per-split test surf_p (σ=3): single=89.54 (+3.1% vs base #3333), rc=86.37 (+0.2%), cruise=49.78 (-3.3% ✓), re_rand=70.41 (-0.8% ✓) — wins 2/4 splits
- mechanism: σ=3 Gaussian centered on moderate frequencies matches smoother regimes (cruise, re_rand) but loses on sharp-feature splits (single_in_dist has highest pressure peaks, log-spaced broad-spectrum coverage wins there)
- clip_frac=1.000 throughout both arms — 0.25 clip is rate-limiter
- artifacts: `models/model-rff-sigma3-n10-tmax20-clip025-20260516-022348/metrics.jsonl`, `models/model-rff-sigma7-n10-tmax20-clip025-20260516-033535/metrics.jsonl`
- verdict: **CLOSED**. Full σ sweep (1, 3, 5, 7) is a thorough negative result — RFF doesn't beat log-spaced at any σ tested. The hypothesis is dead in this regime. Reassigning fern to AdamW β2 sweep (PR #3708).

---

## 2026-05-16 05:25 — Wave-7 assignment: fern #3708

- PR #3708 — charliepai2i48h5-fern: AdamW β2 sweep {0.99, 0.95} vs default 0.999 on n_freqs=14+clip=1.0 stack
- Rationale: β2=0.999 has effective half-life ~700 steps; with heavy-tailed gradients (confirmed by Huber+clip needs and gnorm_max=48 at epoch 1), second-moment estimate is contaminated for too long. Lower β2 (0.99: half-life ~70; 0.95: ~14) accelerates the variance estimate update. Last untouched optimizer hyperparameter.

---

## 2026-05-16 07:30 — PR #3708 (charliepai2i48h5-fern): AdamW β2 sweep — CLOSED

- branch: `charliepai2i48h5-fern/adam-beta2-sweep`
- hypothesis: lowering β2 below 0.999 reduces contamination of second-moment estimator from heavy-tail gradients; hypothesis: faster variance decay → more adaptive per-param LR
- arms:

  | arm | β2 | val_avg/mae_surf_p | test_avg/mae_surf_p | vs current best |
  |---|---|---|---|---|
  | arm-1 | 0.99 | 84.73 | 74.26 | +16.4% val / +14.1% test ✗ |
  | arm-2 | 0.95 | 84.95 | 75.92 | +16.7% val / +16.6% test ✗ |
  | baseline (PR #3593) | 0.999 | **72.77** | **65.12** | — |

- stack: n_freqs=14 + Huber-0.3 + T_max=20 + clip=1.0 (note: clip=1.0, not clip=0.25 from current best)
- per-split test surf_p arm-1 (β2=0.99): single=88.96, rc=83.69, cruise=50.49, re_rand=73.89
- artifacts: `models/model-fourier-n14-clip10-beta2-099-20260516-053002/metrics.jsonl`, `models/model-fourier-n14-clip10-beta2-095-20260516-062337/metrics.jsonl`
- epoch-1 val: arm-1=202.50, arm-2=246.17 — lower β2 is WORSE at epoch 1 (opposite to hypothesis)
- clip_frac stable ~1.000 throughout; grad_norm_mean 13→5; clip already bounds the heavy-tail gradient signal so v_t isn't actually contaminated
- key insight: gradient clipping detoxifies the tail before v_t accumulates it; lower β2 introduces denominator instability instead. Monotone ordering 0.95<0.99<0.999 → default β2=0.999 is optimal.
- verdict: **CLOSED** — hypothesis falsified. No follow-up on β2. New assignment: AdamW eps sweep on LayerScale stack (PR #3782).

---

## 2026-05-16 07:30 — PR #3682 (charliepai2i48h5-thorfinn): Peak LR sweep lr∈{7e-4,1e-3} — CLOSED

- branch: `charliepai2i48h5-thorfinn/peak-lr-sweep`
- hypothesis: higher peak LR accelerates convergence under clip=1.0; lr=1e-3 was untested on n_freqs=14 stack
- arms:

  | arm | lr | epochs | val_avg/mae_surf_p | test_avg/mae_surf_p | vs current best |
  |---|---|---|---|---|---|
  | arm-1 | 7e-4 | 12 | 84.13 | 75.01 | +15.6% / +15.2% ✗ |
  | arm-2 | 1e-3 | 14 | 81.43 | 70.60 | +11.9% / +8.4% ✗ |
  | baseline (PR #3593) | 5e-4 | 13 | **72.77** | **65.12** | — |

- stack: n_freqs=14 + Huber-0.3 + T_max=20 + clip=1.0 (NOT the current best clip=0.25 LayerScale stack)
- per-split test surf_p arm-2 (lr=1e-3): single=82.80, rc=83.87, cruise=47.39, re_rand=68.33
- arm-1 had anomalous epoch-4 slowdown (286s vs 130s typical) — cost 2 epochs; at matched epochs arm-1 actually descends faster per-epoch than arm-2
- arm-2 (lr=1e-3) beats arm-2's cited baseline (n_freqs=14+clip=1.0 val=81.08) on test by -1.3% — promising direction but tested on wrong stack
- key insight: lr=1e-3 is not divergent under clip; clip_frac drops to 0.979 and grad_norm_mean trends to 3.57 (vs 4.8 for arm-1) — optimizer reaching smoother landscape faster under higher LR
- verdict: **CLOSED** — both arms far from current best (72.77); scope was against old n14+clip=1.0 baseline. Positive signal for lr=1e-3 direction warrants retesting on LayerScale stack. Thorfinn reassigned to lr sweep on LayerScale stack (PR #3784).

---

## 2026-05-16 07:30 — PR #3527 (charliepai2i48h5-tanjiro): BF16 mixed precision — SENT BACK

- branch: `tanjiro/mixed-precision-bf16`
- hypothesis: BF16 autocast reduces memory + increases speed → more epochs in 30-min budget → direct metric improvement
- arms:

  | arm | config | epochs | sec/ep | val | test | vs current best |
  |---|---|---|---|---|---|---|
  | arm-1 (n128) | BF16 + full stack | 18 | 102.1s | **72.75** | **65.05** | -0.03% / -0.11% ≈ TIE |
  | arm-2 (n160) | BF16 + n_hidden=160 | 16 | 115.9s | 75.83 | 69.01 | under-converged |
  | baseline (PR #3593) | FP32 + LayerScale | 13 | 132.6s | **72.77** | **65.12** | — |

- arm-1 stack: n_freqs=10 + Huber-0.3 + T_max=20 + clip=0.25 + **BF16** (NO LayerScale — pre-LayerScale scope)
- BF16 speedup: 1.30× (102.1 vs 132.6 s/epoch) + 21% less peak memory (33.4 vs 42.4 GB)
- all 4 test splits uniform 10–13% improvement vs FP32 no-LayerScale baseline (PR #3333)
- arm-1 val=72.75 ≈ current best (LayerScale) at 72.77 — two independent mechanisms reach same level, suggests composition may yield significant gain
- implementation: forward pass in `torch.amp.autocast(dtype=torch.bfloat16)`; eval in FP32; no GradScaler needed (BF16 has same exponent range as FP32)
- merge conflict with LayerScale (train.py modified by both)
- verdict: **SENT BACK** — result is excellent (virtual tie vs LayerScale) but PR needs rebase onto current best. Next test: BF16 + LayerScale composition — expected val ~65-70 if additive, potentially first result below 65. Expected ~18 epochs in budget.

---

## 2026-05-16 07:40 — Wave-8 assignments

- PR #3782 — charliepai2i48h5-fern: AdamW eps sweep {1e-6, 1e-7} on LayerScale stack (default eps=1e-8; theory: small v_t channels with tiny eps get amplified updates, raising eps damps them → more uniform per-param effective LR; especially relevant for near-zero LayerScale γ-attn channels)
- PR #3784 — charliepai2i48h5-thorfinn: Peak LR sweep {7e-4, 1e-3} on LayerScale stack (redo on current best stack; lr=1e-3 showed healthy gradient stats in PR #3682 but wrong baseline; LayerScale's per-channel gating may tolerate higher base LR)

---

## 2026-05-16 08:30 — PR #3732 (charliepai2i48h5-nezuko): n_freqs={18,20} + clip=0.25 — CLOSED

- branch: `charliepai2i48h5-nezuko/fourier-n18-n20-clip025`
- hypothesis: n_freqs=18/20 + clip=0.25 (correct clip) could extend Fourier scaling beyond n=14
- arms:

  | arm | n_freqs | clip | val_avg/mae_surf_p | test_avg/mae_surf_p | vs n=14 baseline | vs current best |
  |---|---|---|---|---|---|---|
  | arm-1 | 18 | 0.25 | 83.06 | 72.25 | +2.4% / +1.0% ✗ | +14.1% / +10.9% ✗ |
  | arm-2 | 20 | 0.25 | 81.24 | 71.97 | +0.2% / +0.6% ✗ | +11.6% / +10.5% ✗ |
  | baseline (PR #3438) | 14 | 0.25 | 81.08 | 71.52 | — | — |
  | current best (PR #3593) | 10 | 0.25 | **72.77** | **65.12** | — | — |

- artifacts: `models/model-fourier-n18-clip025-fullstack-20260516-062742/metrics.jsonl`, `models/model-fourier-n20-clip025-fullstack-20260516-073106/metrics.jsonl`
- clip_frac=1.000 throughout both arms (same as n=14 + clip=0.25)
- key sub-finding: clip=0.25 does help at n=18 vs clip=1.0 (PR #3650 arm-1: val=84.71 vs this arm-1: val=83.06, -2%). Tight-clip hypothesis confirmed but not sufficient.
- pattern: n=10→84.59; n=12≈tied; n=14→81.08 (best); n=18→83.06; n=20→81.24 — non-monotone above n=14, confirming saturation
- arm-2 (n=20) shows best test_rc (83.17 vs 84.95, -2.1%) suggesting finer spectral resolution helps geometry OOD, but cruise/re_rand worse; net wash-to-worse
- both timeout-bound, still improving at ep14
- verdict: **CLOSED** — Fourier spectrum saturated at n=14. Stop scaling n_freqs. Nezuko reassigned to Lookahead optimizer wrapper (PR #3823).

---

## 2026-05-16 08:35 — Wave-8 additions

- PR #3823 — charliepai2i48h5-nezuko: Lookahead optimizer wrapper {k=5 α=0.5, k=10 α=0.5} on LayerScale stack (Zhang et al. 2019; slow anchor weights with periodic pull-back — variance reduction in heavy-tail/high clip_frac regime; orthogonal to β2/eps changes; near-zero compute overhead)

---

## 2026-05-16 14:40 — PR #3527 (charliepai2i48h5-tanjiro): Mixed precision BF16 training — MERGED ★ NEW BEST

- branch: `tanjiro/mixed-precision-bf16`
- hypothesis: BF16 autocast forward buys extra epochs in 30-min budget via 1.30× speedup + −21% memory
- results:

| Arm | Config | epochs | val | test | vs baseline |
|---|---|---|---|---|---|
| arm-1 (BF16+LS+n10) | BF16+LS+n_freqs=10 | 17 | **67.19** | **58.05** | **-5.6%/-7.4%** |
| arm-2 (BF16+LS+n14) | BF16+LS+n_freqs=14 | 17 | 67.00 | 59.31 | marginally worse |
| quad (BF16+LS+n14+EMA) | BF16+LS+n14+EMA 0.998 | 15 | 68.50 | 60.15 | beats old but not arm-1 |

- metric artifacts: `models/model-bf16-layerscale-fullstack-20260516-082748/metrics.jsonl`
- **NEW BEST: arm-1 val=67.19/test=58.05 — uniform improvement across all 4 test splits**
- Key findings:
  - BF16 1.30× speedup → 17 epochs vs 12 (FP32)
  - At 17-epoch horizon, n10 BEATS n14: aliasing advantage dominates at convergence
  - EMA+n14 quad at 68.50/60.15: better than old baseline but worse than n10 alone
  - EMA overhead ~9%/epoch (121s vs 111s) costs ~2 epochs; at 17-epoch horizon the tradeoff is net negative
  - BF16: no GradScaler needed; only forward in BF16, Huber + optimizer in FP32
  - LayerScale γ dynamics unaffected; per-split improvement uniform (cruise -14.5%, re_rand -9.4%)

---

## 2026-05-16 14:45 — PR #3784 (charliepai2i48h5-thorfinn): Peak LR sweep {7e-4, 1e-3} — CLOSED

- branch: `charliepai2i48h5-thorfinn/lr-sweep-on-layerscale`
- hypothesis: higher LR might help LayerScale stack escape local minima
- results: arm-1 (7e-4): val=75.37/test=65.39; arm-2 (1e-3): val=72.06/test=62.18
- both worse than old baseline 71.20; both far from new best 67.19
- Key findings: clip_frac=1.0 throughout — LR can't manifest because clip=0.25 caps every step. Ran on FP32 triple (now superseded). arm-2 descends faster per epoch but can't beat baseline before timeout.

---

## 2026-05-16 14:45 — PR #3883 (charliepai2i48h5-fern): T_max schedule sweep {12, 25} — CLOSED

- branch: `charliepai2i48h5-fern/tmax-sweep`
- hypothesis: T_max=12 (aligned to budget) or T_max=25 (slower decay) might improve convergence
- results: arm-1 (T_max=12): val=79.26/test=70.76 (+11.3%); arm-2 (T_max=25): val=75.94/test=66.29 (+6.7%)
- Key findings: T_max=12 worst — pre-converges to suboptimal basin (LR hits ~0 at ep12, then flat). T_max=20 confirmed optimal. arm-2 still descending at timeout (Δ4.6/epoch). T_max=20 is ~85% cycle at 17 epochs (BF16) — still well-calibrated; no re-sweep needed on new stack.

---

## 2026-05-16 14:46 — PR #3909 (charliepai2i48h5-frieren): Learnable Fourier frequencies — CLOSED

- branch: `charliepai2i48h5-frieren/learnable-fourier-freqs`
- hypothesis: making Fourier freqs nn.Parameter lets gradient descent find optimal freq basis
- results: arm-1 default LR: val=73.20/test=64.21; arm-2 lr10x: val=75.62/test=65.69; replication: val=74.14/test=65.11
- Key findings: frequencies barely migrate — only k=0,1,2 shift >2%; k≥3 essentially frozen by clip=0.25. nn.Parameter overhead costs ~2 epochs (12 vs 14 effective). Log-spaced init is near a fixed point for this architecture. Dead end.

---

## 2026-05-16 14:46 — PR #3941 (charliepai2i48h5-nezuko): AdamW WD sweep {3e-5, 3e-4} — CLOSED

- branch: `charliepai2i48h5-nezuko/adamw-wd-sweep`
- hypothesis: WD=1e-4 may not be optimal on LayerScale+EMA stack
- results: arm-1 (3e-5): val=73.22/test=64.25; arm-2 (3e-4): val=75.66/test=66.35
- Key findings: WD=1e-4 is optimal. Lighter WD closer to baseline than heavier (asymmetric). Model is regularization-constrained — clip+LS already regularize; increasing WD forces larger γ_mlp as compensation (counterintuitive). WD range {3e-5 to 3e-4} falsified.


---

## 2026-05-16 15:30 — PR #4007 (charliepai2i48h5-frieren): Width scaling n_hidden=144 on BF16+LS+n10 — CLOSED

- branch: `charliepai2i48h5-frieren/bf16-width-scaling-n144`
- hypothesis: BF16's −21% memory freed headroom for wider model; LayerScale gates extra channels selectively
- results: val=68.71/test=60.83 vs baseline 67.19/58.05 (+2.3% val, +4.8% test worse)
- Key findings:
  - n=144 ran 15 epochs (vs 17 for n=128) — 14 s/epoch slower (125 vs 111)
  - best_epoch=15 (LAST), val still descending Δ(ep14→15)=−5.9 — needed ~17+ epochs to match baseline
  - Per-split: single_in_dist regresses most (+11%), cruise unchanged, others worse
  - Conclusion: width scaling bottlenecked by epoch count, not capacity. n_hidden=120 (narrower) might fit budget better — assigned as PR #4014
  - Peak GPU memory 39.87 GB (vs 33 GB baseline)
