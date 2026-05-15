# SENPAI Research Results

_Track: `icml-appendix-charlie-pai2i-48h-r5` (round 5)._
_New entries appended as each PR is reviewed._

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

## Wave-2 new assignments (2026-05-15)

- PR #3333 — charliepai2i48h5-frieren: LR schedule alignment (T_max=14 vs T_max=20 with huber_delta=0.3)
- PR #3334 — charliepai2i48h5-tanjiro: Wider Transolver n_hidden=192 (arms: n_head=4 and n_head=6, both huber_delta=0.3)

## Currently rebasing+combining (sent back)

- PR #3199 (fern): dualhead + Huber-0.3
- PR #3182 (askeladd): grad-clip-0.5 + Huber-0.3 + try max_norm=0.25
- PR #3227 (thorfinn): surf-anneal + Huber-0.3 + try terminal weight=30

## Still WIP (original assignments)

- PR #3178 (alphonse): per-sample pressure-scale normalization
- PR #3192 (edward): EMA checkpoint averaging
- PR #3221 (nezuko): Fourier positional features
