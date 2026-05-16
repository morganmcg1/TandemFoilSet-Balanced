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
