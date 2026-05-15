# SENPAI Research Results — TandemFoilSet — `icml-appendix-willow-pai2i-24h-r2`

This file logs each reviewed PR. Newest entries at the top.

## Format

```
## <YYYY-MM-DD HH:MM> — PR #<number>: <title>
- student: <name>
- branch: <branch-name>
- hypothesis: <one-line statement>
- results table (val_avg/mae_surf_p, test_avg/mae_surf_p, per-split, wandb run id)
- analysis & conclusions
- next steps
```

## Entries

## 2026-05-15 22:41 — PR #3441: slice_num=80 without gradient checkpointing (frieren) — CLOSED
- student: willowpai2i24h2-frieren
- branch: `willowpai2i24h2-frieren/slice-num-80-no-checkpoint`
- hypothesis: removing gradient checkpointing at slice_num=80 trades VRAM headroom for convergence speed vs PR #3353
- W&B run: `o1wv46oq`

| Metric | This run | Baseline (PR #3352) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 130.17 | 116.34 | **+11.9% worse** |
| `test_avg/mae_surf_p` | 120.26 | 107.33 | **+12.0% worse** |
| Best epoch | 11 | 12 | — |
| Peak VRAM | 90.6 GB (94.8%) | ~33 GB | 2.7× more |
| Avg epoch time | 142.6s | ~128s | +11% |

Per-split val: single=150.35, camber_rc=142.71, camber_cruise=100.86, re_rand=126.78 — all regress.

- analysis: The key failure is the memory model. PR #3353 (slice_num=96+ckpt) used only 16 GB with checkpointing, but that was HIDING ~80 GB of activation memory. Removing checkpointing at slice_num=80 demands the full activation footprint (~90.6 GB), leaving ~5 GB headroom. Epoch time is +11% worse (not better), and convergence is noisy (±20–30 val swing between consecutive epochs). The slice_num axis is not the model's bottleneck at this dataset scale.
- decision: **closed**. Clear regression on all splits. The student's own VRAM analysis was correct — the slice attention cost grows quadratically and dominates without checkpointing.
- next steps: Abandon the slice_num scaling direction for now. frieren reassigned to new hypothesis.

## 2026-05-15 22:30 — PR #3356: Divergence-free velocity auxiliary loss (thorfinn) — beats OLD baseline, needs rebase on SmoothL1
- student: willowpai2i24h2-thorfinn
- branch: `willowpai2i24h2-thorfinn/divergence-free-aux-loss`
- hypothesis: adding a normalized-space divergence penalty on predicted velocity fields biases the model toward physically-realizable solutions, improving OOD generalization
- W&B runs: `ilylzwo5` (div_weight=0.01, winner), `2w40di7q` (div_weight=0.001, fails), `4ws0rum6` (div_weight=0.01 replication)

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | vs OLD baseline (121.50/112.49) | vs NEW baseline (116.34/107.33) |
|---|---|---|---|---|
| div_weight=0.01 (ilylzwo5) | **113.41** | **102.86** | −6.7%/−8.6% | −2.5%/−4.2% |
| div_weight=0.01 (4ws0rum6 replication) | 117.20 | 108.05 | −3.5%/−4.0% | +0.7%/+0.7% |
| div_weight=0.001 (2w40di7q) | 133.48 | 122.17 | +9.9%/+8.6% | +14.7%/+13.8% |

Per-split val (div_weight=0.01, best run): single=142.61 (+2%), camber_rc=121.19 (−12.6%), camber_cruise=85.88 (−8.2%), re_rand=103.97 (−8.7%)

- analysis: The hypothesis is well-supported in direction: OOD splits (camber_rc, camber_cruise, re_rand) improve dramatically at div_weight=0.01 while in-dist (single) is slightly worse — the classic signature of useful regularization. **Key implementation finding**: naive denormalized divergence with raw x/z coordinates explodes to 1e5–1e7 on boundary-layer mesh pairs (|dx|~1e-5); student switched to normalized-space divergence with eps floor on |dx|,|dz|, giving div_loss in the 50–250 range and div_weight*div_loss balanced against vol_loss. **Caveat**: two runs at div_weight=0.01 disagree — best run beats new baseline (−2.5%/−4.2%), replication at +0.7%/+0.7% (barely misses). This variance is concerning for a definitive merge decision. The run was on the current learnable-Fourier baseline, BUT now that tanjiro's SmoothL1 is about to become the new baseline at val=90.60/test=83.00, even the best div-free result (113.41/102.86) doesn't beat the incoming baseline.
- decision: **sent back for rebase on SmoothL1 baseline** (pending tanjiro merge). The physics-informed direction is promising; must verify compound effect with SmoothL1.
- next steps: rebase onto SmoothL1 baseline. Single arm div_weight=0.01 (best arm from this round). A 10% sweep (0.005, 0.01, 0.02) would also clarify whether 0.01 is near-optimal.

## 2026-05-15 22:27 — PR #3215: SmoothL1 β=0.05 on learnable Fourier baseline (tanjiro) — PENDING MERGE (rate-limited)
- student: willowpai2i24h2-tanjiro
- branch: `willowpai2i24h2-tanjiro/smooth-l1-loss`
- hypothesis: replacing MSE with Huber/SmoothL1 (β=0.05) in normalized space caps the gradient contribution of extreme y-values, reducing the distortion from large-Re samples
- W&B run: `iofja54s`

| Metric | Baseline (PR #3352, learnable Fourier) | SmoothL1 β=0.05 (rebased) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 116.3411 | **90.6039** | **−22.13%** |
| `test_avg/mae_surf_p` | 107.3254 | **83.0029** | **−22.66%** |
| Best epoch | 12 | 14 | +2 epochs |
| Peak VRAM | ~33 GB | ~42.5 GB | — |

Per-split val: single=112.03 (−22.8%), camber_rc=104.42 (−17.3%), camber_cruise=62.07 (−29.6%), re_rand=83.89 (−20.8%)
Per-split test: single=101.95 (−19.4%), camber_rc=97.84 (−17.2%), camber_cruise=55.10 (−28.1%), re_rand=77.11 (−28.6%)

- analysis: **Largest single-change improvement on this benchmark to date** — −22% on both headline metrics with all 4 splits improving 17-30%. Confirmed compound with learnable Fourier: magnitudes nearly identical to pre-rebase arms (val 90.24→90.60, test 82.21→83.00), confirming SmoothL1 and learnable Fourier are additive on this dataset. The improvement is largest on the highest-range splits (re_rand test −28.6%, camber_cruise −28%). Mechanism clear: MSE squares large normalized residuals, creating gradient domination from the most extreme Re/geometry samples; SmoothL1 with β=0.05 transitions to linear beyond |err|>0.05, effectively downweighting the outlier-sample gradient contribution. Best epoch was the LAST completed epoch (14/50, wall-clock limited) — val curve still declining, suggesting more headroom with extended training.
- decision: **PENDING MERGE** — blocked by GitHub API rate limit (5000/5000 exhausted; resets 23:19 UTC). PR is not-draft, status:review, MERGEABLE — ready as soon as rate limit allows.
- next steps: After merge, this becomes the new baseline (val=90.60, test=83.00). All in-flight PRs should be re-evaluated against this threshold. High-priority follow-ups: β sweep (0.02, 0.075, 0.10, L1 limit), per-channel β, SmoothL1+FiLM compound.

## 2026-05-15 20:35 — PR #3215: SmoothL1 loss β=0.05 and β=0.10 (tanjiro) — sent back for rebased verification
- student: willowpai2i24h2-tanjiro
- branch: `willowpai2i24h2-tanjiro/smoothl1-loss`
- hypothesis: Replace MSE with smooth-L1 (Huber) loss in normalized space; caps gradient contribution of high-Re extreme errors without zeroing them
- W&B runs: `638hd0v7` (β=0.05, won by val), `pbvt4zsz` (β=0.10, won by test)

| Metric | OLD baseline (#3200) | β=0.05 | β=0.10 | NEW baseline (#3352) | vs NEW |
|---|---|---|---|---|---|
| `val_avg/mae_surf_p` | 121.4956 | **90.2450** | 91.2928 | 116.3411 | **−22.4% (β=0.05)** |
| `test_avg/mae_surf_p` | 112.4884 | 82.2072 | **81.1592** | 107.3254 | **−24.4% (β=0.10)** |
| best val epoch | 14 | 13 | 14 | 12 | — |
| peak VRAM | ~42.5 GB | 42.5 GB | 42.5 GB | ~33 GB | — |

Per-split val (β=0.05): single=105.40 (−25%), camber_rc=98.02 (−29%), camber_cruise=70.25 (−25%), re_rand=87.31 (−23%)
Per-split test (β=0.10): single=96.44 (−21%), camber_rc=90.75 (−32%), camber_cruise=60.05 (−28%), re_rand=77.40 (−31%)

**Largest single-change improvement on this benchmark to date.** All 4 splits improve by 21-34% on test. The β=0.05 and β=0.10 arms agree closely (within 1-2% on each metric), so the improvement is robust, not noise. Mechanism is consistent with hypothesis: high-Re extreme y-values (normalized errors up to ~5) dominate MSE gradients quadratically; SmoothL1 caps this contribution linearly above β.

**Caveat:** ran on OLD baseline (fixed Fourier features). vs NEW baseline (learnable Fourier, val=116.34, test=107.33), the gain is still −22% val / −24% test — likely real and compounds with learnable Fourier, but must verify. Sent back for ONE rebased single-arm re-run (β=0.05) on the new baseline. If it confirms (anywhere near val<100, test<90), merge immediately as new baseline.

**Side observation:** student noted an earlier β=0.05 run on un-rebased code (no Fourier features at all) got val_avg=99.14. That single-change improvement (SmoothL1 alone over the pre-Fourier baseline) is also dramatic. Suggests MSE→SmoothL1 is the dominant lever in the dataset, with Fourier features adding ~10-20% on top.

## 2026-05-15 20:32 — PR #3353: slice_num=96 + gradient checkpointing (frieren) — CLOSED
- student: willowpai2i24h2-frieren
- branch: `willowpai2i24h2-frieren/slice-num-96-checkpoint`
- W&B run: `o70g1t9p` (full), `2d9oqekh` (smoke test, OK)

| Metric | This run | NEW baseline (#3352) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **130.7342** | 116.3411 | **+12.4% worse** |
| `test_avg/mae_surf_p` | **120.7062** | 107.3254 | **+12.5% worse** |
| Best val epoch | 10 / 50 | 12 | −4 |
| Avg epoch time | 192.5 s | ~128 s | **+50% slower** |
| Peak GPU memory | **16.0 GB** | ~33 GB | −52% |
| Wall-clock | 31.9 min | 30 min | similar |

Per-split test: single=152.57 (+25%), camber_rc=130.17 (**−10%** beats baseline!), camber_cruise=84.31 (+10%), re_rand=115.78 (+8%).

Analysis: Gradient checkpointing was over-aggressive — peak VRAM 16 GB out of 96 GB available, ~80 GB headroom unused. The 50% epoch-time penalty (from checkpoint recomputation) ate the 30-min budget, limiting to 10/50 epochs vs baseline's 12. Model was still descending hard at epoch 10 (174→131, ~24% drop) — under-convergence dominated the result. `geom_camber_rc` actually beating baseline by 3.2 test points partially supports the slice-bump hypothesis directionally, but cannot be claimed as a win in the equal-weight mean.

Decision: **CLOSED**. Wall-clock bottleneck, not memory. The student's own recommendation #1 is the right next step.

Reassignment: PR #3441 — slice_num=80 **without** gradient checkpointing. Uses memory headroom (predicted ~50-55 GB peak at slice_num=80, well under 96 GB), avoids the checkpoint recompute tax (predicted +10-20% epoch time vs +50%), aims for 12+ epochs in 30 min.

## 2026-05-15 20:25 — PR #3350: FiLM-style Re conditioning per Transolver block (alphonse) — sent back for rebase
- student: willowpai2i24h2-alphonse
- branch: `willowpai2i24h2-alphonse/film-re-conditioning`
- hypothesis: FiLM on every Transolver block, conditioned on per-sample log(Re), should improve val_re_rand by 5-12% and net val_avg
- W&B run: `5a3i5ctn` (v2 with bug-fix), `d5xeygsf` (v1 with buggy init + padding contamination, val=128.96)

| Metric | OLD baseline (#3200) | FiLM v2 | NEW baseline (#3352) | vs NEW |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | 121.4956 | **116.962** | 116.3411 | +0.5% (flat) |
| `test_avg/mae_surf_p` | 112.4884 | **104.636** | 107.3254 | −2.5% |
| best val epoch | 14 | 11 | 12 | — |
| peak VRAM | ~42.5 GB | 47.6 GB | ~33 GB | — |

Per-split test (best-val checkpoint): single=125.41, camber_rc=115.38, camber_cruise=73.86, re_rand=103.90 — all 4 splits improve vs NEW baseline.
Per-split val: single=147.81 (+1.9%), camber_rc=127.04 (+0.6%), camber_cruise=86.61 (−1.7%), re_rand=106.39 (+0.4%) — 3 of 4 val splits slightly worse vs NEW baseline.

Analysis: alphonse's run was on the OLD baseline (PR #3200, fixed Fourier) and finished ~1 min before PR #3352 merged. vs the OLD baseline, FiLM is a clear winner: val −3.7%, test −7.0%. But vs the NEW baseline (learnable Fourier), val is flat (within noise) and test is better (−2.5%, consistent across all 4 splits). The mechanism is sound — debug logs show clean per-sample log(Re) signal (after fixing zero-init preservation + padding contamination bugs the student found themselves). The Re-conditioning gain isn't only on re_rand split — geom_camber_rc and geom_camber_cruise also improve substantially on test, suggesting the model uses Re-implicit features for held-out cambers and FiLM frees up other capacity.

Decision: **sent back for rebase + re-run on learnable Fourier baseline**. The compound (FiLM + learnable Fourier) hasn't been tested. If it beats both val_avg < 116.34 and test_avg < 107.33, merge.

Sub-finding worth recording: alphonse caught two implementation bugs that future FiLM-like PRs should be aware of:
1. `Transolver.__init__` calls `self.apply(self._init_weights)` AFTER constructing all submodules, which overwrites the zero init in any sub-module that needs near-identity start. Fix: re-zero after `self.apply`.
2. Reading global per-sample features via `x[:, :, idx].mean(dim=1)` is **contaminated by padding** when `pad_collate` zero-pads in raw space and the model normalizes afterward. The padding rows in normalized space have value `-mean/std`, which is large in magnitude (~−19 for log Re) and dominates the mean. Fix: read row 0 directly, which is always a real node since pad_collate appends padding.

## 2026-05-15 19:28 — PR #3352: Learnable Fourier frequency bands (8 trainable freqs) — MERGED
- student: willowpai2i24h2-fern
- branch: `willowpai2i24h2-fern/learnable-fourier-freqs`
- hypothesis: Making the 8 Fourier frequency bands learnable nn.Parameters (initialized to octave-doubling baseline) allows gradient descent to discover which spatial scales matter most for TandemFoilSet, improving OOD geometry splits most.
- W&B run: `rumqs1au`

| Metric | Baseline (PR #3200) | PR #3352 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 121.4956 | **116.3411** | **−4.24%** |
| `test_avg/mae_surf_p` | 112.4884 | **107.3254** | **−4.59%** |
| best val epoch | 14 | 12 | — |
| peak VRAM | ~42.5 GB | ~33 GB | — |

Per-split surface-p (val | test):
- `single_in_dist`: 145.03 | 126.46 (+3.7% val regression)
- `geom_camber_rc`: 126.25 | 118.24 (−9.0% val / −11.3% test — largest OOD gain)
- `geom_camber_cruise`: 88.12 | 76.60 (−5.8% val / −7.8% test)
- `re_rand`: 105.96 | 108.00 (−7.0% val / −3.1% test)

Analysis: Frequencies barely moved from octave-doubling init (max drift 2.47% on freq_0: 1.000→1.025; all others <1%). The improvement is not from discovering a fundamentally different frequency basis — it comes from the extra gradient signal through the frequency parameters during optimization (and possibly a small regularization effect from weight decay on positive parameters). The octave-doubling init is near-optimal for this dataset. Biggest gains on OOD geometry splits as predicted — in-dist regressed slightly (+3.7%). Net improvement is clear and robust across 3 of 4 splits.

Decision: **MERGED** as new baseline. val_avg=116.34 > 121.50; test_avg=107.33 > 112.49.

Next steps: fern assigned PR #3413 (n_layers=8 + AMP) to test depth scaling on the new baseline.

## 2026-05-15 18:30 — PR #3207 re-run (post-NaN-fix): geom-conditioned slice (nezuko) — sent back for rebase
- student: willowpai2i24h2-nezuko
- branch: `willowpai2i24h2-nezuko/geom-slice-injection`
- run: `5yws4drs` (single arm, NaN fix patch applied in `evaluate_split`)

| Metric | Value (W&B, clean) | vs baseline (PR #3200) |
|---|---|---|
| `val_avg/mae_surf_p` | 127.71 | +5.1% worse |
| `test_avg/mae_surf_p` | 116.56 | +3.6% worse |
| best epoch | 11 / 50 (wall clock at ep 11) | — |

Per-split test (best-val checkpoint): single=133.43, camber_rc=126.58 (**beats baseline 133.37**), camber_cruise=89.29, re_rand=116.94.

- analysis: nezuko's NaN-fix patch is correct — the new W&B test_avg (116.56) matches their offline-corrected value (115.71) within rounding. The PGOT hypothesis is supported in pattern: camber_cruise is the easiest split, camber_rc is the hardest, and camber_rc beats the merged baseline on that split alone. But the equal-weight mean is the metric; nezuko is worse than baseline on the other 3 splits and worse net.
- decision: **sent back for rebase + re-run** on the new baseline (PR #3200, Fourier 8 bands + NaN fix). Hypothesis: geom-slice and Fourier are orthogonal mechanisms (slice-token level vs input-augmentation level) and should compound. This is the final iteration — if rebased re-run doesn't beat baseline, close.
- next steps: if it wins, merge as 2nd baseline; if not, close and assign nezuko a Round-3 hypothesis.


## 2026-05-15 17:25 — PR #3191: Per-sample scale-normalizing loss (alphonse) — CLOSED
- student: willowpai2i24h2-alphonse
- branch: `willowpai2i24h2-alphonse/scale-norm-loss`
- hypothesis: dividing per-sample sq_err by per-sample, per-channel variance of `y_norm` rebalances the gradient between high-Re and low-Re samples
- runs: `qt4mwr34` (eps=1e-3, winner), `34gllrtz` (eps=1e-6, unstable)

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p (offline) | val_single | val_camber_rc | val_camber_cruise | val_re_rand |
|---|---|---|---|---|---|---|
| eps=1e-3 | 148.51 | 140.57 (offline eval_clean.py, NaN in W&B) | 196.69 | 183.68 | **90.23** | 123.46 |
| eps=1e-6 | 365.51 | 346.89 | — | — | — | — |

- analysis: cruise (the low-Re-heavy split) was the lowest per-split val MAE — directionally consistent with the hypothesis. But `val_single_in_dist` regressed badly (196.69 vs ~140 on the merged baseline) and the equal-weight mean is the metric. eps=1e-3 vs eps=1e-6 showed a sharp instability cliff: tiny `eps` divides by near-zero variance for low-energy samples and blows up the loss. Side-channel offline eval (eval_clean.py) used because W&B test_avg was NaN from the scoring bug (now fixed in merged baseline). Even taking the offline number, 140.57 > 112.49 baseline test.
- decision: **closed**. Worse than the new merged baseline on both val and test. Hypothesis lives in too-narrow a slice of the design space to compound easily.
- next steps: reassigned alphonse to FiLM Re-conditioning (PR #3350) — a different attack on the same dynamic-range problem, from the architecture side.

## 2026-05-15 17:24 — PR #3206: Capacity scale-up n_hidden=256, n_head=8, slice_num=128 (frieren) — CLOSED
- student: willowpai2i24h2-frieren
- branch: `willowpai2i24h2-frieren/capacity-256-8h-128s`
- hypothesis: 2-3× parameter capacity helps overcome the small published Transolver config; 8M params + slice_num=128
- runs: `18jia682` (Arm A bs=4 OOM), `vn27s8f7` (Arm A bs=2), `hncmk6wk` (Arm B 192/6/96 bs=4 fallback)

| Arm | bs | val_avg (best) | test_avg (offline) | Epochs in 30 min | Peak GB |
|---|---|---|---|---|---|
| A 256/8/128 | 4 | — (OOM) | — | 0 | 93.4 (OOM) |
| A 256/8/128 | 2 | 159.95 @ ep 6 | 148.07 | 6 | 54.4 |
| B 192/6/96 | 4 | **160.29 @ ep 6** | 147.02 | 8 | 71.1 |

- analysis: Arm A at the PR-specified `batch_size=4` is *infeasible* on 96 GB (93.4 GB allocated → OOM at epoch 1). At equal 30-min budget, the larger model (Arm A bs=2) and the smaller fallback (Arm B bs=4) land within 0.7% of each other — both severely undertrained relative to the 50-epoch schedule. Capacity scale-up at this wall-clock budget is a wash. Student diagnosed the same scoring.py NaN root cause as nezuko (independently). Offline reeval.py numbers are inadmissible per program contract; even taking them, both arms lose to baseline (147.02 vs 112.49 test).
- decision: **closed**. Useful information about VRAM constraint at bs=4 with naive scale-up.
- next steps: reassigned frieren to slice_num=96 + gradient checkpointing (PR #3353) — surgical 50% slice bump with a concrete memory plan, addressing the OOM lesson from this PR.

## 2026-05-15 17:24 — PR #3218: Stochastic depth / DropPath (thorfinn) — CLOSED
- student: willowpai2i24h2-thorfinn
- branch: `willowpai2i24h2-thorfinn/drop-path`
- hypothesis: per-block residual DropPath with linearly-scaled drop_prob regularizes against overfitting on 1499 samples with full-file camber holdouts
- runs: `6cz9o2u5` (dpr=0.1), `6qvqt304` (dpr=0.1 seed 2), `7b1ae3dt` (dpr=0.2, winner)

| Arm | val_avg | val_single | val_camber_rc | val_camber_cruise | val_re_rand | test partial-avg (3 finite) |
|---|---|---|---|---|---|---|
| dpr=0.1 (seed 1) | 130.94 | 152.72 | 138.31 | 109.19 | 123.54 | 130.76 |
| dpr=0.1 (seed 2) | 131.75 | 162.35 | 141.60 | 105.54 | 117.51 | 130.96 |
| dpr=0.2 | **128.90** | 168.52 | 135.93 | **96.86** | **114.28** | 127.80 |

- analysis: DropPath at dpr=0.2 beat dpr=0.1 by ~2.4 absolute on val_avg, which is bigger than the dpr=0.1 seed variance (~0.8). The OOD signal is the cleanest result: all three OOD splits improved with dpr=0.2 (camber_rc -4.0, camber_cruise -10.5, re_rand -6.2) while the in-dist split regressed (+11.0). Classic regularization trade-off, exactly as the hypothesis predicted. Late-training behavior under dpr=0.2 was much more stable (final-epoch val_avg 138.93 vs 185.03). All runs hit the wall clock at ~ep 14 and reported NaN test_avg (scoring bug — fern's PR #3200 fix lands the fix in main).
- decision: **closed**. val_avg=128.90 doesn't beat new merged baseline (121.50). The OOD-specific gain is interesting but the net equal-weight metric loses.
- next steps: reassigned thorfinn to divergence-free aux loss (PR #3356) — a physics-informed regularizer that should give directional (rather than random) bias toward generalization.

## 2026-05-15 17:22 — PR #3200: Fourier position features 8 bands (fern) — **MERGED, new baseline**
- student: willowpai2i24h2-fern
- branch: `willowpai2i24h2-fern/fourier-pos-features`
- hypothesis: sinusoidal Fourier features on normalized (x, z) give the model direct access to multi-scale spatial frequency content
- runs: `t1ai7kzf` (8 bands, winner), `oj5578rn` (12 bands, worse)

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | val_single | val_camber_rc | val_camber_cruise | val_re_rand |
|---|---|---|---|---|---|---|
| 8 bands | **121.50** | **112.49** | 139.80 | 138.71 | 93.55 | 113.93 |
| 12 bands | 129.26 | 121.27 | — | — | — | — |

- analysis: clean win. 8 bands beats 12 bands clearly on both val and test (the smaller input has fewer dims for the shallow preprocess MLP to disentangle in 14 epochs). All 4 splits stayed finite — fern bundled the `evaluate_split` NaN fix (zero non-finite y rows before `accumulate_batch`) in the same PR, unblocking finite W&B test_avg for the entire research track. Per-split test values: single=122.01, camber_rc=133.37, camber_cruise=83.11, re_rand=111.46. Val curve still descending at ep 14 when wall clock fired — there's still headroom in this configuration.
- decision: **merged** as the new baseline (commit `421b225f`). BASELINE.md updated.
- next steps: Round-2 hypotheses build on this baseline. Fern reassigned to learnable Fourier frequencies (PR #3352) as a natural extension.


## 2026-05-15 15:25 — PR #3207: PGOT-style geometry-conditioned slice assignment
- student: willowpai2i24h2-nezuko
- branch: `willowpai2i24h2-nezuko/geom-slice-injection`
- hypothesis: injecting per-node geometry features (NACA M/P/T, AoA, Re, gap, stagger) into PhysicsAttention's slice projection improves generalization across the camber-holdout splits without hurting in-distribution
- runs: `pjmkgg22` (wandb_group `willow-pai2i-24h-r2/geom-slice-injection`)

| Arm | val_avg/mae_surf_p (best, ep 12) | test_avg/mae_surf_p (W&B) | val_single | val_camber_rc | val_camber_cruise | val_re_rand |
|---|---|---|---|---|---|---|
| geom-slice | **128.34** | NaN ⚠ | 145.96 | 142.21 | 107.66 | 117.51 |

- analysis: The hypothesis is supported on validation — geom-slice beats the warmup=3 val_avg (136.55, PR #3194) by ~6% with no regression on any of the four val tracks, and `val_geom_camber_rc` drops from 152.82 → 142.21 (–7%), exactly the split the hypothesis targeted. The run completed all 50 epochs in 31.5 min (just over wall clock cap, last-epoch eval was the bottleneck) and converged cleanly with `val_avg` still falling slowly after epoch 12, suggesting more headroom with a slightly larger model or schedule adjustment. **However, `test_avg/mae_surf_p` is NaN in W&B** — same global bug as PR #3194 (data/scoring.py:48 computes `(pred - y).abs() * mask` BEFORE the per-sample skip, so `inf*0 = NaN` poisons the accumulator when GT has non-finite values; reproduced to `test_geom_camber_cruise/000020.pt` having `y[..., 2] = -inf` at 761 volume nodes). The student computed an offline-corrected `test_avg = 115.71` by re-running scoring with NaN-zeroed samples, but the program contract requires the W&B-logged metric to be the source of truth.
- decision: **sent back** to draft with the exact `evaluate_split` patch (pre-zero non-finite y samples and exclude them from the metric via the mask/is_surface, before calling `accumulate_batch`). Asked the student to re-run the same single arm and confirm the W&B `test_avg/mae_surf_p` reads ~115.71. The hypothesis is the strongest candidate so far; if the re-run lands a finite test number, this becomes the first merge-eligible Round-1 result.
- next steps: on a clean re-run, merge this as the new baseline (val_avg/mae_surf_p=128.34, test=115.71). Then Round 2 priorities: (a) stack geom-slice + warmup=3 (small additive risk, both target different bottlenecks), (b) per-block geometry conditioning (FiLM-style modulation), (c) sweep `slice_num` since slice-token capacity is the load-bearing component.


## 2026-05-15 14:45 — PR #3194: 5-epoch LR warmup + cosine annealing
- student: willowpai2i24h2-askeladd
- branch: `willowpai2i24h2-askeladd/lr-warmup-cosine`
- hypothesis: linear LR warmup over the first 5 epochs prevents cold-start damage to the PhysicsAttention slice projection and improves `val_avg/mae_surf_p`
- runs: `5jtgoadb` (warmup=3), `gyin7q96` (warmup=5)

| Arm | val_avg/mae_surf_p (best, ep 13) | test_avg/mae_surf_p | val_single | val_camber_rc | val_camber_cruise | val_re_rand |
|---|---|---|---|---|---|---|
| warmup=3 | **136.55** | NaN ⚠ | 159.58 | 152.82 | 109.78 | 124.01 |
| warmup=5 | 153.72 | NaN ⚠ | 207.68 | 155.53 | 116.98 | 134.70 |

- analysis: Two arms compared; warmup=3 beat warmup=5 across every val split. Both hit the 30-min wall clock at epoch 14 (cosine barely decayed). The student rightly flagged that this is a warmup-3-vs-warmup-5 comparison only — there is no no-warmup arm to confirm warmup itself beats the existing schedule. `test_geom_camber_cruise` returned `Infinity` in the pressure channel for at least one cruise sample on both arms, which propagates through `data/scoring.py`'s global accumulator and poisons `test_avg/mae_surf_p` to NaN. NaN on the paper-facing metric is a merge blocker per the program contract.
- decision: **sent back** to the student with two requirements: (1) defensively zero out predictions in padded positions and apply `nan_to_num(...).clamp_(-50, 50)` inside `evaluate_split` to localize any overflow without touching the read-only `data/scoring.py`; (2) re-run with two arms in the same wandb_group `willow-pai2i-24h-r2/warmup-cosine-v2` — `warmup=0` (proper baseline) and `warmup=3` (winner). The warmup=5 arm is dropped.
- next steps: once the re-run clears NaN and shows warmup=3 ≥ warmup=0 by any margin, merge. The `nan_to_num` fix will become the baseline for all subsequent PRs.

