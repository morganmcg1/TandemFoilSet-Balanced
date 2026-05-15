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

