# SENPAI Research Results

Results log for `icml-appendix-willow-pai2g-48h-r2`. Wave 1 launched 2026-05-12.

---

## 2026-05-12 18:56 — PR #1454: Enable unified positional encoding (unified_pos=True, ref=8)

- **Branch:** `willowpai2g48h2-tanjiro/unified-pos-ref8`
- **Student:** willowpai2g48h2-tanjiro
- **Hypothesis:** Flip `unified_pos=True, ref=8` in `model_config` to use a grid-based positional encoding instead of raw `(x, z)` coords. Predicted −3 to −8% on `val_avg/mae_surf_p`, biggest on `val_geom_camber_*`.

### Result table (W&B run `mwo6fi5h`, verified)

| Metric | Value | Note |
|---|---|---|
| `val_avg/mae_surf_p` (best, epoch 10) | **147.6498** | first concrete reference number on this branch |
| `val_single_in_dist` surf p | 181.59 | |
| `val_geom_camber_rc` surf p | 170.45 | |
| `val_geom_camber_cruise` surf p | 104.54 | smallest because cruise has smaller pressure scale |
| `val_re_rand` surf p | 134.02 | |
| `test_single_in_dist` surf p | 166.52 | |
| `test_geom_camber_rc` surf p | 157.25 | |
| `test_geom_camber_cruise` surf p | **NaN** | corrupt GT + scoring bug, see below |
| `test_re_rand` surf p | 134.64 | |
| `test_avg/mae_surf_p` | **NaN** | merge-blocker |
| Run time | 22.6 min, 10 epochs | val still descending at last epoch |
| Params | 0.68M | +0.02M vs. baseline (preprocess MLP input 24→86) |

### Discovery: pre-existing bugs surfaced by this PR

1. **Constructor inconsistency in `Transolver`:** the `unified_pos=True` branch used `ref**3 = 512` (3D-Transolver copy) but the `forward` pass never built the encoding, so the flag alone crashed (`mat1 and mat2 shapes cannot be multiplied (200x24 and 534x256)`). Student fixed `train.py` with: (a) switch to `ref**2 = 64` for our 2D problem; (b) build per-mesh min-max-normalized distance encoding in `forward`; (c) plumb `mask` from train/eval call sites into the model dict.
2. **`data/scoring.py` NaN propagation:** `test_geom_camber_cruise/000020.pt` has NaN in the `p` channel of `y` (corrupt preprocessing artifact). `accumulate_batch` filters NaN-GT samples from the node count but `0 * nan = nan` still propagates through the err-sum, yielding a NaN channel total. This affects **every PR this round** that runs end-of-run test evaluation on `test_geom_camber_cruise`. Fix is a one-line `nan_to_num` on err before `* mask`.

### Decision

- **Sent back to student** for (a) the one-line `data/scoring.py` fix (authorized as an infra bug fix), (b) re-run at `--epochs=15` (val curve still descending at epoch 10 + we want to use more of the 30-min wall-clock budget for the cosine anneal), (c) same `unified_pos=True, ref=8` config so we get a clean `test_avg/mae_surf_p` without confounding hypothesis variables.
- Not merged: NaN test metric violates the paper-facing contract per `program.md`.
- Not closed: result is informative (val 147.65 is the first reference point, the val curve looks healthy, and the implementation is the right corrective shape for the broken constructor). The merge-eligible re-run inherits the same unified-pos code.

### Analysis

- **Val curve:** `val_avg/mae_surf_p` over 10 epochs went 261 → 222 → 214 → 179 → 190 → 172 → 168 → 151 → 156 → 148. Not strictly monotonic (epoch 4→5 spike +10.7, epoch 8→9 spike +4.8) but clearly trending down. Final epoch was the best, so undertrained.
- **OOD vs ID:** within-run, `val_geom_camber_cruise` (OOD) has the lowest absolute surf p MAE, but that's largely a function of the smaller pressure scale of the cruise domain (avg per-sample y std ~164 vs. ~458 for raceCar single, per `program.md`). Cannot read the OOD-improvement signal directly without a non-unified-pos baseline to compare against.
- **Implication for other wave-1 PRs:** the scoring NaN bug will hit every PR's `test_avg/mae_surf_p` unless they pull tanjiro's fix. Once tanjiro's re-run lands and merges, the other 7 PRs will need to rebase + rerun for clean test metrics. Plan to send each back individually after they post initial results.

---

## 2026-05-12 19:00 — PR #1452: Swap MSE → Smooth-L1 (Huber β=1.0)

- **Branch:** `willowpai2g48h2-frieren/smooth-l1-loss`
- **Student:** willowpai2g48h2-frieren
- **Hypothesis:** Replace MSE with Smooth-L1 (Huber β=1.0) in both training loop and `evaluate_split` (loss only — metric in `data/scoring.py` is unchanged). Tames high-Re outliers that dominate MSE gradients. Predicted −3 to −10% on `val_avg/mae_surf_p`, biggest on `val_re_rand` and high-Re-heavy splits.

### Result table (W&B run `zkytqdmi`, verified)

| Metric | Value | Note |
|---|---|---|
| `val_avg/mae_surf_p` (best, epoch 10) | **111.0609** | **leading wave-1 result** (~25% better than tanjiro's 147.65) |
| `val_single_in_dist` surf p | 134.74 | hardest val split |
| `val_geom_camber_rc` surf p | 123.58 | |
| `val_geom_camber_cruise` surf p | 85.84 | matches prediction: high-Re-heavy split is the easiest |
| `val_re_rand` surf p | 100.08 | second-lowest, also matches |
| `test_single_in_dist` surf p | 121.89 | |
| `test_geom_camber_rc` surf p | 105.66 | |
| `test_geom_camber_cruise` surf p | **NaN** | same `data/scoring.py` bug as #1454 |
| `test_re_rand` surf p | 99.15 | |
| `test_avg/mae_surf_p` (4-split) | **NaN** | merge-blocker |
| `test_3split_avg/mae_surf_p` | 108.90 | informative but non-contracted |
| Train loss range | 0.07–0.54 | sanity check OK (Huber unsquared range) |
| Peak VRAM | 42.1 GB | well under 96 GB cap |
| Run time | 22.4 min | 10 epochs, room for ~3 more |
| Params | 0.66M | baseline architecture (no model change) |

### Val curve

| Epoch | val_avg/mae_surf_p |
|---|---|
| 1 | 216.75 |
| 2 | 207.85 |
| 3 | 181.87 |
| 4 | 165.49 |
| 5 | 167.87 (+2.4) |
| 6 | 165.17 |
| 7 | 137.73 |
| 8 | 118.89 |
| 9 | 117.32 |
| 10 | **111.06** ⭐ |

Monotonic from epoch 7 onward, one tiny spike epoch 4→5. Final epoch is the best — strongly suggests this run is undertrained, more epochs should help.

### Decision

- **Sent back to student** for (a) one-line `data/scoring.py` NaN-safe fix (authorized as infra bug fix, in parallel with PR #1454's identical fix), (b) re-run at `--epochs=15` since val was still descending steeply at epoch 10 (117→111 in the last 2 epochs), (c) keep Smooth-L1 β=1.0 isolated.
- If clean rerun lands, this is the wave-1 winner.

### Analysis

- **Hypothesis confirmed pattern-wise:** the two splits predicted to benefit most from outlier capping (`val_re_rand`, `val_geom_camber_cruise`) are the two lowest absolute MAEs. The two non-high-Re-dominated splits (`val_single_in_dist`, `val_geom_camber_rc`) are the highest.
- **vs. tanjiro PR #1454:** 111.06 (frieren) vs. 147.65 (tanjiro) on val_avg/mae_surf_p, ~25% lower. Frieren wins on a loss-function change, tanjiro on a positional encoding change. These are orthogonal — they could stack in wave 2.
- **β sweep is a natural follow-up:** β=1.0 was a guess; values in {0.1, 0.3, 1.0, 3.0} could be tested. Lower β acts more like L1 (more aggressive outlier capping); higher β acts more like MSE.

---

## 2026-05-12 19:16 — PR #1455: Batch=8, lr=7.1e-4 (sqrt(2)-scaled)

- **Branch:** `willowpai2g48h2-thorfinn/batch-8-lr-up`
- **Student:** willowpai2g48h2-thorfinn
- **Hypothesis:** Doubling batch size 4→8 with sqrt-scaled lr (5e-4→7.1e-4) reduces gradient noise and improves convergence at no VRAM cost. Predicted −2 to −6% on val_avg/mae_surf_p.

### Result table (W&B run `2glb7y77`, student-reported)

| Metric | Value | Note |
|---|---|---|
| `val_avg/mae_surf_p` (best, epoch 10) | **162.39** | weakest of the three completed wave-1 PRs |
| `val_single_in_dist` surf p | (not posted per-split for val) | |
| Test 3-split avg (ex. cruise) | 162.63 | tracks val — good gen |
| `test_single_in_dist` surf p | 212.97 | highest test split |
| `test_geom_camber_rc` surf p | 155.35 | |
| `test_geom_camber_cruise` surf p | **NaN** | same scoring bug |
| `test_re_rand` surf p | 119.56 | lowest |
| `test_avg/mae_surf_p` (4-split) | **NaN** | merge-blocker |
| Peak VRAM | 84.2 GB / 96 GB | room for batch=10/12 |
| Run time | 21.7 min, 10 epochs | val still improving at last epoch |
| Params | 0.66M | baseline architecture |

### Standings after 3 completed wave-1 PRs

| PR | Hypothesis | val_avg/mae_surf_p |
|---|---|---|
| #1452 frieren | Smooth-L1 (Huber β=1) | **111.06** |
| #1454 tanjiro | unified-pos ref=8 | 147.65 |
| #1455 thorfinn | batch=8, lr=7.1e-4 | 162.39 |

### Decision

- **Sent back to student** for (a) same one-line `data/scoring.py` NaN-safe fix as #1452/#1454 (parallel race), (b) re-run at `--epochs=15` since val was still descending at the last epoch (164.75 → 162.39 over the final 2 epochs), (c) keep `--batch_size=8 --lr=7.1e-4` to give the original hypothesis a fair training budget.
- **Operational note:** GraphQL API rate limit was exhausted during the send-back. Comment posted and label swapped via REST; PR draft conversion deferred to next invocation (after GraphQL reset at 19:48 UTC). Student poll uses labels only (not isDraft), so thorfinn will pick up the work regardless.

### Analysis

- batch+lr scaling at sqrt(2) underperforms relative to Huber loss and unified-pos in the same wave. Possible explanations: (a) larger batch reduces gradient noise — but the surface loss component is computed over a tiny fraction of nodes, where averaging across more samples might *under-emphasize* surface signal; (b) lr=7.1e-4 is mostly held near peak across the 10-epoch cosine (only ~10% lower than peak at epoch 5), so the sqrt(2) scaling is essentially never compensated by anneal-late convergence.
- Generalization is healthy — test 3-split avg (162.63) ≈ val (162.39), so the model isn't overfitting; it's just a less-good optimum than the other variants. 
- If the 15-epoch rerun still lands far above frieren's 111, this is a clean negative for batch+lr scaling and we'd close it. Worth one more shot first.

---

## 2026-05-12 19:55 — PR #1454 (rerun): Enable unified positional encoding (unified_pos=True, ref=8), --epochs=15

- **Branch:** `willowpai2g48h2-tanjiro/unified-pos-ref8`
- **Student:** willowpai2g48h2-tanjiro
- **Change vs. first attempt:** (1) one-line `data/scoring.py` `nan_to_num` fix per advisor authorization, (2) `--epochs=15` (was 10), same `unified_pos=True, ref=8` config.

### Result table (W&B run `24w5a8qx`, verified)

| Metric | Value | Note |
|---|---|---|
| `val_avg/mae_surf_p` (best, epoch 14) | **128.7761** | ↓ from 147.65 (e10 run) → **−12.8%** |
| `val_single_in_dist` surf p | 163.05 | |
| `val_geom_camber_rc` surf p | 138.53 | |
| `val_geom_camber_cruise` surf p | 94.21 | smallest, smaller pressure scale of cruise |
| `val_re_rand` surf p | 119.31 | |
| `test_single_in_dist` surf p | 142.38 | |
| `test_geom_camber_rc` surf p | 130.43 | |
| `test_geom_camber_cruise` surf p | **81.42** ✅ | finite — scoring fix worked |
| `test_re_rand` surf p | 115.07 | |
| `test_avg/mae_surf_p` (4-split) | **117.33** ✅ | finite |
| Run time | ~31.4 min, 14 epochs done (timeout cap hit during epoch 15) |  |
| Params | 0.68M | unchanged from e10 |

### Decision

- **Closed.** Frieren's PR #1452 rerun (val=100.77, test=90.38) landed first as the wave-1 winner; tanjiro's val=128.78 / test=117.33 is 28%/30% worse on the post-merge baseline.
- The unified_pos architecture is genuinely orthogonal to Huber loss, so closing this PR with the explicit follow-up of testing the **stack** (unified_pos on top of merged Huber baseline) in a fresh PR — see new PR #1551 below.
- Rebase rather than fresh PR was rejected because both PRs touch `train.py` (loss site) and `data/scoring.py` (your fix vs. frieren's). Starting fresh is faster than untangling.

### Analysis

- 15 epochs of cosine anneal pulled val from 147.65 → 128.78 (−12.8%), validating both the schedule alignment and the unified-pos forward fix. At epoch 10 the e15 run was already at 143.40 (vs. 147.65 for the e10 run with `T_max=10`), so longer schedules help even at the same epoch index.
- Val still descending sharply at epoch 14 (130.18 → 128.78 = −1.1%) — the run is still undertrained at 15 epochs but the 30-min cap binds.
- OOD-vs-ID pattern: `val_geom_camber_cruise` (94.21) lowest, `val_single_in_dist` (163.05) highest — pressure-scale artifact more than positional-encoding signal (per-domain y_std differs).
- The scoring fix tanjiro wrote is functionally equivalent to frieren's `torch.where` variant; frieren landed first on squash-merge, so frieren's form is in the baseline.

---

## 2026-05-12 19:57 — PR #1452 (rerun, MERGED): Swap MSE → Smooth-L1 (Huber β=1.0) + scoring NaN-safe fix, --epochs=15

- **Branch:** `willowpai2g48h2-frieren/smooth-l1-loss`
- **Student:** willowpai2g48h2-frieren
- **Change vs. first attempt:** (1) `data/scoring.py` NaN-safe fix via `torch.where(mask, err, zero)` (no arithmetic on masked positions), (2) `--epochs=15` (was 10), same Smooth-L1 β=1.0 in both training and `evaluate_split`.

### Result table (W&B run `lo8vp7rj`, verified)

| Metric | Value | Note |
|---|---|---|
| `val_avg/mae_surf_p` (best, epoch 14) | **100.7659** | ↓ from 111.06 (e10) → **−9.3%** |
| `val_single_in_dist` surf p | 119.74 | |
| `val_geom_camber_rc` surf p | 109.38 | |
| `val_geom_camber_cruise` surf p | 80.90 | lowest (matches hypothesis: Huber caps high-Re outliers) |
| `val_re_rand` surf p | 93.04 | second-lowest (matches) |
| `test_single_in_dist` surf p | 106.01 | |
| `test_geom_camber_rc` surf p | 96.25 | |
| `test_geom_camber_cruise` surf p | **68.86** ✅ | finite — scoring fix worked |
| `test_re_rand` surf p | 90.42 | |
| `test_avg/mae_surf_p` (4-split) | **90.3840** ✅ | finite, first 4-split test metric on this branch |
| Peak VRAM | ~42 GB / 96 GB | unchanged from e10 |
| Run time | ~30 min (cap hit during epoch 15) | 14 full epochs |
| Params | 0.66M | baseline arch |

### Final wave-1 standings (val_avg/mae_surf_p)

| PR | Hypothesis | val_avg | test_avg | Status |
|---|---|---|---|---|
| **#1452 frieren** | Smooth-L1 (Huber β=1) + scoring fix | **100.77** | **90.38** | **MERGED — new baseline** |
| #1454 tanjiro | unified-pos ref=8 (+ constructor fix) | 128.78 | 117.33 | CLOSED, follow-up #1551 |
| #1455 thorfinn | batch=8, lr=7.1e-4 (sqrt(2)-scaled) | 162.39 (e10) | NaN (rerun pending) | WIP (rerun in flight) |
| #1446 alphonse | schedule-align (--epochs=10) | — | — | WIP (rate-limit-delayed start) |
| #1448 askeladd | slice_num=128 | — | — | WIP (rate-limit-delayed start) |
| #1449 edward | surf_weight=30 | — | — | WIP (rate-limit-delayed start) |
| #1450 fern | mlp_ratio=4 | — | — | WIP (rate-limit-delayed start) |
| #1453 nezuko | n_hidden=192 | — | — | WIP (rate-limit-delayed start) |

### Decision

- **Merged at 2026-05-12 20:02 UTC** as the wave-1 winner. `BASELINE.md` created with val=100.77 / test=90.38 as the new reference numbers for all future PRs to compare against. Two files changed: `train.py` (loss swap) and `data/scoring.py` (NaN-safe accumulator).
- The scoring fix is the dominant value-add — it unblocks every future PR's test metric. The Huber loss is the headline improvement.

### Analysis

- Five extra epochs of cosine anneal pulled val from 111.06 → 100.77 (−9.3%). Val still descending at epoch 14 (102.88 → 100.77 over the last 2 epochs); a 20-epoch run would likely improve further but exceeds the 30-min cap budget at current per-epoch cost (~130 s/epoch).
- Per-split pattern is monotonically consistent with hypothesis: `val_geom_camber_cruise` (80.90) and `val_re_rand` (93.04) are the two lowest — Huber caps the gradient on high-Re outliers that MSE would have over-penalized.
- Test follows val closely with a slight edge (90.38 < 100.77): the model isn't overfitting and generalizes well across the 4 splits.

---

## 2026-05-12 20:05 — Wave-2 launches: PR #1551 (tanjiro), PR #1554 (frieren)

After merging the wave-1 winner, two newly-idle students were assigned wave-2 stack tests on top of the merged Huber baseline:

| PR | Student | Slug | Hypothesis | Predicted Δ vs. 100.77 val |
|---|---|---|---|---|
| #1551 | tanjiro | `unified-pos-on-huber` | unified_pos=True, ref=8 stacked on Huber baseline (re-applying the constructor fix + forward-side encoding on the new branch) | −3 to −8% (~92–98 val) |
| #1554 | frieren | `swa-on-huber` | Stochastic Weight Averaging on final 4/15 epochs, swa_lr=1e-4, terminal test eval uses `swa_model` | −3 to −7% (~94–98 val) |

Both are pure single-variable add-ons; both have low implementation risk and high stacking-orthogonality with Huber. Wave 1's other 5 PRs (alphonse, askeladd, edward, fern, nezuko) are still running on the pre-merge baseline (MSE) — their results will need to be evaluated against the new baseline (Huber@100.77) when they post, since the Huber win is itself a ~25% improvement that those MSE-arm hypotheses would need to clear.


---

## 2026-05-12 21:10 — PR #1448 askeladd (slice_num=128, wave-1 MSE arm): CLOSED

- Branch: `willowpai2g48h2-askeladd/slice-num-128`
- Hypothesis: Double `slice_num` in the PhysicsAttention block (64 → 128) to give the model more learned latent slices to softmax-route nodes into, on top of the pre-merge MSE baseline.
- 3 seeds (continuing askeladd's wave-1 rigor):

| Seed | best val_avg/mae_surf_p | best epoch |
|---|---:|---:|
| A | 131.67 | (terminal) |
| B | ~134.78 | (terminal) |
| C | ~136.49 | (terminal) |
| Mean ± std | **134.31 ± 2.39** | — |

- Test (best seed A): finite under merged scoring fix but well above new baseline (90.38).
- Decision: **CLOSED**. Best seed is 30.6% worse than the merged Huber baseline (100.77). On the pre-merge MSE baseline alone the lever was a regression (vs. 147.65 → 131.67 is only −10.8%, less than the ~25% Huber win), and stacking with Huber is unlikely to recover that gap.

### Follow-up

- Closed cleanly with a hand-off comment pointing askeladd at a new wave-2 hypothesis (PR #1585, FiLM-on-Huber, research-ideas H5). FiLM is a more principled way to inject the same global flow-context (Re/AoA/NACA/gap/stagger) into the model than widening the latent slice budget.

---

## 2026-05-12 21:12 — PR #1455 thorfinn rerun (batch=8, lr=7.1e-4, wave-1 MSE arm): CLOSED

- Branch: `willowpai2g48h2-thorfinn/batch-8-lr-up`
- Hypothesis (rerun): Increase batch size from 4 → 8 with sqrt(2)-scaled lr (5e-4 → ~7.1e-4); run for full 15 epochs with the merged `data/scoring.py` fix.
- Single-seed result:

| Metric | Value |
|---|---:|
| val_avg/mae_surf_p (best) | 141.94 |
| test_avg/mae_surf_p | 125.92 |
| Peak VRAM | 84.2 GB |
| Wall time | ~28 min |
| best_epoch | 10 |

- Decision: **CLOSED**. 41% worse than new Huber baseline (val=100.77). The lr-batch scaling alone — even with the scoring fix applied — doesn't close the gap to the Huber win. Possible the lr scaling overshot (sqrt(2) was a rule-of-thumb), but the wider-batch regularization story doesn't survive Huber's outlier-gradient capping.

### Follow-up

- Closed cleanly with a hand-off comment pointing thorfinn at a new wave-2 hypothesis (PR #1586, Re-based loss weighting on Huber, research-ideas H4). Per-sample Re-weighting directly addresses the "y std varies 10× across samples" observation from `program.md`, which is mechanism-orthogonal to Huber's gradient capping.

---

## 2026-05-12 21:15 — Wave-2 launches: PR #1585 (askeladd), PR #1586 (thorfinn)

Both newly-idle students were reassigned wave-2 stack tests on top of the merged Huber baseline. With this round, all 4 of the most promising "stack on Huber" levers from `RESEARCH_IDEAS_2026-05-12_round2.md` are now in flight:

| PR | Student | Slug | Hypothesis | Predicted Δ vs. 100.77 val |
|---|---|---|---|---|
| #1551 | tanjiro | `unified-pos-on-huber` | unified_pos=True ref=8 stacked on Huber | −3 to −8% (~92–98 val) |
| #1554 | frieren | `swa-on-huber` | SWA on final 4/15 epochs, swa_lr=1e-4, terminal test eval uses `swa_model` | −3 to −7% (~94–98 val) |
| #1585 | askeladd | `film-on-huber` | FiLM global conditioning (Re/AoA/NACA/gap/stagger → per-layer γ,β), zero-init for identity start, 3 seeds | −4 to −10% (~91–97 val) |
| #1586 | thorfinn | `re-weight-on-huber` | Per-sample loss reweighting by 1/(shifted log Re), normalized to mean=1 per batch, 1 seed | −4 to −9% (~92–97 val) |

If multiple wave-2 levers land in the predicted range, **wave 3 should stack them** — Huber × unified-pos × FiLM × SWA, etc. The predicted compound improvement from 4 stacked levers (each at the midpoint of its range) is ~100.77 × 0.94 × 0.94 × 0.93 × 0.95 ≈ 78–83 val.

### Notes

- All 4 wave-2 PRs touch **train.py only** (per stack-test discipline). No PR touches `target/models/Transolver.py`, and `data/scoring.py` is frozen with the merged frieren fix.
- The FiLM PR (#1585) is the only one that runs 3 seeds; the other three run 1 seed each (different rigor patterns reflect each lever's inherent variance — FiLM adds new params, the others don't).

---

## 2026-05-12 21:06 — PR #1554 frieren (SWA on Huber): MERGED — new baseline

- Branch: `willowpai2g48h2-frieren/swa-on-huber`
- Hypothesis: Stochastic Weight Averaging on final 4/15 epochs of the Huber baseline, swa_lr=1e-4, anneal_epochs=2, eval on `swa_model.module` at terminal step.
- Result:

| Metric | Old baseline (#1452) | New (SWA+Huber, #1554) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 100.7659 | **99.0704** | **−1.69%** |
| test_avg/mae_surf_p | 90.3840 | **88.8955** | **−1.65%** |
| Wall time | 30.0 min | 30.8 min | +2.7% |
| Peak VRAM | ~42 GB | ~42 GB | flat |
| Params | 0.66M | 0.66M | flat |

- All four **test splits improved** (test_single_in_dist −3.4%, test_geom_camber_rc −0.8%, test_geom_camber_cruise −1.8%, test_re_rand −0.4%).
- Val per-split mostly positive: val_single_in_dist −1.7%, val_geom_camber_rc −4.7%, val_geom_camber_cruise −2.1%; **val_re_rand regressed +2.2%** — speculation in PR comment: only 3 SWA-active epochs averaged in the 30-min cap (epoch 15 didn't start), and `swa_lr=1e-4` is above the cosine floor at that point, so the average is integrating over noisier weights.
- W&B run `cnu8v9i2` (https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2/runs/cnu8v9i2) — verified via wandb-primary subagent: all reported numbers match logged metrics to 4+ decimal places, run state = "finished", no NaN in primary surface metrics.
- One minor non-fatal flag: `swa_test/test_geom_camber_cruise/vol_loss = Infinity` (volume-component normalised loss on the corrupt GT sample `000020.pt`). Surface MAE is finite. Not a regression from #1452.

### Decision

- **Merged at 2026-05-12 21:06 UTC** via `gh pr merge 1554 --squash`. Preflight passed. `BASELINE.md` updated with the new numbers.
- The 1.7% headline improvement is smaller than the predicted −3 to −7% range, but firmly above the merge bar. The "SWA effect" within this run (SWA vs. base-best, same trajectory) is −4.0% val / −5.3% test, which is squarely in the predicted range — the gap is fully explained by frieren's wave-1 baseline run having an unusually good epoch-14 base, while this SWA run's base hit best at epoch 12.

### Analysis

- SWA composes cleanly with Huber. The flat-minima effect shows uniformly across test splits, exactly as predicted for OOD generalization.
- The `val_re_rand` regression suggests `swa_lr` is too high; lowering to 0.1× or 0.05× base lr may close that gap (logged in BASELINE.md follow-ups).
- The merged baseline shifts ~95 → 99 territory on val, ~88 → 89 on test. With three wave-2 levers still in flight (unified-pos, FiLM, Re-weight), each predicted to land another −3 to −10%, the compound 4-lever theoretical floor is ~78–83 val.

---

## 2026-05-12 21:15 — Wave-3 launch: PR #1600 (frieren, beta-sweep-on-swa)

After merging frieren's SWA win, they were re-assigned to test a 3-arm β sweep on the new SWA-on-Huber baseline:

| PR | Student | Slug | Hypothesis | Predicted Δ vs. 99.07 val |
|---|---|---|---|---|
| #1600 | frieren | `beta-sweep-on-swa` | 3-arm sweep: β ∈ {0.3, 1.0, 3.0}, single-variable on the Smooth-L1 transition point | best arm: −1 to −4% (~95–98 val), control: 99.07, worst: neutral or slight regress |

- frieren is the right student to own this since they wrote both the Huber (PR #1452) and SWA (PR #1554) implementations. They have full context to debug any divergent arm.
- The β sweep is the natural hyperparameter-tuning follow-up to the merged baseline. Even if no arm wins, the shape of the β-response curve is diagnostic about the residual distribution late in training.

### Current wave-2/3 portfolio (4 in flight)

| PR | Student | Lever | Stacks on |
|---|---|---|---|
| #1551 | tanjiro | unified_pos=True ref=8 | Huber baseline (#1452) — **stale** (needs rebase onto SWA baseline) |
| #1585 | askeladd | FiLM global conditioning | Huber baseline (#1452) — **stale** (needs rebase onto SWA baseline) |
| #1586 | thorfinn | Per-sample Re-based loss weighting | Huber baseline (#1452) — **stale** (needs rebase onto SWA baseline) |
| #1600 | frieren | β ∈ {0.3, 1.0, 3.0} sweep | SWA-on-Huber baseline (#1554) ✓ |

Three of the four wave-2 PRs were created before the SWA merge and currently target their work against the pre-merge Huber baseline. **Each needs to be sent back for rebase** so its result is comparable to the new SWA-on-Huber baseline (val=99.07).

---

## 2026-05-12 21:25 — PR #1453 nezuko (n_hidden=192, wave-1 MSE arm): CLOSED

- Branch: `willowpai2g48h2-nezuko/wider-n-hidden-192`
- Hypothesis: Widen Transolver `n_hidden` 128 → 192 on the pre-merge MSE+10-epoch baseline.
- Result (2 runs, no seed): val_avg/mae_surf_p = **128.28** (best, run `pn7x5dx8`) and **148.57** (worse, run `k3ddvtjm`). 16% inter-run variance.
- Test (best run): test_avg_3split/mae_surf_p = 129.13 (NaN on cruise pressure due to running against the pre-merge `data/scoring.py`).
- Decision: **CLOSED**. Best run is 29% worse than the new SWA-on-Huber baseline (val=99.07).
- Param count came out to 1.47M (~2.2× baseline 0.66M). Capacity expansion plausible but variance-limited at this schedule budget.

### nezuko follow-up

Reassigned to PR #1617: gradient clipping (max_norm=1.0) on SWA-on-Huber baseline. The lever is motivated *directly by their wave-1 observation* of 16% seed-to-seed variance — clipping is the right defensive lever for gradient-spike instability that Huber's per-element capping doesn't cover. 2-seed protocol so we can measure variance reduction.

---

## 2026-05-12 21:25 — PR #1446 alphonse (schedule-align, --epochs=10): CLOSED — not a regression

- Branch: `willowpai2g48h2-alphonse/schedule-align-baseline`
- Hypothesis: Align cosine `T_max=epochs=10` to actual training budget (the pre-merge baseline had `T_max=15` but `--epochs=10`).
- Result: **never trained** — pod was stuck on rate-limit + outdated baseline window.
- Decision: **CLOSED** as moot. The merged baseline (PR #1452 → #1554) already uses `--epochs=15` with `CosineAnnealingLR(T_max=15)` — schedule alignment landed implicitly as part of the Huber merge, not as an isolated test. Re-running this experiment would test something already in baseline.

### alphonse follow-up

Reassigned to PR #1618: split-loss-by-node-type (Huber on surface + MSE on volume), research-ideas H3. The headline metric is `mae_surf_p` so a surface-specialized loss kind is targeted at exactly the right axis. Wave-1's Huber win came from outlier-gradient capping which is most relevant for high-magnitude surface residuals; on volume, MSE may give a stronger learning signal. Single-variable split-loss change.

---

## 2026-05-12 21:25 — Wave-3 portfolio (5 in flight, 2 stale wave-1 still running)

After the cascade of close+reassign, the active portfolio is now:

| PR | Student | Slug | Stacks on | Predicted Δ vs. 99.07 val |
|---|---|---|---|---|
| #1551 | tanjiro | `unified-pos-on-huber` | Huber baseline (#1452) — **stale**, predates SWA merge | will need rebase if it wins |
| #1585 | askeladd | `film-on-huber` | Huber baseline (#1452) — **stale**, predates SWA merge | will need rebase if it wins |
| #1586 | thorfinn | `re-weight-on-huber` | Huber baseline (#1452) — **stale**, predates SWA merge | will need rebase if it wins |
| #1600 | frieren | `beta-sweep-on-swa` (3-arm) | SWA-on-Huber baseline (#1554) ✓ | −1 to −4% best arm |
| #1617 | nezuko | `grad-clip-on-swa` (2-seed) | SWA-on-Huber baseline (#1554) ✓ | −0.5 to −2% + variance reduction |
| #1618 | alphonse | `surf-huber-vol-mse` | SWA-on-Huber baseline (#1554) ✓ | −2 to −5% |
| (#1449) | edward | `surf-weight-30` (wave-1 MSE arm) | MSE baseline — **stale**, training in progress | needs reframe when results land |
| (#1450) | fern | `mlp-ratio-4` (wave-1 MSE arm) | MSE baseline — **stale**, training in progress | needs reframe when results land |

Edward and fern are mid-training on the original MSE baseline (94 GB GPU usage on their pods, no PR comments yet). Letting them complete; will evaluate their lever delta on the MSE frame and decide rebase vs. close when they post.

### Compound improvement target

If wave-3 PRs land at the midpoint of their predicted ranges, the compound effect on val is:
`99.07 × 0.975 (β-sweep) × 0.985 (grad-clip) × 0.965 (surf-Huber/vol-MSE) ≈ 92`
And wave-2's three "Huber-stale" levers, after rebase onto the merged baseline, could plausibly add another 0.94× (FiLM/unified-pos/Re-weight at midpoint) bringing the theoretical floor to ~87 val.

---

## 2026-05-12 21:50 — PR #1449 edward + PR #1450 fern: CLOSED (baseline-stale, never trained)

- Both PRs were wave-1 single-variable assignments (surf_weight=30, mlp_ratio=4) created at 17:55 UTC against the pre-merge MSE baseline.
- Neither posted training results in the ~4 hours between assignment and triage.
- Root cause: GraphQL rate-limit episodes caused student polls to return "no work assigned" intermittently, and by the time the buckets reset their assignment branches were already 2 merges out of date (Huber merge at 20:02, SWA merge at 21:06). Pods went idle ("No assigned PRs or issues") and never resumed.
- Branch inspection: both branches only contained the original advisor-assignment commit — no student code changes were ever pushed.
- Decision: **CLOSED** as **baseline-stale**, not as regressions. The levers are still scientifically valuable; reopening them on fresh branches forked from the current SWA-on-Huber advisor branch HEAD so the comparison is apples-to-apples.

### Reassignments

| Old PR | New PR | Student | Slug | Stacks on |
|---|---|---|---|---|
| #1449 | **#1620** | edward | `surf-weight-30-on-swa` | SWA-on-Huber baseline (#1554) ✓ |
| #1450 | **#1621** | fern | `mlp-ratio-4-on-swa` | SWA-on-Huber baseline (#1554) ✓ |

Both fresh PRs preserve the original lever exactly — only the baseline frame and the supporting infrastructure (Huber + scoring fix + SWA + schedule-aligned cosine) have changed. Predicted improvements:

- edward: −1 to −4% on val (surf_weight=30 aligns training objective to surface-MAE metric)
- fern: −1 to −5% on val (mlp_ratio=4 restores canonical Transolver FFN capacity, ~0.66M → ~1.0M params)

---

## 2026-05-12 21:50 — Wave-3 portfolio (complete, 5 in flight)

After this reassignment cascade, the full active wave-3 stack-test portfolio against the SWA-on-Huber baseline (val=99.07) is:

| PR | Student | Lever | Mechanism axis | Predicted Δ |
|---|---|---|---|---|
| #1600 | frieren | Huber β ∈ {0.3, 1.0, 3.0} (3 arms) | loss-shape | best arm −1 to −4% |
| #1617 | nezuko | `grad_clip_norm=1.0` (2 seeds) | optimizer-stability | −0.5 to −2% + variance reduction |
| #1618 | alphonse | Huber on surface + MSE on volume | loss-by-node-type | −2 to −5% |
| #1620 | edward | `surf_weight=30.0` (3× baseline) | loss-weighting | −1 to −4% |
| #1621 | fern | `mlp_ratio=4` (canonical Transolver FFN) | architecture-capacity | −1 to −5% |

Wave-2 portfolio (3 in flight, stack-stale on Huber baseline, will be evaluated when results land):

| PR | Student | Lever | Stacks on |
|---|---|---|---|
| #1551 | tanjiro | `unified_pos=True` ref=8 | Huber baseline (#1452) |
| #1585 | askeladd | FiLM global conditioning (3 seeds) | Huber baseline (#1452) |
| #1586 | thorfinn | Per-sample Re-based loss weighting | Huber baseline (#1452) |

### Mechanism-axis coverage

- **Loss-shape:** β-sweep (#1600), surface-vs-volume kind split (#1618)
- **Loss-weighting:** surf_weight bump (#1620), per-sample Re (#1586)
- **Optimizer-stability:** gradient clipping (#1617)
- **Architecture-capacity:** mlp_ratio=4 (#1621), positional-encoding (#1551, unified-pos)
- **Architecture-conditioning:** FiLM (#1585)

This is well-spread across orthogonal axes. If any 2-3 wave-3 levers hit their midpoints, the merged baseline could compound to ~93-95 val. Wave-2 stack-stale arms (if rebased after winning on Huber baseline) could push another 0.94× to ~88-90 val.

### Open question for next review wave

When results land, prioritize:
1. **Which mechanism axis dominates** the compound improvement — is it loss-shape, weighting, stability, or capacity?
2. **Per-split impact pattern** — does any wave-3 lever specifically rescue val_re_rand (the split that regressed under SWA)?
3. **Variance signal** — nezuko's 2-seed grad-clip will measure whether SWA + clipping reduces seed-to-seed variance from the ~16% baseline observed on n_hidden=192.
