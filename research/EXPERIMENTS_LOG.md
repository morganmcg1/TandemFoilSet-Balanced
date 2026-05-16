# SENPAI Research Results

Track: `charlie-pai2i-24h-r1`
Advisor branch: `icml-appendix-charlie-pai2i-24h-r1`

## 2026-05-16 — Loop 18 actions (compile axis closed; 1 dispatch)

**Context**: Pure L1 (#3798) still in-flight pending frieren's seed-pin confirmation. SmoothL1 baseline still active (94.97/85.04). Three open WIP from Loop 16 (#3676 edward), Loop 17 dispatches (5 new), and #3802 alphonse compile arrived for review.

**PR closure (1)**:

### PR #3802 — alphonse — torch.compile determinism + 23-ep budget-soak — CLOSED (determinism failed)

- **Student branch**: `charliepai2i24h1-alphonse/compile-determinism`
- **Hypothesis**: Test whether `torch.compile(mode='reduce-overhead', dynamic=True)` produces deterministic, quality-preserving training on SmoothL1; if so, trade speedup for more epochs.
- **Arm A result (18 epochs, compile vs eager baseline)**: val_avg=**97.5971** vs baseline 94.9723 (+2.76% regression), test_avg=**87.5734** vs 85.0372 (+2.99%). **Outside the ±0.5 determinism window by 2.12 — determinism failed.** Per PR stop-condition, Arm B (23-epoch budget-soak) correctly NOT run.
- **Throughput finding (HIGH VALUE)**:
  - Mean epoch 2-18 wall: 54.40 s vs eager 98.27 s → **1.81× speedup (44.6%)** — much better than the 22% prior estimate
  - peak_memory_gb: 23.83 vs eager 32.95 (−27.7%)
  - Total 18-ep wall: 17.1 min vs eager 29.5 min
- **Divergence shape**: gap small early (within ±5 through epoch 6), widens sharply at epochs 8-11 (+8.54 peak), partially recovers via cosine anneal but doesn't close (+2.62 at final epoch). Mid-training divergence > cosine-anneal recovery in 18 epochs.
- **Verdict**: CLOSED. Quality regression makes the throughput win unusable. The compile axis on `mode='reduce-overhead' + dynamic=True + in-place EMA update` pattern is closed; future revisits should try `mode='default'` (drop cudagraphs to eliminate EMA-aliasing risk) or static-shape compile with batch-level padding bucketing. Alphonse reassigned to Adam β2=0.99 (#3869).
- **Mechanistic hypothesis (from student)**: EMA + cudagraph interaction. EMA model is compile-wrapped but updated via in-place `mul_/add_` from eager code; cudagraph replay path may interact unexpectedly with EMA mutation cadence. Well-documented compile gotcha — published recipes either don't compile EMA or use state_dict swap instead of in-place mutation.
- **Metric artifacts**: `models/model-compile-arm-a-18ep-20260516-083129/metrics.jsonl`

**PR awaiting student recovery (1)**:

### PR #3676 — edward — slice_num=48 + 21 epochs + rebased to SmoothL1 — pod blocked by GH API rate limit

- **Status**: Sent back Loop 16 with rebased SmoothL1 single-arm instructions. Pod is on the correct branch but heartbeat polls fail with `gh: API rate limit exceeded for user ID 20516801` since at least iteration 6 (10:09 UTC). Pod's poll-for-work skill returns "No assigned PRs or issues" because the gh-API JSON parse fails.
- **Verdict**: Leave PR as-is. Operational issue (token rate limit) — closing/reassigning would not help because the same rate limit blocks any new assignment too. Student will pick up work when GH API recovers (typically 30-60 min). Re-check next loop.

**1 new dispatch**:

| PR | Student | Axis | One-line summary |
|---|---|---|---|
| #3869 | alphonse | Optimizer (Adam) | Adam β2 = 0.99 (default 0.999); more responsive second-moment for small-batch transformer |

**Loop 18 systemic findings**:

1. **`torch.compile(mode='reduce-overhead', dynamic=True)` is NOT numerically equivalent** to eager on this Blackwell + bf16 + SmoothL1 + EMA stack. Divergence accumulates mid-training (epochs 8-11) and isn't fully recovered by cosine anneal in 18 epochs. The cause is likely EMA + cudagraph interaction (in-place EMA update from eager code into a compiled forward).
2. **Throughput speedup under compile is real and large** (44.6%, 1.81×) — much higher than prior 22% estimate. The infrastructure win exists if we can find a quality-preserving compile configuration. Defer revisits to a future plateau.
3. **GitHub API rate limits can effectively idle a student pod** (#3676 case): poll-for-work fails silently because the API call returns rate-limit error and the json parser dies. Systemic observation: pods should have rate-limit-aware retry with backoff.

## 2026-05-16 — Loop 17 actions (Pure L1 BIG win in flight; 3 closures, 1 send-back, 2 stale closures, 5 fresh dispatches)

**Context**: 4 PRs landed for review against the SmoothL1 baseline (#3127, val_avg=94.97 / test_avg=85.04). Pure L1 (#3798 frieren) crushed the field with val_avg=86.66 (−8.75%), test_avg=77.21 (−9.20%) — but single-seed and `use_l1` default left at False. Sent back for seed-pin confirmation + default flip before merge. 3 of the other PRs dominated by the Pure L1 mechanism and closed. 2 stale_wip PRs (#3588 Lookahead, #3589 SWA) had no commits across 2 baseline shifts — closed as stale and the students reassigned with fresh hypotheses.

**PR send-back (1)**:

### PR #3798 — frieren — Pure L1 (F.l1_loss) — SENT BACK (huge win pending confirmation)

- **Student branch**: `charliepai2i24h1-frieren/pure-l1`
- **Hypothesis**: F.l1_loss is the β→0 limit of SmoothL1; pushes the L1 tail down to the gradient discontinuity at r=0. Should compound with the SmoothL1 win.
- **Result**: val_avg=**86.66** (−8.75% vs SmoothL1 94.97), test_avg=**77.21** (−9.20% vs SmoothL1 85.04). **Uniform improvement across all 8 val+test splits.**
  - Per-val: single_in_dist=101.41 (−8.51%), camber_rc=94.41 (−9.03%), cruise=69.93 (−7.79%), re_rand=80.91 (−9.51%)
  - Per-test: test_single_in_dist=89.51 (−7.89%), camber_rc=84.96 (−9.31%), cruise=58.54 (−8.72%), re_rand=75.84 (−10.94%)
- **Why sent back (not merged)**: (a) single seed — #3676 found ~6-unit seed variance at narrow+bf16, asked for 1 confirmation seed (seed=42); (b) `use_l1: bool = False` default would mean post-merge students still get SmoothL1 unless they pass `--use_l1` — asked frieren to flip default to True for consistency with #3127 convention.
- **Metric artifacts**: from #3798 PR comments

**PR closures (3 superseded + 2 stale)**:

### PR #3763 — askeladd — SmoothL1 β sweep (β=0.5, β=0.25) — CLOSED (superseded by Pure L1)

- **Student branch**: `charliepai2i24h1-askeladd/smoothl1-beta-sweep`
- **Hypothesis**: SmoothL1 with smaller β pushes the L1 tail closer to r=0. β=0.5 and β=0.25 should monotonically improve.
- **Two-arm result**:
  - β=0.5: val_avg=91.57 (−3.58% vs 94.97), test_avg=83.20 (−2.16%). Best β=0.5 arm.
  - β=0.25: val_avg=91.81 (−3.33%), test_avg=82.71 (−2.74%). Slightly worse than β=0.5 on val, slightly better on test.
- **Per-channel signal (HIGH VALUE)**: β=0.25 uniformly better on Ux/Uy; β=0.5 better on pressure (the primary metric channel) on 2 of 4 splits. The optimal β depends on the channel.
- **Verdict**: CLOSED. Both arms beat SmoothL1 baseline but are decisively dominated by Pure L1 at 86.66 (−8.75%). Progression {β=1.0: 94.97, β=0.5: 91.57, β=0.25: 91.81, β=0.0: 86.66} is non-monotonic with a 4.91-unit gap at the L1 limit — strongly suggesting Pure L1's gradient discontinuity at r=0 is the load-bearing mechanism, not just shrinking β. Askeladd reassigned to Charbonnier loss (#3861) which directly tests the smoothness-at-r=0 question.
- **Metric artifacts**: from #3763 PR comments

### PR #3800 — fern — Per-channel surf_p 4× weighting — CLOSED (regression)

- **Student branch**: `charliepai2i24h1-fern/surf-p-4x`
- **Hypothesis**: Pressure is the primary metric channel; weighting surf_p 4× inside surface loss should directly improve mae_surf_p.
- **Result**: val_avg=97.92 (+3.10% vs 94.97), test_avg=87.39 (+2.77%). **Uniform regression across all 8 val+test splits.**
- **Verdict**: CLOSED. The mechanism is global gradient-budget reallocation (more pressure weight = less Ux/Uy weight) which under SmoothL1's L1 tail is already partially solved. Decisively closed by Pure L1's win — Pure L1 dominates per-channel reweighting on this baseline.
- **Metric artifacts**: `models/model-surf-p-4x-...` (path on student branch)

### PR #3804 — nezuko — n_hidden=128 → 160 — CLOSED (regression + under-completion)

- **Student branch**: `charliepai2i24h1-nezuko/n-hidden-160`
- **Hypothesis**: Widening trunk to 160 (intermediate point between narrow 128 and previously-wider 192) compounds with SmoothL1.
- **Result**: val_avg ≈ 96.4 (+1.5%), 17/18 epochs realized (under-completed cosine at 111 s/epoch vs predicted 118). camber_rc trajectory consistent with width-helping mechanism, but cosine under-completion bites at this trunk size.
- **Verdict**: CLOSED. Reinforces #3478's "schedule completion > raw capacity" finding under SmoothL1. Pure L1 has more headroom at the existing trunk than widening does. Nezuko reassigned to lr_min=1e-5 schedule floor (#3864).
- **Metric artifacts**: from #3804 PR comments

### PR #3588 — tanjiro — Lookahead optimizer (k=5, α=0.5) — CLOSED (stale)

- **Student branch**: `charliepai2i24h1-tanjiro/lookahead-optimizer`
- **Status**: 5+ days stale_wip across 2 baseline shifts (#3478 narrow+bf16, then #3127 SmoothL1). Pod alive but never picked up the work. Closed as stale; reassigned to LLRD (#3865) — the natural per-block-lr follow-up to #3719's global-lr closure.

### PR #3589 — thorfinn — SWA tail (last 3 epochs) — CLOSED (stale)

- **Student branch**: `charliepai2i24h1-thorfinn/swa-tail`
- **Status**: 5+ days stale_wip across 2 baseline shifts. Reassigned with fresh restart on SWA tail (#3866) with cleaner spec on the current SmoothL1 baseline.

**5 new dispatches (all idle students addressed; zero idle GPUs)**:

| PR | Student | Axis | One-line summary |
|---|---|---|---|
| #3861 | askeladd | Loss formulation | Charbonnier `sqrt(eps² + r²) - eps`; eps∈{1e-3, 1e-2}. Tests "is L1's discontinuity at r=0 the mechanism?" |
| #3863 | fern | Optimizer stability | Gradient norm clipping max_norm=1.0 + per-epoch grad-norm telemetry |
| #3864 | nezuko | Schedule shape | CosineAnnealingLR eta_min=1e-5 — directly attack underfit-baseline late-cosine descent |
| #3865 | tanjiro | Per-block lr | LLRD (γ∈{0.9, 0.75}) — natural follow-up to #3719 global-lr closure |
| #3866 | thorfinn | Weight averaging | SWA uniform tail-3 averaging — complementary to EMA |

**Loop 17 systemic findings**:

1. **L1's discontinuity at r=0 is likely the load-bearing mechanism** (not just smaller β): the non-monotonic progression {β=1.0: 94.97, β=0.5: 91.57, β=0.25: 91.81, β=0.0: 86.66} with a 4.91-unit gap at the L1 limit is much larger than the inter-step gap inside the β sweep. Charbonnier (#3861) directly tests this.
2. **Schedule completion is still the binding constraint** on width: #3804 n_hidden=160 reinforces #3478's finding that under-completed cosine bites harder than capacity helps.
3. **Per-channel optima diverge by channel** (#3763 finding): β=0.25 wins on Ux/Uy but β=0.5 wins on pressure on 2/4 splits. Suggests per-channel loss formulation is a future axis worth exploring.
4. **Global gradient-budget reallocation hurts** under L1-tail losses: #3800 surf_p 4× regression confirms that any per-channel reweighting needs to coexist with L1 mechanics, not against them.

## 2026-05-16 — Loop 16 actions (post-SmoothL1 merge)

**Context**: PR #3127 (SmoothL1) merged Loop 15 with new best val_avg=94.97 / test_avg=85.04 (−15.0%). 5 PRs landed for review against the new baseline; all 5 had results measured on the OLD MSE baseline (111.75).

**PR closures (4)**:

### PR #3719 — frieren — lr=5e-4 → 1e-3 (+warmup arm) — CLOSED

- **Student branch**: `charliepai2i24h1-frieren/lr-1e3`
- **Hypothesis**: Baseline at 18 epochs is step-quality limited (loss descending at final cosine-annealed epoch). Doubling effective lr via integral should yield 1–5% improvement.
- **Two-arm result**:
  - Arm 1 (lr=1e-3 no-warm): val_avg=112.69 (+0.85% on OLD 111.75), asymmetric per-split (only camber_rc improved −1.69%)
  - Arm 2 (lr=1e-3 + 375-step linear warmup): val_avg=112.09 (+0.31%), test_avg=99.83. Warmup recovered single_in_dist/cruise but lost camber_rc win.
- **Verdict**: CLOSED — student's own conclusion: "Closes the global-lr-tuning axis."
- **Systemic finding**: A single global lr cannot simultaneously optimize all 4 val splits at this epoch budget. Per-split convergence profiles diverge: camber_rc wants higher lr; single_in_dist/re_rand are at convergence at lr=5e-4; cruise is roughly lr-invariant in [5e-4, 1e-3]. The new SmoothL1 baseline reduced the camber_rc gap from +18.5% to +9.3%, weakening the motivation for further global-lr work.
- **Metric artifacts**: `models/model-lr-1e3-20260516-053453/metrics.jsonl`, `models/model-lr-1e3-warmup-20260516-064253/metrics.jsonl`

### PR #3624 — nezuko — Scale-only LayerNorm (bias=False) — CLOSED

- **Student branch**: `charliepai2i24h1-nezuko/scale-only-layernorm`
- **Hypothesis**: If RMSNorm's matched-epoch gain (#3496) comes from dropping bias terms, scale-only LayerNorm should reproduce most of it.
- **Result (4 independent seeds)**: val_avg mean=115.69 ±0.78 (+2.69% on OLD baseline), test_avg=104.42 ±1.10 (+5.16%). n_params verified at 660,951 (−1,408 from baseline 662,359 = exact 11 LN sites × 128 bias).
- **Verdict**: CLOSED. Regression confirmed across 4 seeds; bias-drop alone is NOT the RMSNorm gain mechanism.
- **Systemic finding (HIGH VALUE)**: Bias-drop closes only ~14% of the RMSNorm matched-epoch gap (#3496). **Mean-subtraction in LayerNorm is the load-bearing piece** (~86% of the gain). Bias term is mildly load-bearing for late-stage fine-tuning. Future norm work should target mean-subtraction removal (RMSNorm proper), not bias removal.
- **Metric artifacts**: 4 `models/model-scale-only-layernorm-2026051*/metrics.jsonl` files

### PR #3585 — fern — Per-domain weight (racecar_tandem 2.0×) — CLOSED

- **Student branch**: `charliepai2i24h1-fern/per-domain-loss-weight`
- **Hypothesis**: Upweighting `racecar_tandem` loss 2.0× should push parameters serving camber_rc generalization (5-way convergent evidence on wider trunk).
- **Result**: val_avg=112.59 (+0.76% on OLD baseline), test_avg=102.77 (+3.49%). camber_rc improved −0.78% (direction confirmed, magnitude small), but cruise regressed +3.91% (the budget squeeze: tandem 30%→46% share crowded cruise 30%→22%).
- **Verdict**: CLOSED. Direction partially confirmed but net-negative; not sending back at 1.5× because (a) SmoothL1 already reduced camber_rc-as-discriminator gap from +18.5% to +9.3% (weakening the mechanism), and (b) the 5-way convergent evidence was from WIDER trunk (n_hidden=192); on narrow+bf16 the camber_rc cluster may be capacity-clipped.
- **Metric artifacts**: `models/model-per-domain-weight-tandem2-20260516-063248/metrics.jsonl`

### PR #3555 — alphonse — Coord jitter σ=0.01 on (x, z) — CLOSED

- **Student branch**: `charliepai2i24h1-alphonse/coord-jitter-sigma01`
- **Hypothesis**: Input-side denoising regularizer forces smoothness for memorized templates; biggest predicted improvement on val_geom_camber_rc.
- **Result**: val_avg=112.64 (+0.80% on OLD baseline), test_avg=102.28 (+2.99%). Direction confirmed on rc (−0.91%) and single_in_dist (−1.20%), but cruise regressed +5.16% (large) and re_rand regressed +1.63%.
- **Verdict**: CLOSED. Cost asymmetry (cruise −5.16% vs rc −0.91%) makes the axis low-EV. SmoothL1's L1 tail already absorbs much of the "force smoothness on memorized templates" mechanism by de-emphasizing large residuals.
- **Metric artifacts**: `models/model-alphonse-coord-jitter-sigma01-20260516-062942/metrics.jsonl`

**Send-back (1)**:

### PR #3676 — edward — slice_num=48 — SENT BACK

- **Student branch**: `charliepai2i24h1-edward/slice-num-48`
- **Hypothesis**: slice_num=64 is over-parameterized at narrow trunk; reducing to 48 should save compute → more cosine-completed epochs.
- **Result (2 seeds on OLD baseline)**: Run A val_avg=105.77 (−5.36%), Run B val_avg=111.76 (+0.012%). Mean −2.67%. Compute savings unambiguous and structural: 98s → 88s/epoch (−10%), 32.95 → 29.84 GB (−9.4%), 21 cosine-annealed epochs in 30.8 min.
- **Verdict**: SENT BACK for rebased re-run on SmoothL1 baseline (94.97) with 3 seeds (42, 123, 7) at 21 epochs. Compute savings + Run A's promise make this a cheap repeat. Also asked edward to add deterministic seeding (since 2-seed variance was 5.99 — too noisy to interpret).
- **Metric artifacts**: `models/model-slice-num-48-20260516-{052338,062231}/metrics.jsonl`

**Dispatches (4 new + 1 send-back = 5 students re-engaged)**:

| PR | Student | Slug | Axis | Why |
|---|---|---|---|---|
| #3676 (sent back) | edward | slice-num-48 | Slice mechanism | 3-seed rebased re-run on SmoothL1 + 21 epochs |
| #3798 | frieren | pure-l1 | Loss formulation | F.l1_loss — limit test of SmoothL1 beta→0 |
| #3800 | fern | surf-p-weight-4x | Per-channel weighting | surf_p 4× inside surface loss — direct metric attack |
| #3802 | alphonse | compile-determinism | Throughput | torch.compile + 23-epoch budget-soak on SmoothL1 underfit |
| #3804 | nezuko | n-hidden-160 | Capacity | Wider trunk on SmoothL1 (width × loss compounding test) |

## 2026-05-15 13:34 — PR #3130: Wider Transolver: n_hidden 128->192, n_head 4->6 — MERGED

- **Student branch**: `charliepai2i24h1-edward/wider-h192-h6`
- **Hypothesis**: Baseline at n_hidden=128 / n_head=4 (~0.65M params) is under-capacity for 74K–242K-node meshes across 3 domains; widening to n_hidden=192 / n_head=6 (dim_head=32 unchanged) should give monotonic capacity gains.
- **Verdict**: MERGED. First measured reference on this advisor branch.

### Results

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p (primary, best)** | **166.5037** (epoch 8) |
| test_avg/mae_surf_p | NaN (cruise test pressure overflowed; other 3 splits avg 166.58) |
| n_params | 1,447,521 (1.45 M) |
| peak_memory_gb | 63.0 / 96 |
| epochs completed | 9 of 50 (cut by `SENPAI_TIMEOUT_MINUTES=30`) |

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 209.69 | 184.18 |
| geom_camber_rc | 177.40 | 169.65 |
| geom_camber_cruise | **126.99** | **NaN** |
| re_rand | 151.93 | 145.91 |
| **avg** | **166.50** | **NaN (166.58 over valid 3)** |

- **Metric artifacts**: `models/model-charliepai2i24h1-edward-wider-h192-h6-20260515-124423/metrics.jsonl`, `metrics.yaml`, `config.yaml`

### Analysis

- val loss decreased monotonically through the available 9-epoch budget (241.75 → 220.13 → 195.00 → 186.74 → 167.31 → 166.50 → 184.61). One bump at epoch 5 (237.13) is consistent with the still-high LR under T_max=50 not annealing.
- The cosine schedule never annealed because `T_max=50` while only 9 epochs completed under the 30-min wall-clock cap. The model is evaluated at near-peak LR rather than after a low-LR fine-tune. This is a systemic issue affecting every round-1 PR; documented in `BASELINE.md`.
- The cruise-test NaN comes from a single non-finite pressure prediction propagating through the unguarded scoring accumulator. The val split for the same domain (`val_geom_camber_cruise`) is fine (in fact the best of the four). Root cause is likely a high-Re cruise test sample pushing pressure logits past float32 range under the still-warm / partially-trained model.
- Param count was 1.45M (not the ~1.6M PR-body estimate). VRAM headroom comfortable.

### Follow-ups queued

- **Edward's next PR (ReScaler / log(Re)-conditioned output scaler).** Directly addresses the cruise NaN AND the predicted per-sample y-std variation that drives high-Re training instability. This is researcher-agent Idea 4. → **dispatched as PR #3273**.
- **Schedule alignment** (set epochs ≈ realized budget, T_max=epochs). Systemic fix; queued for round 2 once we see the other round-1 PRs land.

## 2026-05-15 14:24 — PR #3137: Add EMA model weights (decay 0.999) for eval/test — MERGED

- **Student branch**: `charliepai2i24h1-nezuko/ema-0999`
- **Hypothesis**: Maintaining an exponential moving average of model weights and evaluating with the EMA copy smooths the iterate evaluated; particularly useful here because (a) Transolver has noisy slice-projector weights early in training, and (b) the loss landscape is sharp due to extreme target magnitudes. Predicted 2–5% relative improvement on `val_avg/mae_surf_p`.
- **Verdict**: MERGED. Largest single win on this track so far (~22% relative over #3130 baseline).

### Results

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p (primary, best)** | **129.4217** (epoch 14) |
| test_avg/mae_surf_p | NaN (cruise sample 20 has 761 Inf in GT pressure → scoring bug; other 3 test splits avg 128.44) |
| peak_memory_gb | 42.11 / 96 |
| epochs completed | 14 of 50 (cut by `SENPAI_TIMEOUT_MINUTES=30`) |
| trunk config used by student | n_hidden=128, n_head=4 (pre-#3130) |

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 164.34 | 145.46 |
| geom_camber_rc | 145.10 | 129.24 |
| geom_camber_cruise | **97.32** | **NaN** (corrupt GT) |
| re_rand | 110.93 | 110.61 |
| **avg** | **129.42** | **NaN (128.44 over valid 3)** |

- **Per-epoch val_avg/mae_surf_p trajectory (EMA-evaluated)**: ep1 362.62 → ep2 313.94 → ep3 273.01 → ep4 240.87 → ep5 217.20 → ep6 196.64 → ep7 180.42 → ep8 168.01 → ep9 158.96 → ep10 151.93 → ep11 144.14 → ep12 137.49 → ep13 130.75 → **ep14 129.42** (still strictly monotonically decreasing at wall-clock cutoff).
- **Metric artifacts**: `models/model-ema-0999-20260515-125245/metrics.jsonl`, `metrics.yaml`, `config.yaml`

### Analysis

- **Big win, but the magnitude is suspicious**: EMA was predicted to give 2–5%, we got 22%. Likely a combination of (a) the EMA effect itself smoothing out the noisy near-peak-LR iterate, and (b) the val curve being monotonically decreasing — so any later epoch is better, and EMA-evaluation effectively gives us a smoothed version of the epoch-14 weights vs the raw epoch-14 weights. The student also noted that with `decay=0.999` and ~375 steps/epoch, the EMA's effective averaging window is ~1000 steps — it's still tracking a fast-moving target at the cutoff. Under a budget that actually completes the cosine schedule, the relative EMA gain may be smaller.
- **Cruise-test NaN root cause identified**: Nezuko's diagnostic isolated it precisely. Cruise test sample 20 has 761 `Inf` values in ground-truth `p`. `accumulate_batch` computes `err = (pred - y).abs()` BEFORE masking, then multiplies by `surf_mask`. IEEE-754 says `Inf * 0 = NaN`, so the accumulator goes NaN on the `p` channel specifically (Ux/Uy GT are clean for the same sample). Reproduced deterministically at bs=4, doesn't reproduce at bs=1 (no batching with other samples). **Fix is one-line in `data/scoring.py`**: `err.nan_to_num_(0.0, posinf=0.0, neginf=0.0)` after the subtraction. `data/scoring.py` is marked read-only in `program.md` — requires advisor waiver. Queued.
- **Trust-in-orthogonality stacking**: Student branched from pre-#3130 config (n_hidden=128, n_head=4); squash-merge layered EMA onto the already-merged wider config. The combined `n_hidden=192 + n_head=6 + EMA` config is untested — we trust orthogonality. Documented in BASELINE.md.

### Follow-ups queued

- **Nezuko's next PR (separate per-channel prediction heads).** Researcher-agent Idea 12; addresses channel-scale mismatch (p has 10× the dynamic range of Ux/Uy). Independent of all in-flight round-1 PRs.
- **Scoring bug-fix PR.** One-line `nan_to_num` patch on `data/scoring.py`. Will fix `test_avg/mae_surf_p` for every future result. Needs advisor waiver of read-only — separate from regular experiment PRs.
- **EMA-vs-raw delta logging.** Cheap to add to the trainer (one extra `evaluate_split` per epoch). Would have made the EMA hypothesis test cleaner; folding into a future infra PR.

## 2026-05-15 14:35 — PR #3136: Increase surf_weight 10->25 to bias toward primary metric — MERGED

- **Student branch**: `charliepai2i24h1-frieren/surfw25`
- **Hypothesis**: The primary metric is surface pressure MAE but the trainer uses `loss = vol_loss + 10 * surf_loss`. Surface nodes are only a small fraction of mesh nodes, so even at surf_weight=10 the volumetric loss may dominate the gradient. Raising the weight should make the model prioritize surface accuracy.
- **Verdict**: MERGED. Beat current baseline (#3137 EMA at 129.42) by 2.4% relative.

### Results

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p (primary, best)** | **126.3241** (epoch 14) |
| test_avg/mae_surf_p | NaN (same cruise GT bug; 3-split mean = 123.43) |
| peak_memory_gb | 42.11 / 96 |
| epochs completed | 14 of 50 (cut by `SENPAI_TIMEOUT_MINUTES=30`) |
| trunk config used by student | n_hidden=128, n_head=4 (pre-#3130) |

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 158.79 | 136.67 |
| geom_camber_rc | 127.26 | 117.86 |
| geom_camber_cruise | **102.20** | **NaN** (corrupt GT) |
| re_rand | 117.04 | 115.75 |
| **avg** | **126.32** | **NaN (123.43 over valid 3)** |

- **Metric artifacts**: `models/model-surfw25-20260515-132819/metrics.jsonl`, `metrics.yaml`, `config.yaml`

### Analysis

- Single attributable change applied cleanly. Training stable, no NaN/Inf in val for any of 14 epochs.
- Per-split surface pressure improved on every val split vs the #3137 baseline numbers (most notably val_geom_camber_rc 145.10 → 127.26, -12.3%). This is consistent with the surf_weight knob directly biasing the optimizer toward surface accuracy.
- **Stacking note**: frieren ran with n_hidden=128 (pre-#3130). The squash-merge layers only the surf_weight change onto the merged wider+EMA. Active recipe is now **wider + EMA + surf_weight=25** — a 3-axis stack that has never been measured end-to-end. The first PR based on this new advisor branch will produce the first true measurement; documented in BASELINE.md.
- Same cruise-test NaN as every other PR. Frieren's diagnostic chain (`Inf * 0 = NaN` through the mask) matches nezuko's and tanjiro's independently. The pattern is now well-established and a fix is overdue.

### Follow-ups queued

- **Idea 2 refinement (per-channel p_surf_weight).** With surf_weight=25 now in the baseline, layer a per-channel weight specifically on the p component of surf_loss (e.g. `p_surf_weight=3`). This isolates the channel-level boost without further raising the global surface weight.
- **Schedule alignment for round 2.** The cosine T_max=50 issue continues to affect every run.

## 2026-05-15 14:35 — PR #3273: log(Re)-conditioned output scaler (ReScaler) — SENT BACK

- **Student branch**: `charliepai2i24h1-edward/rescaler-logre`
- **Hypothesis**: Per-sample target std varies by 10× within a single domain (e.g., `val_single_in_dist` y_std range 458/2077). A learned MLP(log_Re) → exp(scale) per-sample output multiplier should let the trunk predict a small normalized residual while the scaler recovers physical magnitude. Predicted 5-12% improvement on val_avg/mae_surf_p, with biggest gain on Re_rand and geom_camber_cruise splits. Predicted side benefit: eliminate the cruise-test NaN.

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best) | 153.3094 (epoch 9 of 9 realized) |
| Δ vs pre-#3137 baseline (166.50) | -7.9% (matches predicted 5-12% range) |
| Δ vs current baseline (#3137, 129.42) | +18.5% (WORSE; no EMA in student's run) |
| test_avg/mae_surf_p | NaN (cruise pressure still overflowed → 3-split mean 154.42) |
| peak_memory_gb | 63.0 / 96 |
| epochs completed | 9 of 50 (cut by 30-min cap) |
| trunk config used by student | n_hidden=192, n_head=6 (post-#3130, pre-#3137) |
| Per-split val Δ vs #3130 | single_in_dist -8.9%, rc -6.1%, cruise -7.2%, re_rand -9.3% |

- **Metric artifacts**: `models/model-rescaler-logre-20260515-135045/metrics.jsonl`, `metrics.yaml`

### Verdict: send back for rebase + tighter bound

- The hypothesis was validated on its own terms (-7.9% vs the baseline that existed when assigned). But the current baseline now includes EMA, and the student didn't see EMA — so 153.31 doesn't beat 129.42.
- The ReScaler approach is independent of EMA, so the combined effect is likely additive. The send-back asks for: (a) rebase onto the post-#3137 advisor branch (wider+EMA+surf_weight=25), (b) tighten `max_log_scale` from 2.0 → 1.0 (the y-std ratio across splits is only ~3×, so exp(±1)=[0.37, 2.72] is plenty), (c) optional `pred.clamp(-50, 50)` defense before scaling to prevent trunk overflow on outlier samples.
- If the rebased rerun shows ReScaler + EMA + surf_weight=25 beats 126.32, we merge.

## 2026-05-15 14:35 — PR #3141: Random Fourier features on position coordinates — SENT BACK

- **Student branch**: `charliepai2i24h1-tanjiro/fourier-pos`
- **Hypothesis**: Position coordinates fed directly through the preprocess MLP suffer from neural-network spectral bias — the network struggles to fit high-frequency surface-pressure variation. Random Fourier features (num_freqs=16, sigma=4.0) on (x, z) before preprocess should let the network express higher-frequency content cleanly. Predicted 5-8% improvement.

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best) | 136.1451 (epoch 14 of 15 realized) |
| Δ vs current baseline (#3137, 129.42) | +5.2% (WORSE) |
| test_avg/mae_surf_p (raw) | NaN |
| **test_avg/mae_surf_p (NaN-safe re-eval)** | **122.9001** |
| peak_memory_gb | 42.33 / 96 |
| epochs completed | 15 of 50 |
| trunk config used by student | n_hidden=128, n_head=4 (pre-#3130) |

- **Metric artifacts**: `models/model-charliepai2i24h1-tanjiro-fourier-pos-20260515-131052/metrics.jsonl`, `test_metrics_clean.json`, `eval_test_clean.py`

### Verdict: send back for rebase + longer-budget rerun

- Val didn't beat current baseline (136.14 vs 129.42), BUT the NaN-safe test re-eval (122.90) is the lowest test number we've ever seen on this branch — and val was still descending at the cutoff (best was epoch 14 of 15 realized).
- The student also delivered a working NaN-safe re-eval (`eval_test_clean.py`) which respects the read-only constraint on `data/scoring.py`. **This pattern is now the recommended standard for reporting test numbers** until the scoring bug is patched.
- Send back asks for: (a) rebase onto post-#3137 advisor (wider+EMA+surf_weight=25), (b) consider a sigma sweep (σ ∈ {2, 4, 8}) since σ=4 may not be optimal at this dataset's spatial frequency content, (c) re-run with longer effective horizon (the cosine schedule was sized for 50 but only ~15 ran — at minimum, request `T_max=epochs` alignment).
- If the rebased rerun shows Fourier + EMA + surf_weight=25 beats 126.32 val_avg, we merge.

## 2026-05-15 16:35 — PR #3134: slice_num 64->128 — CLOSED

- **Student branch**: `charliepai2i24h1-fern/slice-num-128`
- **Hypothesis**: Doubling the number of physics-attention slice tokens from 64 to 128 should give the model finer-grained representation of mesh-local physics regimes, with predicted 3-6% improvement on `val_avg/mae_surf_p`.

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best) | 191.65 (epoch 7 of 7 realized) |
| Δ vs pre-#3130 baseline (#3130 n_hidden=192 = 166.50) | +15% (WORSE) |
| Δ vs current baseline (#3136, 126.32) | +52% (WORSE — much worse) |
| trunk config used by student | n_hidden=128, n_head=4, surf_weight=10 (pre-#3130) |
| epochs completed | 7 of 50 (cut by 30-min cap) |
| Per-epoch wall time | ~4.3 min (≈2× #3130 baseline) |

- **Metric artifacts**: `models/model-charliepai2i24h1-fern-slice-num-128-20260515-132506/metrics.jsonl`

### Verdict: closed

- Doubling `slice_num` doubles the per-epoch cost (~4.3 min/epoch vs ~2.2 min on baseline). Under the 30-min cap, this means only ~7 epochs realize instead of ~14. The capacity gain (more slice tokens) is more than offset by the budget loss (fewer epochs). At fixed wall-clock, finer slice resolution is strictly worse here.
- Fern also independently diagnosed the cruise-test NaN root cause (Inf in `test_geom_camber_cruise/sample_20` GT × scoring-bug `Inf*0=NaN`) — same chain as nezuko #3137, tanjiro #3141, frieren #3136. The diagnostic agreement across 4 independent students is conclusive evidence; the read-only-waiver scoring-fix PR is overdue.
- Closed rather than sent back: the cost-benefit math doesn't work under the 30-min cap. Even with budget alignment (T_max=epochs=7), the floor is the wall-clock budget, not the schedule. Fern reassigned to round-2 `p_surf_weight` (#3298, see below).

## 2026-05-15 16:42 — PR #3127: SmoothL1 (Huber) loss — SENT BACK (strong-but-stale result)

- **Student branch**: `charliepai2i24h1-askeladd/smoothl1-loss`
- **Hypothesis**: MSE-in-normalized-space squares residuals; with per-sample y_std varying 10× within a split, a handful of high-Re samples dominate the gradient. Switching to SmoothL1 (Huber, beta=1.0) — quadratic near zero, linear in the tails — better matches the L1-in-physical-space evaluation metric and de-emphasizes outliers. Predicted 3-8% relative improvement, biggest on `val_re_rand` and `val_geom_camber_cruise`.

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best) | **114.1433** (epoch 13 of 14 realized) |
| test_avg/mae_surf_p (NaN-safe via per-sample y_finite filter) | **102.3205** |
| Δ vs current baseline (#3136, 126.32) | **-9.6%** (BIG win) |
| Δ vs nominal old-config-MSE (#3130, 166.50) | -31% |
| trunk config used by student | n_hidden=128, n_head=4, surf_weight=10, no EMA (PRE-#3130) |
| epochs completed | 14 of 50 (cut by 30-min cap) |

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 129.90 | 112.53 |
| geom_camber_rc | 129.94 | 112.80 |
| geom_camber_cruise | **92.61** | **80.85** |
| re_rand | 104.12 | 103.11 |
| **avg** | **114.14** | **102.32** |

- **Metric artifacts**: `models/model-charliepai2i24h1-askeladd-smoothl1-loss-20260515-140900/metrics.jsonl`

### Verdict: send back for rebase + budget-align

- **The hypothesis is strongly validated**: SmoothL1 vs MSE is clearly a big win on the same architecture — 114.14 vs the closest comparable MSE measurement (#3130 = 166.50, wider but no EMA) is -31%.
- **But the reported 114.14 is on the OLD pre-merge config** (n_hidden=128, n_head=4, surf_weight=10, no EMA — i.e., 4 axes different from current baseline), so it isn't a clean drop-in measurement for the merged stack. GitHub shows `mergeStateStatus: CLEAN` because the merge base is far enough back that nothing conflicts textually, but a hypothetical squash-merge would land SmoothL1 onto wider+EMA+surf_weight=25 — a recipe nobody has measured. The published 114.14 would not reproduce.
- **Notable askeladd contribution**: a per-sample `y_finite` filter in `evaluate_split` (lines 236-249 of their `train.py`) — drops any sample with non-finite GT before accumulator math. Lives in `train.py` (not `data/scoring.py`), so it respects the read-only constraint. Different from tanjiro's mask-before-sum approach (#3141), but functionally equivalent. Either pattern is acceptable.
- Send-back asks for: (a) `git rebase origin/icml-appendix-charlie-pai2i-24h-r1` to inherit wider+EMA+surf_weight=25, (b) keep SmoothL1 and the NaN-safe shim, (c) `--epochs 12` with `T_max=12` for proper cosine annealing, (d) re-report `val_avg/mae_surf_p` and `test_avg/mae_surf_p`.
- **If the rebased rerun beats 126.32**, we merge — and given the magnitude of the gap opened on the old config, this is one of the higher-probability merges queued.

## 2026-05-15 16:50 — PR #3298: per-channel p_surf_weight=3.0 on surface MSE (Idea 2 refinement) — DISPATCHED

- **Student**: charliepai2i24h1-fern (round 2 reassignment after #3134 close)
- **Hypothesis**: Now that surf_weight=25 is in the merged baseline, the surface loss dominates the total objective — but within surface loss all 3 channels (Ux, Uy, p) contribute equally. The primary metric only cares about p. Add a `p_surf_weight=3.0` multiplier inside the surface loss so the p channel gets 3× the gradient weight inside surf_loss (which is then multiplied by surf_weight=25 against vol_loss). Per-channel decomposition is the natural follow-up to frieren's global surf_weight win.
- **Predicted delta**: 2-5% on val_avg/mae_surf_p; should also show that mae_surf_Ux and mae_surf_Uy don't degrade by more than ~2-3%.
- **Schedule**: budget-aligned `epochs=12, T_max=12`.
- Single-attributable axis: introduces a new config field `p_surf_weight: float = 3.0` and replaces the global surface MSE with a per-channel sum where channel index 2 (p) carries the multiplier. At `p_surf_weight=1.0` the formulation collapses to the original MSE, so the change is a clean knob.

## 2026-05-15 17:00 — PR #3273: log(Re)-conditioned output scaler (ReScaler) rebased rerun — CLOSED

- **Student branch**: `charliepai2i24h1-edward/rescaler-logre` (rebased on advisor branch with `max_log_scale=2.0→1.0`, pre-scale `clamp(±50)`, schedule `epochs=8, T_max=8`)
- **Hypothesis (rebased rerun)**: ReScaler with tightened bound + clamp defense, layered onto the merged wider+EMA+surf_w25 stack, should beat 126.32. Schedule budget-aligned for the realized 8-epoch horizon.

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, epoch 8 of 8 realized) | **152.7870** |
| test_avg/mae_surf_p (NaN-safe re-eval) | 138.7311 |
| Δ vs current baseline (#3136, 126.32) | **+20.9% (WORSE)** |
| Δ vs #3136 at matched epoch 8 | -2.7% (BETTER at matched-epoch) |
| trunk config used by student | wider+EMA+surf_w25+ReScaler (post-#3136 rebase, correctly inherited) |
| epochs completed | 8 of 8 (cosine T_max=8 fully annealed) |
| Per-epoch wallclock | ~205 s (wider trunk) |

| Split | val mae_surf_p (best) | test mae_surf_p (NaN-safe) |
|---|---|---|
| single_in_dist | 187.40 | 162.54 |
| geom_camber_rc | 169.25 | 152.58 |
| geom_camber_cruise | 118.44 | **101.80** (cruise sample 20 skipped) |
| re_rand | 136.05 | 137.99 |
| **avg** | **152.79** | **138.73** |

- **Metric artifacts**: `models/model-rescaler-logre-tight-20260515-154035/metrics.jsonl` + `metrics.yaml`

### Verdict: closed — schedule/budget tradeoff dominates

- The ReScaler hypothesis is **technically validated**: at matched epoch 8, ReScaler+wider+EMA+surf_w25 (152.79) beats no-ReScaler+narrow+EMA+surf_w25 (#3136 at epoch 8 ≈ 157.02) by ~3% — but the trunk axis differs, so the comparison isn't perfectly apples-to-apples.
- **The absolute number doesn't beat baseline** (-21% regression vs 126.32). Per CLAUDE.md decision criteria (>5% regression → close), this is a clear close.
- **Edward's analysis identified the real blocker**: at n_hidden=192, only 8 epochs fit in the 30-min cap; the narrower trunk fit 14 epochs and got further along the val descent curve, even with a partially-annealed cosine. The wider trunk's capacity advantage cannot manifest within the current per-epoch wallclock budget. **This is now the binding systemic constraint.**
- **Useful artifacts produced**: `ReScaler` module (163 params, clean implementation), `max_log_scale=1.0` + `pre_scale clamp(±50)` defense, and `evaluate_split_nan_safe()` — the third NaN-safe re-eval pattern on this track (alongside tanjiro's `eval_test_clean.py` and askeladd's `y_finite` filter).
- **Edward reassigned to bf16 mixed-precision** (#3332) — directly addresses the per-epoch wallclock budget that ReScaler couldn't overcome.

## 2026-05-15 17:05 — PR #3332: bf16 mixed-precision training — DISPATCHED

- **Student**: charliepai2i24h1-edward (round 2 reassignment after #3273 close)
- **Hypothesis**: The dominant bottleneck is now schedule/budget alignment at the wider trunk (identified directly by edward's #3273 analysis). bf16 autocast on forward + loss is expected to deliver ~30-40% per-epoch speedup on the 96GB GPU, unlocking ~12-13 epochs in the 30-min cap (vs 8 at fp32). Combined with budget-aligned scheduling (`epochs=12, T_max=12`), this finally lets the wider trunk realize its capacity advantage.
- **Predicted delta on val_avg**: 2-5% better than 126.32 baseline (i.e., 119-123), with the gain attributable to (a) wider trunk capacity finally usable, (b) cosine fully annealing through 12 epochs.
- Single attributable: wrap forward + loss in `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)` in both the training loop and `evaluate_split`. No GradScaler (bf16 has the dynamic range), no parameter conversion, no batch_size change. EMA stays fp32 by default since autocast doesn't affect parameter storage.
- Critical instrumentation requested: **per-epoch wallclock** (the directly-attributable measurement) and **epochs realized in 30-min cap** (the lever this PR is pulling). If bf16 speedup is <20%, that's an important systems-level finding informing whether the next throughput unlock should be `batch_size=8` instead.

## 2026-05-15 17:10 — PR #3298: per-channel p_surf_weight=3.0 — CLOSED

- **Student branch**: `charliepai2i24h1-fern/p-surf-weight`
- **Hypothesis (rerun)**: p_surf_weight=3.0 multiplier on the p channel inside surface MSE; identity-at-1.0 knob; budget-aligned epochs=12.

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, epoch 9 of 12) | **158.5363** |
| test_avg/mae_surf_p (NaN-safe via tanjiro's `eval_test_clean.py` pattern) | 144.4903 |
| Δ vs current baseline (#3136, 126.32) | **+25.5% (WORSE)** |
| Δ at matched epoch 9 vs no-p_surf_weight matched-epoch | -0.3% (essentially neutral, 158.54 vs 158.96) |
| trunk config used by student | wider+EMA+surf_w25+p_surf_weight=3.0 (correctly inherited from advisor) |
| epochs completed | 9 of 12 (cut by 30-min cap, schedule annealed faster due to T_max=12) |
| Per-epoch wallclock | ~203 s (wider trunk; same as edward #3273) |

| Channel | This run (epoch 9) | Baseline (epoch 14) |
|---|---:|---:|
| val_avg/mae_surf_p | 158.5363 | 126.3241 |
| val_avg/mae_surf_Ux | 3.0786 | 1.9848 |
| val_avg/mae_surf_Uy | 1.1146 | 0.8943 |

- **Metric artifacts**: `models/model-p-surf-weight-20260515-153455/metrics.jsonl` + `metrics.yaml` + `test_metrics_clean.json`

### Verdict: closed — same schedule/budget wall as #3273

- **Same systemic pattern as edward's #3273**: wider trunk fits only 9 epochs vs baseline's 14, and the matched-epoch comparison shows the hypothesis-attributable effect is approximately **neutral** (158.54 vs 158.96 = -0.3% at epoch 9). The +25% regression in absolute terms is dominated by the schedule/budget tradeoff, not by the p-channel weighting itself.
- Fern's matched-epoch comparison was the cleanest analysis of this kind on the track to date — exactly the right diagnostic in a budget-constrained regime.
- **Per-channel surface reweighting is approximately neutral at this scale** — the test was clean enough that we can call this. The hypothesis-attributable effect would have to come from epochs 10+ where p_surf_weight=3.0 starts diverging from baseline, but there's no strong reason to expect this if it's neutral at epoch 9.
- Could be revisited after bf16 (edward #3332) lands and unlocks the 12-13 epoch budget.
- **Fern reassigned to gradient clipping (#3336)** — orthogonal to all in-flight, single-line change, doesn't require full schedule annealing to show signal.

## 2026-05-15 17:15 — PR #3336: Global gradient norm clipping (max_norm=1.0) — DISPATCHED

- **Student**: charliepai2i24h1-fern (round 2 reassignment after #3298 close)
- **Hypothesis**: With `surf_weight=25` and per-sample y_std varying 10× within a single split, high-Re samples produce gradient spikes amplified twice (by surf_weight and by the squared-error tail). Global `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` bounds the per-step update magnitude, damping high-Re gradient spikes without changing the average update direction.
- **Predicted delta**: modest (2-5%) on val_avg/mae_surf_p, with largest effect on val_re_rand (the stratified Re holdout). Per-step effect — does NOT require full schedule annealing to validate.
- **Schedule**: budget-aligned `--epochs 12 --T_max=12` (matching the post-#3136 rerun policy).
- Single attributable: one line inserted between `loss.backward()` and `optimizer.step()`. Instrumentation: track `train/grad_norm_mean`, `_max`, `_min` per epoch — primary diagnostic for whether clipping is actually firing.
- Designed to be orthogonal to bf16, separate-heads, FiLM, SmoothL1, warmup, Fourier, and deeper — none of those control gradient norm.

## 2026-05-15 17:40 — PR #3144: Deeper Transolver: n_layers 5->8 (rebased) — CLOSED

- **Student branch**: `charliepai2i24h1-thorfinn/deeper-l8`
- **Hypothesis**: At the merged wider trunk, n_layers 5→8 should deliver capacity gains (more transformer block depth).
- **Verdict**: CLOSED. val_avg/mae_surf_p = **177.6026** (epoch 6) vs baseline 126.32 = **+40.6% regression**, well above the 5% close threshold. **Same wider-trunk-budget wall as #3273 and #3298** — third independent diagnosis of the pattern.

### Results

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p (primary, best)** | **177.6026** (epoch 6 of 8 realized) |
| test_avg/mae_surf_p | NaN (cruise scoring bug); 3-split finite mean = 177.1244 |
| n_params | 2,242,995 (2.24M — more than doubled from 1.03M baseline) |
| peak_memory_gb | **96.61 / 96** (at the hardware ceiling) |
| Per-epoch wallclock | ~317 s (~5.3 min/epoch — vs ~205s baseline) |
| epochs realized | 6 of 8 (still mid-descent: 193→177 between epochs 5→6) |

### Verdict commentary

- **The capacity gain isn't testable at current budget**: wider × deeper drove per-step cost +55% AND pushed VRAM to 96.61/96 GB ceiling. The model is clearly still mid-descent at termination (8% val improvement between epochs 5→6).
- **Thorfinn's analysis was excellent**: correctly identified depth=8 is gated on bf16 (or bs=2, or both) landing first. After bf16 lands and unlocks ~30% more budget, this hypothesis becomes testable as part of a depth sweep (5/6/7/8).
- Per-channel: mae_surf_Ux=3.83, mae_surf_Uy=1.10 — not damaged, consistent with baseline rails (3.65 / 1.04). The pressure channel just isn't getting enough training.
- **Thorfinn reassigned to scoring-fix PR (#3378)** — system-fix with advisor waiver to edit `data/scoring.py`. Unblocks `test_avg/mae_surf_p` reporting paper-wide.

## 2026-05-15 18:20 — PR #3332: bf16 mixed-precision (autocast forward + loss) — SENT BACK

- **Student branch**: `charliepai2i24h1-edward/bf16-amp`
- **Hypothesis**: Wrap forward + loss in `torch.amp.autocast(dtype=torch.bfloat16)`; ~30-40% per-epoch speedup → 12-13 epochs realized at wider trunk; wider trunk's capacity manifests with the extra epochs.
- **Verdict**: SENT BACK. The infrastructure unlock is clean and works as predicted, but val_avg/mae_surf_p = 129.76 is +2.7% over baseline 126.32 — doesn't pass the merge-must-improve-primary rule. Sent back with bs=8 + epochs=22 retry recipe.

### Results

| Metric | Value (bf16, 12 epochs) | Baseline #3136 |
|---|---|---|
| **val_avg/mae_surf_p (primary, best)** | **129.7591** (epoch 12) | 126.3241 (epoch 14) → **+2.7% regression** |
| test_avg/mae_surf_p | **117.7643** (full 4-split, NaN-safe) | NaN (3-split partial 123.43) → **+5.7 better, full coverage** |
| Per-epoch wallclock | **143.7s** (~30% speedup) | 205s (fp32 wider) → 30% landed (low end of 30-40% prediction) |
| epochs realized in 30-min cap | **12** | 8 (wider fp32) → +4 epochs |
| peak_memory_gb | **49.24** / 96 | (vs 63 fp32) → ~22% memory savings, lots of headroom for bs=8 |
| nonfinite_loss_count | 0 | n/a |

### Verdict commentary

- **bf16 itself is fully clean**: zero NaN/Inf events across 12 epochs and 4 test splits. Per-channel ratios (Ux=2.12, Uy=0.86) intact, no precision damage.
- **Val regression is concentrated** on `val_geom_camber_rc` (+12.3 vs baseline; other 3 splits move <1 point). Wider trunk still mid-descent on the geometry-unseen split at 12 epochs.
- **30% speedup confirmed as predicted** — peer-epoch 205s → 143.7s. Memory-bandwidth-bound parts absorbed some of the GEMM/conv speedup (compute would have given 40%).
- **NaN-safe eval working**: all 4 test splits finite, test_avg covers all 4 splits (vs baseline's 3-split partial). This pattern is now standardized via thorfinn's scoring-fix PR (#3378).
- **batch_size=8 is the natural next step**: peak memory 49 GB / 96 GB means there's a clean 2× headroom. Doubling batch at bf16 → ~75-85s/epoch (memory-bandwidth-bound so not full 2×) → 21-24 epochs realized.
- Sent back with: bs=8, epochs=22, T_max=22, keep bf16 plumbing, keep lr=5e-4 (no scaling for now to keep bs axis attributable).

## 2026-05-15 18:30 — PR #3378: NaN-safe scoring fix (torch.where, advisor waiver) — DISPATCHED

- **Student**: charliepai2i24h1-thorfinn (round 2 reassignment after #3144 close)
- **Hypothesis**: `data/scoring.py::accumulate_batch` uses `err * mask.double()` which propagates Inf×0=NaN when GT contains Inf (cruise sample 20). Replace with `torch.where(mask, err, 0)` — preserves documented per-sample-skip semantic, eliminates NaN propagation. **Not expected to change val** (val GT is finite); **expected to change test** from NaN to a finite 4-split mean.
- **Advisor waiver**: explicit permission to edit `data/scoring.py` (normally a protected file). Scoped to `accumulate_batch` only, one-spot substitution. Adds unit test `tests/test_scoring_nan_safe.py` to verify Inf-in-GT doesn't propagate NaN.
- **Project value**: unblocks `test_avg/mae_surf_p` reporting paper-wide. Edward's `evaluate_split_nan_safe` (in #3332) and tanjiro's `eval_test_clean.py` (#3141) are workarounds; this is the source-level fix.
- 5 students have independently diagnosed this issue (nezuko, tanjiro, frieren, fern, edward). Three NaN-safe patterns established as workarounds; tanjiro's element-mask is the cleanest and what this PR adopts.

## 2026-05-15 18:41 — PR #3277: Separate per-channel prediction heads — CLOSED

- **Student branch**: `charliepai2i24h1-nezuko/separate-heads`
- **Hypothesis**: Three independent `Linear→GELU→Linear` heads (one per Ux/Uy/p channel) on the shared trunk can specialize per-channel non-linearities, addressing the channel-scale-mismatch problem. Total parameter increase predicted to be tiny.
- **Verdict**: CLOSED. val_avg/mae_surf_p = **148.83** (epoch 9 of 9 realized) vs baseline 129.42 = **+15% regression**. Wider-trunk-budget pattern again — though the dominant cost driver here was activation traffic (three [B, N, 384] hidden states), not parameter count.

### Results

| Metric | Value | Baseline |
|---|---|---|
| **val_avg/mae_surf_p (primary, best @ ep9)** | **148.8304** | 129.42 → +15% regression |
| test_avg/mae_surf_p | NaN (cruise bug); 3-split partial = 149.92 | (baseline partial 128.44) |
| Per-epoch wallclock | ~218 s (~75% slower than 125s baseline) | n/a |
| n_params | 1,633,377 (+12.8% over 1.45M) | 1,447,521 |
| peak_memory_gb | **71.21** | 42.11 |
| epochs realized in 30-min cap | 9 | 14 |

### Verdict commentary

- **Memory-bandwidth issue, not parameter-count issue**: +12.8% params → +75% per-epoch wallclock. Three [B, N=242K, 384] hidden states at mlp_ratio=2 carry ~3× the activation traffic of the baseline's single [B, N, 192] head. The architecture works in principle but doesn't fit the compute budget.
- **Strict monotonic descent at termination**: every epoch was a new best (338→285→248→218→198→182→168→157→149); model is far from converged at the 30-min cap.
- **Nezuko's analysis is excellent**: correctly identified the +75% slowdown vs +12.8% param change as the dominant signal. Three of four suggested follow-ups are sound (mlp_ratio=1 cheaper heads, matched-epoch confirmation, channel-wise LR/scale). One (per-channel scalar gain) is mathematically redundant.
- **Nezuko reassigned to LayerScale (#3404)** — orthogonal architectural change with ~1920 added params and zero compute overhead. Tests the "cold-start" hypothesis directly.

## 2026-05-15 19:20 — PR #3404: LayerScale residual gating (CaIT) — DISPATCHED

- **Student**: charliepai2i24h1-nezuko (round 2 reassignment after #3277 close)
- **Hypothesis**: At budget-limited training (12 epochs, monotonically descending at termination), the cold-start problem dominates. LayerScale (Touvron et al. CaIT 2021) adds a learnable per-channel multiplicative gate on each residual branch, initialized at `1e-4`, so each TransolverBlock contributes ≈0 to the residual stream at init. Training learns to scale up each block's contribution as evidence accumulates — natural curriculum across depth.
- **Predicted delta**: 2-5% on val_avg/mae_surf_p, with the biggest effect on early-epoch val (faster initial descent translates to better final number at the 30-min cap).
- **Cost**: 1920 added params (+0.13% over 1.45M), zero per-step compute overhead (element-wise multiplication on residual).
- **Single attributable**: insert `LayerScale(dim=192, init_value=1e-4)` on both residual branches of `TransolverBlock`. No other changes — same trunk config, EMA, surf_weight, optimizer, schedule.
- **Diagnostic to track**: per-block `ls_attn.gamma.mean()` and `ls_mlp.gamma.mean()` over training. Cold-start hypothesis predicts late blocks (closer to readout) grow gamma first; early blocks come online later.

## 2026-05-15 20:25 — PR #3378: NaN-safe `data/scoring.py` fix — MERGED (system fix)

- **Student branch**: `charliepai2i24h1-thorfinn/scoring-nan-safe`
- **Hypothesis**: Bug fix with advisor waiver. `data/scoring.py::accumulate_batch` computes `err = (pred - y).abs()` before masking, then multiplies by mask. When GT contains Inf (cruise test sample 20 has 761 Inf values in `p`), IEEE-754 `Inf × 0 = NaN` poisons the accumulator. Replace element-wise product with `torch.where(mask, err, 0)`. Mathematically identical on finite GT; never reads `err` where mask=False on Inf GT.
- **Verdict**: MERGED. System fix that doesn't change val_avg (validation has no Inf, behavior identical). Unblocks `test_avg/mae_surf_p` finite reporting paper-wide.

### Results

| Metric | Value | Baseline |
|---|---|---|
| **val_avg/mae_surf_p (primary, best @ ep9)** | **148.4154** | 126.32 (gap is schedule effect, not fix attribution) |
| **test_avg/mae_surf_p (NOW FINITE, 4-split)** | **136.0053** | NaN (was 3-split partial 123.43) |
| test_geom_camber_cruise.mae_surf_p | **97.4897** | NaN (previously broken under bug) |
| Unit test (`tests/test_scoring_nan_safe.py`) | PASS | n/a |
| epochs realized | 9 of 12 | n/a |
| peak_memory_gb | 63.00 | 42.11 |

### Verdict commentary

- **Val gap is purely schedule-effect**: 148.42 (9 ep, T_max=12) matches the schedule-truncation trajectory of every other budget-aligned wider-trunk run. Validation GT is finite, so the fix is bytewise identical there.
- **Test reporting unblocked**: cruise term goes from NaN → 97.49 (lower than the other three splits, consistent with that being the "easier" cruise regime per `program.md`). All future PRs on this track now report 4-split finite test_avg directly from `metrics.jsonl`.
- **Unit test catches the failure mode**: passes with NaN-safe code, fails with old code on Inf GT input.
- **Acknowledged side note**: `train.py`'s eval-loss aggregation (separate from scoring) has the same Inf×0=NaN pattern. Doesn't affect paper-facing MAE metric. Logged for a future one-spot patch.
- **Excellent PR discipline**: kept the change atomic (only `accumulate_batch`, no `aggregate_splits` or `finalize_split` touched), flagged the residual issue without expanding scope.

## 2026-05-15 20:30 — PR #3336: Gradient norm clipping `max_norm=1.0` — SENT BACK

- **Student branch**: `charliepai2i24h1-fern/grad-clip-1p0`
- **Hypothesis**: Per-batch `clip_grad_norm_(max_norm=1.0)` would damp rare high-Re gradient spikes that drive instability in re_rand and cruise splits while leaving median batches alone. Predicted 2-5% improvement.
- **Verdict**: SENT BACK with max_norm sweep recipe. Direction signal correct but `max_norm=1.0` was 100× too aggressive.

### Results

| Metric | Value | Baseline |
|---|---|---|
| **val_avg/mae_surf_p (best @ ep9 of 12)** | **129.7342** | 126.32 → +2.7% |
| **test_avg/mae_surf_p (NaN-safe clean, 4-split)** | **116.5846** | (was NaN under bug; clean 3-split=129.08) |
| val_re_rand | 111.84 | 117.04 → **-4.4%** (improved) |
| val_geom_camber_cruise | 95.64 | 102.20 → **-6.4%** (improved) |
| val_geom_camber_rc | 149.45 | 127.26 → +17.4% (regressed) |
| val_single_in_dist | 162.01 | 158.79 → +2.0% |
| Per-epoch wallclock | ~204s | n/a |

### Diagnostic punchline

Per-epoch pre-clip gradient L2 norms:
- min: 4-20, mean: 100-200, max: 600-1900
- **clip_rate = 1.00 EVERY EPOCH** — every batch hit the clip threshold
- `max_norm=1.0` renormalized every gradient to a unit vector; optimizer saw direction only, never magnitude

### Send-back recipe

- **Single arm**: `max_norm=100.0` (not a 4-way sweep). Based on observed distribution, this should clip ~5-20% of batches.
- **Rebase** onto current advisor branch to pick up #3378 (no more eval_test_clean.py needed)
- **Keep grad_norm logging permanently** — it's the diagnostic that made this PR valuable
- Same recipe otherwise: budget-aligned epochs=12, T_max=12, n_hidden=192, EMA=0.999, surf_weight=25

### Verdict commentary

- **Direction signal is real**: re_rand and camber_cruise (high-Re spike-prone splits) improved exactly as the hypothesis predicted. The other two splits regressed due to median-batch info loss.
- **`max_norm=1.0` is not the regime the hypothesis lives in**: clipping every batch is not "damp the spikes" — it's "kill all magnitude signal". The 100× scaling fix should isolate the spike-damping benefit.
- **Grad-norm logging is the killer feature**: every future PR should expose `train/grad_norm_{mean,max,min,clip_rate}` per epoch. Cheap, gives mechanism-level insight on what optimization looks like.

## 2026-05-15 20:35 — PR #3121: Linear warmup + cosine annealing (rebased, budget-aligned) — CLOSED

- **Student branch**: `charliepai2i24h1-alphonse/warmup-cosine-12ep`
- **Hypothesis**: 2-epoch linear warmup from 5e-7 → 5e-4 followed by cosine annealing (T_max=10) over remaining epochs would stabilize early training and improve val_avg by 3-8% relative.
- **Verdict**: CLOSED. val=158.75 vs baseline 126.32 = **+25.7% regression**. Schedule axis exhausted under 30-min cap.

### Results

| Metric | Value | Baseline |
|---|---|---|
| **val_avg/mae_surf_p (best @ ep9 of 12)** | **158.7495** | 126.32 → +25.7% |
| **test_avg/mae_surf_p (NaN-safe clean, 4-split)** | **143.8544** | (was NaN under bug; clean 3-split = 123.43) |
| val_single_in_dist | 203.45 | 158.79 → +28.1% |
| val_geom_camber_rc | 172.78 | 127.26 → +35.8% |
| val_geom_camber_cruise | 121.21 | 102.20 → +18.6% |
| val_re_rand | 137.56 | 117.04 → +17.5% |
| Realized epochs | 9 of 12 | 14 of 50 |
| Peak memory | 62.97 GB | 42.11 GB |

### Verdict commentary

- **Uniform regression across splits** (17-36%) — consistent with a uniform schedule cost, not a per-split feature interaction
- **Epoch 1 burned at lr=5e-7** (start_factor=1e-3) — 1/9 of realized budget wasted on near-zero LR
- **Cosine never reached floor**: at epoch 9 LR was still 35% of peak (1.73e-4); the "low-lr fine-tune at end of cosine" never happened
- **Val curve still decreasing at -10/epoch** when timeout hit — model is in rapid-improvement phase; burning an epoch on warmup is net-negative at this budget
- **Three-way confirmation of schedule-axis exhaustion**: combined with #3273/#3298/#3144 (wider-trunk-budget wall) and #3277 (memory-bandwidth wall), schedule perturbations cannot improve val_avg in the current regime
- **Student's analysis is correct and articulate**: "the real lever is the budget, not the schedule" — exactly right. Schedule axis goes back on the shelf until bf16+bs=8 unlocks 21-24 epochs of realized training.

### Follow-up

- Alphonse reassigned to a fresh non-schedule axis (not a warmup-variant; the hypothesis class itself is closed at this budget)

## 2026-05-15 21:25 — PR #3287: Domain-conditional FiLM (gap+AoA → scale/shift on LayerNorm) — CLOSED

- **Student branch**: `charliepai2i24h1-frieren/domain-film`
- **Hypothesis**: Three geometric regimes (single-foil, racecar-tandem, cruise-tandem) produce qualitatively distinct pressure fields. Per-block FiLM that emits `(scale, shift)` from `(gap, AoA1)` should give the model an explicit per-sample regime conditioner without requiring it to discover it implicitly. Predicted disproportionate gains on `val_geom_camber_rc` and `val_single_in_dist`.
- **Verdict**: CLOSED. Clear negative result with thorough mechanism analysis. Hypothesis disconfirmed by the *predicted-to-improve splits being the most regressed*.

### Results (canonical run, EMA-evaluated best-val checkpoint)

| Metric | FiLM (#3287) | Baseline (post-#3136) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p (primary)** | **145.9856** | 126.3241 | **+15.6%** |
| test_avg/mae_surf_p (NaN-safe 4-split) | 138.3074 | n/a (3-split partial 123.43 pre-#3378) | n/a |
| n_params | 1,572,513 | 1,447,521 | +8.6% |
| peak_memory_gb | 71.20 | 42.11 | +69% |
| epochs realized | 9 of 12 | 14 of 50 | -36% |
| Per-epoch wall time | 223.6 s | 128 s | +75% |
| NaN/Inf events | 0 | 0 | — |

### Per-val-split mae_surf_p (best epoch=9)

| Split | FiLM | Baseline | Δ | Hypothesis prediction |
|---|---:|---:|---:|---|
| val_single_in_dist | 184.65 | 158.79 | **+16.3%** | predicted improvement |
| val_geom_camber_rc | 157.07 | 127.26 | **+23.4%** | predicted **largest** improvement |
| val_geom_camber_cruise | 112.96 | 102.20 | +10.5% | secondary improvement |
| val_re_rand | 129.26 | 117.04 | +10.4% | secondary improvement |

- **Metric artifacts**: `models/model-charliepai2i24h1-frieren-domain-film-canonical-20260515-192839/{metrics.jsonl,metrics.yaml}`

### Analysis

- **The predicted-to-improve splits are the most degraded.** This is the strongest possible disconfirmation: if `(gap, AoA1)` is the right conditioner, raceCar tandem should improve, not regress 23.4%. Hypothesis is not budget-starved — it is wrong about which mesh features actually matter.
- **Dual failure mode** (student's diagnosis was correct):
  1. **Memory-induced wall-clock slowdown**: 5 blocks × 2 LayerNorms × `[4, ~200K, 192]` ≈ 6 GB of additional autograd-retained activations → peak 71 GB → 74% GPU utilization → per-epoch +75% slower. Under 30-min cap, lost 5 realized epochs.
  2. **2-D conditioner too coarse** for the multi-modal regime. `gap=0` and AoA ranges overlap between single/tandem in ways that the cleanest taxonomy doesn't capture.
- **Implementation quality**: zero NaN/Inf, identity-init verified, schedule-aligned (T_max=12), feature indices verified against `program.md` (no fallback path).

### Verdict commentary

- This is a **methodologically valuable negative result** — the disconfirmation analysis (predicted splits most degraded) is exactly what a careful reviewer would value. It rules out the *family* of low-dimensional per-sample conditioners, not just this specific implementation.
- Student's suggested follow-ups (input-only FiLM injection, 7-scalar richer conditioner, LayerNorm-affine-fused FiLM) are noted but represent fresh hypothesis classes; not pursued in this PR.
- The high-quality scaffolding (clean numerics, 4-way sweep, exact `program.md` feature indexing) leaves room for a fresh axis without inheriting any negative-result baggage.

### Follow-up

- Frieren reassigned to **DropPath (stochastic depth)** — a regularization axis, orthogonal to all conditioning work, zero param overhead, zero inference cost, compatible with LayerScale (#3404 nezuko in-flight). Different mechanism class than FiLM (regularization vs feature conditioning).

## 2026-05-15 22:30 — PR #3332: bf16 + bs=8 retry — CLOSED

- **Student branch**: `charliepai2i24h1-edward/bf16-amp`
- **Hypothesis (retry arm)**: With peak memory 49 GB / 96 GB at bs=4, bs=8 should give 1.7-1.9× per-epoch speedup → 21-24 epochs realized at the wider trunk. Combined with cosine T_max=22, would unlock the first wider-trunk run with fully-annealed cosine.
- **Verdict**: CLOSED. bs=8 axis cleanly refuted — Transolver is memory-bandwidth-bound, not compute-bound. bs=8 was ~10% **slower** per-epoch and hit the 96 GB HBM ceiling (98.42 GB peak with `expandable_segments`).

### Results (two arms in one PR)

| Arm | val_avg/mae_surf_p | test_avg (NaN-safe 4-split) | epochs realized | s/epoch | peak_GB |
|---|---:|---:|---:|---:|---:|
| **bf16-bs8 (latest, T_max=22)** | **187.83** | 172.03 | 12/22 | 158.9 | **98.42** |
| bf16-bs8 (1st relaunch, T_max=22) | 178.45 | — | 12/22 | 160.6 | 98+ |
| bf16-bs4 (prior arm, T_max=12) | 129.76 | 117.76 | 12/12 | 143.7 | 49.24 |
| Baseline #3136 (fp32 narrow, T_max=50) | 126.32 | NaN (3-split=123.43 pre-fix) | 14/50 | ~205 (wider fp32) | 42.11 |

- **Metric artifacts**: `models/model-bf16-bs8-20260515-{192740,202644}/{metrics.jsonl,metrics.yaml}`

### Mechanism findings

1. **bs=8 is slower than bs=4** on this model. Per-batch time 382 ms (bs=4) → 846 ms (bs=8) — doubling the batch more than doubled per-batch time. The Transolver is **memory-bandwidth-bound**, so the bs=8 GEMM density gain doesn't compensate for the doubled memory traffic.
2. **Peak memory jumps to 98 GB at bs=8**, hitting the HBM ceiling and forcing `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid OOM. bs=12/16 would OOM hard.
3. **Schedule misalignment compounds the regression**: T_max=22 with 12 realized epochs leaves LR at ~30% of peak at "best epoch" — cosine never anneals. All 4 val splits regress substantially (single_in_dist 285 vs baseline 159, geom_camber_rc 197 vs 127, etc.).
4. **bf16 plumbing remains clean**: zero NaN/Inf, channel ratios well-behaved. Not a precision damage event; purely a schedule + batch-size mismatch.

### Verdict commentary

- **bs=8 axis is exhausted.** Memory-bandwidth bound + HBM ceiling = no future for batch-size throughput unlocks on this stack. bs=12/16 OOMs; bs=8 already underperforms bs=4.
- **bf16 alone (the prior arm) was +2.7% on val** — close but not winning. Not enough to merge bf16 as standalone infrastructure.
- **Stop/pivot criterion fires**: BASELINE.md stacking note documented that if bf16+bs=8 retry still doesn't beat 126.32, the wider-trunk thesis itself should be tested by reverting to narrow+bf16 (~18-20 epochs in cap). That test gets dispatched as the next PR.
- **Student's analysis was excellent** — per-batch-time breakdown (382→846 ms) was the cleanest possible diagnostic. The 10-point run-to-run noise on bs=8 (178.45 vs 187.83) is concerning but not blocking the close decision.

### Follow-up

- Edward reassigned to **narrow trunk (n_h=128, n_head=4) + bf16** (PR #3478) — directly tests the stop/pivot criterion. If narrow+bf16 at 18-20 epochs beats 126.32, we revert the wider-trunk merge (#3130). If it ties, width is a wash and we keep the merged wider config. If narrow+bf16 regresses, the wider thesis is confirmed and we need throughput unlocks (torch.compile) as the next axis.

## 2026-05-15 22:32 — PR #3127 askeladd SmoothL1 rerun: NUDGE (no rebased commits in 7h)

- **Student branch**: `charliepai2i24h1-askeladd/smoothl1-loss`
- **Status**: still on `status:wip`. Original result (val=114.14, big -9.6% improvement) was on pre-merge config (pre-#3130 narrow, pre-#3136 surf_weight=10, pre-#3137 no EMA, pre-#3378 NaN-safe). Sent back at 15:33 UTC for rebase + rerun on current advisor stack. No new commits or training output since.
- **Action**: explicit advisor reminder posted. Confirmed exact `train.py` command and emphasis on canonical `beta=1.0` arm. Pod is heartbeating but Claude sessions are short — likely waiting on a clarifying instruction.

## 2026-05-15 22:33 — PR #3404 nezuko LayerScale: NUDGE (no commits in 3h)

- **Student branch**: `charliepai2i24h1-nezuko/layerscale-residual`
- **Status**: assigned 19:25 UTC. Only one commit on the branch (the assignment commit `a605cf6a`). No comments, no metrics, no training output. Pod started seeing the assignment at iteration 90 (22:22 UTC) but training hadn't begun.
- **Action**: explicit advisor reminder posted with reiteration of LayerScale module signature (per-channel `gamma`, init=1e-4, +1,920 params), forward wrapping pattern, and exact `train.py` command. Pod monitoring will catch the next iteration's status.

## 2026-05-15 22:38 — PR #3404: LayerScale residual gating (CaIT init=1e-4) — CLOSED

- **Student branch**: `charliepai2i24h1-nezuko/layerscale-residual`
- **Hypothesis**: CaIT-style learnable per-channel scalar gate on each residual branch (init=1e-4) should act as a cold-start curriculum — blocks contribute ~0 at init, training ramps them up, with late blocks predicted to activate first. +1,920 params, zero compute overhead.

### Results

| Metric | Value | Baseline (#3136) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p (best @ ep9/12)** | **135.1100** | 126.3241 | **+6.9% (WORSE)** |
| epochs realized | 9 of 12 | 14 of 50 | — |
| peak_memory_gb | 70.45 GB | — | — |
| NaN/Inf events | None | — | — |

| Block | ls_attn gamma (ep9) | ls_mlp gamma (ep9) |
|---|---|---|
| Block 1 | ~0.010 | ~0.019 |
| Block 5 | ~0.006 | ~0.015 |

### Verdict: CLOSED — both pre-registered predictions disconfirmed

1. **Block-curriculum failed**: late blocks (closer to readout) should activate first per the CaIT cold-start rationale. Opposite observed: `ls_attn` block 1 gamma > block 5. MLP gammas uniform across depth.
2. **Convergence speed didn't improve**: val at epoch 1 = 330, descending at a **slowing** rate (44→35→30→26→20→15→14→11 per epoch) — classic signature of a model whose residual branches haven't unfolded. At epoch 9 gammas are still ~0.01–0.02 (50–200× smaller than 1.0).

**Root cause**: CaIT uses init=1e-4 for 36-layer transformers with ~300 training epochs. Our 5-layer, 12-epoch, 30-min-cap setup cannot give the gammas time to grow. The model wasted its entire compute budget on gate-growing rather than CFD optimization.

**Methodological value**: cleanly rules out the CaIT cold-start axis at this budget. A larger init (1e-2 or 1.0) would be a regularization hypothesis, not a cold-start curriculum, and belongs as a fresh PR if ever revisited. Student's per-block gamma instrumentation made the close decision unambiguous.

## 2026-05-15 22:40 — PR #3141: Random Fourier features on (x,z) position (rebased rerun) — CLOSED

- **Student branch**: `charliepai2i24h1-tanjiro/fourier-pos-rebased`
- **Hypothesis**: Random Fourier feature mapping σ=4.0 on (x,z) coordinates injects high-frequency spatial basis into the Transolver input, counteracting spectral bias. Predicted improvement on surface pressure captures near sharp geometry edges.

### Results

| Metric | Value | Baseline (#3136) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p (best @ ep9/12)** | **133.6001** | 126.3241 | **+5.7% (WORSE)** |
| test_avg/mae_surf_p (NaN-safe 4-split) | 121.5384 | — | — |
| epochs realized | 9 of 12 | 14 | — |
| peak_memory_gb | 63.22 GB | — | — |

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 160.6934 | 144.6801 |
| geom_camber_rc | 145.2853 | 129.7991 |
| geom_camber_cruise | 107.8672 | 89.7765 |
| re_rand | 120.5545 | 121.8979 |
| **avg** | **133.60** | **121.54** |

- **Metric artifacts**: `models/model-charliepai2i24h1-tanjiro-fourier-pos-rebased-20260515-212626/metrics.jsonl`

### Verdict: CLOSED — two-run negative signal confirms closure

- Pre-rebase: val=136.14; post-rebase (current advisor stack): val=133.60. Improvement of 2.5 points from rebasing, but still +5.7% above baseline. The direction is **stable in the regression band** across both runs.
- Budget was correctly aligned (T_max=12). Schedule misalignment cannot explain the gap.
- **Spectral-bias pattern confirmed**: surface `p` MAE far higher than `Ux`/`Uy` across splits — Fourier encoding didn't close the gap, but the diagnosis was correct.
- σ sweep (σ ∈ {2, 4, 8}) would be a different hypothesis for a future PR if positional encoding is revisited.

## 2026-05-15 22:42 — PR #3435: Aux task — predict is_surface indicator (BCE, aux_weight=0.1) — CLOSED

- **Student branch**: `charliepai2i24h1-alphonse/aux-issurface`
- **Hypothesis**: Predicting a binary `is_surface` indicator alongside the primary CFD regression (aux_weight=0.1, BCE loss) should improve boundary-layer representations, sharpening surface pressure accuracy. +193 params for the aux head.

### Results

| Metric | Value | Baseline (#3136) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p (best @ ep9/12)** | **141.2732** | 126.3241 | **+11.8% (WORSE)** |
| test_avg/mae_surf_p (NaN-safe 4-split) | 128.6019 | — | — |
| n_params | 1,447,714 | 1,447,521 | +193 (exactly as predicted) |
| peak_memory_gb | 63.0 GB | — | — |

| Split | val mae_surf_p | Baseline | Δ |
|---|---|---|---|
| val_single_in_dist | 180.9047 | 158.79 | +13.9% |
| val_geom_camber_rc | 151.6076 | 127.26 | +19.1% |
| val_geom_camber_cruise | 108.4551 | 102.20 | +6.1% |
| val_re_rand | 124.1254 | 117.04 | +6.1% |

### Aux head diagnostics

| Epoch | aux_acc | aux_loss |
|---|---|---|
| 1 | 0.8997 | 0.5889 |
| 6 | 0.9892 | 0.0988 |
| 9 | 0.9909 | 0.0595 |

- **Metric artifacts**: `models/model-aux-issurface-w0p1-20260515-212151/metrics.jsonl`

### Verdict: CLOSED — aux head saturates early; learning signal diminishes before primary task converges

The aux head reaches ~99% accuracy by epoch 6 of 9 realized, with aux_loss dropping 10× by epoch 3. Once the head saturates, the aux_weight=0.1 gradient signal becomes trivially small — the model gets no additional supervision. Meanwhile the primary regression task (surface pressure) regressed on all 4 splits. The 11.8% regression exceeds the 5% threshold.

**Note on label discipline**: student didn't swap `status:wip` → `status:review` after posting the SENPAI-RESULT. PR was reviewed in the `status:wip` queue after identifying the committed metrics. Future PRs should invoke the label-swap or `senpai:submit-experiment-results` skill as documented in the PR body.

**Mechanism insight**: aux task saturation at epoch 6 means the representation-shaping effect only operates for the first half of training. With a later-saturating target (e.g., `near_surface_distance` regression rather than binary classification), this might behave differently. Closed without a follow-up — representation-shaping via aux supervision has other active routes (DropPath, RMSNorm).

## 2026-05-15 22:44 — PRs #3496, #3498, #3500: New round-2 dispatches

- **#3496 nezuko**: RMSNorm swap — replace all `nn.LayerNorm` with `RMSNorm` (T5/PaLM/LLaMA standard). Drop-in, -2,112 params, 5-10% per-epoch speedup predicted.
- **#3498 tanjiro**: SwiGLU MLP — replace GELU FFN with `SwiGLU(d_ff=256)` (Shazeer 2020, iso-param per PaLM convention). Gated FFN for multi-scale targets.
- **#3500 alphonse**: slice_num 64→32 — halve PhysicsAttention slice tokens, direct memory-bandwidth attack. Predicted 15-25% per-epoch speedup (10-18 realized epochs vs 9 at fp32).

## 2026-05-15 23:00 — PR #3455: DropPath stochastic depth (linear 0→0.1) on TransolverBlock residuals — CLOSED

- **Student branch**: `charliepai2i24h1-frieren/droppath-residuals`
- **Hypothesis**: DropPath linear schedule [0, 0.025, 0.05, 0.075, 0.1] on TransolverBlock residual branches; CaIT-canonical zero-parameter regularizer; predicted to help OOD splits (re_rand, geom_camber_rc) disproportionately by reducing depth co-adaptation.

### Results

| Metric | Value | Baseline (#3136) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p (best @ ep9/12)** | **146.6597** | 126.3241 | **+16.1% (WORSE)** |
| test_avg/mae_surf_p (NaN-safe 4-split) | 133.8657 | — | — |
| n_params | 1,447,521 | 1,447,521 | 0 (matches exactly) |
| peak_memory_gb | 62.997 | — | — |
| epochs realized | 9 of 12 | 14 | — |
| Per-epoch wall time | ~213 s | ~205 s | wider-trunk variance, not DropPath |

| Split | val mae_surf_p | Baseline | Δ |
|---|---|---|---|
| val_single_in_dist | 190.92 | 158.79 | +20.2% |
| val_geom_camber_rc | 156.28 | 127.26 | +22.8% |
| val_geom_camber_cruise | 112.51 | 102.20 | +10.1% |
| val_re_rand | 126.93 | 117.04 | +8.5% |
| Split-spread max/min | 1.70 | ~1.55 | got WORSE |

- **Metric artifacts**: `models/model-droppath-r0p1-20260515-222516/metrics.jsonl`

### Verdict: CLOSED — directional disconfirmation, not just magnitude

The hypothesis predicted **disproportionate OOD wins**. Data shows the opposite:

1. **Split-spread got worse** (1.70 vs ~1.55 baseline) — DropPath made the distribution gap larger, not smaller.
2. **In-dist regressed MORE than OOD** in absolute terms (+20.2% single_in_dist vs +8.5% re_rand).
3. **Lower rates wouldn't fix this** — they'd shrink the regression magnitude but the directional pattern would persist. The data says depth co-adaptation isn't the failure mode this model is suffering from.

Methodologically clean: closes the CaIT-style residual regularization axis at this budget. Different from a "rate is wrong" close.

## 2026-05-15 23:03 — PR #3336: grad-clip max_norm=100 rerun — CLOSED

- **Student branch**: `charliepai2i24h1-fern/grad-clip` (rebased after #3378)
- **Hypothesis (rerun)**: max_norm=100 should clip only spike batches (predicted clip_rate 0.1-0.3) instead of every batch as max_norm=1.0 did. Predicted recovery of the noisy-split improvements without losing median-batch info.

### Results

| Metric | Value | max_norm=1.0 round 1 | Baseline (#3136) |
|---|---|---|---|
| **val_avg/mae_surf_p (best @ ep9/12)** | **127.3480** (committed, best of 3 seeds) | 129.7342 | 126.3241 |
| Mean of 3 seeds (127.35, 130.59, 131.21) | 129.7158 | 129.73 | — |
| test_avg/mae_surf_p (NaN-safe 4-split) | 114.5905 | 116.5846 | — |
| Δ vs baseline (committed) | **+0.81% (worse)** | +2.7% | — |
| Δ vs baseline (3-seed mean) | **+2.7% (worse, == max_norm=1.0)** | +2.7% | — |

| Split | mae_surf_p (committed) | max_norm=1.0 | Baseline | Δ vs baseline |
|---|---|---|---|---|
| val_single_in_dist | 158.47 | 162.01 | 158.79 | -0.2% |
| val_geom_camber_rc | 142.36 | 149.45 | 127.26 | **+11.9%** (regression persists across both clip levels) |
| val_geom_camber_cruise | 96.96 | 95.64 | 102.20 | **-5.1%** (improved as hypothesis predicted) |
| val_re_rand | 111.60 | 111.84 | 117.04 | **-4.7%** (improved as hypothesis predicted) |

### Clip-rate distribution landed exactly in the canonical regime

| Epoch | grad_norm_mean | grad_norm_max | clip_rate |
|---|---|---|---|
| 1 | 232 | 1085 | **0.797** |
| 5 | 138 | 899 | **0.512** |
| 9 | 106 | 2008 | **0.283** |

Decayed from 80% → 28% as training stabilized. Textbook spike-clipping regime.

### Verdict: CLOSED — multi-seed evidence shows effect is noise-equivalent

1. **3-seed mean = 129.72** is essentially identical to max_norm=1.0 result (129.73). On a multi-seed basis grad-clip is **neutral**, not improving.
2. **Seed variance (~4 MAE units)** is comparable to the entire effect size. Three more seeds wouldn't shift the conclusion.
3. **camber_rc regression persists across both clip levels** (+17.4% at max_norm=1.0, +11.9% at max_norm=100). This is **not stochastic** — it's a real mechanism: camber_rc samples have systematically high gradient norms and grad-clip discourages the large optimizer steps they need.
4. **Clip distribution landed in the canonical regime** (0.80 → 0.28 over 9 epochs). If it doesn't help here, it won't help at max_norm=50 or 200 either.

### Permanent findings to preserve

- **Camber_rc gradient-norm-driven systemic regression**: this dataset has a split that benefits from large optimizer steps while two other splits benefit from clip. The grad-clip mechanism is a wash because these effects cancel.
- **`train/grad_norm_{mean,max,min,clip_rate}` per-epoch logging**: cheap, high-information diagnostic that revealed both the loss-scale issue (round 1) and the canonical regime (round 2). Should persist regardless of whether clipping is active.
- **`eval_test_clean.py` is dead code** post-#3378 — flagged for future cleanup but not in scope here.

Student's research hygiene (exposing 3 uncommitted seed runs and explicitly calling out variance > effect size) is exemplary. Closed cleanly with no follow-up — the optimization-stability axis is exhausted at this budget via the grad-clip mechanism class.

## 2026-05-15 23:05 — PRs #3525, #3526: Round-3 dispatches

- **#3525 fern**: Lion optimizer (Chen et al. 2023) — sign-momentum, half the optimizer state of AdamW, replaces per-param LR adaptation with uniform sign updates. Tests whether the camber_rc grad-norm asymmetry that grad-clip exposed is mediated by AdamW's `v` term. lr=1.5e-4, wd=1e-3 (Lion-scaled from AdamW 5e-4/1e-4).
- **#3526 frieren**: torch.compile(dynamic=True) — operator fusion attacking the per-kernel dispatch overhead. Different mechanism from slice_num=32 (smaller tensors) and bf16 (smaller dtype). Predicted 10-20% per-epoch speedup → 12-13 realized epochs vs 9 baseline. Inductor cache fresh on first run.

## 2026-05-15 23:30 — PR #3500: slice_num 64→32 — CLOSED

- **Student branch**: `charliepai2i24h1-alphonse/slice-num-32`
- **Hypothesis**: Halve PhysicsAttention slice tokens (64→32) as a direct memory-bandwidth attack. Predicted 15-25% per-epoch speedup → 15-18 realized epochs vs 8-9 baseline.

### Results

| Metric | Value | Baseline (#3136) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p (best @ ep11/12)** | **136.5227** | 126.3241 | **+8.1% (WORSE)** |
| test_avg/mae_surf_p (NaN-safe 4-split) | 123.1666 | — | — |
| Per-epoch wall time | 174.3 s (mean of 11) | ~205 s | **-15.4% (throughput win)** ✓ |
| epochs realized | **11** of 12 | 8-9 | +22-37% (throughput win) ✓ |
| n_params | 1,442,241 | 1,447,521 | -6,720 (PR overestimated at -30,720) |
| peak_memory_gb | 55.53 | — | higher than predicted ~42 GB |

| Split | val mae_surf_p | Baseline | Δ |
|---|---|---|---|
| val_single_in_dist | 173.09 | 158.79 | +9.0% |
| **val_geom_camber_rc** | **151.29** | **127.26** | **+18.9%** (absorbs ~all regression) |
| val_geom_camber_cruise | 103.09 | 102.20 | +0.9% (≈ neutral) |
| val_re_rand | 118.61 | 117.04 | +1.3% (≈ neutral) |

- **Metric artifacts**: `models/model-slice-num-32-20260515-232341/metrics.jsonl`

### Verdict: CLOSED — throughput win real but capacity asymmetric

**The hypothesis split into two outcomes**:
1. ✅ **Throughput**: confirmed at +15.4% per-epoch, 11 realized epochs vs 8-9 baseline (low end of predicted 15-25%).
2. ❌ **Quality**: +8.1% regression dominated by val_geom_camber_rc (+18.9% vs ≤+1.3% on other splits).

**Key diagnostic findings**:
- **Slice mechanism is capacity-bottlenecked, not memory-bottlenecked**: peak_memory came in at 55.5 GB, substantially higher than the PR-body's ~42 GB anchor. The per-node-to-slice projection (`B × N≈240K × slice_num × heads × 4 bytes`) dominates VRAM, not the slice-attention matrix. The wider trunk's residual stream + intermediate activations are the real memory consumers.
- **camber_rc requires slice-token resolution**: +18.9% regression on rc while cruise and re_rand are within ±1.3% of baseline. The held-out tandem-camber split needs more mesh-structure resolution than slice_num=32 provides. Other splits are slice_num-insensitive at this scale.
- **Param-count delta was overestimated**: actual -6,720 (not -30,720). Slice projection is `dim_head × slice_num` per head, not `n_hidden × slice_num` per layer.

**Permanent finding**: slice axis is mapped — 64 baseline, 32 too narrow for rc. Future PRs should expect tandem-geometry-sensitive operators to be parametrically expensive on the wider trunk. **camber_rc is the most slice-sensitive split** — a useful invariant for future capacity-vs-throughput tradeoffs.

## 2026-05-15 23:32 — PR #3437: Domain curriculum (racecar_single first → uniform) — CLOSED

- **Student branch**: `charliepai2i24h1-thorfinn/domain-curriculum`
- **Hypothesis**: WeightedRandomSampler with easy_boost=3.0 starting from racecar_single-only, transitioning to uniform by epoch 6 (transition_frac=0.5). Predicted easy-domain foundation → better OOD generalization on geom and re_rand splits.

### Results

| Metric | Value | Baseline (#3136) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p (best @ ep11/12)** | **146.4934** | 126.3241 | **+15.96% (WORSE)** |
| test_avg/mae_surf_p (NaN-safe 4-split) | 134.5920 | — | — |
| Per-epoch wall time | 181 s (mean) | ~120 s | **+50% overhead** (DataLoader rebuild) |
| epochs realized | 11 of 12 (just over 30-min cap) | 14 | — |

| Split | val mae_surf_p | Baseline | Δ |
|---|---|---|---|
| val_single_in_dist | 161.54 | 158.79 | +1.7% (≈ neutral) |
| val_geom_camber_rc | 159.80 | 127.26 | **+25.6%** (predicted to win, regressed worst) |
| val_geom_camber_cruise | 128.18 | 102.20 | +25.4% |
| val_re_rand | 136.45 | 117.04 | **+16.6%** (predicted to win, regressed) |

### Curriculum diagnostic — domain mix fired exactly as designed

| Epoch | rc_single | rc_tandem | cruise | val_avg/mae_surf_p |
|---|---|---|---|---|
| 1 | 0.94 | 0.03 | 0.03 | 330.69 |
| 4 | 0.65 | 0.18 | 0.18 | 231.14 |
| 7 (uniform) | 0.33 | 0.33 | 0.33 | 182.29 |
| 11 (terminal) | 0.33 | 0.33 | 0.33 | 146.49 |

Mix reaches uniform at epoch 7 as designed. val_avg/mae_surf_p still descending steeply at epoch 11 (~4% per-epoch), confirming under-training.

- **Metric artifacts**: `models/model-domain-curriculum-20260515-232319/metrics.jsonl`

### Verdict: CLOSED — three independent failure modes

1. **Hypothesis predictions inverted**: PR predicted val_geom_camber_rc and val_re_rand wins (OOD splits should benefit from broader foundation). Data shows rc regressed +25.6% (worst) and re_rand +16.6% — both predicted-to-win splits regressed.

2. **Structural wall-time overhead +50%**: DataLoader rebuilt each epoch + WeightedRandomSampler weights rebuilt each epoch + worker cold-start. Mean 181 s vs ~120 s baseline. Cut realized epochs from 14 → 11.

3. **Curriculum is budget-elastic**: needs slack at both ends (easy phase + uniform-recovery phase). The 30-min cap doesn't have either headroom — 6 of 11 epochs were "starving" hard domains.

**Student's suggested follow-ups** all retain the DataLoader-rebuild overhead. Even milder curriculum (easy_boost=1.5) couldn't recover the 50% wall-time penalty. Closed cleanly; if curriculum is revisited, **infrastructure (in-place WeightedRandomSampler weight mutation)** must come first.

**Permanent finding for the log**: per-batch loss `surf_loss` can still produce NaN even after #3378 NaN-safe accumulator fix (cruise test sample's Inf GT propagates pre-accumulator). One-line fix candidate for future PR; not in current scope.

## 2026-05-15 23:35 — PR #3478 edward narrow+bf16: NUDGE (6+ hours stale, only assignment commit)

- **Student branch**: `charliepai2i24h1-edward/narrow-bf16-confirm`
- **Status**: dispatched at 17:05 UTC; only 1 commit (the assignment commit `35e4bac...`). Pod heartbeating, Claude not running training.
- **Action**: explicit advisor reminder posted with recap of single-attributable change (n_hidden 192→128, n_head 6→4, EMA+surf_weight=25+bf16, epochs=18 T_max=18) and the three decisive outcomes that gate the stop/pivot decision. Three round-3 architectural PRs (#3496, #3498, #3500) stacking on the wider trunk depend on this answer.

## 2026-05-15 23:38 — PRs #3555, #3556: New round-4 dispatches

- **#3555 alphonse**: coordinate jitter augmentation (σ=0.01 Gaussian noise on input (x,z) coords during training, identity at eval). Data-axis regularizer attacking camber_rc overfitting from input side. Untouched mechanism class on this track.
- **#3556 thorfinn**: mixed MSE+L1 surface loss (0.7*MSE + 0.3*L1). Loss-formulation axis, distinct mechanism from askeladd's in-flight SmoothL1 (smooth blend vs linear combination). Attacks heavy-tailed pressure residual distribution + the camber_rc gradient-norm asymmetry fern uncovered.

## 2026-05-16 — Loop 6: PR #3525 fern Lion CLOSED

- **Student branch**: `charliepai2i24h1-fern/lion-optimizer`
- **Hypothesis**: Lion optimizer (sign-momentum, lr=1.5e-4, wd=1e-3) as drop-in for AdamW — tests whether camber_rc's high gradient norms depend on AdamW's per-param v adaptation.
- **Result**: val_avg/mae_surf_p = 126.7473 (best ep 9/9 realized, 30-min cap), +0.33% vs baseline 126.3241. test_avg = 113.4676 (NaN-safe 4-split).

| Split | Lion | Baseline | Delta |
|---|---|---|---|
| single_in_dist | 152.19 | 158.79 | -4.16% improved |
| geom_camber_rc | 141.06 | 127.26 | +10.84% REGRESSED |
| geom_camber_cruise | 99.02 | 102.20 | -3.11% improved |
| re_rand | 114.72 | 117.04 | -1.98% improved |

- Metric artifacts: `models/model-charliepai2i24h1-fern-lion-optimizer-20260515-234226/metrics.jsonl`

**Analysis**: 3 of 4 splits improved; camber_rc regressed +10.84%. This is the **third mechanism class** to produce a camber_rc-specific regression (after grad-clip max_norm=1.0 and max_norm=100 from #3336). Lion removes AdamW's per-parameter v adaptation (sign-only momentum, no adaptive scaling) — the same pattern holds. Throughput prediction (2-5% speedup) did not materialize; 204.7 s/ep vs baseline ~205 s. Schedule alignment caveat: Lion got T_max=12 vs baseline T_max=50, so Lion actually got a more aggressive low-LR phase — this probably *helped* Lion vs a fair matched-schedule comparison.

**Verdict**: CLOSED. Headline ~neutral (within seed variance), but camber_rc-as-discriminator-3rd-mechanism-class is the permanent finding. Closing optimizer-swap axis as primary lever per student's own recommendation.

## 2026-05-16 — Loop 6: PR #3498 tanjiro SwiGLU CLOSED

- **Student branch**: `charliepai2i24h1-tanjiro/swiglu-mlp`
- **Hypothesis**: SwiGLU FFN (iso-param d_ff=256, no biases) as drop-in for GELU FFN — tests whether gated activation improves generalization and gives 5-10% throughput speedup at wider trunk.
- **Result**: val_avg/mae_surf_p = 132.3848 (best ep 9/9 realized, 30-min cap), +4.79% vs baseline. test_avg = 118.4290.

| Split | SwiGLU | Baseline | Delta |
|---|---|---|---|
| single_in_dist | 169.58 | 158.79 | +6.8% |
| geom_camber_rc | 146.69 | 127.26 | +15.3% |
| geom_camber_cruise | 100.29 | 102.20 | -1.9% |
| re_rand | 112.98 | 117.04 | -3.5% |

- Metric artifacts: `models/model-swiglu-mlp-20260515-232558/metrics.jsonl`

**Analysis**: Throughput unchanged (~211 s/ep) — MLP is small fraction of FLOPs on n_hidden=192 trunk vs PhysicsAttention on 74K-242K node meshes. PaLM-cited 5-10% speedup is a large-hidden LLM regime result. Budget cut at 9/12 epochs (211s × 12 = 42 min > 30 min cap) noted; but even with budget cut, per-split signature does not support the hypothesis (camber_rc worst at +15.3%, cruise and re_rand the small winners). n_params = 1,444,641 (-2,880, near iso-param as intended).

**Verdict**: CLOSED. Gated-FFN axis exhausted at n_hidden=192: no throughput win, no quality win, and camber_rc remains discriminator-worst.

## 2026-05-16 — Loop 6: PR #3556 thorfinn MSE+L1 mix CLOSED

- **Student branch**: `charliepai2i24h1-thorfinn/mse-l1-mix`
- **Hypothesis**: Mixed surface loss 0.7*MSE + 0.3*L1 — attacks heavy-tail pressure residuals + camber_rc gradient-norm asymmetry by linearizing the tail gradient.
- **Result**: val_avg/mae_surf_p = 137.3291 (best ep 9/9 realized, 30-min cap), +8.7% vs baseline. test_avg = 123.7493.

| Split | MSE+L1 | Baseline | Delta | Hypothesis predicted |
|---|---|---|---|---|
| single_in_dist | 179.21 | 158.79 | +12.9% | neutral |
| geom_camber_rc | 149.42 | 127.26 | +17.4% | biggest WIN — INVERTED |
| geom_camber_cruise | 101.11 | 102.20 | -1.1% | biggest win — tiny |
| re_rand | 119.58 | 117.04 | +2.2% | — |

- Metric artifacts: `models/model-surf-mse-l1-mix-20260516-003558/metrics.jsonl`
- Diagnostics: mix functionally engaged (train_surf_l1 0.34 > train_surf_mse 0.27 at ep 9, not degenerating to pure MSE). Per-epoch wall ~203 s, VRAM 63.0 GB — both unchanged.

**Analysis**: Hypothesis falsified directionally: the L1 gradient-floor term `0.3*sign(r)` rotates the gradient toward sign(r) on heavy-tail rc residuals, *destroying* useful curvature instead of providing outlier robustness. This is the **fifth mechanism class** to regress camber_rc when gradient-magnitude information is flattened or replaced.

**5-way camber_rc-as-discriminator synthesis**:

| Mechanism class | camber_rc regression | What is removed from gradient |
|---|---|---|
| slice_num=32 (#3500) | +18.9% | attention capacity |
| grad-clip max_norm=1.0 (#3336) | +17.4% | gradient L2 magnitude (clipped) |
| MSE+L1 mix 0.7/0.3 (this PR) | +17.4% | partial sign(r) replacement |
| grad-clip max_norm=100 (#3336) | +11.9% | gradient L2 magnitude (loose) |
| Lion sign-momentum (#3525) | +10.84% | AdamW per-param v adaptation |

**Verdict**: CLOSED. +8.7% regression (well above 5% close threshold). Closes the linear-combination-loss axis. SmoothL1 (#3127 askeladd, still WIP) is a separate mechanism class (quadratic-near-zero smooth blend) and is unaffected by this close.

## 2026-05-16 — Loop 6: New dispatches #3585, #3588, #3589

- **#3585 fern per-domain-loss-weight**: multiply racecar_tandem surface loss by 2.0× during training. First data-distribution-side attack on the 5-way camber_rc-as-discriminator finding; all prior 5 mechanism classes touched gradient-flow side. Upweights the training domain most structurally similar to held-out geom_camber_rc.
- **#3588 tanjiro lookahead-optimizer**: Lookahead(AdamW, k=5, alpha=0.5). Meta-optimizer wrapper preserving AdamW's v adaptation (which the 5-way finding says camber_rc needs) while adding slow-weight trajectory smoothing. Distinct from Lion (which replaced AdamW) and from grad-clip (which rate-limits gradient magnitude).
- **#3589 thorfinn swa-tail**: Stochastic Weight Averaging over last 3 epochs via torch.optim.swa_utils. Uniform tail averaging complementing EMA's exponential decay. Single-line addition (AveragedModel), evaluated as min(val_ema, val_swa) at end of training.

## 2026-05-16 — Loop 7: PR #3478 edward narrow+bf16 MERGED ← NEW BEST

- **Student branch**: `charliepai2i24h1-edward/narrow-bf16-confirm`
- **Hypothesis**: Revert wider trunk (#3130) to n_hidden=128, n_head=4; add bf16 autocast. Stop/pivot test — does narrower trunk + bf16 + budget-aligned cosine beat wider trunk fp32?
- **Result**: val_avg/mae_surf_p = **111.7473** (epoch 18 of 18, full cosine anneal). test_avg = **99.3066** (NaN-safe 4-split — first sub-100 on this track). **-11.5% vs baseline 126.32.**

| Split | Narrow+bf16 | Old baseline (#3136) | Delta |
|---|---|---|---|
| single_in_dist | 133.64 | 158.79 | -15.8% |
| geom_camber_rc | 121.33 | 127.26 | -4.7% |
| geom_camber_cruise | 88.92 | 102.20 | -13.0% |
| re_rand | 103.10 | 117.04 | -11.9% |

- n_params: 662,359 (vs 1,447,521 wider — 54% smaller)
- peak_memory: 32.95 GB (vs 42.11 GB wider)
- per_epoch_wall: ~98 s (vs ~205 s wider — 52% faster)
- Metric artifacts: `models/model-narrow-bf16-aligned-20260516-002225/metrics.jsonl`

**Analysis**: The budget constraint was the binding factor. Wider trunk at fp32 got 8-9 epochs at ~205s/ep (28% cosine annealed). Narrow+bf16 gets 18 full epochs at ~98s/ep (100% annealed). The model crossed the old baseline at epoch 13, then kept improving through 18. camber_rc improved least (-4.7%) vs -12-16% for others — consistent with camber_rc remaining the discriminator gap even after the budget fix.

**Verdict**: MERGED. New best. Active config reverted to n_hidden=128, n_head=4, bf16, T_max=18, epochs=18.

## 2026-05-16 — Loop 7: PR #3526 frieren torch.compile CLOSED

- **Student branch**: `charliepai2i24h1-frieren/torch-compile-dynamic`
- **Hypothesis**: torch.compile(dynamic=True) gives 22% throughput speedup via operator fusion.
- **Result**: val_avg = 135.4121 (+7.2% regression). 22% speedup confirmed (160s vs 205s). 12 epochs realized.

| Split | Compile | Baseline | Delta |
|---|---|---|---|
| single_in_dist | 170.55 | 158.79 | +7.4% |
| geom_camber_rc | 141.42 | 127.26 | +11.1% |
| geom_camber_cruise | 107.78 | 102.20 | +5.5% |
| re_rand | 121.91 | 117.04 | +4.2% |

- Metric artifacts: `models/model-torch-compile-dynamic-20260516-002538/metrics.jsonl`

**Analysis**: Throughput win is real (22% speedup, 12 epochs realized, zero graph breaks). Quality regression comes from kernel fusion reordering reduction operations in PhysicsAttention's slice-attention — cumulative numerical drift. Train loss upticks at ep 4 and ep 9 indicate noisier optimization landscape. Note: this PR ran on the WIDER trunk baseline; irrelevant post-#3478 merge since the architecture has changed.

**Verdict**: CLOSED. Throughput win is real but quality regression is unacceptable (+7.2%). Axis not permanently closed — torch.compile + determinism check is a valid follow-up on the new narrow+bf16 config.

## 2026-05-16 — Loop 7: PR #3496 nezuko RMSNorm CLOSED

- **Student branch**: `charliepai2i24h1-nezuko/rmsnorm-swap`
- **Hypothesis**: RMSNorm replaces LayerNorm for throughput + quality gain.
- **Result**: val_avg = 144.7053 (+14.5% regression, best of fused arm). 53% wall-time regression killed it.

Two arms: naive (+91% wall), fused F.rms_norm (+53% wall). Fused arm: 9 epochs realized vs baseline 14.

**Key permanent finding**: matched-epoch comparison shows RMSNorm fused BETTER than LayerNorm by ~8% from ep≥5 onward. The channel-mean preservation effect in RMSNorm (no mean subtraction) is real — but the backward kernel for F.rms_norm on this Blackwell stack is missing a aten::native_rms_norm_backward equivalent, causing 53% wall-time regression.

**Diagnostic**: isolated micro-benchmark showed norm-kernel 3% faster; end-to-end 53% slower → secondary autograd effect dominates. The RMSNorm quality benefit is real but inaccessible in this PyTorch/hardware configuration.

**Student follow-up flagged**: scale-only LayerNorm (bias=False, keeps mean-subtraction, keeps fused kernel) would isolate whether the bias-drop or mean-subtraction drives the quality gain. Assigned as nezuko's next PR (#3624).

**Verdict**: CLOSED. Full RMSNorm axis closed. "Channel-mean preservation" question survives as a separate test.

## 2026-05-16 04:26 — Loop 10: PR #3620 edward depth-nlayers7 CLOSED

- **Student branch**: `charliepai2i24h1-edward/depth-nlayers7`
- **Hypothesis**: n_layers 5→7 increases representational capacity; predicted biggest improvement on val_geom_camber_rc (the split that improved least from narrow+bf16 win).
- **Result**: val_avg = 132.9310 (+19.0% regression), test_avg = 119.677 (+20.5% regression).

| Split | n_layers=7 | Baseline (n_layers=5) | Delta |
|---|---|---|---|
| val_single_in_dist | 167.05 | 133.64 | +25.0% worse |
| val_geom_camber_rc | 143.33 | 121.33 | +18.1% worse |
| val_geom_camber_cruise | 102.33 | 88.92 | +15.1% worse |
| val_re_rand | 119.02 | 103.10 | +15.4% worse |
| **val_avg** | **132.93** | **111.75** | **+19.0% worse** |

- n_params: 904,671 (+36.6% vs 662,359 baseline)
- peak_memory_gb: 44.91 (+36.3% vs 32.95)
- per-epoch wall time: ~135 s (+38.8% vs ~98 s)
- realized epochs: 13/13 (budget-aligned) vs baseline 18/18
- Metric artifacts: `models/model-depth-nlayers7-20260516-032429/metrics.jsonl`

**Analysis**: Two effects confounded. (1) Budget compression: +38.8% per-epoch cost → 13 realized epochs vs 18 for baseline. Trajectory still descending at -3.25/epoch at end of cosine schedule — not converged. (2) Per-split signature falsifies the hypothesis directly: camber_rc regressed the SAME ~18% as the other splits, not less. If depth were specifically helpful for camber_rc, we'd expect camber_rc to lag the others LESS at the equivalent training step. We see the opposite — single_in_dist held up best in relative terms.

**Conclusion**: depth is NOT the binding capacity lever for camber_rc at n_hidden=128. Close the depth axis at this width. Depth can only be revisited with reduced slice_num/batch_size to recover per-epoch budget.

**Verdict**: CLOSED. Depth axis closed (n_hidden=128 budget). Data-side and slice-mechanism levers are the remaining architectural options.

## 2026-05-16 07:10 — Loop 15: PR #3127 askeladd SmoothL1-rebased MERGED ← NEW BEST

- **Student branch**: `charliepai2i24h1-askeladd/smoothl1-loss`
- **Hypothesis**: Replace MSE with `F.smooth_l1_loss(beta=1.0)` in training loop and evaluate_split. SmoothL1 better matches the `mae_surf_p` L1 evaluation metric and de-emphasizes outlier samples via bounded gradient.
- **Result**: val_avg = 94.9723 (**−15.0%** vs 111.7473 baseline), test_avg = 85.0372 (−14.4%). MERGED.

| Split | SmoothL1 | MSE Baseline | Δ |
|---|---|---|---|
| val_single_in_dist | 110.85 | 133.64 | −17.0% |
| val_geom_camber_rc | 103.78 | 121.33 | −14.5% |
| val_geom_camber_cruise | 75.84 | 88.92 | −14.7% |
| val_re_rand | 89.42 | 103.10 | −13.3% |
| **val_avg** | **94.97** | **111.75** | **−15.0%** |
| test_single_in_dist | 97.18 | 113.39 | −14.3% |
| test_geom_camber_rc | 93.68 | 109.86 | −14.7% |
| test_geom_camber_cruise | 64.13 | 73.92 | −13.2% |
| test_re_rand | 85.16 | 100.05 | −14.9% |
| **test_avg** | **85.04** | **99.31** | **−14.4%** |

- n_params: 662,359 (unchanged — pure loss-axis change)
- peak_memory_gb: 32.95 (unchanged)
- per-epoch wall: ~98 s (unchanged)
- epochs realized: 18/18 (full cosine anneal, monotone descent throughout)
- Metric artifacts: `models/model-charliepai2i24h1-askeladd-smoothl1-rebased-20260516-052919/metrics.jsonl`

**Analysis**: Dramatic and uniform win. Every epoch was a new best (monotone training throughout). The improvement is symmetric across all 4 val splits (13-17%) — consistent with the loss-metric alignment mechanism (SmoothL1 gradient targets the same L1 objective as `mae_surf_p`). No instability, no NaN. Zero compute overhead. This is the highest single-PR gain on the track since #3478 narrow+bf16 (-11.5%). Combined with #3478, the two changes account for: 126.32 → 111.75 → **94.97** — a cumulative -24.8% from the pre-narrow baseline.

The original #3127 submission (val=114.14) was on the pre-merge config (n_hidden=128, surf_weight=10, no EMA). The rebased run on narrow+bf16+EMA+surf_w25 shows SmoothL1 compounds cleanly with all prior wins rather than substituting for them.

**camber_rc note**: improved from 121.33 → 103.78 (-14.5%). Still the highest-error split but no longer an outlier — all 4 splits now cluster in a narrower range (75-111 vs 89-134 before).

**Verdict**: MERGED. New best. Active loss changed to SmoothL1 beta=1.0. All 7 in-flight WIPs notified of new baseline 94.97 and instructed to rebase before training.

## 2026-05-16 — Loop 15: New dispatch #3763

- **#3763 askeladd smoothl1-beta-sweep**: 2-arm sweep of beta=0.5 and beta=0.25 on the new SmoothL1 baseline. No source edits needed — `smooth_l1_beta` is a Config CLI field. Tests whether pushing the L1/L2 transition lower improves further given the underfit baseline (loss still descending at epoch 18).

## 2026-05-16 05:26 — Loop 12: PR #3621 frieren batch-size-8 CLOSED

- **Student branch**: `charliepai2i24h1-frieren/batch-size-8`
- **Hypothesis**: bs=4→8 reduces per-step gradient variance; predicted 0.5-2% improvement, biggest on camber_rc (noisy-grad split).
- **Result**: val_avg = 162.1397 (+45.1% regression), test_avg = 146.10 (+47.1%). Two arms (12 ep, 15 ep) both deeply regressed.

| Split | bs=8 (ep15) | Baseline (bs=4, ep18) | Delta |
|---|---|---|---|
| val_single_in_dist | 218.98 | 133.64 | +63.9% |
| val_geom_camber_rc | 175.53 | 121.33 | +44.7% |
| val_geom_camber_cruise | 119.93 | 88.92 | +34.9% |
| val_re_rand | 134.11 | 103.10 | +30.1% |
| **val_avg** | **162.14** | **111.75** | **+45.1% worse** |

- peak_memory_gb: 65.87 (vs 32.95, +99%)
- per-epoch wall: 104.4 s (vs ~98 s, +6%) — barely budged
- realized epochs: 15/15 (cosine fully annealed) — best == last epoch (still descending)
- grad steps: 2,820 vs baseline 6,750 (-58%)
- Metric artifacts: `models/model-batch8-20260516-042449/metrics.jsonl`

**Critical systemic finding** (high-value diagnostic from frieren's analysis): per-epoch time barely changed (+6%) with 2× batch → compute-bound, not memory-bandwidth-bound at n_hidden=128 + bf16. Doubling batch_size HALVES gradient steps in the budget. Loss still monotonically descending at the cosine-annealed final epoch — **the narrow+bf16 baseline at 18 epochs is itself underfit**. The model is step-quality limited, not capacity limited.

Per-split signature also rules out the gradient-noise hypothesis: regression is roughly uniform (30-64%) across splits, including camber_rc (+44.7%). Gradient noise at bs=4 is NOT a dominant bottleneck for any split.

**Conclusion**: Close batch-size axis at this budget (2nd close — also failed at wider trunk). **The new high-priority lever is LR/schedule tuning** to attack the underfit-baseline finding.

**Verdict**: CLOSED. Frieren reassigned to attack underfit via lr=5e-4→1e-3 (#3719).

## 2026-05-16 — Loop 12: New dispatch #3719

- **#3719 frieren lr-1e3**: lr 5e-4→1e-3 at narrow+bf16. Directly attacks underfit-baseline finding from #3621. Doubles effective lr integral over cosine schedule. Zero budget cost (per-step time unchanged, same 18 epochs). bf16 stability is the only real risk — instrumented to halt on early NaN. If wins, follow-up is lr=2e-3 + warmup. If diverges, indicates 5e-4 was at the bf16 stability ceiling and the next move is warmup ramp.

## 2026-05-16 — Loop 10: New dispatch #3676

- **#3676 edward slice-num-48**: slice_num 64→48 at narrow+bf16 baseline. Tests whether slice_num=64 is over-parameterized at the narrow trunk. Reduces per-epoch compute (~-15%), enabling ~21 epochs vs the baseline's 18 — more cosine completion with zero budget penalty. If 48≈64 quality, slice mechanism has slack; if 48 regresses, validates #3500 "slice is capacity-bottlenecked" finding and the next move is slice_num=96. Budget-adjusted epochs=21 (est. ~85s/ep × 21 ≈ 30 min).

## 2026-05-16 — Loop 7: New dispatches #3620, #3621, #3624

- **#3620 edward depth-nlayers7**: n_layers 5→7 at narrow+bf16 baseline. camber_rc improved least (-4.7%) from the big win — testing whether it's capacity-limited. Budget-adjusted epochs=13 (est. 137s/ep × 13 ≈ 30 min).
- **#3621 frieren batch-size-8**: batch_size 4→8 at narrow+bf16. 63 GB VRAM headroom makes this feasible. Targets gradient variance reduction; different from per-domain weighting. Estimated ~66 GB peak, ~12 epochs in 30 min.
- **#3624 nezuko scale-only-layernorm**: LayerNorm bias=False, keeps fused aten::native_layer_norm kernel. Tests whether bias-drop alone explains the 8% matched-epoch RMSNorm quality gain from #3496. -1,408 params.

All 5 in-flight WIPs (#3127, #3555, #3585, #3588, #3589) notified of new baseline 111.75, new config (n_hidden=128, bf16, epochs=18), and instructed to rebase before training.
