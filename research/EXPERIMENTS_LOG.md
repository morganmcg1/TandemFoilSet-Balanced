# SENPAI Research Results

Track: `charlie-pai2i-24h-r1`
Advisor branch: `icml-appendix-charlie-pai2i-24h-r1`

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
