# SENPAI Research Results — willow-pai2g-24h-r1

Track: `willow-pai2g-24h-r1` | Advisor branch: `icml-appendix-willow-pai2g-24h-r1`

## 2026-05-12 20:35 — PR #1515 (SENT BACK): grad-clip max_norm=1.0
- Branch: `willowpai2g24h1-frieren/grad-clip-1.0`
- Hypothesis: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` between `loss.backward()` and `optimizer.step()` will damp the per-batch outliers that produce non-finite pressure predictions on `test_geom_camber_cruise`.
- Single isolated change: insert grad-clip + a `grad_norm` W&B log key.

| Metric | Value | Notes |
|---|---|---|
| val_avg/mae_surf_p (best, epoch 13) | **115.78** | **strongest reading on the fleet** |
| test_avg/mae_surf_p | **NaN** | confirmed data bug (see #1377 and #1567), NOT a model failure |
| test_avg/mae_surf_p (partial 3 of 4) | 114.96 | excluding cruise-p split |
| Params | 0.66 M | unchanged |
| Peak VRAM | 42.1 GB / 96 GB | comfortable |
| Epochs completed | 14 / 50 | hit `SENPAI_TIMEOUT_MINUTES=30` cap |
| Per-epoch | ~131–134 s | comparable to baseline (~120 s) |
| W&B run id | `53avo9qv` | group `stability` |

- Per-split val (epoch 13): single_in_dist clean, geom_camber_rc clean, **geom_camber_cruise=85.58 (clean)**, re_rand clean.
- Per-split test (best-val checkpoint @ epoch 13): single_in_dist=119.72 · geom_camber_rc=122.09 · geom_camber_cruise=**NaN** (data bug) · re_rand=103.08.

**Grad-norm distribution over 5255 train steps (pre-clip):** min=2.37, p1=6.00, p10=13.45, **p50=44.68**, mean=64.57, p90=136.41, p99=332.30, p99.9=502.62, max=940.84. **100% of steps clipped.** Effective per-step LR shrink ~1/45 on average → effective LR ≈ 1.1e-5. So the "occasional outlier" hypothesis was wrong: every step's gradient is being divided by a large constant, and the val win likely comes from this global LR-shrink (combined with adaptive damping of the long tail).

**Status — SENT BACK for one more iteration.** This is the strongest val signal on the fleet (115.78 vs 149.40 prior best on the same branch), but `test_avg/mae_surf_p = NaN` blocks the paper-facing merge. The NaN is the fleet-wide data bug nezuko diagnosed in #1377 (`test_geom_camber_cruise/000020.pt` has `y` containing `+inf`; `data/scoring.py` propagates it via `inf * 0 = NaN`), not a model failure. Since `data/` and `data/scoring.py` are read-only, the workaround is a train.py-side filter that drops samples with non-finite GT before `accumulate_batch`. Sending the PR back to frieren to bundle that ~10-line filter into the same PR — bundle is justified because the hypothesis cannot be evaluated on the paper-facing metric without the filter. Issue #1567 filed with the human research team.

## 2026-05-12 20:30 — PR #1377 (CLOSED): mlp_ratio 2 -> 4 (hidden FFN 256 -> 512)
- Branch: `willowpai2g24h1-nezuko/mlp-ratio-4`
- Hypothesis: doubling FFN hidden dim expands capacity per Transolver block, especially helpful on geometric OOD splits.
- Single isolated change: `model_config["mlp_ratio"]: 2 → 4` in `train.py`.

| Metric | Value | Notes |
|---|---|---|
| val_avg/mae_surf_p (best, epoch 13) | 146.34 | mildly better than round-1 trio (149–155) |
| test_avg/mae_surf_p | **NaN** | same fleet-wide data bug |
| test_avg/mae_surf_p (partial 3 of 4) | 146.32 | excluding cruise-p split |
| Params | 0.99 M | ~+29% vs baseline |
| Peak VRAM | 54.5 GB / 96 GB | comfortable |
| Epochs completed | 13 / 50 | hit `SENPAI_TIMEOUT_MINUTES=30` cap |
| Per-epoch | ~148 s | ~+25% vs baseline-implied |
| W&B run id | `u4meqcav` | group `mlp-ratio-sweep` |

- Per-split val (epoch 13): single_in_dist=197.11 · geom_camber_rc=151.28 · geom_camber_cruise=112.87 · re_rand=124.08.
- Per-split test: single_in_dist=183.17 · geom_camber_rc=132.25 · geom_camber_cruise=**NaN** · re_rand=123.54.

**Conclusion — CLOSED (superseded on val).** mlp_ratio=4 lands at val=146.34, marginally ahead of the round-1 trio but well behind #1515 grad-clip-1.0 (val=115.78). The +29% params and +25% per-epoch cost convert wall-clock into latency without commensurate val improvement under the 30-min cap. mlp_ratio=4 is plausibly viable once tanjiro's bf16 throughput fix (#1516) makes more epochs available — kept as a future revisit candidate.

**Critical bug diagnosis (the headline contribution of this PR).** Nezuko identified that `test_geom_camber_cruise/000020.pt` has `y` containing `+inf` in the hidden pressure GT. `data/scoring.py::accumulate_batch` intends to skip non-finite-GT samples via `sample_mask`, but does `err = (pred - y).abs()` (→ inf for that sample) before applying the mask, and `err * surf_mask = inf * 0 = NaN`. This is the only sample with non-finite GT across all 8 val/test splits. The diagnosis explains the NaN test_avg pattern in **every** completed PR on this fleet (#1372, #1378, #1382, #1377, #1515) and reframes them all from "model-stability failure" to "data infrastructure issue, model otherwise fine." Filed as issue #1567 for the human research team; bundled into #1515 rework as a train.py-side workaround.

## 2026-05-12 19:30 — PR #1382 (CLOSED): AdamW weight_decay 1e-4 -> 3e-4
- Branch: `willowpai2g24h1-thorfinn/wd-3e-4`
- Hypothesis: Stronger weight decay regularizes toward lower-norm solutions, expected to improve OOD splits at minor cost on in-distribution split.
- Single isolated change: `--weight_decay 3e-4` CLI flag (no code change).

| Metric | Value | Notes |
|---|---|---|
| val_avg/mae_surf_p (best, epoch 10) | **149.4008** | best round-1 val reading so far |
| test_avg/mae_surf_p | **NaN** | same `test_geom_camber_cruise/mae_surf_p` inf pathology |
| test_avg/mae_surf_p (partial 3 of 4) | 153.20 | excluding cruise split |
| Params | 0.66 M | unchanged |
| Peak VRAM | 49.1 GB / 96 GB | comfortable |
| Epochs completed | 10 / 50 | hit `SENPAI_TIMEOUT_MINUTES=30` cap |
| Final LR | 4.5e-4 (vs 5.0e-4 peak) | ~10% decay only |
| W&B run id | `n86jz5o4` | group `wd-sweep` |

- Per-split val (epoch 10): single_in_dist=193.07 · geom_camber_rc=163.62 · geom_camber_cruise=107.34 · re_rand=133.57.
- Per-split test (best-val checkpoint @ epoch 10): single_in_dist=180.01 · geom_camber_rc=149.82 · geom_camber_cruise=**NaN** · re_rand=129.77.

**Conclusion — INCONCLUSIVE → CLOSED.** Same dual blocker as PR #1372 and #1378: (a) only 10/50 epochs reached, cosine T_max=50 → LR ~flat at peak, model in early-noise regime; (b) `test_geom_camber_cruise` pressure inf corrupts `test_avg/mae_surf_p`. Best validation reading on this fleet so far (149.40 vs 153.84 from n_head=8 and 155.16 from n_hidden=192) but the wd hypothesis is unresolved without a paired wd=1e-4 baseline at the same epoch budget — and ALL three completed runs hit the same cruise-test inf so the wd-vs-baseline comparison would also have NaN test metrics. The pattern is now unambiguous: undertrained Transolver at peak-LR produces extreme pressure outputs on the hardest OOD split. Closing for consistency with #1372 and #1378; reassigning thorfinn to `huber-loss-vol` as an orthogonal robust-loss experiment that may indirectly stabilize high-Re extremes.

**Student observation worth flagging (not actioned).** Thorfinn pointed out that `data/scoring.py`'s masked-MAE uses `err * mask.double()` accumulation, which propagates NaN if `err` contains `inf` anywhere (even at masked positions, because `inf * 0 = NaN`). `torch.where(mask, err, 0)` or `err.masked_fill(~mask, 0)` would be NaN-safe. `data/scoring.py` is **read-only** per program.md, so we cannot make this fix. The right path is to prevent inf at the source — grad clipping (#1515) and bf16 autocast (#1516) are the active stability/throughput levers. Worth flagging to the human researcher team as a robustness improvement that would help every experiment on this benchmark, but out of scope for this launch.

## 2026-05-12 18:55 — PR #1378 (CLOSED): Widen Transolver hidden dim 128->192
- Branch: `willowpai2g24h1-tanjiro/n-hidden-192`
- Hypothesis: Increase model width from 128 → 192 to expand all projections, expected to improve fitting capacity on all splits with biggest gain on geometric OOD.
- Single isolated change: `model_config["n_hidden"]: 128 → 192` in `train.py:421`.

| Metric | Value | Notes |
|---|---|---|
| val_avg/mae_surf_p (best, epoch 8) | 155.1646 | best of 10 epochs run |
| test_avg/mae_surf_p | **NaN** | corrupted by inf on `test_geom_camber_cruise/mae_surf_p` |
| test_avg/mae_surf_p (partial 3 of 4) | 159.6191 | excluding cruise split |
| Params | 1.47 M (vs ~0.65 M baseline) | +2.25× |
| Peak VRAM | 58.0 GB / 96 GB | comfortable |
| Epochs completed | 10 / 50 | hit `SENPAI_TIMEOUT_MINUTES=30` cap |
| Per-epoch | ~183 s | vs ~120 s baseline-implied |
| W&B run id | `m66sbam1` | group `width-sweep` |

- Per-split test (best-val checkpoint @ epoch 8): single_in_dist=194.51 · geom_camber_rc=151.73 · geom_camber_cruise=**inf** · re_rand=132.62.
- Per-split val (epoch 8): single_in_dist=231.42 · geom_camber_rc=149.30 · geom_camber_cruise=123.25 · re_rand=131.50.

**Conclusion — INCONCLUSIVE → CLOSED.** Two blockers compound under the 30-min cap: (1) wider model + same wall clock = only 10/50 epochs reached, and the cosine T_max=50 schedule means LR is effectively unchanged from peak across the entire run; (2) test_geom_camber_cruise produces non-finite pressure on at least one OOD sample, so the paper-facing `test_avg/mae_surf_p` is NaN. Can't isolate width effect without first fixing throughput (more epochs in the budget) and stability (no inf pressure outputs). Both fixes are good single-knob hypotheses on their own — re-assigning tanjiro to bf16-autocast which directly attacks throughput.

## 2026-05-12 18:54 — PR #1372 (CLOSED): Increase attention heads 4->8
- Branch: `willowpai2g24h1-frieren/n-head-8`
- Hypothesis: Doubling heads (with `dim_head=64` unchanged, raising inner_dim 256→512) gives more parallel physics features and should help OOD splits most.
- Single isolated change: `model_config["n_head"]: 4 → 8` in `train.py:423`.

| Metric | Value | Notes |
|---|---|---|
| val_avg/mae_surf_p (best, epoch 10) | **153.8400** | best of 11 epochs run |
| test_avg/mae_surf_p | **NaN** | corrupted by inf on `test_geom_camber_cruise/mae_surf_p` |
| test_avg/mae_surf_p (partial 3 of 4) | 141.5282 | excluding cruise split |
| Params | 0.65 M | same order as baseline |
| Peak VRAM | 54.5 GB / 96 GB | comfortable |
| Epochs completed | 11 / 50 | hit `SENPAI_TIMEOUT_MINUTES=30` cap |
| Per-epoch | ~170 s | vs ~120 s baseline-implied |
| W&B run id | `pfxtxxms` | group `head-sweep` |

- Per-split test (best-val checkpoint @ epoch 10): single_in_dist=152.99 · geom_camber_rc=133.32 · geom_camber_cruise=**NaN** · re_rand=138.28.
- Per-split val (epoch 10): single_in_dist=231.42 · geom_camber_rc=149.30 · geom_camber_cruise=123.25 · re_rand=131.50.

**Conclusion — INCONCLUSIVE → CLOSED.** Same wall-clock + cosine-schedule mismatch as #1378; only 11/50 epochs and LR essentially flat at peak. Same `test_geom_camber_cruise/mae_surf_p` inf as #1378 — strongly suggests the issue is a model-stability blow-up on extreme OOD samples under undertraining, not a knob-specific failure. Mildly better val than width arm (153.84 vs 155.16) and better partial test (141.53 vs 159.62), but cannot anchor a baseline with NaN test_avg. Re-assigning frieren to grad-clip-1.0 to attack the cruise-pressure inf directly so future capacity experiments produce clean test_avg numbers.

## Cross-result note (round 1 first results)

Three completely independent single-knob changes (`n_hidden=192`, `n_head=8`, `wd=3e-4`) produced the same pathology: model output goes non-finite on `test_geom_camber_cruise/mae_surf_p` while remaining finite on the other 3 test splits and on Ux/Uy of the cruise split. The aggregate `test_avg/mae_surf_p` is NaN for all three. Hypothesis: under-trained Transolver (LR near peak for the entire 10-epoch run under T_max=50 cosine) produces extreme pressure predictions on the unseen camber range (M=2–4) in the cruise tandem domain. Round-2 priorities:
1. **Stability**: `clip_grad_norm_(parameters, 1.0)` — assigned to frieren (#1515).
2. **Throughput**: `torch.autocast(bf16)` mixed precision — assigned to tanjiro (#1516).
3. **Loss robustness**: Huber loss on volume term — assigned to thorfinn (queued).
4. **Schedule mismatch**: separate follow-up — once we know roughly how many epochs fit, future experiments should pass `--epochs` close to that count so `CosineAnnealingLR(T_max=epochs)` actually anneals.

Round-1 informational val ranking (NOT a settled baseline since all test_avg are NaN):

| PR | Change | val_avg/mae_surf_p | partial test_avg (3 of 4) | status |
|---|---|---:|---:|---|
| #1382 | wd=3e-4 | **149.40** | 153.20 | closed |
| #1372 | n_head=8 | 153.84 | 141.53 | closed |
| #1378 | n_hidden=192 | 155.16 | 159.62 | closed |
