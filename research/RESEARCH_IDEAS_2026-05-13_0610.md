# Round N Research Ideas — 2026-05-13 06:10

**Context:** 12 merges; current best val_avg/mae_surf_p = 82.56 / test = 74.13 (#1812 lr-warmup-1ep). Three idle students: edward, fern, frieren.

**Key finding driving these hypotheses:** In every examined run (warmup, eta_min, T_max=18), val is still actively descending at the final training epoch. The 30-min wall-clock cap terminates training before convergence. The model is undertrained, not overfit. This means schedule changes that maximize effective learning within the budget have the highest expected ROI.

**Per-split MAE at best checkpoint (#1812):** cruise=66.68, re_rand=81.77, single=90.40, rc=91.39. The raceCar OOD split (`val_geom_camber_rc`) is 37% harder than cruise — it is the dominant bottleneck.

---

## H1 — CosineAnnealingWarmRestarts (fern, slug `cawr-t0-9`)

**Change:** Replace `SequentialLR(LinearLR(1ep) + CosineAnnealingLR(T_max=17, eta_min=5e-5))` with `SequentialLR(LinearLR(1ep warmup, start_factor=0.01, total_iters=1) + CosineAnnealingWarmRestarts(T_0=9, T_mult=2, eta_min=5e-5))`. Keep all other hyperparameters identical.

**Why:** The first cosine period ends at epoch 10 (1 warmup + 9 cosine), refreshing the LR back to 5e-4 for a second period of 18 epochs (10 to 28). In the 30-min budget this second period runs to timeout, effectively extending the productive high-LR phase beyond the standard single cosine cycle. The restart injects fresh gradient signal exactly at midpoint, which has repeatedly shown gains on irregular-mesh regression problems where the model is undertrained.

**Mechanism:** Current schedule decays LR monotonically to 5e-5 by epoch 18 while val is still falling. CAWR reschedules to a second high-LR phase during the budget, converting the tail of training from "low-lr fine-tuning" to "fresh mid-lr descent."

**Risk:** Second cycle LR peak may cause a transient bump; val should recover within 2 epochs of restart. If it does not, the restart period is wrong. Also: T_mult=2 means if somehow 28 epochs fit in budget, the third period would be 36 epochs long (irrelevant — wall-clock will cut it).

**Expected gain:** 2–5 val points.

**Student instruction:** In `train.py`, find the `SequentialLR` block. Replace `CosineAnnealingLR(optimizer, T_max=17, eta_min=5e-5)` with `CosineAnnealingWarmRestarts(optimizer, T_0=9, T_mult=2, eta_min=5e-5)`. Keep `LinearLR(start_factor=0.01, end_factor=1.0, total_iters=1)` warmup unchanged. Do not change lr, wd, or any other setting.

---

## H2 — 2-epoch linear warmup (edward, slug `warmup-2ep`)

**Change:** Extend the warmup from 1 epoch to 2 epochs. Replace `LinearLR(start_factor=0.01, end_factor=1.0, total_iters=1)` with `LinearLR(start_factor=0.01, end_factor=1.0, total_iters=2)`. Adjust cosine phase to `CosineAnnealingLR(T_max=16, eta_min=5e-5)` to keep total epochs at 18.

**Why:** 1-epoch warmup was just merged as the 12th improvement (val 83.64→82.56, a 1.1-point gain). Warmup works by damping early AdamW momentum corruption. The natural next bracket is 2 epochs. The model reaches lr=5e-4 at epoch 3 instead of epoch 2, giving the optimizer more time in the low-lr safe zone before the high-lr cosine descent. With the cosine phase shortened by 1 epoch (T_max=16), the LR floor at eta_min=5e-5 is reached slightly earlier, but epoch 17-18 are already near eta_min so this cost is minimal.

**Mechanism:** Direct extension of the winning warmup mechanism. Tests whether the corruption-damping effect saturates at 1 epoch or whether a second epoch provides incremental benefit.

**Risk:** Very low. If 2ep warmup hurts relative to 1ep, the mechanism saturates at 1 epoch and this axis is closed. Result is clean and interpretable.

**Expected gain:** 0.5–2 val points if mechanism continues; could be null.

**Student instruction:** In `train.py`, change `LinearLR(start_factor=0.01, end_factor=1.0, total_iters=1)` to `total_iters=2`. Change `CosineAnnealingLR(T_max=17, eta_min=5e-5)` to `T_max=16`. Do not change lr, wd, or anything else.

---

## H3 — mlp_ratio=1 (frieren, slug `mlp-ratio-1`)

**Change:** Set `mlp_ratio=1` in the model config (currently `mlp_ratio=2`). No other changes.

**Why:** `mlp_ratio=4` just closed as a null result — wider FFN did not help. The natural reverse bracket is `mlp_ratio=1`, which halves the FFN hidden dimension per attention block. With ~1500 training samples and OOD generalization as the primary challenge, a smaller FFN reduces the capacity for memorization. Per-parameter efficiency often improves for small-dataset OOD regression tasks. This is the cheapest possible model capacity experiment.

**Mechanism:** Tests whether current FFN is over-parameterized for the training set size. If `mlp_ratio=4 > mlp_ratio=2 > mlp_ratio=1`, then capacity is binding and we need a different path to more params. If `mlp_ratio=1 < mlp_ratio=2 <= mlp_ratio=4`, then capacity is NOT the bottleneck and the FFN axis is fully closed. Either outcome is high-information.

**Risk:** Low. Smaller model is slightly faster per epoch, so epoch count may increase marginally within the 30-min budget — this is a free bonus.

**Expected gain:** Uncertain — could be null (consistent with both 4 and 2 being optimal) or a 1–3 point gain from reduced overfit. Most informative to close the axis cleanly.

**Student instruction:** In `train.py`, find `mlp_ratio=2` in the model config dict and change it to `mlp_ratio=1`. Do not change any other hyperparameter.

---

## Evaluation

| Idea | Mode | Mechanistic grounding | Research-state value | Execution value | Priority |
|---|---|---|---|---|---|
| CAWR T_0=9 | Frontier refinement | 3 — targets the monotonically-decaying LR hitting floor as val still descends | 4 — would distinguish "more budget at high LR" from "model needs more epochs" | 3 — trivial change, 30-min budget is the test | HIGH |
| 2-epoch warmup | Frontier refinement | 4 — direct next bracket of the merged 1ep winner | 3 — closes the warmup bracket cleanly | 4 — minimal change, result is fully interpretable | HIGH |
| mlp_ratio=1 | Diagnostic | 3 — closes the FFN capacity axis in the downward direction | 3 — paired with mlp_ratio=4 null result, axis will be fully characterized | 4 — fastest run (fewer params), zero risk | MEDIUM-HIGH |

---

## Ruled-out alternatives (considered and rejected)

- **surf-only fine-tune last 3 epochs**: Novel but requires mask-switching logic. Risk of instability when vol_loss is dropped mid-training. Implementation complexity higher than the expected gain.
- **DropPath rate=0.1**: Interesting but requires model code modification and interacts with existing grad-clip. Better tested after simpler levers are exhausted.
- **Label smoothing / target noise**: Would interfere with the physical interpretation of the metric. Not appropriate for regression tasks where true values are meaningful.
- **Stochastic weight averaging**: Overlaps with EMA (#1540 in flight — highest priority). Do not duplicate.
- **OneCycleLR**: Higher LR peak (up to 1e-3) than current max (5e-4). Could win but has more knobs. Try CAWR first — it has a cleaner mechanism story for this specific failure mode.
