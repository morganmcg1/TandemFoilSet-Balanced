# Assignment: frieren — Specialized surf/vol decoders + epochs=50 COMPOUND

**Branch (use exactly):** `charliepai2g48h3-frieren/surfvoldecoders-epochs50-nlayers2-slicenum16`

**Base branch:** `icml-appendix-charlie-pai2g-48h-r3`

## Hypothesis

Your #2883 result validated the specialized-decoders mechanism cleanly:
- Cosine similarity between final-layer surf/vol decoder weights = **0.0396** (orthogonal — two distinct functions, not redundant copies)
- Asymmetric L2 norms (vol 1.70 vs surf 1.13, +51%) consistent with vol readout needing larger magnitude for wider per-channel ranges
- Test improved on **all 4 splits** (avg −1.08% vs new baseline 29.916)
- Val regressed +1.73% vs new baseline 34.544 — but the trajectory was **still descending at epoch 46** (best_epoch=46/46), same epoch-budget binding-constraint pattern as the old #2468 baseline before #2872 unlocked the epochs=50 plateau

**Hypothesis (compound):** The specialized-decoders mechanism is **orthogonal** to the epochs=50 schedule extension. Both add capacity along independent dimensions:
- epochs=50 (#2872): more training-budget headroom — unlocks the plateau at e47-50 from a still-descending e46 trajectory
- specialized_decoders (#2883): per-node-type readout structure — orthogonal final-layer decomposition

Compounding them tests whether specialized decoders can find a deeper plateau than 34.544 when given the proper training budget. The #2883 trajectory (still descending at e46) suggests the val regression was an **early-stopping artifact**, not a fundamental loss.

## Why compound, why now

- **Both mechanisms are mechanistically independent.** Epoch extension changes *when* training stops; specialized decoders change *what gradient flow each readout sees*. No interaction risk.
- **Test wins on all 4 splits at e46 already** — extending to e50 plausibly converts the val "loss" (within seed noise) into a clear win as the plateau is reached.
- **The plateau is at e47-50 specifically** — your old run never reached it. With +5.4% per-epoch overhead, you may converge slightly later, but the schedule has +4 epochs of headroom.

## Implementation

Re-use your #2883 specialized_decoders implementation **exactly as-is** — no code changes from that PR. Just run it on the epochs=50 schedule.

Your existing code (verified working in #2883):
```python
if cfg.specialized_decoders:
    self.surf_decoder = nn.Sequential(
        nn.Linear(self.n_hidden, self.n_hidden), nn.GELU(),
        nn.Linear(self.n_hidden, len(output_fields)),
    )
    self.vol_decoder = nn.Sequential(
        nn.Linear(self.n_hidden, self.n_hidden), nn.GELU(),
        nn.Linear(self.n_hidden, len(output_fields)),
    )
# Forward:
surf_pred = self.surf_decoder(h)
vol_pred = self.vol_decoder(h)
pred = surf_pred * surf_mask.unsqueeze(-1) + vol_pred * vol_mask.unsqueeze(-1)
```

## Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name surfvoldecoders-epochs50-nlayers2-slicenum16 \
  --epochs 50 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 --slice_num 16 \
  --specialized_decoders true
```

## Baseline to beat

PR #2872 (n_layers=2 + slice_num=16 + **epochs=50**, best_epoch=47, **shared decoder, no SWA**) — current best:

| Metric | Value |
|---|---:|
| **val_avg/mae_surf_p** | **34.544** |
| val_single_in_dist | 35.113 |
| val_geom_camber_rc | 48.106 |
| val_geom_camber_cruise | 18.895 |
| val_re_rand | 36.060 |
| **test_avg/mae_surf_p** | **29.916** |

## Per-run constraints

- Hard timeout: 30 min (`SENPAI_TIMEOUT_MINUTES=30`).
- **Wall-clock risk:** #2883 was 37s/epoch (+5.4% vs 35.1s baseline). 50 × 37s ≈ 30.8 min — slightly over the 30-min cap.
- **If the run gets cut at e48-49**, that's still informative — the plateau in #2872 was [e47, e48, e49, e50] = [34.544, 34.794, 34.638, 34.646], so we'd capture the bulk of plateau even with 1-2 epochs cut.
- Save metrics to JSONL after every epoch (you already do this) so partial results are recoverable.
- **In your final result, report:**
  - The actual `best_epoch` reached
  - Whether the run was cut by timeout
  - Per-epoch val_avg for the last 4-5 epochs (so we can see the plateau, or lack of one)
- Hard epoch cap: `SENPAI_MAX_EPOCHS` (do not override).
- **Local JSONL metrics only.** Do NOT log to W&B.
- Branch only from `icml-appendix-charlie-pai2g-48h-r3`.

## Terminal result format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/.../metrics.jsonl"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best_val_avg>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test_at_best_val>}}
```

Include:
1. Full per-split val and test table for the **best_epoch checkpoint**
2. Decoder cosine similarity diagnostic (should still be ~0 if mechanism stable)
3. Per-epoch val_avg trajectory for last 4-5 epochs (to confirm plateau)
4. Total wall-clock + epochs actually completed (whether timeout cut occurred)

## Decision criteria

- **Win (val < 34.544):** Compound confirmed; the mechanism stacks. Suggested follow-up: try widened vol_decoder (asymmetric capacity per your #2883 suggestion) on the epochs=50 stack.
- **Neutral (|Δ| < 0.5 val):** Mechanism is real but doesn't compound — single-mechanism wins (epochs=50) already extracted the plateau. Close axis.
- **Loss (val > 34.544 by >0.5):** Specialized decoders impose an objective cost not recouped by training budget. Close axis. Note: this would mean your #2883 was likely just lucky on test splits.

## EV assessment

**High.** This is the cleanest possible compound test: two independent mechanisms, both validated separately at this exact n_layers=2 + slice_num=16 + surf_weight=10 stack. The #2883 test wins on all 4 splits are a real signal worth chasing on the proper training schedule. Worst case: confirms specialized decoders don't compound (closes one axis decisively). Best case: 1-3% additional val improvement on top of the new baseline 34.544.
