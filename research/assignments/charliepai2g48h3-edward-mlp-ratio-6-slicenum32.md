# mlp_ratio=6 on new compact stack: explore unexplored midpoint of MLP capacity axis

## Hypothesis

The mlp_ratio axis on this codebase has been tested at:
- **mlp_ratio=2**: +9.95% worse (PR #2007, OLD n_layers=6 stack) — speedup too small to unlock epoch budget; capacity loss hurt
- **mlp_ratio=4** (current default, PR #1408): chosen as baseline; combined with GeGLU=mlp_ratio*2/3 effective expansion
- **mlp_ratio=8**: +5.95% worse (PR #1872, GeGLU stack) — fc2 expansion beyond 256 channels adds noise pathways
- **mlp_ratio=6**: UNTESTED — the unexplored midpoint between current 4 and lost 8

The current model on the new compact stack (n_layers=4 + slice_num=32) has only **667,923 params** — a 31.6% reduction from the original 976,827 baseline. The model's per-epoch time has dropped from 138s to 74s. We've consistently observed `best_epoch=final` indicating the model is **budget-limited**, not capacity-limited at training time.

But the val splits still show substantial room:
- geom_camber_rc: 56.766 val (hardest split)
- single_in_dist: 44.963 val

These suggest more representation capacity *could* help if added without breaking the epoch budget. mlp_ratio=6 is a targeted ~25% MLP expansion that:
- Adds ~24% to total params (667K → ~830K)
- Adds ~8-10% to per-epoch wall-clock (~74s → ~80s)
- Still fits ~22 epochs in 30-min budget at T_max=22

**Two predictions:**
1. **If mlp_ratio=6 wins:** the new compact stack was under-capacity in the MLP dimension; capacity can be selectively restored where it pays off. Would then test mlp_ratio=5 for finer tuning.
2. **If mlp_ratio=6 loses or ties:** mlp_ratio=4 is at or near the optimum; the MLP capacity axis on the new compact stack saturates at 4. Confirms a clean U-shape with 2 and 8 both worse.

**Mechanism rationale:** GeGLU's gated structure means each MLP has two parallel projections — `fc1_a, fc1_b` (one becomes the gate, one becomes the value, multiplied), then `fc2`. Width expansion in `fc1` increases gate diversity; if the gates were collapsing or saturating on the compact model, more gate dimensions help. The fact that mlp_ratio=8 lost suggests there's a sweet spot somewhere in [4, 8); we test 6.

## Instructions

Change ONLY `--mlp_ratio` from 4 to 6. Use `--slice_num 32` and `--n_layers 4` to match the current baseline. Set `--epochs 22` to give the slightly slower training one extra cosine-aligned epoch beyond baseline's 21 (since you'll likely have ~80s/epoch instead of 74s, 22 epochs ≈ 29.3 min).

If `--mlp_ratio` is not yet a CLI argument, the student should add it to the Config dataclass (default value 4) and pass it through `model_config['mlp_ratio']` or however it's plumbed (check `train.py` lines around `model_config`). Use the same plumbing pattern as `slice_num` (added in PR #2108).

Note: `n_layers` defaults to 5 in train.py — you MUST pass `--n_layers 4`.

**Budget guardrail:** If epoch 1 takes >85s, the 30-min cap may cut you before epoch 22. Report however many epochs completed.

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-edward \
  --experiment_name mlp-ratio-6-slicenum32-nlayers4 \
  --epochs 22 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 4 \
  --slice_num 32 \
  --mlp_ratio 6
```

### Reporting requirements

1. Per-split val and test `mae_surf_p` vs current baseline (val=42.815 / test=36.899)
2. Per-split `mae_vol_p` — diagnostic for whether MLP expansion benefits volume reconstruction
3. Per-epoch wall-clock — confirm timing estimate
4. Best epoch, total wall-clock, epochs completed
5. Parameter count (should be ~830K with mlp_ratio=6)
6. Peak memory

**Specifically check:** Did `geom_camber_rc` improve disproportionately? (it's the hardest split and would benefit most from more capacity). Did `geom_camber_cruise` regress? (it reacted poorly to capacity changes in past experiments — would suggest cruise prefers smaller models).

## Baseline (PR #2108: n_layers=4 + slice_num=32 + T_max=21, mlp_ratio=4)

| Split | val `mae_surf_p` | test `mae_surf_p` |
|---|---|---|
| single_in_dist | 44.963 | 40.717 |
| geom_camber_rc | 56.766 | 51.074 |
| geom_camber_cruise | 25.476 | 21.158 |
| re_rand | 44.053 | 34.646 |
| **avg** | **42.815** | **36.899** |

**Target to beat:** `val_avg/mae_surf_p < 42.815`

Baseline reproduce:
```bash
cd target/ && python train.py --epochs 21 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10 --n_layers 4 --slice_num 32
```

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path-to-jsonl>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
