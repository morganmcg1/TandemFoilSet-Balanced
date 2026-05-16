# Round-4 Research Ideas — 2026-05-16 04:30

Baseline: val_avg/mae_surf_p = 83.4954, test_avg/mae_surf_p = 73.7918 (PR #3632)
Stack: n_hidden=160, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, Fourier PE num_freq=4, coord_noise_std=0.01, lr=5e-4, L1 loss, surf_weight=10.0

---

## Idea 1: lr-1e3-coord-noise

**Name/slug:** `lr-1e3-coord-noise`

**Hypothesis:**
The current baseline was won at lr=5e-4 (Config default), but the previous best Fourier PE baseline (#3372) used lr=1e-3. The coord noise win (#3632) was never tested at lr=1e-3. Since coord noise acts as a regularizer (reducing effective capacity), it may tolerate a higher learning rate better — allowing the optimizer to reach a lower loss in the same number of epochs. Combining lr=1e-3 with coord noise is explicitly flagged as an open experiment in BASELINE.md with the note "expected to compound."

**Concrete implementation:**
Single flag change — no code modification needed:
```bash
python train.py --epochs 10 --lr 1e-3 --coord_noise_std 0.01
```
Config.coord_noise_std is already 0.01 by default; just pass `--lr 1e-3` to override the 5e-4 default.

**Expected gain:** −2–5% val (approximately recovering the gap between Fourier PE at lr=1e-3 vs lr=5e-4)

**Risk level:** Low — both lr=1e-3 and coord_noise are individually validated; this is a composition test only.

---

## Idea 2: feature-noise-aug

**Name/slug:** `feature-noise-aug`

**Hypothesis:**
Coord noise augments the spatial position (x,z) of mesh nodes, improving robustness to geometric variation. But the flow condition features — log(Re) (dim 13), AoA foil1 (dim 14), AoA foil2 (dim 18), and NACA geometry (dims 15-17, 19-21) — are never augmented. Adding small Gaussian noise to these condition features during training acts as a continuous data augmentation over the flow/geometry parameter space, pushing the model to learn smoother, more interpolation-friendly representations. This is especially relevant for the OOD splits (geom_camber_rc, geom_camber_cruise, re_rand) where the model must generalize to unseen parameter values.

**Concrete implementation:**
Add after the existing coord noise block in the training loop (around line where coord noise is applied):

```python
# Feature noise augmentation on flow condition dims
COND_COLS = list(range(2, 24))  # dims 2-23 (saf, dsdf, is_surface, Re, AoA, NACA, gap, stagger)
if cfg.feature_noise_std > 0:
    pad_mask = mask.unsqueeze(-1).to(x_norm.dtype)
    feat_noise = torch.randn_like(x_norm[..., COND_COLS]) * cfg.feature_noise_std * pad_mask
    x_norm = x_norm.clone()
    x_norm[..., COND_COLS] = x_norm[..., COND_COLS] + feat_noise
```

Add to Config:
```python
feature_noise_std: float = 0.0
```

Add CLI arg:
```python
parser.add_argument("--feature_noise_std", type=float, default=cfg.feature_noise_std)
```

Run command:
```bash
python train.py --epochs 10 --feature_noise_std 0.005
```

Note: is_surface (dim 12) is boolean and should ideally be excluded from noise. A cleaner slice would target dims 2-11 (saf+dsdf) and 13-23 (Re, AoA, NACA, gap, stagger) but adding noise to a boolean that's already 0/1 with std=0.005 has negligible effect. If results are mixed, next step is to target only continuous features.

**Expected gain:** −1–3% val (primarily on OOD splits geom_camber_rc, geom_camber_cruise)

**Risk level:** Low-Med — augmentation in feature space is well-established; the exact std needs tuning.

---

## Idea 3: longer-training-12ep

**Name/slug:** `longer-training-12ep`

**Hypothesis:**
Val curves are still descending at epoch 10 for every experiment in this research programme. The 30-min budget at ~170s/epoch yields exactly 10 epochs. However, the SENPAI_TIMEOUT_MINUTES env var is a hard wall-clock cap — not epoch cap. If we set `--epochs 12` and leave SENPAI_TIMEOUT_MINUTES at 30, training will run up to 12 epochs but halt at 30 min (~10.5 epochs at 170s/epoch). The key insight: setting T_max=12 in the cosine scheduler means the LR anneals more slowly over 12 epochs, giving better coverage of the loss landscape even in the same wall-clock time. The best checkpoint is saved whenever val improves, so any epochs that complete are captured.

**Concrete implementation:**
Single flag change:
```bash
python train.py --epochs 12
```

The scheduler already uses `T_max = cfg.epochs` so this automatically adjusts the cosine decay slope. With 30-min budget and ~170s/epoch, expect ~10-11 epochs to complete. The LR will be higher for longer (slower decay), which may help navigate out of local minima that 10-epoch cosine reaches.

If budget allows, also test `--epochs 14` for a slower cosine tail with a follow-up run.

**Expected gain:** −1–3% val (free — zero architecture change, just a longer cosine tail)

**Risk level:** Low — only risk is that the cosine LR doesn't fully decay, which is already the case at epochs=10. The best checkpoint is always checkpointed regardless.

---

## Idea 4: mlp-ratio-4

**Name/slug:** `mlp-ratio-4`

**Hypothesis:**
The Transolver FFN (MLP inside each TransolverBlock) currently uses mlp_ratio=2, meaning the inner dimension is 2×n_hidden = 320. Standard transformer practice uses mlp_ratio=4 (inner dim 4×n_hidden = 640). This is the single largest capacity lever not yet tested — it nearly doubles the FFN parameters, adding ~830K parameters (from ~1.03M to ~1.86M total), still well within VRAM. The FFN in transformers is where feature interactions are learned; at mlp_ratio=2 the model may be bottlenecked in its capacity to represent complex flow patterns. The 30-min budget should still allow ~8-9 epochs at the higher parameter count.

**Concrete implementation:**
Single flag change (if mlp_ratio is exposed as CLI arg) or Config change:

Check if `--mlp_ratio` CLI arg exists; if not, add it:
```python
# In Config dataclass, change default:
mlp_ratio: int = 2  # keep default; override via CLI

# Add CLI arg:
parser.add_argument("--mlp_ratio", type=int, default=cfg.mlp_ratio)
cfg.mlp_ratio = args.mlp_ratio

# Pass to model:
model_config = dict(
    ...,
    mlp_ratio=cfg.mlp_ratio,  # already there in current code
    ...
)
```

Run command:
```bash
python train.py --epochs 10 --mlp_ratio 4
```

If per-epoch time exceeds ~185s (10 epochs would exceed budget), fall back to `--epochs 8` — the scheduler will still fully anneal over 8 epochs.

**Expected gain:** −2–5% val (significant upside; FFN capacity is a known bottleneck in narrow transformers)

**Risk level:** Med — adds complexity; may slow training enough to reduce effective epochs, or may not improve if the model is already bottlenecked elsewhere.

---

## Idea 5: surf-weight-tuning

**Name/slug:** `surf-weight-sweep`

**Hypothesis:**
The current surf_weight=10.0 was set early (pre-L1, pre-Fourier-PE, pre-coord-noise) and has never been revisited. Since then, the model has become much better calibrated. The primary metric is surface-pressure MAE, so upweighting surface nodes in the loss directly optimizes the ranking metric. However, too high a surf_weight may cause the model to sacrifice volume accuracy in a way that harms surface generalization (overfitting to training surface patterns). The sweet spot for surf_weight may have shifted with the improved baseline stack. Testing surf_weight=20 was tried on an old stack (val=111.92) but that was before L1, Fourier PE, and coord noise — the interaction is unknown. Testing surf_weight=15 is a modest step that's likely safe.

**Concrete implementation:**
Single flag change:
```bash
python train.py --epochs 10 --surf_weight 15
```

If val improves, follow up with surf_weight=20. If val degrades, follow up with surf_weight=7 to check if the current 10 is already too high.

Note: The previous surf_weight=20 test (#3095) used MSE loss on an old stack — the L1 loss landscape is fundamentally different and that result should not be used as a proxy.

**Expected gain:** −1–3% val (direct metric alignment)

**Risk level:** Low-Med — the lever is well-understood; the risk is primarily that surf_weight=20 caused a regression on an older stack, but L1 loss changes the scale relationship.

---

## Idea 6: n-head-8

**Name/slug:** `n-head-8`

**Hypothesis:**
The current Transolver uses n_head=4 with n_hidden=160, giving head_dim=40. Standard transformer practice suggests head_dim=32-64 is optimal; head_dim=40 is fine but increasing to n_head=8 (head_dim=20) may allow more specialized attention patterns at the cost of narrower per-head representations. More importantly, with n_head=8 the slice_num=64 maps to 8 slices per head, which may allow the PhysicsAttention mechanism to discover 8 distinct physics "modes" per attention head rather than 4. This is unexplored — every architecture experiment so far has been on n_layers and n_hidden, not n_head.

**Concrete implementation:**
Check if `--n_head` CLI arg exists; if not, add it:
```python
# In Config dataclass:
n_head: int = 4  # keep default; override via CLI

# Add CLI arg:
parser.add_argument("--n_head", type=int, default=cfg.n_head)
cfg.n_head = args.n_head

# Pass to model:
model_config = dict(
    ...,
    n_head=cfg.n_head,
    ...
)
```

Run command:
```bash
python train.py --epochs 10 --n_head 8
```

Note: n_head=8 requires n_hidden divisible by 8 (160/8=20 — valid). Parameter count change is minimal (only Q/K/V projection shapes change slightly). Per-epoch time should remain ~170s.

**Expected gain:** −1–3% val (more attention specialization in PhysicsAttention)

**Risk level:** Med — head_dim=20 may be too narrow for complex flow patterns; the prior experiment history shows no n_head experiments so this is genuinely unexplored.

---

## Priority ranking

1. **lr-1e3-coord-noise** — explicitly flagged as untested and "expected to compound" in BASELINE.md; highest confidence, lowest risk
2. **longer-training-12ep** — free improvement via LR schedule adjustment; zero architecture change
3. **feature-noise-aug** — extends the augmentation philosophy of coord noise to the condition space; directly targets OOD splits
4. **mlp-ratio-4** — largest untested capacity lever; high upside if FFN is the bottleneck
5. **surf-weight-sweep** — direct metric alignment; surf_weight never revisited post-L1/Fourier-PE
6. **n-head-8** — genuinely unexplored attention axis; medium confidence but orthogonal to all prior work
