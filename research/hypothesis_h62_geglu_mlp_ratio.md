## Hypothesis

**H62: GEGLU's FFN expansion ratio is under-tuned — sweep mlp_ratio (3, 4) at GEGLU baseline.**

The current mlp_ratio=2 means FFN hidden dim = 2 × n_hidden = 256. With vanilla FFN (`Linear(128, 256) → GELU → Linear(256, 128)`), this is a single 2× expansion. But GEGLU uses 3 matrices: `gate_proj`, `up_proj`, `down_proj`. The "effective expansion" is similar in parameter count to a 2× expansion vanilla FFN with the same hidden dim, but the multiplicative gate uses parameters differently than the additive expand+contract path of vanilla FFN.

The standard convention from Llama / PaLM literature: when switching from vanilla FFN to SwiGLU/GEGLU, the mlp_ratio is increased to compensate for the slightly different parameter efficiency. Llama uses (2/3) × 4 ≈ 2.67 for SwiGLU (vs 4 for vanilla FFN), but the absolute hidden dim is often *increased* in gated architectures. Our mlp_ratio=2 was tuned for vanilla FFN; the GEGLU optimum may be 3 or 4.

**Mechanism:** GEGLU's `(xW_up) ⊙ σ(xW_gate)` produces a vector in the expanded space, then projects back. The expanded representation must encode both the "what" (W_up) and the "where" (W_gate selection). With mlp_ratio=2 (hidden=256), each of these has only 128×256 = 32k parameters. Doubling to mlp_ratio=4 (hidden=512) gives each 65k parameters — more capacity per spatial selectivity decision.

**Two arms:**
- **Arm A: mlp_ratio=3** (FFN hidden = 384) — moderate widening, conservative
- **Arm B: mlp_ratio=4** (FFN hidden = 512) — full Llama-style widening

Both at H48 GEGLU config. Param count roughly: Arm A ≈ 1.1M, Arm B ≈ 1.3M (vs H48's 0.89M).

**Predicted:** Arm A (mlp_ratio=3) ~57-58 likely. Arm B (mlp_ratio=4) is wider — may be slower per epoch within wall budget, but more capacity could win 0.5-1 pt.

**Risk:** Wider FFN = slower per-step compute, fewer epochs in wall budget. If Arm B completes <13 epochs (vs H48's 13), comparison is unfair. Report epochs_completed prominently.

## Instructions

The current FFN expansion factor is likely a constant (`mlp_ratio=2` or `mlp_hidden = 2 * n_hidden` hardcoded in the Transolver block). You'll need to expose it as a flag.

Add `--mlp_ratio` CLI flag (default 2 to preserve current behavior). Propagate it to the Transolver block's FFN:

```python
# Inside the Transolver block FFN construction
ffn_hidden = mlp_ratio * n_hidden  # was: 2 * n_hidden hardcoded

if ffn_act == "geglu":
    self.gate_proj = nn.Linear(n_hidden, ffn_hidden, bias=False)
    self.up_proj = nn.Linear(n_hidden, ffn_hidden, bias=False)
    self.down_proj = nn.Linear(ffn_hidden, n_hidden, bias=False)
# else: vanilla FFN with same ffn_hidden
```

Run both arms:

```bash
# Arm A — mlp_ratio=3 at GEGLU base
cd target/ && python train.py --epochs 50 \
  --experiment_name h62-geglu-mlp3-nhead2-wd5e5 \
  --agent charliepai2i48h3-askeladd \
  --ffn_act geglu --mlp_ratio 3 \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0

# Arm B — mlp_ratio=4 at GEGLU base
cd target/ && python train.py --epochs 50 \
  --experiment_name h62-geglu-mlp4-nhead2-wd5e5 \
  --agent charliepai2i48h3-askeladd \
  --ffn_act geglu --mlp_ratio 4 \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15 (current merged defaults).

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- **Epochs completed before wall** (CRITICAL — wider FFN may complete fewer epochs)
- Best epoch
- Per-epoch val_avg trajectory — overlay against H48 GEGLU (mlp_ratio=2) trajectory
- **Parameter count** for each arm (vs H48 0.89M baseline)
- Peak GPU memory and mean s/epoch (wider arm will be slower)
- Final-epoch comparison vs H48 at same epoch (account for fewer completed epochs)

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report.

## Baseline

**Current best — PR #3834 — H48: GEGLU gated FFN (askeladd, mlp_ratio=2)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **58.6268** |
| val_single_in_dist/mae_surf_p | 61.6193 |
| val_geom_camber_rc/mae_surf_p | 73.8983 |
| val_geom_camber_cruise/mae_surf_p | 40.4338 |
| val_re_rand/mae_surf_p | 58.5556 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **56.6976** |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=5e-5 + ffn_act=geglu + **mlp_ratio=2** (current default).

**Beat this: val_avg/mae_surf_p < 58.6268**

Predicted: Arm A (mlp_ratio=3) ≈ 57-58. Arm B (mlp_ratio=4) ≈ 57-58 if compute budget holds.

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
