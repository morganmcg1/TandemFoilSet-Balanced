## Hypothesis

**H59: RMSNorm replacing LayerNorm in the GEGLU Transolver — match the normalization to the gated architecture.**

GEGLU's gate path computes `σ(xW_gate)` which depends on the *direction* of the layer activations more than their absolute mean. LayerNorm subtracts the mean and divides by the standard deviation; RMSNorm only divides by the root-mean-square (no mean subtraction). For gated activations, the mean-subtraction step of LayerNorm can introduce a systematic bias in the gate output: if pre-norm activations have a non-zero mean shifted by a small amount, LayerNorm aggressively centers it, but the gate then sees activations centered around 0 — slowing the gate's ability to select tokens.

RMSNorm preserves the directional structure that the gate exploits. This is a small (parameter-free in the bias dimension) change that's been shown in T5 (Zhang & Sennrich 2019) and Llama (Touvron et al. 2023) to:
1. Reduce per-step compute by ~10-20% (no mean operation)
2. Improve training stability at high LR (less aggressive normalization → smoother loss landscape)
3. Particularly help gated architectures (the gate weights see less distorted input)

**Hypothesis:** GEGLU + RMSNorm gives a 0.5-2 pt val_avg improvement over GEGLU + LayerNorm by preserving the directional signal the gate path uses. Compounding the small benefit with the GEGLU architecture, predicted val_avg ≈ **57-58**.

**Mechanism in this specific dataset:** TandemFoilSet boundary-layer pressure gradients are mathematically captured by spatial derivatives of slowly-varying mean fields with sharp deviations near the wall. LayerNorm's mean subtraction effectively removes the slowly-varying component of the activation pattern, which is informative for the gate to attend to. RMSNorm preserves it. Empirically, RMSNorm has been a small-but-positive lever in most modern Transformer applications.

**Single arm (the question is binary: does RMSNorm help GEGLU?):** ffn_act=geglu + RMSNorm replacing all LayerNorm.

## Instructions

You need to add RMSNorm to `train.py` and expose a `--norm_type` flag.

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
    def forward(self, x):
        # x: [..., dim]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms
```

Then expose `--norm_type {layernorm, rmsnorm}` (default `layernorm`) and propagate it through the Transolver block: replace all `torch.nn.LayerNorm(n_hidden)` instantiations with `RMSNorm(n_hidden) if norm_type == "rmsnorm" else torch.nn.LayerNorm(n_hidden)`. This affects all norm layers inside the Transolver blocks — typically the pre-attention LN and the pre-FFN LN per block.

**Important:** Do not change the FiLM embedding normalization or the input projection norm (if present) — only the *block-level* LayerNorms inside the Transolver layers. Those are the ones interacting with GEGLU.

Run a single arm:

```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h59-geglu-rmsnorm \
  --agent charliepai2i48h3-fern \
  --norm_type rmsnorm \
  --ffn_act geglu \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15.

**Sanity check before running:** Verify RMSNorm forward pass against a known input. For x of shape (2, 4) with values [1.0, 2.0, 3.0, 4.0] repeated, the rms is sqrt((1+4+9+16)/4) ≈ 2.7386, so output[0] should be [1.0/2.7386, 2.0/2.7386, ...] ≈ [0.365, 0.730, 1.095, 1.460] when self.weight is initialized to ones. Print this once at startup to verify.

**Report:**
- val_avg/mae_surf_p, per-split breakdown
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- Number of epochs completed before wall, best epoch
- Per-epoch val_avg trajectory — compare to H48 GEGLU's trajectory at the same epochs (do not need to re-run baseline; reference H48 PR #3834 metrics on advisor branch under `models/model-h48-geglu-nhead2-wd5e5-*/metrics.jsonl`)
- **Speedup measurement**: report mean s/epoch for this run vs H48 GEGLU's s/epoch. RMSNorm should be 5-15% faster per epoch.
- Peak GPU memory (should be ~same or slightly lower — one fewer mean operation per norm layer)
- **Gate output health**: at epoch 13, log mean/std of GEGLU sigmoid output across a fixed validation batch (sanity-check that the gate is still active in [0.1, 0.9] range and not collapsed)

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml`.

**Stop early if diverging:** if val_avg at epoch 3 exceeds 250, kill and report. RMSNorm should be at most slightly different from LayerNorm in the first few epochs.

## Baseline

**Current best — PR #3834 — H48: GEGLU gated FFN (askeladd)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **58.6268** |
| val_single_in_dist/mae_surf_p | 61.6193 |
| val_geom_camber_rc/mae_surf_p | 73.8983 |
| val_geom_camber_cruise/mae_surf_p | 40.4338 |
| val_re_rand/mae_surf_p | 58.5556 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **56.6976** |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=5e-5 + ffn_act=geglu. Default norm: LayerNorm.

**Beat this: val_avg/mae_surf_p < 58.6268**

Predicted ≈ 57-58 (modest gain from cleaner gate-friendly normalization). A larger win (≤57) would be notable and worth pursuing further.

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
