## Hypothesis

**H58: Lion optimizer + GEGLU FFN mega-stack — combine two orthogonal wins from this round.**

PR #3859 (Lion, edward — closed) demonstrated Lion Arm A (lr=1e-4, wd=1e-3, β₁=0.9, β₂=0.99) reaches val_avg=60.30 on the H37b base — a **−5.80 pt gain** over H37b's 66.11. The mechanism (sign-based updates normalizing per-coordinate gradient magnitude) was confirmed: biggest gains landed on `val_single_in_dist` (−9.4) and `val_re_rand` (−4.26), exactly the splits where high-Re/mixed-Re gradient imbalance should hurt most.

PR #3834 (H48 GEGLU, askeladd — merged) demonstrated GEGLU gated FFN reaches val_avg=58.63 on the H37b stack — a **−7.48 pt gain** via multiplicative spatial selectivity near boundary-layer pressure gradients.

**These two wins use completely orthogonal levers:**
- Lion changes the *optimizer*'s step-direction normalization (sign rather than magnitude)
- GEGLU changes the *FFN architecture* (multiplicative gate over expansion path)

If the gains compound (as n_head=2 + lr=2e-3 + wd=5e-5 stacked super-additively in H39 Arm C), the predicted improvement: 58.63 − 5.80 = **~52.8** (lower is better). This would be the biggest single PR gain since the T_max=15 fix.

**Mechanism:** GEGLU's gate `(xW₁) ⊙ σ(xW₂)` is sensitive to update direction. Lion's sign-based step prevents any single coordinate (e.g., a noisy gate weight) from dominating updates — the gate weights and the linear-expansion weights both see uniform-magnitude updates, preserving the gate's selectivity throughout training. AdamW, with its second-moment normalization per coordinate, can saturate gate weights when the gate output gradient is consistently small (a known failure mode of gated FFNs). Lion's sign rule sidesteps this.

**Two arms (LR sensitivity probe in GEGLU context):**
- **Arm A (proven Lion sweet spot):** lr=1e-4, wd=1e-3, β₁=0.9, β₂=0.99, ffn_act=geglu
- **Arm B (intermediate LR):** lr=2e-4, wd=1e-3, β₁=0.9, β₂=0.99, ffn_act=geglu

Arm A reproduces the H49 Arm A winner with GEGLU added. Arm B probes whether GEGLU's gate path needs slightly more update magnitude than the vanilla FFN — gate weights have lower effective gradient magnitude than expansion weights, so a modest LR bump may help. Lion's H49 Arm B at lr=3e-4 *was* worse than lr=1e-4, so we cap at 2e-4 to stay inside the safe region.

**Risk:** Lion + cosine schedule may converge slower than AdamW + cosine in absolute epoch count. At 15-epoch wall budget, this could hide gains. Mitigation: if val_avg at epoch 8 looks behind the H48 GEGLU baseline at epoch 8 by more than 3 pts, that signals undertraining — but per H49 Arm A, Lion's val trajectory tracked AdamW closely from epoch 6 onward.

## Instructions

You'll need to re-add the Lion optimizer to `train.py`. Your prior Lion implementation in PR #3859 is the reference — recreate the `Lion` class and the `--optimizer`, `--beta1`, `--beta2` flags from there.

```python
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]
                # Lion update
                update = (exp_avg * beta1 + grad * (1 - beta1)).sign_()
                p.add_(update, alpha=-group["lr"])
                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss
```

Add `--optimizer {adamw,lion}` (default adamw), `--beta1` (default 0.9), `--beta2` (default 0.999 for adamw, but pass 0.99 for Lion in your commands below).

Then run both arms back-to-back:

```bash
# Arm A — proven Lion sweet spot + GEGLU
cd target/ && python train.py --epochs 50 \
  --experiment_name h58-lion-lr1e4-geglu \
  --agent charliepai2i48h3-edward \
  --optimizer lion --lr 1e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0

# Arm B — intermediate Lion LR + GEGLU
cd target/ && python train.py --epochs 50 \
  --experiment_name h58-lion-lr2e4-geglu \
  --agent charliepai2i48h3-edward \
  --optimizer lion --lr 2e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15 (current merged defaults).

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- Number of epochs completed before wall, best epoch
- Per-epoch val_avg trajectory for both arms — compare to H48 GEGLU's trajectory (AdamW baseline) at the same epoch indices
- **Gate-weight health check at epoch 7 and final epoch**: log the mean and std of the GEGLU sigmoid gate output (`F.sigmoid(x @ W_gate)` mean and std across batch dimension) on a fixed validation minibatch. If gates collapse to ~0 or ~1 (mean far from 0.5 OR std < 0.05), Lion is saturating gates and that's the failure mode to flag.
- Peak GPU memory

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** if val_avg at epoch 3 exceeds 250, kill and report.

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

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=5e-5 + ffn_act=geglu.

**Reference — H49 Arm A (Lion best, AdamW base, PR #3859 closed):** val_avg=60.30, test 3-split=59.02. Lion's contribution at H37b base = **−5.80 pts**.

**Beat this: val_avg/mae_surf_p < 58.6268**

Predicted ≈ 52-55 if gains compound additively. If Arm A lands in the 53-55 range, this is the most significant gain since the T_max fix.

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
