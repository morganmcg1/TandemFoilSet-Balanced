## Hypothesis

**H56: Lower grad clip — clip=0.5 and clip=0.7 at the H39 Arm C stack (n_head=2 + lr=2e-3 + wd=5e-5).**

H20 confirmed clip=1.0 wins over no-clip; H40 confirmed clip=2.0/3.0 regress vs clip=1.0. But ALL prior clip sweeps were at lr=1e-3, where pre-clip grad norms peak ~3-5 at epoch 1. **At lr=2e-3 the binding ratio has shifted**: H39 Arm C reported pre-clip norms 7→1.5 across epochs (binds even at epoch 13-15), meaning at the higher LR more updates are scaled down by clipping. The natural follow-up is: at lr=2e-3 where clipping is binding throughout the entire training run, does tightening the clip further help by *enforcing* even more conservative step sizes, effectively decoupling the optimizer's exploration radius from peak LR?

**Mechanism:** At lr=2e-3 with clip=1.0, effective per-step weight changes are bounded by `lr · 1.0 = 2e-3`. Lowering clip to 0.5 cuts the effective max step to `lr · 0.5 = 1e-3` — equivalent to lr=1e-3 with clip=1.0 in step magnitude, BUT preserves the higher LR's effect on (a) the wd shrinkage rate (`-lr · wd · θ`), (b) the cosine schedule shape (lr decay still indexed by peak=2e-3), and (c) Adam's second-moment bias correction at early steps. This is a *fine-grained* exploration of the lr×clip product surface.

**Why this could win:**
1. **Higher LR's exploration benefit + tighter clipping's stability.** The H39 monotone LR trend (1e-3→1.5e-3→2e-3 monotone-improving) suggests higher LR helps optimization geometry. But the wd-coupled shrinkage at higher LR may be *too aggressive*. Tighter clip caps the destabilizing updates without removing the higher peak LR's coupling effects.
2. **Equivalent effective step size, different schedule.** clip=0.5 at lr=2e-3 = clip=1.0 at lr=1e-3 in update magnitude, but the cosine peak is 2× higher, so the implicit warmup-like effect of the schedule's high portion may matter independently of clip.
3. **Anti-overshoot insurance.** H51 (lr=2.5e-3 and 3e-3 push, alphonse) tests upward LR ceiling. H56 tests the orthogonal lever: *what if 2e-3 is right but updates are still too large?*

**Risk:** If clip is too tight (e.g., 0.3), the optimizer can't make progress at all — gradients of mostly the same direction get truncated to the same magnitude regardless of confidence, so the model under-trains. clip=0.5 is at the boundary of "tight but still functional" based on standard practice.

**Two arms test clip values below 1.0:**

- **Arm A — clip=0.7 + lr=2e-3 + n_head=2 + wd=5e-5**: Conservative tightening. Effective step max = 1.4e-3.
- **Arm B — clip=0.5 + lr=2e-3 + n_head=2 + wd=5e-5**: Aggressive tightening. Effective step max = 1e-3, matching H37b's effective step but with lr=2e-3's schedule shape.

Predicted val_avg if mechanism holds ≈ 62.5-64.5. If both regress vs H39 Arm C (63.44), clip=1.0 at lr=2e-3 is the optimum and lowering it just under-trains the model. If Arm A wins (clip=0.7), it suggests the optimum is between 0.5 and 1.0 — recommend a follow-up clip=0.8.

## Instructions

No code changes needed. `--clip_grad_norm` flag already exists.

**Arm A — clip=0.7:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h56-clip07-nhead2-lr2e3-wd5e5 \
  --agent charliepai2i48h3-frieren \
  --n_head 2 --lr 2e-3 --weight_decay 5e-5 --clip_grad_norm 0.7
```

**Arm B — clip=0.5:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h56-clip05-nhead2-lr2e3-wd5e5 \
  --agent charliepai2i48h3-frieren \
  --n_head 2 --lr 2e-3 --weight_decay 5e-5 --clip_grad_norm 0.5
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15 (current merged defaults).

**Report per-arm:**
- val_avg/mae_surf_p, per-split breakdown
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- Number of epochs completed before wall, best epoch
- **Pre-clip gradient norms at epochs 1, 7, 13, 15** (critical — show the actual norm distribution vs the clip threshold; we need to see how often the clip binds)
- **Fraction of steps where clip binds** (i.e., pre-clip norm > clip threshold). H39 Arm C was 100% binding throughout — does it stay 100% at clip=0.5?
- Per-epoch val_avg trajectory (does the tighter clip slow convergence early on?)
- Peak GPU memory

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**No early stopping needed:** clip=0.5 is conservative; the model won't diverge. The risk is only that convergence stalls — let it run to wall.

## Baseline (pending merge)

**Current best — PR #3683 — H39 Arm C: n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0 (thorfinn)** (in rebase, will merge shortly)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p (best of 2 seeds)** | **63.4385** |
| val_avg/mae_surf_p (2nd seed) | 65.5093 |
| test_avg/mae_surf_p (3-split, excl. cruise, best seed) | **61.3910** |
| Best epoch | 15/50 (cut by timeout) |

**Beat this: val_avg/mae_surf_p < 63.44 (best) or < 64.47 (mean of 2 seeds)**

Prior baseline (merged) — **PR #3629 — H37b: n_head=2 + lr=1e-3 + clip=1.0**:
| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p | 66.1060 |
| test_avg/mae_surf_p (3-split) | 64.4522 |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=1e-4 (default), AdamW (β₁=0.9, β₂=0.999).

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.

**Reproduce H39 Arm C baseline (lr=2e-3, clip=1.0):**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h39c-nhead2-lr2e3-wd5e5-clip1 \
  --agent charliepai2i48h3-frieren \
  --n_head 2 --lr 2e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```
