## Hypothesis

**H57: GEGLU + lr=2e-3 mega-stack — combine the gated FFN architecture win with the LR ceiling finding.**

H48 (GEGLU, PR #3834) established a new baseline at val_avg=58.6268 with:
- ffn_act=geglu
- lr=1e-3 (same as H37b baseline)
- n_head=2, wd=5e-5, clip=1.0

H39 Arm C (PR #3683) showed that lr=2e-3 at the n_head=2 + wd=5e-5 + clip=1.0 stack gives val_avg=63.4385 — a −2.67 pt gain over H37b's 66.11.

**These two wins use completely orthogonal levers:**
- GEGLU changes the *architecture* (FFN gating)
- lr=2e-3 changes the *optimizer's peak exploration radius*

If the gains are additive (as they have been throughout this research: n_head=2 + wd=5e-5 + clip stacked super-additively), the predicted improvement: 58.63 - 2.67 ≈ **55.96** (lower is better).

**Mechanism:** GEGLU's multiplicative gating allows the FFN to selectively amplify features most relevant to near-wall pressure gradients. A higher LR (2e-3 vs 1e-3) during the first 6-8 epochs (when the cosine schedule is above 50% peak) lets the model cover more of the loss landscape before annealing locks in. With clip=1.0 still binding (confirmed at lr=2e-3 for the full cosine run), stability is preserved. The combination should work because the LR lever acts on the optimizer trajectory, while GEGLU acts on what information is computed at each step — fundamentally independent.

**Single arm:** We've validated both levers independently. Run one arm: GEGLU + lr=2e-3 at full H48 config.

**Risk:** GEGLU's gating may be sensitive to the update step size. If lr=2e-3 causes the gate weights to overshoot and saturate early (GEGLU sigmoid approaching 0 or 1 for all tokens), the gating loses selectivity and the GEGLU advantage disappears. Watch for: val_avg at epoch 1-3 is comparable to H48 at epoch 3 (GEGLU should converge faster at higher LR), and val_single_in_dist should improve the most (LR was the lever that most helped in-dist in H39).

## Instructions

No code changes needed. `--ffn_act geglu` flag exists since H48 merged.

```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h57-geglu-lr2e3-nhead2-wd5e5 \
  --agent charliepai2i48h3-askeladd \
  --ffn_act geglu \
  --n_head 2 --lr 2e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15 (current merged defaults).

**Report:**
- val_avg/mae_surf_p, per-split breakdown
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- Number of epochs completed before wall, best epoch
- **Pre-clip gradient norms at epochs 1, 3, 7, 13** — confirm clip still binds at lr=2e-3 + GEGLU. If pre-clip norms are much lower (< 0.5) relative to H48's baseline at lr=1e-3, GEGLU's gating is collapsing updates.
- Per-epoch val_avg trajectory (compare to H48's trajectory at same epochs)
- Peak GPU memory (GEGLU adds one extra weight matrix per FFN block; expect slight VRAM increase)

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml`.

**Stop early if diverging:** if val_avg at epoch 3 exceeds 200, kill and report. Indicates lr=2e-3 is unstable with GEGLU. But given H39 Arm C ran lr=2e-3 stably with clip=1.0, this is unlikely.

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

**Beat this: val_avg/mae_surf_p < 58.6268**

Predicted ≈ 55-57 if gains from lr=2e-3 (−2.67 vs H37b) are additive with GEGLU.

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.

**Reproduce H48 GEGLU baseline (lr=1e-3):**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h48-geglu-nhead2-wd5e5 \
  --agent charliepai2i48h3-askeladd \
  --ffn_act geglu \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```
