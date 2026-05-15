# Hypothesis: naca-fourier-stacked (thorfinn)

## Hypothesis
NACA camber/thickness Fourier features, stacked on the Huber loss baseline. In round-3,
thorfinn's NACA Fourier features showed val_avg=123.35 against fresh-slate MSE baseline.
This was worse than baseline (129.99→123.35 = the result _was_ better! it was 5.1% better than baseline,
just not better than the round-3 winner frieren's 107.46).

Wait - actually let me reconsider: 123.35 is worse than baseline 129.99 in the sense that 
lower is better. 123.35 < 129.99 means 123.35 IS BETTER. So thorfinn's result DID beat
the fresh-slate baseline by 5.1%. It just didn't beat frieren's Huber (107.46).

The hypothesis: geometric Fourier features provide OOD geometry generalization that is
orthogonal to Huber loss stability. Huber addresses the gradient noise problem; NACA
features address the geometric representation problem. They should stack.

**Predicted improvement:** −3 to −7 on val_avg/mae_surf_p vs 107.46.

## Instructions

### 1. Use your round-3 feature branch as a starting point

Your branch `willowpai2i24h3-thorfinn/naca-camber-fourier-features` still exists. 
**Rebase it onto the current advisor branch** (`icml-appendix-willow-pai2i-24h-r3`)
to incorporate Huber loss (now default in train.py):

```bash
git fetch origin
git checkout willowpai2i24h3-thorfinn/naca-camber-fourier-features
git rebase origin/icml-appendix-willow-pai2i-24h-r3
```

Resolve any conflicts (likely minimal since NACA feature code and Huber code touch
different parts of train.py).

### 2. Run the experiment

```bash
cd target/ && python train.py \
    --wandb_group naca-fourier-stacked \
    --wandb_name naca-fourier-huber \
    --agent willowpai2i24h3-thorfinn
```

### 3. Key things to report

- Per-split val, especially `val_geom_camber_rc` and `val_geom_camber_cruise` — these
  are the OOD geometry splits that NACA features are designed to help with.
- Compare the per-split improvement vs baseline to see if geometry features help
  the OOD splits specifically.

## Baseline

- **val_avg/mae_surf_p:** 107.4641 (frieren PR #3248, Huber δ=2.0)
- **test_avg_nansafe/mae_surf_p:** 101.9848
- **Per-split val:** single_in_dist=127.91, geom_camber_rc=118.48, geom_camber_cruise=83.35, re_rand=100.11
- **W&B run:** `mp8s8okf`
- **Reproduce baseline:** `cd target/ && python train.py --wandb_group huber-robust-loss --wandb_name huber-delta2 --agent willowpai2i24h3-frieren`
