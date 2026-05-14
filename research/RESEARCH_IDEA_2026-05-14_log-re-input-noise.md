# Round 138 — Input feature noise on log_Re (σ=0.05) during training

## Hypothesis

Add Gaussian noise σ=0.05 to the `log_Re` input feature during TRAINING ONLY (no noise at eval). Tests whether train-time data augmentation in CONDITIONING-VARIABLE space regularizes in_dist over-specialization by making the model robust to small shifts in Reynolds number.

## Motivation

After 5 routing-axis closures (#2884, #2923, #2934, #2944, #2955, #2958) all LOSS via the same "block-3 routing softness" failure mode, plus the head-axis fully closed (3/3 LOSS), the meta-signal (cruise WIN / in_dist LOSS) clearly lives at a level deeper than any single REPRESENTATION-axis structural lever can reach.

Fresh angle: attack at the CONDITIONING-VARIABLE level. `log_Re` is the primary conditioning input that drives FiLM modulation, slice-routing, and downstream attention. Adding small Gaussian noise to log_Re during training:
1. Smooths the model's response surface w.r.t. Re
2. Should improve OOD generalization on `re_rand` (random Re held out) and `single_in_dist` (specific Re extremes)
3. Different mechanism than anything tried (data-axis intervention at the SCALAR conditioning level, not coord-space mixup of #2918)

Cf. #2918 input-mixup-0.2 LOSS catastrophic (mixed input features + masks across samples — non-physical). This PR is much more conservative: noise on ONE scalar feature, σ small, only at training, no cross-sample mixing.

## Architecture

```python
# In training forward (before passing log_Re to model):
if model.training:
    log_Re_noise = torch.randn_like(log_Re) * 0.05  # σ=0.05 in log_Re space
    log_Re_train = log_Re + log_Re_noise
else:
    log_Re_train = log_Re

# Pass log_Re_train through the rest of the model
```

Effect: σ=0.05 in log_Re space → multiplicative noise of ~exp(0.05) ≈ 5.1% in linear Re. Small enough to remain physically plausible but large enough to regularize.

Zero new params, ~3 lines of code.

## Falsifiable predictions

- **WIN** (val < 30.5605): Conditioning-variable robustness helps. Especially likely to help re_rand. Suggests log_Re sensitivity is part of the meta-signal mechanism. Try σ=0.02 and σ=0.10 sweeps.
- **PARTIAL** (val ≈ 30.5605 ± 1%): Mild regularization, marginal effect.
- **LOSS** (val > 31.0): Conditioning-variable noise hurts — either σ too large, or in_dist/cruise specialization is NOT driven by log_Re sensitivity. Closes input-feature-augmentation axis at this magnitude.

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-edward \
    --experiment_name "charliepai2g48h5-edward/log-re-input-noise" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

## Reporting

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown
3. **Especially close attention to re_rand (random Re held-out split)** — is it most-improved? If WIN concentrates on re_rand, log_Re sensitivity hypothesis confirmed.
4. Param count (unchanged 407,940)
5. Train→val gap at convergence (does noise make training harder?)
6. **Meta-signal check:** does cruise WIN / in_dist LOSS pattern repeat, attenuate, or invert?
7. Plain-language verdict: WIN / PARTIAL / LOSS
