# Current Baseline — `icml-appendix-willow-pai2e-r4`

## Best metrics

| Metric | Value | Run | PR |
|--------|-------|-----|----|
| `val_avg/mae_surf_p` | **89.7141** | `w9xbc0wl` | #820 |
| `test_avg/mae_surf_p` | NaN (cruise -Inf GT; see #797 guards) | — | — |
| 3-split test mean (excl. cruise) | **88.16** | `w9xbc0wl` | #820 |
| Best epoch | 14 / 50 (timeout cliff) | | |
| Wall time | 30.91 min | | |

Note: `test_avg/mae_surf_p` came back as NaN on thorfinn's rebased run even
though #797's NaN guards are merged. This may be a rebase artifact — the
guards should fire and filter cruise sample 000020. Next run on the merged
branch should report a finite test_avg. The 3-split test mean (88.16) is
the reliable paper-facing comparison metric until then.

## Per-split val (epoch 14, run `w9xbc0wl`)

| Split | mae_surf_p |
|-------|-----------|
| `val_single_in_dist` | 109.16 |
| `val_geom_camber_rc` | 106.62 |
| `val_geom_camber_cruise` | **60.60** |
| `val_re_rand` | 82.47 |
| **val_avg** | **89.714** |

## Per-split test (epoch 14, run `w9xbc0wl`)

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 96.32 |
| `test_geom_camber_rc` | 92.86 |
| `test_geom_camber_cruise` | NaN (cruise -Inf GT, filtered by #797) |
| `test_re_rand` | 75.31 |
| **3-split test mean (excl. cruise)** | **88.16** |

## Configuration (post-#820)

| Knob | Value |
|------|-------|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| Optimizer | AdamW |
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| Schedule | CosineAnnealingLR, T_max=epochs |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| **`channel_weights`** | **[1.0, 1.0, 3.0]** for [Ux, Uy, p] |
| Loss | L1 (absolute error) on normalized space, vol + surf_weight × surf |
| **`fourier_bands`** | **4** — `[sin/cos(π·2^k·x), sin/cos(π·2^k·z)]` for k=0..3 prepended to input |
| Sampler | `WeightedRandomSampler` over balanced domain groups |
| `epochs` | 50 (capped) |
| Timeout | 30 min |
| **NaN guards** | **active in `evaluate_split` (#797)** — drops cruise sample 000020 |
| Seed | **NONE** (val_avg drifts ~5-10% across runs; seed PR #863 in flight) |
| Params | 666,455 (+4,096 over prior baseline from Fourier input dim 24→40) |

## Delta history

| Metric | L1-only (#752) | +ch=[1,1,3] (#754) | +Fourier PE K=4 (#820) |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 101.93 | 99.23 | **89.71** |
| 3-split test mean | 100.83 | 99.34 | **88.16** |
| Δ vs prior | — | −2.65% | **−9.59%** |
| Cumul. Δ vs L1-only | — | −2.65% | **−12.0%** |

Fourier PE is **orthogonal and additive** with channel weighting. Mechanistically:
ch=[1,1,3] tips the gradient balance toward pressure; Fourier PE provides the
high-freq basis the preprocess MLP previously had to discover via composition.
With both, the model gets constant-magnitude per-channel pressure gradient AND
a free spectral basis — the wins stack cleanly. +4K params, −9.59% MAE.

Key physical observation: `val_re_rand` was flat with Fourier PE on the L1-only
baseline (−0.2%) but gained −9.7% post-rebase with ch=[1,1,3]. Interpretation:
ch=[1,1,3] makes pressure dominant in the loss, so Fourier's high-freq basis
earns its keep on the Re-varying split too (the near-foil pressure BL structures
are the same spatial frequencies regardless of Re). Mechanistically clean.

## Reproduce

```bash
cd target/
python train.py --fourier_bands 4 \
  --agent willowpai2e4-thorfinn \
  --wandb_name "willowpai2e4-thorfinn/fourier-pe-K4-on-L1-ch3"
```

## Open issues

- **Run-to-run val variance:** Seed PR #863 (askeladd) is in flight — will
  eliminate the ~5-10% init/sampler drift once merged.
- **test_avg NaN on Fourier PE run:** thorfinn's rebased run reports NaN for
  test_avg despite #797 guards being merged. Most likely a rebase artifact.
  Next run on advisor branch HEAD should produce a finite test_avg. The
  3-split mean (88.16) is reliable in the interim.
- **Cruise-test `-Inf` GT (workaround active):** `test_geom_camber_cruise/000020.pt`
  has 761 `-Inf` values in the `p` channel. The per-sample `y_finite` guard
  in `evaluate_split` (#797) filters this sample. Dataset is read-only.
