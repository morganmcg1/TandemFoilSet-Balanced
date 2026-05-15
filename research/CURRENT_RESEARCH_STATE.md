# SENPAI Research State

- **Date:** 2026-05-15
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.

## Current research focus

Round 3 of the willow-pai2i-48h cycle. The advisor branch was just freshly cut and no canonical baseline run exists yet on this branch state. The primary ranking metric is `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 validation tracks), with `test_avg/mae_surf_p` as the paper-facing number.

The fleet of 8 idle students is being assigned a first round of 8 single-variable experiments spanning four families of intervention:

1. **Model capacity** — does the small Transolver under-fit?
   - `alphonse`: wider hidden + more heads (128→192, 4→6)
   - `nezuko`: depth (5→8 layers)
   - `tanjiro`: MLP ratio (2→4)
2. **Optimization recipe**
   - `askeladd`: LR warmup + higher peak (5e-4→1e-3 with 3-epoch linear warmup)
3. **Loss formulation**
   - `edward`: per-channel loss weighting (upweight p by 3x)
   - `fern`: Huber loss instead of MSE
   - `frieren`: per-sample loss normalization (equal-weight per sample, not per node)
4. **Inputs**
   - `thorfinn`: Fourier position features on (x, z) + slice_num bump 64→96

Each PR runs **dual-arm** (baseline + variant in same wandb_group) so we simultaneously establish baseline metrics for this branch state AND get clean A/B attribution per hypothesis.

## Potential next research directions

After round-1 results come in:

- **Stack winners.** If multiple families of intervention win, the next round should combine compatible winners (e.g., wider model + Huber loss + p-upweighted loss).
- **Loss reformulations beyond Huber.** Log-cosh, gradient-matching (Sobolev), frequency-domain MSE, signed-loss for high-magnitude regions.
- **Position-aware tricks.** RoPE on slice tokens, learned global tokens, NeRF-style multiresolution hash encoding.
- **Slice-side experiments.** Slice attention variants (cross-attention between slices), slice-token initialization, slice dropout, slice-num scaling > 96.
- **Better samplers.** Re-stratified sampling within the balanced sampler, harder-domain upweighting, curriculum from low-Re to high-Re.
- **Data augmentation.** Reflection / sign-flipping augmentation (CFD has reflection symmetry across the chord), modest geometry / scale jitter.
- **Optimizer alternatives.** Lion, Shampoo / SOAP, Muon.
- **EMA + SWA.** Exponential moving average and stochastic weight averaging at the end of cosine.
- **Mixed precision.** bf16 training to free VRAM for bigger batches / wider models.
- **Surface-conditioned heads.** Separate output MLP for surface vs volume nodes.
