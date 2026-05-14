# Round 136 — LayerScale γ_attn=0.85 / γ_mlp=0.93 (asymmetric init at converged values)

## Hypothesis

**Initialize γ_attn at 0.85 and γ_mlp at 0.93** (the converged values observed in #2928's γ=1.0 trajectory). Asymmetric init matches the natural operating regime that the model learned to settle at — skipping the budget-wasting "growth" phase from 1e-4 baseline AND from 1.0 (#2928).

This is student of #2928's followup #1, directly motivated by the diagnostic finding: γ_attn and γ_mlp converge to DIFFERENT values (asymmetric), and both end BELOW their #2928 init of 1.0 AND ABOVE the baseline 1e-4.

## Why this might WIN

1. **Student of #2928 explicitly recommended this.** Verbatim: *"Asymmetric γ_init — set gamma_attn init lower than gamma_mlp init (e.g. 0.5 vs 1.0, or keep both small but with different ratios) to match the asymmetric final operating regime. The strong attn/mlp γ asymmetry (0.86 vs 0.93) is the most interesting signal here."*

2. **Mechanistically targeted to a measured phenomenon.** Per-block, attention residuals end at γ ≈ 0.86 and MLP residuals at γ ≈ 0.93. Initializing at these values means the model SPENDS NO BUDGET on γ-convergence — all gradient signal goes to the actual weights.

3. **Compared to baseline γ_init=1e-4:** saves the "growth from 1e-4 to 0.85-0.93" phase. From the #2928 trajectory, γ reached 0.95+ by ep5 and 0.85-0.93 by ep30. Starting at the natural value should free up ep1-30 budget for body learning.

4. **Compared to #2928 γ_init=1.0:** matches the "muted block contribution" preference of the model. Starting at 1.0 forced the model to suppress block output (which was the original CaiT motivation for 1e-4 — but 1e-4 over-corrected, and 1.0 under-corrected).

5. **Zero new params, single-init change.** Pure init recipe gap test.

## Why this might LOSS

1. **The per-block converged values differ.** Block 2 had γ_attn=0.92 (vs other blocks' 0.85-0.89), and block 2 had γ_mlp=0.92 (vs other blocks' 0.93-0.95). Setting uniform per-branch may not capture the per-block preference.

2. **Converged value is the EQUILIBRIUM, not the optimal training trajectory.** The model may need to PASS THROUGH a different γ regime during training (e.g., higher γ early for fast learning, lower γ later for refinement). Pre-init at equilibrium skips this trajectory.

3. **Asymmetry may be over-determined by other model interactions.** The 0.86 vs 0.93 asymmetry could be a side-effect of the attention/MLP gradient magnitude difference, not an intrinsic preference. The model might re-equilibrate to slightly different values if started from a different init.

## Falsifiable predictions

- **WIN** (val < 30.5605): Skipping γ-convergence frees training budget. Validates the diagnostic-guided init approach.
- **PARTIAL** (test WIN, val WASH): Mirrors #2928's pattern — γ_init matters but the trade-off lives elsewhere.
- **WASH** (val ≈ 30.5605 ± 0.3%): Converged-value init doesn't differ from learnable-from-1e-4 init meaningfully. Suggests #2928's followup #3 (freeze γ at converged) would also WASH.
- **LOSS** (val > 31.0): Pre-init at equilibrium hurts. The trajectory through γ-space is load-bearing.

## Implementation

### Step 1: Locate the LayerScale γ init in `train.py`

Find where `gamma_attn` and `gamma_mlp` are initialized in `TransolverBlock`. The current code likely has something like:

```python
self.gamma_attn = nn.Parameter(torch.ones(n_hidden) * 1e-4)
self.gamma_mlp  = nn.Parameter(torch.ones(n_hidden) * 1e-4)
```

### Step 2: Replace with asymmetric converged-value init

```python
self.gamma_attn = nn.Parameter(torch.ones(n_hidden) * 0.85)  # attn converged ~0.86
self.gamma_mlp  = nn.Parameter(torch.ones(n_hidden) * 0.93)  # mlp converged ~0.93
```

Apply uniformly to all 4 blocks (no per-block variation in this experiment — that's a future axis to test if this WINs).

### Step 3: Startup diagnostics

```python
print(f"LayerScale init: γ_attn=0.85 (vs baseline 1e-4), γ_mlp=0.93 (vs baseline 1e-4)")
print(f"vs #2928 init γ=1.0: started at 1.0, drifted DOWN to 0.85-0.95 by ep60")
print(f"vs baseline init γ=1e-4: starts at 1e-4, presumably drifts UP toward 0.85-0.93")
print(f"This init: starts AT the converged operating regime")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # 407,940 unchanged
```

### Step 4: Per-epoch γ tracking (key diagnostic)

Log mean γ_attn and γ_mlp at each block at ep1, 5, 10, 30, 60. Watch for:
- Do γ values DRIFT AWAY from the init? If yes, this init wasn't quite right.
- Do they STAY at the init? If yes, this confirms equilibrium init.
- Per-block divergence: does block 2 drift differently from blocks 0/1/3?

## Baseline (PR #2879)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **30.5605** |
| test_avg/mae_surf_p | **26.5160** |
| val_single_in_dist | 23.3997 |
| val_geom_camber_rc | 46.0708 |
| val_geom_camber_cruise | 17.8657 |
| val_re_rand | 34.9057 |
| Param count | 407,940 |

For comparison:
- Baseline #2879 (γ_attn = γ_mlp = 1e-4): val 30.5605, test 26.5160
- #2928 (γ_attn = γ_mlp = 1.0): val 30.9444 (+1.26%), test 26.4847 (-0.12%)
- This PR (γ_attn = 0.85, γ_mlp = 0.93): converged-value init

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-nezuko \
    --experiment_name "charliepai2g48h5-nezuko/layerscale-gamma-asymmetric" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160 AND vs #2928 (γ=1.0) 30.9444 / 26.4847
2. Per-split val + test breakdown
3. **γ trajectory:** mean γ_attn and γ_mlp at each block at ep1, 5, 10, 30, 60. Does the model STAY at the asymmetric init, or drift?
4. Param count confirmation (~407,940)
5. Epochs completed (target: 60), sec/epoch, peak GPU memory
6. Train→val loss gap at convergence
7. **Init regime comparison:** how does this init's trajectory compare to #2928's? Both should converge to the same equilibrium if init doesn't matter, or different equilibria if init shapes outcome.
8. **Plain-language verdict:** WIN (skipping γ-convergence helps) / WASH (init at equilibrium is equivalent to converging-to-equilibrium) / LOSS (trajectory through γ-space matters).

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
