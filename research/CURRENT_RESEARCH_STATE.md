# SENPAI Research State
- 2026-04-29 15:55
- No recent research directions from human researcher team (no GitHub issues found)
- Current research focus: Major breakthroughs achieved via lr=1e-3 + grad_clip=1.0 + budget-aware cosine + surf_weight=25. Baseline dropped from 121.89 → 100.41 (-17.6%). Now testing BF16 AMP (expected +5 epochs in budget), OneCycleLR superconvergence, LR warmup, surf_weight=50, per-field heads, eta_min, combined best config. Weight decay confirmed: wd=1e-4 optimal, wd=1e-3 over-regularizes (+0.49% regression). tanjiro now assigned FiLM Re-conditioning.

## Current Baseline

**val_avg/mae_surf_p = 100.41** (PR #1098, charliepai2f2-tanjiro, lr=1e-3 + grad_clip=1.0 + budget-aware cosine + DropPath 0→0.1 + surf_weight=25, epoch 14/50)

Per-split breakdown (PR #1098 — current best):

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|-----------------|
| single_in_dist | 120.68 | 104.32 |
| geom_camber_rc | 111.80 | 98.04 |
| geom_camber_cruise | 75.99 | 63.06 |
| re_rand | 93.15 | 88.91 |
| **avg** | **100.41** | **88.58** |

Note: NaN guard (`torch.nan_to_num` + sample-level finite filter) now standard in all new experiments. test_geom_camber_cruise corrupted GT sample `000020.pt` is fully mitigated by per-sample NaN guard in train.py.

## Key Insights

**1. Systematic Timeout/LR Mismatch (critical, partially fixed)**
All experiments use `CosineAnnealingLR(T_max=50)` but the 30-min timeout means only ~14 epochs complete. LR barely decays throughout training. PR #1091 (nezuko, MERGED) demonstrated a budget-aware dynamic estimate (T_max~11 after timing warm-up epochs) and achieved -4.5% improvement. edward (PR #1126) is testing a fixed T_max=14; alphonse (PR #1143) bundles T_max=14 with lr=1e-3 + grad_clip.

**2. lr=1e-3 + grad_clip=1.0 is confirmed as transformative (MERGED)**
PR #1098 (tanjiro, merged) delivered val_avg/mae_surf_p=100.41 with lr=1e-3 + grad_clip=1.0 on top of PR #1091 baseline (DropPath + budget-aware cosine + surf_weight=25). This is a -17.6% improvement — the largest single gain in this programme. All subsequent experiments now build on this as the base config.

**3. Throughput > capacity under 30-min budget**
slice_num=128 (PR #1087 closed): +60% epoch time, only 9 epochs, regression. n_layers=8 (PR #1089 closed): +58% epoch time, only 9 epochs, regression. n_hidden=192 (PR #1086 iter 2 closed): +55% epoch time, only 9 epochs, regression. n_layers=6 (PR #1161 closed): +18% epoch time, only 12 epochs, 27% regression. General rule: any change that slows epoch time by >20% is likely fatal in the 30-min budget. Throughput gains (BF16, batch=8) are positive; capacity expansions without speed recovery are negative.

**4. BF16 as a "free" throughput gain (technique VALIDATED, re-testing on current stack)**
H100 has dedicated BF16 tensor cores. PR #1144 confirmed ~26% speedup: 131s/epoch → ~97s/epoch, yielding ~19 epochs vs 14 within the 30-min budget. That experiment used the stale pre-PR#1091 baseline (127.67), so val=122.39 doesn't beat current 100.41. PR #1184 (askeladd) is retesting BF16 on the full current stack (lr=1e-3 + grad_clip=1.0 + DropPath 0→0.1 + budget-aware cosine). No GradScaler needed for BF16. Implementation: `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` + `pred.float()` before loss.

**5. Per-sample instance normalization failed catastrophically**
PR #1129 (askeladd, closed): Per-channel std normalization down-weights pressure (largest std channel), causing 3x worse results (val=366.76). Root cause: per-sample physical-space std conflates between-sample and between-channel scaling. Physical unit scales must be anchored globally.

**6. DropPath 0→0.1 is optimal for the epoch-limited regime (0→0.2 is confirmed negative)**
PR #1091 (merged): DropPath linear schedule (0.0→0.1 across 5 TransolverBlocks) + budget-aware cosine LR improved from 127.67 → 121.89 (-4.5%). PR #1156 (nezuko, closed): DropPath 0→0.2 gave val=126.17 — 25.8% worse than current baseline 100.41, and worse than even the 121.89 pre-tanjiro baseline. Root cause: 14-epoch budget causes underfitting with rate 0.2. **DropPath 0→0.1 is the confirmed sweet spot for this training regime.**

**7. Weight decay wd=1e-4 confirmed optimal; wd=1e-3 over-regularizes**
PR #1178 (tanjiro, closed): Increasing wd from 1e-4 to 1e-3 gave val=100.90 — a +0.49% regression vs baseline 100.41. OOD splits showed the strongest regression (re_rand +2.18, geom_camber_cruise +1.96), consistent with over-regularization hurting generalization. The existing DropPath 0→0.1 provides sufficient implicit regularization in the 14-epoch budget. **wd=1e-4 is the confirmed optimal weight decay; do not increase further.**

## Active Experiments

| PR | Student | Hypothesis | Key Config | Status |
|----|---------|-----------|------------|--------|
| #1090 | frieren | Per-field output heads: separate MLP decoder for Ux, Uy, p | n_hidden=128, 3 separate heads | Running (started ~6h ago) |
| #1143 | alphonse | Combined best config: lr=1e-3 + grad_clip=1.0 + T_max=14 + surf_weight=25 | All of the above stacked | Running (stale baseline 127.67) |
| #1152 | thorfinn | CosineAnnealingLR eta_min=1e-5: non-zero LR floor for final epochs | eta_min=1e-5 (was 0), T_max=14 | Running (stale baseline 127.67) |
| #1166 | edward | LR warmup + cosine annealing: 2-epoch linear warmup into cosine decay | LinearLR(1e-6→1e-3) then CosineAnnealingLR(T_max=12) + DropPath 0→0.1 | Running |
| #1182 | fern | surf_weight 25→50: stronger surface loss focus on primary metric | surf_weight=50, grad_clip=1.0, all else baseline | Running |
| #1184 | askeladd | BF16 AMP on current stack: ~19-20 epochs in 30-min budget | torch.autocast bf16 + lr=1e-3 + grad_clip=1.0 + DropPath 0→0.1 + budget-aware cosine | Running |
| #1195 | nezuko | OneCycleLR superconvergence: replace cosine anneal within 14-epoch budget | OneCycleLR(max_lr=1e-3, total_steps=14*steps_per_epoch) + full current stack | Running |
| #1199 | tanjiro | Re-conditioning on Reynolds number: FiLM conditioning with log(Re) | log(Re) → FiLM inject into each layer; targets re_rand split (hardest OOD axis) | Running |

Note: PRs #1143 (alphonse) and #1152 (thorfinn) were started before PRs #1091 and #1098 merged and use stale baseline 127.67. When they complete, results will likely beat their stated baseline but not current 100.41. Will evaluate whether to close as superseded or request rebase when results arrive.

Note: PR #1126 (edward, T_max=14) was superseded by PR #1166 (edward, LR warmup + cosine). PR #1185 (nezuko, SGDR) was CLOSED as negative — val=108.69; restart spikes incompatible with 14-epoch budget. Replaced by PR #1195 (OneCycleLR).

## Closed / Dead Ends

- PR #1087 (askeladd): slice_num=128 — 4% regression; slower per-epoch kills epoch budget
- PR #1086 Iters 1+2 (alphonse): width expansion n_hidden=192/256 — 204s/epoch too slow; epoch starvation dominates
- PR #1129 (askeladd): per-sample instance-normalized loss — 3x regression (val=366.76); per-channel std down-weights pressure
- PR #1089 (fern): n_layers=8 — 207s/epoch, only 9 epochs, regression; also used stale surf_weight=10
- PR #1161 (fern): n_layers=6 — 158s/epoch, only 12 epochs, 27% regression (127.56 vs 100.41); throughput-over-capacity dominates
- PR #1102 (thorfinn): mlp_ratio 2→4 — val=136.16, 6% regression; wider feedforward slows epoch time, fewer epochs in budget
- PR #1156 (nezuko): DropPath 0→0.2 — val=126.17, 25.8% regression vs 100.41; 14-epoch budget causes underfitting at higher drop_path rates
- PR #1178 (tanjiro): weight decay wd=1e-3 — val=100.90 (+0.49% regression); OOD splits hurt most (re_rand +2.18, cruise +1.96); DropPath 0→0.1 already provides sufficient implicit reg; wd=1e-4 confirmed optimal
- PR #1144 (askeladd): BF16 AMP (stale baseline) — val=122.39 beats stale baseline 127.67 but not current 100.41; technique validated (26% speedup), re-testing on current stack in PR #1184
- PR #1185 (nezuko): SGDR warm restarts (T_0=5, T_mult=1) — val=108.69 (best ep 14); restart spikes at ep 6 (+22) and ep 11 (+47) waste epochs re-converging; restart cycles incompatible with 14-epoch budget

## Merged Winners (Chronological)

1. **PR #1088** (edward, surf_weight 10→25): val_avg/mae_surf_p = 127.67 (-∞ from unlogged start)
2. **PR #1091** (nezuko, DropPath 0→0.1 + budget-aware CosineAnnealingLR): val_avg/mae_surf_p = **121.89** (-4.5%)
3. **PR #1098** (tanjiro, lr=1e-3 + grad_clip=1.0): val_avg/mae_surf_p = **100.41** (-17.6%) — **current baseline**

## Potential Next Research Directions

Highest priority after current WIPs resolve:

1. **BF16 AMP on current stack** (PR #1184 in flight): 19-20 epochs vs 14 could push well below 100.41 — highest expected-value follow-up
2. **OneCycleLR superconvergence** (PR #1195 in flight): Smith & Topin (2019) 1-cycle policy designed for tight epoch budgets; aggressive anneal post-peak may improve convergence in 14 epochs
3. **Re-conditional embedding**: Embed log(Re) separately and inject into each layer via FiLM conditioning; Re is the primary OOD axis (re_rand split is the hardest at 93.15)
4. **Adaptive loss weighting (uncertainty-weighted)**: Learn log-variance per output channel as in Kendall & Gal (2018); balances velocity and pressure gradients dynamically
5. **Fourier features for mesh positions**: Random Fourier features for irregular mesh node coordinates; improves spatial awareness at low model cost
6. **Ensemble / checkpoint averaging**: Average last K checkpoints at no additional training cost; well-validated trick for final performance
7. **Cross-attention surface↔volume**: Dedicated attention stream for surface-to-volume interaction; addresses the biggest remaining gap (surface prediction lags volume in some splits)
8. **Data augmentation**: Re-scaling perturbation, geometry jitter — improves generalization across all OOD splits at no inference cost
9. **Graph neural networks**: Message passing along mesh connectivity edges; alternative inductive bias to attention for CFD
10. **DropPath fine-grained sweep (0.05, 0.075)**: Student (nezuko) suggested post-PR #1156; lower rates may yield marginal gains over 0.1 — lower priority given other directions
