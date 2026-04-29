# SENPAI Research State
- 2026-04-29 (updated: PR #948 n_hidden=192 merged — new best 93.0126, sub-93 is next target; 6 new experiments assigned to all students)
- No directives from the human researcher team
- **Current research focus**: Round 4 — stacked wins: `surf_weight=20` + `per_sample_norm_mse` + `lr=2e-4` + `grad_clip=1.0` + `n_hidden=192`. Current best is PR #948 at **93.0126**. Active experiments probe weight decay sensitivity, depth scaling (n_layers=6/7), surf_weight tuning, and output clamping for the val_single_in_dist bottleneck. **Sub-90 is the next target.**
- **Pre-flight checklist for new assignments**: branch must be cut from advisor HEAD (post-#948) so `--n_hidden` arg exists; reproduce command must explicitly include `--surf_weight 20.0 --lr 2e-4 --grad_clip 1.0 --loss_kind per_sample_norm_mse --n_hidden 192`.

## Baseline

- **val_avg/mae_surf_p: 93.0126** (PR #948 — nezuko/n-hidden-192)
- Architecture: `n_hidden=192`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2` (~1.12M params)
- Training: `--surf_weight 20.0 --lr 2e-4 --grad_clip 1.0 --loss_kind per_sample_norm_mse --weight_decay 1e-4 --epochs 12`
- Cosine annealing T_max=12 (properly calibrated to realistic epoch budget)
- Reproduce: `cd target/ && python train.py --surf_weight 20.0 --lr 2e-4 --grad_clip 1.0 --loss_kind per_sample_norm_mse --n_hidden 192 --epochs 12`

## Key Technical Findings

- **grad_clip=1.0 is mandatory**: Pre-clip gradient norms were 35–105× over max_norm every epoch — the sub-100 breakthrough came from PR #871. All new experiments must include `--grad_clip 1.0`.
- **n_hidden=192 is the new best**: PR #948 improved 95.66 → 93.01 (~2.7%). All new experiments must use `--n_hidden 192`.
- **T_max confound**: `CosineAnnealingLR(T_max=MAX_EPOCHS)` where default is 50 epochs. With 30-min timeout only 10-14 epochs run. Fix: pass `--epochs N` where N matches realistic budget.
- **Per-epoch timing**: n_hidden=128 n_layers=5 ~120s/epoch; n_hidden=192 n_layers=5 ~150s/epoch; n_hidden=192 n_layers=6 ~185s/epoch; n_hidden=192 n_layers=7 ~200s/epoch
- **surf_weight=20 is mandatory for fair comparison**: all new experiments must include `--surf_weight 20.0`
- **per_sample_norm_mse is mandatory**: all new experiments must include `--loss_kind per_sample_norm_mse`
- **lr=2e-4 is the best LR found**: `--lr 2e-4` required in all new experiments
- **NaN bug**: test_geom_camber_cruise sample 20 has -inf ground truth → NaN in scoring.py; val splits unaffected
- **Re-stratified sampling is redundant**: per-sample normalized loss already achieves Re-balancing — explicit sampler creates conflict (PR #839 dead end)
- **surf_weight=40 hurts volume**: no net gain (PR #849 confirmed dead end)
- **best=last pattern**: PR #871 best at epoch 14/14 — model still improving at cutoff; PR #948 best at epoch 10/12 — suggests epochs=12 may be slightly over-budget at n_hidden=192
- **val_single_in_dist bottleneck**: still the largest single split contributor (~31% of average, 107.8467 at baseline) — output clamping experiment (#994) targets this
- **per_sample_norm_mse already implicitly balances channels**: explicit per-channel weighting (PR #876) degraded all metrics; the loss naturally normalizes by per-sample variance which covers channel scale differences

## Active WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #988 | edward   | Zero weight_decay (0.0) on n_hidden=192 best config — tests if per_sample_norm_mse makes L2 redundant | wip |
| #989 | tanjiro  | weight_decay 1e-4 → 1e-5 on n_hidden=192 best config — lighter regularization test | wip |
| #990 | frieren  | n_layers=6 on n_hidden=192 best config (epochs=9) — depth test on wider model | wip |
| #991 | alphonse | surf_weight=25 on n_hidden=192 best config — surface/volume trade-off with wider model | wip |
| #994 | fern     | Output clamping ±6σ on n_hidden=192 best config — targets val_single_in_dist 107.8467 bottleneck | wip |
| #995 | nezuko   | n_layers=7 on n_hidden=192 best config (epochs=8) — deeper capacity test with wider hidden dim | wip |

## Merged / Closed History

| PR | Student | Hypothesis | Outcome |
|----|---------|------------|---------|
| #738 | edward   | surf_weight 10→20 | MERGED — 128.83 (first baseline) |
| #812 | edward   | lr 5e-4→2e-4 + surf_weight=20 | MERGED — 112.94 (−12.3%) |
| #845 | fern     | per_sample_norm_mse + surf_weight=20 | MERGED — 105.96 (−6.2%) |
| #868 | edward   | per_sample_norm_mse + lr=2e-4 + surf_weight=20 | MERGED — 102.94 (−2.85%) |
| #871 | askeladd | grad_clip=1.0 + lr=2e-4 + per_sample_norm_mse | MERGED — 95.66 (−7.1%, sub-100!) |
| #948 | nezuko   | n_hidden=192 capacity test on full best config | MERGED — 93.01 (−2.7%) |
| #741 | frieren  | Wider FFN mlp_ratio 2→4 | CLOSED — +9.9% regression |
| #735 | alphonse | Wider hidden dim 128→256 + surf_weight=20 | CLOSED — fork conflict |
| #736 | askeladd | More slices 64→128 + surf_weight=20 | CLOSED — 135.96 |
| #746 | tanjiro  | More attention heads 4→8 + surf_weight=20 | CLOSED — 128.96, essentially tied |
| #802 | edward   | bf16 + batch_size=8 | CLOSED — 129.14, regression |
| #849 | askeladd | surf_weight 20→40 + T_max=15 | CLOSED — 121.41, pre-norm result |
| #839 | tanjiro  | Re-stratified mini-batch sampling | CLOSED — 137.63, +30% regression |
| #813 | frieren  | Zero weight decay (old assignment, superseded) | N/A — reassigned |
| #876 | nezuko   | Per-channel pressure weighting (w_p=5.0) | CLOSED — 109.39, +14.4% vs best; per_sample_norm already balances channels implicitly |
| #870 | alphonse | n_layers=6 with per-sample norm | CLOSED — 110.99 on stale config (no grad_clip, lr=5e-4) |
| #874 | fern     | Checkpoint averaging across final K=5 epochs | CLOSED — 110.20 averaged vs 108.29 single-best; SWA premise doesn't hold under 14-epoch cosine |
| #946 | askeladd | SGDR warm restarts (T_0=7, T_mult=1) | CLOSED — +1.32% regression; model not in local minimum |
| #932 | edward   | Zero weight_decay on post-#871 config | SUPERSEDED — rebased as #988 on post-#948 HEAD |
| #933 | tanjiro  | weight_decay 1e-5 on post-#871 config | SUPERSEDED — rebased as #989 on post-#948 HEAD |
| #935 | frieren  | Extended training epochs=18 on post-#871 config | SUPERSEDED — replaced by n_layers=6 (#990) on post-#948 HEAD |
| #956 | alphonse | surf_weight=25 on post-#871 config | SUPERSEDED — rebased as #991 on post-#948 HEAD |
| #958 | fern     | Output clamping ±6σ on post-#871 config | SUPERSEDED — rebased as #994 on post-#948 HEAD |

## Potential Next Research Directions

After the active WIP experiments complete, promising next ideas include:

1. **n_hidden=256 revisit** — prior PR #735 had a fork conflict, not a real result; now worth retrying on post-#948 HEAD with correct config
2. **n_head=6 or n_head=8** — more attention heads on the wider model (n_hidden=192); prior n_head=8 was on stale config (128.96), might be better now
3. **surf_weight=15** — if surf_weight=25 wins, also try 15 to bracket the optimum
4. **AdamW → SOAP/Muon optimizer** — modern optimizers may converge faster within the epoch budget
5. **Input feature enrichment** — curvature, arc-length gradients, local normals as extra node features
6. **Cross-attention surface/volume** — dedicated cross-attention between surface and volume node sets
7. **Physics constraints** — soft continuity equation loss as auxiliary signal (e.g., soft divergence-free constraint on velocity)
8. **lr=1e-4 with n_hidden=192** — test lower learning rate on wider model; wider models sometimes benefit from smaller steps
9. **Cosine annealing eta_min tuning** — askeladd (#973) tests eta_min=1e-5/1e-6; results pending
10. **slice_num=128** — more physics slices; prior #736 was on old config without norm/grad_clip/wider hidden
