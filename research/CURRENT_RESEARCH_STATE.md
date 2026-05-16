# SENPAI Research State

- **Date:** 2026-05-16 08:30 UTC (Round 4 in progress on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Current research focus

**Six axes confirmed and merged.** Compound stack: Huber + bf16 + cosine T_max=15 + EMA decay=0.999 + single-shot FiLM + two-shot FiLM.

**Current best: 89.784 val_avg/mae_surf_p** (frieren #3584, two-shot FiLM on full stack, 2026-05-16 04:50)

### 🔥 Two huge unmerged signals on stale stacks — both need composition verify

- **🔥 T_max=20 vs T_max=15 (thorfinn #3390 R2):** R1 showed Arm C (bf16+T_max=20) hits **88.229** — *better than current 89.784* but on bf16-only stack. Mechanism: T_max=15 was sized for fp32 14-epoch budget; on bf16's 19-epoch budget it hits LR floor at ep 16 wasting 3 epochs. T_max=20 keeps cosine arc decaying through ep 19. Sent back for verify on full FiLM stack. Predicted Arm B: **80-85** if compose, **86-89** if sub-additive.
- **🔥 Schedule-Free AdamW (alphonse #3594 R2):** R1 showed −20.75% intra-PR (71.492 vs 90.207) on pre-two-shot-FiLM stack. Parallel attack on same LR-floor problem. Sent back for verify on current full stack.

**Both T_max=20 and SF-AdamW attack the same problem: cosine T_max=15 floors out early under bf16's 19-epoch budget.** Whichever has the larger paired Δ on full FiLM stack will dominate the merge order.

### Other in-flight composition tests

- **n_hidden=192 (nezuko #3492 R2):** R1 showed −2.99% intra-PR (93.989 vs 96.886) on pre-FiLM stack. Mechanism: "better inductive bias at same fit" (not raw memorization). Sent back for FiLM-stack verify. Predicted: **86.5-88.5** if compose.
- **Gradient clipping clip=1.0 (tanjiro #3511 R2):** −4.03% on pre-FiLM stack; sent back for rebase on FiLM stack, expected ~86-88 if additive.
- **Three-shot FiLM (frieren #3681):** preprocess injection as third FiLM site, expected −1-3%.
- **Slice-num sweep (edward #3684):** test slice_num=32/64/96 on full stack.
- **n_layers=4 (fern #3758):** depth axis ablation; 4 layers faster per epoch → more fine-tune epochs in 30-min budget.
- **SDF input features (askeladd #3777):** per-node distance-to-nearest-surface; orthogonal to FiLM, expected −1-3%.

**Fourier features CLOSED (subsumed by FiLM, #3117 R4):** Paired Δ −0.10% at R4 (was −9.10% at R2, −3.16% at R3).
**Batch size CLOSED (askeladd #3365):** bs=4 best on bf16-only stack; bigger batches cut SGD steps without reducing sec/epoch (GPU compute-bound).
**LR axis closed:** 5e-4 is optimal for bf16+T_max=15.

**Primary metric:** `val_avg/mae_surf_p` (lower is better)

## Merged winners (chronological)

| PR | Hypothesis | val_avg delta | New val_avg |
|----|-----------|------------|---------|
| #3094 | Huber loss (alphonse) | −15.7% vs MSE | 111.531 |
| #3290 | bf16 AMP (askeladd) | −8.98% vs Huber | 101.519 |
| #3289 | Cosine T_max=15 (thorfinn) | −10.3% vs Huber fp32 / −1.4% vs bf16 | 100.059 (fp32) |
| #3126 | EMA decay=0.999 Karras-ramp (nezuko) | −1.06% vs bf16+T_max=15 arm | 96.464 |
| #3122 | FiLM conditioning — log Re, AoA, NACA, gap, stagger (frieren) | −4.00% vs EMA baseline | 92.606 |
| #3584 | Two-shot FiLM — attn + MLP per block, shared module, +0 params (frieren) | −3.05% vs FiLM baseline | **89.784** |

## Falsified / closed hypotheses

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #3278 | Per-channel p-upweighting | +3-21% regression | Domain variance > channel bias |
| #3364 | lr=1e-3+warmup on bf16 | +8.3% regression | bf16 noise amplifies higher LR |
| #3321 | lr=1e-3/1.5e-3+warmup on fp32+bf16 | +2.2-12% regression (6 arms) | Higher LR dead end, two-seed confirmed |
| #3443 | lr ∈ {2.5e-4, 3.5e-4} on bf16+T_max=15 | +0.78-2.60% regression | LR axis closed: 5e-4 is magnitude optimum |
| #3595 | n_layers=6 vs 5 on full FiLM stack | +2.47% regression | Depth costs wall-clock epochs; 6 layers net-negative under 30-min budget |
| #3117 | Fourier scale=2 (R2-R4) | −0.10% at R4 (tie) | FiLM absorbed Fourier's spatial-frequency signal; test direction +0.86% |
| #3365 | batch_size=6/8 on bf16 | bs=4 best (monotonic regression) | GPU compute-bound; bigger batch cuts SGD steps without reducing sec/epoch |

## Active experiments

| Student | PR | Hypothesis | Stack | Status |
|---------|----|-----------|----|----|
| thorfinn | #3390 | **🔥 T_max=20 vs T_max=15 verify on FiLM stack** | bf16+T_max+EMA+2xFiLM | Sent back for rebase (R1 showed 88.229 on bf16-only) |
| askeladd | #3777 | SDF input features — distance-to-surface | bf16+T_max=15+EMA+2xFiLM | Just assigned |
| tanjiro | #3511 | grad_clip=1.0 on two-shot FiLM stack (rebase) | bf16+T_max+EMA+2xFiLM | Sent back to rebase |
| nezuko | #3492 | **n_hidden=192 verify on FiLM stack** | bf16+T_max+EMA+2xFiLM | Sent back for rebase (R1 showed −2.99% on pre-FiLM) |
| alphonse | #3594 | **🔥 Schedule-Free AdamW verify on FiLM stack** | bf16+T_max=15+EMA+2xFiLM | Sent back for verify+rebase (R1 showed −20.75% on pre-two-shot) |
| edward | #3684 | slice_num=32/64/96 sweep | bf16+T_max=15+EMA+2xFiLM | Just assigned |
| frieren | #3681 | Three-shot FiLM (preprocess + attn + MLP) | bf16+T_max=15+EMA+2xFiLM | Just assigned |
| fern | #3758 | n_layers=4 depth ablation | bf16+T_max=15+EMA+2xFiLM | Just assigned |

## Key research questions (ordered by potential impact)

1. **🔥 T_max=20 + FiLM stack (thorfinn #3390 R2):** Predicted **80-85** if additive. Largest possible single-change advance currently in flight.
2. **🔥 SF-AdamW + FiLM stack (alphonse #3594 R2):** Predicted **70-75** if R1 (−20.75%) reproduces. Even larger if true, but mechanism overlaps with T_max=20.
3. **n_hidden=192 + FiLM (nezuko #3492 R2):** Predicted **86.5-88.5**. Wider model adds capacity AND grows FiLM head proportionally.
4. **Grad clip=1.0 + FiLM (tanjiro #3511 R2):** Predicted ~86-88. Gradient direction normalization orthogonal to FiLM.
5. **Three-shot FiLM (frieren #3681):** Predicted −1-3%.
6. **SDF input features (askeladd #3777):** Predicted −1-3% on geom_camber splits specifically.
7. **n_layers=4 (fern #3758):** Predicted −0.5% to −2% if epoch budget matters more than capacity.
8. **slice_num sweep (edward #3684):** Direction unknown.

## LR axis (closed)

**5e-4 is the optimum** for bf16+T_max=15. Both lower (2.5e-4, 3.5e-4) and higher (1e-3, 1.5e-3) directions falsified across 4+ seeds.

## Depth axis (explored)

- n_layers=6: regresses under 30-min budget (+2.47%, epoch cost kills fine-tune time)
- n_layers=4: IN TEST (#3758) — hypothesis: faster epochs → more fine-tune
- Current best stays at n_layers=5 until #3758 resolves

## Cosine schedule axis (active investigation)

**Current default T_max=15 is suboptimal:**
- Calibrated for fp32 14-epoch budget; on bf16's 19-epoch budget it hits LR floor at ep 16, wasting 3 epochs at LR=5e-8
- Two parallel attacks in flight: (a) extend T_max=20 (#3390), (b) eliminate schedule via SF-AdamW (#3594)
- Whichever has larger paired Δ on full FiLM stack will dominate. Both might compose with other hypotheses; rerun cascades if needed.

## Fourier axis (CLOSED)

Fourier features subsumed by FiLM. The spatial-frequency signal FiLM provides via γ,β modulation is functionally equivalent to what Fourier basis expansion added on the input side. The per-round compression (−9.10% → −3.16% → −0.10%) cleanly tracks each merged feature's absorption of the Fourier signal. Not worth re-testing unless the FiLM stack is removed.

## Potential next hypotheses (not yet assigned)

1. **Per-block independent FiLM** — give each block its own conditioner (more expressive, ~5K more params/block). Three-shot FiLM result may inform priority.
2. **Lion optimizer** — gradient-sign updates directly; tanjiro's grad-clip finding suggests normalized GD helps. SF-AdamW result may make this redundant.
3. **Sobolev loss** — gradient supervision near surface (dU/dx, dp/dx). Physically motivated, untested. Complex to implement.
4. **n_layers=4 + n_hidden=192** — if both individually win, compose them together for one compact model.
5. **LR-scaled bigger batch** — askeladd #3365 falsified flat-LR retry; LR-scaled (lr × √(B/4)) could still win.
6. **Wider FiLM hidden** — `film_mlp_hidden=256` instead of 128; orthogonal to n_hidden of main model.

## Operational notes

- **Current best command:** `python train.py --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 --film_cond --two_shot_film`
- **GH API rate limits:** recurring ~40-min windows; student pods auto-recover (nezuko and thorfinn were stuck earlier, recovered by 07:40).
- **test_geom_camber_cruise NaN:** cruise-sample overflow in read-only `data/scoring.py`; use 3-split partial mean for test comparisons.
- **Depth-vs-epochs insight:** Under 30-min budget, per-epoch cost matters as much as model capacity. n_layers=6 loses 3 fine-tune epochs to n_layers=5.
- **FiLM-owns-velocity insight:** Fourier per-split decomposition (#3117 R4) shows mae_surf_Ux regresses +4.64% with Fourier under FiLM stack. FiLM fully owns velocity channel.
- **Cosine T_max=15 wasted-epochs insight:** T_max=15 hits LR floor at epoch 16 on bf16's 19-epoch budget. This is the largest unfixed inefficiency in the current stack. Two attacks in flight.
- **Stack staleness pattern:** Multiple PRs ran on pre-FiLM stack and need rebase verifies (thorfinn, nezuko, tanjiro, alphonse). The baseline moved fast through Round 4.
