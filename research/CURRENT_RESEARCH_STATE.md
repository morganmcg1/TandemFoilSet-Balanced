# SENPAI Research State

- **Date:** 2026-05-13 ~06:10
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).
- **Plateau Protocol:** RESOLVED — Lion broke plateau (-14.3%), GeGLU+Lion shattered it (-25.3%).

## Current baseline

**`val_avg/mae_surf_p` = 52.798** (T_max=12 + RMSNorm+GeGLU+Lion+surf_weight=10, PR #1793, epoch 12)
**`test_avg/mae_surf_p` = 44.972**

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 58.907 | 50.239 |
| geom_camber_rc | **67.658** | 59.561 |
| geom_camber_cruise | 33.380 | 27.740 |
| re_rand | 51.248 | 42.345 |
| **avg** | **52.798** | **44.972** |

**Note:** Compound `T_max=12 + surf_weight=5` is in flight (nezuko #1956). If positive, the baseline shifts down again.

## What we've learned

### Big wins (merged)
1. **L1 loss**: −20.5% (PR #1358)
2. **GeGLU+Lion compound**: −25.3% (PR #1769)
3. **Lion optimizer lr=1e-4**: −14.3% (PR #1725)
4. **n_layers=6**: −9.4% (PR #1392)
5. **surf_weight=5**: −9.0% val / −9.8% test (PR #1836) ← single_in_dist −20.5%
6. **T_max=12 (cosine aligned to epoch budget)**: −7.9% val / −8.9% test (PR #1793) ← all 4 splits improved, cruise −13.8%, re_rand −11.0%
7. **mlp_ratio=4**: −5% (PR #1408)
8. **RMSNorm**: −2.9% val / −5.9% test (PR #1837) ← geom_camber_rc −17.2%
9. **bf16 mixed precision**: −0.34% (PR #1724) ← infrastructure win, +1-2 epochs/run

### Current stack (defaults + CLI overrides)
- L1 (MAE) loss in normalized space, **surf_weight=10** (CLI; reverted from 5 with T_max=12 merge — compound still in flight)
- n_layers=6, **mlp_ratio=4, GeGLU activation** (PR #1769)
- **RMSNorm** (PR #1837, replaces LayerNorm)
- n_hidden=128, n_head=4, slice_num=64
- Lion optimizer lr=1e-4, weight_decay=1e-4
- **CosineAnnealingLR T_max=12 (epochs=12)** ← PR #1793, cosine fully decays to 0
- bf16 mixed precision (autocast)
- 12 epochs in 30 min (~138s/epoch)

**Reproduce command:** `python train.py --epochs 12 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10`

### Dead ends
- **AdamW hyperparameter space fully exhausted:** WD (0, 1e-4 optimal, 5e-4), LR (5e-4 only), betas (0.85/0.9/0.95 for β1, 0.99/0.999 for β2), eps (1e-8, 1e-4), schedule (T_max=14/50, warmup, cosine restarts)
- Dropout=0.1: +11.8% (model is underfitting — regularizing makes it worse)
- Gradient clipping max_norm=1 and max_norm=10: both worse (oscillations = useful search)
- Huber loss β=1.0: +15.7% (mostly-MSE; L1's constant gradient is the key)
- **Channel-weighted L1 [0.03,0.03,1.0] on GeGLU+Lion**: +11.0% (PR #1767 — GeGLU gates do implicit channel balancing; manual Ux/Uy downweighting disrupts routing)
- **SwiGLU vs GeGLU**: +1.6% (PR #1824 — LLM finding doesn't transfer; GELU's slightly negative gate range benefits CFD pressure-gradient features)
- **SmoothL1 β=0.1 on GeGLU+Lion**: +7.1% (PR #1859 — all loss modifications exhausted; pure L1 optimal for Lion)
- **n_layers=7 (re-test with RMSNorm+Lion)**: +4.6% (PR #1890 — 12 epochs at 160s/epoch; single_in_dist catastrophic +18%; depth expansion incompatible with budget under any tested condition)
- **WD=1e-1**: +2.72% (PR #1889 — over-regularizes; best_epoch=10, train descending while val climbs; WD space above 1e-2 exhausted)
- **lr=2e-4 on RMSNorm stack**: +0.98% val (PR #1765 — RMSNorm tightened loss surface; lr=2e-4 overshoots on geom_camber_rc; test −1.23% ✓; pivot to lr=1.5e-4)
- **mlp_ratio=8 + GeGLU**: +5.95% (PR #1872 — gating wins outright; fc2 capacity expansion beyond 256 channels adds noise pathways; mlp_ratio=4 optimal)
- **CosineAnnealingLR eta_min=1e-5**: +12.05% vs current baseline (PR #1920 — LR floor above 0 conflicts with T_max=12 which cleanly decays to 0; T_max=12 strictly dominates)
- **Lion WD=3e-2**: +0.06% (PR #1925 — WD valley confirmed flat [1e-4→3e-2]; WD=1e-1 bends up; entire WD axis exhausted on this stack)
- **Lion 2-epoch warmup**: mechanism conflict with T_max=12 (PR #1790 — warmup costs 17% of 12-epoch budget; cold-start problem already addressed by T_max=12 cosine; student stale on rerun)
- n_head=8: +43% per-epoch cost, +15.7% worse
- slice_num=128: +12% per-epoch cost, +17.8% worse
- EMA decay=0.999: cold-start drag (+41% worse)
- Batch=8 (accum_steps=2): +23.6% worse (step-count limited)
- Fourier L=4: +5.6% worse (doesn't compound with L1)
- Width n_hidden=192: too slow/epoch

### Key insights
1. **Lion is structurally complementary to L1**: Both operate via sign direction; combined signal is clean
2. **Budget is the constraint**: 30 min → ~14 epochs with current stack (~138s/epoch).
3. **L1 loss in normalized space is validated**: channel-weighted loss hurts on GeGLU+Lion (+11%); GeGLU gates do implicit channel balancing — the gradient channel weights must stay equal.
4. **surf_weight=5 mechanism confirmed**: halving surface:volume ratio reallocates L1 gradient to volume nodes → richer volumetric features → better surface via geometric context. All 4 splits improved, vol MAE improved −7% to −26%.
5. **AdamW hyperparameter space is exhausted**: All optimizer knobs tested. Lion is the new baseline optimizer.
6. **Lion WD=1e-2 confirmed on Lion+GELU (−10.4%)**: awaiting retest on full RMSNorm+GeGLU+Lion+surf_weight=5 stack.
7. **RMSNorm shifts the hardest split**: After RMSNorm, geom_camber_rc became easier; single_in_dist became the primary bottleneck. surf_weight=5 cracked single_in_dist (−20.5% val).
8. **geom_camber_rc (72.0 val) is now the hardest split** — primary target for further improvement.

## Active experiments (Round 12 — all on T_max=12 + RMSNorm+GeGLU+Lion baseline, val=52.798)

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1765 | Lion lr=1.5e-4 (pivot from 2e-4): midpoint LR | WIP (rerun) |
| askeladd | #1766 | Lion WD=1e-2: paper-recommended on full stack | WIP (stale) |
| edward | #1995 | n_layers=5: shallower model → ~15 epochs in 30-min budget | NEW |
| fern | #1996 | slice_num=48: tighter PhysicsAttention → ~14 epochs in budget | NEW |
| nezuko | #1956 | **T_max=12 + surf_weight=5 compound** | WIP |
| thorfinn | #1948 | surf_weight=3: sweep gradient budget further toward volume | WIP |
| frieren | #1983 | CosineAnnealingLR T_max=10: push cosine floor to epoch 10 | WIP |
| tanjiro | #1984 | n_hidden=160: widen attention dim for richer aerodynamic features | WIP |

**Recently merged:**
- nezuko #1793: T_max=12 on RMSNorm+GeGLU+Lion (−7.9% val / −8.9% test) ← NEW BASELINE 52.798/44.972
- thorfinn #1836: surf_weight=5 on RMSNorm+GeGLU+Lion (−9.03% val / −9.76% test)
- frieren #1837: RMSNorm on GeGLU+Lion (−2.9% val / −5.9% test)

**Recently closed:**
- edward #1925: WD=3e-2 (+0.06% on prior baseline; +19.4% vs current) — WD axis confirmed flat [1e-4→3e-2]; exhausted
- fern #1790: Lion 2-epoch warmup — stale + mechanism conflicts with T_max=12 (warmup wastes 17% of budget when cosine already handles full decay)
- frieren #1920: eta_min=1e-5 (+12.05% vs current baseline) — mechanism redundant with T_max=12; T_max=12 cleanly decays LR to 0, which is strictly better than a 1e-5 floor
- tanjiro #1872: mlp_ratio=8 (+5.95%) — "gating wins outright"; fc2 capacity expansion adds noise pathways that gate doesn't fully suppress at 12 epochs
- frieren #1890: n_layers=7 (+4.6%) — depth incompatible with budget
- edward #1889: WD=1e-1 (+2.72%) — over-regularizes
- edward #1859: SmoothL1 β=0.1 (+7.1%) — all loss modifications exhausted
- edward #1767: channel-weighted L1 (+11%) — GeGLU gates do implicit channel balancing
- tanjiro #1824: SwiGLU (+1.6%) — GELU's negative gate range benefits CFD features

## Critical infra issue: train.py:440 LR hardcoding bug

Discovered by askeladd in #1766; alphonse's #1765 also contains the same fix (`lr=cfg.lr`, plus `Config.lr` default updated to 1e-4). Once either PR rebases cleanly onto the new baseline and is merged, the bug is resolved. **Until then, any LR experiment with `--lr != 1e-4` is silently broken.** alphonse confirmed fix works (config shows lr=0.0002 was applied correctly in their rerun).

## Round 11 priorities (T_max=12 + RMSNorm+GeGLU+Lion baseline, val=52.798)

**Tier 1 (compound the two recent wins):**
1. **T_max=12 + surf_weight=5 compound** (nezuko #1956): pure orthogonal stacking; predicted val ~48-50 if mechanisms compound additively.
2. **surf_weight=3 with T_max=12** (thorfinn #1948): if running on old T_max=50 stack, may need re-evaluation against new baseline.

**Tier 2 (optimizer tuning — older PRs may need rebase to T_max=12 baseline):**
3. **Lion WD=1e-2** (askeladd #1766, stale): confirmed −10.4% on Lion+GELU; needs rebase to T_max=12.
4. **Lion WD=3e-2** (edward #1925): brackets WD optimum; may need to re-evaluate at T_max=12.
5. **Lion lr=1.5e-4** (alphonse #1765): pivot from 2e-4; at T_max=12 the cosine fully decays — higher initial LR may now be tolerable.

**Tier 3 (schedule/architecture — older PRs may need rebase):**
6. **Lion + 2-epoch warmup** (fern #1790): at T_max=12, warmup eats 17% of budget — may not compound.
7. **mlp_ratio=8 + GeGLU** (tanjiro #1872, stale — nudged): recover fc2 capacity halved by GeGLU split.
8. **CosineAnnealingLR eta_min=1e-5** (frieren #1920): non-zero LR floor for Lion tail; potentially redundant with T_max=12 now that LR reaches 0 cleanly.

**Queued ideas for next idle students (after current round lands):**
- **T_max=11**: T_max=10 assigned to frieren (#1983); if T_max=10 wins, try 11 to bracket optimum
- **surf_weight=2 with T_max=12**: if T_max=12+sw=5 compound (nezuko #1956) holds, keep sweeping
- **geom_camber_rc-targeted experiments**: 67.7 val still dominates the average; worth targeted interventions
- **PhysicsAttention slice_num=48**: slight reduction for faster epochs, more steps in budget
- **Higher LR (1.5e-4 or 2e-4) + T_max=12 compound**: now that LR decays to 0, larger initial steps may be tolerable; alphonse #1765 testing 1.5e-4

## Key constraints

- 30 min / run cap: 12 epochs at ~138s/epoch with T_max=12 (cosine fully decays)
- Per-epoch time eliminates: n_head=8 (+43%), slice_num=128 (+12%), n_layers=7 (~160s/epoch even with RMSNorm+Lion; confirmed dead in PR #1890)
- EMA: cold-start drag, incompatible with short budget
- Batch increase: always worse (step-count limited)
- Gradient clipping: always worse (oscillations are useful)
- Dropout: always worse (model is underfitting at 14 epochs)
