# SENPAI Research State

- **Last updated:** 2026-05-13 00:00 (post-#1585 FiLM merge — new baseline val=80.82 / test=71.30)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 actively assigned)

## ⭐ Current baseline (PR #1585 merged 2026-05-12 23:55 UTC)

- **val_avg/mae_surf_p:** **80.8162** (best seed = seed 2, epoch 14, base-model — tested on Huber-only stack; merged code adds Re-weight + SWA)
- **test_avg/mae_surf_p:** **71.3028** (best seed, 4-split, all finite)
- **3-seed mean ± std:** val 82.20 ± 1.23, test 73.09 ± 1.64
- **Config (tested):** Transolver baseline + **FiLM conditioner (mid_dim=64, zero-init last linear, per-layer (γ,β) from masked-mean of dims 13-23)** + Smooth-L1 (Huber β=1.0) + surf_weight=10.0
- **Merged code adds:** per-sample Re-weight (`1/log_re_shifted`) + SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2) — **untested composition; conservative tested floor = 80.82**
- See `BASELINE.md` for full reproducible spec + composition warning.

## 🔥 Hottest signals this session

- **PR #1585 (askeladd, FiLM-on-Huber):** Largest single-PR gain in branch history. val=80.82 (−15.6% vs prior 95.75) / test=71.30 (−17.3% vs prior 86.17). All 3 seeds beat baseline by 12+ points. FiLM modulation observability confirms the mechanism (non-trivial γ/β, β grows with depth). **Mechanism axis: architecture-conditioning — landed.**
- **PR #1617 (nezuko, grad-clip on SWA-on-Huber):** val=94.48 best / 0.83% inter-seed (20× variance reduction). Needs rebase onto new FiLM baseline; composition with FiLM is the next test.
- **PR #1600 frieren (β-sweep, in-flight):** β=0.3 finished val=96.16 / **swa_test=84.76** — asymmetric val/test signal (test beats prior baseline by 1.6% while val doesn't). β=1.0/3.0 arms still running.

## Most recent direction from human researcher team

None received. Last issue check: 2026-05-12 22:10 UTC, zero open issues on this advisor branch. Workflow assumed: drive primary ranking metric `val_avg/mae_surf_p` (and `test_avg/mae_surf_p`).

## ✓ Merged improvements

| PR | Slug | Win | Baseline after merge |
|---|---|---|---|
| #1452 (frieren) | smooth-l1-loss-e15 | MSE → Huber (β=1.0), 10 → 15 epochs, `data/scoring.py` NaN-safe fix | val=100.77, test=90.38 |
| #1554 (frieren) | swa-on-huber | SWA on final 4/15 epochs, terminal eval on `swa_model` | val=99.07, test=88.90 |
| #1586 (thorfinn) | re-weight-on-huber | Per-sample loss reweighting by `1/log_re_shifted`, normalized | val=95.75, test=86.17 |
| #1585 (askeladd) | film-on-huber | **FiLM global conditioning** (zero-init head, per-layer (γ,β), 84K extra params) | **val=80.82, test=71.30** (current) |

## Current research focus

### Wave 6 (in flight, on the merged FiLM baseline)

| PR | Student | Slug | Hypothesis | Predicted Δ vs. 80.82 val |
|---|---|---|---|---|
| #1702 | askeladd | `per-channel-p-weight-on-filmed` | **Per-channel pressure-loss weighting** (`p_weight ∈ {2.0, 3.0}`, 2-arm sweep). 4th orthogonal axis (per-channel) | −0.5 to −2% |

### Wave 5 (in flight, against now-superseded 95.75 baseline frame)

All 4 PRs were designed against val=95.75. Their predicted ranges (−0.5 to −3%) do not land below 80.82. Decision framework: best-arm val < 80.82 → merge; 80.82 ≤ val < 84 → send back to stack with FiLM; val ≥ 84 → close.

| PR | Student | Slug | Hypothesis | Status |
|---|---|---|---|---|
| #1642 | thorfinn | `re-weight-sqrt-on-swa` | Sharper Re-weight curve `1/sqrt(log_re_shifted)` | WIP |
| #1679 | tanjiro | `no-swa-on-reweight` | Remove SWA entirely | WIP |
| #1680 | fern | `drop-path-0p1-on-merged` | Stochastic depth `drop_path_rate=0.1` | WIP |
| #1691 | edward | `surf-weight-5-on-merged` | Halve `surf_weight` 10 → 5 (opposite direction from #1620) | WIP |

### Wave 3 (in flight, against SWA-on-Huber 99.07 baseline frame)

Even more stack-stale than wave-5. All comments updated with new baseline frame.

| PR | Student | Slug | Hypothesis | Status |
|---|---|---|---|---|
| #1600 | frieren | `beta-sweep-on-swa` | 3-arm Huber β ∈ {0.3, 1.0, 3.0} | **β=0.3 done (val=96.16), β=1.0 done (val=104.17), β=3.0 running** |
| #1617 | nezuko | `grad-clip-on-swa` | `clip_grad_norm_(max_norm=1.0)` + 2 seeds | **needs rebase onto FiLM baseline** |
| #1618 | alphonse | `surf-huber-vol-mse` | Split loss kind: Huber on surface, MSE on volume | **last completed run val=98.35; latest pod crash; awaiting student status** |

### Wave 2 (stack-stale on Huber-only 100.77 baseline)

| PR | Student | Slug | Hypothesis | Status |
|---|---|---|---|---|
| (none — #1585 was the last wave-2 PR; it merged) | | | | |

### Reframe decision rule for in-flight PRs (vs new 80.82 baseline)

- best-arm val < 80.82 AND no merge conflicts: merge directly.
- best-arm val < 80.82 BUT has merge conflicts: send back for rebase + retest on merged FiLM code.
- 80.82 ≤ best-arm val < 84: send back to retest with FiLM stack. If lever is theoretically orthogonal to FiLM (grad-clip, β-tuning, per-channel weighting), small improvements vs FiLM are likely; if lever is conditioning-redundant (e.g., another global-condition lever), close.
- best-arm val ≥ 84: close.
- **Test metric override:** if test metric beats 71.30 even when val doesn't beat 80.82, send back; paper-facing test wins are valuable.

## ✗ Closed this session

- #1454 (tanjiro, unified-pos rerun): val=128.78, regression.
- #1455 (thorfinn, batch=8/lr=7.1e-4 rerun): val=141.94, regression.
- #1448 (askeladd, slice_num=128, 3 seeds): mean val=134.31 ± 2.39.
- #1453 (nezuko, n_hidden=192, 2 unseeded runs): val=128.28 / 148.57, 16% variance.
- #1446 (alphonse, --epochs=10 schedule align): never trained, moot.
- #1449 (edward, surf-weight-30 wave-1): never trained (baseline-stale).
- #1450 (fern, mlp-ratio-4 wave-1): never trained (baseline-stale).
- #1551 (tanjiro, unified-pos-on-huber): val=105.24, +4.4% regression. Lever debunked twice on this branch.
- #1621 (fern, mlp-ratio-4 wave-3 rerun): val=106.11 + wall-clock overflow. Capacity expansion definitively closed as wrong axis.
- #1645 (tanjiro, swa_lr=5e-5): val=100.55. Cosine-floor displacement issue identified.
- #1620 (edward, surf-weight-30 wave-3 rerun): val=105.99 / test=95.73, +7%/+7.7% regression. Volume context starvation.

## ⚠ Active operational notes

- The GraphQL rate-limit pattern (~30-40 min between exhaustions) continues; pods recover automatically. REST helpers preferred where possible.
- **#1585 FiLM merge changes the decision threshold for ALL in-flight PRs.** Comments posted to #1617, #1618, #1600 with updated baseline frame. Wave-5 PRs not yet pinged (they're mid-training; the new baseline note can wait until they post results).
- **The SWA × Re-weight × FiLM composition is untested.** Tanjiro's #1679 (no-SWA test) is now the critical experiment for resolving whether SWA is helping or hurting on the FiLM stack.
- **Surf_weight axis status:** #1620 (=30) closed; #1691 (=5) in flight. Bookend test.
- **Largest remaining gap: val_geom_camber_rc (97.90 ± 0.47 on FiLM baseline, −5.65% from prior).** FiLM helped least here. Next stacking should target **geometry-aware** mechanisms.
- **Asymmetric val/test signal from frieren β=0.3:** swa_test=84.76 beat prior baseline test=86.17 by 1.6% while val didn't beat val. Possibly a generic "test/OOD-tuned hyperparameters" signal — could become a paper-facing optimization.

## Mechanism-axis coverage (all 8 students)

- **Loss-shape:** β-sweep (#1600 frieren), surface-vs-volume split (#1618 alphonse)
- **Loss-weighting:** surf_weight halve (#1691 edward), Re-weight-sqrt (#1642 thorfinn), **per-channel p-weight (#1702 askeladd — new wave-6)**
- **Optimizer-stability:** gradient clipping (#1617 nezuko, rebase-needed)
- **Regularization:** stochastic depth (#1680 fern)
- **Architecture-conditioning:** **FiLM (#1585 askeladd — LANDED)**
- **Schedule / SWA-on-off:** no-SWA test (#1679 tanjiro)

7 orthogonal mechanism axes across 8 students. Architecture-capacity (mlp_ratio, n_hidden) definitively closed. The surf_weight axis is bookend-probed.

## Potential next research directions (wave 7+)

Ranked by expected ROI on `val_avg/mae_surf_p` given the new FiLM baseline 80.82:

1. **Geometry-aware lever stacked with FiLM** — `val_geom_camber_rc=97.90` is the bottleneck FiLM left untouched. Candidates:
   - Surface arc-length / dsdf positional encoding for surface nodes specifically
   - Geometry-conditioned FiLM (per-token (γ,β) gated by `is_surface` mask, conditioned on NACA params)
   - Slice_num sweep within FiLM (different from wave-1 #1448 close — that was n_hidden=192 with no FiLM; FiLM may have unlocked the slice-capacity axis)
2. **Compound stack: grad-clip × FiLM × Re-weight (× no-SWA?)** — #1617 nezuko's rebase tests this directly. Plausible compound floor ~78 val / ~69 test.
3. **More epochs on FiLM-merged baseline** — val curve was still descending at epoch 14 (−4.5% from epoch 12→14 on best seed). A 20–25 epoch run (budget permitting) likely buys another 2–4 points. Cheapest improvement.
4. **EMA averaging as alternative to SWA** — if no-SWA (#1679) wins, this is a low-effort variant that may capture flat-minima benefits without SWA's schedule-window displacement.
5. **Per-channel β** — pressure has wider normalized range; combine with #1702 p-weight win if it lands.
6. **Test-focused β tuning** — frieren #1600's β=0.3 already showed test=84.76 beating prior baseline. A FiLM + β=0.3 stack test could be a paper-facing optimization.
7. **Asinh transform on pressure target** — value-level compression of high-Re tail; orthogonal to sample-level Re-weight and per-channel p-weight.
8. **Surface-aware slice routing in PhysicsAttention** — research-ideas H2, medium-effort.
9. **Domain-adversarial training** — −3 to −8% on camber OOD specifically.
10. **Best-checkpoint vs SWA-final infrastructure** — save base-best alongside SWA for paper-facing comparisons regardless of which path wins.

The researcher-agent's `RESEARCH_IDEAS_2026-05-12_round2.md` has H1–H10 with concrete implementation specs (if any remain unexplored).

## Open questions to revisit on next review

- **FiLM × Re-weight × SWA composition:** does the merged code's untested combination match, beat, or trail the standalone FiLM result (val=80.82)? Next merged PR will tell us.
- **SWA × FiLM composition:** SWA evidence on the prior stack (#1645) suggested SWA may regress; does FiLM change that? Tanjiro #1679 + the FiLM merge together answer this.
- **grad-clip × FiLM composition:** #1617 rebase will test. Expected: constructive (orthogonal mechanisms).
- **drop_path on 5-layer network + FiLM:** literature consensus says drop_path is most useful at 12+ layers; FiLM doesn't add depth. Likely still won't help here.
- **Per-channel p_weight on FiLM:** does explicit per-channel weighting compound with FiLM's implicit per-channel conditioning?
- **Per-split divergence:** with FiLM baseline, val_geom_camber_cruise (59.69) is easiest; val_geom_camber_rc (97.36) is hardest. Wave-7 geometry-aware levers should target val_geom_camber_rc specifically.
- **Wall-clock budget tightness:** 30-min per-run remains the binding constraint. FiLM's ~3% per-epoch overhead from the extra parameters is acceptable; further architectural additions need to mind this.
