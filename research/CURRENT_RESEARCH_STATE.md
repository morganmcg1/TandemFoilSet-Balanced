# SENPAI Research State

- **Last updated:** 2026-05-16 ~01:38 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline

**val_avg/mae_surf_p = 96.0997, test_avg/mae_surf_p = 85.5256** — from PR #3507 (alphonse, n_hidden=160), merged 2026-05-16 ~00:30 UTC. See `BASELINE.md` for full details.

Per-split test: single_in_dist=103.75, geom_camber_rc=92.42, geom_camber_cruise=61.38, re_rand=84.55.

Previous baseline: val=100.5275, test=90.1489 (PR #3089 alphonse L1 loss).

## Pending winners (W&B-verified, awaiting/queued merge)

| PR | Student | Hypothesis | W&B run | val_avg/mae_surf_p | test_avg/mae_surf_p | State |
|---|---|---|---|---:|---:|---|
| **#3372** | askeladd | **Fourier PE 4-freq (rebased)** | `qyc68z5k` | **88.244** 🏆 | **77.088** 🏆 | Awaiting SENPAI-RESULT comment (nudged 01:35) |
| **#3479** | frieren | **Per-channel mlp2 heads (rebased)** | `99phangs` | **95.602** | **85.309** | status:review, mergeable |

Merge order plan (best-first):
1. Merge #3372 askeladd first → new baseline val≈88.24, test≈77.09.
2. Re-evaluate #3479 frieren: val=95.60 was a win vs 96.10 baseline but would be a regression vs the new 88.24 baseline. Will need rebase onto post-#3372 advisor + one confirm arm to validate per-channel heads compose with Fourier PE.

## Width scaling curve so far

| n_hidden | params | val_avg | test_avg | Δ val vs 128 | PR |
|---|---|---|---|---|---|
| 128 | 662k | 100.527 | 90.149 | baseline | #3089 (MERGED) |
| 160 | 1.03M | 96.100 | 85.526 | −4.4% | #3507 (MERGED) |
| 192 | ~1.49M | — | — | in-flight | #3552 (alphonse) |

Val curve still descending at epoch 10 for both merged widths → continued width scaling expected to help.

## Most recent research direction from human researcher team

No GitHub Issues open for this track. Proceeding from the program contract only.

## Cross-cutting findings (apply to ALL in-flight PRs)

1. **L1 loss is the default** (Config.loss_type = "l1"). Advisor branch uses L1 everywhere.
2. **n_hidden=160 is the new default** (Config.n_hidden = 160). All branches rebase to get this.
3. **Scoring NaN fixed** (torch.isfinite per-sample mask in train.py's evaluate_split).
4. **Use `--epochs 10`** (or `--epochs 8` for very wide models > 185s/epoch) so cosine fully anneals.
5. **Grad clip max_norm=1.0**, warmup 2 epochs, lr=1e-3, batch=4.

## Active in-flight PRs (status as of 01:38 UTC)

| # | Student | Hypothesis | State | val_avg/mae_surf_p |
|---|---|---|---|---|
| **#3372** | askeladd | **Fourier PE 4-freq (rebased)** | wip, terminal run exists | **88.244** 🏆 (run `qyc68z5k`) |
| #3469 | tanjiro | Depth n_layers=5→6 | **CLOSED** (01:38, stale baseline regress) | 108.45 vs 96.10 baseline |
| **#3479** | frieren | Per-channel heads (rebased) | **status:review** | **95.602** (run `99phangs`) |
| #3288 | edward | Scoring fix + lr default verify | wip | 96.53 (run `hlzvyl3y`); ties baseline → close |
| #3490 | nezuko | L1 LR sweep {3e-4, 2e-3, 4e-3} | wip, lr=4e-3 arm running | best so far 98.88 (run `tv96lmop`, lr=2e-3) |
| #3552 | alphonse | Width n_hidden=192 | wip, arm 2 running | best so far 102.73 (run `g5m772d7`) |
| #3508 | fern | Cosine warm restarts (SGDR T_0=4) | wip, arm 3 running | best so far 100.79 (run `tkheirrd`) |
| #3524 | thorfinn | Huber loss β=1.0 | wip, arm 2 running | best so far 104.43 (run `f1lexwcq`) |

**Action queue (next wakeup):**
- Merge #3372 (askeladd), then potentially #3479 (frieren) after rebase
- Close: #3288 edward (ties baseline), #3490 nezuko (best 98.88 regressed), #3508 fern (best 100.79 regressed), #3524 thorfinn (104.43 regressed), #3552 alphonse width-192 (102.73 regressed)
- Reassign all 6 students to new experiments after closures

## Merged wins (cumulative)

| PR | Description | val_avg | test_avg |
|---|---|---|---|
| **#3507** | **Width n_hidden=160 (alphonse)** | **96.0997** ← current | **85.5256** |
| #3089 | L1 loss + scoring fix (alphonse) | 100.5275 | 90.1489 |
| #3091 | LR warmup + clip + lr=1e-3 (edward) | 109.42 | NaN |

## Closed experiments

| PR | Hypothesis | val | Reason |
|---|---|---|---|
| #3095 | nezuko surf_weight=20 | 111.92 | +11.3% vs new baseline |
| #3371 | thorfinn EMA-9999 | 106.22 | +5.7% vs new baseline; training not converged enough |
| #3414 | tanjiro SWA | swa=109.48 | EMA/SWA premature at 10 epochs |
| #3093 | frieren bf16+bs=8 | 120.43 | MSE+bf16+clip incompatible schedule |
| #3092 | fern slice_num=128 | 106.82 | No win vs baseline |

## Merge resolution status (W&B-verified 2026-05-16 01:30 UTC)

1. ✅ **#3372 askeladd Fourier PE 4freq (rebased + L1)** — `qyc68z5k` val=88.244, test=77.088. **Awaiting SENPAI-RESULT comment** (nudged at 01:35).
2. ✅ **#3479 frieren per-channel heads (rebased)** — `99phangs` val=95.602, test=85.309. **status:review**; merge after #3372 (will need rebase + confirm to validate compose with Fourier PE).
3. ❌ **#3469 tanjiro depth-6** — `5y4w4b45` val=108.45 (stale code). **CLOSED**.
4. ❌ **#3552 alphonse width-192** — `g5m772d7` val=102.73 (over-parameterized at --epochs 8). Close after arm 2 finishes.
5. ❌ **#3490 nezuko LR sweep** — best `tv96lmop` lr=2e-3 val=98.88 (+2.78 vs 96.10). Close.
6. ❌ **#3508 fern warm restarts** — best `tkheirrd` val=100.79 (+4.69). Close.
7. ❌ **#3524 thorfinn Huber β=1.0** — `f1lexwcq` val=104.43 (+8.33). Close.
8. ⚠️ **#3288 edward scoring fix + lr default** — `hlzvyl3y` val=96.53 (~ties baseline 96.10). Close as superseded by #3089 + #3507.

## Potential next research directions (round 6+)

1. **Stack Fourier PE + per-channel heads** — both are pending-merge wins; if frieren rebase confirms compose with Fourier-PE'd advisor we have ~7+ point + 0.5 point composition test.
2. **Longer training (15–20 epochs)** — fern's warm restarts arms ran 10 ep cosine that was still descending; same true for frieren, alphonse, thorfinn. Strongest single lever NOT yet explored. Reassign fern to `--epochs 15`.
3. **Data augmentation** — foil reflection / AoA perturbation; doubles effective training set. Reassign tanjiro.
4. **Physics-aware auxiliary loss** — divergence-free penalty for incompressible flow (∇·u = 0 in interior). Reassign edward.
5. **Learnable Fourier frequencies** — askeladd's suggested follow-up from #3372 (Tancik random-Fourier or learnable σ).
6. **num_freq sweep 2/6/8** — confirm 4 is optimum on the new compose-ready stack.
7. **bf16 retry at n_hidden=160** — correct bf16 schedule co-design (lower LR or smaller warmup); 2× epochs in budget.
8. **Asymmetric head capacity** — pressure-dominated head wider/deeper than Ux/Uy (frieren's #3 suggested follow-up).
9. **Width n_hidden=176 or 184** (sweet spot between 160 and 192) — width-192 over-parameterized at --epochs 8; try smaller bump with full --epochs 10.
10. **Slice token increase** at n_hidden=160 (revisit fern's #3092 idea with new baseline) — slice_num=64 → 96/128.

## Cross-cutting observations

- **Width-160 + L1 + Fourier PE composes spectacularly** — the +8.1% L1 gain (#3089), +4.4% width gain (#3507), and +8.2% Fourier PE gain (#3372 unmerged) are all roughly orthogonal levers that have STACKED cleanly to bring val from 109.42 → 88.24 (a 19.4% improvement in 3 merges).
- **L1 loss gave 8.1% gain** on top of the base optimizer — now canonical.
- **Width scaling has a sweet spot near 160.** Width-192 at --epochs 8 regressed to val=102.73 (over-parameterized for budget); the 168s/epoch at width-160 is the right wall-clock vs capacity tradeoff at the 30-min budget.
- **Per-channel heads compose with width** but the marginal gain shrinks (val −2.5% at width-128 → val −0.5% at width-160) — width absorbed most of the per-channel upside.
- **Huber β=1.0 worse than L1** at this scale (104.43 vs 96.10). L1 stays canonical.
- **Cosine warm restarts (SGDR) worse than vanilla cosine** at this budget (100.79 vs 96.10). Cyclic LR resets restart from too-high LR for the under-converged solver state.
- **Higher LR (2e-3) worse than 1e-3** for L1 (98.88 vs 96.10). 1e-3 + warmup is well-tuned.
- **EMA/SWA don't help** with 10-epoch budget; training still descending.
- **Val curves still descending at epoch 10** for every experiment → longer training is an unexplored axis.
