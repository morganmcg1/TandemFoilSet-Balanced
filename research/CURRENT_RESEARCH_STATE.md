# SENPAI Research State

- **Updated**: 2026-04-29 (PR #821 MERGED — new best val=55.90/test=49.64; askeladd → #971 LR warmup)
- **Branch**: `icml-appendix-willow-pai2e-r2`
- **Tag**: `willow-pai2e-r2`
- **Most recent human researcher direction**: none; no GitHub Issues open.
- **Lab**: 11 students, 1 GPU each (96 GB), 30 min wall-clock, 50 epochs cap.

## Current baseline (MERGED)

**PR #821 (askeladd, tooling stack) — MERGED 2026-04-29** ← NEW BEST
- `val_avg/mae_surf_p` = **55.90** at epoch 50 (seed42 `66c4gac6`, still descending!)
- Per-split test: single=63.94, rc=62.62, cruise=26.87, re_rand=45.11
- `test_avg/mae_surf_p` = **49.64** (all 4 splits finite)
- Config: compound base + `--loss_type relative_mae --lr 2e-3 --batch_size 16 --compile`
- Wall: 22.5 min / 50 epochs (vs prior 30.4 min / 32 epochs)
- ⚠️ Seed-variance caveat: default seed landed val=82.97 / test=72.01 (27-pt spread). Future PRs: run ≥ 2 seeds.

**Prior baseline**: PR #840 (edward, rel MAE) — val=64.16, test=55.73 (superseded)
**Earlier**: PR #783 (fern, Huber δ=1.0) — val=75.93 (superseded)

## Current assignments (active WIP PRs)

| Student | PR | Hypothesis | Axis | Status |
|---------|----|-----------|------|--------|
| alphonse | #853 | Huber δ sweep: δ=0.5 and δ=2.0 on compound+Huber base | loss (δ tuning) | WIP |
| frieren  | #854 | Huber + grad accum (accum_steps=2): double throughput, ~60 epochs in budget | training throughput | WIP |
| fern     | #855 | Huber + surf_weight sweep: sw=5 and sw=20 vs baseline sw=10 | loss weighting | WIP |
| askeladd | #971 | LR warmup (5ep linear, 0→2e-3) + flip loss_type default to relative_mae | optimization (stability) | WIP |
| edward   | #940 | Relative MAE ε sweep: ε ∈ {1e-3, 1e-2, 1e-1} vs default 1e-6 | loss (ε tuning) | WIP |
| stark    | #842 | compound + SwiGLU param-matched h=168 | architecture (activation) | WIP |
| himmel   | #843 | compound + gradient norm clipping (max_norm sweep 0.5 / 1.0) | optimization (stability) | WIP |
| charlie  | #844 | compound + mlp_ratio=4 (FFN capacity at nh1) | architecture (MLP capacity) | WIP |
| thorfinn | #865 | AdamW weight decay sweep: wd=1e-5 and wd=0 on Huber base | optimization (regularization) | WIP |
| tanjiro  | #864 | Bugfix: sanitize GT y in evaluate_split (cruise NaN poison fix) | infrastructure | WIP |
| nezuko   | #866 | EMA model weights for val/test eval (decay=0.999) | optimization (eval smoothing) | WIP |

**Note on PRs #840–#844**: Assigned against old compound anchor (96.80). Compare against new baseline (64.16) when they finish.

**Idle-detection caveat (2026-04-29)**: The entrypoint harness reports 6 "idle" students (alphonse, fern, frieren, nezuko, tanjiro, thorfinn) because it queries `student:willowpai2e2-<name>` while their PRs (#853, #854, #855, #864, #865, #866) use the short-form `student:<name>` label. **Do NOT re-assign these students** — verify with `gh pr list --base $ADVISOR_BRANCH` before treating any "idle" report as actionable. All 11 GPUs are productively occupied.

**PR #940 (edward, ε sweep)**: ε=1e-6 (default) may over-weight cruise/low-magnitude samples, starving rc/single splits. Testing ε ∈ {1e-3, 1e-2, 1e-1} to soften the small-denominator dominance and recover the 84.10 rc / 77.07 single headroom.

## Key events this review pass

1. **PR #821 (askeladd, tooling stack) MERGED** — NEW BEST: val=55.90/test=49.64 (seed42). Full tooling stack now on advisor branch: AMP/bf16, bs=16, lr=2e-3, torch.compile, NaN-safe eval. All 4 test splits finite. 50 epochs in 22.5 min. Askeladd assigned to LR warmup PR #971 to address 27-pt seed variance.

2. **PR #840 (relative MAE) MERGED** as prior baseline (val=64.16, test=55.73 — superseded by #821).

3. **PR #900 (edward, loss curriculum) CLOSED**: Hard Huber→rel-MAE curriculum rejected. 10ep (+0.38 val, +1.73 test) and 20ep (+1.54, +1.95) both regress. Root causes: optimizer-reset stall at switch-over, plus Huber pre-training builds high-Re biased representations. Edward reassigned to ε sweep (#940).

4. **PR #821 round 3**: rebase + rel-MAE re-validation. C1+C2+C3 all pass. Seed42: val=55.90, test=49.64. Default seed: val=82.97, test=72.01. 27-pt spread → LR warmup follow-up.
   - C1 PASS: 50/50 epochs in 22.2 / 22.3 min (~26% headroom).
   - C2 PASS: cruise test=65.56 / 63.23 finite (3rd & 4th time on branch).
   - C3 strict-fail: val_avg=136.22 / 97.84 with vanilla MSE loss. Wide seed spread (38 pts).
   - **Reason for send-back**: PR was branched pre-#840, so train.py has merge conflicts AND validation used vanilla MSE (not the now-canonical relative-MAE loss). Round-3 ask: rebase + re-validate with `--loss_type relative_mae` to confirm the tooling stack preserves the 64.73 / 56.92 baseline.
   - The torch.compile addition (~1.5–1.8× speedup on top of AMP) was an unexpected upside.

## Current research focus

**The relative-MAE mechanism is working.** Both Huber (PR #783) and relative MAE (PR #840) attack the same root cause — high-Re tail dominance — at different abstraction levels, and they compound. The test_avg has improved from NaN (cruise bug) to 56.92 (all splits finite) with a clear path to the reference target of 40.93.

**The tooling stack has landed.** val=55.90/test=49.64 (seed42, 50 epochs). The model was still descending at epoch 50 — more wall-clock budget would help. Reference target of 40.93 is now realistically in reach. Next hypothesis priority: stabilize the LR warmup so all seeds land close to 55.90, then push on loss (ε sweep, surf_weight) and architecture (n_hidden=192 now feasible with AMP).

Current open questions:
1. Does 5-epoch LR warmup narrow the seed spread from 27 pts to ≤ 15 pts? (PR #971 in-flight)
2. Does ε tuning (1e-6 → 1e-2/1e-1) help rc/single at cruise's expense? (PR #940 in-flight)
3. What happens with the full hyperparam stack (Huber δ #853, surf_weight #855, grad clip #843, EMA #866) on top of the new tooling defaults?
4. Can n_hidden=192 + AMP/bs=16 now fit in budget? (VRAM was 49.8 GB with bs=16; n_hidden 128→192 adds ~40% params; should still fit at 96 GB)
5. Can we break below the prior-round reference of 40.93 with the full 50-epoch budget?

## Settled facts from this round

- **Relative MAE > Huber MSE**: per-sample scale normalization gives −14.7% on top of Huber's −21.6%. Total loss gain from loss reformulation: −36.3% from anchor.
- **Cruise split is the loss-reform beneficiary**: 40.13 val / 32.35 test — best OOD split, confirming the scale-equalization mechanism.
- **All test splits finite after relative-MAE**: The relative loss suppresses extreme pressure predictions that caused the cruise Inf bug.
- **Slice floor at sn=16**: sn=4 (val=98.25) and sn=8 (val=92.5) both regress.
- **Mean-centering is load-bearing**: RMSNorm decisive negative (val=109.17).
- **Loss reformulation beats architecture tweaks**: GeGLU, Fourier PE, RMSNorm all regress. Loss-first principle confirmed.

## Ruled-out directions (do not repeat)

- Gaussian Fourier PE on (x,z) — PR #787, val=100.12, decisive negative
- GeGLU activation in FFN — PR #782, val=94.41 (param-matched), decisive negative
- RMSNorm replacing LayerNorm — PR #786, val=109.17, decisive negative
- FiLM conditioning — failed in prior round
- OneCycleLR — PR #784 round 2, val=92.25; gradient-step-limited
- slice_num=4 — PR #841, val=98.25, floor at sn=16
- n_hidden=192 without AMP — throughput-blocked; re-test after #821 lands

## Pending new assignments (all students active)

All 11 students now have active WIP PRs. Next priority assignments (for when students finish):

1. **Relative MAE + full 50-epoch run (post-AMP)** — once PR #821 merges with lr=2e-3, run edward's config for a full 50 epochs. Most likely to push val below 60.
2. **ε sweep for relative MAE** (in-flight as PR #940) — ε ∈ {1e-3, 1e-2, 1e-1}.
3. **Domain-adaptive Huber δ** — different δ per domain (cruise vs rc vs single vs re_rand). Per-domain residual distributions are structurally different (verified by split metrics).
4. **n_hidden=192 + relative-MAE** — width scaling after AMP/bf16 lands.
5. **Cosine annealing LR schedule** — add CosineAnnealingLR on top of relative-MAE base; may squeeze extra performance in final 10 epochs.
6. **Relative MAE + surf_weight tuning** — compound the surf_weight sweep (#855) with the relative MAE loss rather than Huber base.

## Potential longer-horizon directions

- **Curriculum learning**: train on single-foil first, then add tandem. Motivated by the split disparity (cruise best with loss reform, single worst).
- **PerceiverIO cross-attention decoder**: if plateau persists at ~60.
- **Physics-constrained output layer**: divergence-free velocity prior for Ux/Uy channels.
- **Graph attention network**: compare against Transolver if plateau persists.
- **Multi-fidelity training**: use lower-resolution CFD samples to pre-train.
