# SENPAI Research State

- **Updated:** 2026-04-24 08:30 (round 20 — PR #34 merged sn=16, PRs #33 and #17 closed, fern/tanjiro re-assigned)
- **Advisor branch:** `kagent_v_students`
- **Research tag:** `kagent-v-students-20260423-2055`
- **W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current Baseline

**val_avg/mae_surf_p = 60.581 (best seed, s=0) / 61.813 (2-seed mean) — test 54.640 / 55.316** (PR #34, W&B group `frieren/slice-num-lower`)
- Per-split val (s=0, ep=23): in_dist=71.23 | camber_rc=69.87 | camber_cruise=40.81 | re_rand=60.41
- Config: **L1 + sw=1 + AMP + grad_accum=4 + Fourier PE fixed m=160 σ=0.7 + SwiGLU + slice_num=16**
- VRAM: 28.5 GB; per-epoch: ~78s; budget: ~23 epochs in 30-min
- **Both seeds hit best val at final epoch** — headroom confirmed; more epochs would help

**Round-20 takeaway:** sn=16 is a massive −10% win. The slice_num monotonic trend (96→48→32→16 all improving) continues. sn=8 single seed also passed (62.476, trailing-5=64.05 — lowest of any run). The floor is below sn=16. sn=24 is unstable (std=6.07). Closed PRs #33 (α-gated Fourier, degenerate fixed point) and #17 (input jitter, 3-round exhaustion without beating current stack).

**Round-19 takeaway:** PR #32 (alphonse n_head) found n_head=2 at sn=64 beats n_head=4 by ~4pts (2-seed mean 66.37 vs 70.67). Sent back for rebase on sn=16 + 5-seed confirmation.

**Key prior insights still binding:**
- L1 (PR #3), sw=1 (PR #11), AMP+grad_accum=4 (PR #12), Fourier PE fixed σ=0.7 m=160 (PR #24), SwiGLU (PR #20), sn=32 (PR #27), sn=16 (PR #34). Seven compounding components.
- **Seed variance floor at sn=16 recipe:** std~1.742 val (2-seed). Use sn=16 3-seed anchor std from new PRs for merge criterion.
- **Multi-seed protocol:** 2-seed anchors mandatory for merge claims; 3-seed for strong claims.
- sn=24 is a stability dead-zone. sn=8 is stable but single-seed. sn=4/6 untested.
- Per-block Fourier injection dead (PRs #30, #33 both failed — mechanism either harmful or dead).
- Input jitter (gap/stagger) exhausted over 3 rounds — doesn't beat current stack.

---

## Student Status

| Student | Status | PR | Branch |
|---------|--------|----|--------|
| fern     | WIP (r20) | #36: slice_num floor sweep sn∈{4,6,8} | `fern/slice-num-floor-sweep` |
| tanjiro  | WIP (r20) | #37: n_head sweep {1,2,4,8} on sn=16 recipe | `tanjiro/n-head-sweep-sn16` |
| frieren  | idle (PR #34 MERGED) | — | — |
| nezuko   | WIP (r18) | #35: n_layers depth sweep on sn=32 recipe — STALE (pre-sn=16) | `nezuko/n-layers-sweep-sn32` |
| alphonse | WIP (r19) | #32: nh=2/sn=32 5-seed compound — STALE (pre-sn=16) | `alphonse/n-head-sweep` |
| edward   | WIP (r2) | #8: EMA + grad-clip on L1 — STALE (very old) | `edward/ema-gradclip-stability` |
| thorfinn | WIP (r13) | #29: Slice-bottleneck residual decoder — likely stale | `thorfinn/slice-bottleneck-decoder` |

**Note:** frieren is now idle (PR #34 merged). Will assign a new experiment.

**Idle students:** frieren. All others active or in-flight.

---

## PRs Ready for Review

None at this time.

---

## PRs In Progress (`status:wip`)

| PR | Student | Hypothesis | Status | Target |
|----|---------|-----------|--------|--------|
| #36 | fern     | slice_num floor sweep sn∈{4,6,8} vs sn=16 anchor | Fresh (r20) | Beat **60.581** / **61.813** |
| #37 | tanjiro  | n_head sweep {1,2,4,8} on sn=16 recipe | Fresh (r20) | Beat **60.581** / **61.813** |
| #35 | nezuko   | n_layers depth sweep {3,4,5,6,7} on sn=32 recipe | STALE (pre-sn=16) | Will need rebase |
| #32 | alphonse | nh=2×sn=32 5-seed compound (rebased for sn=32 at time) | STALE (pre-sn=16) | Will need rebase |
| #8  | edward   | EMA 0.999 + wider grad-clip on L1 | VERY STALE (r2) | Will need rebase |
| #29 | thorfinn | Slice-bottleneck residual decoder (PhysicsAttention, zero-init) | Likely stale | Will need rebase |

---

## Recent Decisions Log

- **Round 20 (r20 — 2026-04-24 08:30):**
  - **MERGED PR #34 (frieren slice_num=16):** New baseline **60.581 best-seed / 61.813 2-seed-mean val; 54.640 / 55.316 test**. −10% val vs sn=32. Decisive: 2-seed mean 5.21 pts below merge gate. Both seeds hit best val at final epoch (headroom confirmed). sn=8 (single-seed 62.476, trailing-5=64.05) very promising.
  - **Closed PR #33 (fern α-gated Fourier injection):** Mathematically degenerate fixed point — α_init=0 + zero-init projector has zero gradient for all parameters (dead mechanism). α_init=0.01 variant shows optimizer actively suppressing injection (alphas shrank 5× in epoch 1). Conclusively falsified.
  - **Closed PR #17 (tanjiro input jitter):** Three rounds of iteration; 3rd round on current stack shows gap-jitter 2-seed mean (75.08) missing baseline (69.85) by 5pts. Within-sweep lift (−2.3pts) is real but swamped by throughput-contention noise floor. Direction exhausted under current stack.
  - **Assigned PR #36 (fern): slice_num floor sweep sn∈{4,6,8}** — natural continuation of sn trend; sn=8 single-seed passed merge gate; find the floor. 3-seed anchor at sn=16 included.
  - **Assigned PR #37 (tanjiro): n_head sweep {1,2,4,8} on sn=16 recipe** — never cleanly tested at sn=16; PR #32 found n_head=2 beats n_head=4 on sn=64 recipe; hypothesis: fewer wider heads better at sn=16 where only 16 tokens exist.

- **Round 19 (r19 — 2026-04-24 07:30):**
  - **Sent back PR #32 (alphonse n_head + 3-seed anchor):** nh=2 at sn=64 gave 2-seed mean val 66.372 (best single-seed 64.161 / test 55.337). Monotonic n_head landscape: nh=2 << nh=4 < nh=8 << nh=16 (catastrophic). Sent back for rebase + 5-seed nh=2/sn=32 confirmation + nh=1 probe. **Now stale again due to sn=16 merge.**

---

## Current Research Focus and Themes

**Current baseline: 60.581 val (best seed) / 61.813 (2-seed mean) — test 54.640 / 55.316** (PR #34, sn=16 + σ=0.7 + SwiGLU + L1 + AMP + grad_accum=4)

The recipe has compounded 7 improvements. The sn=16 merge opened more compute headroom (~23 epochs vs 21 at sn=32; ~27 at sn=8). The dominant research frontier right now:

**Theme 1: Finish mapping the slice_num axis (HIGHEST PRIORITY)**
- sn=8/6/4 haven't been 2-seed confirmed. PR #36 (fern) addresses this.
- sn=8 trailing-5 (64.05) is the lowest of any run — strongly suggests sn=8 or below will beat sn=16 on 2-seed.
- Finding the true floor is essential — this axis has been the most productive of all.

**Theme 2: n_head sweep on sn=16 recipe**
- PR #37 (tanjiro) sweeps n_head ∈ {1, 2, 4, 8} at sn=16. Never tested on current recipe.
- Prior evidence (PR #32): n_head=2 beat n_head=4 on sn=64 by ~4pts.
- With only 16 tokens, n_head=1 or 2 may be strongly optimal (fewer, wider heads = richer cross-token representation).

**Theme 3: n_layers sweep on sn=16 recipe (pending PR #35 rebase)**
- PR #35 (nezuko) is stale (designed for sn=32). Needs rebase onto sn=16 recipe.
- sn=16's faster per-epoch makes deeper models (n_layers=6,7) more feasible in budget.
- Orthogonal to n_head and slice_num.

**Theme 4: Architecture experiments (medium priority)**
- Conditional LayerNorm / AdaLN on log(Re) — val_re_rand (60.41) is still the second-hardest split.
- n_hidden scaling (128→192) — sn=16 freed VRAM (28.5 GB), headroom for wider models.
- Horizontal-flip augmentation with Uy sign-flip — doubles effective training set.

**Theme 5: frieren idle — needs new assignment**
- frieren is now idle after merging PR #34. Assign next highest-EV experiment.
- Best candidate: **mlp_ratio sweep {2, 3, 4} on sn=16 recipe** — PR #25 (pre-σ=0.7) showed mr=3 gave 73.35 vs 73.66 anchor (~0.3pt improvement) but was stale; never tested cleanly on merged config. Low implementation cost.
- Alternative: **Conditional LayerNorm (AdaLN) on log(Re)** — strong physics motivation, targets val_re_rand directly.

---

## Potential Next Research Directions

### Highest-priority (next idle students)
- **mlp_ratio sweep {2,3,4} on sn=16 recipe** — PR #25 found mr=3 marginally better (stale baseline). Now needs clean test. 10 LOC diff.
- **Conditional LayerNorm / AdaLN on log(Re) + domain flags** — val_re_rand (60.41) is structurally the hardest remaining split. Re fundamentally changes flow regime. ~60 LOC.
- **Horizontal-flip augmentation with Uy sign-flip** — doubles effective training set from 1499 samples. ~80 LOC. Low risk, strong prior.
- **LR re-sweep on sn=16 recipe** — LR=5e-4 set at ~100 val; optimal may differ at 61 val. Sweep {2e-4, 5e-4, 1e-3} × cosine. ~10 LOC.

### Architecture (medium priority)
- **n_hidden scaling (128→192)** — sn=16 freed VRAM (28.5 GB); h=192 at sn=16 might be ~35 GB (feasible).
- **n_layers sweep on sn=16** — PR #35 needs rebase; sn=16's speed makes n_layers=6,7 viable in budget.
- **Coordinate-based slice assignment** (MLP on x,z,is_surface) — spatial clustering for better OOD inductive bias.

### Physics-informed
- **Near-surface volume band weighting** (3-tier loss using dsdf distance) — boundary layer nodes.
- **Kutta condition soft loss** at trailing edge — penalizes pressure discontinuity at TE.
- **Pressure-gradient weighted L1** — upweight nodes where |∇p| is large (boundary layer, stagnation).

### Speculative
- **Mamba surface decoder** — treat surface nodes as 1D sequence sorted by saf; Mamba block to refine.
- **Hard-example mining** — after epoch 10, oversample worst 20% val samples 2× in subsequent epochs.
- **sn sweep below 4** — if sn=4 doesn't collapse, try sn=2.

---

## Most Recent Research Direction from Human Team

No human issues received. No new directives.
