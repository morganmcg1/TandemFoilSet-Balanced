# SENPAI Research State

- **Updated:** 2026-04-24 09:30 (round 22 — 3 closes, 3 assignments, mlp_ratio fix cherry-picked)
- **Advisor branch:** `kagent_v_students`
- **Research tag:** `kagent-v-students-20260423-2055`
- **W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current Baseline

**val_avg/mae_surf_p = 54.210 (best seed, s=1) / 54.476 (2-seed mean) — test 47.484 / 47.336** (PR #35, W&B group `nezuko/n-layers-sn32`)
- Per-split val (s=1, ep=32): in_dist=61.24 | camber_rc=67.65 | camber_cruise=33.63 | re_rand=54.33
- Config: **L1 + sw=1 + AMP + grad_accum=4 + Fourier PE fixed m=160 σ=0.7 + SwiGLU + slice_num=32 + n_layers=3**
- Best epoch: 32 (both seeds at final epoch — headroom remains)
- nl=3 2-seed std: 0.376 val (tight)

**Round-22 outcomes (this round):**
- Closed PR #36 (fern sn floor): null at nl=5; floor is sn=16, not lower. Capacity-bound, not throughput-bound.
- Closed PR #37 (tanjiro n_head at sn=16): within-recipe nh=1 wins (2-seed mean 58.38 vs 62.46 anchor), but stale (+3.91 above current baseline). **n_head monotonic trend robust across sn=16 & sn=64.**
- Closed PR #38 (frieren mlp_ratio): null (mr=2 wins). **Critical infrastructure fix: `--mlp_ratio` CLI was unwired; frieren's 2-line fix (commit b8330ac) cherry-picked.** PR #25's earlier mr=3 claim was silently using mr=2.
- Assigned 3 fresh experiments on nl=3/sn=32 recipe (PR #40, #41, #42).

**Key prior insights still binding:**
- L1 (PR #3), sw=1 (PR #11), AMP+grad_accum=4 (PR #12), Fourier PE σ=0.7 m=160 (PR #24), SwiGLU (PR #20), sn=32 (PR #27), nl=3 (PR #35). Seven compounding components.
- **Compute-reduction theme has won 4×:** AMP, sn=32, sn=16, nl=3. n_hidden shrink untested.
- **n_head monotonic trend robust (nh=1 < nh=2 < nh=4 << nh=8) across sn=16, sn=64, nl=5.** Alphonse's PR #32 r3 tests at nl=3/sn=32 — the critical compound.
- **Noise floor recipe-dependent:** nl=3/sn=32 2-seed std 0.376 val (tight); earlier recipes wider.
- `--mlp_ratio` CLI fix cherry-picked (commit b8330ac). All future flag values work.

---

## Student Status

| Student | Status | PR | Branch |
|---------|--------|----|--------|
| frieren  | WIP (r22) | #40: LR warmup + cosine-floor revival on nl=3 recipe | `frieren/lr-warmup-nl3` |
| fern     | WIP (r22) | #41: n_hidden shrink sweep {64,96,128,160} on nl=3/sn=32 | `fern/n-hidden-shrink-sweep` |
| tanjiro  | WIP (r22) | #42: Dropout + DropPath sweep on nl=3 (regularization) | `tanjiro/dropout-sweep-nl3` |
| nezuko   | WIP (r21) | #39: nl=3 × sn=16 compound + sn=8 & nl=2 probes | `nezuko/nl3-sn16-compound` |
| alphonse | WIP (r21) | #32 r3: nh=1/nh=2 × nl=3/sn=32 compound + shape-preserving | `alphonse/n-head-sweep` |
| edward   | WIP (r2) | #8: EMA + grad-clip on L1 — VERY STALE | `edward/ema-gradclip-stability` |
| thorfinn | WIP (r13) | #29: Slice-bottleneck residual decoder — likely stale | `thorfinn/slice-bottleneck-decoder` |

**Idle students:** none. All 7 GPUs occupied.

---

## PRs Ready for Review

None at this time.

---

## PRs In Progress (`status:wip`)

| PR | Student | Hypothesis | Target |
|----|---------|-----------|--------|
| #40 | frieren  | LR warmup + cosine-floor revival on nl=3 | Beat **54.48** (2-seed mean) |
| #41 | fern     | n_hidden shrink sweep {64, 96, 128, 160} on nl=3/sn=32 | Beat **54.48** |
| #42 | tanjiro  | Dropout + DropPath sweep on nl=3 (regularization opens up at 32 epochs) | Beat **54.48** |
| #39 | nezuko   | nl=3 × sn=16 compound + sn=8 & nl=2 probes | Beat **54.48** |
| #32 | alphonse | r3: nh=1/nh=2 × nl=3/sn=32 compound + shape-preserving control | Beat **54.48** |
| #8  | edward   | EMA 0.999 + wider grad-clip on L1 — VERY STALE | Will need full rebase |
| #29 | thorfinn | Slice-bottleneck residual decoder — likely stale | Will need rebase |

**Merge threshold:** winner 2-seed mean ≤ **54.10** for strict merge (nl=3 anchor std 0.376).

---

## Recent Decisions Log

- **Round 22 (r22 — 2026-04-24 09:30):**
  - **Cherry-picked PR #38's mlp_ratio fix (b8330ac)** — `--mlp_ratio` was unwired before. PR #25's earlier mr=3 claim was silently mr=2. Infrastructure now correct.
  - **Closed PR #36 (fern sn floor):** sn floor at nl=5 recipe is sn=16; below that is capacity-bound (sn=4 shows instability). Current baseline (nl=3/sn=32) beats all variants by 8+ val.
  - **Closed PR #37 (tanjiro n_head at sn=16):** within-recipe win (nh=1 2-seed 58.38 beats nh=4 anchor 62.46 by 2.45σ), monotonic trend robust across sn values. But +3.91 above current baseline. Alphonse's PR #32 r3 will test the compound at nl=3/sn=32.
  - **Closed PR #38 (frieren mlp_ratio):** mr=2 wins at nl=5/sn=16 recipe (as expected). Infrastructure fix is the real deliverable.
  - **Assigned PR #40 (frieren):** LR warmup + min_lr + higher peak lr sweep on nl=3 (new regime never tested — nezuko's LR work was at nl=5).
  - **Assigned PR #41 (fern):** n_hidden shrink {64, 96, 128, 160} on nl=3/sn=32. Tests whether the compute-reduction theme (now 4/4 wins) extends to n_hidden.
  - **Assigned PR #42 (tanjiro):** dropout + DropPath sweep on nl=3. At 32 epochs of training, regularization is potentially useful for the first time.

- **Round 21:** PR #35 merged (nl=3 → val 54.48). PR #32 sent back.
- **Round 20:** sn=16 merged (PR #34). α-gated Fourier + input jitter closed.
- **Earlier:** PR #27 merged (sn=32 first), PR #24 merged (σ=0.7 + SwiGLU compound), PR #20 SwiGLU, PR #12 AMP.

---

## Current Research Focus and Themes

**Baseline at val 54.48 / test 47.34** after 7 compounding merges. The compute-reduction theme (AMP → sn=32 → sn=16 → nl=3) has won 4 times in a row. Active hypotheses split into three families:

**Theme 1 (highest priority): Continue compute-reduction axis**
- PR #39 (nezuko): nl=3 × sn=16 compound — most likely to land <52 val if additive.
- PR #41 (fern): n_hidden shrink {64, 96} — extends theme to width axis, never shrunk.
- PR #32 r3 (alphonse): nh=1/nh=2 at nl=3/sn=32 — compute-reduction via fewer heads.

**Theme 2: LR/regularization regime at 32 epochs**
- PR #40 (frieren): nl=3's 32-epoch regime is genuinely new; LR schedule effects may transfer differently.
- PR #42 (tanjiro): regularization (dropout/DropPath) opens up at 32-epoch regime.

**Theme 3: Stale revival (low priority)**
- #29 (thorfinn decoder), #8 (edward EMA) need rebase if anyone touches them.

---

## Potential Next Research Directions (round 23+)

### High-priority if current round produces winners
- **Triple compound:** if nl=3 × sn=16 and n_hidden=96 both win independently, test nl=3 × sn=16 × n_hidden=96.
- **n_hidden × n_head compound:** if n_head=1 wins at nl=3/sn=32 and n_hidden=96 wins at nl=3/sn=32, test both.
- **Extended epoch analysis:** if nl=3/sn=16 still hits best val at final epoch, formally request longer training budget from human team.

### Long-standing unaddressed
- Horizontal-flip augmentation (closed; revisit with the physics corrections alphonse found).
- Cross-attention surface decoder (thorfinn's WIP PR #29 pre-dates nl=3; needs rebase).
- Kutta condition / physics-informed auxiliary loss.
- Warm-start from merged checkpoint for architectural expansion experiments.

### Methodological
- Request human team input: should we increase SENPAI_MAX_EPOCHS now that multiple configs are budget-bound at 32 epochs?
