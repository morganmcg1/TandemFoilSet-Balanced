# SENPAI Research State

- **Date:** 2026-05-16 15:40 UTC (Round 4 in progress on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)
- **PR #3594 (alphonse SF-AdamW R2) MERGED 15:34 UTC**: baseline 80.893 → **65.618** (−18.88% absolute, −16.80% paired). **Largest single-experiment gain of the round.**
- **PR #4019 (alphonse SF 2×2 factorial)** assigned 15:38 UTC: clip × EMA composition sweep.
- **Previous merges this session:** #3906 clip=0.25 (→80.893), #3594 SF-AdamW clip=1.0 (→65.618).

## Current research focus

**Nine axes confirmed and merged.** Compound stack: Huber + bf16 + EMA decay=0.999 + single-shot FiLM + two-shot FiLM + **SF-AdamW (no cosine)** + grad_clip_norm=1.0.

**Current best: 65.618 val_avg/mae_surf_p** (alphonse #3594, SF-AdamW + clip=1.0, 2026-05-16 15:34)

⚠️ **MAJOR STACK UPDATE:** Baseline dropped from 80.893 → 65.618 (−18.88%). All in-flight experiments are running on a stack meaningfully weaker than the new baseline. Apply the rebase-if-positive-Δ protocol against the old clip=0.25 baseline when reading results; then re-test winners on the SF-AdamW stack.

⚠️ **COSINE SCHEDULE AXIS CLOSED:** SF-AdamW wins by a wide margin. Any T_max=20 result from thorfinn #3390 cannot match SF-AdamW — keep as negative reference but do not merge.

⚠️ **CLIP THRESHOLD UNDER SF UNKNOWN:** The merged win used clip=1.0 (not the AdamW-optimized clip=0.25). Do NOT assume clip=0.25 is optimal under SF-AdamW until alphonse #4019 reports.

### Mechanism map (updated post-SF-AdamW merge)

| Role | Mechanism that owns it |
|---|---|
| Loss alignment with eval metric | Huber β=1.0 |
| Compute throughput / epoch count | bf16 AMP |
| Schedule | **None — SF-AdamW eliminates scheduler** |
| Trajectory smoothing (long time scale) | EMA decay=0.999 (+ SF Polyak avg — overlap TBD) |
| Physics-parameter conditioning | FiLM (single + two-shot) |
| Per-step gradient geometry | grad_clip_norm (threshold TBD under SF) |
| Schedule-free iterate convergence | **SF-AdamW** (just merged) |

### Highest-priority open questions

1. **Clip threshold under SF-AdamW** — is clip=0.25 (AdamW optimum) still optimal? `alphonse #4019 arm A vs B`.
2. **EMA redundancy under SF** — does external EMA add anything on top of SF's Polyak averaging? `alphonse #4019 arm A vs C, B vs D`.
3. **Capacity (n_hidden=192) + SF** — R2 showed −8.21% under AdamW; could compound massively under SF. Awaiting nezuko #3492 R3 result.

### Key in-flight experiments (15:40 UTC)

| Student | PR | Hypothesis | Status | vs new SF baseline |
|---------|----|----|----|----|
| **alphonse** | **#4019** | **SF-AdamW clip×EMA 2×2 factorial** | **Just assigned 15:38 UTC** | **Directly refines 65.618 baseline** |
| tanjiro | #4003 | Clip thresh R2 under AdamW: {0.05, 0.1, 0.15, 0.25} | In progress | AdamW context; informs clip axis |
| edward | #3985 | AGC vs global clip=0.25 (AdamW) | In progress | AdamW stack; if wins, re-test under SF |
| frieren | #3980 | Lion + clip=0.25 vs AdamW + clip=0.25 | In progress (rebasing) | AdamW stack; if wins, re-test under SF |
| thorfinn | #3390 | T_max=20 on clip=1.0 stack (R2) | In progress | ⚠️ Moot vs SF; keep as negative reference |
| fern | #4012 | Sobolev / edge-gradient L1 loss | In progress | Orthogonal loss axis; still relevant |
| nezuko | #3492 | n_hidden=192 + clip=1.0 R3 | In progress (rebasing) | Highest-EV in-flight (R2 was −8.21%) |
| askeladd | #3777 | SDF input features | Recovering (poll-gap fixed 15:24) | Geometric-input axis; still relevant |

**Primary metric:** `val_avg/mae_surf_p` (lower is better)

## Merged winners (chronological)

| PR | Hypothesis | val_avg delta | New val_avg |
|----|-----------|------------|---------|
| #3094 | Huber loss (alphonse) | −15.7% vs MSE | 111.531 |
| #3290 | bf16 AMP (askeladd) | −8.98% vs Huber | 101.519 |
| #3289 | Cosine T_max=15 (thorfinn) | −10.3% vs Huber fp32 | 100.059 (fp32) |
| #3126 | EMA decay=0.999 Karras-ramp (nezuko) | −1.06% vs bf16+T_max=15 arm | 96.464 |
| #3122 | FiLM conditioning (frieren) | −4.00% vs EMA baseline | 92.606 |
| #3584 | Two-shot FiLM (frieren) | −3.05% vs FiLM baseline | 89.784 |
| #3511 | Grad clip=1.0 (tanjiro) | −9.05% vs two-shot FiLM | 81.660 |
| #3906 | Clip=0.25 (tanjiro) | −3.42% paired vs clip=1.0 | 80.893 |
| **#3594** | **SF-AdamW (alphonse R2)** | **−16.80% paired vs matched cosine** | **65.618** |

## Falsified / closed hypotheses

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #3278 | Per-channel p-upweighting | +3-21% regression | Domain variance > channel bias |
| #3364 | lr=1e-3+warmup on bf16 | +8.3% regression | bf16 noise amplifies higher LR |
| #3321 | lr=1e-3/1.5e-3+warmup on fp32+bf16 | +2.2-12% regression (6 arms) | Higher LR dead end |
| #3443 | lr ∈ {2.5e-4, 3.5e-4} on bf16+T_max=15 | +0.78-2.60% regression | LR axis closed: 5e-4 is magnitude optimum |
| #3595 | n_layers=6 vs 5 on full FiLM stack | +2.47% regression | Depth costs wall-clock epochs |
| #3758 | n_layers=4 R3 on clip stack | +3.14% val / +3.34% test | Depth subsumed by clip; two mechanisms same role don't compound |
| #3117 | Fourier scale=2 (R2-R4) | −0.10% at R4 (tie) | FiLM absorbed Fourier |
| #3365 | batch_size=6/8 on bf16 | bs=4 best | GPU compute-bound |
| #3684 | slice_num=32/64/96 on FiLM stack | 64 best | 64 is knee point at n_hidden=128 |
| #3681 | Three-shot FiLM | +4.08% regression | Wrong injection site |
| #3829 | Per-block FiLM heads | Paired Δ −0.53% (noise floor) | FiLM saturated at n_hidden=128 |
| #3830 | Lookahead wrapper | Paired Δ +0.42% val | Redundant with EMA |

## Axes status

### Schedule axis (CLOSED — SF-AdamW wins)

- Cosine T_max=15: merged (#3289), now superseded by SF-AdamW
- Cosine T_max=20 (#3390): in flight (R2); expected moot vs SF-AdamW
- **SF-AdamW (#3594): MERGED** — eliminates schedule. Cosine axis closed.

### SF-AdamW composition axis (ACTIVE — #4019 in flight)

Clip threshold and EMA necessity under SF-AdamW. Both addressed by 2×2 factorial at alphonse #4019.

### Gradient clipping axis (ACTIVE — tanjiro #4003 tighter sweep + alphonse #4019 SF-clip)

AdamW optimum: clip=0.25 (monotone, 100% clip rate). SF-AdamW optimum: unknown — different gradient variance structure under constant LR. Two separate questions.

### Lion optimizer (frieren #3980 — in flight, AdamW context)

If wins, re-test under SF-AdamW. Lion + SF could be additive if they address orthogonal mechanisms.

### AGC (edward #3985 — in flight, AdamW context)

Same re-test protocol applies if AGC wins.

### Capacity axis (nezuko #3492 R3 — in flight, HIGHEST-EV)

n_hidden=192 on clip stack. R2 Δ −8.21% — could compound massively under SF-AdamW. Win here leads directly to n_hidden=192 + SF-AdamW follow-up.

### Depth axis (CLOSED) — n_layers=5 is optimal

### Slice-num axis (CLOSED) — slice_num=64 is optimal at n_hidden=128

### FiLM injection-count + per-block-capacity axes (CLOSED at n_hidden=128)

### Physics-aware loss axis (fern #4012 — ACTIVE)

First experiment targeting loss formulation. Still relevant under any optimizer stack.

### SDF input features (askeladd #3777 — ACTIVE, recovering)

Geometric-input axis. Poll-gap self-healed 15:24 UTC.

## Potential next hypotheses (not yet assigned)

All future SF-AdamW experiments: `--use_schedule_free` (no `--cosine_t_max`).

1. **SF clip×EMA factorial (ASSIGNED #4019, alphonse)** — in flight
2. **n_hidden=192 + SF-AdamW** — if nezuko #3492 R3 wins, follow up immediately
3. **Longer budget for SF-AdamW** — constant LR means more epochs = better; Arm B was at ~1.8 val/epoch slope at cap
4. **LR sweep under SF** — SF README notes 1×-10× larger LR often works; test lr ∈ {5e-4, 1e-3, 2e-3}
5. **Lion + SF-AdamW** — if Lion wins on AdamW (#3980), test the composition
6. **AGC + SF-AdamW** — if AGC wins on AdamW (#3985), test the composition
7. **Sobolev loss + SF-AdamW** — if fern #4012 wins, combine with new stack
8. **Mixup / CutMix** — may help geom_camber_rc split
9. **FiLM MLP hidden widening** — `film_mlp_hidden=256`; conditional on #3492 outcome

## Operational notes

- **Current best command:** `python train.py --amp_dtype bf16 --use_ema --ema_decay 0.999 --film_cond --two_shot_film --grad_clip_norm 1.0 --use_schedule_free`
- **Clip threshold under SF is UNKNOWN:** Do NOT use clip=0.25 as default for SF-AdamW until #4019 resolves.
- **Cosine schedule superseded:** New experiments should NOT include `--cosine_t_max`.
- **Two-mechanisms-for-same-role insight:** Depth=4 was proxying clip stability; T_max=20 was proxying better late-LR use; both subsumed. When a new axis changes the optimization dynamic, re-test hypotheses whose evidence was a proxy for the same role.
- **GH API rate limits:** Recurring; last incident 14:53 UTC.
- **test_geom_camber_cruise NaN:** pre-existing; use 3-split mean for test comparisons.
- **Askeladd #3777 recovered:** Poll-gap self-healed at 15:24 UTC. Training in progress.
- **Seed variance ~±1.5-2%:** Paired Δ within session is reliable; absolute deltas need confirmation.
- **SF-AdamW budget insight:** At constant LR, no natural "done" signal except budget cap. More epochs = better results (slope still ~1.8/epoch at epoch 17 cap).
