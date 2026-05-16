# SENPAI Research State

- **Date:** 2026-05-16 16:12 UTC (Round 4 in progress on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)
- **PR #3594 (alphonse SF-AdamW R2) MERGED 15:34 UTC**: baseline 80.893 → **65.618** (−18.88% absolute, −16.80% paired). Largest single-experiment gain of the round.
- ⭐ **PR #3980 (frieren Lion) STRONG WIN, SENT BACK FOR REBASE 16:05 UTC**: paired Δ −24.43% val / −23.72% test. Arm B absolute val=63.336 < current baseline 65.618 (−3.48%), test=60.549 < 62.853 (−3.67%). **Largest paired Δ on the track, lowest single-run absolute.** Mergeable once branch rebased and reproduction confirmed.
- **PR #3492 (nezuko n_hidden=192 R3) SENT BACK 16:10 UTC**: third paired replication of capacity win (−5.17% val / −5.22% test paired) but on stale stack (absolute 79.409 > 65.618). R4 retest under SF-AdamW assigned.
- **PR #3390 (thorfinn T_max=20 R2) CLOSED 16:08 UTC**: superseded by SF-AdamW. Cosine axis formally closed.
- **PR #3777 (askeladd SDF) CLOSED 15:55 UTC**: +2.52% paired regression. Geometric-input axis closed.
- **PR #3985 (edward AGC R1) SENT BACK 15:50 UTC**: real paired Δ −2.02% but on stale baseline. R2 retest under SF-AdamW assigned.
- **PR #4019 (alphonse SF 2×2 factorial)** assigned 15:38 UTC: clip × EMA composition sweep.
- **PR #4038 (askeladd SF-LR sweep)** assigned 16:00 UTC: lr ∈ {5e-4 control, 1e-3, 2e-3, 5e-3}.
- **PR #4051 (thorfinn SF-wd sweep)** assigned 16:12 UTC: wd ∈ {1e-4 control, 3e-4, 1e-3, 1e-2}.
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
3. **LR under SF-AdamW** — SF paper recommends 1×-10× larger LR than scheduled approach; Arm B in #3594 was still descending at cap. `askeladd #4038 4-arm sweep`.
4. **AGC under SF-AdamW** — Edward's R1 showed −2.02% paired on stale baseline; does per-tensor direction normalization compound with SF's Polyak averaging? `edward #3985 R2 retest`.
5. **Capacity (n_hidden=192) + SF** — R2 showed −8.21% under AdamW; could compound massively under SF. Awaiting nezuko #3492 R3 result.

### Key in-flight experiments (16:12 UTC)

| Student | PR | Hypothesis | Status | vs SF baseline 65.618 |
|---------|----|----|----|----|
| ⭐ **frieren** | **#3980** | **Lion vs AdamW (post-rebase reproduction)** | Sent back 16:05 for rebase | **Strongest candidate — −24% paired, 63.336 absolute** |
| **alphonse** | **#4019** | **SF-AdamW clip×EMA 2×2 factorial** | In progress (since 15:38) | Directly refines 65.618 baseline |
| **askeladd** | **#4038** | **SF-AdamW LR sweep** {5e-4, 1e-3, 2e-3, 5e-3} | In progress (since 16:00) | Directly refines 65.618 baseline |
| **thorfinn** | **#4051** | **SF-AdamW weight-decay sweep** {1e-4, 3e-4, 1e-3, 1e-2} | Just assigned 16:12 UTC | Directly refines 65.618 baseline |
| **nezuko** | **#3492** | **n_hidden=192 + SF R4** (paired retest) | Sent back 16:10 for SF retest | Highest-EV: 3rd paired Δ replication of capacity win |
| **edward** | **#3985** | **AGC R2: SF + AGC vs SF + clip=1.0** | Sent back 15:50 for SF retest | If wins, replaces `--grad_clip_norm 1.0` |
| tanjiro | #4003 | Clip thresh R2 under AdamW: {0.05, 0.1, 0.15, 0.25} | In progress | AdamW context; informs clip axis |
| fern | #4012 | Sobolev / edge-gradient L1 loss | In progress | Orthogonal loss axis; still relevant |

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
| #3777 | SDF input features | +2.52% val / +3.99% test | Redundant with `dsdf`+FiLM; per-sample p95 norm breaks comparability |
| #3390 | T_max=20 R2 (Arm A only, B aborted) | superseded mid-run | Cosine axis closed by SF-AdamW; another "two mechanisms same role" |

## Axes status

### Schedule axis (CLOSED — SF-AdamW wins, T_max=20 closed)

- Cosine T_max=15: merged (#3289), now superseded by SF-AdamW
- Cosine T_max=20 (#3390 R2): **CLOSED 16:08 UTC** — Arm A completed (84.480 control reproduction), Arm B aborted by student when SF-AdamW merged mid-run (smart hold call). Cosine axis formally closed.
- **SF-AdamW (#3594): MERGED** — eliminates schedule.

### SF-AdamW composition axis (ACTIVE — #4019 in flight)

Clip threshold and EMA necessity under SF-AdamW. Both addressed by 2×2 factorial at alphonse #4019.

### Gradient clipping axis (ACTIVE — tanjiro #4003 tighter sweep + alphonse #4019 SF-clip)

AdamW optimum: clip=0.25 (monotone, 100% clip rate). SF-AdamW optimum: unknown — different gradient variance structure under constant LR. Two separate questions.

### Lion optimizer (frieren #3980 — STRONG WIN, awaiting rebase reproduction)

R1 result: **paired Δ −24.43% val / −23.72% test on AdamW+cosine+clip=0.25 stack**. Arm B absolute val 63.336 BEATS the current SF-AdamW baseline (65.618, −3.48%); test 60.549 vs 62.853 (−3.67%). **Largest paired Δ on the track, lowest single-run absolute number on any single experiment.** Mechanism reading: sign projection is the L∞ extreme of "direction normalization" — internally consistent vs AdamW + L2 clip which combines two normalizers in series (Adam's per-coordinate `v̂` then global rescaling). PR has merge conflicts post-#3594 — sent back for rebase and re-run confirmation. **If reproduction holds, Lion becomes the new canonical optimizer** displacing AdamW; next experiment Lion + SF-AdamW (mechanisms likely orthogonal: Lion = update direction, SF = schedule removal + iterate averaging).

### AGC (edward #3985 — R2 retest under SF-AdamW, in flight)

R1 showed −2.02% paired val / −3.97% paired test on the AdamW + clip=0.25 stack (absolute 81.552 > current 65.618 baseline). R2 retest now in flight: SF-AdamW + AGC vs SF-AdamW + clip=1.0. Key diagnostic from R1: `any_clip` rate = 100% every step, meaning AGC at λ=0.01 behaves as a permanent per-tensor direction normalizer rather than the safety-clamp regime the paper assumes. The mechanism that won is structurally similar to the clip=0.25 win in #3906 (direction normalization at full saturation).

### Capacity axis (nezuko #3492 R4 SF retest — in flight, HIGHEST-EV non-Lion)

R3 result: **third consecutive paired replication of n_hidden=192 win** (R1 −2.99%, R2 −8.21%, R3 −5.17% on FiLM+clip stack). Mechanism is robust: wider features compose super-additively with global inductive priors. But R3 absolute 79.409 > 65.618 — cannot merge on stale stack. R4 retest now in flight: paired n_hidden=128 vs 192 on the full SF-AdamW stack. If wins, capacity + SF compound could be the largest absolute win on the track. **Seed=1 pinning now available** (student added `--seed` flag mid-R3).

### Depth axis (CLOSED) — n_layers=5 is optimal

### Slice-num axis (CLOSED) — slice_num=64 is optimal at n_hidden=128

### FiLM injection-count + per-block-capacity axes (CLOSED at n_hidden=128)

### Physics-aware loss axis (fern #4012 — ACTIVE)

First experiment targeting loss formulation. Still relevant under any optimizer stack.

### SDF input features (CLOSED — #3777 askeladd)

Geometric-input axis closed. SDF regressed +2.52% paired val / +3.99% test 3-split. Three independent mechanism causes: redundant with `dsdf` (8-D shape descriptor already in input), competes with FiLM for geometric conditioning role, per-sample p95 normalization breaks cross-sample comparability (2.2× spread). Lesson: when a new feature targets a role another mechanism already owns, it has to replace not stack.

### Learning-rate axis under SF-AdamW (askeladd #4038 — ACTIVE, NEW)

NEW HIGHEST-PRIORITY axis. Under cosine, lr=5e-4 averaged to ~1.5e-4 effective over the run; under SF it stays at 5e-4 throughout — so SF-effective LR is 3× higher than AdamW-effective LR. SF README recommends 1×-10× larger LR than the scheduled approach (so 5e-4 → 5e-3). #4038 sweeps the full recommended range with paired control.

## Potential next hypotheses (not yet assigned)

All future SF-AdamW experiments: `--use_schedule_free` (no `--cosine_t_max`).

1. **SF clip×EMA factorial (ASSIGNED #4019, alphonse)** — in flight
2. **SF LR sweep (ASSIGNED #4038, askeladd)** — in flight
3. **SF weight-decay sweep (ASSIGNED #4051, thorfinn)** — in flight
4. **AGC under SF (ASSIGNED #3985 R2, edward)** — in flight
5. **n_hidden=192 + SF (ASSIGNED #3492 R4, nezuko)** — highest-EV in-flight
6. **Lion reproduction post-rebase (ASSIGNED #3980 rebase, frieren)** — STRONGEST candidate
7. **Lion + SF-AdamW** — if Lion lands, immediate next experiment to test mechanism orthogonality
8. **Lion + n_hidden=192** — if both win independently, compound test
9. **Lion + AGC** — Lion is L∞ direction normalization, AGC is per-tensor; could compose
10. **Longer budget for SF-AdamW** — constant LR means more epochs = better; Arm B was at ~1.8 val/epoch slope at cap
11. **Best-of-(LR × clip × EMA × wd)** — after #4019 #4038 #4051 land, combine winners
12. **Sobolev loss + winning stack** — if fern #4012 wins, combine
13. **Mixup / CutMix** — may help geom_camber_rc split
14. **FiLM MLP hidden widening** — `film_mlp_hidden=256`; conditional on #3492 R4 outcome

## Operational notes

- **Current best command:** `python train.py --amp_dtype bf16 --use_ema --ema_decay 0.999 --film_cond --two_shot_film --grad_clip_norm 1.0 --use_schedule_free`
- **Clip threshold under SF is UNKNOWN:** Do NOT use clip=0.25 as default for SF-AdamW until #4019 resolves.
- **Cosine schedule superseded:** New experiments should NOT include `--cosine_t_max`.
- **Two-mechanisms-for-same-role insight:** Now confirmed across **three independent cases**: depth=4 proxying clip stability (subsumed by clip); T_max=20 proxying better late-LR use (subsumed by SF-AdamW); SDF input feature proxying geometric conditioning (owned by FiLM + `dsdf`). When a new axis changes the optimization dynamic, re-test hypotheses whose evidence was a proxy for the same role — they often falsify. Research-program-level principle worth its own writeup section.
- **Direction-normalization mechanism family:** clip=0.25 (#3906, L2 with 100% sat), AGC (#3985, per-tensor), Lion (#3980, L∞ via sign projection). All three win on the same mechanism via different geometries. Lion is the L∞ extreme; clip is L2; AGC is per-tensor-L2. Lion's −24% paired Δ suggests L∞ is the dominant projection for this task's gradient geometry.
- **GH API rate limits:** Recurring; last incident 14:53 UTC.
- **test_geom_camber_cruise NaN:** pre-existing; use 3-split mean for test comparisons.
- **Askeladd next assignment #4038:** LR sweep under SF-AdamW (highest-leverage hyperparameter on the new baseline).
- **Thorfinn next assignment #4051:** wd sweep under SF-AdamW (orthogonal regularization knob, never re-tuned for new stack).
- **Zero idle students at 16:12 UTC:** all 8 actively assigned (5 in flight refining SF baseline, 1 rebasing Lion winner, 2 on orthogonal axes Sobolev+clip-tighter).
- **Seed variance ~±1.5-2%:** Paired Δ within session is reliable; absolute deltas need confirmation.
- **SF-AdamW budget insight:** At constant LR, no natural "done" signal except budget cap. More epochs = better results (slope still ~1.8/epoch at epoch 17 cap).
