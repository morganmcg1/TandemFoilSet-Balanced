# SENPAI Research State

- **Updated:** 2026-04-23 21:45
- **Advisor branch:** `kagent_v_students`
- **Research tag:** `kagent-v-students-20260423-2055`
- **W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current Baseline

**val_avg/mae_surf_p = 103.036** (PR #3, frieren/loss-l1-sw10, W&B run `w2jsabii`)
- Per-split: in_dist=133.19 | camber_rc=117.21 | camber_cruise=70.11 | re_rand=91.63
- Config: L1 loss, surf_weight=10, n_hidden=128, n_layers=5, slice_num=64, lr=5e-4

**Known blocker:** `test_avg/mae_surf_p` is NaN on all runs (GH issue #10 — corrupt `test_geom_camber_cruise/000020.pt`). Val-based decisions continue unaffected.

---

## Student Status

| Student | Status | PR | Branch |
|---------|--------|----|--------|
| frieren  | WIP | #11: L1 + fine surf_weight sweep | `frieren/l1-surf-weight-sweep` |
| fern     | WIP | #12: Throughput — AMP + grad accumulation | `fern/throughput-amp` |
| tanjiro  | WIP | #5 (round 2): Channel-weighting on top of L1 | `tanjiro/channel-weighted-loss` |
| nezuko   | WIP | #6 (round 2): LR + WSD + seed-replay on L1 | `nezuko/lr-schedule-sweep` |
| alphonse | WIP | #7: Fourier PE + FiLM on log(Re) | `alphonse/fourier-pe-film-re` |
| edward   | WIP | #8: EMA + gradient clipping | `edward/ema-gradclip-stability` |
| thorfinn | WIP | #9: Pressure target reparameterization | `thorfinn/pressure-target-reparam` |

**Idle students:** none.

---

## PRs Ready for Review

None.

---

## PRs In Progress (`status:wip`)

### Round 2 (rebased on L1 baseline)

| PR | Student | Hypothesis | Target |
|----|---------|-----------|--------|
| #11 | frieren  | Fine surf_weight sweep under L1 (sw ∈ {1,2,3,5,10,15,20,30}) | Beat 103.036 |
| #12 | fern     | AMP (bf16) + grad accumulation to unlock 25–35 epochs vs current 14 | Beat 103.036 |
| #5  | tanjiro  | Channel-weight fine sweep (psurf ∈ {14,17,20,23,27}) + vol_weight on top of L1 | Beat 103.036 |
| #6  | nezuko   | 3-seed replay of LR floor + WSD scheduler + 1e-3 LR cosine on L1 | Beat 103.036 |

### Round 1 still in flight (will rebase on completion)

| PR | Student | Hypothesis |
|----|---------|-----------|
| #7  | alphonse | Fourier PE on (x,z) + FiLM conditioning on log(Re) |
| #8  | edward   | EMA of weights × gradient clipping grid |
| #9  | thorfinn | Pressure target reparameterization (asinh/robust/per-domain) |

PRs #7–9 started before L1 was merged. Their results will be evaluated vs the track baseline (103.036). If their internal sweeps show improvement relative to their in-PR MSE baseline, we'll determine whether the effect likely compounds with L1 or not and decide merge vs send-back accordingly.

---

## Most Recent Research Direction from Human Team

No human issues received as of 2026-04-23 21:45.

---

## Current Research Focus and Themes

**Round 2 priorities (ordered by expected impact):**

1. **Throughput unlock (fern #12)** — The most critical experiment. Every run stops at epoch 14/50 due to 30-min wall-clock. If AMP + larger batch gets us to epoch 30+, this compounds with EVERYTHING else and raises the ceiling for all subsequent experiments. Highest EV of any experiment in flight.

2. **Loss shape signal (frieren #11)** — L1-sw10 ≈ L1-sw20 suggests the optimum may be BELOW sw=10. Fine sweep will clarify. Low-complexity, potentially −2–6%.

3. **Channel-weighting on L1 (tanjiro #5 r2)** — The signal was real on MSE (psurf=20 gave −4.9%). It may amplify or cancel on L1. If it compounds: psurf=20 on L1 could give another 5%.

4. **Schedule disambiguation (nezuko #6 r2)** — LR floor effect may be noise. Seed replay and WSD will clarify before we commit this to the baseline.

5. **Fourier PE + FiLM (alphonse #7)** — Testing feature conditioning; expected to help on OOD splits (camber generalization, Re generalization). Medium complexity, strong theoretical motivation.

6. **EMA + grad clipping (edward #8)** — Stability tricks; cheapest addition, likely small but stackable gain.

7. **Target reparameterization (thorfinn #9)** — asinh/robust/per-domain normalization for heavy-tailed p. Orthogonal to loss shape, may compound.

---

## Potential Next Research Directions (Round 3+)

Top candidates from `research/RESEARCH_IDEAS_2026-04-23_21:00.md`:

### High-priority physics-informed
- **Horizontal-flip augmentation + Uy sign-flip** — free 2× training set at exact physics; high expected lift on OOD generalization.
- **Panel-method residual learning** — predict viscous correction on top of potential-flow prior. Hardest to implement but highest OOD ceiling.
- **Near-surface volume-band weighting** — three-tier (far-vol / near-vol / surf) loss structure; boundary layer resolution drives surface pressure.

### Architectural
- **Cross-attention surface decoder** — dedicated high-capacity decoder for surface nodes; shared trunk does volume.
- **Sample-wise z-score normalization** — normalize each sample's targets by its own y_std (predicted from log(Re)) to fix the 10× per-sample y_std variance.

### Compounding plan
Once throughput (fern #12) and the best of {#5, #6, #11} merge:
1. Combine AMP with the best loss weights.
2. Re-run Fourier/FiLM + EMA on the improved baseline.
3. Introduce horizontal-flip augmentation as a standalone PR.
4. Target reparameterization + channel-weighting together as a combined experiment.
