# SENPAI Research State — willow-pai2g-24h-r1

- 2026-05-12 19:35 (round 1: 3 of 8 done; round 2 in flight)
- Most recent research direction from human researcher team: This launch is a
  controlled 24h/48h Charlie-vs-Willow logging ablation. Each training run is
  capped at `SENPAI_TIMEOUT_MINUTES=30`. The fleet is scoped to research tag
  `willow-pai2g-24h-r1` and advisor branch
  `icml-appendix-willow-pai2g-24h-r1`. Treat experiments as isolated; do not
  inspect or compare to any other PR / branch / experiment outside this scope.

- Current research focus and themes:

  Three of eight round-1 PRs are complete (#1372 n_head=8, #1378 n_hidden=192,
  #1382 wd=3e-4) and all three exposed the same dual-blocker pattern under the
  30-min wall-clock cap:

  1. **Throughput.** Standard runs hit ~10–11 epochs out of the spec'd 50
     before the cap fires. `CosineAnnealingLR(T_max=50)` decays LR by only
     ~10% across the whole run, so models stay near peak LR (high-noise
     regime) all the way through evaluation.
  2. **Stability.** All three completed runs produced non-finite pressure
     predictions on the hardest OOD split (`test_geom_camber_cruise`) while
     Ux/Uy and the other three test splits stayed finite. This corrupted
     `test_avg/mae_surf_p` to NaN. The pattern is now unambiguous: the
     undertrained Transolver blows up the pressure head on extreme OOD
     samples in the cruise tandem domain (unseen camber M=2–4).

  Subtler issue flagged by thorfinn's results comment: `data/scoring.py`
  uses `err * mask.double()` accumulation which propagates NaN if `err`
  contains any `inf` (since `inf * 0 = NaN`). `torch.where(mask, err, 0)`
  would be NaN-safe. **`data/` is read-only per program.md**, so we cannot
  fix scoring directly — instead the active strategy is to prevent inf at
  the source via grad clipping (#1515) and to make `test_avg/mae_surf_p`
  computable on every future run.

  Round-1 informational ranking (not a settled baseline since all test_avg
  are NaN):

  | PR | Change | val_avg/mae_surf_p | partial test_avg (3 of 4) |
  |---|---|---:|---:|
  | #1382 | wd=3e-4 | **149.40** | 153.20 |
  | #1372 | n_head=8 | 153.84 | 141.53 |
  | #1378 | n_hidden=192 | 155.16 | 159.62 |

  Round 2 attacks the blockers head-on so capacity-changing variants can be
  retried with clean test numbers later:

  - **frieren #1515** — `grad-clip-1.0`: insert
    `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` between
    `loss.backward()` and `optimizer.step()`. Directly addresses the
    cruise-test pressure inf. If this works, every subsequent experiment
    can report a finite `test_avg/mae_surf_p`.
  - **tanjiro #1516** — `bf16-autocast`: wrap forward + (val/test) forward
    in `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`. Expected
    ~1.3–1.8× per-step speedup → ~15–22 epochs in 30 min vs 10–12 today.
    Compounds with every future hypothesis (more anneal, more training).
  - **thorfinn (queued)** — `huber-loss-vol`: replace MSE on the volume
    term with Huber loss (delta in normalized space). Robustness against
    high-Re outliers; may indirectly help cruise-test pressure stability.

- Round 1 in-flight (5 PRs still WIP — currently quiet but pods running
  ~2h+; expected to finish within next hour as the wider/deeper variants
  have higher per-epoch cost):
  - alphonse #1353 — `surf-weight-25`
  - askeladd #1354 — `lr-1e-3`
  - edward #1356 — `n-layers-7`
  - fern #1360 — `slice-num-128`
  - nezuko #1377 — `mlp-ratio-4`

- Round 2 (so far):
  - frieren #1515 — `grad-clip-1.0`
  - tanjiro #1516 — `bf16-autocast`
  - thorfinn — `huber-loss-vol` (this turn)

- Potential next research directions once round 1/2 land:
  - **Schedule fix**: pass `--epochs <fits-in-30min>` so `T_max=epochs`
    actually anneals — likely worth ~5–15 MAE points on its own.
  - **Compound winners**: best loss-weight × best LR × stability/throughput
    fixes into a multi-knob frontier run.
  - **LR schedule variants**: OneCycleLR (built-in warmup+anneal for short
    budgets), warmup+cosine, lower min-lr.
  - **Architecture**: RoPE / Fourier positional encoding on (x, z), gated
    FFN (SwiGLU), pre-norm vs post-norm placement.
  - **Data**: coordinate normalization (centering on foil 1), per-domain
    stats vs global stats.
