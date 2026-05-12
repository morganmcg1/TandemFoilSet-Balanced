# SENPAI Research State — willow-pai2g-24h-r1

- 2026-05-12 18:55 (round 1 first results in)
- Most recent research direction from human researcher team: This launch is a
  controlled 24h/48h Charlie-vs-Willow logging ablation. Each training run is
  capped at `SENPAI_TIMEOUT_MINUTES=30`. The fleet is scoped to research tag
  `willow-pai2g-24h-r1` and advisor branch
  `icml-appendix-willow-pai2g-24h-r1`. Treat experiments as isolated; do not
  inspect or compare to any other PR / branch / experiment outside this scope.

- Current research focus and themes:

  Round 1 first two completions (PRs #1372 n_head=8 and #1378 n_hidden=192,
  both now closed) revealed two crosscutting blockers that affect ALL
  capacity-changing variants under the 30-min wall-clock cap:

  1. **Throughput.** A standard run reaches only ~10–11 of the spec'd 50
     epochs before the timeout fires. Wider/deeper variants reach even fewer
     (~10 epochs at n_hidden=192). Under this regime the
     `CosineAnnealingLR(T_max=epochs)` schedule barely moves the LR from
     peak across the whole run, so models are still in their early-noise
     regime when evaluated.
  2. **Stability.** Both completed runs produced non-finite pressure
     predictions on `test_geom_camber_cruise` (one of four test splits)
     while staying finite on the other three. This corrupted
     `test_avg/mae_surf_p` for both PRs. Strong signal that an undertrained
     Transolver blows up the pressure head on extreme OOD samples.

  Round 2 attacks these blockers head-on while round 1 finishes:

  - **frieren #1383** — `grad-clip-1.0`: add
    `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` to the
    training step. Directly addresses the cruise-test pressure inf. If this
    works, all subsequent capacity sweeps can produce clean `test_avg`.
  - **tanjiro #1384** — `bf16-autocast`: wrap the forward + loss in
    `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`. Cheapest
    path to ~1.3–1.8× per-step speedup on Hopper/Ampere → more epochs fit
    inside the 30-min cap. Compounds with every future hypothesis.

- Round 1 in-flight (6 PRs still WIP, results expected within the next
  hour or so):
  - alphonse #1353 — `surf-weight-25`
  - askeladd #1354 — `lr-1e-3`
  - edward #1356 — `n-layers-7`
  - fern #1360 — `slice-num-128`
  - nezuko #1377 — `mlp-ratio-4`
  - thorfinn #1382 — `wd-3e-4`

- Round 2 (so far):
  - frieren #1383 — `grad-clip-1.0`
  - tanjiro #1384 — `bf16-autocast`

- Potential next research directions and themes once round 1/2 land:
  - **Schedule fix**: pass `--epochs <fits-in-30min>` so `T_max=epochs`
    actually anneals. Likely worth ~5–15 MAE points on its own.
  - **Compound winners**: best loss-weight × best LR × best stability fix
    into a multi-knob frontier run.
  - **Loss reformulation**: Huber on volume term (high-Re samples push
    extreme y values that pure MSE penalizes disproportionately), or
    per-channel weighting (Ux/Uy/p with separate weights).
  - **LR schedule variants**: OneCycle, warmup+cosine, lower min-lr.
  - **Architecture**: RoPE / Fourier positional encoding on (x, z), gated
    FFN (SwiGLU), pre-norm vs post-norm placement.
  - **Data**: coordinate normalization (centering on foil 1), per-domain
    stats vs global stats.
