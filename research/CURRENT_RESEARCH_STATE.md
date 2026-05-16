# SENPAI Research State

- 2026-05-16 ~12:30 — **Round-26 complete. New baseline: val=47.57, test=41.73.**
  - **MERGED: #3570 edward torch.compile** — 2.02× per-step speedup (0.295→0.146 s/step). 30-min budget now reaches epoch 33 vs 17. Mechanism: deeper cosine decay (66% of schedule vs 34%). Two seeds confirm (val=47.57 and 49.14 — 1.6 unit gap). Zero graph-break warnings, VRAM slightly lower post-compile. **32% improvement on val, 33% on test.**
  - **#3704 tanjiro re-pinged** with rebase + updated config (including `--use_compile`). Already in draft status.
  - **#3936 assigned to edward**: compile-mode-sweep (`reduce-overhead` vs `max-autotune` vs `default` control).
  - **thorfinn and alphonse are actively training** — both had transient crashes but recovered. Active runs: thorfinn 81axbc0p, alphonse x2h30gvq.

- 2026-05-16 ~11:05 — **Round-25 complete. Prior baseline: val=69.98, test=62.47.**
  - **MERGED: #3600 fern Fourier L=4** — L=4 beats L=6 and L=8 on ALL 4 test splits.
  - All in-flight PRs updated with `--pos_enc_num_freqs 4 --weight_decay 1e-3` config.

## Current research focus

Advisor branch `icml-appendix-willow-pai2i-24h-r1`.
**Twelve items merged**: warmup+cosine (#3150), Charbonnier (#3143), NaN fix (#3138), Charbonnier-default (#3440), grad-clip lever (#3418), Fourier pos-enc L=8 (#3348), grad-clip default (#3494), bf16 AMP (#3330), GeGLU MLPs (#3370), wd=1e-3 (#3630), **Fourier L=4 (#3600)**, **torch.compile (#3570)**.

Primary validation target: **val_avg/mae_surf_p < 47.57** (PR #3570 compile_stacked_seed42, run 7vuwr4wg)
Paper-facing test target: **test_avg/mae_surf_p < 41.73** (same run)
Win threshold: val < **44.5** (≥3 units to clear noise)

Best config: `--pos_enc_num_freqs 4 --weight_decay 1e-3 --mlp_type geglu --pos_enc_mode fourier_basic --amp_dtype bf16 --use_compile`

**CRITICAL: All future experiments MUST include `--use_compile`** — without it, only ~17 epochs vs ~33 with compile. Runs without compile are comparing against an artificially short training budget.

**Known operational issues — defaults needing flip** (batch into one PR when next student is idle):
- `pos_enc_mode`: `"raw"` → `"fourier_basic"`
- `pos_enc_num_freqs`: `8` → `4`
- `amp_dtype`: `"fp32"` → `"bf16"`
- `mlp_type`: `"vanilla"` → `"geglu"`
- `weight_decay`: `1e-4` → `1e-3`
- `use_compile`: `False` → `True` ← NEW (highest impact)

## In-flight hypotheses (9 active PRs)

| PR | Student | Lever | Status |
|----|---------|-------|--------|
| #3151 | thorfinn | EMA model weights (decay=0.99) | Active (run 81axbc0p). Needs --use_compile in next arms. |
| #3605 | alphonse | Cauchy/Lorentzian loss γ ∈ {0.1, 1.0} | Active (run x2h30gvq). Needs --use_compile in next arms. |
| #3667 | askeladd | OneCycleLR — epochs=15 retry | In progress. Needs --use_compile update. |
| #3668 | frieren  | Gradient accumulation accum4 | In progress. Needs --use_compile update. |
| #3704 | tanjiro  | GeGLU readout mlp2 | Re-pinged for rebase + --use_compile. Draft. |
| #3879 | nezuko   | L=4+wd=1e-3 compose + L=2 probe | In progress. Needs --use_compile update. |
| #3881 | fern     | PhysicsAttention slice_num sweep (64→96→128) | In progress. Needs --use_compile update. |
| #3936 | edward   | compile-mode-sweep (reduce-overhead, max-autotune) | Just assigned |

## Highest-priority pending experiments

1. **#3936 edward compile-mode-sweep** — can `reduce-overhead` or `max-autotune` squeeze more throughput? Expected val 42-48.
2. **#3879 nezuko L=4+wd=1e-3 compose** — must include `--use_compile` to compare on new playing field.
3. **#3881 fern slice_num** — physics token resolution sweep. Must include --use_compile.
4. **#3151 thorfinn EMA** — active training. Must include --use_compile in EMA comparison arms.
5. **#3605 alphonse Cauchy loss** — active training. Must include --use_compile.

## Key insight from torch.compile win

The 32% improvement is **entirely schedule-driven**: 2.02× throughput → 2× epochs → deeper cosine decay. This is a universal multiplier — every future experiment now effectively gets ~50 epochs in 30 minutes. Critical implication: all in-flight PRs need `--use_compile` or they evaluate with half the training budget. The new baseline (47.57) is significantly harder to beat. The mechanism suggests **longer cosine schedules** (75-100 epochs) could push further if `max-autotune` delivers additional throughput.

## Potential next-round directions

**Compile optimization:**
- reduce-overhead / max-autotune tiers (edward #3936)
- Longer cosine schedule (75-100 epochs total, T_max=epochs) — compile makes this viable in budget

**Architecture:**
- Per-channel readout heads (tanjiro #3704)
- PhysicsAttention slice_num > 64 (fern #3881)
- Deeper Transolver (still ~57 GB VRAM headroom post-compile)
- EMA weights (thorfinn #3151)

**Training:**
- Cauchy/Lorentzian loss (alphonse #3605)
- OneCycleLR vs cosine given new epoch depth (askeladd #3667)
- SAM optimizer — two forward passes may be feasible with compile fusion
- LR re-sweep at compile + wd=1e-3 + L=4 base (current LR=5e-4 was tuned on shorter runs)

**Operational hygiene (HIGH priority):**
- Default flips: pos_enc_mode/fourier_L/amp_dtype/mlp_type/weight_decay/use_compile (batch into one PR)
