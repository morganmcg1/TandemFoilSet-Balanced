# SENPAI Research State

- **Last updated**: 2026-05-16 ~09:30 UTC
- **Branch**: `icml-appendix-charlie-pai2i-24h-r3`
- **Target**: TandemFoilSet 2D CFD surrogate; Transolver
- **Primary metric**: `val_avg/mae_surf_p` — lower is better
- **Per-run budget**: SENPAI_MAX_EPOCHS=50, SENPAI_TIMEOUT_MINUTES=30 (hard caps)

## Current best baseline

- `val_avg/mae_surf_p` = **86.77** (PR #3753, alphonse, `dsdf-clip`, epoch 19)
- **MERGED 2026-05-16 08:30 UTC**
- Change: `x_norm = x_norm.clamp(-3.0, 3.0)` after feature normalization (global, all 24 dims). Gain came from position/saf dims 0-3 tail clipping. DSDF dims 4-11 had 0% clipping. val_single_in_dist regressed +3.56 — follow-up #3818 tests surgical and soft clip.

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 102.00 |
| val_geom_camber_rc | 93.75 |
| val_geom_camber_cruise | 69.15 |
| val_re_rand | 82.17 |
| **val_avg** | **86.77** |

_Prior best (for tracking): PR #3513, val_avg=87.62. Cumulative stack: BF16 + Huber δ=1.0 + cosine T_max=20 + global ±3σ clip._

## High-value in-flight result

**PR #3759 askeladd (per-point-τ)**: val_avg=85.479 on OLD 87.62 baseline (−2.45%). If the mechanism stacks with the clip, expected val_avg ≈ 84.5–85.5. Sent back for rebase + single re-run to confirm stacking.

## Active PRs (round 6)

| # | Student | Slug | Status | Hypothesis |
|---|---|---|---|---|
| **#3759** | **askeladd** | **`per-point-temp`** | **Rebase pending — HIGH PRIORITY** | Per-point adaptive slice temperature; val_avg=85.479 on OLD baseline; awaiting rebased re-run on 86.77 base |
| **#3818** | **alphonse** | **`surgical-clip`** | **WIP** | Surgical dims-0-3 clip vs tanh soft-clip — recover val_single_in_dist from #3753 regression |
| **#3778** | **tanjiro** | **`rmsnorm`** | **WIP** | RMSNorm replacement for LayerNorm — LLaMA-family normalization, BF16 stability |
| **#3780** | **nezuko** | **`focal-loss`** | **WIP** | EMA per-sample focal weight γ=2 — adaptive hard-sample focus |
| **#3844** | **fern** | **`surf-attn`** | **WIP — NEW** | Surface-only all-to-all cross-attention after block stack; direct foil-surface coupling |
| **#3846** | **edward** | **`stream-fn`** | **WIP — NEW** | Stream function ψ auxiliary head; L_div = ‖u − curl(ψ)‖² soft incompressibility constraint |
| **#3847** | **thorfinn** | **`re-consistency`** | **WIP — NEW** | Re-perturbed forward pass; consistency loss penalizing p ∝ Re² scaling violations |
| **#3848** | **frieren** | **`dual-scale`** | **WIP — NEW** | DualScalePhysicsAttention: G_fine=64 + G_coarse=16 parallel heads with linear gate |

## Just closed (round 5/6)

| # | Student | Slug | Outcome |
|---|---|---|---|
| #3779 | thorfinn | re-stratified-loss | Closed — failure (+2.26); Re range too narrow, weights nearly uniform, regresses tandem |
| #3754 | edward | per-domain-norm | Closed — severe regression on val_single_in_dist (+5.94); per-domain norm backfires |
| #3755 | fern | swa | Closed — failure (+3.02); SWA unsuitable for heterogeneous-mesh training |
| #3756 | frieren | grad-accum-2 | Closed — stale_wip; no training output |
| #3757 | tanjiro | pre-ln | Closed — baseline already Pre-LN; student caught no-op; reassigned to RMSNorm |
| #3393 | thorfinn | surf-p-channel-weight | Closed — failure (90.98, +3.36); fails to stack with BF16 |
| #3709 | nezuko | cosine-t-max-25 | Closed — stale_wip; reassigned to focal loss |
| #3235 | askeladd | local-re-feature | Closed — 5h stale; saf_norm insight preserved (-9.7% on OLD 117.66) but unvalidated on current |

## Current research themes (round 6 — post plateau escalation)

**Promising pending result:**
- **Per-point adaptive temperature** (askeladd #3759): −2.45% on old baseline; pending rebase confirmation. If stacking is confirmed, this becomes the new baseline and changes round-7 priorities.

**Architecture experiments (in flight):**
- **Surface-only cross-attention** (fern #3844): all-to-all communication over is_surface nodes after the Transolver block stack. Direct path for Kutta-Joukowski circulation coupling. Expected to help val_single_in_dist and val_re_rand.
- **Dual-scale slice hierarchy** (frieren #3848): G_fine=64 + G_coarse=16 parallel attention heads with learned gate. Multi-scale operator learning (MNO ICLR 2026 precedent). Doubles attention compute; expected ~50-60 GB VRAM.

**Physics-informed regularization (in flight):**
- **Stream function auxiliary head** (edward #3846): psi_head predicts ψ; autograd through positions gives ∂ψ/∂x, ∂ψ/∂z; L_div on surface nodes with lambda_div=0.001. Divergence-free constraint from first principles.
- **Re-consistency loss** (thorfinn #3847): perturb log(Re) by δ∈[0.95,1.05]; penalize pressure deviation from p∝Re² scaling law. Soft physics constraint, ~2× forward compute.

**Input representation cleanup (in flight):**
- **Surgical clip** (alphonse #3818): dims 0-3 only at ±3σ vs tanh soft-clip. Targeting recovery of val_single_in_dist (+3.56 regression from #3753).
- **RMSNorm** (tanjiro #3778): LLaMA-family normalization; BF16 gradient stability hypothesis.
- **Focal loss** (nezuko #3780): EMA per-sample difficulty weighting γ=2.

## Refuted approaches (do NOT re-assign)

- **Per-channel output weighting**: fails to stack with BF16 (3.36 worse).
- **Per-domain output normalization**: backfires on low-sample splits.
- **SWA**: incompatible with heterogeneous-mesh training.
- **Re-stratified sample weighting**: Re range too narrow; weights nearly uniform.
- **Gradient accumulation N=2**: no benefit identified; stale_wip pattern.
- **Cosine T_max=25**: stale_wip; low priority schedule tweak.
- **Pre-LN swap**: already implemented in baseline.
- **n_head=8 with dim_head=16**: too thin; major failure.
- **mlp_ratio=4**: FFN not bottlenecked at n_hidden=128.
- **slice_num=32**: slice budget G=64 is binding; don't reduce.

## Round-4 ideas still unassigned (candidates for round 7)

- **Temperature annealing with schedule** (edward #3700 was flat τ, not schedule-based): τ linear 1.0→0.1 over epochs
- **β2=0.99 retest** at lower LR (4e-4) — test/val asymmetry in frieren #3707 suggests noise helps generalization
- **Incompressibility direct FD penalty** (original idea #5) — harder version of stream fn; requires mesh stencils
- **AoA/geometry-conditioned slice routing** — sinusoidal AoA embedding into slice logits

## Scoring.py NaN bug (branch-wide)
`test_geom_camber_cruise/000020.pt` has 761 `inf` values in GT. Workaround: rank on val_avg/mae_surf_p; report test_avg as mean over 3 finite splits. Fix requires modifying `data/scoring.py` (marked read-only).
