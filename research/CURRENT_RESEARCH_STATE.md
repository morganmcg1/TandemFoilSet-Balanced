# SENPAI Research State — charlie-r3

- **Updated:** 2026-04-27 16:00 (round 1 — kickoff)
- **Advisor branch:** `icml-appendix-charlie-r3`
- **Research tag:** `charlie-r3`
- **Track:** ICML appendix
- **W&B:** disabled (local metrics only)

---

## Most recent direction from human researcher team

No directives in flight. Charlie-r3 is the third round of ICML-appendix experiments on TandemFoilSet, starting from a fresh vanilla `train.py` (no W&B, strengthened local metric logging).

## Current state on charlie-r3

- **Baseline:** vanilla Transolver, MSE loss, no AMP, no Fourier PE — no measured `val_avg/mae_surf_p` yet on this branch. The first merged PR seeds the entry in `BASELINE.md`.
- **Reference target:** kagent_v_students recipe → val 49.077 / test 42.473 (best seed); 2-seed mean val 49.443 / test 42.450.
- **Open PRs:** none.
- **Idle students (8):** charlie3-alphonse, charlie3-askeladd, charlie3-edward, charlie3-fern, charlie3-frieren, charlie3-nezuko, charlie3-tanjiro, charlie3-thorfinn.

## Round 1 strategy (assigning now)

For an ICML appendix track, we want a clean, well-scoped set of experiments that simultaneously:
1. **Re-establish the proven kagent recipe** on charlie-r3 (anchor PR).
2. **Document per-component contributions** as appendix-quality ablations.
3. **Push beyond the kagent best** with novel directions on the strong recipe.

### Round 1 assignments

| Student | Slug | Theme | Goal |
|---------|------|-------|------|
| charlie3-alphonse | `kagent-recipe-replication` | Anchor | Port full proven recipe (L1+sw=1+AMP+grad_accum=4+Fourier+SwiGLU+nl=3/sn=8). 3-seed verification. **Aim val ≤ 50.** |
| charlie3-askeladd | `loss-and-surf-weight-ablation` | Ablation #1 | Clean 2×2 ablation: {MSE, L1} × {sw=10, sw=1} on vanilla baseline. Multi-seed. |
| charlie3-edward | `fourier-sigma-sensitivity` | Ablation #2 | On L1+sw=1+AMP base, sweep Fourier σ ∈ {0.5, 0.7, 1.0, 1.5} + no-Fourier control. |
| charlie3-fern | `swiglu-vs-gelu-ablation` | Ablation #3 | On L1+sw=1+AMP+Fourier base, swap GELU FFN for SwiGLU. Multi-seed. |
| charlie3-frieren | `slice-num-depth-floor` | Push beyond | On full recipe, 2-D probe (n_layers, slice_num) ∈ {2,3,4} × {4,8,16}. Find floor. |
| charlie3-nezuko | `lr-warmup-cosine-floor` | Push beyond | Full recipe + LR warmup (3 / 5 ep) + cosine min_lr_ratio ∈ {0, 0.1}. (Resumes kagent PR #40 thread.) |
| charlie3-tanjiro | `n-head-on-recipe` | Push beyond | Full recipe (nl=3/sn=8), sweep n_head ∈ {1, 2, 4, 8}. (Resumes kagent PR #32 thread.) |
| charlie3-thorfinn | `near-surface-band-loss` | Novel | Full recipe + 3-tier loss with extra weight on volume nodes within distance d of surface. |

## Themes in flight

**Theme A — Recipe re-establishment.** Alphonse's compound replication is the highest-priority PR for round 1. Without it, every other PR's "novel direction on top of strong recipe" claim is unanchored. If alphonse confirms ~49 val on charlie-r3, that becomes the new baseline for round 2.

**Theme B — Appendix-grade ablations.** Askeladd, edward, fern produce the per-component ablation table the paper appendix needs.

**Theme C — Push beyond kagent.** Frieren (depth/sn floor), nezuko (LR schedule), tanjiro (n_head), thorfinn (BL band loss) probe the open directions the kagent round didn't finish.

## Potential next research directions (round 2+)

- **Regularization at the small-model regime** (nl=3/sn=8 trains 38 epochs; dropout/DropPath might prevent overfit on remaining budget).
- **Extended training budget** if env limits permit (kagent was bumping the ceiling).
- **Boundary-layer aware decoder head** (cross-attn or zero-init residual on surface tokens).
- **Pressure asinh / robust target reparameterization** — kagent PR #9 closed; might be revived on the new strong base.
- **Horizontal-flip augmentation with Uy sign-flip** — physics-exact 2× data; kagent PR #15 closed without strong evidence.
- **n_hidden shrink** {64, 96, 128} — extends compute-reduction theme to width.
- **Multi-resolution attention** (coarse-to-fine slice tokens).
- **SAM / SWA / model soups** — late-stage optimization tricks for the appendix.
- **Physics-informed auxiliary losses** (∇·u = 0, Kutta condition near trailing edge).
- **Sample-wise normalization with Re-predicted scale** — kagent PR #26 closed but mechanism is sound.

The researcher-agent is generating a fresh idea slate (`research/RESEARCH_IDEAS_2026-04-27_16:00.md`) for round 2+.
