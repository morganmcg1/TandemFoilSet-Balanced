# SENPAI Research State — willow-pai2-r5

- **Updated:** 2026-04-27 18:00 (round 1 — bootstrap)
- **Advisor branch:** `icml-appendix-willow-pai2-r5`
- **Research tag:** `willow-pai2-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-pai2-r5`

---

## Most recent direction from human researcher team

None at boot — no human issues at this time.

## Current Baseline

**Bare-baseline state** on this branch (Transolver MSE, sw=10, slice_num=64, n_layers=5, no AMP, no Fourier, no SwiGLU). No PRs merged yet.

**Target to beat (prior best from kagent_v_students round):**
- `val_avg/mae_surf_p = 49.077 (best seed) / 49.443 (2-seed mean)` — test 42.473 / 42.450 (PR #39 `nezuko/nl3-sn16-compound`)
- Recipe: L1 + sw=1 + AMP + grad_accum=4 + Fourier PE fixed m=160 σ=0.7 + SwiGLU + slice_num=8 + n_layers=3 + n_head=4 + mlp_ratio=2

**Best UNMERGED frontier (PR #32 from kagent_v_students):** single-head triple compound (nh=1, nl=3, sn=16) — `test_avg ≈ 40.927` single-seed. Never reproduced or merged.

## Current research focus and themes

This is a fresh advisor branch (round 1 of 5 willow-pai2 cohort). The full kagent_v_students research history is preserved on the `kagent_v_students` branch and is the source of truth for prior knowledge.

**Round-1 strategy: port the proven recipe + close the unmerged frontier in parallel with diverse hypothesis families.**

8 ideas dispatched (see `research/RESEARCH_IDEAS_2026-04-27_18:00.md`):

| # | Slug | Student | Theme |
|---|------|---------|-------|
| 1 | recipe-port-frontier-anchor | nezuko | **ANCHOR** — port full proven recipe + nh=1 triple compound |
| 2 | sn-floor-mapping-extended | frieren | Compute-reduction: sn ∈ {2, 4} probe |
| 3 | nl-floor-and-mlp-depth | fern | Compute-reduction: nl=1 + deeper preprocess MLP |
| 4 | n-hidden-shrink-floor | alphonse | Compute-reduction: n_hidden ∈ {64, 96} |
| 5 | dim-head-shape-preserving | askeladd | Architecture: decouple dim_head from n_head |
| 6 | chord-aligned-frame-features | edward | Feature engineering: chord-relative coords + normals |
| 7 | per-block-film-conditioning | tanjiro | Conditioning: per-block FiLM on (Re, AoA, NACA) |
| 8 | surface-finetune-phase | thorfinn | Training schedule: surface-only finetune phase |

**Coverage:** Compute-reduction (1-4), architecture (1, 5), features (6), conditioning (7), training (8). No two students overlap on the same knob.

## Student status

| Student | Status | PR | Hypothesis |
|---------|--------|----|------------|
| alphonse | idle (round 1) | — | (assigning Idea 4) |
| askeladd | idle (round 1) | — | (assigning Idea 5) |
| edward | idle (round 1) | — | (assigning Idea 6) |
| fern | idle (round 1) | — | (assigning Idea 3) |
| frieren | idle (round 1) | — | (assigning Idea 2) |
| nezuko | idle (round 1) | — | (assigning Idea 1) |
| tanjiro | idle (round 1) | — | (assigning Idea 7) |
| thorfinn | idle (round 1) | — | (assigning Idea 8) |

## Potential next research directions (round 2+)

### High-priority follow-ups
- Ideas 9 (wall-distance features) and 10 (torch.compile + FlashAttn) are queued for round 2 if students from this round become idle quickly or if fresh ideas miss.
- Triple compounds: if Idea 1 confirms nh=1 wins and Idea 4 confirms n_hidden=96 wins, test the nh=1 × n_hidden=96 × nl=3/sn=8 quadruple in round 2.

### Long-standing unaddressed (deferred from prior rounds)
- Multi-scale slice tokens (2 sets at different scales).
- Cross-attention surface readout head.
- Curriculum on Re or mesh size.
- Per-channel loss weighting with empirical (Ux, Uy, p) weighting.
- Replace `placeholder` mean parameter with proper learnable cls/sink/registers.

### Methodological
- Increase `SENPAI_MAX_EPOCHS` if multiple recipes remain budget-bound at terminal.
- Standardize a 2-seed protocol with explicit `--seed` flag from the anchor PR onward.
