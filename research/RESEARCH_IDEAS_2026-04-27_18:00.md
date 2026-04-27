# Research Ideas — 2026-04-27 (willow-pai2 round 5, round-1)

**Compiled:** 2026-04-27 18:00 UTC by researcher-agent
**Advisor branch:** `icml-appendix-willow-pai2-r5` (target repo `morganmcg1/TandemFoilSet-Balanced`)
**W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-pai2-r5`
**Primary metric:** `val_avg/mae_surf_p` (lower is better); paper-facing: `test_avg/mae_surf_p`

## Context summary (load-bearing)

This branch's `train.py` is the **bare baseline** (Transolver, MSE, sw=10, no AMP, no Fourier PE, no SwiGLU, slice_num=64, n_layers=5, n_head=4, mlp_ratio=2). The CLI exposes only `--lr`, `--weight_decay`, `--batch_size`, `--surf_weight`, `--epochs`. Every proven recipe component (L1 loss, AMP+grad_accum, Fourier PE, SwiGLU FFN, slice_num override, n_layers override, n_head override, seed) has to be wired into both code AND CLI by the student.

**Proven recipe from `kagent_v_students` (best 2-seed merge, PR #39):**
- L1 + sw=1 + AMP + grad_accum=4 + Fourier PE fixed m=160 σ=0.7 + SwiGLU + slice_num=8 + n_layers=3 + n_head=4 + mlp_ratio=2
- val_avg/mae_surf_p = **49.443** (2-seed mean), test_avg = **42.450** (2-seed mean)
- Both seeds at terminal epoch (ep=38) — model NOT converged, headroom remains.

**Best UNMERGED frontier (PR #32 closed):** nh=1 single-head triple compound (nh=1, sn=16, nl=3) → test_avg ≈ **40.93** single-seed. Never reproduced/merged.

**The compound trends still descending at end of prior round:**
- sn axis at nl=3: sn=32 (54.48) → sn=16 (51.98) → sn=8 (49.44). Floor not found.
- Depth at sn=16: nl=5 (61.81) → nl=3 (51.98) → nl=2 (50.72 single-seed). Floor not found.
- n_head: nh=4 → nh=2 → nh=1 monotonic across multiple recipes. nh=1 × nl=3 × sn=16 NEVER merged.

**Mechanism of all compute-reduction wins:** shrink → cheaper epoch → more epochs in 30-min budget → better val. The 8 in-flight architectural fixes are layered onto this dynamic.

---

## Top 8-10 ideas for parallel assignment

### Idea 1: recipe-port-frontier-anchor — Port full proven recipe + nh=1 single-head triple compound (the leaderboard top-1 frontier closure)
- **Hypothesis:** With the entire `kagent_v_students` proven stack (L1 + sw=1 + AMP + ga=4 + Fourier PE m=160 σ=0.7 + SwiGLU + sn=8 + nl=3 + mlp_ratio=2 + seed flag), AND the closed-but-unmerged nh=1 frontier from PR #32, 2-seed mean val ≤ **44** and test ≤ **40** are both achievable. Triple compound (nh=1 × sn=16 × nl=3) reproduces test ~40.9 (PR #32 best run `ip8hn4tx`); applied at sn=8 it should land below 40.
- **Why it could win:** This is a known-good stack — every component independently merged in the prior round. The only piece never confirmed at full compound is nh=1, which has shown a robust monotonic trend (nh=1 < nh=2 < nh=4 across nl=5/sn=16, nl=5/sn=64, and the unverified nl=3/sn=16 PR #32 point) and reaches test 40.93 single-seed. Frontier closure with strict 2-seed multi-seed protocol provides the round's anchor.
- **Concrete change:**
  - Wire all flags into `Config` and CLI: `--loss_type {mse,l1}`, `--amp bool`, `--grad_accum int`, `--fourier_features {none,fixed}`, `--fourier_m int`, `--fourier_sigma float`, `--swiglu bool`, `--slice_num int`, `--n_layers int`, `--n_head int`, `--mlp_ratio int`, `--seed int`.
  - Code edits in `train.py`:
    - L1 loss: replace `(pred - y_norm)**2` with `(pred - y_norm).abs()` when `cfg.loss_type=="l1"`.
    - bf16 autocast: wrap forward+loss in `with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if cfg.amp:`.
    - Grad accum: scale loss by `1/cfg.grad_accum`, only `optimizer.step()/zero_grad()` every `grad_accum` micro-batches.
    - Fourier PE: in `Transolver.__init__`, build `B = torch.randn(2, fourier_m) * fourier_sigma; self.register_buffer("B", B)`; in `preprocess`, prepend `[sin(2π x B), cos(2π x B)]` to features (input dim becomes `X_DIM + 2*fourier_m`). Adjust `MLP` `n_input` accordingly.
    - SwiGLU: replace the standard MLP inside `TransolverBlock` with three projections `(W_gate, W_up, W_down)`, hidden = `(2/3) * mlp_ratio * n_hidden` (round to multiple of 8). Use `silu(x@W_gate) * (x@W_up) @ W_down`.
    - `seed`: `torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)` early in `train.py`; also seed numpy.
    - Override `model_config` from CLI: `slice_num=cfg.slice_num`, `n_layers=cfg.n_layers`, `n_head=cfg.n_head`, `mlp_ratio=cfg.mlp_ratio`.
  - First run `--debug` to verify the wired stack reaches a finite val number after 3 epochs.
  - Sweep: 4 GPUs × 2 configs × 2 seeds = 8 runs.
    - GPU pair A (anchor reproduction): `nh=4 sn=8 nl=3` (full proven recipe), seeds {0, 1}.
    - GPU pair B (frontier closure): `nh=1 sn=16 nl=3` (PR #32 frontier), seeds {0, 1}.
    - GPU pair C (frontier extension): `nh=1 sn=8 nl=3` (extend frontier to merged sn floor), seeds {0, 1}.
    - GPU pair D (depth-floor probe): `nh=1 sn=8 nl=2` (push depth too), seeds {0, 1}.
- **Recipe baseline to start from:** This IS the recipe baseline. Reproduce command:
  ```bash
  python train.py --loss_type l1 --surf_weight 1 --amp true --grad_accum 4 --batch_size 4 \
    --fourier_features fixed --fourier_m 160 --fourier_sigma 0.7 --swiglu \
    --slice_num 8 --n_layers 3 --n_head 4 --mlp_ratio 2 --seed 0 \
    --wandb_group <student>/recipe-port --wandb_name <student>/recipe-port-anchor-s0
  ```
- **Risk / falsification:** If wired stack 2-seed mean val > 52 (vs prior 49.4), there is a wiring bug; debug. If nh=1×sn=16×nl=3 2-seed mean > 50, the PR #32 single-seed top result was lucky; close. Expected fail mode: SwiGLU or Fourier PE wired with wrong shape causing a +20 val regression.
- **Suggested student:** **nezuko** (led 3 of top 4 architectural wins in prior round; trustworthy on multi-seed protocol).

---

### Idea 2: sn-floor-mapping-extended — Push slice_num floor below 8 (sn ∈ {2, 4} on full recipe at nl=3 and nl=2)
- **Hypothesis:** sn axis monotonic 32→16→8 buys ~2.5 val per halving with no observed floor. sn=4 should reach val ≈ 47, sn=2 ≈ 45 (or break to instability). nl=2/sn=8 single-seed (50.72) suggests nl=2 still works; nl=2/sn=4 may compound to ≈48.
- **Why it could win:** Five consecutive compute-reduction wins (AMP, sn=32, sn=16, sn=8, nl=3). Best epochs both sn=8 seeds hit terminal; ~40 epochs landed in budget. Halving sn again should buy 4-5 more epochs and continue the trend. Frontier mapping rather than tweaking.
- **Concrete change:**
  - Wires same flags as Idea 1.
  - Sweep at fixed nh=4, nl=3:
    - GPU 0-1: sn=8 anchor (re-pin baseline), seeds {0,1}
    - GPU 2-3: sn=4, seeds {0,1}
    - GPU 4-5: sn=2, seeds {0,1}
    - GPU 6-7: sn=4 nl=2 double-compound, seeds {0,1}
  - Watch trailing-5 val and best_epoch. If sn=2 destabilizes (trailing-5 spike), report training instability cleanly.
- **Recipe baseline to start from:** Same as Idea 1 reproduce command, change `--slice_num` to {2, 4} per arm. wandb_group: `<student>/sn-floor-extended`.
- **Risk / falsification:** sn=2 may produce a degenerate slice partition (1 active head, others dead — same failure mode that killed sn=24 in PR #34 with trailing-5 73.95). If sn=4 2-seed std > 2 val, mark as unstable; close. Expected: sn=4 buys a clean ~2 val improvement, sn=2 hits a wall.
- **Suggested student:** **frieren** (led original sn=16 win in prior round, knows the regime).

---

### Idea 3: nl-floor-and-mlp-depth — Push n_layers floor (nl=1) + compensate depth loss with deeper preprocess MLP
- **Hypothesis:** Depth axis monotonic 5→3→2 (single-seed). nl=1 alone may break (only 1 attention layer leaves no inter-token mixing across depth). Compensating with `MLP n_layers=2` in `preprocess` and `mlp2` head (currently both n_layers=0/explicit GELU) gives the model representational depth at near-zero compute cost. Predicted: nl=1 + deeper MLPs reaches val ≈ 48-50.
- **Why it could win:** The Transolver attention block is the expensive piece per-epoch; the preprocess MLP and `mlp2` head are essentially free in compute. If representational depth lives in the attention, nl=1 fails; if it lives in the FFN/MLPs, nl=1 + deeper MLPs wins. Free probe on a major axis. nl=2/sn=16 single-seed (50.72) is already at the depth floor; nl=1 is the next probe.
- **Concrete change:**
  - Wire `--n_layers`, plus `--preprocess_mlp_layers` (default 0 — current behavior), `--head_mlp_layers` (default 0 — applied to last-block `mlp2`).
  - Code edits: `MLP(n_input, n_hidden*2, n_hidden, n_layers=cfg.preprocess_mlp_layers, ...)` for `Transolver.preprocess`. Replace `mlp2 = nn.Sequential(Linear, GELU, Linear)` with `MLP(hidden_dim, hidden_dim, out_dim, n_layers=cfg.head_mlp_layers, ...)`.
  - Sweep:
    - GPU 0-1: nl=2 sn=8 anchor (verify single-seed PR #39 number), seeds {0,1}
    - GPU 2-3: nl=1 sn=8 baseline, preprocess_mlp_layers=0, seeds {0,1}
    - GPU 4-5: nl=1 sn=8 + preprocess_mlp_layers=2, seeds {0,1}
    - GPU 6-7: nl=1 sn=8 + preprocess_mlp_layers=2 + head_mlp_layers=2, seeds {0,1}
- **Recipe baseline to start from:** Idea 1 recipe + `--n_layers {1,2}` + new flags. wandb_group: `<student>/nl-floor-mlp-depth`.
- **Risk / falsification:** If nl=1 + deeper MLPs > 55 val, the attention depth IS load-bearing; close. If nl=2 single-seed (50.72) does not replicate at 2-seed (mean > 51.5), the depth floor was lucky.
- **Suggested student:** **fern** (closed prior n_hidden capacity scaling work cleanly; comfortable with MLP architecture diffs).

---

### Idea 4: n-hidden-shrink-floor — Shrink n_hidden {64, 96} on best recipe, find the width floor
- **Hypothesis:** Compute-reduction theme (now 5/5 wins) extends to width axis. n_hidden=128 default never shrunk (PR #41 unverified outcome). At nh=1 + sn=8 + nl=3, halving width to 64 saves ~3× compute and should buy ~5 epochs, reaching val ≈ 47. n_hidden=96 is the compromise probe.
- **Why it could win:** Width is the only compute-reduction axis untested. n_head=1 already uses dim_head=128 (full width per single head), making the network "wide-and-shallow" already. Shrinking n_hidden completes the orthogonal axis. Width × time tradeoff likely leans toward time at 32-epoch budget where every model is still descending at terminal.
- **Concrete change:**
  - Wire `--n_hidden int` (default 128).
  - Override `model_config["n_hidden"] = cfg.n_hidden`.
  - Sweep at nh=1, sn=8, nl=3 (combine with Idea 1 frontier):
    - GPU 0-1: n_hidden=128 anchor, seeds {0,1}
    - GPU 2-3: n_hidden=96, seeds {0,1}
    - GPU 4-5: n_hidden=64, seeds {0,1}
    - GPU 6-7: n_hidden=64, sn=4 (compute-reduction stacked), seeds {0,1}
  - Note: param count for SwiGLU FFN scales with n_hidden^2 — total param drop ~75% at n_hidden=64.
- **Recipe baseline to start from:** Idea 1 recipe + `--n_head 1 --slice_num 8 --n_hidden {128, 96, 64}`. wandb_group: `<student>/n-hidden-shrink-floor`.
- **Risk / falsification:** If n_hidden=64 2-seed mean > 53, capacity floor reached; close. If n_hidden=96 wins by < 1 val, it's noise; close. Expected: n_hidden=96 wins with ~+2 epochs in budget.
- **Suggested student:** **alphonse** (ran prior n_head and n_hidden-adjacent sweeps with strict multi-seed protocol).

---

### Idea 5: dim-head-shape-preserving — Decouple n_head and dim_head; sweep dim_head ∈ {32, 64, 128, 192} at n_head=1
- **Hypothesis:** When n_head=1 wins on the leaderboard, "single head" actually means "single attention path with head-dim = full hidden". Real architectural question is the dim_head axis. At n_head=1, dim_head defaults to n_hidden=128; sweeping dim_head independently of n_head explores whether 1 head with dim_head=192 (wider single head) beats 1 head with dim_head=128. PR #32 has nh=1 with dim_head=128; never tried wider.
- **Why it could win:** Param count `Linear(dim_head, slice_num)` scales with dim_head, so changing dim_head independently from n_head reveals what's really driving the n_head=1 win — slice projection capacity vs single-head bias. 1-head + dim_head=192 has 50% more slice-projection capacity than 1-head + dim_head=128 and may keep budget cost manageable.
- **Concrete change:**
  - Wire `--dim_head int` (default `n_hidden // n_head`).
  - In `PhysicsAttention.__init__`, take `dim_head` as an explicit arg (currently hard-derived from `dim // heads`). Adjust `inner_dim = dim_head * heads`. Output proj `to_out` already maps `inner_dim -> dim`, so it auto-handles the case where `inner_dim != dim`.
  - Sweep at nh=1, sn=8, nl=3:
    - GPU 0-1: dim_head=128 anchor (matches existing nh=1), seeds {0,1}
    - GPU 2-3: dim_head=64, seeds {0,1}
    - GPU 4-5: dim_head=192, seeds {0,1}
    - GPU 6-7: dim_head=64 + n_head=2 (compound, reverse direction), seeds {0,1}
- **Recipe baseline to start from:** Idea 1 + `--n_head 1 --dim_head {64,128,192}`. wandb_group: `<student>/dim-head-sweep`.
- **Risk / falsification:** If dim_head=192 2-seed mean > 51 (worse than dim_head=128), capacity is not the bottleneck. If dim_head=64 wins (slimmer single head), the n_head=1 finding was about slice-token regularization, not capacity.
- **Suggested student:** **askeladd** (fresh assignment; methodologically rigorous task suits a careful new student).

---

### Idea 6: chord-aligned-frame-features — Add chord-aligned coordinates and surface normals as input features
- **Hypothesis:** Foil orientation (AoA) is currently encoded only as a scalar at dims 14,18 of x. Adding chord-aligned `(x', z')` rotated by AoA and normal vectors at surface nodes (computed from `dsdf`/`saf`) gives the model explicit foil-relative geometry. Predicted: val drops ~3-5 on geom_camber_rc and re_rand splits, target val ≈ 46.
- **Why it could win:** The two OOD geometry splits (val_geom_camber_rc, val_geom_camber_cruise) are the largest contributors to the average val_avg/mae_surf_p. Explicit chord-aligned features inject the inductive bias that pressure depends on chord position not absolute (x, z). Surface normals make the surface a marked submanifold — currently encoded only by `is_surface` (dim 12) which is binary and uninformative about local geometry. PR #74 (RoPE/SE2) closed but never tried this simpler featurization.
- **Concrete change:**
  - In `train.py`, before normalization, add 4 derived features:
    - `chord_x = x[:,:,0]*cos(aoa) - x[:,:,1]*sin(aoa)` (rotated x by foil-1 AoA dim 14)
    - `chord_z = x[:,:,0]*sin(aoa) + x[:,:,1]*cos(aoa)`
    - `n_x, n_z`: surface normal direction. Computed offline via finite-difference of `saf`/`dsdf` along the foil; for non-surface nodes, use vector to nearest surface node from `dsdf[0]`. Implement either (a) on-the-fly in the dataloader (preferred — keeps `data/` read-only requires the additional features to be appended in `train.py` after batch load) or (b) as a precomputed cache on PVC.
  - Adjust `X_DIM` import and `model_config["fun_dim"]` to `X_DIM + 4 - 2`.
  - Update normalization stats: compute mean/std for the new 4 features in `train.py` (or use `(0,1)` normalization since rotated coords are bounded).
  - Sweep:
    - GPU 0-1: anchor (no extra features), seeds {0,1}
    - GPU 2-3: chord_x, chord_z only (orientation only), seeds {0,1}
    - GPU 4-5: chord + normals (full), seeds {0,1}
    - GPU 6-7: full + tandem-frame (rotated by foil-2 AoA dim 18 too), seeds {0,1}
- **Recipe baseline to start from:** Idea 1 + the feature additions. wandb_group: `<student>/chord-frame-features`.
- **Risk / falsification:** If chord-aligned features alone regress the normalized recipe (>50 val), the model already extracts orientation from learned features. Surface normal computation must respect padding mask — verify in `--debug`.
- **Suggested student:** **edward** (this requires careful feature-engineering execution + correct mask handling — exactly the kind of careful work a fresh assignment needs).

---

### Idea 7: per-block-film-conditioning — Inject (Re, AoA, foil-2 features) into every block via FiLM, not just at input
- **Hypothesis:** PR #7 (round 1) and follow-ups showed input-only FiLM hurts; per-block FiLM was tested briefly but only on a stale recipe (pre-Fourier, pre-SwiGLU, pre-sn=16, pre-nl=3). Fresh per-block FiLM on the FULL merged recipe should let conditioning reach deep layers where Re-dependent physics regimes matter most. Predicted: val drops ~2-3 on val_re_rand and val_geom_camber_*, target val ≈ 47.
- **Why it could win:** The model currently has 3 attention layers and only sees Re/AoA/NACA at preprocess input — those signals attenuate through the residual stack. Per-block FiLM (γ, β as functions of pooled global features) gives every block direct access to the regime indicator. NeRF, diffusion, and conditional norm work all support this pattern. PR #7's failure was attributed to "single-point injection underpowered" — exactly what per-block FiLM fixes.
- **Concrete change:**
  - Add `FiLMBlock` module: takes pooled-global `g = mean(x[:,:,[13,14,15,16,17,18,19,20,21,22,23]], dim=-2)` (Re, AoAs, NACA, gap, stagger), passes through 2-layer MLP → outputs `(γ_l, β_l)` for each layer l.
  - In `TransolverBlock.forward`, apply `fx = γ_l * fx + β_l` after `ln_1` (or after `ln_2`).
  - Wire `--film bool` (default false) to gate the new module.
  - Sweep at nh=4, sn=8, nl=3 (or compound with nh=1 if Idea 1 confirms):
    - GPU 0-1: anchor (no FiLM), seeds {0,1}
    - GPU 2-3: FiLM after ln_1, seeds {0,1}
    - GPU 4-5: FiLM after ln_2, seeds {0,1}
    - GPU 6-7: FiLM both, seeds {0,1}
- **Recipe baseline to start from:** Idea 1 recipe + `--film {false,after_ln1,after_ln2,both}`. wandb_group: `<student>/per-block-film`.
- **Risk / falsification:** If 2-seed mean > 52 with FiLM enabled, the residual block already captures conditioning sufficiently — close. If FiLM helps at nh=4 but not at nh=1, single-head trunk has different conditioning needs.
- **Suggested student:** **tanjiro** (fresh assignment; clean architectural diff with conservative gate).

---

### Idea 8: surface-finetune-phase — Two-phase training: full-objective for 30 epochs, then surface-only fine-tune for 8 epochs
- **Hypothesis:** Surface MAE is the only metric that ranks the leaderboard, but training treats surface and volume nodes uniformly (sw=1). A short fine-tune phase that flips to surface-only loss after the model has stabilized should drop val_avg/mae_surf_p by 2-4 with negligible epoch cost. Predicted: val ≈ 46-47.
- **Why it could win:** Heads on the full objective serve dual purpose (volume + surface). The terminal phase of training (where best val occurs at terminal in the merged recipe) is the right time to refocus. Standard CV and CFD-ML practice. Direct optimization of the ranking metric. Orthogonal to all merged components.
- **Concrete change:**
  - Wire `--finetune_epochs int` (default 0), `--finetune_surf_only bool` (default false).
  - In the training loop, when `epoch >= cfg.epochs - cfg.finetune_epochs`, set effective `loss = surf_loss` only (drop `vol_loss + sw*surf_loss`).
  - Optionally add `--finetune_lr float` (default 0.1× cfg.lr) to mimic the cosine tail.
  - Sweep at full recipe nl=3 sn=8 nh=4:
    - GPU 0-1: anchor (no finetune), seeds {0,1}
    - GPU 2-3: finetune_epochs=8, finetune_lr=cosine-tail-default, seeds {0,1}
    - GPU 4-5: finetune_epochs=4, seeds {0,1}
    - GPU 6-7: finetune_epochs=12, finetune_lr=1e-5, seeds {0,1}
  - Note: finetune phase adds N epochs ON TOP of `--epochs 50` regular schedule. Adjust scheduler T_max=cfg.epochs (not cfg.epochs+finetune_epochs) so cosine completes before fine-tune, then fine-tune uses constant low LR.
- **Recipe baseline to start from:** Idea 1 recipe + `--finetune_epochs {0,4,8,12}`. wandb_group: `<student>/surface-finetune`.
- **Risk / falsification:** If finetune_epochs=8 destabilizes (val regresses during finetune), the trunk loses its volume-derived prior; close. If both seeds win by <1 val, noise; close.
- **Suggested student:** **thorfinn** (fresh assignment; clean two-phase training is a tractable architectural diff).

---

### Idea 9: signed-distance-and-wall-distance-features — Add wall-distance / distance-to-nearest-surface-node features
- **Hypothesis:** The 8-dim `dsdf` (dims 4-11) is a multi-scale shape descriptor but does NOT directly include signed distance to nearest surface node. Adding `wall_distance = min over surface nodes of euclidean distance` and `signed_wall_distance = wall_distance * sign(dsdf[0])` as 2 extra input features should boost the boundary-layer encoding directly. Predicted: val ≈ 46-47 with strongest gains on cruise (largest meshes, most volume nodes far from surface).
- **Why it could win:** Surface pressure depends critically on the boundary-layer separation point, which depends on wall distance. The current `dsdf` features are SDF-derived but not the SDF itself. Adding the explicit wall-distance signal is a tiny cheap injection that direct competitors (GNO, FNO-style models) all include. PR #103 (Sobolev arc-length grad loss) and PR #71 (signed distance) status uncertain — explicit signed distance was never wired and verified.
- **Concrete change:**
  - In `train.py`, after batch load, compute per-batch:
    - For each sample: `surface_pos = x[is_surface, :2]` and `wall_distance[i] = min over j of ||x[i,:2] - surface_pos[j]||`.
    - Vectorize via `torch.cdist` and reduce min along surface dim. Mask out padding.
    - Append wall_distance and signed wall_distance as 2 additional features.
  - Update `model_config["fun_dim"] = X_DIM + 2 - 2`.
  - Compute normalization stats for these 2 features (mean ~0.3 chord, std varies by domain).
  - Sweep at full recipe nl=3 sn=8 nh=4:
    - GPU 0-1: anchor, seeds {0,1}
    - GPU 2-3: + wall_distance only, seeds {0,1}
    - GPU 4-5: + signed wall_distance, seeds {0,1}
    - GPU 6-7: + both, seeds {0,1}
- **Recipe baseline to start from:** Idea 1 recipe + feature additions. wandb_group: `<student>/wall-distance-features`.
- **Risk / falsification:** If wall_distance computation is too slow per batch (>10% epoch time hit), epoch budget shrinks. Time it in `--debug`. If features regress (>52 val), the dsdf already encodes this; close.
- **Suggested student:** **frieren** (alternative if not assigned to Idea 2; comfortable with on-the-fly feature derivation).

---

### Idea 10: throughput-compile-flashattn — torch.compile + FlashAttention v2 for raw throughput unlock
- **Hypothesis:** PR #102 closed but reasons unclear from prior log. With sn=8 and nl=3, the model is small (<1M params) but the dataloader-padding-bound regime means torch.compile + FlashAttn v2 may unlock 1.3-1.5× throughput → 12-15 more epochs in budget → val drops to ≈45. Pure throughput unlock, orthogonal to all knobs.
- **Why it could win:** Compute-reduction theme is the single dominant win category; raw throughput is the OTHER axis (nezuko's PR #39 made the merged stack budget-bound). All merged configs hit best val at terminal epoch — every extra epoch buys real progress. FA v2 explicit + torch.compile is the lowest-effort throughput unlock.
- **Concrete change:**
  - Wire `--compile bool` (default false), `--flash bool` (default false).
  - At model construction: `if cfg.compile: model = torch.compile(model, mode="reduce-overhead")`.
  - In `PhysicsAttention.forward`: when `cfg.flash`, use `torch.nn.functional.scaled_dot_product_attention` (already used) but with `with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):` context.
  - Padding mask: SDPA already handles `attn_mask`; ensure padded slice tokens are masked. (Currently the `slice_token` branch may not mask properly — check carefully).
  - Sweep at full recipe nl=3 sn=8 nh=4:
    - GPU 0-1: anchor (no compile, no flash), seeds {0,1}
    - GPU 2-3: + compile only, seeds {0,1}
    - GPU 4-5: + flash only, seeds {0,1}
    - GPU 6-7: + both, seeds {0,1}
  - Check: report epochs/min, peak_GB, best_val_avg, mean train/loss for the first 5 epochs (compile recompile triggers can dominate first 1-2 epochs).
- **Recipe baseline to start from:** Idea 1 recipe + `--compile --flash`. wandb_group: `<student>/throughput-compile-flash`.
- **Risk / falsification:** torch.compile may need warmup epochs that eat into the 30-min budget; if compile run takes >15s on first epoch, document and proceed. FA may not work with current SDPA usage on padded inputs. If compile alone doesn't move epochs/min > 1.1×, close. PR #102 closed for a reason — student should investigate that PR's failure mode in the prior round and avoid it.
- **Suggested student:** **askeladd** (alternative if not Idea 5; throughput diagnosis suits methodically careful student) OR **edward** (alternative if not Idea 6).

---

## Summary table

| # | Slug | Student (suggested) | Recipe diff vs proven recipe | Predicted Δ val |
|---|------|---------------------|------------------------------|------------------|
| 1 | recipe-port-frontier-anchor | nezuko | Full port + nh=1 sn={8,16} nl={2,3} | val ≈ 44-46, test ≈ 40 |
| 2 | sn-floor-mapping-extended | frieren | sn ∈ {2, 4} probes | val ≈ 45-47 |
| 3 | nl-floor-and-mlp-depth | fern | nl=1 + deeper preprocess MLP | val ≈ 48-50 |
| 4 | n-hidden-shrink-floor | alphonse | n_hidden ∈ {64, 96} | val ≈ 47 |
| 5 | dim-head-shape-preserving | askeladd | dim_head ∈ {64, 128, 192} at nh=1 | val ≈ 47-49 |
| 6 | chord-aligned-frame-features | edward | + chord-relative coords + normals | val ≈ 46 |
| 7 | per-block-film-conditioning | tanjiro | + per-block FiLM | val ≈ 47 |
| 8 | surface-finetune-phase | thorfinn | + 4-12 epoch surface-only finetune | val ≈ 46-47 |
| 9 | signed-distance-and-wall-distance-features | frieren (alt) | + wall-distance features | val ≈ 46-47 |
| 10 | throughput-compile-flashattn | askeladd (alt) | + torch.compile + FA v2 | val ≈ 45 |

**Coverage and orthogonality:**
- Ideas 1-5 are compute/architecture reduction extensions on the dominant winning axis.
- Ideas 6-9 are feature, conditioning, loss, finetune diversifications (orthogonal to compute-reduction).
- Idea 10 is the raw throughput axis (orthogonal to everything).
- No two ideas overlap on the same knob.

**Anchor:** Idea 1 is the highest-confidence shot at the leaderboard top-1 in round 1 (recipe port + frontier closure). It also produces the round's BASELINE.md update needed for all subsequent merge decisions.
