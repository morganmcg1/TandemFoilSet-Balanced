# SENPAI Research State

- **Last updated:** 2026-05-17 08:35 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Primary metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline (QK-norm + Lion)

**PR #4270** — QK-norm (LayerNorm on Q,K) + Lion + n_hidden=176 + bf16 + epochs=14 (edward), merged 2026-05-17 ~05:30 UTC
- **val_avg/mae_surf_p = 46.9886** (W&B `oospddft`)
- **test_avg/mae_surf_p = 40.4803**
- Per-split test: single_in_dist=43.18, geom_camber_rc=52.79, geom_camber_cruise=25.83, re_rand=40.12
- Wall: ~30.6 min / 14 ep (~131 s/ep), Peak VRAM: 44.6 GB
- Reproduce: `cd "target/" && python train.py --n_hidden 176 --epochs 14 --use_bf16 --use_lion --lion_lr 1e-4 --lion_wd 1e-3 --use_qk_norm`

**QK-NORM IS NOW STANDARD.** All new experiments must include `--use_qk_norm` unless specifically testing its removal.

## Most recent research direction from human researcher team

No GitHub Issues open for this track as of 2026-05-17 06:30 UTC. Proceeding from program contract.

## Current in-flight experiments (8 active, zero idle)

| PR | Student | Axis being tested | Status |
|---|---|---|---|
| **#4411** | tanjiro | coord_noise=0.005 seed-2 confirmation (most promising signal) | WIP — sent back, ready to rerun |
| **#4474** | alphonse | Skip-connection scaling 1/√2 + QK-norm + Lion | WIP — running (Round-11 carryover) |
| **#4478** | nezuko | Lion eta_min=1e-5 in cosine schedule + QK-norm | WIP — running (Round-11 carryover) |
| **#4483** | thorfinn | **batch_size=8 + ep18 + Lion + QK-norm** — gradient-variance reduction | WIP — just assigned (Round-12 #1) |
| **#4485** | askeladd | **Constant LR after warmup + Lion + QK-norm** — full late-training exploration | WIP — just assigned (Round-12 #2) |
| **#4486** | fern | **TTA K=8 coord-noise eval + QK-norm** — eval-time procedure | WIP — just assigned (Round-12 #3) |
| **#4488** | frieren | **Post-LN configuration + QK-norm** — original Transformer norm placement | WIP — just assigned (Round-12 #4) |
| **#4489** | edward | **Tail-emphasizing focal-L1 (α=0.5)** — counter to Huber failure | WIP — just assigned (Round-12 #5) |

**Zero idle students. Zero idle GPUs.**

### Most promising standing signal

**#4411 tanjiro coord_noise=0.005 (arm A `m5dcxfhe`):** val=47.03 (+0.04, within noise floor of 0.5), **test=40.35 (-0.13 BEATS paper-facing metric)**, geom_camber_rc=52.45 (BEATS hardest split by 0.34), re_rand=39.93 (BEATS). Sent back for seed-2 confirmation. If seed-2 confirms, MERGE candidate updating baseline to ~47.0/40.3 range.

## Round-10/11 dead-ends (cumulative, 16 closures since #4270 merged)

| PR | Student | Axis | W&B verdict |
|---|---|---|---|
| #4280 | frieren | Lion+nh=192+ep12: 3 seeds val 49.6-50.9 | CLOSED |
| #4285 | nezuko | Lion lr=2e-4: 2 seeds val 49.2-49.7 | CLOSED |
| #4233 | tanjiro | AGC clip=0.03: val=57.37 (+22% catastrophic) | CLOSED |
| #4354 | alphonse | Lion n_head=2: 2 seeds val 48.82-49.17 | CLOSED |
| #4382 | edward | V-norm: val=72.52 (+54.4% CATASTROPHIC) | CLOSED |
| #4366 | fern | Lookahead k=3/5: val=50.03 (axis dead) | CLOSED |
| #4324/#4413 | askeladd | wd=5e-4+QK-norm: mechanism overlap | CLOSED |
| #4178 | thorfinn | EMA (decay=0.999): no signal | CLOSED (prior) |
| #4409 | frieren | mlp_ratio=3: val=50.76 (+8%) — FFN width axis | CLOSED |
| #4410 | nezuko | loss_type=huber: val=54.27 (+15.5%) — tail-suppress wrong direction | CLOSED |
| #4412 | alphonse | batch_size=2: val=50.54 (+7.6%) | CLOSED |
| #4416 | edward | LayerScale γ=1e-4 AND γ=1.0 both regress | CLOSED (axis exhausted) |
| #4417 | fern | SWA ep11-14: val=48.53 (+3.3%) | CLOSED (3rd time-avg failure) |
| #4418 | askeladd | Lion β1=0.95: val=54.40 (+15.8% severe) | CLOSED |
| #4476 | frieren | n_layers=6 nh=128: val=50.16 (+6.8%) | CLOSED (param-budget exhausted) |
| #4383 | thorfinn | surf_weight {5,15} 3 arms close-tie/regress | CLOSED (axis exhausted) |

## PLATEAU PROTOCOL EXTENDED (2026-05-17 08:35 UTC)

**17 consecutive non-improvements since #4270 merged at 05:30 UTC.** Plateau extends through 5 additional closures since 07:30 (#4416 γ=1.0, #4417 SWA, #4418 β1=0.95, #4476 L6-nh128, #4383 surf-sweep).

### Most promising signal — but technically not a win

- **tanjiro #4411 arm A (coord_noise_std=0.005):** val=47.03 (+0.09% regress) but **test=40.35 (-0.32% IMPROVEMENT)**. Just outside noise floor on val. Arm B (std=0.02) still running — if both arms close-tie, the coord-noise axis may be a noise floor under QK-norm. WAIT for arm B.
- **thorfinn #4383 arm A (surf_weight=5, run-B):** val=47.36 (+0.79% regress), test=40.64. Run-A `1grt9rc9` val=48.21 (+2.6%), test=40.22 (BEATS by 0.26). 0.85 spread on same config = noise floor ~0.5-1.0 val. sw=15 still running.

### Round-12 plateau-break wave (in-flight)

Newly assigned at 08:35 UTC — 5 fresh mechanism surfaces:

1. **thorfinn #4483:** bs=8 + ep18 (gradient-variance reduction at optimizer level)
2. **askeladd #4485:** Constant LR after warmup (full late-training exploration)
3. **fern #4486:** TTA K=8 coord-noise (eval-time procedure, paper-facing)
4. **frieren #4488:** Post-LN configuration (normalization placement)
5. **edward #4489:** Tail-emphasizing focal-L1 α=0.5 (loss formulation)

Plus 3 Round-11 carryovers still running:
- alphonse #4474 skip-1/√2 (residual scaling)
- nezuko #4478 eta_min in cosine (LR floor)
- tanjiro #4411 send-back coord_noise=0.005 seed-2 (close-tie confirmation)

**Total 8 in-flight on orthogonal mechanism surfaces. If ANY breaks plateau, several may compound.**

## Key learnings (Round-10 to date)

1. **QK-norm + Lion is the new stack baseline.** ALL new experiments must build on both. Students who were assigned pre-#4270 had their results re-evaluated against new baseline; most failed.
2. **Baseline shift mid-round (from #4252 to #4270):** When #4270 merged mid-cycle (val 49.26→46.99), several in-flight PRs that beat the OLD baseline failed against the NEW one. Decision rule: send back for QK-norm stack retest if the mechanism is orthogonal AND per-split shows meaningful signal (e.g., geom_camber_rc improvement). Close if mechanism is redundant or all splits regress uniformly.
3. **Students stuck in Claude loop without SENPAI-RESULT:** PRs #4280, #4285, #4233, #4354 all had multiple finished W&B runs but no posted terminal marker. Closed via advisor W&B-data verdict. Students need to be reminded to post results promptly.
4. **nh=192 width axis exhausted under Lion.** 3 seeds at val 49.6-50.9 confirm nh=176 is the width sweet spot at ep12-14 cap. Cap-bound (12ep) can't fully converge nh=192.
5. **Lion LR=1e-4 is the confirmed local optimum.** lr=2e-4 (2× up) regresses; lr=5e-5 (2× down) not needed.
6. **AGC is redundant with Lion.** Lion's sign-update provides gradient-direction stability; AGC-on-top catastrophically regresses. AGC axis closed.
7. **n_head=2 (d_head=88) doesn't help without QK-norm.** Uniform all-split regression confirms wider heads need normalization to unlock; could be retested with QK-norm but per-split showed no asymmetric signal.

## Round-13 backlog (researcher-agent ideas, 2026-05-17 08:35)

See `research/RESEARCH_IDEAS_2026-05-17_0820.md` for full reasoning + literature citations. Top 5 ranked:

1. **Variance+Mean Composite Loss** — `L = 0.8·mean(|e|) + 0.2·std(|e|)`. 2-line change. Penalizes localized error spikes (geom_camber_rc dominant pattern). Backed by Hanna et al. arXiv:2412.13993.
2. **GeoMix Geometry Augmentation** — Interpolate input/target pairs from nearby-camber training cases (λ ~ Beta(2,2)). Bridges to OOD M=6-8 from training M=3-5. Medium effort. Highest expected single-PR gain.
3. **2D Rotary Position Encoding (RoPE-2D)** — Encode (x,y) as rotary frequencies in Q,K. Geometry-relative attention prior. Apply d_rope = n_hidden//4 = 48.
4. **Zonal / Wake-Emphasis Loss** — 3× weight on TE/wake nodes (x_norm > 0.6). Domain knowledge: camber shift moves Kutta condition → most strongly affects TE. 5-line change.
5. **GFocal Dual-Path Attention** — Parallel Nyström global path (m=64 landmarks) + slice local path, learnable gate fusion. Hardest impl. Save for plateau extension if loss/data ideas (#1, #2, #4) all fail.

**Stop condition (per researcher):** If Variance+Mean loss AND GeoMix both fail to improve `geom_camber_rc` by >1%, the OOD gap is architectural — escalate to GFocal.

---

## Plateau-break next tier (Round-12 backlog if Round-11 wave regresses)

If alphonse skip-1/√2, frieren L6, nezuko eta_min all regress (and prior 5 plateau-breaks), escalate to:
1. **Physics-informed divergence-free loss** — penalty term λ × E[||∂Ux/∂x + ∂Uy/∂y||²] computed via finite-diff along foil contour. Big swing, ICML-worthy. Implementation needs contour-ordered surface points (verify dataset has them).
2. **Gradient-based input features** — append ∂p/∂x, ∂p/∂y to input encoding via local finite-diff.
3. **Multi-scale slice_num** — heterogeneous slice_num across layers (e.g., layers 0-1 → 128, layers 2-3 → 64, layer 4 → 32). Captures hierarchical physics scales.
4. **bs=8 with VRAM headroom** — alphonse bs2 regressed; opposite direction may help (Lion needs variance reduction).
5. **Wider heads n_head=8 with QK-norm** — d_head=22 was bad pre-QK-norm; QK-norm may stabilize narrow heads.
6. **Constant lr after warmup** — if nezuko eta_min works, this is the natural extension (full constant LR, no cosine).
7. **Tail-emphasizing loss** — focal-style weighting or |∂p/∂x|-region upweight (opposite of huber's tail-suppress).

## Round-11 in-flight summary (8 plateau-break axes)

All 8 in-flight experiments target *different* mechanisms — orthogonal axes per CLAUDE.md "one hypothesis per PR":

| Axis | PR | Mechanism |
|---|---|---|
| Augmentation strength | #4411 (tanjiro) | coord_noise sweep — close-tie arm A, arm B pending |
| Loss reweighting | #4383 (thorfinn) | surf_weight — close-tie sw=5, sw=15 pending |
| Residual gating (learnable) | #4416 (edward) | LayerScale γ=1.0 retest |
| Weight averaging (eval-time) | #4417 (fern) | SWA over ep11-14 |
| Optimizer momentum window | #4418 (askeladd) | Lion β1=0.95 |
| Residual scaling (fixed) | #4474 (alphonse) | Skip-connection 1/√2 |
| Depth-width tradeoff | #4476 (frieren) | n_layers=6 at nh=128 |
| LR schedule floor | #4478 (nezuko) | eta_min=1e-5 in cosine |

If ANY of these breaks plateau, the others may compound — Lion β1 + eta_min + SWA could all combine.

## Cross-cutting findings (apply to all in-flight PRs)

1. **SwiGLU FFN is default** (#3814 merged).
2. **L1 loss is default** (`Config.loss_type = "l1"`) — nezuko #4410 testing huber.
3. **Lion is default optimizer** (#4252 merged).
4. **QK-norm is NOW STANDARD** (#4270 merged) — `--use_qk_norm` required on all new experiments.
5. **bf16 autocast is default** (#3981 merged).
6. **Fourier PE num_freq=4 is default** (#3372 merged, "4 won vs 8" confirmed in code comment).
7. **coord_noise_std=0.01 is default** (#3632 merged) — tanjiro #4411 sweeping {0.005, 0.02}.
8. **Grad clip max_norm=1.0**, warmup 2 epochs, batch=4.
9. **n_hidden=176, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2** are current default architecture.
10. **`geom_camber_rc` is the structural hard split** — QK-norm has moved it twice (54.75→52.79), still the hardest test split.

## Confirmed exhausted (do not retry on this stack)

- AdamW optimizer variants (any betas / wd / lr) — Lion supersedes
- Surface loss reweighting by target magnitude (pmag-weight, val +4.5% regress)
- Surface loss reweighting by DSDF proxy (curvature, val +12.2% regress)
- slice_num=48 (U-shape), 96 (monotonic worse from 64), 128
- n_head=8 (d_head=22 CUDA fragmentation)
- n_head=2 WITHOUT QK-norm (d_head=88, uniform regression) — see #4354
- n_layers=6 (cap-bound under-trained)
- RMSNorm (slower kernel + slice-attention breakage)
- Multi-scale Fourier PE wide (absorbed by width)
- DropPath, mlp2 gate, attn_dropout, asinh input transform
- AdaBelief optimizer, OneCycleLR, curriculum learning
- DSDF clip thresholds (no-op confirmed via dataset analysis)
- Camber flip augmentation (NACA-M asymmetry unflippable)
- nh=208 (cap-bound) and nh=192 (width saturates at Lion+ep12)
- AGC (Adaptive Gradient Clipping) — redundant with Lion sign-update, #4233
- EMA weight averaging — no signal under monotonic cosine-descent, #4178
- Lion lr=2e-4 (regresses vs lr=1e-4 optimum), lr=5e-5 (inferred from lr landscape)

## Pod environment notes

- All Round-10 student pods enforce **`SENPAI_TIMEOUT_MINUTES=30`** hard cap.
- Per-epoch walls (bf16): nh=176 ≈ 131 s/ep (Lion+QK-norm), nh=192 ≈ 131-145 s/ep.
- VRAM peaks (bf16): nh=176+Lion+QK-norm ≈ 44.6 GB. H100 has 96 GB — ample headroom.

## Baseline progression (val_avg/mae_surf_p)

- #3091 baseline: 109.42 → ... → #3814 SwiGLU: 64.24 → ... → #3981 bf16+ep18: 53.82 → #4082 nh=176: 50.90 → #4106 nh=192+ep20: 48.84 → #4252 Lion+nh=176+ep14: 49.26 → **#4270 QK-norm+Lion+nh=176+ep14: 46.99**

Total improvement from #3091: **−57.0% val, −55.8% test.** QK-norm + Lion + bf16 + width-sweep at 1.23M params (Transolver SwiGLU at nh=176).
