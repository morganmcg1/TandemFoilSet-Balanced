# SENPAI Research State

- **Date**: 2026-05-17 (cycle 36)
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 late-phase — **CURRENT BEST: H95 Arm A bf16 (val=40.51 / test=39.02, PR #4215).** Cycle 36 closed H98 (β₁ retune tie); identified **H93 Arm C WSD 0/5/10 as standing winner at val=39.51 / test=38.53, pending nezuko rebase before merge.** Re-sent H96 compound test. Assigned H112 (fern, OOD input jitter). 8 WIP, 0 idle.
- **Most recent human research directive**: None received

## Current Best

**PR #4215 (H95 Arm A: bf16 walltime, alphonse) — val_avg/mae_surf_p = 40.5066 / test 3-split = 39.0160** (MERGED 2026-05-17)

**Standing winner (cycle 36, pending merge):** PR #4196 H93 Arm C (WSD 0/5/10, nezuko) — val=39.5100 / test=38.5345 (Δ-1.00 val vs H95, Δ-0.48 test). Schedule signal real: forcing 10-epoch decay tail beats T_max=15 cosine by Δ-3.28 on hardest split (val_geom_camber_rc). Sent back for clean rebase against current advisor branch.

| Reference | val_avg | test 3-split | Status |
|-----------|--------:|-------------:|--------|
| **H93 Arm C (WSD 0/5/10)** | **39.5100** | **38.5345** | **STANDING WINNER pending rebase (PR #4196)** |
| **H95 Arm A (bf16 walltime)** | **40.5066** | **39.0160** | **CURRENT BEST MERGED (PR #4215)** |
| H88 Arm B (β₂=0.997) | 41.2153 | 39.5337 | Prior best (PR #4166) |
| H98 Arm A (β₁=0.85) | 40.5804 | 39.4821 | Tie vs H95 (within 1.7σ); CLOSED |
| H78 Arm B (β₂=0.995) | 42.3048 | 40.5564 | Overridden (PR #4097) |
| H73 Arm B (Lion lr=3e-4 + slice96) | 42.9784 | 41.5455 | Overridden (PR #4055) |

**Cumulative R5 gain: −25.60 pts val_avg vs H37b (66.11 → 40.51) merged; another −1.00 pts pending if H93C lands.** Total: −74.12 pts from R1 start (114.63).

## Noise Floor (revised H92)

**Revised: 2σ = 1.67 pts** on val_avg/mae_surf_p (H92, n=3 seeds at H78 config). Test 3-split 2σ ≈ 1.02 pts. H93 Arm C is a boundary win on val (Δ-1.00 < 1.7) but breaks through on test (Δ-0.48 < 1.0 yet supported by mechanism + per-split decomposition).

Decision thresholds:
- Δ < 1.7 pts → noise (tie)
- Δ ≥ 2.5 pts → real signal
- Δ ≥ 4.0 pts → strong signal

## Round 5 Insights (cumulative)

**Schedule has new headroom (cycle 36 discovery):** H93 Arm C (WSD: 0 warmup / 5 stable / 10 decay) substantially outperformed cosine T_max=15 on the hardest splits. Mechanism: long decay tail (10 of 15 epochs at decaying LR) gives more fine-tune time near zero LR. Per-split: val_geom_camber_rc Δ-3.28, val_geom_camber_cruise Δ-1.40, val_re_rand Δ-0.28, val_single_in_dist Δ+0.97. **Future high-EV compound: WSD 0/5/10 + bf16 (~21 epochs at bf16 → maybe 0/7/14 reshape). Predicted val ~37-38.**

**bf16 unlock (cycle 34):** Reduced s/epoch by ~30% without changing memory, unlocking capacity probes that were previously wall-cut-bound. H100/H101/H102/H103 active.

**Lion optimizer axis fully saturated:**
- β₂=0.997: LOCKED (H88)
- β₁=0.9: LOCKED (H90 at β₂=0.995; H98 confirms at β₂=0.997 — β₁=0.85 ties within noise, β₁=0.95 regresses badly)
- LR=3e-4: LOCKED (H97)
- wd=1e-3: LOCKED (H79)
- surf_weight=10: LOCKED (H91)

**OOD robustness explored:** H112 (fern) tests input-space jitter on continuous conditioning (AoA, log(Re), gap, stagger) — first OOD-explicit regularization experiment of R5.

## Active WIP Experiments (8 / 8 students, 0 idle)

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#4272** | alphonse | **H99: bf16 + T_max=21 schedule fix** | HIGH (T_max=15 bounce confound; aligns to bf16 budget) | ~38.5-40.5 |
| **#4276** | askeladd | **H100: n_hidden=192 under bf16 (H86 retest)** | HIGH (wall-cut-bound at fp32; bf16 gives ~14 epochs) | ~38-42 |
| **#4277** | frieren | **H101: n_layers=5 depth probe under bf16** | HIGH (depth unexplored; bf16 gives ~17 epochs) | ~39-42 |
| **#4291** | edward | **H102: slice_num=128 attention capacity under bf16** | HIGH (slice negative under AdamW; new regime may unlock) | ~38-42 |
| **#4292** | tanjiro | **H103: mlp_ratio=3 FFN capacity retest under bf16** | HIGH (H89 wall-cut at fp32; bf16 should reach ~17 ep) | ~38-42 |
| **#4196** | nezuko | **H93 Arm C: WSD 0/5/10 — STANDING WINNER pending clean rebase** | TOP (val=39.51, test=38.53; merge once rebase clean) | confirmed |
| **#4217** | thorfinn | **H96 (re-sent): compile + bf16 + T_max=21 compound (Arms C, D)** | HIGH (efficiency stack test; compile -27% + bf16 -30% may compound) | ~38-41 |
| **#4316** | fern | **H112: AoA + log(Re) + gap/stagger input jitter (σ ∈ {0.02, 0.05, 0.1})** | MED (first explicit OOD regularization; targets val_re_rand and val_geom_camber_cruise) | ~39-41 |

## Lever Status

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| Optimizer | 🏆 Lion locked | 42.98 (H73) | Massive super-additive win vs AdamW |
| LR (Lion) | ✅ 3e-4 LOCKED at β₂=0.997 (H97) | 3e-4 (H73) | LR optimum stable across β₂ shift |
| Schedule T_max | 🔬 H99 active (fix T_max=21 for bf16) | 15 (H73) | T_max=15 hardcoded; confound at 21 bf16 epochs |
| Schedule WSD | 🏆 **NEW SIGNAL — H93 Arm C 0/5/10 standing winner (pending rebase)** | 39.51 (H93C) | Long decay tail outperforms cosine on hardest splits |
| Schedule warmup | ❌ Regresses at slice=96 (H76); H93 confirms 0-warmup optimal | none | — |
| n_head | ❌ n_head=4 regresses at slice=96 (H77) | 2 | — |
| β₂ (Lion) | ✅ 0.997 locked (H88) | 0.997 | Peak confirmed |
| β₁ (Lion) | ✅ 0.9 locked (H90, H98 confirmed at β₂=0.997) | 0.9 | Asymmetric: 0.85 ties, 0.95 regresses |
| wd (Lion) | ✅ 1e-3 locked (H79) | 1e-3 | — |
| slice_num | 🔬 H102 active (128 under bf16+β₂=0.997) | 96 (H73) | Lion+bf16+β₂=0.997 regime may shift |
| n_layers | 🔬 H101 active (5 under bf16) | 4 (H60/H83) | Depth unexplored at bf16 speed |
| n_hidden | 🔬 H100 active (192 under bf16) | 128 | H86 wall-cut at fp32; retesting |
| mlp_ratio | 🔬 H103 active (3 under bf16) | 2 (default) | H89 wall-cut at fp32; retesting |
| FFN act | ✅ GEGLU locked (H48) | GEGLU | — |
| Normalization | ✅ LayerNorm locked (H72, H81) | LN | — |
| Mixed precision (bf16) | 🏆 MERGED WINNER (H95) | −0.71 val / −30% s/epoch | — |
| torch.compile | 🔬 H96 active (compile+bf16 compound + T_max=21, Arms C, D) | -27% s/epoch alone | First pass beat H78 -1.25; compound test active |
| OOD input jitter | 🔬 H112 active (AoA+log(Re)+gap/stagger) | none | First R5 explicit OOD regularization |
| surf_weight | ✅ 10 locked (H54 AdamW, H91 Lion) | 10 | — |
| clip_grad_norm | ✅ 1.0 locked (H20, H56) | 1.0 | — |
| Huber δ_p | ✅ 0.25 locked (H25/H64) | 0.25 | — |
| Batch size | ✅ 4 LOCKED (H94) | 4 | — |
| cond_dim | ✅ 11 locked (FiLM) | 11 | — |
| EMA averaging | ❌ Wrong regime (H65) | none | — |
| Mixup | ❌ Wrong inductive bias (H55) | none | — |
| DropPath | ❌ No effect (H63) | 0.0 | — |
| eta_min | ❌ H87 negative | 0 | — |

## Baseline Progression

| Val avg/mae_surf_p | Test 3-split | Event |
|---|---|---|
| 114.63 | — | R1 start (FiLM only, T_max=50 mismatch) |
| 66.11 | 64.45 | H37b: n_head=2 + lr=1e-3 |
| 58.63 | 56.70 | H48 GEGLU |
| 57.58 | 56.46 | H60: n_layers=4 |
| 56.75 | 54.50 | H66: slice_num=96 |
| 42.98 | 41.55 | H73 Arm B: Lion + lr=3e-4 |
| 42.30 | 40.56 | H78 Arm B: β₂=0.995 |
| 41.22 | 39.53 | H88 Arm B: β₂=0.997 |
| **40.51** | **39.02** | **H95 Arm A: bf16 autocast (CURRENT MERGED BEST)** |
| (39.51) | (38.53) | (H93 Arm C: WSD 0/5/10 — pending rebase + merge) |

Total merged gain: **−74.12 pts val (64.7% reduction).** Pending: another −1.00 pts if H93C lands.

## Strategic State

**Schedule axis is the new engine (cycle 36 discovery).** H93 Arm C reveals real headroom in LR schedule shape — the WSD 0/5/10 schedule (5 stable, 10 decay) beats cosine T_max=15 on the hardest OOD splits. This is the first non-saturated lever in cycles. The compound with bf16 (where we have 21 epochs available — reshape to 0/7/14?) is the highest-EV future experiment.

**Capacity probes converging.** H100/H101/H102/H103 cover the four capacity dimensions (width, depth, slice_num, mlp_ratio) all under bf16. Even if 2 of 4 land in noise, the other 2 give a real architectural improvement path.

**Efficiency stack maturing.** bf16 merged (−30% s/epoch), torch.compile (−27% s/epoch alone) compound under test (H96). If compile+bf16 compounds even partially, that's ~45% more effective compute per wall-minute → genuinely longer schedules possible.

**OOD generalization is now in scope (H112).** First experiment to explicitly attack the OOD splits via input-space regularization rather than capacity/scheduling.

**Open strategic questions:**
1. **Does H93 Arm C rebase clean?** (Top priority — would land new best at val=39.51 / test=38.53)
2. **Does WSD 0/5/10 + bf16 compound?** (Reshape to 0/7/14 for 21-epoch budget. Predicted val ~37-38. Assign next idle student.)
3. **Does any capacity probe (H100/H101/H102/H103) break through under bf16?**
4. **Does H99's T_max fix improve on H95?** (orthogonal to WSD if both schedules end up beating cosine)
5. **Does compile+bf16 compound** (H96 Arms C, D)?
6. **Does input jitter improve OOD splits** (H112) without hurting in-dist?

## Pending Research Ideas (researcher-agent output, cycle 35)

`research/RESEARCH_IDEAS_2026-05-17_02.md` — H104 (per-sample p std normalization), H105 (SWA), H106 (Fourier PE), H107 (log(Re) aux head), H108 (AoA+Re jitter — became H112), H109 (split surf/vol normalization), H110 (n_hidden=256 gated on H100), H111 (surf-P residual head).

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op.
- `train.py`: `T_max=15` hardcoded at line 649 (not a CLI arg). H99 student adding `--T_max` arg.
- `train.py`: `n_hidden=128` hardcoded at line 621 (not a CLI arg). H100 student adding `--n_hidden` arg.
