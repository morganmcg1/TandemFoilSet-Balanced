# SENPAI Research State

- **Date:** 2026-05-17 11:50
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best — MAJOR JUMP THIS TURN

- **PR #4424 (tanjiro, MERGED THIS TURN):** BF16 + LayerScale γ=0.01 + **n_freqs=8** + batch_size=2 + Huber δ=0.10 + **lr=9e-4** + wd=0.0001 (default) + T_max=20 + clip=0.25 + slice_num=32
- **val_avg/mae_surf_p: 53.595** | **test_avg/mae_surf_p: 46.395**
- Per-split test surf_p: single=49.322, rc=60.151, cruise=30.349, re_rand=45.757 — WINS ALL 4 SPLITS vs prior best
- best_epoch=20/20 (timeout-bound at cosine floor — but cosine min reached precisely at timeout, no information lost)
- **Cumulative improvement: -58.3% val from round-5 start (~128.69)**

**Key finding from #4424**: Monotone lr improvement on n=8 stack: 7e-4 (54.959) → 8e-4 (54.558) → 9e-4 (53.595). Mechanism: at cosine LR floor, clip_frac descends with peak LR (0.94 at lr=9e-4 — most slack). n=8's shallower Fourier landscape tolerates wider lr window than n=10 (where lr ceiling was 8e-4, #4513).

**Inverse coupling**: lr ceiling vs n_freqs. Coarser Fourier → higher lr ceiling. Open question: is n=6 ceiling even higher?

**Other notable result this turn (arm-1)**: lr=8e-4 val=54.558 / test=**45.909** — has BEST TEST of any run, beating arm-2 by 0.5 units. Val/test divergence suggests val_single overfits slightly at high lr.

## Improvement history (recent)

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
| **#4424 (tanjiro, merged)** | **BF16 + LS + n=8 + bs=2 + lr=9e-4 + default wd + δ=0.10 + slice=32** | **53.595** | **46.395** | **-2.48%** |
| #4425 (edward, merged) | BF16 + LS + n=8 + bs=2 + lr=7e-4 + wd=0.0001 default + δ=0.10 + slice=32 | 54.959 | 47.521 | -0.076% |
| #4448 (alphonse, merged) | BF16 + LS + n=10 + bs=2 + lr=8e-4 + wd=0.001 + δ=0.10 + slice=32 | 55.001 | 47.946 | -0.45% |
| #4349 (tanjiro, merged) | BF16 + LS + n=8 + bs=2 + lr=7e-4 + δ=0.10 + slice=32 | 55.250 | 47.592 | -0.98% |
| #4322 (askeladd, merged) | BF16 + LS + n=10 + bs=2 + δ=0.10 + slice=32 + wd=0.001 | 55.799 | 48.846 | -0.58% |

## Active WIP (7 students + 1 idle this turn → tanjiro reassigned)

| Student | PR | Hypothesis | Stack |
|---|---|---|---|
| alphonse | #4581 | LR warmup {3, 5} epochs on prior best stack (with Huber δ fallback) | n=8+lr=7e-4+wd=0.0001+δ=0.10+slice=32 ⚠️ stack now obsolete (lr=9e-4 is winner) |
| **tanjiro** | **TBD this turn** | **lr=1e-3 push + T_max bracket on new best n=8+lr=9e-4 stack** | TBD |
| edward | #4543 | lr=8e-4 cross-lineage transfer (n=8+lr=8e-4 ± wd=0.001) | n=8+δ=0.10+slice=32 ⚠️ partial result already known: arm-1 lr=8e-4=54.558 |
| askeladd | #4561 | slice bracket {40, 48} on current best stack | n=8+lr=7e-4+wd=0.0001+δ=0.10 ⚠️ stack pre-tanjiro-merge |
| frieren | #4484 | T_max bracket {18, 22} on n=8 stack | n=8+lr=7e-4+δ=0.10+slice=32 ⚠️ pre-tanjiro-merge |
| nezuko | #4544 | clip WIDER {0.30, 0.35} on n=8+lr=7e-4 stack | n=8+lr=7e-4+δ=0.10+slice=32 ⚠️ pre-tanjiro-merge |
| fern | #4559 | n_freqs={11, 12} on current best stack | n=8+lr=7e-4+wd=0.0001+δ=0.10+slice=32 ⚠️ pre-tanjiro-merge |
| thorfinn | #4407 | T_max bracket {16, 18} on n=10+wd=0.001 stack | n=10+wd=0.001+slice=32 |

⚠️ **IMPORTANT**: Many in-flight experiments use the pre-tanjiro-merge n=8+lr=7e-4 stack. Their results will still be useful for lever characterization but will be measured against the NEW best (val=53.595) not the prior best (val=54.959). If a result beats 54.959 but not 53.595, it's still a useful data point but not mergeable.

## Settled levers (do not re-sweep)

| Lever | Settled value | Source |
|---|---|---|
| fourier_base | 2.0 | #4331 |
| slice_num | 32 (val) / 48 (test) | #4298 |
| n_head | 4 | #4367 |
| surf_weight on n=8 stack | 10 (default) | #4439 |
| n_hidden | 128 | #4289 |
| δ on lineage A (n=8) | 0.10 (current best) | #4220, current best uses 0.10 |
| clip × δ=0.10 | clip ≤ 0.25 only | #4222, #4223 |
| **clip on n=8+lr=7e-4** | **clip=0.25 (default) — DO NOT tighten** | **#4449** |
| **wd on n=8+lr=7e-4** | **wd=0.0001 (default) — fully characterized 4-point bracket** | **#4425 + #4479** |
| **lr ceiling on n=10 stack** | **lr=8e-4** | **#4513 — lr=9e-4 saturates clip_frac** |
| **wd at lr=8e-4 on n=10** | **wd=0.001** | **#4513 — wd=0.002 over-regularizes** |
| **lr on n=8 stack** | **lr=9e-4 (current best) — ceiling untested above** | **#4424 — monotone improvement 7→8→9e-4; clip_frac=0.940 has slack** |
| EMA | dead on current stack | #4288 |
| bs | 2 | #4147 |

## Current research priorities

### Priority 1: Push lr ceiling on new best n=8 stack
- **#4424 closed**: lr=9e-4 won. clip_frac=0.940 has slack → lr=1e-3 is the natural next push
- **TBD tanjiro this turn**: lr=1e-3 + T_max bracket on new best stack (assign now)

### Priority 2: Re-evaluate in-flight against new baseline (val=53.595)

All in-flight experiments on n=8+lr=7e-4 stack are now measured against val=53.595 (NEW best) not val=54.959 (prior best). Realistic ceiling for these experiments is probably val ≥ 54.0 unless they introduce a method that compounds with lr=9e-4 — unlikely to beat new best.

The IMPORTANT in-flight results to harvest:
- **#4543 edward**: arm-1 lr=8e-4 should land near tanjiro's arm-1 (val=54.558) — useful as cross-lineage replication
- **#4484 frieren** T_max bracket: if T_max=18 or 22 helps at lr=7e-4, may compound with lr=9e-4
- **#4559 fern** n_freqs=12: if it helps at lr=7e-4, would test the inverse lr-ceiling-vs-n_freqs hypothesis (n=12 should have lower lr ceiling)
- **#4561 askeladd** slice bracket: orthogonal to lr — could compound with new best
- **#4581 alphonse** LR warmup: orthogonal to peak lr — could compound

### Priority 3: Lineage B (n=10) continued
- **#4407 thorfinn** T_max {16, 18} on n=10+wd=0.001 — completes n=10 lever characterization

## Potential next research directions

**Highest priority (post-current wave):**
- **lr=1e-3 + T_max push on n=8 stack**: tanjiro's natural next step (assigned this turn)
- **n_freqs=6 + lr=9e-4 or 1e-3**: tests the inverse lr-ceiling-vs-n_freqs hypothesis. If n=6 has even higher lr ceiling, this is a major mechanism finding
- **lr push + warmup compound**: if alphonse #4581 warmup shows value at lr=7e-4, immediately test at lr=9e-4
- **arm-1 lr=8e-4 had best TEST**: investigate val/test divergence — maybe ensemble of lr=8e-4 + 9e-4 checkpoints

**Other directions:**
- **Cross-lineage compound**: n=10+lr=8e-4+wd=0.001 was prior best on n=10. n=8+lr=9e-4 is new best on n=8. Mixed-lineage features?
- **Per-split δ**: cruise + re_rand respond differently than single + rc — per-split Huber δ
- **Per-domain Fourier basis**: different freq for cruise vs single leading edge

**Bold directions for plateau protocol (if current wave stalls):**
- **Architecture change**: all improvements so far are training/regularization. Transolver's slice mechanism untested architecturally
- **Positional encoding**: per-domain mixed Fourier basis
