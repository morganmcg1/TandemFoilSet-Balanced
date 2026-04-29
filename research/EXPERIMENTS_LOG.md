# SENPAI Research Results — icml-appendix-charlie-pai2f-r3

## 2026-04-29 10:52 — PR #1093: Compound baseline anchor (Lion+L1+EMA+bf16+n_layers=1+sw=28+cosine+clip)
- charliepai2f3-alphonse/compound-baseline-lion-l1-ema-bf16-n1
- Hypothesis: re-run the charlie-pai2e-r5 compound recipe as a clean anchor on the new round, so subsequent experiments measure against a freshly executed baseline rather than a referenced number.
- Results:

  | Metric | Value |
  |---|---|
  | val_avg/mae_surf_p | **47.3987** |
  | val_single_in_dist/mae_surf_p | 50.0824 |
  | val_geom_camber_rc/mae_surf_p | 62.7615 |
  | val_geom_camber_cruise/mae_surf_p | 28.5501 |
  | val_re_rand/mae_surf_p | 48.2009 |
  | Peak VRAM | 9.02 GB |
  | Wall time | ~22 min, 50 epochs |
  | Metrics path | `target/models/model-charliepai2f3-alphonse-compound-baseline-lion-l1-ema-bf16-n1-20260429-102214/metrics.jsonl` |

- Verdict: **MERGED** as new round-3 anchor. Improved on the referenced charlie-pai2e-r5 number (47.7385 → 47.3987, −0.34). Per-split is slightly different from the reference (cruise camber improved meaningfully, single-in-dist and rc regressed a touch), so future PRs should treat the new per-split numbers as the comparison target.

## 2026-04-29 11:01 — PR #1104: FiLM global conditioning (Re/AoA/NACA via scale+shift) — sent back for rebase
- charliepai2f3-edward/film-global-conditioning
- Hypothesis: inject the 11-dim global scalar vector (log Re, AoA1, NACA1, AoA2, NACA2, gap, stagger) into each Transolver block as DiT-style scale+shift on both attention and MLP sublayers, with zero-init on the final FiLM projection so the network starts identical to the non-FiLM baseline.
- Results (against reference baseline 47.7385, before #1093 anchored 47.3987):

  | Metric | Reference | Edward | Δ |
  |---|---|---|---|
  | val_avg/mae_surf_p | 47.7385 | **42.3822** | −5.36 (−11.2%) |
  | val_single_in_dist/mae_surf_p | 49.68 | 43.0534 | −6.63 |
  | val_geom_camber_rc/mae_surf_p | 60.82 | 56.9802 | −3.84 |
  | val_geom_camber_cruise/mae_surf_p | 30.55 | 25.1076 | −5.44 |
  | val_re_rand/mae_surf_p | 49.90 | 44.3876 | −5.51 |
  | test_avg/mae_surf_p (bf16, post-fix rerun) | — | 35.8802 | — |
  | test_avg/mae_surf_p (fp32, post-fix rerun) | — | 35.8504 | — |
  | Peak VRAM | — | 3.4 GB | low |
  | Wall time | — | 22.2 min, 50 epochs | matched budget |
  | n_params | ~117K | 245,319 | ~2.1× |
  | Metrics path | — | `target/models/model-charliepai2f3-edward-film-global-conditioning-20260429-100550/metrics.jsonl` | — |

- Notes: best epoch = 50 (final) → still descending; student suggests longer training, FiLM on the preprocess MLP, Fourier on log(Re) for high-Re tail. PR also ships an alternate fix for the NaN bug (drops samples with non-finite `y` from each batch in both train and eval, plus an extra `event: test_rerun_with_nan_filter` line in metrics.jsonl). The originally committed `test_avg/mae_surf_p` is NaN due to the upstream scoring bug; the post-fix rerun line provides clean test numbers.
- Verdict: **REQUEST CHANGES (rebase, top priority)** — strongest signal of the round so far. Squash-merge conflicted with #1093 anchor. Sent back: rebase onto icml-appendix-charlie-pai2f-r3, re-run the same command, keep the NaN filter in evaluate_split. Gate to merge: `val_avg ≤ ~45` on the rebased run.

## 2026-04-29 10:57 — PR #1106: Fourier positional encoding on (x,z) — sent back for rebase
- charliepai2f3-frieren/fourier-positional-encoding
- Hypothesis: enrich (x, z) with sinusoidal features at frequencies {1, 2, 4, 8, 16}×π so the attention can resolve fine-scale boundary-layer geometry, raising input dim 24 → 44.
- Results (against reference baseline 47.7385, before #1093 anchored 47.3987):

  | Metric | Baseline (ref 47.7385) | Frieren | Δ |
  |---|---|---|---|
  | val_avg/mae_surf_p | 47.7385 | **45.3304** | −2.41 (−5.05%) |
  | val_single_in_dist/mae_surf_p | 49.68 | 46.87 | −2.81 |
  | val_geom_camber_rc/mae_surf_p | 60.82 | 60.82 | ≈0 |
  | val_geom_camber_cruise/mae_surf_p | 30.55 | 26.77 | −3.78 |
  | val_re_rand/mae_surf_p | 49.90 | 46.86 | −3.04 |
  | test_avg/mae_surf_p | — | 38.1284 | — |
  | Peak VRAM | — | 9.32 GB | +0.30 GB |
  | Wall time | — | 21.3 min, 50 epochs | matched budget |
  | Metrics path | — | `target/models/model-charliepai2f3-frieren-fourier-pos-enc-compound-v2-20260429-103213/metrics.jsonl` | — |

- Notes: best epoch = 50 (final) → still descending; student suggests longer training, frequency sweep, and applying Fourier to dsdf channels next. PR also ships a critical bug fix in `evaluate_split` that masks samples whose ground truth contains non-finite entries (sample 20 of `test_geom_camber_cruise` has 761 inf entries that were leaking NaN into bf16 test metrics).
- Verdict: **REQUEST CHANGES (rebase)** — would have merged outright but advisor branch already advanced via PR #1093 so the squash conflicted. Sent back: rebase onto icml-appendix-charlie-pai2f-r3, re-run the same command to confirm the improvement still holds on top of the anchor, keep the NaN guard regardless. Gate to merge after rebase: `val_avg ≤ ~46`.
