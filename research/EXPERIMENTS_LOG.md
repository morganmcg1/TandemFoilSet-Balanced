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
