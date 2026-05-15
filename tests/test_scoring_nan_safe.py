import torch

from data.scoring import accumulate_batch


def test_inf_gt_does_not_propagate_nan():
    B, N = 2, 10
    pred = torch.randn(B, N, 3)
    y = torch.randn(B, N, 3)
    y[0, 3, 2] = float("inf")
    is_surface = torch.zeros(B, N, dtype=torch.bool)
    is_surface[:, :5] = True
    mask = torch.ones(B, N, dtype=torch.bool)

    mae_surf = torch.zeros(3, dtype=torch.float64)
    mae_vol = torch.zeros(3, dtype=torch.float64)
    n_surf, n_vol = accumulate_batch(pred, y, is_surface, mask, mae_surf, mae_vol)

    assert torch.isfinite(mae_surf).all(), f"NaN/Inf in mae_surf: {mae_surf}"
    assert torch.isfinite(mae_vol).all(), f"NaN/Inf in mae_vol:  {mae_vol}"
    assert n_surf == 5, f"expected 5 surf nodes (only sample 1), got {n_surf}"
    assert n_vol == 5, f"expected 5 vol nodes (only sample 1), got {n_vol}"

    expected_err = (pred[1].double() - y[1].double()).abs()
    expected_surf = expected_err[:5].sum(dim=0)
    expected_vol = expected_err[5:].sum(dim=0)
    assert torch.allclose(mae_surf, expected_surf), (
        f"surface MAE mismatch: {mae_surf} vs {expected_surf}"
    )
    assert torch.allclose(mae_vol, expected_vol), (
        f"volume MAE mismatch: {mae_vol} vs {expected_vol}"
    )
