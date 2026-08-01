import torch

from data.scoring import accumulate_batch


def test_accumulate_batch_excludes_nonfinite_ground_truth_samples():
    pred = torch.tensor(
        [
            [[2.0, 3.0, 4.0], [6.0, 8.0, 10.0]],
            [[100.0, 100.0, 100.0], [100.0, 100.0, 100.0]],
        ]
    )
    target = torch.tensor(
        [
            [[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]],
            [[float("inf"), 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    is_surface = torch.tensor([[True, False], [True, False]])
    mask = torch.ones((2, 2), dtype=torch.bool)
    mae_surf = torch.zeros(3, dtype=torch.float64)
    mae_vol = torch.zeros(3, dtype=torch.float64)

    n_surf, n_vol = accumulate_batch(
        pred, target, is_surface, mask, mae_surf, mae_vol
    )

    assert (n_surf, n_vol) == (1, 1)
    torch.testing.assert_close(
        mae_surf, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        mae_vol, torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
    )
