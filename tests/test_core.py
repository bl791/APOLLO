from __future__ import annotations

import torch

from apollo._core import MatrixLayout, apply_norm_growth_limit, compute_scale


def test_matrix_layout_round_trip_for_matrix():
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    layout = MatrixLayout.from_shape(tensor.shape)
    restored = layout.restore(layout.flatten(tensor))
    assert torch.equal(restored, tensor)


def test_matrix_layout_round_trip_for_higher_rank_tensor():
    tensor = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    layout = MatrixLayout.from_shape(tensor.shape)
    restored = layout.restore(layout.flatten(tensor))
    assert torch.equal(restored, tensor)


def test_channel_scale_matches_columnwise_norm_ratio():
    projected_grad = torch.tensor([[3.0, 0.0], [4.0, 5.0]])
    projected_update = torch.tensor([[6.0, 0.0], [8.0, 10.0]])
    scale = compute_scale(projected_grad, projected_update, scaling="channel", eps=1e-12)
    assert torch.allclose(scale, torch.tensor([[2.0, 2.0]]))


def test_norm_growth_limit_caps_update_norm():
    update = torch.tensor([6.0, 8.0])
    previous_norm = torch.tensor(5.0)
    limited, new_norm = apply_norm_growth_limit(update, previous_norm, 1.1)
    assert torch.allclose(torch.linalg.vector_norm(limited), torch.tensor(5.5), atol=1e-6)
    assert torch.allclose(new_norm, torch.tensor(5.5), atol=1e-6)
