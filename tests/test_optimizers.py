from __future__ import annotations

import math

import torch

from apollo import Apollo, ApolloMini


def test_apollo_state_shape_matches_effective_rank():
    param = torch.nn.Parameter(torch.zeros(3, 5))
    param.grad = torch.ones_like(param)
    optimizer = Apollo([param], lr=1e-2, rank=16, update_proj_gap=5, projection="random")

    optimizer.step()

    state = optimizer.state[param]
    assert state["exp_avg"].shape == (3, 5)
    assert state["exp_avg_sq"].shape == (3, 5)
    assert isinstance(state["proj_seed"], int)
    assert state["projection_basis"] is None


def test_random_projection_seed_refreshes_on_schedule():
    param = torch.nn.Parameter(torch.zeros(2, 4))
    optimizer = Apollo([param], lr=1e-2, rank=2, update_proj_gap=2, projection="random")

    param.grad = torch.ones_like(param)
    optimizer.step()
    seed_step_1 = optimizer.state[param]["proj_seed"]

    param.grad = torch.full_like(param, 2.0)
    optimizer.step()
    seed_step_2 = optimizer.state[param]["proj_seed"]

    param.grad = torch.full_like(param, 3.0)
    optimizer.step()
    seed_step_3 = optimizer.state[param]["proj_seed"]

    assert seed_step_1 == seed_step_2
    assert seed_step_3 != seed_step_2


def test_apollo_mini_uses_rank_one_and_tensor_scaling():
    param = torch.nn.Parameter(torch.zeros(4, 4))
    param.grad = torch.ones_like(param)
    optimizer = ApolloMini([param], lr=1e-2)
    optimizer.step()

    state = optimizer.state[param]
    assert state["exp_avg"].shape == (1, 4)
    assert optimizer.param_groups[0]["rank"] == 1
    assert optimizer.param_groups[0]["scaling"] == "tensor"
    assert math.isclose(optimizer.param_groups[0]["alpha"], math.sqrt(128.0))


def test_apollo_optimizes_simple_quadratic():
    param = torch.nn.Parameter(torch.tensor([5.0, -4.0]))
    optimizer = Apollo(
        [param],
        lr=0.1,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        rank=1,
        projection="svd",
        norm_growth_limit=None,
    )

    initial_loss = None
    final_loss = None
    for step in range(30):
        optimizer.zero_grad()
        loss = (param**2).sum()
        loss.backward()
        if step == 0:
            initial_loss = float(loss.detach())
        optimizer.step()
        final_loss = float(loss.detach())

    assert final_loss is not None
    assert initial_loss is not None
    assert final_loss < initial_loss


def test_apollo_mini_optimizes_simple_quadratic():
    param = torch.nn.Parameter(torch.tensor([2.0, -3.0, 1.5]))
    optimizer = ApolloMini(
        [param],
        lr=0.02,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        projection="svd",
        norm_growth_limit=None,
    )

    initial_loss = None
    final_loss = None
    for step in range(40):
        optimizer.zero_grad()
        loss = (param**2).sum()
        loss.backward()
        if step == 0:
            initial_loss = float(loss.detach())
        optimizer.step()
        final_loss = float(loss.detach())

    assert final_loss is not None
    assert initial_loss is not None
    assert final_loss < initial_loss
