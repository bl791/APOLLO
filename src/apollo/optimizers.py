from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

from ._core import (
    MatrixLayout,
    ProjectionType,
    ScalingType,
    apply_norm_growth_limit,
    compute_scale,
    compute_svd_projection,
    refresh_seed,
    sample_random_projection,
    select_compute_dtype,
)


class _ApolloBase(Optimizer):
    r"""Shared clean-room implementation of the Apollo optimizer family.

    The implementation follows the algorithm described in arXiv:2412.05270,
    using projected first/second moments to estimate either channel-wise or
    tensor-wise gradient scaling factors in the full parameter space.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        rank: int = 128,
        alpha: float = 1.0,
        update_proj_gap: int = 200,
        projection: ProjectionType = "random",
        scaling: ScalingType = "channel",
        norm_growth_limit: float | None = 1.01,
        maximize: bool = False,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if rank <= 0:
            raise ValueError(f"Invalid rank value: {rank}")
        if update_proj_gap <= 0:
            raise ValueError(f"Invalid update_proj_gap value: {update_proj_gap}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
        if projection not in {"random", "svd"}:
            raise ValueError(f"Invalid projection type: {projection}")
        if scaling not in {"channel", "tensor"}:
            raise ValueError(f"Invalid scaling type: {scaling}")
        if norm_growth_limit is not None and norm_growth_limit <= 0.0:
            raise ValueError(f"Invalid norm_growth_limit value: {norm_growth_limit}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            alpha=alpha,
            update_proj_gap=update_proj_gap,
            projection=projection,
            scaling=scaling,
            norm_growth_limit=norm_growth_limit,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            requested_rank = group["rank"]
            alpha = group["alpha"]
            update_proj_gap = group["update_proj_gap"]
            projection = group["projection"]
            scaling = group["scaling"]
            norm_growth_limit = group["norm_growth_limit"]
            maximize = group["maximize"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Apollo does not support sparse gradients.")

                grad = grad if not maximize else -grad
                layout = MatrixLayout.from_shape(param.shape)
                matrix = layout.flatten(grad)
                compute_dtype = select_compute_dtype(matrix)
                matrix = matrix.to(dtype=compute_dtype)

                rows, cols = matrix.shape
                effective_rank = min(requested_rank, rows)
                state = self.state[param]

                if not state or state["exp_avg"].shape != (effective_rank, cols):
                    state["step"] = 0
                    state["rank"] = effective_rank
                    state["exp_avg"] = torch.zeros(
                        effective_rank,
                        cols,
                        device=matrix.device,
                        dtype=compute_dtype,
                    )
                    state["exp_avg_sq"] = torch.zeros(
                        effective_rank,
                        cols,
                        device=matrix.device,
                        dtype=compute_dtype,
                    )
                    state["prev_update_norm"] = torch.zeros((), device=matrix.device, dtype=compute_dtype)
                    state["proj_seed"] = refresh_seed()
                    state["projection_basis"] = None

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                prev_update_norm = state["prev_update_norm"]

                state["step"] += 1
                if projection == "random":
                    if (state["step"] - 1) % update_proj_gap == 0:
                        state["proj_seed"] = refresh_seed()
                    projection_matrix = sample_random_projection(
                        rows=rows,
                        rank=effective_rank,
                        seed=state["proj_seed"],
                        device=matrix.device,
                        dtype=compute_dtype,
                    )
                else:
                    basis = state.get("projection_basis")
                    needs_refresh = (
                        basis is None
                        or basis.shape != (effective_rank, rows)
                        or basis.device != matrix.device
                        or basis.dtype != compute_dtype
                        or (state["step"] - 1) % update_proj_gap == 0
                    )
                    if needs_refresh:
                        basis = compute_svd_projection(matrix, effective_rank)
                        state["projection_basis"] = basis
                    projection_matrix = basis

                projected_grad = projection_matrix @ matrix

                exp_avg.mul_(beta1).add_(projected_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(projected_grad, projected_grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                projected_update = (exp_avg / bias_correction1) / denom

                effective_scaling = scaling
                if min(matrix.shape) == 1:
                    # The paper defines channel-wise updates for matrices; tiny
                    # tensors fall back to tensor-wise scaling for stability.
                    effective_scaling = "tensor"
                scale = compute_scale(
                    projected_grad,
                    projected_update,
                    scaling=effective_scaling,
                    eps=eps,
                )

                full_update = matrix * scale * alpha
                full_update, new_norm = apply_norm_growth_limit(
                    full_update,
                    prev_update_norm,
                    norm_growth_limit,
                )
                state["prev_update_norm"] = new_norm

                update = layout.restore(full_update).to(dtype=param.dtype)
                if weight_decay != 0:
                    param.add_(param, alpha=-lr * weight_decay)
                param.add_(update, alpha=-lr)

        return loss


class Apollo(_ApolloBase):
    r"""Apollo optimizer from arXiv:2412.05270.

    Defaults follow the paper's random-projection, channel-wise variant:
    `projection="random"`, `scaling="channel"`, `alpha=1.0`,
    and `update_proj_gap=200`.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        rank: int = 128,
        alpha: float = 1.0,
        update_proj_gap: int = 200,
        projection: ProjectionType = "random",
        norm_growth_limit: float | None = 1.01,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            alpha=alpha,
            update_proj_gap=update_proj_gap,
            projection=projection,
            scaling="channel",
            norm_growth_limit=norm_growth_limit,
            maximize=maximize,
        )


class ApolloMini(_ApolloBase):
    r"""Apollo-Mini from arXiv:2412.05270.

    Apollo-Mini uses a rank-1 auxiliary space and tensor-wise scaling,
    matching the paper's extreme memory-efficiency variant.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        alpha: float = math.sqrt(128.0),
        update_proj_gap: int = 200,
        projection: ProjectionType = "random",
        norm_growth_limit: float | None = 1.01,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=1,
            alpha=alpha,
            update_proj_gap=update_proj_gap,
            projection=projection,
            scaling="tensor",
            norm_growth_limit=norm_growth_limit,
            maximize=maximize,
        )
