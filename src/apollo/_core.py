from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

ProjectionType = Literal["random", "svd"]
ScalingType = Literal["channel", "tensor"]


@dataclass(frozen=True)
class MatrixLayout:
    """Maps arbitrary tensors to the paper's matrix-shaped update rule."""

    original_shape: torch.Size
    channel_axis: int
    moved_shape: tuple[int, ...]
    transposed: bool

    @classmethod
    def from_shape(cls, shape: torch.Size | tuple[int, ...]) -> "MatrixLayout":
        shape = torch.Size(shape)
        if len(shape) == 0:
            return cls(shape, 0, (1, 1), False)
        if len(shape) == 1:
            return cls(shape, 0, (1, shape[0]), False)

        channel_axis = max(range(len(shape)), key=shape.__getitem__)
        moved = list(shape)
        channel_size = moved.pop(channel_axis)
        moved.append(channel_size)
        moved_shape = tuple(moved)

        rows = math.prod(moved_shape[:-1])
        cols = moved_shape[-1]
        return cls(shape, channel_axis, moved_shape, rows > cols)

    def flatten(self, tensor: Tensor) -> Tensor:
        if len(self.original_shape) == 0:
            return tensor.reshape(1, 1)
        if len(self.original_shape) == 1:
            return tensor.reshape(1, -1)

        moved = tensor.movedim(self.channel_axis, -1).reshape(-1, self.moved_shape[-1])
        return moved.t() if self.transposed else moved

    def restore(self, matrix: Tensor) -> Tensor:
        if self.transposed:
            matrix = matrix.t()

        if len(self.original_shape) == 0:
            return matrix.reshape(())
        if len(self.original_shape) == 1:
            return matrix.reshape(self.original_shape)

        moved = matrix.reshape(self.moved_shape)
        return moved.movedim(-1, self.channel_axis)

    @property
    def matrix_shape(self) -> tuple[int, int]:
        rows = math.prod(self.moved_shape[:-1])
        cols = self.moved_shape[-1]
        if self.transposed:
            return cols, rows
        return rows, cols


def select_compute_dtype(tensor: Tensor) -> torch.dtype:
    if tensor.dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return tensor.dtype


def sample_random_projection(
    *,
    rows: int,
    rank: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    gen_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)

    projection_device = device if gen_device == device.type else torch.device(gen_device)
    projection = torch.randn(
        rank,
        rows,
        generator=generator,
        device=projection_device,
        dtype=dtype,
    )
    projection.mul_(1.0 / math.sqrt(rank))
    if projection.device != device:
        projection = projection.to(device=device)
    return projection


def compute_svd_projection(matrix: Tensor, rank: int) -> Tensor:
    u, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return u[:, :rank].transpose(0, 1).contiguous()


def refresh_seed() -> int:
    return random.SystemRandom().randrange(0, 2**63 - 1)


def safe_ratio(numerator: Tensor, denominator: Tensor, eps: float) -> Tensor:
    zeros = torch.zeros_like(numerator)
    return torch.where(denominator > 0, numerator / denominator.clamp_min(eps), zeros)


def compute_scale(
    projected_grad: Tensor,
    projected_update: Tensor,
    *,
    scaling: ScalingType,
    eps: float,
) -> Tensor:
    if scaling == "channel":
        update_norm = torch.linalg.vector_norm(projected_update, dim=0)
        grad_norm = torch.linalg.vector_norm(projected_grad, dim=0)
        return safe_ratio(update_norm, grad_norm, eps).unsqueeze(0)

    update_norm = torch.linalg.vector_norm(projected_update)
    grad_norm = torch.linalg.vector_norm(projected_grad)
    return safe_ratio(update_norm, grad_norm, eps).reshape(1, 1)


def apply_norm_growth_limit(
    update: Tensor,
    previous_norm: Tensor | None,
    growth_limit: float | None,
) -> tuple[Tensor, Tensor]:
    current_norm = torch.linalg.vector_norm(update)

    if growth_limit is None:
        return update, current_norm.detach()
    if previous_norm is None or float(previous_norm) <= 0.0 or not math.isfinite(float(previous_norm)):
        return update, current_norm.detach()

    allowed_norm = float(previous_norm) * growth_limit
    current_norm_value = float(current_norm)
    if current_norm_value <= allowed_norm or current_norm_value == 0.0:
        return update, current_norm.detach()

    limited = update * (allowed_norm / current_norm_value)
    return limited, torch.as_tensor(allowed_norm, device=update.device, dtype=update.dtype)
