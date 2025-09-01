from __future__ import annotations
import torch
from typing import Callable


def tsallis_divergence(p: torch.Tensor,
                       q: torch.Tensor,
                       q_param: float = 1.5,
                       eps: float = 1e-6) -> torch.Tensor:
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    inner = p_safe.pow(q_param) * q_safe.pow(1 - q_param)
    summed = torch.sum(inner, dim=-1) - 1
    return torch.mean(summed / (q_param - 1), dim=1)


def renyi_divergence(p: torch.Tensor,
                     q: torch.Tensor,
                     q_param: float = 1.5,
                     eps: float = 1e-6) -> torch.Tensor:
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    inner = p_safe.pow(q_param) * q_safe.pow(1 - q_param)
    summed = torch.sum(inner, dim=-1)
    return torch.mean(torch.log(summed) / (q_param - 1), dim=1)


def kl_divergence(p: torch.Tensor,
                  q: torch.Tensor,
                  eps: float = 1e-6) -> torch.Tensor:
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    kl = p_safe * (torch.log(p_safe) - torch.log(q_safe))
    return torch.mean(torch.sum(kl, dim=-1), dim=1)


def get_divergence_fn(divergence_type: str,
                      q_param: float = 1.5) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    divergence_type = divergence_type.lower()
    if divergence_type == 'tsallis':
        return lambda p, q: tsallis_divergence(p, q, q_param)
    if divergence_type == 'renyi':
        return lambda p, q: renyi_divergence(p, q, q_param)
    if divergence_type == 'kl':
        return lambda p, q: kl_divergence(p, q)
    raise ValueError(f"Unknown divergence type: {divergence_type}")


__all__ = [
    "tsallis_divergence",
    "renyi_divergence",
    "kl_divergence",
    "get_divergence_fn",
]