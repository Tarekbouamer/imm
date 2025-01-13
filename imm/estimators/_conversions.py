from typing import Union

import numpy as np
import torch
from torch import Tensor


def convert_points_to_homogeneous(points: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    Convert points from Euclidean to homogeneous coordinates.

    Args:
        points: Input points in Euclidean coordinates, shape (..., D).

    Returns:
        Points in homogeneous coordinates, shape (..., D+1).
    """
    if not isinstance(points, (np.ndarray, Tensor)):
        raise TypeError(f"Input type is not a numpy array or torch tensor. Got {type(points)}")

    # Append ones to the last dimension
    if isinstance(points, np.ndarray):
        return np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    else:
        return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def convert_points_from_homogeneous(points: Union[np.ndarray, Tensor], eps: float = 1e-8) -> Union[np.ndarray, Tensor]:
    """
    Convert points from homogeneous to Euclidean coordinates.

    Args:
        points: Input points in homogeneous coordinates, shape (..., D).
        eps: Small value to avoid division by zero.

    Returns:
        Points in Euclidean coordinates, shape (..., D-1).
    """
    if not isinstance(points, (np.ndarray, Tensor)):
        raise TypeError(f"Input type is not a numpy array or torch tensor. Got {type(points)}")

    # Divide by the last dimension
    return points[..., :-1] / (points[..., -1:] + eps)
