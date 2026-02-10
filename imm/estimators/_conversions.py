from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor


def convert_points_to_homogeneous(points: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """Convert points from Euclidean to homogeneous coordinates.

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
    """Convert points from homogeneous to Euclidean coordinates.

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


def essential_from_Rt(R: Union[np.ndarray, Tensor], t: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """Compute the essential matrix from the rotation matrix and translation vector.

    Args:
        R (Union[np.ndarray, Tensor]): Rotation matrix, shape (3, 3).
        t (Union[np.ndarray, Tensor]): Translation vector, shape (3,).

    Returns:
        Union[np.ndarray, Tensor]: Essential matrix, shape (3, 3).
    """
    if isinstance(R, np.ndarray):
        t_x = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        return t_x @ R
    else:
        t_x = torch.tensor([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=R.dtype, device=R.device)
        return t_x @ R


def compute_epipolar_lines(
    F: Union[np.ndarray, Tensor], points: Union[np.ndarray, Tensor]
) -> Union[np.ndarray, Tensor]:
    """Compute the epipolar lines for a set of points.

    Args:
        F (Union[np.ndarray, Tensor]): The fundamental matrix, shape (3, 3).
        points (Union[np.ndarray, Tensor]): The points in the first image, shape (N, 2).

    Returns:
        Union[np.ndarray, Tensor]: The epipolar lines in the second image, shape (N, 3).
    """
    # Ensure points are homogeneous
    points = convert_points_to_homogeneous(points)

    # Compute the epipolar lines
    if isinstance(F, np.ndarray):
        return F @ points.T
    else:
        return F @ points.transpose(-1, -2)


def fundamental_from_essential(
    E: Union[np.ndarray, Tensor], K1: Union[np.ndarray, Tensor], K2: Union[np.ndarray, Tensor]
) -> Union[np.ndarray, Tensor]:
    """Compute the fundamental matrix from the essential matrix and camera matrices.

    Args:
        E (Union[np.ndarray, Tensor]): The essential matrix, shape (3, 3).
        K1 (Union[np.ndarray, Tensor]): The camera matrix of the first image, shape (3, 3).
        K2 (Union[np.ndarray, Tensor]): The camera matrix of the second image, shape (3, 3).

    Returns:
        Union[np.ndarray, Tensor]: The fundamental matrix, shape (3, 3).
    """
    if isinstance(E, np.ndarray):
        return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    else:
        return torch.inverse(K2).transpose(-1, -2) @ E @ torch.inverse(K1)
