from typing import Optional, Tuple, Type, TypeVar

import numpy as np
import torch

T = TypeVar("T")


def CHECK(condition: bool, msg: Optional[str] = None, raises: bool = True) -> bool:
    """
    Checks a condition, optionally raising an exception if it fails.
    """
    if not condition:
        if raises:
            raise Exception(f"{msg or 'Condition Failed'}")
        return False
    return True


def CHECK_TYPE(x: object, typ: Type[T] | Tuple[Type[T], ...], msg: Optional[str] = None, raises: bool = True) -> bool:
    """
    Checks if a variable is of a specified type.
    """
    #
    return CHECK(isinstance(x, typ), f"Invalid type: {type(x)}. {msg or ''}", raises=raises)


def CHECK_SHAPE(data, expected_shape: Tuple[int, ...], raises: bool = True) -> bool:
    """
    Checks if a NumPy array or PyTorch tensor has a specific shape.
    """
    # Ensure type
    CHECK_TYPE(data, (np.ndarray, torch.Tensor), "Input is not a NumPy array or PyTorch tensor.", raises=raises)

    actual_shape = data.shape

    # Check dimensions
    if len(actual_shape) != len(expected_shape):
        return CHECK(False, f"Expected {len(expected_shape)} dimensions, but got {len(actual_shape)}.", raises=raises)

    # Check each dimension
    for i, (actual_dim, expected_dim) in enumerate(zip(actual_shape, expected_shape)):
        if expected_dim != -1 and actual_dim != expected_dim:
            return CHECK(
                False, f"Dimension mismatch at index {i}: expected {expected_dim}, got {actual_dim}.", raises=raises
            )

    return True
