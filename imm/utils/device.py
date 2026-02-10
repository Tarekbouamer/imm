import numpy as np
import torch
from loguru import logger


def detect_device(force_cpu: bool = False) -> str:
    """Detects the device to use."""

    if force_cpu or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    logger.info(f"Using device: {device}")
    return device


def to_tensor(data):
    """Convert data to PyTorch tensor recursively."""
    if isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_tensor(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_tensor(v) for v in data)
    elif isinstance(data, set):
        return [to_tensor(v) for v in data]
    elif isinstance(data, (int, float, np.ndarray)):
        return torch.tensor(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        raise TypeError("Unsupported data type")


def to_numpy(data):
    """Convert data to NumPy array recursively."""
    if isinstance(data, dict):
        return {k: to_numpy(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_numpy(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_numpy(v) for v in data)
    elif isinstance(data, set):
        return [to_numpy(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, (int, float, np.ndarray)):
        return np.array(data)
    else:
        raise TypeError("Unsupported data type")


def to_cpu(data):
    """Move data to CPU recursively."""
    if isinstance(data, dict):
        return {k: to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_cpu(v) for v in data)
    elif isinstance(data, set):
        return [to_cpu(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cpu()
    else:
        return data


def to_cuda(data):
    """Move data to CUDA recursively, if CUDA is available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if isinstance(data, dict):
        return {k: to_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cuda(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_cuda(v) for v in data)
    elif isinstance(data, set):
        return [to_cuda(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    else:
        return data


def to_half(data):
    """Convert data to half precision recursively, equivalent to float16."""
    if isinstance(data, dict):
        return {k: to_half(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_half(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_half(v) for v in data)
    elif isinstance(data, set):
        return [to_half(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.half() if data.is_floating_point() else data
    elif isinstance(data, np.ndarray):
        return data.astype(np.float16) if np.issubdtype(data.dtype, np.floating) else data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def to_short(data):
    """Convert data to short precision recursively, equivalent to int16."""
    if isinstance(data, dict):
        return {k: to_short(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_short(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_short(v) for v in data)
    elif isinstance(data, set):
        return [to_short(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.short()
    elif isinstance(data, np.ndarray):
        return data.astype(np.int16)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
