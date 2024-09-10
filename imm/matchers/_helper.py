from loguru import logger
from typing import Optional, Dict, Any

import torch
from imm.registry.register import ModelRegistry

MATCHERS_REGISTRY = ModelRegistry("matchers", location=__file__)


def process_tensor_data(data: Any) -> Any:
    if isinstance(data, torch.Tensor):
        # Process single tensor
        if data.dim() == 1:
            return data.unsqueeze(0).unsqueeze(0)  # [D] -> [1, 1, D]
        elif data.dim() == 2:
            return data.unsqueeze(0)  # [N, D] -> [1, N, D]
        elif data.dim() == 3:
            return data  # Already 3D
        else:
            raise ValueError(f"Cannot convert tensor of shape {data.shape} to 3D")

    elif isinstance(data, (list, tuple)) and all(isinstance(item, torch.Tensor) for item in data):
        # Process list or tuple of tensors
        stacked = torch.stack(data)
        return process_tensor_data(stacked)  # Recursive call to ensure 3D

    else:
        # Return unchanged if not a tensor or list/tuple of tensors
        return data


def create_matcher(
    name: str,
    cfg: Optional[Dict[str, Any]] = None,
    pretrained: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Create a matcher model.

    Args:
        name (str): Name of the matcher model.
        cfg (Optional[Dict[str, Any]], optional): Configuration for the model. Defaults to None.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        **kwargs: Additional keyword arguments for model creation.

    Returns:
        Any: The created matcher model.

    Raises:
        ValueError: If the matcher is not available in the registry.
        Exception: If there's an error during model creation.
    """
    logger.info(f"Create matcher: {name}" + (f" with config: {cfg}" if cfg is not None else ""))

    try:
        if not MATCHERS_REGISTRY.is_model(name):
            available_models = MATCHERS_REGISTRY.list_models
            print(MATCHERS_REGISTRY)
            raise ValueError(f"Matcher '{name}' not available. Available matchers: {available_models}")

        model = MATCHERS_REGISTRY.create_model(name, cfg=cfg, pretrained=pretrained, **kwargs)
        logger.info(f"Successfully created matcher: {name}")
        return model

    except Exception as e:
        logger.error(f"Error creating matcher '{name}': {str(e)}")
        raise
