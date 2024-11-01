from typing import Any, Dict, Optional

from loguru import logger

from imm.registry.register import ModelRegistry

MATCHERS_REGISTRY = ModelRegistry("matchers", location=__file__)


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
