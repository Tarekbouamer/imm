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
    """Create a matcher model from registry.

    Args:
        name: Model name.
        cfg: Configuration dict.
        pretrained: Use pretrained weights.
        **kwargs: Additional args for model creation.

    Returns:
        Created matcher model.
    """
    logger.info(
        f"Create matcher: {name}" + (f" with config: {cfg}" if cfg is not None else ""))

    try:
        if not MATCHERS_REGISTRY.is_model(name):
            available_models = MATCHERS_REGISTRY.list_models
            raise ValueError(
                f"Matcher '{name}' is not available. Available models are: {', '.join(available_models)}"
            )
        model = MATCHERS_REGISTRY.create_model(
            name, cfg=cfg, pretrained=pretrained, **kwargs)
        logger.info(f"Successfully created matcher: {name}")
        return model

    except Exception as e:
        logger.error(f"Error creating matcher '{name}': {e}")
        raise
