from typing import Any, Dict, Optional

from loguru import logger

from imm.registry.register import ModelRegistry

EXTRACTORS_REGISTRY = ModelRegistry("extractors", location=__file__)


def create_extractor(
    name: str,
    cfg: Optional[Dict[str, Any]] = None,
    pretrained: bool = True,
    **kwargs: Any,
) -> Any:
    """Create an extractor model from registry.

    Args:
        name: Model name.
        cfg: Configuration dict.
        pretrained: Use pretrained weights.
        **kwargs: Additional args for model creation.

    Returns:
        Created extractor model.
    """
    logger.info(f"Create extractor: {name}" +
                (f" with config: {cfg}" if cfg is not None else ""))

    try:
        if not EXTRACTORS_REGISTRY.is_model(name):
            available_models = EXTRACTORS_REGISTRY.list_models
            raise ValueError(
                f"Extractor '{name}' is not available. Available models are: {', '.join(available_models)}"
            )
        model = EXTRACTORS_REGISTRY.create_model(
            name, cfg=cfg, pretrained=pretrained, **kwargs)
        logger.info(f"Successfully created extractor: {name}")
        return model

    except Exception as e:
        logger.error(f"Error creating extractor '{name}': {e}")
        raise
