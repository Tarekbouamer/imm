from typing import Any, Mapping, Optional

from loguru import logger

from imm.registry.register import ModelRegistry

EXTRACTORS_REGISTRY = ModelRegistry("extractors", location=__file__)
LINE_EXTRACTORS_REGISTRY = ModelRegistry("extractors", location=__file__)


def create_extractor(
    name: str,
    cfg: Optional[Mapping[str, Any]] = None,
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
    logger.info(
        f"Create extractor: {name} "
        f"pretrained={pretrained} "
        f"cfg={cfg}"
    )

    if not EXTRACTORS_REGISTRY.is_model(name):
        available_models = EXTRACTORS_REGISTRY.list_models()
        raise ValueError(
            f"Extractor '{name}' is not available.\n"
            f"Available models:\n"
            f"  - " + "\n  - ".join(sorted(available_models))
        )

    model = EXTRACTORS_REGISTRY.create_model(
        name, cfg=cfg, pretrained=pretrained, **kwargs)

    if model is None:
        raise RuntimeError(
            f"Failed to create extractor '{name}': create_model returned None")

    logger.info(f"Successfully created extractor: {name}")
    return model


def create_line_extractor(
    name: str,
    cfg: Optional[Mapping[str, Any]] = None,
    pretrained: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Create a line extractor model.

    Args:
        name (str): Name of the line extractor model.
        cfg (Optional[Mapping[str, Any]], optional): Configuration for the model. Defaults to None.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        **kwargs: Additional keyword arguments for model creation.

    Returns:
        Any: The created line extractor model.

    Raises:
        ValueError: If the line extractor is not available in the registry.
        Exception: If there's an error during model creation.
    """
    logger.info(f"Create line extractor: {name}" +
                (f" with config: {cfg}" if cfg is not None else ""))

    if not LINE_EXTRACTORS_REGISTRY.is_model(name):
        available_models = LINE_EXTRACTORS_REGISTRY.list_models()
        raise ValueError(
            f"Line extractor '{name}' is not available. " f"Available models are: {', '.join(available_models)}"
        )
    model = LINE_EXTRACTORS_REGISTRY.create_model(
        name, cfg=cfg, pretrained=pretrained, **kwargs)
    logger.info(f"Successfully created line extractor: {name}")
    return model
