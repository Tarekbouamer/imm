from typing import Any, Dict, Optional

from loguru import logger

from imm.registry.register import ModelRegistry

EXTRACTORS_REGISTRY = ModelRegistry("extractors", location=__file__)
LINE_EXTRACTORS_REGISTRY = ModelRegistry("extractors", location=__file__)


def create_extractor(
    name: str,
    cfg: Optional[Dict[str, Any]] = None,
    pretrained: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Create an extractor model.

    Args:
        name (str): Name of the extractor model.
        cfg (Optional[Dict[str, Any]], optional): Configuration for the model. Defaults to None.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        **kwargs: Additional keyword arguments for model creation.

    Returns:
        Any: The created extractor model.

    Raises:
        ValueError: If the extractor is not available in the registry.
        Exception: If there's an error during model creation.
    """
    logger.info(f"Create extractor: {name}" + (f" with config: {cfg}" if cfg is not None else ""))

    try:
        if not EXTRACTORS_REGISTRY.is_model(name):
            available_models = EXTRACTORS_REGISTRY.list_models
            raise ValueError(
                f"Extractor '{name}' is not available. " f"Available models are: {', '.join(available_models)}"
            )
        model = EXTRACTORS_REGISTRY.create_model(name, cfg=cfg, pretrained=pretrained, **kwargs)
        logger.info(f"Successfully created extractor: {name}")
        return model

    except Exception as e:
        print(EXTRACTORS_REGISTRY)
        logger.error(f"Error creating extractor '{name}': {str(e)}")
        raise


def create_line_extractor(
    name: str,
    cfg: Optional[Dict[str, Any]] = None,
    pretrained: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Create a line extractor model.

    Args:
        name (str): Name of the line extractor model.
        cfg (Optional[Dict[str, Any]], optional): Configuration for the model. Defaults to None.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        **kwargs: Additional keyword arguments for model creation.

    Returns:
        Any: The created line extractor model.

    Raises:
        ValueError: If the line extractor is not available in the registry.
        Exception: If there's an error during model creation.
    """
    logger.info(f"Create line extractor: {name}" + (f" with config: {cfg}" if cfg is not None else ""))

    try:
        if not LINE_EXTRACTORS_REGISTRY.is_model(name):
            available_models = LINE_EXTRACTORS_REGISTRY.list_models
            raise ValueError(
                f"Line extractor '{name}' is not available. " f"Available models are: {', '.join(available_models)}"
            )
        model = LINE_EXTRACTORS_REGISTRY.create_model(name, cfg=cfg, pretrained=pretrained, **kwargs)
        logger.info(f"Successfully created line extractor: {name}")
        return model

    except Exception as e:
        logger.error(f"Error creating line extractor '{name}': {str(e)}")
        raise
