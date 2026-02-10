from typing import Any, Mapping, Optional

from loguru import logger

from imm.registry.register import ModelRegistry

EXTRACTORS_REGISTRY = ModelRegistry("extractors", location=__file__)


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
