from typing import Any, Mapping, Optional

from loguru import logger

from imm.registry.register import ModelRegistry

MATCHERS_REGISTRY = ModelRegistry("matchers", location=__file__)


def create_matcher(
    name: str,
    cfg: Optional[Mapping[str, Any]] = None,
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
        f"Create matcher: {name} "
        f"pretrained={pretrained} "
        f"cfg={cfg}"
    )

    if not MATCHERS_REGISTRY.is_model(name):
        available_models = MATCHERS_REGISTRY.list_models()
        raise ValueError(
            f"Matcher '{name}' is not available.\n"
            f"Available models:\n"
            f"  - " + "\n  - ".join(sorted(available_models))
        )

    model = MATCHERS_REGISTRY.create_model(
        name, cfg=cfg, pretrained=pretrained, **kwargs)

    if model is None:
        raise RuntimeError(
            f"Failed to create matcher '{name}': create_model returned None")

    logger.info(f"Successfully created matcher: {name}")
    return model
