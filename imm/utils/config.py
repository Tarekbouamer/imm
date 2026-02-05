
def merge_config(cfg_base=None, **kwargs):
    """Merge base config with additional kwargs.

    Args:
        cfg_base: Base configuration dict
        **kwargs: Additional config overrides

    Returns:
        Merged configuration dict
    """
    cfg = cfg_base.copy() if cfg_base else {}
    cfg.update(kwargs)
    return cfg
