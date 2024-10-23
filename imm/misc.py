def _cfg(cfg_base=None, **kwargs):
    cfg = cfg_base.copy() if cfg_base else {}
    cfg.update(kwargs)
    return cfg


def extend_keys_with_suffix(data, suffix="0"):
    return {k + suffix: v for k, v in data.items()}
