from omegaconf import OmegaConf


class Estimator:
    default_cfg = {}

    def __init__(self, cfg=None):
        # Merge default cfg with user cfg
        if cfg is None:
            cfg = {}
        cfg = {**self.default_cfg, **cfg}
        self.cfg = OmegaConf.create(cfg) if not isinstance(cfg, OmegaConf) else cfg

    def estimate(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses")

    def __call__(self, *args, **kwargs):
        return self.estimate(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(cfg={self.cfg})"
