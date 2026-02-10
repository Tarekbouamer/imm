from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from pytlsd import lsd as lsd_py
from torch import Tensor

from imm.base import FeatureModel, tfn_grayscale
from imm.extractors._helper import EXTRACTORS_REGISTRY
from imm.misc import _cfg


class LSD(FeatureModel):
    """A feature extraction model for detecting line segments in images using the LSD (Line Segment Detector) algorithm."""

    required_data_keys = ["image"]

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

    def detect_lines(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detects line segments in the input image using the LSD algorithm.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - lines: Detected line segments as a numpy array of shape (N, 2, 2).
                - scores: Scores for each detected line segment as a numpy array of shape (N,).
        """
        # Run LSD to detect line segments
        segments = lsd_py(image)

        # Filter out line segments that do not meet the minimum length criteria
        lengths = np.linalg.norm(segments[:, 2:4] - segments[:, 0:2], axis=1)
        valid_mask = lengths >= self.cfg["min_length"]
        segments, lengths = segments[valid_mask], lengths[valid_mask]

        # Calculate scores for each line segment (score = confidence * sqrt(length))
        scores = segments[:, -1] * np.sqrt(lengths)
        lines = segments[:, :4].reshape(-1, 2, 2)

        # Sort line segments by score in descending order
        sorted_indices = np.argsort(-scores)

        # Keep only the top-scoring line segments if max_lines is specified
        if self.cfg["max_lines"] is not None and self.cfg["max_lines"] > 0:
            sorted_indices = sorted_indices[: self.cfg["max_lines"]]
            lines = lines[sorted_indices]
            scores = scores[sorted_indices]

        return np.array(lines), np.array(scores)

    def transform_inputs(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Prepares the input data for line detection."""
        # Ensure the image tensor is 4D (batch, channel, height, width)
        if data["image"].dim() == 3:
            data["image"] = data["image"].unsqueeze(0)

        # Convert the image to grayscale
        data["image"] = tfn_grayscale(data["image"])
        _, _, height, width = data["image"].shape
        data["size"] = torch.tensor([width, height])

        return data

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass for line detection."""
        image = data["image"]
        device = image.device

        # Convert the image to unsigned 8-bit integer
        image_np = np.uint8(image.squeeze(1).cpu().numpy() * 255)

        # Detect lines
        lines, scores = self.detect_lines(image_np[0])

        # Convert to tensors
        lines = torch.tensor(lines, dtype=torch.float, device=device)
        scores = torch.tensor(scores, dtype=torch.float, device=device)

        return {"lines": lines.reshape(-1, 4), "scores": scores}


# Default cfgs
default_cfgs = {
    "lsd": _cfg(
        url=None,
        min_length=30,
        max_lines=-1,
    )
}


def _make_model(
    name: str,
    cfg: Dict[str, Any] = {},
    pretrained: bool = True,
    **kwargs: Dict[str, Any],
) -> nn.Module:
    """Creates an instance of the LSD model."""
    # Create the LSD model
    model = LSD(cfg=cfg)

    # Warn if pretrained weights are requested (not supported for LSD)
    if pretrained:
        logger.warning("LSD does not have pretrained weights available.")

    return model


@EXTRACTORS_REGISTRY.register(name="lsd", default_cfg=default_cfgs["lsd"])
def lsd(cfg: Dict[str, Any] = {}, **kwargs) -> nn.Module:
    return _make_model(name="lsd", cfg=cfg, **kwargs)
