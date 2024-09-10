from typing import Any, Dict, List

import torch

from .model_base import ModelBase


class FeatureModel(ModelBase):
    """
    A base class for feature extraction models.

    This class extends ModelBase and provides additional functionality
    specific to feature extraction tasks.
    """

    required_inputs: List[str] = []

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """
        Initialize the FeatureModel.

        Args:
            cfg (Dict[str, Any]): Configuration dictionary for the model.
        """
        super().__init__(cfg)

    def transform_inputs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform input data before feature extraction.

        Args:
            data (Dict[str, Any]): Input data dictionary.

        Returns:
            Dict[str, Any]: Transformed input data dictionary.
        """
        # Default implementation: no transformation
        return data

    @torch.no_grad()
    def extract(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Extract features from input data.

        Args:
            data (Dict[str, Any]): Input data dictionary.

        Returns:
            Dict[str, torch.Tensor]: Extracted features dictionary.

        Raises:
            RuntimeWarning: If called in training mode.
            AssertionError: If required inputs are missing.
        """
        if self.training:
            raise RuntimeWarning("extract() should be called in eval mode")

        for k in self.required_inputs:
            assert k in data, f"missing required input '{k}'"

        data = self.transform_inputs(data)
        return self.forward(data)

    def forward(self, x: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the feature extraction model.

        This method should be implemented in subclasses to define the
        specific feature extraction process.

        Args:
            x (Dict[str, Any]): Input data dictionary.

        Returns:
            Dict[str, torch.Tensor]: Extracted features dictionary.
        """
        raise NotImplementedError("forward method must be implemented in subclass")

    def __repr__(self) -> str:
        """
        Return a string representation of the FeatureModel instance.

        Returns:
            str: String representation of the instance.
        """
        return f"{self.__class__.__name__}"
