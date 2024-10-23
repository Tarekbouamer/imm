from typing import Any, Callable, Dict, List, Union

from omegaconf import OmegaConf
import torch
from omegaconf import DictConfig

from .model_base import ModelBase

from enum import Enum


# enum matcher to sparse and dense matchers
class MatcherType(Enum):
    SPARSE = "sparse"
    DENSE = "dense"


class MatcherModel(ModelBase):
    """
    A generic base class for implementing matching models.

    This class extends ModelBase and provides a structure for implementing
    various matching algorithms. It includes methods for input transformation,
    matching operations, and processing of matches.

    Attributes:
        required_inputs (List[str]): A list of required input keys. Should be defined in subclasses.
    """

    required_inputs: List[str] = []  # Define this in subclasses

    def __init__(self, cfg: Union[dict, DictConfig] = {}) -> None:
        """
        Initialize the MatcherModel.

        Args:
            cfg (Union[dict, DictConfig]): Configuration dictionary or DictConfig object.
        """
        super().__init__(cfg=cfg)
        self.cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg

    def transform_inputs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform input data to the format required by the model.

        This method should be implemented in subclasses to perform any necessary
        preprocessing or data formatting.

        Args:
            data (Dict[str, Any]): Input data dictionary.

        Returns:
            Dict[str, Any]: Transformed input data.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError("transform_inputs method must be implemented in subclass")

    def process_matches(self, data: Dict[str, Any], preds: torch.Tensor) -> Dict[str, Any]:
        """
        Process matches using the input data.

        This method should be implemented in subclasses to perform any necessary
        post-processing on the matches, such as finding mutual keypoints.

        Args:
            data (Dict[str, Any]): Input data dictionary.
            matches (torch.Tensor): Tensor containing match indices.

        Returns:
            Dict[str, Any]: Dictionary containing processed match results.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError("process_matches method must be implemented in subclass")

    @torch.no_grad()
    def match(
        self,
        data: Dict[str, Any],
        process_fn: Callable[[Dict[str, Any], torch.Tensor], Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform matching operation in evaluation mode and process matches.

        This method checks for required inputs, transforms the data,
        calls the forward method to perform the actual matching,
        and then processes the matches using the provided or default process_matches method.

        Args:
            data (Dict[str, Any]): Input data dictionary.
            process_fn (Callable): Optional custom function to process matches.
                If not provided, the default process_matches method will be used.
            max_keypoints (int): Maximum number of keypoints to return.

        Returns:
            Dict[str, Any]: Dictionary containing matching results and processed match data.

        Raises:
            RuntimeWarning: If called in training mode.
            AssertionError: If any required input is missing.
        """
        if self.training:
            raise RuntimeWarning("match() should be called in eval mode")

        for k in self.required_inputs:
            assert k in data, f"Missing required input '{k}', we got {data.keys()}"

        data = self.transform_inputs(data)
        preds = self.forward(data)

        # Process matches
        if process_fn is None:
            process_fn = self.process_matches

        return process_fn(data, preds)

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass of the matcher model.

        This method should be implemented in subclasses to define the
        specific matching algorithm.

        Args:
            data (Dict[str, Any]): Transformed input data.

        Returns:
            Dict[str, Any]: Dictionary containing matching results.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError("forward method must be implemented in subclass")

    def __repr__(self) -> str:
        """
        Return a string representation of the MatcherModel.

        Returns:
            str: A string containing the class name and device.
        """
        return f"{self.__class__.__name__}(device={self.device})"
