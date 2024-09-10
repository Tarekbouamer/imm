import os
from typing import Dict, Any, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from loguru import logger

try:
    from torchsummary import summary
except ImportError:
    summary = None

try:
    from thop import profile
except ImportError:
    profile = None

try:
    from torchviz import make_dot
except ImportError:
    make_dot = None


class ModelBase(nn.Module):
    """
    A base class for PyTorch models providing common functionality.

    This class extends nn.Module and provides a foundation for building
    custom models with additional utility methods for training, evaluation,
    and model analysis.
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize the ModelBase.

        Args:
            cfg (Dict[str, Any]): Configuration dictionary for the model.
        """
        super().__init__()
        self.cfg = cfg

    def build_model(self) -> None:
        """
        Abstract method to build the model architecture.

        This method should be implemented in subclasses to define the
        specific architecture of the model.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pass

    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch (Any): A batch of training data.

        Returns:
            Dict[str, float]: A dictionary containing training metrics.
        """
        raise NotImplementedError("train_step method is not yet implemented.")

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on the validation set.

        Args:
            val_loader (DataLoader): Validation data loader.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        raise NotImplementedError("evaluate method is not yet implemented.")

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the model.

        Args:
            data (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Predictions tensor.
        """
        self.eval()
        with torch.no_grad():
            return self(data)

    def save(self, save_path: str) -> None:
        """
        Save the model state dictionary to a file.

        Args:
            save_path (str): Path to save the model.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.cfg,
            },
            save_path,
        )
        logger.info(f"Model saved to {save_path}")

    def load(self, save_path: str) -> None:
        """
        Load the model state dictionary from a file.

        Args:
            save_path (str): Path to load the model from.
        """
        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.cfg = checkpoint["config"]
        logger.info(f"Model loaded from {save_path}")

    def summary(self, input_size: Optional[Tuple[int, ...]] = None) -> None:
        """
        Print a summary of the model architecture.

        Args:
            input_size (Optional[Tuple[int, ...]]): The size of the input tensor.
        """
        if summary is None:
            logger.error("torchsummary is not installed. Please install it with 'pip install torchsummary'")
            return

        if input_size is None:
            input_size = self.cfg.get("input_shape")

        if input_size:
            summary(self, input_size=input_size)
        else:
            logger.warning("Input shape not provided. Unable to generate summary.")

    def get_flops(self, input_size: Tuple[int, ...]) -> int:
        """
        Estimate the number of FLOPs for a forward pass.

        Args:
            input_size (Tuple[int, ...]): The size of the input tensor.

        Returns:
            int: Estimated number of FLOPs.
        """
        if profile is None:
            logger.error("thop library is not installed. Please install it with 'pip install thop'")
            return -1

        input = torch.randn(1, *input_size).to(self.get_device())
        flops, _ = profile(self, inputs=(input,))
        logger.info(f"Estimated FLOPs: {flops}")
        return flops

    def plot_architecture(self, save_path: str, input_size: Optional[Tuple[int, ...]] = None) -> None:
        """
        Generate and save a visual representation of the model architecture.

        Args:
            save_path (str): Path to save the architecture plot.
            input_size (Optional[Tuple[int, ...]]): The size of the input tensor.
        """
        if make_dot is None:
            logger.error("torchviz library is not installed. Please install it with 'pip install torchviz'")
            return

        if input_size is None:
            input_size = self.cfg.get("input_shape")

        if input_size is None:
            logger.error("Input shape not provided. Unable to generate architecture plot.")
            return

        x = torch.randn(1, *input_size).to(self.get_device())
        y = self(x)

        dot = make_dot(y, params=dict(self.named_parameters()))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dot.render(save_path, format="png", cleanup=True)
        logger.info(f"Model architecture plot saved to {save_path}.png")

    @property
    def num_parameters(self) -> int:
        """
        Return the number of parameters in the model.

        Returns:
            int: Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self) -> int:
        """
        Return the number of trainable parameters.

        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Model parameters frozen.")

    def unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Model parameters unfrozen.")

    def reset_weights(self) -> None:
        """Reset all model parameters."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                m.reset_parameters()
        logger.info("Model weights have been reset.")

    def to(self, device: Union[str, torch.device]) -> "ModelBase":
        """
        Move the model to the specified device.

        Args:
            device (Union[str, torch.device]): The device to move the model to.

        Returns:
            ModelBase: The model instance.
        """
        device = torch.device(device)
        super().to(device)
        return self

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None) -> "ModelBase":
        """
        Move the model to a CUDA device.

        Args:
            device (Optional[Union[int, str, torch.device]]): The CUDA device to move the model to.
                If None, uses the current CUDA device.

        Returns:
            ModelBase: The model instance.

        Raises:
            RuntimeError: If CUDA is not available.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot move model to CUDA device.")

        if device is None:
            device = torch.cuda.current_device()

        return self.to(f"cuda:{device}")

    def cpu(self) -> "ModelBase":
        """
        Move the model to the CPU.

        Returns:
            ModelBase: The model instance.
        """
        return self.to("cpu")

    def get_device(self) -> torch.device:
        """
        Return the device of the model parameters.

        Returns:
            torch.device: The device of the model parameters.
        """
        return next(self.parameters()).device

    def update_cfg(self, new_cfg: Dict[str, Any]) -> None:
        """
        Update the model configuration.

        Args:
            new_cfg (Dict[str, Any]): New configuration to update with.
        """
        self.cfg.update(new_cfg)
        logger.info("Model configuration updated.")
