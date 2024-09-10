from typing import Dict, Type, List, Any, Optional
from rich.table import Table
from rich.console import Console
import torch.nn as nn


class ModelRegistry:
    def __init__(self, name: str, location: str):
        """
        Initialize a new ModelRegistry.

        Args:
            name (str): The name of the registry.
            location (str): The location or context of the registry.
        """
        self.name: str = name
        self.location: str = location
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, default_cfg: Optional[Dict[str, Any]] = None):
        """
        Decorator to register a new model under a given name with an optional default configuration.

        Args:
            name (str): The name under which to register the model.
            default_cfg (Optional[Dict[str, Any]]): Default configuration for the model.

        Returns:
            callable: The decorator function.

        Raises:
            KeyError: If the name is already registered.
        """

        def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
            if name in self._registry:
                raise KeyError(f"Model '{name}' is already registered in {self.name}")
            # if not issubclass(cls, nn.Module):
            #     raise TypeError(f"Registered model must be a subclass of nn.Module, got {cls}")
            self._registry[name] = {
                "class": cls,
                "default_cfg": default_cfg or {},
            }
            return cls

        return decorator

    def get(self, name: str) -> Dict[str, Any]:
        """
        Retrieve a registered model by name.

        Args:
            name (str): The name of the model to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the model class and default configuration.

        Raises:
            KeyError: If the model is not found in the registry.
        """
        if name not in self._registry:
            available_models = ", ".join(self._registry.keys())
            raise KeyError(f"No model registered under name '{name}' in {self.name}. " f"Available models are: {available_models}")
        return self._registry[name]

    def create_model(
        self,
        name: str,
        cfg: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
        **kwargs,
    ) -> nn.Module:
        """
        Create an instance of a registered model with given configuration, pretrained option, and parameters.

        Args:
            name (str): The name of the model to create.
            cfg (Optional[Dict[str, Any]]): Configuration dictionary to override the default config.
            pretrained (bool): Whether to use pretrained weights. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
            nn.Module: An instance of the requested model.

        Raises:
            KeyError: If the model is not found in the registry.
        """
        model_info = self.get(name)
        model_class = model_info["class"]

        model_cfg = model_info["default_cfg"].copy()
        if cfg is not None:
            model_cfg.update(cfg)
        model_cfg.update(kwargs)

        return model_class(cfg=model_cfg, pretrained=pretrained, **kwargs)

    def is_model(self, name: str) -> bool:
        """
        Check if a model with the given name is registered.

        Args:
            name (str): The name of the model to check.

        Returns:
            bool: True if the model is registered, False otherwise.
        """
        return name in self._registry

    def is_pretrained(self, name: str) -> bool:
        """
        Check if a model has pretrained weights available.

        Args:
            name (str): The name of the model to check.

        Returns:
            bool: True if the model has pretrained weights, False otherwise.

        Raises:
            KeyError: If the model is not found in the registry.
        """
        model_info = self.get(name)
        return model_info["default_cfg"].get("pretrained", False)

    def get_default_cfg(self, name: str) -> Dict[str, Any]:
        """
        Get the default configuration for a model.

        Args:
            name (str): The name of the model.

        Returns:
            Dict[str, Any]: The default configuration dictionary for the model.

        Raises:
            KeyError: If the model is not found in the registry.
        """
        model_info = self.get(name)
        return model_info["default_cfg"].copy()

    def unregister(self, name: str) -> None:
        """
        Unregister a model from the registry.

        Args:
            name (str): The name of the model to unregister.

        Raises:
            KeyError: If the model is not found in the registry.
        """
        if name not in self._registry:
            raise KeyError(f"No model registered under name '{name}' in {self.name}")
        del self._registry[name]

    def __len__(self) -> int:
        """Return the number of registered models."""
        return len(self._registry)

    def __repr__(self) -> str:
        """
        Represent the full registry as a table using the rich library.

        Returns:
            str: A string representation of the registry.
        """
        table = Table(title=f"{self.name} Model Registry at {self.location}")
        table.add_column("Model Name", style="magenta")
        table.add_column("Model Class", style="cyan")
        table.add_column("Default Config", style="green")
        for name, info in self._registry.items():
            table.add_row(name, info["class"].__name__, str(info["default_cfg"]))
        console = Console()
        console.print(table)
        return f"{self.name} Model Registry with {len(self)} models."

    @property
    def list_models(self) -> List[str]:
        """
        List all registered models.

        Returns:
            List[str]: A list of all registered model names.
        """
        return list(self._registry.keys())

    def get_model_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            name (str): The name of the model.

        Returns:
            Dict[str, Any]: A dictionary containing model information.

        Raises:
            KeyError: If the model is not found in the registry.
        """
        model_info = self.get(name)
        model_class = model_info["class"]
        return {
            "name": name,
            "class": model_class.__name__,
            "module": model_class.__module__,
            "doc": model_class.__doc__,
            "default_cfg": model_info["default_cfg"],
        }
