from abc import ABC, abstractmethod
from typing import Any, Dict


class Estimator(ABC):
    @abstractmethod
    def estimate(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Estimate the parameters of the model."""
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Allow the estimator to be called like a function."""
        return self.estimate(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
