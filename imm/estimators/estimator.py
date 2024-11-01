from typing import Any


class Estimator:
    def __init__(self):
        """Estimators."""
        pass

    def estimate(self, *args) -> Any:
        """Placeholder for the estimate method."""
        raise NotImplementedError("The estimate method should be implemented in the derived class.")

    def __call__(self, *args) -> Any:
        """Allow the estimator to be called like a function."""
        return self.estimate(*args)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
