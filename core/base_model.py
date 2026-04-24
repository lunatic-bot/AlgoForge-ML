# Abstract Base Class (BaseMLModel)
from abc import ABC, abstractmethod


class BaseMLModel(ABC):
    """Abstract base class for all ML models."""

    @abstractmethod
    def fit(self, X, y):
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions on the given data."""
        pass

    @abstractmethod
    def score(self, X, y):
        """Return the model score on the given data."""
        pass