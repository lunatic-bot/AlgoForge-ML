from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Any


class BaseMLModel(ABC):
    """
    Abstract base class for all machine learning models in the workbench.
    """
    def __init__(self, **kwargs):
        # self.model will hold the actual scikit-learn/xgboost instance
        self.model = None 
        self.hyperparameters = kwargs

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fits the model to the training data."""
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generates predictions on new data."""
        pass

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Returns a dictionary of performance metrics (Accuracy, MSE, etc.)."""
        pass