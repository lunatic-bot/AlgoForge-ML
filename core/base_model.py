from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.model_selection import GridSearchCV


class BaseMLModel(ABC):
    """
    Abstract base class for all machine learning models in the workbench.
    """
    def __init__(self, **kwargs):
        # self.model will hold the actual scikit-learn/xgboost instance
        self.model = None 
        self.hyperparameters = kwargs

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, tune: bool=False, param_grid: dict = None) -> None:
        """Fits the model to the training data."""
        if tune and param_grid:
            # cv = 3 means 3-fold cross-validation. n_jobs = -1, use all the CPU cores
            search = GridSearchCV(self.model, param_grid, cv=3, n_jobs=-1)
            search.fit(X_train, y_train)

            # overwrite the model with the best estimator
            self.model = search.best_estimator_
        else:
            self.model.fit(X_train, y_train)
        

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generates predictions on new data."""
        pass

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Returns a dictionary of performance metrics (Accuracy, MSE, etc.)."""
        pass