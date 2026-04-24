import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any

from core.base_model import BaseMLModel

class KNNRunner(BaseMLModel):
    """
    Concrete implementation of a K-Nearest Neighbors model.
    """

    def __init__(self, **kwargs):
        # Pass any kwargs (hyperparameters) up to the base class
        super().__init__(**kwargs)
        # Initialize the scikit-learn model
        self.model = KNeighborsClassifier(**self.hyperparameters)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fits the model to the training data."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generates predictions on new data."""
        return self.model.predict(X_test)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Returns a dictionary of performance metrics."""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)

        return {
            "accuracy": accuracy,
            "detailed_report": report
        }

class SVMRunner(BaseMLModel):
    """
    Concrete implementation of a Support Vector Machine model.
    """

    def __init__(self, **kwargs):
        # Pass any kwargs (hyperparameters) up to the base class
        super().__init__(**kwargs)
        # Initialize the scikit-learn model
        self.model = SVC(**self.hyperparameters)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fits the model to the training data."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generates predictions on new data."""
        return self.model.predict(X_test)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Returns a dictionary of performance metrics."""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)

        return {
            "accuracy": accuracy,
            "detailed_report": report
        }