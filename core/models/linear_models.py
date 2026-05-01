import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
from typing import Dict, Any

from core.base_model import BaseMLModel

class LogisticRegressionRunner(BaseMLModel):
    """
    Concrete implementation of a Logistic Regression model.
    """

    def __init__(self, **kwargs):
        # Pass any kwargs (hyperparameters) up to the base class
        super().__init__(**kwargs)
        # We pass hyperparams like C (inverse regularization strength) or penalty (l1/l2)
        self.model = LogisticRegression(**self.hyperparameters)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fits the model to the training data."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generates predictions on new data."""
        return self.model.predict(X_test)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Returns a dictionary of performance metrics (Accuracy, MSE, etc.)."""
        predictions = self.predict(X_test)

        #calculate core metrics
        accuracy = accuracy_score(y_test, predictions)

        # output_dict=True makes this incredibly easy to return as JSON later
        report = classification_report(y_test, predictions, output_dict=True)

        return {
            "accuracy": accuracy,
            "detailed_report": report
        }
    

class LinearRegressionRunner(BaseMLModel):
    """
    Concrete implementation of a Linear Regression model.
    """

    def __init__(self, **kwargs):
        # Pass any kwargs (hyperparameters) up to the base class
        super().__init__(**kwargs)
        # We pass hyperparams like C (inverse regularization strength) or penalty (l1/l2)
        self.model = LinearRegression(**self.hyperparameters)

    # def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    #     """Fits the model to the training data."""
    #     self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generates predictions on new data."""
        return self.model.predict(X_test)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2_score": r2_score(y_test, predictions)
        }