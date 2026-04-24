import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any

from core.base_model import BaseMLModel

class RandomForestRunner(BaseMLModel):
    """
    Concrete implementation of a Random Forest model.
    """
    def __init__(self, task_type='classification', **kwargs):
        # Pass any kwargs (hyperparameters) up to the base class
        super().__init__(**kwargs)
        self.task_type = task_type
        # Initialize the scikit-learn model with the provided hyperparameters
        if self.task_type == "classification":
            self.model = RandomForestClassifier(**self.hyperparameters)
        else:
            # You can add RandomForestRegressor logic here later
            raise ValueError("Only 'classification' is supported in the MVP.")
        
    
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
