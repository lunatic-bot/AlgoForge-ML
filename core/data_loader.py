# Logic to load and preprocess datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataLoader:
    """Load and preprocess datasets for ML training."""

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_csv(self, path):
        """Load data from a CSV file."""
        self.data = pd.read_csv(path)
        return self.data

    def load_titanic(self):
        """Load the Titanic dataset."""
        # Placeholder - would load from data/raw/titanic.csv
        pass

    def load_housing(self):
        """Load the Housing dataset."""
        # Placeholder - would load from data/raw/housing.csv
        pass

    def split(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def encode_labels(self, y):
        """Encode target labels."""
        return self.label_encoder.fit_transform(y)

    def preprocess(self, data, target_column=None):
        """Full preprocessing pipeline."""
        if target_column and target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            # Handle categorical columns
            X = pd.get_dummies(X, drop_first=True)
            return X, y
        return data, None