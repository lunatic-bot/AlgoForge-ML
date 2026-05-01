import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple
from sklearn.impute import SimpleImputer

class MLDataLoader:
    """
    Handles data ingestion, preprocessing, and safe splitting to avoid data leakage.
    """
    def __init__(self, target_column: str, test_size: float = 0.2, random_state: int = 42):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = None
    
    def process_data(self, df: pd.DataFrame, requires_scaling: bool = True)->Tuple:
        """
        Processes a pandas DataFrame and returns safe train/test splits.
        
        Args:
            df: The raw pandas DataFrame.
            requires_scaling: True for distance/linear models, False for tree models.
        """

        # # 1. Basic Imputation (Dropping missing values for MVP, can upgrade to SimpleImputer later)
        # df_clean = df.dropna().copy()

        # 1. Separate Features (X) and Target (y)
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
    
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # 2. Advanced Imputation (Handling missing real-world data)
        # We isolate numeric columns to fill missing numbers with the median
        numeric_cols = X.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            self.imputer = SimpleImputer(strategy='median')
            # Fit and transform the data, then rebuild the DataFrame
            X_imputed = self.imputer.fit_transform(X[numeric_cols])
            X.loc[:, numeric_cols] = X_imputed

        # 3. Handle Categorical Features (One-Hot Encoding)
        # Converts text columns into 0s and 1s so math algorithms can process them
        X = pd.get_dummies(X, drop_first=True)
        # Forceing all columns (including new True/False dummies) to be strictly numeric floats
        X = X.astype(float)

        # 4. Encode the Target Variable (if it's text like 'Yes'/'No')
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = pd.Series(self.target_encoder.fit_transform(y), index=y.index)

        # 5. Train-Test Split (CRITICAL: Do this BEFORE scaling)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        # 6. Scale features(if required)
        if requires_scaling:
            # Fit only on training data, trasnform both training and test data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # convert back to dataframe
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        return X_train, X_test, y_train, y_test
