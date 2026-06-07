import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple
from sklearn.impute import SimpleImputer

# Import your centralized logger utility
from api.logger import setup_logger

# Initialize a dedicated logger for data lifecycle and engineering routines
logger = setup_logger("api.dataloader")

class MLDataLoader:
    """
    Handles data ingestion, preprocessing, and safe splitting to avoid data leakage.
    """
    def __init__(self, target_column: str, test_size: float = 0.2, random_state: int = 42):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()  # Initialized label encoder instance
        self.imputer = None
        logger.info(f"MLDataLoader engine initialized. Target feature: '{self.target_column}' | Test Split ratio: {self.test_size}")
    
    def process_data(self, df: pd.DataFrame, requires_scaling: bool = True) -> Tuple:
        """
        Processes a pandas DataFrame and returns safe train/test splits.
        
        Args:
            df: The raw pandas DataFrame.
            requires_scaling: True for distance/linear models, False for tree models.
        """
        logger.info(f"Data engineering pipeline started. Matrix dimensions: {df.shape[0]} rows x {df.shape[1]} columns. Requires feature scaling: {requires_scaling}")

        # 1. Separate Features (X) and Target (y)
        if self.target_column not in df.columns:
            logger.error(f"Data separation aborted: target column '{self.target_column}' is missing from feature matrix keys.")
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
    
        try:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            logger.debug("Successfully partitioned feature space vector from target labels.")
        except Exception as e:
            logger.critical(f"Unexpected split fault separating feature columns: {str(e)}", exc_info=True)
            raise RuntimeError(f"Core data matrix splitting partition failure: {str(e)}")

        # 2. Advanced Imputation (Handling missing real-world data)
        try:
            numeric_cols = X.select_dtypes(include=['number']).columns
            missing_count = X[numeric_cols].isnull().sum().sum()
            
            if len(numeric_cols) > 0 and missing_count > 0:
                logger.info(f"Found {missing_count} missing entries across numeric features. Executing SimpleImputer median filling matrix strategy.")
                self.imputer = SimpleImputer(strategy='median')
                X_imputed = self.imputer.fit_transform(X[numeric_cols])
                X.loc[:, numeric_cols] = X_imputed
                logger.debug("Numeric entries imputed completely.")
            else:
                logger.debug("Data imputation stage bypassed: No missing data values observed in feature arrays.")
        except Exception as imp_err:
            logger.error(f"Imputation pipeline crashed fitting attributes: {str(imp_err)}", exc_info=True)
            raise RuntimeError(f"Feature matrix imputer execution crash: {str(imp_err)}")

        # 3. Handle Categorical Features (One-Hot Encoding)
        try:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            if len(categorical_cols) > 0:
                logger.info(f"Categorical features detected: {categorical_cols}. Transforming categories via One-Hot structural dummy variables.")
            
            X = pd.get_dummies(X, drop_first=True)
            # Forcing all columns (including new True/False dummies) to be strictly numeric floats
            X = X.astype(float)
            logger.debug(f"Feature transformation complete. Expanded matrix layout width: {X.shape[1]} properties.")
        except Exception as cat_err:
            logger.error(f"One-Hot dummy matrix layout compilation failed: {str(cat_err)}", exc_info=True)
            raise RuntimeError(f"Categorical encoding structure pipeline failure: {str(cat_err)}")

        # 4. Encode the Target Variable (if it's text like 'Yes'/'No')
        try:
            if y.dtype == 'object' or y.dtype.name == 'category':
                distinct_labels = y.dropna().unique().tolist()
                logger.info(f"Categorical target label detected (Type: {y.dtype}). Processing class mappings using label encoder. Labels found: {distinct_labels}")
                # FIX: Mapped to use self.label_encoder which was defined in __init__
                y = pd.Series(self.label_encoder.fit_transform(y), index=y.index)
                logger.debug("Target categories encoded successfully.")
        except Exception as target_err:
            logger.error(f"Target variable label mapping failed transformation: {str(target_err)}", exc_info=True)
            raise RuntimeError(f"Target encoder configuration failure: {str(target_err)}")

        # 5. Train-Test Split (CRITICAL: Do this BEFORE scaling)
        try:
            logger.info(f"Slicing matrix arrays into train/test states. Validation holdout target: {self.test_size * 100}%")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            logger.info(f"Data separation successful. Training set shape: {X_train.shape[0]} samples | Holdout subset shape: {X_test.shape[0]} samples")
        except Exception as split_err:
            logger.critical(f"Array separation phase collapsed: {str(split_err)}", exc_info=True)
            raise RuntimeError(f"Train/Test index slicing failure: {str(split_err)}")

        # 6. Scale features (if required)
        if requires_scaling:
            try:
                logger.info("Executing standard Z-Score normalization scaling transformations over split array dimensions.")
                # Fit only on training data, transform both training and test data
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                # convert back to dataframe
                X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
                X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
                logger.debug("Standard scaling calculations computed and mapped back to dataframes safely.")
            except Exception as scale_err:
                logger.error(f"Z-Score scaler computation collapsed transforming feature parameters: {str(scale_err)}", exc_info=True)
                raise RuntimeError(f"Feature scaling pipeline computation failure: {str(scale_err)}")

        logger.info("Data loader pipeline executed completely. Passing generated frames back to operational runtime environment.")
        return X_train, X_test, y_train, y_test



# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from typing import Tuple
# from sklearn.impute import SimpleImputer

# class MLDataLoader:
#     """
#     Handles data ingestion, preprocessing, and safe splitting to avoid data leakage.
#     """
#     def __init__(self, target_column: str, test_size: float = 0.2, random_state: int = 42):
#         self.target_column = target_column
#         self.test_size = test_size
#         self.random_state = random_state
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()
#         self.imputer = None
    
#     def process_data(self, df: pd.DataFrame, requires_scaling: bool = True)->Tuple:
#         """
#         Processes a pandas DataFrame and returns safe train/test splits.
        
#         Args:
#             df: The raw pandas DataFrame.
#             requires_scaling: True for distance/linear models, False for tree models.
#         """

#         # # 1. Basic Imputation (Dropping missing values for MVP, can upgrade to SimpleImputer later)
#         # df_clean = df.dropna().copy()

#         # 1. Separate Features (X) and Target (y)
#         if self.target_column not in df.columns:
#             raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
    
#         X = df.drop(columns=[self.target_column])
#         y = df[self.target_column]

#         # 2. Advanced Imputation (Handling missing real-world data)
#         # We isolate numeric columns to fill missing numbers with the median
#         numeric_cols = X.select_dtypes(include=['number']).columns
#         if len(numeric_cols) > 0:
#             self.imputer = SimpleImputer(strategy='median')
#             # Fit and transform the data, then rebuild the DataFrame
#             X_imputed = self.imputer.fit_transform(X[numeric_cols])
#             X.loc[:, numeric_cols] = X_imputed

#         # 3. Handle Categorical Features (One-Hot Encoding)
#         # Converts text columns into 0s and 1s so math algorithms can process them
#         X = pd.get_dummies(X, drop_first=True)
#         # Forceing all columns (including new True/False dummies) to be strictly numeric floats
#         X = X.astype(float)

#         # 4. Encode the Target Variable (if it's text like 'Yes'/'No')
#         if y.dtype == 'object' or y.dtype.name == 'category':
#             y = pd.Series(self.target_encoder.fit_transform(y), index=y.index)

#         # 5. Train-Test Split (CRITICAL: Do this BEFORE scaling)
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=self.test_size, random_state=42
#         )

#         # 6. Scale features(if required)
#         if requires_scaling:
#             # Fit only on training data, trasnform both training and test data
#             X_train_scaled = self.scaler.fit_transform(X_train)
#             X_test_scaled = self.scaler.transform(X_test)

#             # convert back to dataframe
#             X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
#             X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

#         return X_train, X_test, y_train, y_test
