import pytest
import pandas as pd
import numpy as np
from core.data_loader import MLDataLoader

def test_data_loader_imputation_and_scaling():
    """Test that the DataLoader correctly fills missing values and scales data."""
    # 1. Creating a fake dataset with a missing value (NaN)
    data = {
        "feature1": [1.0, 2.0, np.nan, 4.0, 5.0], # Median is 3.0
        "feature2": ["A", "B", "A", "B", "A"],    # Text column to test get_dummies
        "target": [0, 1, 0, 1, 0]
    }

    df = pd.DataFrame(data)
    
    loader = MLDataLoader(target_column="target", test_size=0.2)

    # 2. Process the data (requires_scaling=True)
    X_train, X_test, y_train, y_test = loader.process_data(df, requires_scaling=True)

    # 3. Assertions (the tests)
    # Check that get_dummies worked (feature2_B should exist)
    assert "feature2_B" in X_train.columns

    # Check that imputation worked (No NaN values should remain)
    assert not X_train.isnull().values.any()
    assert not X_test.isnull().values.any()

    # Check that the test split worked correctly (5 rows * 0.2 = 1 row in test)
    assert len(X_test) == 1
    assert len(X_train) == 4