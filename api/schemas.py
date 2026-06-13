from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class TrainRequest(BaseModel):
    """Request model for training a ML model."""
    model_type:str = Field(..., description="Type of ML model to train (e.g. 'random_forest)")
    dataset_name:str = Field(..., description="Name of the dataset to use for training")
    target_column:str = Field(..., description="Name of the target column in the dataset")
    test_size:float = 0.2
    random_state:int = 42

    drop_columns: list[str] = [] #Default to empty list if none provided

    #hyperparameters
    hyperparameters: Optional[Dict[str, Any]] = Field(default={}, description="Hyperparameters for the model")

    # For Auto-Tuning (GridSearchCV)
    tune_hyperparameters: bool = False


class TrainResponse(BaseModel):
    """Response model for training endpoint."""
    model_id:str
    model_type:str
    task_type:str #NEW: "classification" or "regression"
    metrics:Dict[str, Any] #Flexible dictionary for any math metrics
    message:str
    expected_features: List[str]


class PredictRequest(BaseModel):
    """Request model for making predictions."""
    model_id:str = Field(..., description="ID of the model to use for prediction")
    #Dict ensures feature names match training data perfectly
    features:Dict[str, Any] = Field(..., description="Features to use for prediction(mapped by column name)")

    


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    model_id:str
    prediction:Any
    message:str
    explanation: Optional[dict] = None #Feature importance dictionary
    cache_hit: bool


class DatasetInfo(BaseModel):
    """Information about a dataset."""
    name:str
    description:Optional[str] = None
    target_column:str #UI to know what the default target is


class ModelInfo(BaseModel):
    """Information about a model type"""
    name: str
    type: str # e.g., "Tree", "Linear", "Distance"
    requires_scaling: bool # Crucial for our DataLoader!




