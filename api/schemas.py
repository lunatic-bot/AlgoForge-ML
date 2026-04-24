# Pydantic models for strict data validation
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class TrainRequest(BaseModel):
    """Request model for training an ML model."""
    model_type: str = Field(..., description="Type of model to train")
    data_path: str = Field(..., description="Path to the CSV data file")
    target_column: str = Field(..., description="Name of the target column")
    test_size: float = Field(default=0.2, description="Test set proportion")
    random_state: int = Field(default=42, description="Random seed")


class TrainResponse(BaseModel):
    """Response model for training endpoint."""
    model_id: str
    model_type: str
    score: float
    message: str


class PredictRequest(BaseModel):
    """Request model for making predictions."""
    model_id: str = Field(..., description="ID of the trained model")
    features: List[float] = Field(..., description="Input features for prediction")


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    model_id: str
    prediction: List[Any]
    message: str


class DatasetInfo(BaseModel):
    """Information about a dataset."""
    name: str
    path: str
    description: Optional[str] = None


class ModelInfo(BaseModel):
    """Information about a model type."""
    name: str
    type: str
    description: Optional[str] = None