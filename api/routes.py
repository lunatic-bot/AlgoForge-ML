# /train, /predict, /datasets endpoints
from fastapi import APIRouter, HTTPException
from api.schemas import TrainRequest, TrainResponse, PredictRequest, PredictResponse
from core.models.tree_models import RandomForestRunner, DecisionTreeRunner
from core.models.linear_models import LogisticRegressionRunner, LinearRegressionRunner
from core.models.distance import KNNRunner, SVMRunner
from core.data_loader import DataLoader
import pandas as pd

router = APIRouter(prefix="/api", tags=["ml"])

# Model registry
MODELS = {
    "random_forest": RandomForestRunner,
    "decision_tree": DecisionTreeRunner,
    "logistic_regression": LogisticRegressionRunner,
    "linear_regression": LinearRegressionRunner,
    "knn": KNNRunner,
    "svm": SVMRunner,
}

# Store trained models
trained_models = {}


@router.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest):
    """Train an ML model on the provided dataset."""
    try:
        # Load data
        loader = DataLoader()
        data = loader.load_csv(request.data_path)
        
        # Preprocess
        X, y = loader.preprocess(data, target_column=request.target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = loader.split(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = loader.scale_features(X_train, X_test)
        
        # Get model
        model_class = MODELS.get(request.model_type)
        if not model_class:
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model_type}")
        
        # Train
        model = model_class()
        model.fit(X_train_scaled, y_train)
        
        # Score
        score = model.score(X_test_scaled, y_test)
        
        # Store model
        model_id = f"{request.model_type}_{len(trained_models)}"
        trained_models[model_id] = model
        
        return TrainResponse(
            model_id=model_id,
            model_type=request.model_type,
            score=score,
            message="Model trained successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Make predictions using a trained model."""
    try:
        model = trained_models.get(request.model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Convert input to array
        import numpy as np
        X = np.array(request.features).reshape(1, -1)
        
        # Predict
        prediction = model.predict(X)
        
        return PredictResponse(
            model_id=request.model_id,
            prediction=prediction.tolist(),
            message="Prediction made successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
def list_datasets():
    """List available datasets."""
    return {
        "datasets": [
            {"name": "titanic", "path": "data/raw/titanic.csv"},
            {"name": "housing", "path": "data/raw/housing.csv"},
        ]
    }


@router.get("/models")
def list_models():
    """List available model types."""
    return {"models": list(MODELS.keys())}