from fastapi import APIRouter, HTTPException
import uuid
import pandas as pd
from typing import List, Dict, Any 

from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes


# Import custom architecture
from core.data_loader import MLDataLoader
from core.models.tree_models import RandomForestRunner, RandomForestRegressorRunner
from core.models.linear_models import LogisticRegressionRunner, LinearRegressionRunner
from core.models.distance import KNNRunner, SVMRunner, SVRRunner

# Import schemas

from api.schemas import (
    TrainRequest, TrainResponse, PredictRequest, 
    PredictResponse, DatasetInfo, ModelInfo)

# initialize the router
router = APIRouter()

# state management
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}

AVAILABLE_MODELS ={
    # Classification
    "Random Forest Classifier": {"class" : RandomForestRunner, "requires_scaling": False, "type":"Tree", "task_type": "classification"},
    "Logistic Regression": {"class" : LogisticRegressionRunner, "requires_scaling": True, "type":"Linear", "task_type": "classification"},
    "KNN Classifier" : {"class" : KNNRunner, "requires_scaling": True, "type":"Distance", "task_type": "classification"},
    "SVM Classifier" : {"class" : SVMRunner, "requires_scaling": True, "type":"Distance", "task_type": "classification"},   

    # Regression
    "Random Forest Regressor": {"class": RandomForestRegressorRunner, "requires_scaling": False, "type": "Tree", "task_type": "regression"},
    "Linear Regression": {"class": LinearRegressionRunner, "requires_scaling": True, "type": "Linear", "task_type": "regression"},
    "SVR (Support Vector Regressor)": {"class": SVRRunner, "requires_scaling": True, "type": "Distance", "task_type": "regression"}
}

#Endpoints
@router.get("/datasets", response_model=List[DatasetInfo])
def get_datasets():
    return [
        DatasetInfo(name="iris", description="Multiclass classification (Flower types)", target_column="target"),
        DatasetInfo(name="breast_cancer", description="Binary classification (Cancer detection)", target_column="target"),
        # Regression
        DatasetInfo(name="diabetes", description="Regression (Disease progression)", target_column="target")
    ]

@router.get("/models", response_model=List[ModelInfo])
def get_models():
    return [
        ModelInfo(name=name, type=info["type"], requires_scaling=info["requires_scaling"]) for name, info in AVAILABLE_MODELS.items()
    ]


@router.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest):
    if request.model_type not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}")
    
    if request.dataset_name == "breast_cancer":
        data = load_breast_cancer()
        # Creates {0: 'malignant', 1: 'benign'}
        target_mapping = {i: str(name) for i, name in enumerate(data.target_names)}
    elif request.dataset_name == "iris":
        data = load_iris()
        # Creates {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        target_mapping = {i: str(name) for i, name in enumerate(data.target_names)}
    elif request.dataset_name == "diabetes":
        data = load_diabetes()
        target_mapping = None # Regression doesn't use text labels
    else:
        raise HTTPException(status_code=400, detail=f"Dataset not found: {request.dataset_name}")
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[request.target_column] = data.target

    model_config = AVAILABLE_MODELS[request.model_type]
    requires_scaling = model_config["requires_scaling"]
    ModelClass = model_config["class"]

    loader = MLDataLoader(target_column=request.target_column, test_size=request.test_size)
    try:
        X_train, X_test, y_train, y_test = loader.process_data(df, requires_scaling=requires_scaling)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data processing failed: {str(e)}")

    model = ModelClass(**(request.hyperparameters or {}))
    model.train(X_train, y_train)
    results = model.evaluate(X_test, y_test)

    model_id = str(uuid.uuid4())
    MODEL_REGISTRY[model_id] = {
        "model": model,
        "loader": loader,
        "requires_scaling": requires_scaling,
        "feature_names": list(X_train.columns),
        "target_mapping": target_mapping  # Save the translation dictionary
    }

    return TrainResponse(
        model_id=model_id,
        model_type=request.model_type,
        task_type=model_config["task_type"], # task type
        metrics=results, #dictionary from evaluate()
        message="Model trained and saved successfully",
        expected_features=list(X_train.columns)

    )


@router.post("/predict", response_model=PredictResponse)
def make_prediction(request: PredictRequest):
    if request.model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")
    
    saved_state = MODEL_REGISTRY[request.model_id]
    model = saved_state["model"]
    loader = saved_state["loader"]
    requires_scaling = saved_state["requires_scaling"]
    feature_names = saved_state["feature_names"]

    try:
        input_df = pd.DataFrame([request.features], columns=feature_names)
    except KeyError:
        raise HTTPException(status_code=400, detail="Provided features do not match training data.")
    
    if requires_scaling:
        # The scaler strips the column names and returns a NumPy array
        scaled_values = loader.scaler.transform(input_df)
        # FIX: We must wrap that raw array BACK into a Pandas DataFrame 
        # using the exact column names the model is expecting!
        input_df = pd.DataFrame(scaled_values, columns=feature_names)


    
    # model.predict returns an array like [0]. Grabing that first item and make it an integer.
    raw_prediction = int(model.predict(input_df)[0]) 
    
    # Grab the translation dictionary saved during training
    target_mapping = saved_state.get("target_mapping", {})
    

    # Fork the logic based on the task type
    if target_mapping is not None:
        # It's Classification
        raw_pred_int = int(raw_prediction)
        final_prediction = target_mapping.get(raw_pred_int, raw_pred_int)
    else:
        # It's Regression
        final_prediction = float(raw_prediction)

    return PredictResponse(
        model_id=request.model_id,
        prediction=final_prediction,
        message="Prediction made successfully"
    )

    


    
