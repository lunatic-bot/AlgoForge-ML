from fastapi import APIRouter, HTTPException, UploadFile, File
import uuid
import os
import joblib
import shap
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

UPLOAD_DIR = "data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create the persistent models directory
MODELS_DIR = "models/saved"
os.makedirs(MODELS_DIR, exist_ok=True)

# state management
# MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}

AVAILABLE_MODELS ={
    # Classification
    "Random Forest Classifier": {"class" : RandomForestRunner, "requires_scaling": False, "type":"Tree", "task_type": "classification", "param_grid": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}},
    "Logistic Regression": {"class" : LogisticRegressionRunner, "requires_scaling": True, "type":"Linear", "task_type": "classification", "param_grid": {"C": [0.1, 1.0, 10.0]}},
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


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Accepts a CSV, saves it locally, and returns the column headers."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    #save the file to data/raw folder
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # read the coluumns using pandas 
    try:
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {str(e)}")
    

    return {
        "filename": file.filename,
        "columns": columns,
        "message": "Dataset uploaded successfully"
    }

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
    
    # load custom uploaded CSV
    elif request.dataset_name.endswith(".csv"):
        file_path = os.path.join(UPLOAD_DIR, request.dataset_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail=f"Uploaded dataset not found on server.")
        # Load the custom CSV directly into the dataframe
        df = pd.read_csv(file_path)

        # If the target is categorical (like text), we leave target_mapping empty for MVP 
        # (Scikit-Learn handles text targets automatically, we just won't translate them back yet)
        target_mapping = None
    
    else:
        raise HTTPException(status_code=400, detail=f"Dataset not found: {request.dataset_name}")
    
    #Only run this mapping if it's a built-in Scikit-Learn dataset
    if not request.dataset_name.endswith(".csv"):
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df[request.target_column] = data.target

    #Drop user-selected columns before any processing!
    if request.drop_columns:
        ## errors='ignore' ensures it doesn't crash if a column name is slightly wrong
        df = df.drop(columns=request.drop_columns, errors='ignore')

    model_config = AVAILABLE_MODELS[request.model_type]
    requires_scaling = model_config["requires_scaling"]
    ModelClass = model_config["class"]

    loader = MLDataLoader(target_column=request.target_column, test_size=request.test_size)
    try:
        X_train, X_test, y_train, y_test = loader.process_data(df, requires_scaling=requires_scaling)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data processing failed: {str(e)}")

    model = ModelClass(**(request.hyperparameters or {}))
    model.train(X_train, y_train, tune=request.tune_hyperparameters, 
        param_grid=model_config.get("param_grid"))
    results = model.evaluate(X_test, y_test)

    model_id = str(uuid.uuid4())

    # Sample the training data for the SHAP Explainer
    # If the dataset is smaller than 50 rows, take them all. Otherwise, take 50.
    sample_size = min(50, X_train.shape[0])
    backgroud_data = shap.sample(X_train, sample_size)

    
    # MODEL_REGISTRY[model_id] = {
    #     "model": model,
    #     "loader": loader,
    #     "requires_scaling": requires_scaling,
    #     "feature_names": list(X_train.columns),
    #     "target_mapping": target_mapping,  # Save the translation dictionary
    #     "task_type": model_config["task_type"]
    # }
    # Persistent State Saving (Writing to Disk)
    saved_state = {
        "model": model,
        "loader": loader,
        "requires_scaling": requires_scaling,
        "feature_names": list(X_train.columns),
        "target_mapping": target_mapping,  # Save the translation dictionary
        "task_type": model_config["task_type"],
        "background_data": backgroud_data,
    }

    # Create the file path and dump the dictionary to the hard drive
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    joblib.dump(saved_state, model_path)


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
    """Make a prediction using a persistently saved model."""
    # if request.model_id not in MODEL_REGISTRY:
    #     raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")
    
    # saved_state = MODEL_REGISTRY[request.model_id]

    #Search the hard drive for the model file
    model_path = os.path.join(MODELS_DIR, f"{request.model_id}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id} on server storage.")

    #Load the state back into memory
    try:
        saved_state = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model file : {str(e)}")
    
    model = saved_state["model"]
    loader = saved_state["loader"]
    requires_scaling = saved_state["requires_scaling"]
    feature_names = saved_state["feature_names"]

    try:
        input_df = pd.DataFrame([request.features])
        # Force incoming Streamlit data to be floats to prevent SHAP crashes
        input_df = input_df.astype(float) 
        
        # Ensure column order matches EXACTLY what the model expects
        input_df = input_df[feature_names]
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
    target_mapping = saved_state.get("target_mapping")
    task_type = saved_state.get("task_type", "classification") # Fallback to classification
    
    #Fork logic based on the actual algorithm's task_type
    if task_type == "classification":
        # It's Classification
        raw_pred_int = int(raw_prediction)
        # Translate to text if we have a dictionary, otherwise just return the int (0 or 1)
        if target_mapping is not None:
            final_prediction = target_mapping.get(raw_pred_int, raw_pred_int)
        else:
            final_prediction = raw_pred_int
    else:
        # It's Regression
        final_prediction = float(raw_prediction)

    # # Fork the logic based on the task type
    # if target_mapping is not None:
    #     # It's Classification
    #     raw_pred_int = int(raw_prediction)
    #     final_prediction = target_mapping.get(raw_pred_int, raw_pred_int)
    # else:
    #     # It's Regression
    #     final_prediction = float(raw_prediction)


    # Explain the prediction using SHAP
    backgroud_data = saved_state.get("background_data")
    explanation_dict = None

    try:
        if backgroud_data is not None:
            # We have the universal explainer. It figures out the math based on the model type.
            explainer = shap.Explainer(model.predict, backgroud_data)
            shap_values = explainer(input_df)
            
            # map the feature names to their absolute SHAP impact values for this specific prediction
            impact_scores = abs(shap_values.values[0])
            explanation_dict = dict(zip(feature_names, impact_scores))

    except Exception as e:
        print(f"SHAP Error: {str(e)}") #if SHAP can't handle a specific model type yet
        explaination_dict = None 

    return PredictResponse(
        model_id=request.model_id,
        prediction=final_prediction,
        message="Prediction made successfully",
        explanation=explanation_dict,
    )

    


    
