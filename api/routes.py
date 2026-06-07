import os
import uuid
import json
import glob
import joblib
from datetime import datetime
from typing import List
import pandas as pd
import shap
import mlflow
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status

# Assuming these schemas and dependencies exist in your setup
from .schemas import DatasetInfo, ModelInfo, TrainRequest, TrainResponse, PredictRequest, PredictResponse
from .auth import get_current_user
from .cache import generate_cache_key, get_cached_prediction, set_cached_prediction

from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes

# Import custom architecture
from core.data_loader import MLDataLoader
from core.models.tree_models import RandomForestRunner, RandomForestRegressorRunner
from core.models.linear_models import LogisticRegressionRunner, LinearRegressionRunner
from core.models.distance import KNNRunner, SVMRunner, SVRRunner

# Centralized logger import
from .logger import setup_logger


# Initialize router-specific logger
logger = setup_logger("api.routes")

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

@router.get("/models/history")
def get_model_history():
    """Returns a list of all historically trained models and their metadata."""
    logger.info("Fetching historical model metadata records.")
    history = []

    search_pattern = os.path.join(MODELS_DIR, "*_meta.json")
    meta_files = glob.glob(search_pattern)
    logger.debug(f"Found {len(meta_files)} potential metadata files on disk storage.")

    for file_path in meta_files:
        try:
            with open(file_path, "r") as f:
                meta = json.load(f)
                history.append(meta)
        except json.JSONDecodeError as jde:
            logger.error(f"Skipping corrupted JSON metadata file {file_path}: {str(jde)}")
        except IOError as ioe:
            logger.error(f"Disk read access failure for file {file_path}: {str(ioe)}")
        except Exception as e:
            logger.error(f"Unexpected error processing metadata file {file_path}: {str(e)}", exc_info=True)

    # sort the list so that the newest models appear on the top
    try:
        history.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    except Exception as sort_err:
        logger.warning(f"Failed sorting metadata history stream by date field: {str(sort_err)}")

    logger.info(f"Successfully compiled model history. Returning {len(history)} records.")
    return {"history": history}


@router.get("/datasets", response_model=List[DatasetInfo])
def get_datasets():
    logger.debug("Datasets options endpoint accessed.")
    return [
        DatasetInfo(name="iris", description="Multiclass classification (Flower types)", target_column="target"),
        DatasetInfo(name="breast_cancer", description="Binary classification (Cancer detection)", target_column="target"),
        DatasetInfo(name="diabetes", description="Regression (Disease progression)", target_column="target")
    ]


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Accepts a CSV, saves it locally, and returns the column headers."""
    logger.info(f"Dataset upload request received. Filename: {file.filename}")
    
    if not file.filename.endswith(".csv"):
        logger.warning(f"Rejected upload: File '{file.filename}' is not a valid CSV.")
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # save the file to data/raw folder
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        logger.info(f"File system write complete. Saved: {file_path}")
    except IOError as io_err:
        logger.critical(f"Disk write failure during custom CSV upload operation: {str(io_err)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server disk IO write failure.")

    # read the columns using pandas 
    try:
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()
        logger.info(f"CSV read successfully. Columns found: {len(columns)} rows parsed: {df.shape[0]}")
    except Exception as e:
        logger.error(f"Pandas processing failed for file '{file.filename}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {str(e)}")
    
    return {
        "filename": file.filename,
        "columns": columns,
        "message": "Dataset uploaded successfully"
    }


@router.get("/models", response_model=List[ModelInfo])
def get_models():
    logger.debug("Algorithms lookup directory endpoint accessed.")
    return [
        ModelInfo(name=name, type=info["type"], requires_scaling=info["requires_scaling"]) 
        for name, info in AVAILABLE_MODELS.items()
    ]


@router.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest, current_user: dict = Depends(get_current_user)):
    """Trains an ML Model, tracks metrics via MLflow, and logs pipeline state."""
    username = current_user.username if current_user else "unknown"
    logger.info(f"User '{username}' initiated train run. Algorithm: {request.model_type} | Dataset: {request.dataset_name}")

    # 1. Setup MLflow with tracking safeguards
    try:
        tracking_uri = "http://host.docker.internal:5000"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(f"AlgoForge_{request.model_type}")
        logger.info(f"Connected to MLflow server at {tracking_uri} | Experiment: AlgoForge_{request.model_type}")
    except Exception as mlflow_connect_err:
        logger.error(f"MLflow connection handshake failed: {str(mlflow_connect_err)}", exc_info=True)
        raise HTTPException(
            status_code=503, 
            detail="Tracking infrastructure unreachable. Train sequence halted to protect experiment telemetry metrics."
        )

    # 2. Algorithm & Dataset Validation
    if request.model_type not in AVAILABLE_MODELS:
        logger.warning(f"Training aborted. Invalid algorithm selection requested: {request.model_type}")
        raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}")
    
    try:
        if request.dataset_name == "breast_cancer":
            data = load_breast_cancer()
            target_mapping = {i: str(name) for i, name in enumerate(data.target_names)}
        elif request.dataset_name == "iris":
            data = load_iris()
            target_mapping = {i: str(name) for i, name in enumerate(data.target_names)}
        elif request.dataset_name == "diabetes":
            data = load_diabetes()
            target_mapping = None 
        elif request.dataset_name.endswith(".csv"):
            file_path = os.path.join(UPLOAD_DIR, request.dataset_name)
            if not os.path.exists(file_path):
                logger.warning(f"Training aborted. Target custom upload missing from volume: {file_path}")
                raise HTTPException(status_code=400, detail=f"Uploaded dataset not found on server.")
            df = pd.read_csv(file_path)
            target_mapping = None
        else:
            logger.warning(f"Training aborted. Unrecognized data mapping directive: {request.dataset_name}")
            raise HTTPException(status_code=400, detail=f"Dataset not found: {request.dataset_name}")

        if not request.dataset_name.endswith(".csv"):
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df[request.target_column] = data.target

    except HTTPException:
        raise
    except FileNotFoundError as fnf:
        logger.error(f"Disk tracking lookup error reading data block: {str(fnf)}", exc_info=True)
        raise HTTPException(status_code=404, detail="Data target missing from engine storage volume matrix.")
    except Exception as data_prep_err:
        logger.error(f"General breakdown during data collection mapping step: {str(data_prep_err)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Data parsing error environment side: {str(data_prep_err)}")

    # Drop user-selected columns before any processing!
    if request.drop_columns:
        logger.info(f"Dropping explicit features from frame: {request.drop_columns}")
        df = df.drop(columns=request.drop_columns, errors='ignore')

    model_config = AVAILABLE_MODELS[request.model_type]
    requires_scaling = model_config["requires_scaling"]
    ModelClass = model_config["class"]

    # 3. Model Pipeline Execution Tracking
    loader = MLDataLoader(target_column=request.target_column, test_size=request.test_size)
    try:
        X_train, X_test, y_train, y_test = loader.process_data(df, requires_scaling=requires_scaling)
        logger.info(f"Data separation successful. Train shapes: {X_train.shape} | Test shapes: {X_test.shape}")
    except Exception as e:
        logger.error(f"Data engineering preprocessing framework crashed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Data processing failed: {str(e)}")
    
    try:
        logger.info("Executing Scikit-Learn training execution cycle wrapped inside active MLflow run block.")
        with mlflow.start_run():
            mlflow.log_param("model_type", request.model_type)
            mlflow.log_param("test_size", request.test_size)
            mlflow.log_param("dataset", request.dataset_name)

            if request.hyperparameters:
                mlflow.log_params(request.hyperparameters)

            model = ModelClass(**(request.hyperparameters or {}))
            model.train(X_train, y_train, tune=request.tune_hyperparameters, param_grid=model_config.get("param_grid"))
            results = model.evaluate(X_test, y_test)

            flat_metrics = {k: v for k, v in results.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(flat_metrics)
            mlflow.sklearn.log_model(sk_model=model.model, artifact_path="model")
            logger.info("Successfully recorded experiment tracking run metrics and artifacts into MLflow storage server.")

    except Exception as mlflow_e:
        # Non-blocking infrastructure tracking error warning
        logger.error(f"MLflow lifecycle recording failed during train telemetry phase: {mlflow_e}", exc_info=True)

    # 4. Serialization and Metadata Disk Persistence
    model_id = str(uuid.uuid4())
    logger.info(f"Pipeline executed successfully. Generating asset identity signature ID: {model_id}")

    try:
        sample_size = min(50, X_train.shape[0])
        backgroud_data = shap.sample(X_train, sample_size)
        
        saved_state = {
            "model": model,
            "loader": loader,
            "requires_scaling": requires_scaling,
            "feature_names": list(X_train.columns),
            "target_mapping": target_mapping,  
            "task_type": model_config["task_type"],
            "background_data": backgroud_data,
        }

        metadata = {
            "model_id": model_id,
            "algorithm": request.model_type,
            "dataset": request.dataset_name,
            "task_type": model_config["task_type"],
            "metrics": results,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        meta_path = os.path.join(MODELS_DIR, f"{model_id}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
        joblib.dump(saved_state, model_path)
        logger.info(f"Model state binaries and structural json metadata successfully saved to storage volumes. Model ID: {model_id}")

    except IOError as io_store_err:
        logger.critical(f"Storage allocation exception serialization write block: {str(io_store_err)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server block allocation breakdown: Failed serialization of final training state.")

    return TrainResponse(
        model_id=model_id,
        model_type=request.model_type,
        task_type=model_config["task_type"], 
        metrics=results, 
        message="Model trained and saved successfully(Logged to Mlflow).",
        expected_features=list(X_train.columns)
    )


@router.post("/predict", response_model=PredictResponse)
def make_prediction(request: PredictRequest, current_user: dict = Depends(get_current_user)):
    """Make a prediction using a persistently saved model, guarded by OAuth2 and optimized with Redis Caching."""
    username = current_user.username if current_user else "unknown"
    logger.info(f"User '{username}' submitted observation payload against model ID: {request.model_id}")
    
    # 1. Redis Cache Lookup Strategy
    try:
        cache_key = generate_cache_key(request.model_id, request.features)
        cached_result = get_cached_prediction(cache_key)
        if cached_result:
            logger.info(f"🚀 Redis Cache Hit intercept! Returning compiled inference state instantly for key: {cache_key}")
            cached_result["cache_hit"] = True
            return cached_result
        logger.info(f"🐢 Redis Cache Miss for key: {cache_key}. Proceeding to disk evaluation path matrix.")
    except Exception as redis_err:
        # Fall-through resilience rule: if Redis container goes missing or breaks, catch it silently and evaluate via CPU
        logger.error(f"Redis memory tracking layer connection error: {str(redis_err)}. Proceeding with raw processing pipeline loop safely.")

    # 2. Storage Check and Joblib Deserialization
    model_path = os.path.join(MODELS_DIR, f"{request.model_id}.joblib")
    if not os.path.exists(model_path):
        logger.warning(f"Inference prediction aborted. Model identifier path asset target not found: {model_path}")
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id} on server storage.")

    try:
        saved_state = joblib.load(model_path)
        logger.debug(f"Successfully loaded joblib model state wrapper into engine RAM memory space.")
    except Exception as e:
        logger.critical(f"Binary object graph construction failure during model package instantiation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model file : {str(e)}")
    
    model = saved_state["model"]
    loader = saved_state["loader"]
    requires_scaling = saved_state["requires_scaling"]
    feature_names = saved_state["feature_names"]

    # 3. Data Transformation and Processing Pipeline
    try:
        input_df = pd.DataFrame([request.features])
        input_df = input_df.astype(float) 
        input_df = input_df[feature_names]
    except KeyError as missing_feat_err:
        logger.warning(f"Bad Request: Input features array missing required model signature elements. Details: {str(missing_feat_err)}")
        raise HTTPException(status_code=400, detail="Provided features matrix configuration doesn't match baseline model layout parameters.")
    except ValueError as val_cast_err:
        logger.warning(f"Bad Request: Input properties translation failed typing schema casts. Details: {str(val_cast_err)}")
        raise HTTPException(status_code=400, detail="Feature properties casting exception: Ensure values can translate to real floats.")

    if requires_scaling:
        try:
            logger.debug("Applying normalization transformation transformations onto feature row matrix vectors.")
            scaled_values = loader.scaler.transform(input_df)
            input_df = pd.DataFrame(scaled_values, columns=feature_names)
        except Exception as scale_err:
            logger.error(f"Scaler calculation transformer step failed processing input inputs: {str(scale_err)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal inference calculation processing transformation engine crash.")

    # 4. Core Prediction Calculations
    try:
        raw_prediction = int(model.predict(input_df)[0]) 
        target_mapping = saved_state.get("target_mapping")
        task_type = saved_state.get("task_type", "classification")
        
        if task_type == "classification":
            raw_pred_int = int(raw_prediction)
            if target_mapping is not None:
                final_prediction = target_mapping.get(raw_pred_int, raw_pred_int)
            else:
                final_prediction = raw_pred_int
        else:
            final_prediction = float(raw_prediction)
        logger.info(f"Prediction execution calculation successful. Raw inference index: {raw_prediction} | Transformed score output value: {final_prediction}")
    except Exception as infer_err:
        logger.critical(f"Mathematical execution fault inside model prediction block function layer: {str(infer_err)}", exc_info=True)
        raise HTTPException(status_code=500, detail="ML Model execution calculations processing failure.")

    # 5. SHAP Explainability Engine Generation Block
    backgroud_data = saved_state.get("background_data")
    explanation_dict = None

    try:
        if backgroud_data is not None:
            logger.debug("Generating prediction local attribution explainability impacts array matrix utilizing SHAP kernel explainer objects.")
            raw_estimator = model.model if hasattr(model, "model") else model
            explainer = shap.Explainer(raw_estimator, backgroud_data)
            shap_values = explainer(input_df)
        
            if len(shap_values.values.shape) == 3:
                impact_scores = abs(shap_values.values[0, :, int(raw_prediction)])
            else:
                impact_scores = abs(shap_values.values[0])
            
            explanation_dict = dict(zip(feature_names, impact_scores.tolist()))
            logger.debug("SHAP explanation vectors compiled successfully.")
    except Exception as shap_crash_err:
        # Non-blocking explainability error capture
        logger.error(f"Attribution analysis handling collapsed during calculation: {str(shap_crash_err)}", exc_info=True)
        explanation_dict = None 

    # 6. Response Construction and Cache Populating Execution
    response_payload = {
        "model_id": request.model_id,
        "prediction": final_prediction,
        "message": "Prediction made successfully",
        "explanation": explanation_dict,
        "cache_hit": False 
    }

    try:
        set_cached_prediction(cache_key, response_payload, expire_seconds=3600)
        logger.info(f"Inference record saved to Redis cluster for key: {cache_key} with an eviction TTL profile of 1 hour.")
    except Exception as redis_write_err:
        logger.error(f"Redis memory caching engine persistence execution error caught: {str(redis_write_err)}")

    return response_payload




# from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
# import uuid
# import os
# import joblib
# import json
# import glob
# import shap
# import mlflow
# import mlflow.sklearn
# from datetime import datetime
# import pandas as pd
# from typing import List, Dict, Any 

# from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes


# # Import custom architecture
# from core.data_loader import MLDataLoader
# from core.models.tree_models import RandomForestRunner, RandomForestRegressorRunner
# from core.models.linear_models import LogisticRegressionRunner, LinearRegressionRunner
# from core.models.distance import KNNRunner, SVMRunner, SVRRunner


# from .auth import get_current_user
# from .cache import generate_cache_key, get_cached_prediction, set_cached_prediction

# # Import schemas

# from api.schemas import (
#     TrainRequest, TrainResponse, PredictRequest, 
#     PredictResponse, DatasetInfo, ModelInfo)


# from .logger import setup_logger

# logger = setup_logger("api.routes")


# # initialize the router
# router = APIRouter()

# UPLOAD_DIR = "data/raw"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Create the persistent models directory
# MODELS_DIR = "models/saved"
# os.makedirs(MODELS_DIR, exist_ok=True)

# # state management
# # MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}

# AVAILABLE_MODELS ={
#     # Classification
#     "Random Forest Classifier": {"class" : RandomForestRunner, "requires_scaling": False, "type":"Tree", "task_type": "classification", "param_grid": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}},
#     "Logistic Regression": {"class" : LogisticRegressionRunner, "requires_scaling": True, "type":"Linear", "task_type": "classification", "param_grid": {"C": [0.1, 1.0, 10.0]}},
#     "KNN Classifier" : {"class" : KNNRunner, "requires_scaling": True, "type":"Distance", "task_type": "classification"},
#     "SVM Classifier" : {"class" : SVMRunner, "requires_scaling": True, "type":"Distance", "task_type": "classification"},   

#     # Regression
#     "Random Forest Regressor": {"class": RandomForestRegressorRunner, "requires_scaling": False, "type": "Tree", "task_type": "regression"},
#     "Linear Regression": {"class": LinearRegressionRunner, "requires_scaling": True, "type": "Linear", "task_type": "regression"},
#     "SVR (Support Vector Regressor)": {"class": SVRRunner, "requires_scaling": True, "type": "Distance", "task_type": "regression"}
# }


# @router.get("/models/history")
# def get_model_history():
#     """Returns a list of all historically trained models and their metadata."""
#     history = []

#     # find all paths ending with meta.json in models directory
#     search_pattern = os.path.join(MODELS_DIR, "*_meta.json")

#     for file_path in glob.glob(search_pattern):
#         try:
#             with open(file_path, "r") as f:
#                 meta = json.load(f)
#                 history.append(meta)
#         except Exception as e:
#             print(f"Skipping corrupted metadata file {file_path}: {str(e)}")

#     # sort the list so that the newest models appear on the top
#     history.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
#     return {"history": history}


# #Endpoints
# @router.get("/datasets", response_model=List[DatasetInfo])
# def get_datasets():
#     return [
#         DatasetInfo(name="iris", description="Multiclass classification (Flower types)", target_column="target"),
#         DatasetInfo(name="breast_cancer", description="Binary classification (Cancer detection)", target_column="target"),
#         # Regression
#         DatasetInfo(name="diabetes", description="Regression (Disease progression)", target_column="target")
#     ]


# @router.post("/upload")
# async def upload_dataset(file: UploadFile = File(...)):
#     """Accepts a CSV, saves it locally, and returns the column headers."""
#     if not file.filename.endswith(".csv"):
#         raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
#     file_path = os.path.join(UPLOAD_DIR, file.filename)

#     #save the file to data/raw folder
#     with open(file_path, "wb") as buffer:
#         content = await file.read()
#         buffer.write(content)

#     # read the coluumns using pandas 
#     try:
#         df = pd.read_csv(file_path)
#         columns = df.columns.tolist()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to read CSV: {str(e)}")
    

#     return {
#         "filename": file.filename,
#         "columns": columns,
#         "message": "Dataset uploaded successfully"
#     }

# @router.get("/models", response_model=List[ModelInfo])
# def get_models():
#     return [
#         ModelInfo(name=name, type=info["type"], requires_scaling=info["requires_scaling"]) for name, info in AVAILABLE_MODELS.items()
#     ]


# @router.post("/train", response_model=TrainResponse)
# def train_model(request: TrainRequest, current_user: dict = Depends(get_current_user)):

#     # 1. Setup MLflow
#     # mlflow.set_tracking_uri("http://mlflow:5000")
#     mlflow.set_tracking_uri("http://host.docker.internal:5000")
#     mlflow.set_experiment(f"AlgoForge_{request.model_type}")

    
#     if request.model_type not in AVAILABLE_MODELS:
#         raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}")
    
#     if request.dataset_name == "breast_cancer":
#         data = load_breast_cancer()
#         # Creates {0: 'malignant', 1: 'benign'}
#         target_mapping = {i: str(name) for i, name in enumerate(data.target_names)}
    
#     elif request.dataset_name == "iris":
#         data = load_iris()
#         # Creates {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
#         target_mapping = {i: str(name) for i, name in enumerate(data.target_names)}
    
#     elif request.dataset_name == "diabetes":
#         data = load_diabetes()
#         target_mapping = None # Regression doesn't use text labels
    
#     # load custom uploaded CSV
#     elif request.dataset_name.endswith(".csv"):
#         file_path = os.path.join(UPLOAD_DIR, request.dataset_name)
#         if not os.path.exists(file_path):
#             raise HTTPException(status_code=400, detail=f"Uploaded dataset not found on server.")
#         # Load the custom CSV directly into the dataframe
#         df = pd.read_csv(file_path)

#         # If the target is categorical (like text), we leave target_mapping empty for MVP 
#         # (Scikit-Learn handles text targets automatically, we just won't translate them back yet)
#         target_mapping = None
    
#     else:
#         raise HTTPException(status_code=400, detail=f"Dataset not found: {request.dataset_name}")
    
#     #Only run this mapping if it's a built-in Scikit-Learn dataset
#     if not request.dataset_name.endswith(".csv"):
#         df = pd.DataFrame(data.data, columns=data.feature_names)
#         df[request.target_column] = data.target

#     #Drop user-selected columns before any processing!
#     if request.drop_columns:
#         ## errors='ignore' ensures it doesn't crash if a column name is slightly wrong
#         df = df.drop(columns=request.drop_columns, errors='ignore')

#     model_config = AVAILABLE_MODELS[request.model_type]
#     requires_scaling = model_config["requires_scaling"]
#     ModelClass = model_config["class"]

#     loader = MLDataLoader(target_column=request.target_column, test_size=request.test_size)
#     try:
#         X_train, X_test, y_train, y_test = loader.process_data(df, requires_scaling=requires_scaling)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Data processing failed: {str(e)}")
    
#     try:
#         # create a new run
#         with mlflow.start_run():
#             # Log Hyperparameters
#             mlflow.log_param("model_type", request.model_type)
#             mlflow.log_param("test_size", request.test_size)
#             mlflow.log_param("dataset", request.dataset_name)

#             # Log any specific hyperparameters if they exist
#             if request.hyperparameters:
#                 mlflow.log_params(request.hyperparameters)

#             model = ModelClass(**(request.hyperparameters or {}))
#             model.train(X_train, y_train, tune=request.tune_hyperparameters, 
#                 param_grid=model_config.get("param_grid"))
#             results = model.evaluate(X_test, y_test)

#             # 3. Log Metrics (Handles both classification and regression)
#             # MLflow's log_metrics expects a dict of {name: float}
#             # We filter out the 'detailed_report' because it's a nested dict (MLflow doesn't like those in metrics)
#             flat_metrics = {k: v for k, v in results.items() if isinstance(v, (int, float))}
#             mlflow.log_metrics(flat_metrics)

#             # 4. Log the Model Artifact (The actual .joblib file)
#             # This allows you to download the model directly from the MLflow UI!
#             mlflow.sklearn.log_model(sk_model=model.model, artifact_path="model")

#     except Exception as mlflow_e:
#         print(f"MLflow Logging Failed: {mlflow_e}")

#     model_id = str(uuid.uuid4())

#     # Sample the training data for the SHAP Explainer
#     # If the dataset is smaller than 50 rows, take them all. Otherwise, take 50.
#     sample_size = min(50, X_train.shape[0])
#     backgroud_data = shap.sample(X_train, sample_size)

    

#     # Persistent State Saving (Writing to Disk)
#     saved_state = {
#         "model": model,
#         "loader": loader,
#         "requires_scaling": requires_scaling,
#         "feature_names": list(X_train.columns),
#         "target_mapping": target_mapping,  # Save the translation dictionary
#         "task_type": model_config["task_type"],
#         "background_data": backgroud_data,
#     }

#     metadata = {
#         "model_id": model_id,
#         "algorithm": request.model_type,
#         "dataset": request.dataset_name,
#         "task_type": model_config["task_type"],
#         "metrics": results,
#         "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     }

#     meta_path = os.path.join(MODELS_DIR, f"{model_id}_meta.json")
#     with open(meta_path, "w") as f:
#         json.dump(metadata, f)

#     # Create the file path and dump the dictionary to the hard drive
#     model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
#     joblib.dump(saved_state, model_path)


#     return TrainResponse(
#         model_id=model_id,
#         model_type=request.model_type,
#         task_type=model_config["task_type"], # task type
#         metrics=results, #dictionary from evaluate()
#         message="Model trained and saved successfully(Logged to Mlflow).",
#         expected_features=list(X_train.columns)

#     )

# # Ensure your PredictResponse Pydantic model includes a cache_hit: bool field!
# @router.post("/predict", response_model=PredictResponse)
# def make_prediction(request: PredictRequest, current_user: dict = Depends(get_current_user)):
#     """Make a prediction using a persistently saved model, guarded by OAuth2 and optimized with Redis Caching."""
    
#     # 1. Generate a deterministic cache key based on model and current user input features
#     cache_key = generate_cache_key(request.model_id, request.features)
    
#     # 2. Check the Redis Cache layer first (Cache-Aside Strategy)
#     cached_result = get_cached_prediction(cache_key)
#     if cached_result:
#         # Senior Flex: inject the hit status telemetry flag
#         cached_result["cache_hit"] = True
#         return cached_result

#     # 3. CACHE MISS: Execute standard model logic
#     # Search the hard drive for the model file
#     model_path = os.path.join(MODELS_DIR, f"{request.model_id}.joblib")
#     if not os.path.exists(model_path):
#         raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id} on server storage.")

#     # Load the state back into memory
#     try:
#         saved_state = joblib.load(model_path)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to load model file : {str(e)}")
    
#     model = saved_state["model"]
#     loader = saved_state["loader"]
#     requires_scaling = saved_state["requires_scaling"]
#     feature_names = saved_state["feature_names"]

#     try:
#         input_df = pd.DataFrame([request.features])
#         # Force incoming Streamlit data to be floats to prevent SHAP crashes
#         input_df = input_df.astype(float) 
        
#         # Ensure column order matches EXACTLY what the model expects
#         input_df = input_df[feature_names]
#     except KeyError:
#         raise HTTPException(status_code=400, detail="Provided features do not match training data.")
    
#     if requires_scaling:
#         # The scaler strips the column names and returns a NumPy array
#         scaled_values = loader.scaler.transform(input_df)
#         # Wrap that raw array BACK into a Pandas DataFrame using exact column names
#         input_df = pd.DataFrame(scaled_values, columns=feature_names)

#     # Calculate prediction values
#     raw_prediction = int(model.predict(input_df)[0]) 
    
#     # Grab the translation dictionary saved during training
#     target_mapping = saved_state.get("target_mapping")
#     task_type = saved_state.get("task_type", "classification") # Fallback to classification
    
#     # Fork logic based on the actual algorithm's task_type
#     if task_type == "classification":
#         raw_pred_int = int(raw_prediction)
#         if target_mapping is not None:
#             final_prediction = target_mapping.get(raw_pred_int, raw_pred_int)
#         else:
#             final_prediction = raw_pred_int
#     else:
#         final_prediction = float(raw_prediction)

#     # Explain the prediction using SHAP
#     backgroud_data = saved_state.get("background_data")
#     explanation_dict = None

#     try:
#         if backgroud_data is not None:
#             raw_estimator = model.model if hasattr(model, "model") else model
#             explainer = shap.Explainer(raw_estimator, backgroud_data)
#             shap_values = explainer(input_df)
        
#             # Check how many dimensions SHAP returned
#             if len(shap_values.values.shape) == 3:
#                 # [samples, features, classes]
#                 impact_scores = abs(shap_values.values[0, :, int(raw_prediction)])
#             else:
#                 # [samples, features] (Regression)
#                 impact_scores = abs(shap_values.values[0])
            
#             # Zip it up and convert to a native Python list
#             explanation_dict = dict(zip(feature_names, impact_scores.tolist()))

#     except Exception as e:
#         print(f"SHAP Error: {str(e)}")
#         explanation_dict = None 

#     # Construct the final prediction dictionary payload
#     response_payload = {
#         "model_id": request.model_id,
#         "prediction": final_prediction,
#         "message": "Prediction made successfully",
#         "explanation": explanation_dict,
#         "cache_hit": False # Explicit status check flag
#     }

#     # 4. Write to memory cache so next execution completes in < 2ms
#     set_cached_prediction(cache_key, response_payload, expire_seconds=3600)

#     return response_payload