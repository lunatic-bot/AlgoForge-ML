# Main Streamlit entry point
import streamlit as st
import pandas as pd
import requests
import json
from components import render_sidebar, render_model_card

# API base URL (Removed the trailing /api since our backend routes are at the root)
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AlgoForge-ML",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AlgoForge-ML")
st.markdown("Machine Learning made simple - Train models and make predictions")

# --- Session State Management ---
# This allows the app to remember the model_id between page clicks
if "model_id" not in st.session_state:
    st.session_state["model_id"] = ""

def main():
    # Render sidebar
    page = render_sidebar()
    
    if page == "Train":
        render_train_page()
    elif page == "Predict":
        render_predict_page()
    elif page == "Datasets":
        render_datasets_page()

def render_train_page():
    """Render the model training page."""
    st.header("⚙️ Train a Model")
    
    # 1. Fetch available options from the backend dynamically
    try:
        models_res = requests.get(f"{API_URL}/models").json()
        datasets_res = requests.get(f"{API_URL}/datasets").json()
        model_options = [m["name"] for m in models_res]
        dataset_options = [d["name"] for d in datasets_res]
    except Exception:
        st.error("⚠️ Cannot connect to backend. Make sure FastAPI is running on port 8000.")
        return
    
    # Radio toggle for data source
    data_source = st.radio("Data Source", ["Use Built-in Dataset", "Upload Custom CSV"], horizontal=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox("Select Algorithm", model_options)
        test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        
    with col2:
        #Dynamic UI based on Data Source toggle
        if data_source == "Use Built-in Dataset":
            dataset_name = st.selectbox("Select Dataset", dataset_options)
            target_column = st.text_input("Target Column", "target")
        else:
            # Custom CSV upload logic
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
            if uploaded_file is not None:
                with st.spinner("Upploading and analysing the file..."):
                    # send the file to our fastAPI endpoint
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    upload_res = requests.post(f"{API_URL}/upload", files=files)

                    if upload_res.status_code == 200:
                        st.session_state["custom_filename"] = upload_res.json()["filename"]
                        st.session_state["custom_columns"] = upload_res.json()["columns"]
                    else:
                        # If FastAPI throws an error, show it gracefully in the UI without crashing
                        st.error(f"Backend Error: {upload_res.status_code} - {upload_res.text}")

            # if file has been successfully uploaded and saved in state
            if "custom_filename" in st.session_state and "custom_columns" in st.session_state:
                dataset_name = st.session_state["custom_filename"]
                st.success(f"Successfully Loaded : {dataset_name}")
                # dynamically generate the target dropdown from csv headers
                target_column = st.selectbox("Select Target Variable (What are you predicting?)", st.session_state["custom_columns"])
            else:
                dataset_name = None
                target_column = None

        # we disable the train button if they choose upload but haven't uploaded a file
        disable_train = data_source == "Upload Custom CSV" and dataset_name is None

    
    if st.button("Train Model", type="primary", disabled=disable_train):
        with st.spinner(f"Training {model_type} on {dataset_name}..."):
            try:
                # 2. Match the exact Pydantic TrainRequest schema
                payload = {
                    "model_type": model_type,
                    "dataset_name": dataset_name,
                    "target_column": target_column,
                    "test_size": test_size,
                    "random_state": 42,
                    "hyperparameters": {}
                }
                
                response = requests.post(f"{API_URL}/train", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(result["message"])
                    
                    # Save ID to session state so it auto-fills on the Predict page
                    st.session_state["model_id"] = result["model_id"]

                    #Save the required features to session state
                    st.session_state["expected_features"] = result["expected_features"]
                    
                    st.subheader("📊 Training Results")
                    col_a, col_b = st.columns(2)

                    # col_a.metric("Model Score (Accuracy)", f"{result['accuracy'] * 100:.2f}%")
                    col_b.info(f"Model ID: {result['model_id']}")

                    # dashboard routing
                    if result["task_type"] == "classification":
                        col_a.metric("Model Score (Accuracy)", f"{result['metrics']['accuracy'] * 100:.2f}%")
                        st.markdown("**Classification Report:**")
                        # The classification runners return "detailed_report" inside the metrics dict
                        report_df = pd.DataFrame(result["metrics"]["detailed_report"]).transpose()
                        st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
                    
                    elif result["task_type"] == "regression":
                        # Render regression dashboard
                        col_a.metric("Model Score (R2)", f"{result['metrics']['r2_score']:.4f}")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Mean Squared Error (MSE)", f"{result['metrics']['mse']:.2f}")
                        col2.metric("Root Mean Squared Error (RMSE)", f"{result['metrics']['rmse']:.2f}")
                        st.info("💡 **R² Score** closer to 1.0 means the model explains the variance well. Lower **RMSE** means the model's predictions are closer to the actual values.")

                        
                    
                    # Display the detailed classification report beautifully
                    # st.subheader("Classification Report")
                    # report_df = pd.DataFrame(result["report"]).transpose()
                    # st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")

def render_predict_page():
    """Render the prediction page."""
    st.header("🔮 Make Predictions")
    
    # Auto-fill from session state if they just trained a model
    model_id = st.text_input("Model ID", value=st.session_state["model_id"])
    
    st.markdown("Enter features as a JSON dictionary matching the dataset columns:")

    #Dynamically generate the JSON template!
    expected_features = st.session_state.get("expected_features", [])

    if expected_features:
        # Create a dictionary with 0.0 for every required feature
        default_dict = {feature: 0.0 for feature in expected_features}
        default_json = json.dumps(default_dict, indent=2)
    else:
        # Default JSON asking user to train before testing
        default_json = "{\n  // Train a model first to auto-generate the required fields\n}"
    
    features_input = st.text_area("Features (JSON)", value=default_json, height=150)
    
    if st.button("Predict", type="primary"):
        if not model_id:
            st.warning("Please enter a Model ID first.")
            return
            
        with st.spinner("Analyzing data..."):
            try:
                # Parse the JSON string from the text area into a Python dictionary
                feature_dict = json.loads(features_input)
                
                # Match the exact Pydantic PredictRequest schema
                payload = {
                    "model_id": model_id,
                    "features": feature_dict
                }
                
                response = requests.post(f"{API_URL}/predict", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(result["message"])
                    # st.metric("Predicted Class", result["prediction"][0])
                    # Convert it to string and title-case it so "setosa" becomes "Setosa"
                    # st.metric("Predicted Class", str(result["prediction"]).title())
                    
                    prediction_value = result["prediction"]
                    # If the API returned a float, it's a Regression prediction
                    if isinstance(prediction_value, float):
                        st.metric("Predicted Value (Continuous)", f"{prediction_value:.2f}")
                        st.info("💡 Note: For the Diabetes dataset, this number represents disease progression one year after baseline (Scale: 25 - 346).")
                    # Otherwise, it's a Classification label
                    else:
                        st.metric("Predicted Class", str(prediction_value).title())
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please ensure your features use double quotes and standard JSON syntax.")
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")

def render_datasets_page():
    """Render the datasets page."""
    st.header("📊 Available Datasets")
    
    try:
        response = requests.get(f"{API_URL}/datasets")
        if response.status_code == 200:
            # Our backend returns a direct list [{}, {}], not {"datasets": []}
            datasets = response.json() 
            
            for ds in datasets:
                with st.expander(f"📁 **{ds['name']}**", expanded=True):
                    st.write(f"**Description:** {ds['description']}")
                    st.write(f"**Target Column:** `{ds['target_column']}`")
        else:
            st.error("Failed to load datasets")
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")

if __name__ == "__main__":
    main()