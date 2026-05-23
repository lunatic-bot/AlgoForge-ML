# Main Streamlit entry point
import streamlit as st
import pandas as pd
import requests
import json
from components import render_sidebar, render_model_card

import os 

# Try to get API_URL from Streamlit secrets (for cloud), 
# then from OS environment (for local/docker), 
# fallback to localhost
# if "API_URL" in st.secrets:
#     API_URL = st.secrets["API_URL"]
# elif os.getenv("API_URL"):
#     API_URL = os.getenv("API_URL")
# else:
API_URL = "http://localhost:8000"

# Debugging (Remove this after it works)
st.sidebar.write(f"Connecting to: {API_URL}")

# API base URL (Removed the trailing /api since our backend routes are at the root)
# API_URL = "http://localhost:8000"

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
    # page = render_sidebar()


    # 1. ALWAYS render the login sidebar first
    page = render_login_sidebar()
    
    # 2. Only show the rest of the app if the user is logged in
    if "access_token" in st.session_state:
        st.title("🚀 AlgoForge-ML Dashboard")
        
        # Your tab/navigation logic here
        page = st.sidebar.selectbox("Navigate", ["Train", "Predict", "Experiments"])
        
        if page == "Train":
            render_train_page()
        elif page == "Predict":
            render_predict_page()
        elif page == "Datasets":
            render_datasets_page()
        elif page == "History":
            render_history_page()
    else:
        st.info("👈 Please enter your credentials in the sidebar to begin.")
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJueGZ4bmZ4bmZ4/3o7TKMGpxx0R0A1N4k/giphy.gif", caption="Security First!")
    


import requests
import streamlit as st

def login_user(username, password):
    try:
        # FastAPI's OAuth2 expects data as form-data, not JSON
        response = requests.post(
            f"{API_URL}/login", 
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            return response.json() # Returns {"access_token": "...", "token_type": "bearer"}
        else:
            st.sidebar.error("Invalid credentials")
            return None
    except Exception as e:
        st.sidebar.error(f"Auth Server Error: {e}")
        return None
    
# def render_login_sidebar():
#     st.sidebar.title("🔐 Authentication")
    
#     # Check if user is already logged in
#     if "access_token" not in st.session_state:
#         with st.sidebar.form("login_form"):
#             user = st.text_input("Username")
#             pw = st.text_input("Password", type="password")
#             submit = st.form_submit_button("Login")
            
#             if submit:
#                 auth_data = login_user(user, pw)
#                 if auth_data:
#                     st.session_state["access_token"] = auth_data["access_token"]
#                     st.session_state["username"] = user
#                     st.success("Logged in!")
#                     st.rerun() # Refresh to show protected content
#     else:
#         st.sidebar.write(f"Logged in as: **{st.session_state['username']}**")
#         if st.sidebar.button("Logout"):
#             del st.session_state["access_token"]
#             del st.session_state["username"]
#             st.rerun()

def render_login_sidebar():
    with st.sidebar:
        st.title("🔐 Authentication")
        
        if "access_token" not in st.session_state:
            # Let the user choose what they want to do
            auth_mode = st.radio("Choose Action", ["Login", "Sign Up"], horizontal=True)
            
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if auth_mode == "Login":
                if st.button("Login", type="primary"):
                    data = login_user(username, password)
                    if data:
                        st.session_state["access_token"] = data["access_token"]
                        st.session_state["username"] = username
                        st.success("Welcome back!")
                        st.rerun()
            else:
                if st.button("Create Account"):
                    if username and password:
                        res = requests.post(f"{API_URL}/register", json={"username": username, "password": password})
                        if res.status_code == 201:
                            st.success("Account created! You can now switch to Login mode.")
                        else:
                            st.error(res.json().get("detail", "Registration failed"))
                    else:
                        st.warning("Please fill in all fields")
        else:
            st.write(f"Active Session: **{st.session_state['username']}**")
            if st.button("Logout"):
                del st.session_state["access_token"]
                del st.session_state["username"]
                st.rerun()

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
    drop_columns = []
    
    with col1:
        model_type = st.selectbox("Select Algorithm", model_options)
        test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        # Add the tuning switch
        tune_hyperparameters = st.toggle("🧪 Enable Hyperparameter Tuning (GridSearchCV)")
        
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
                # Build a list of features the user can drop
                available_features = [col for col in st.session_state["custom_columns"] if col != target_column]
                # The Multiselect widget
                drop_columns = st.multiselect("Select columns to DROP (e.g., Name, ID, Ticket)", options=available_features,
                                              default=[])

            else:
                dataset_name = None
                target_column = None
                drop_columns = []

        # we disable the train button if they choose upload but haven't uploaded a file
        disable_train = data_source == "Upload Custom CSV" and dataset_name is None

    
    if st.button("Train Model", type="primary", disabled=disable_train):
        
        if "access_token" not in st.session_state:
            st.error("Please login via the sidebar to train models.")
        else:
            headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}
            
            with st.spinner("Training..."):
                response = requests.post(
                    f"{API_URL}/train", 
                    json=payload, 
                    headers=headers  # <--- CRITICAL ADDITION
                )
                
                if response.status_code == 200:
                    st.success("Trained successfully!")
                elif response.status_code == 401:
                    st.error("Session expired. Please log in again.")


        with st.spinner(f"Training {model_type} on {dataset_name}..."):
            try:
                # 2. Match the exact Pydantic TrainRequest schema
                payload = {
                    "model_type": model_type,
                    "dataset_name": dataset_name,
                    "target_column": target_column,
                    "test_size": test_size,
                    "random_state": 42,
                    "hyperparameters": {},
                    "drop_columns": drop_columns,
                    "tune_hyperparameters": tune_hyperparameters
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
                    
                    prediction_value = result["prediction"]
                    # If the API returned a float, it's a Regression prediction
                    if isinstance(prediction_value, float):
                        st.metric("Predicted Value (Continuous)", f"{prediction_value:.2f}")
                    # Otherwise, it's a Classification label
                    else:
                        st.metric("Predicted Class", str(prediction_value).title())


                    if result.get("explanation"):
                        st.markdown("---")
                        st.subheader("🧠 Why did the model make this decision?")
                        st.markdown("This chart shows which features had the highest impact on this specific prediction.")
                        # # Convert dict to dataframe for Streamlit charting
                        shap_df = pd.DataFrame.from_dict(
                            result["explanation"],
                            orient="index",
                            columns=["Impact"],
                        ).sort_values(by="Impact", ascending=False)
                        st.bar_chart(shap_df, horizontal=True)

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


def render_history_page():
    """Render the Model Registry history page."""
    st.header("📚 Model Registry")
    st.markdown("View all historically trained models, their configurations, and performance metrics.")
    try:
        response = requests.get(f"{API_URL}/models/history")
        if response.status_code == 200:
            history = response.json().get("history", [])
            if not history:
                st.info("No models have been trained yet.")
                return
            # Loop through the history and create a neat expander for each model
            for idx, model in enumerate(history):
                # Use a prominent header showing the Algorithm and Dataset
                title = f"{model.get('algorithm')} ➔ {model.get('dataset')} ({model.get('created_at')})"

                with st.expander(title, expanded=(idx == 0)):
                    st.code(model.get("model_id"), language="text")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Configuration**")
                        st.write(f"- **Task:** {model.get('task_type').title()}")
                        st.write(f"- **Dataset:** {model.get('dataset')}")

                    with col2:
                        st.write("**Performance Metrics**")
                        # display the metrics dictionary 
                        metrics = model.get("metrics", {})
                        for key, value in metrics.items():
                            if isinstance(value, float):
                                st.write(f"- **{key.title()}:** {value:.4f}")
                            else:
                                st.write(f"- **{key.title()}:** {value}")
                        
                    #add a quick-copy button feature
                    if st.button("Use this model", key=f"btn_{model.get('model_id')}"):
                        st.session_state["model_id"] = model.get("model_id")
                        st.success("Model ID copied to session state! Head over to the Predict tab.")

        else:
            st.error("Failed to load history from the server.")
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")


if __name__ == "__main__":
    main()