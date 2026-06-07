# Main Streamlit entry point
import streamlit as pd
import streamlit as st
import pandas as pd
import requests
import json
import logging
import sys
import os
from datetime import datetime

# Import structural components from setup modules
from components import render_sidebar, render_model_card

# ---------------- CENTRALIZED STREAMLIT LOGGER SETUP ---------------- #
def setup_frontend_logger():
    """Configures a standardized console tracking logger for the Streamlit UI frame."""
    logger = logging.getLogger("algoforge.frontend")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | FRONTEND | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

logger = setup_frontend_logger()

# Resolve the API endpoint configuration dynamically across your laptop environment matrix
if os.getenv("API_URL"):
    API_URL = os.getenv("API_URL").rstrip("/")
else:
    API_URL = "http://localhost:8000"

logger.info(f"Streamlit client engine lifecycle initialized. Targeting backend engine context at: {API_URL}")

st.set_page_config(
    page_title="AlgoForge-ML",
    page_icon="🤖",
    layout="wide"
)

# --- Session State Management Initialization ---
if "model_id" not in st.session_state:
    st.session_state["model_id"] = ""

# --- Helper Logic Network Handler Functions ---
def login_user(username, password):
    """Submits form data to the authentication endpoint securely."""
    logger.info(f"Submitting credential validation request for username: '{username}'")
    try:
        response = requests.post(
            f"{API_URL}/login", 
            data={"username": username, "password": password},
            timeout=10 # Prevent UI locking on network stalls
        )
        if response.status_code == 200:
            logger.info(f"User identity confirmed for '{username}'. Access token granted.")
            return response.json()
        else:
            logger.warning(f"Authentication rejected for user '{username}'. Status code: {response.status_code}")
            st.sidebar.error("Invalid credentials profile. Check inputs.")
            return None
    except requests.exceptions.Timeout:
        logger.error("Authentication handshake timed out on connection gateway.")
        st.sidebar.error("Auth server connection timeout. Try again.")
        return None
    except requests.exceptions.RequestException as network_err:
        logger.error(f"Network subsystem drop out during login routine: {str(network_err)}", exc_info=True)
        st.sidebar.error("Unable to establish communication with authorization gateway server.")
        return None


# --- Core Page Rendering Components Layout Blocks ---
def main():
    st.title("🤖 AlgoForge-ML")
    st.markdown("Machine Learning made simple - Train models and make predictions")

    # 1. Render identity check panel instantly inside side viewport
    render_login_sidebar()
    
    # 2. Only project protected visualization dashboards if access keys exist inside session state memory
    if "access_token" in st.session_state:
        # Map protected navigation indexes
        page = st.sidebar.selectbox("Navigate Workspace", ["Train", "Predict", "Datasets", "History"])
        logger.info(f"User context page transition triggered: view shifted to target panel -> [{page}]")
        
        if page == "Train":
            render_train_page()
        elif page == "Predict":
            render_predict_page()
        elif page == "Datasets":
            render_datasets_page()
        elif page == "History":
            render_history_page()
    else:
        logger.debug("Anonymous browsing session intercepted. Rendering generic security lockouts.")
        st.info("👈 Please enter your credentials in the sidebar to begin.")
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJueGZ4bmZ4bmZ4/3o7TKMGpxx0R0A1N4k/giphy.gif", caption="Security Clearance Enforced")


def render_login_sidebar():
    """Generates user authorization context layout elements inside Streamlit sidebar container."""
    with st.sidebar:
        st.markdown(f"**Gateway Diagnostic:** `Connecting to {API_URL}`")
        st.title("🔐 Authentication")
        
        if "access_token" not in st.session_state:
            auth_mode = st.radio("Choose Action", ["Login", "Sign Up"], horizontal=True)
            username = st.text_input("Username", key="auth_user_input")
            password = st.text_input("Password", type="password", key="auth_pw_input")
            
            if auth_mode == "Login":
                if st.button("Login", type="primary", use_container_width=True):
                    if not username or not password:
                        st.warning("Credential properties fields cannot be blank.")
                        return
                    data = login_user(username, password)
                    if data:
                        st.session_state["access_token"] = data["access_token"]
                        st.session_state["username"] = username
                        st.success("Welcome back!")
                        st.rerun()
            else:
                if st.button("Create Account", use_container_width=True):
                    if username and password:
                        logger.info(f"Submitting registration packet creation request for username property: '{username}'")
                        try:
                            res = requests.post(f"{API_URL}/register", json={"username": username, "password": password}, timeout=10)
                            if res.status_code == 201:
                                logger.info(f"Registration successful. Database schema user record appended for: '{username}'")
                                st.success("Account created! You can now switch to Login mode.")
                            else:
                                detail_msg = res.json().get("detail", "Registration failed")
                                logger.warning(f"Registration request denied by backend logic gate: {detail_msg}")
                                st.error(f"Registration Denied: {detail_msg}")
                        except requests.exceptions.RequestException as req_err:
                            logger.error(f"Registration processing failed: Connection breakdown: {str(req_err)}", exc_info=True)
                            st.error("Registration server unreachable.")
                    else:
                        st.warning("Please fill in all fields")
        else:
            st.write(f"Active Session: **{st.session_state['username']}**")
            if st.button("Logout", type="secondary", use_container_width=True):
                logger.info(f"Active session logout procedure finalized for user profile: '{st.session_state.get('username')}'")
                del st.session_state["access_token"]
                del st.session_state["username"]
                st.rerun()


def render_train_page():
    """Render the model training orchestration management visualization layout canvas."""
    st.header("⚙️ Train a Model")
    
    # 1. Fetch available structural option matrices from backend storage definitions dynamically
    try:
        models_res = requests.get(f"{API_URL}/models", timeout=5).json()
        datasets_res = requests.get(f"{API_URL}/datasets", timeout=5).json()
        model_options = [m["name"] for m in models_res]
        dataset_options = [d["name"] for d in datasets_res]
        logger.debug("Successfully updated dynamic algorithm arrays lists from backend directories mapping vectors.")
    except Exception as fetch_err:
        logger.error(f"Dashboard metadata catalog configuration collection step collapsed: Unable to map endpoint choices. Error context: {str(fetch_err)}", exc_info=True)
        st.error("⚠️ Cannot establish structural handshake with backend service stack dependencies. Verify FastAPI running parameters on port 8000.")
        return
    
    data_source = st.radio("Data Source Selection Matrix", ["Use Built-in Dataset", "Upload Custom CSV"], horizontal=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    drop_columns = []
    
    with col1:
        model_type = st.selectbox("Select Target Training Algorithm Architecture", model_options)
        test_size = st.slider("Validation Train Split Holdout Ratio (Test Set Size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        tune_hyperparameters = st.toggle("🧪 Enable Hyperparameter Optimization Pipeline (GridSearchCV)")
        
    with col2:
        if data_source == "Use Built-in Dataset":
            dataset_name = st.selectbox("Select Target Benchmark Dataset Profile", dataset_options)
            target_column = st.text_input("Target Ground Truth Label Variable Header", "target")
        else:
            uploaded_file = st.file_uploader("Upload your custom structured properties file (CSV format)", type=["csv"])
            if uploaded_file is not None:
                with st.spinner("Streaming data stream file to server analysis engines..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                        upload_res = requests.post(f"{API_URL}/upload", files=files, timeout=30)

                        if upload_res.status_code == 200:
                            st.session_state["custom_filename"] = upload_res.json()["filename"]
                            st.session_state["custom_columns"] = upload_res.json()["columns"]
                            logger.info(f"Custom user tracking file accepted and recorded on backend allocation disk: '{uploaded_file.name}'")
                        else:
                            logger.error(f"File upload interface processing rejected by endpoint parsing gates. Error response: {upload_res.text}")
                            st.error(f"Backend Processing Fault: {upload_res.status_code} - {upload_res.text}")
                    except requests.exceptions.RequestException as network_upload_err:
                        logger.error(f"File transfer streaming process failed processing network vectors: {str(network_upload_err)}", exc_info=True)
                        st.error("Data upload transmission failure.")

            if "custom_filename" in st.session_state and "custom_columns" in st.session_state:
                dataset_name = st.session_state["custom_filename"]
                st.success(f"Successfully Registered Storage Reference: {dataset_name}")
                target_column = st.selectbox("Select Target Variable (What are you predicting?)", st.session_state["custom_columns"])
                available_features = [col for col in st.session_state["custom_columns"] if col != target_column]
                drop_columns = st.multiselect("Select columns to DROP (e.g., Name, ID, Ticket)", options=available_features, default=[])
            else:
                dataset_name, target_column, drop_columns = None, None, []

        disable_train = data_source == "Upload Custom CSV" and dataset_name is None

    # --- UNIFIED COMPILING TRIGGER TRIGGER PIPELINE EXECUTION ---
    if st.button("Train Model Instance", type="primary", disabled=disable_train, use_container_width=True):
        if "access_token" not in st.session_state:
            st.error("Session keys expired or unauthenticated. Return to workspace entry portal.")
            return

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
        headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}
        
        logger.info(f"User matching request blueprint payload parameter blocks. Transmitting optimization run directive parameters...")
        with st.spinner(f"Compiling tracking configurations. Training {model_type} on dataset: {dataset_name}..."):
            try:
                response = requests.post(f"{API_URL}/train", json=payload, headers=headers, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Optimization run executed flawlessly. Assigned UUID tracking model keys signature: {result['model_id']}")
                    st.success("Model compilation sequence finalized successfully! Experiments recorded in MLflow registry panel.")
                    
                    st.session_state["model_id"] = result["model_id"]
                    st.session_state["expected_features"] = result["expected_features"]
                    
                    st.subheader("📊 Operational Analytics Evaluation Summary Metrics")
                    col_a, col_b = st.columns(2)
                    col_b.info(f"Model ID Key String: `{result['model_id']}`")

                    if result["task_type"] == "classification":
                        col_a.metric("Model Classification Accuracy", f"{result['metrics']['accuracy'] * 100:.2f}%")
                        st.markdown("**Detailed Classification Matrix Evaluation Report Summary:**")
                        report_df = pd.DataFrame(result["metrics"]["detailed_report"]).transpose()
                        st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
                    elif result["task_type"] == "regression":
                        col_a.metric("Model Variance Explanation Score (R²)", f"{result['metrics']['r2_score']:.4f}")
                        col_r1, col_r2, col_r3 = st.columns(3)
                        col_r1.metric("Mean Squared Error (MSE)", f"{result['metrics']['mse']:.2f}")
                        col_r2.metric("Root Mean Error (RMSE)", f"{result['metrics']['rmse']:.2f}")
                elif response.status_code == 401:
                    logger.warning("FastAPI intercept returned token authentication passport rejection code: 401.")
                    st.error("Authorization passport keys validation failure. Refresh identity validation components via panel sidebar.")
                else:
                    logger.error(f"FastAPI execution container optimization error caught: {response.status_code} - {response.text}")
                    st.error(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.Timeout:
                logger.error("Training execution connection context surpassed maximum timeout parameters boundaries.")
                st.error("Heavy operational model training computation limit threshold elapsed on server side execution limits.")
            except Exception as e:
                logger.error(f"Data streaming or operational compilation pipeline connection interrupted: {str(e)}", exc_info=True)
                st.error(f"Failed to communicate with MLOps core computing environment framework layer: {str(e)}")


def render_predict_page():
    """Render the evaluation prediction inference execution cockpit canvas panel."""
    st.header("🔮 Make Real-Time Predictions")
    model_id = st.text_input("Active Target Model Registry ID signature keys", value=st.session_state["model_id"])
    st.markdown("Configure observation parameters variables vectors as a standardized structured clean JSON dictionary layout format maps:")

    expected_features = st.session_state.get("expected_features", [])
    if expected_features:
        default_dict = {feature: 0.0 for feature in expected_features}
        default_json = json.dumps(default_dict, indent=2)
    else:
        default_json = "{\n  // Enter model registry ID key elements above to resolve signature features template\n}"
    
    features_input = st.text_area("Features Row Input Stream (JSON Matrix Format Map Specification)", value=default_json, height=150)
    
    if st.button("Compute Real-Time Inference Result", type="primary", use_container_width=True):
        if not model_id:
            st.warning("Prediction operation suspended. Missing mandatory pipeline execution variable target Model Registry ID string signature keys.")
            return
            
        logger.info(f"Submitting prediction evaluation request row payload variables to target identity block reference keys: {model_id}")
        with st.spinner("Querying real-time prediction orchestration modules layer framework..."):
            try:
                feature_dict = json.loads(features_input)
                payload = {"model_id": model_id, "features": feature_dict}
                
                # Check performance timestamps metrics logs counters triggers points
                start_time = datetime.now()
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                latency = (datetime.now() - start_time).total_seconds() * 1000

                if response.status_code == 200:
                    result = response.json()
                    
                    # Log telemetry lookup evaluation analysis output stats to dashboard terminals consoles screens
                    if result.get("cache_hit"):
                        logger.info(f"Performance Optimization Analytics Tracker Hit: [CACHE HIT] resolved inference properties details in {latency:.2f}ms")
                        st.info(f"⚡ **Redis Caching Optimization layer Hit!** Prediction pipeline calculations bypassed completely. Intercepted response record instantly from local volatile RAM memory in {latency:.1f}ms.")
                    else:
                        logger.info(f"Performance Optimization Analytics Tracker Miss: [CACHE MISS] executed heavy pipeline execution parameters computing layers in {latency:.2f}ms")
                        st.warning(f"⚙️ **Redis Caching layer Miss.** Backend loaded model serialization artifacts from structural block volume storage partitions, processed transformations, and computed SHAP attributions calculations arrays in {latency:.1f}ms.")
                                    
                    prediction_value = result["prediction"]
                    if isinstance(prediction_value, float):
                        st.metric("Continuous Target Scalar Value Estimation Outcome (Regression Output)", f"{prediction_value:.4f}")
                    else:
                        st.metric("Discrete Class Categorical Categorized Classification Label Outcome", str(prediction_value).upper())

                    if result.get("explanation"):
                        st.markdown("---")
                        st.subheader("🧠 SHAP Local Prediction Attribution Explainability Analytics Plots")
                        shap_df = pd.DataFrame.from_dict(result["explanation"], orient="index", columns=["Local Attribution Impact Weight Value Strength Factor"]).sort_values(by="Local Attribution Impact Weight Value Strength Factor", ascending=False)
                        st.bar_chart(shap_df, horizontal=True)
                else:
                    logger.error(f"Prediction logic router rejected row payload processing execution. Status context code: {response.status_code} - Text info: {response.text}")
                    st.error(f"Error {response.status_code}: {response.text}")
            except json.JSONDecodeError as json_parse_err:
                logger.warning(f"User validation dropout exception: Form input payload structural JSON syntax validation failure formatting errors: {str(json_parse_err)}")
                st.error("Invalid JSON syntax format template mapping rules parameters. Ensure keys use dual string markers properties and numbers format properly.")
            except Exception as e:
                logger.error(f"Prediction interface gateway computation module failed communication handshakes routines: {str(e)}", exc_info=True)
                st.error(f"Failed to communicate with runtime deployment server environment endpoints: {str(e)}")


def render_datasets_page():
    """Render the available benchmark data assets listing directory page canvas."""
    st.header("📊 Registered Datasets Directory Catalogs")
    logger.info("Accessing dynamic backend benchmark static data registry information elements.")
    try:
        response = requests.get(f"{API_URL}/datasets", timeout=5)
        if response.status_code == 200:
            datasets = response.json() 
            for ds in datasets:
                with st.expander(f"📁 **Dataset Identifier Asset Target Reference Key: {ds['name'].upper()}**", expanded=True):
                    st.write(f"**Structural Meta Profiling Functional Description:** {ds['description']}")
                    st.write(f"**Default Target Optimization Variable Key Label Column:** `{ds['target_column']}`")
        else:
            logger.error(f"Failed mapping listing configurations definitions from database directory endpoints: {response.status_code}")
            st.error("Failed to load datasets")
    except Exception as e:
        logger.error(f"Dataset page listing handler connection failure exceptions: {str(e)}", exc_info=True)
        st.error(f"Failed to connect to API: {str(e)}")


def render_history_page():
    """Render the structural historical local Model Registry data tracking metrics dashboard visualizer page."""
    st.header("📚 Model Registry Metadata Catalog Tracking Dashboard")
    st.markdown("Inspect performance metrics telemetry arrays records compiled across execution instances pipelines historical states.")
    logger.info("Compiling and processing structural JSON history records tracks fields configurations documents.")
    
    try:
        response = requests.get(f"{API_URL}/models/history", timeout=10)
        if response.status_code == 200:
            history = response.json().get("history", [])
            if not history:
                st.info("No active models have been recorded inside the tracking database matrix registry yet.")
                return
                
            for idx, model in enumerate(history):
                title = f"🏷️ {model.get('algorithm')} Model Tracking Block Run -> Target Source Domain Dataset: [{model.get('dataset')}] (Recorded Execution Timestamp Target: {model.get('created_at')})"
                
                with st.expander(title, expanded=(idx == 0)):
                    st.code(f"Model Serialization Registry Mapping ID Signature Key: {model.get('model_id')}", language="text")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Configuration Run Hyper-Parameters Settings Context Attributes:**")
                        st.write(f"- **Task Optimization Directive Profile Type Signature:** {model.get('task_type').upper()}")
                        st.write(f"- **Data Ingestion Source Reference Domain Label File:** `{model.get('dataset')}`")

                    with col2:
                        st.markdown("**Performance Evaluation Analytics Tracking Summary Metrics Dictionary Fields Attributes:**")
                        metrics = model.get("metrics", {})
                        for key, value in metrics.items():
                            if isinstance(value, float):
                                st.write(f"- **Evaluation Parameter Attribute Metric Key: [{key.upper()}]:** `{value:.4f}`")
                            else:
                                st.write(f"- **Evaluation Parameter Attribute Metric Key: [{key.upper()}]:** `{value}`")
                        
                    if st.button("Inject Model ID and features blueprint to prediction workspace", key=f"btn_{model.get('model_id')}", use_container_width=True):
                        logger.info(f"User extracted configuration parameters records states from registry tracking document blocks for historical deployment: {model.get('model_id')}")
                        st.session_state["model_id"] = model.get("model_id")
                        st.success("Operational Model Identity target context saved cleanly inside global Streamlit session state memory tracking keys! Navigate over onto the Predict workspace cockpit section layout tab safely.")
        else:
            logger.error(f"Model registry lookups endpoint returned error processing status flag response matrix: {response.status_code}")
            st.error("Failed to load history from the server.")
    except Exception as e:
        logger.error(f"Historical registry analyzer visualization dashboard collapsed: {str(e)}", exc_info=True)
        st.error(f"Failed to connect to API: {str(e)}")


if __name__ == "__main__":
    main()


# # Main Streamlit entry point
# import streamlit as st
# import pandas as pd
# import requests
# import json
# from components import render_sidebar, render_model_card

# import os 

# # Try to get API_URL from Streamlit secrets (for cloud), 
# # then from OS environment (for local/docker), 
# # fallback to localhost
# # if "API_URL" in st.secrets:
# #     API_URL = st.secrets["API_URL"]
# # elif os.getenv("API_URL"):
# #     API_URL = os.getenv("API_URL")
# # else:
# API_URL = "http://localhost:8000"

# # Debugging (Remove this after it works)
# st.sidebar.write(f"Connecting to: {API_URL}")

# # API base URL (Removed the trailing /api since our backend routes are at the root)
# # API_URL = "http://localhost:8000"

# st.set_page_config(
#     page_title="AlgoForge-ML",
#     page_icon="🤖",
#     layout="wide"
# )

# st.title("🤖 AlgoForge-ML")
# st.markdown("Machine Learning made simple - Train models and make predictions")

# # --- Session State Management ---
# # This allows the app to remember the model_id between page clicks
# if "model_id" not in st.session_state:
#     st.session_state["model_id"] = ""

# def main():
#     # Render sidebar
#     # page = render_sidebar()


#     # 1. ALWAYS render the login sidebar first
#     page = render_login_sidebar()
    
#     # 2. Only show the rest of the app if the user is logged in
#     if "access_token" in st.session_state:
#         st.title("🚀 AlgoForge-ML Dashboard")
        
#         # Your tab/navigation logic here
#         page = st.sidebar.selectbox("Navigate", ["Train", "Predict", "Experiments"])
        
#         if page == "Train":
#             render_train_page()
#         elif page == "Predict":
#             render_predict_page()
#         elif page == "Datasets":
#             render_datasets_page()
#         elif page == "History":
#             render_history_page()
#     else:
#         st.info("👈 Please enter your credentials in the sidebar to begin.")
#         st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJueGZ4bmZ4bmZ4/3o7TKMGpxx0R0A1N4k/giphy.gif", caption="Security First!")
    


# import requests
# import streamlit as st

# def login_user(username, password):
#     try:
#         # FastAPI's OAuth2 expects data as form-data, not JSON
#         response = requests.post(
#             f"{API_URL}/login", 
#             data={"username": username, "password": password}
#         )
#         if response.status_code == 200:
#             return response.json() # Returns {"access_token": "...", "token_type": "bearer"}
#         else:
#             st.sidebar.error("Invalid credentials")
#             return None
#     except Exception as e:
#         st.sidebar.error(f"Auth Server Error: {e}")
#         return None
    
# # def render_login_sidebar():
# #     st.sidebar.title("🔐 Authentication")
    
# #     # Check if user is already logged in
# #     if "access_token" not in st.session_state:
# #         with st.sidebar.form("login_form"):
# #             user = st.text_input("Username")
# #             pw = st.text_input("Password", type="password")
# #             submit = st.form_submit_button("Login")
            
# #             if submit:
# #                 auth_data = login_user(user, pw)
# #                 if auth_data:
# #                     st.session_state["access_token"] = auth_data["access_token"]
# #                     st.session_state["username"] = user
# #                     st.success("Logged in!")
# #                     st.rerun() # Refresh to show protected content
# #     else:
# #         st.sidebar.write(f"Logged in as: **{st.session_state['username']}**")
# #         if st.sidebar.button("Logout"):
# #             del st.session_state["access_token"]
# #             del st.session_state["username"]
# #             st.rerun()

# def render_login_sidebar():
#     with st.sidebar:
#         st.title("🔐 Authentication")
        
#         if "access_token" not in st.session_state:
#             # Let the user choose what they want to do
#             auth_mode = st.radio("Choose Action", ["Login", "Sign Up"], horizontal=True)
            
#             username = st.text_input("Username")
#             password = st.text_input("Password", type="password")
            
#             if auth_mode == "Login":
#                 if st.button("Login", type="primary"):
#                     data = login_user(username, password)
#                     if data:
#                         st.session_state["access_token"] = data["access_token"]
#                         st.session_state["username"] = username
#                         st.success("Welcome back!")
#                         st.rerun()
#             else:
#                 if st.button("Create Account"):
#                     if username and password:
#                         res = requests.post(f"{API_URL}/register", json={"username": username, "password": password})
#                         if res.status_code == 201:
#                             st.success("Account created! You can now switch to Login mode.")
#                         else:
#                             st.error(res.json().get("detail", "Registration failed"))
#                     else:
#                         st.warning("Please fill in all fields")
#         else:
#             st.write(f"Active Session: **{st.session_state['username']}**")
#             if st.button("Logout"):
#                 del st.session_state["access_token"]
#                 del st.session_state["username"]
#                 st.rerun()

# # def render_train_page():
# #     """Render the model training page."""
# #     st.header("⚙️ Train a Model")
    
# #     # 1. Fetch available options from the backend dynamically
# #     try:
# #         models_res = requests.get(f"{API_URL}/models").json()
# #         datasets_res = requests.get(f"{API_URL}/datasets").json()
# #         model_options = [m["name"] for m in models_res]
# #         dataset_options = [d["name"] for d in datasets_res]
# #     except Exception:
# #         st.error("⚠️ Cannot connect to backend. Make sure FastAPI is running on port 8000.")
# #         return
    
# #     # Radio toggle for data source
# #     data_source = st.radio("Data Source", ["Use Built-in Dataset", "Upload Custom CSV"], horizontal=True)
# #     st.markdown("---")

# #     col1, col2 = st.columns(2)
# #     drop_columns = []
    
# #     with col1:
# #         model_type = st.selectbox("Select Algorithm", model_options)
# #         test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
# #         # Add the tuning switch
# #         tune_hyperparameters = st.toggle("🧪 Enable Hyperparameter Tuning (GridSearchCV)")
        
# #     with col2:
# #         #Dynamic UI based on Data Source toggle
# #         if data_source == "Use Built-in Dataset":
# #             dataset_name = st.selectbox("Select Dataset", dataset_options)
# #             target_column = st.text_input("Target Column", "target")
# #         else:
# #             # Custom CSV upload logic
# #             uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
# #             if uploaded_file is not None:
# #                 with st.spinner("Upploading and analysing the file..."):
# #                     # send the file to our fastAPI endpoint
# #                     files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
# #                     upload_res = requests.post(f"{API_URL}/upload", files=files)

# #                     if upload_res.status_code == 200:
# #                         st.session_state["custom_filename"] = upload_res.json()["filename"]
# #                         st.session_state["custom_columns"] = upload_res.json()["columns"]
# #                     else:
# #                         # If FastAPI throws an error, show it gracefully in the UI without crashing
# #                         st.error(f"Backend Error: {upload_res.status_code} - {upload_res.text}")

# #             # if file has been successfully uploaded and saved in state
# #             if "custom_filename" in st.session_state and "custom_columns" in st.session_state:
# #                 dataset_name = st.session_state["custom_filename"]
# #                 st.success(f"Successfully Loaded : {dataset_name}")
# #                 # dynamically generate the target dropdown from csv headers
# #                 target_column = st.selectbox("Select Target Variable (What are you predicting?)", st.session_state["custom_columns"])
# #                 # Build a list of features the user can drop
# #                 available_features = [col for col in st.session_state["custom_columns"] if col != target_column]
# #                 # The Multiselect widget
# #                 drop_columns = st.multiselect("Select columns to DROP (e.g., Name, ID, Ticket)", options=available_features,
# #                                               default=[])

# #             else:
# #                 dataset_name = None
# #                 target_column = None
# #                 drop_columns = []

# #         # we disable the train button if they choose upload but haven't uploaded a file
# #         disable_train = data_source == "Upload Custom CSV" and dataset_name is None

    
# #     if st.button("Train Model", type="primary", disabled=disable_train):
        
# #         if "access_token" not in st.session_state:
# #             st.error("Please login via the sidebar to train models.")
# #         else:
# #             headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}
            
# #             with st.spinner("Training..."):
# #                 response = requests.post(
# #                     f"{API_URL}/train", 
# #                     json=payload, 
# #                     headers=headers  # <--- CRITICAL ADDITION
# #                 )
                
# #                 if response.status_code == 200:
# #                     st.success("Trained successfully!")
# #                 elif response.status_code == 401:
# #                     st.error("Session expired. Please log in again.")


# #         with st.spinner(f"Training {model_type} on {dataset_name}..."):
# #             try:
# #                 # 2. Match the exact Pydantic TrainRequest schema
# #                 payload = {
# #                     "model_type": model_type,
# #                     "dataset_name": dataset_name,
# #                     "target_column": target_column,
# #                     "test_size": test_size,
# #                     "random_state": 42,
# #                     "hyperparameters": {},
# #                     "drop_columns": drop_columns,
# #                     "tune_hyperparameters": tune_hyperparameters
# #                 }
                
# #                 response = requests.post(f"{API_URL}/train", json=payload)
                
# #                 if response.status_code == 200:
# #                     result = response.json()
# #                     st.success(result["message"])
                    
# #                     # Save ID to session state so it auto-fills on the Predict page
# #                     st.session_state["model_id"] = result["model_id"]

# #                     #Save the required features to session state
# #                     st.session_state["expected_features"] = result["expected_features"]
                    
# #                     st.subheader("📊 Training Results")
# #                     col_a, col_b = st.columns(2)

# #                     # col_a.metric("Model Score (Accuracy)", f"{result['accuracy'] * 100:.2f}%")
# #                     col_b.info(f"Model ID: {result['model_id']}")

# #                     # dashboard routing
# #                     if result["task_type"] == "classification":
# #                         col_a.metric("Model Score (Accuracy)", f"{result['metrics']['accuracy'] * 100:.2f}%")
# #                         st.markdown("**Classification Report:**")
# #                         # The classification runners return "detailed_report" inside the metrics dict
# #                         report_df = pd.DataFrame(result["metrics"]["detailed_report"]).transpose()
# #                         st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
                    
# #                     elif result["task_type"] == "regression":
# #                         # Render regression dashboard
# #                         col_a.metric("Model Score (R2)", f"{result['metrics']['r2_score']:.4f}")
# #                         col1, col2, col3 = st.columns(3)
# #                         col1.metric("Mean Squared Error (MSE)", f"{result['metrics']['mse']:.2f}")
# #                         col2.metric("Root Mean Squared Error (RMSE)", f"{result['metrics']['rmse']:.2f}")
# #                         st.info("💡 **R² Score** closer to 1.0 means the model explains the variance well. Lower **RMSE** means the model's predictions are closer to the actual values.")

# #                 else:
# #                     st.error(f"Error {response.status_code}: {response.text}")
# #             except Exception as e:
# #                 st.error(f"Failed to connect to API: {str(e)}")

# def render_train_page():
#     """Render the model training page."""
#     st.header("⚙️ Train a Model")
    
#     # 1. Fetch available options from the backend dynamically
#     try:
#         models_res = requests.get(f"{API_URL}/models").json()
#         datasets_res = requests.get(f"{API_URL}/datasets").json()
#         model_options = [m["name"] for m in models_res]
#         dataset_options = [d["name"] for d in datasets_res]
#     except Exception:
#         st.error("⚠️ Cannot connect to backend. Make sure FastAPI is running on port 8000.")
#         return
    
#     # Radio toggle for data source
#     data_source = st.radio("Data Source", ["Use Built-in Dataset", "Upload Custom CSV"], horizontal=True)
#     st.markdown("---")

#     col1, col2 = st.columns(2)
#     drop_columns = []
    
#     with col1:
#         model_type = st.selectbox("Select Algorithm", model_options)
#         test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
#         # Add the tuning switch
#         tune_hyperparameters = st.toggle("🧪 Enable Hyperparameter Tuning (GridSearchCV)")
        
#     with col2:
#         # Dynamic UI based on Data Source toggle
#         if data_source == "Use Built-in Dataset":
#             dataset_name = st.selectbox("Select Dataset", dataset_options)
#             target_column = st.text_input("Target Column", "target")
#         else:
#             # Custom CSV upload logic
#             uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
#             if uploaded_file is not None:
#                 with st.spinner("Uploading and analysing the file..."):
#                     # send the file to our fastAPI endpoint
#                     files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
#                     upload_res = requests.post(f"{API_URL}/upload", files=files)

#                     if upload_res.status_code == 200:
#                         st.session_state["custom_filename"] = upload_res.json()["filename"]
#                         st.session_state["custom_columns"] = upload_res.json()["columns"]
#                     else:
#                         st.error(f"Backend Error: {upload_res.status_code} - {upload_res.text}")

#             # if file has been successfully uploaded and saved in state
#             if "custom_filename" in st.session_state and "custom_columns" in st.session_state:
#                 dataset_name = st.session_state["custom_filename"]
#                 st.success(f"Successfully Loaded : {dataset_name}")
#                 # dynamically generate the target dropdown from csv headers
#                 target_column = st.selectbox("Select Target Variable (What are you predicting?)", st.session_state["custom_columns"])
#                 # Build a list of features the user can drop
#                 available_features = [col for col in st.session_state["custom_columns"] if col != target_column]
#                 # The Multiselect widget
#                 drop_columns = st.multiselect("Select columns to DROP (e.g., Name, ID, Ticket)", options=available_features, default=[])
#             else:
#                 dataset_name = None
#                 target_column = None
#                 drop_columns = []

#         # disable train button if custom data is missing
#         disable_train = data_source == "Upload Custom CSV" and dataset_name is None

#     # --- SINGLE UNIFIED TRAINING TRIGGER ---
#     if st.button("Train Model", type="primary", disabled=disable_train):
#         if "access_token" not in st.session_state:
#             st.error("Please login via the sidebar to train models.")
#             return

#         # 1. Build the payload first so it safely exists in scope
#         payload = {
#             "model_type": model_type,
#             "dataset_name": dataset_name,
#             "target_column": target_column,
#             "test_size": test_size,
#             "random_state": 42,
#             "hyperparameters": {},
#             "drop_columns": drop_columns,
#             "tune_hyperparameters": tune_hyperparameters
#         }
        
#         headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}
        
#         with st.spinner(f"Training {model_type} on {dataset_name}..."):
#             try:
#                 response = requests.post(
#                     f"{API_URL}/train", 
#                     json=payload, 
#                     headers=headers
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     st.success(result["message"])
                    
#                     # Save metrics to session state for navigation tracking
#                     st.session_state["model_id"] = result["model_id"]
#                     st.session_state["expected_features"] = result["expected_features"]
                    
#                     st.subheader("📊 Training Results")
#                     col_a, col_b = st.columns(2)
#                     col_b.info(f"Model ID: {result['model_id']}")

#                     # Dashboard routing based on prediction task metadata
#                     if result["task_type"] == "classification":
#                         col_a.metric("Model Score (Accuracy)", f"{result['metrics']['accuracy'] * 100:.2f}%")
#                         st.markdown("**Classification Report:**")
#                         report_df = pd.DataFrame(result["metrics"]["detailed_report"]).transpose()
#                         st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
                    
#                     elif result["task_type"] == "regression":
#                         col_a.metric("Model Score (R2)", f"{result['metrics']['r2_score']:.4f}")
#                         col1, col2, col3 = st.columns(3)
#                         col1.metric("Mean Squared Error (MSE)", f"{result['metrics']['mse']:.2f}")
#                         col2.metric("Root Mean Squared Error (RMSE)", f"{result['metrics']['rmse']:.2f}")
#                         st.info("💡 **R² Score** closer to 1.0 means the model explains the variance well. Lower **RMSE** means the predictions are closer to actual values.")
                
#                 elif response.status_code == 401:
#                     st.error("Session expired or unauthorized. Please log in via the sidebar.")
#                 else:
#                     st.error(f"Error {response.status_code}: {response.text}")
                    
#             except Exception as e:
#                 st.error(f"Failed to connect to API: {str(e)}")

# def render_predict_page():
#     """Render the prediction page."""
#     st.header("🔮 Make Predictions")
    
#     # Auto-fill from session state if they just trained a model
#     model_id = st.text_input("Model ID", value=st.session_state["model_id"])
    
#     st.markdown("Enter features as a JSON dictionary matching the dataset columns:")

#     #Dynamically generate the JSON template!
#     expected_features = st.session_state.get("expected_features", [])

#     if expected_features:
#         # Create a dictionary with 0.0 for every required feature
#         default_dict = {feature: 0.0 for feature in expected_features}
#         default_json = json.dumps(default_dict, indent=2)
#     else:
#         # Default JSON asking user to train before testing
#         default_json = "{\n  // Train a model first to auto-generate the required fields\n}"
    
#     features_input = st.text_area("Features (JSON)", value=default_json, height=150)
    
#     if st.button("Predict", type="primary"):
#         if not model_id:
#             st.warning("Please enter a Model ID first.")
#             return
            
#         with st.spinner("Analyzing data..."):
#             try:
#                 # Parse the JSON string from the text area into a Python dictionary
#                 feature_dict = json.loads(features_input)
                
#                 # Match the exact Pydantic PredictRequest schema
#                 payload = {
#                     "model_id": model_id,
#                     "features": feature_dict
#                 }
                
#                 response = requests.post(f"{API_URL}/predict", json=payload)
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     st.success(result["message"])

#                     # Visual verification badge for Redis
#                     if result.get("cache_hit"):
#                         st.info("⚡ **Cache Hit!** Response retrieved instantly from Redis memory (under 2ms).")
#                     else:
#                         st.warning("⚙️ **Cache Miss.** Loaded model from disk and executed SHAP calculations.")
                                    
#                     prediction_value = result["prediction"]
#                     # If the API returned a float, it's a Regression prediction
#                     if isinstance(prediction_value, float):
#                         st.metric("Predicted Value (Continuous)", f"{prediction_value:.2f}")
#                     # Otherwise, it's a Classification label
#                     else:
#                         st.metric("Predicted Class", str(prediction_value).title())


#                     if result.get("explanation"):
#                         st.markdown("---")
#                         st.subheader("🧠 Why did the model make this decision?")
#                         st.markdown("This chart shows which features had the highest impact on this specific prediction.")
#                         # # Convert dict to dataframe for Streamlit charting
#                         shap_df = pd.DataFrame.from_dict(
#                             result["explanation"],
#                             orient="index",
#                             columns=["Impact"],
#                         ).sort_values(by="Impact", ascending=False)
#                         st.bar_chart(shap_df, horizontal=True)

#                 else:
#                     st.error(f"Error {response.status_code}: {response.text}")
#             except json.JSONDecodeError:
#                 st.error("Invalid JSON format. Please ensure your features use double quotes and standard JSON syntax.")
#             except Exception as e:
#                 st.error(f"Failed to connect to API: {str(e)}")

# def render_datasets_page():
#     """Render the datasets page."""
#     st.header("📊 Available Datasets")
    
#     try:
#         response = requests.get(f"{API_URL}/datasets")
#         if response.status_code == 200:
#             # Our backend returns a direct list [{}, {}], not {"datasets": []}
#             datasets = response.json() 
            
#             for ds in datasets:
#                 with st.expander(f"📁 **{ds['name']}**", expanded=True):
#                     st.write(f"**Description:** {ds['description']}")
#                     st.write(f"**Target Column:** `{ds['target_column']}`")
#         else:
#             st.error("Failed to load datasets")
#     except Exception as e:
#         st.error(f"Failed to connect to API: {str(e)}")


# def render_history_page():
#     """Render the Model Registry history page."""
#     st.header("📚 Model Registry")
#     st.markdown("View all historically trained models, their configurations, and performance metrics.")
#     try:
#         response = requests.get(f"{API_URL}/models/history")
#         if response.status_code == 200:
#             history = response.json().get("history", [])
#             if not history:
#                 st.info("No models have been trained yet.")
#                 return
#             # Loop through the history and create a neat expander for each model
#             for idx, model in enumerate(history):
#                 # Use a prominent header showing the Algorithm and Dataset
#                 title = f"{model.get('algorithm')} ➔ {model.get('dataset')} ({model.get('created_at')})"

#                 with st.expander(title, expanded=(idx == 0)):
#                     st.code(model.get("model_id"), language="text")

#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.write("**Configuration**")
#                         st.write(f"- **Task:** {model.get('task_type').title()}")
#                         st.write(f"- **Dataset:** {model.get('dataset')}")

#                     with col2:
#                         st.write("**Performance Metrics**")
#                         # display the metrics dictionary 
#                         metrics = model.get("metrics", {})
#                         for key, value in metrics.items():
#                             if isinstance(value, float):
#                                 st.write(f"- **{key.title()}:** {value:.4f}")
#                             else:
#                                 st.write(f"- **{key.title()}:** {value}")
                        
#                     #add a quick-copy button feature
#                     if st.button("Use this model", key=f"btn_{model.get('model_id')}"):
#                         st.session_state["model_id"] = model.get("model_id")
#                         st.success("Model ID copied to session state! Head over to the Predict tab.")

#         else:
#             st.error("Failed to load history from the server.")
#     except Exception as e:
#         st.error(f"Failed to connect to API: {str(e)}")


# if __name__ == "__main__":
#     main()