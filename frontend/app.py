# Main Streamlit entry point
import streamlit as st
import pandas as pd
import requests
from frontend.components import render_sidebar, render_model_card

# API base URL
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AlgoForge-ML",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AlgoForge-ML")
st.markdown("Machine Learning made simple - Train models and make predictions")


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
    st.header("Train a Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Model",
            ["random_forest", "decision_tree", "logistic_regression", 
             "linear_regression", "knn", "svm"]
        )
        
    with col2:
        data_path = st.text_input("Data Path", "data/raw/titanic.csv")
    
    target_column = st.text_input("Target Column", "survived")
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/train",
                    json={
                        "model_type": model_type,
                        "data_path": data_path,
                        "target_column": target_column
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success(result["message"])
                    st.metric("Model Score", f"{result['score']:.4f}")
                    st.info(f"Model ID: {result['model_id']}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")


def render_predict_page():
    """Render the prediction page."""
    st.header("Make Predictions")
    
    model_id = st.text_input("Model ID", "")
    st.markdown("Enter features (comma-separated):")
    features = st.text_input("Features", "1,2,3,4,5")
    
    if st.button("Predict"):
        with st.spinner("Making prediction..."):
            try:
                feature_list = [float(x.strip()) for x in features.split(",")]
                response = requests.post(
                    f"{API_URL}/api/predict",
                    json={
                        "model_id": model_id,
                        "features": feature_list
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success(result["message"])
                    st.metric("Prediction", result["prediction"][0])
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")


def render_datasets_page():
    """Render the datasets page."""
    st.header("Available Datasets")
    
    try:
        response = requests.get(f"{API_URL}/api/datasets")
        if response.status_code == 200:
            datasets = response.json()["datasets"]
            for ds in datasets:
                st.markdown(f"- **{ds['name']}**: {ds['path']}")
        else:
            st.error("Failed to load datasets")
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")


if __name__ == "__main__":
    main()