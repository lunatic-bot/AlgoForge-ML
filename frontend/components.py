# Reusable UI parts (e.g., sidebar logic)
import streamlit as st

def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select a module:",
        ["Train", "Predict", "Datasets"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About AlgoForge")
    st.sidebar.info(
        "AlgoForge-ML is an end-to-end machine learning platform "
        "built with FastAPI and Streamlit. It strictly separates "
        "ML logic from the UI via REST APIs."
    )
    
    # Optional status indicator for the user
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status:** 🟢 Online")
    
    return page

def render_model_card(model_name, model_type, score=None):
    """Render a card displaying model information."""
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; background-color: #f8f9fa; margin-bottom: 10px;">
            <h4 style="margin-top: 0px;">{model_name}</h4>
            <p style="margin-bottom: 5px;"><strong>Type:</strong> {model_type}</p>
            {f'<p style="margin-bottom: 0px;"><strong>Score:</strong> {score:.4f}</p>' if score is not None else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_metric_card(label, value):
    """Render a metric card."""
    st.metric(label=label, value=value)

def render_success_message(message):
    """Render a success message."""
    st.success(message)

def render_error_message(message):
    """Render an error message."""
    st.error(message)

def render_info_message(message):
    """Render an info message."""
    st.info(message)