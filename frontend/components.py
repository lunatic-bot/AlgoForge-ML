# Reusable UI parts (e.g., sidebar logic)
import streamlit as st


def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Go to",
        ["Train", "Predict", "Datasets"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "AlgoForge-ML is a machine learning platform "
        "that makes it easy to train models and make predictions."
    )
    
    return page


def render_model_card(model_name, model_type, score=None):
    """Render a card displaying model information."""
    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;">
            <h4>{model_name}</h4>
            <p><strong>Type:</strong> {model_type}</p>
            {f'<p><strong>Score:</strong> {score:.4f}</p>' if score is not None else ''}
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