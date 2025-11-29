import streamlit as st

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main {
            padding: 0;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Credit Risk Assessment Platform")

st.markdown("""
    Welcome to the **Credit Risk Assessment Platform**, powered by advanced machine learning.

    ### Features
    - 📊 **Single Application**: Assess risk for one loan application
    - 📁 **Batch Processing**: Upload CSV/Excel for multiple applications
    - 📈 **Model Analysis**: Explore model performance, CV stability, and metrics

    ### Getting Started
    1. Navigate to **Risk Calculator** (sidebar) to assess applications
    2. Visit **Model Analysis** to understand model performance

    ### About the Model
    Our **Ensemble Stacking** model combines Random Forest, Gradient Boosting, Logistic Regression, and Neural Networks
    to predict loan default probability with **93.1% AUC** on test data.

    ---
    *Built with ❤️ using Streamlit + Scikit-learn*
""")