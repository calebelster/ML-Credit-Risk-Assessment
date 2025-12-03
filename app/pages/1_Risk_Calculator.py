# streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add app directory to path so imports work
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from utils.predictor import CreditRiskPredictor
from utils.data_processor import DataProcessor

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Risk Calculator", layout="wide", initial_sidebar_state="expanded")

st.title("💳 Credit Risk Assessment")
st.markdown("**Assess loan default risk for individual or batch applications**")

# -------------------------
# Cached loaders
# -------------------------
@st.cache_resource
def load_predictor():
    # instantiate and load your model once
    return CreditRiskPredictor()

@st.cache_resource
def load_processor():
    return DataProcessor()

@st.cache_data
def load_file_dataframe(_processor, uploaded_file_bytes):
    """
    Cache file read. Note: first arg begins with underscore so Streamlit
    does not try to hash the complex processor object.
    uploaded_file_bytes can be the raw bytes or file-like object (Streamlit uploader)
    """
    # If uploaded_file_bytes is a BytesIO or tmp file from st.file_uploader, pass directly
    return _processor.csv_to_dataframe(uploaded_file_bytes)

# initialize cached objects (only executed once)
predictor = load_predictor()
processor = load_processor()

# -------------------------
# Sidebar settings
# -------------------------
st.sidebar.markdown("### ⚙️ Model Settings")
decision_threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.30,
    max_value=0.70,
    value=0.50,
    step=0.05,
    help="Lower threshold = more conservative (catch more defaults). Higher threshold = more lenient (approve more loans)."
)
st.sidebar.markdown(f"**Current Threshold:** {decision_threshold:.2f}")

# -------------------------
# Tabs
# -------------------------
tab1, tab2 = st.tabs(["Single Application", "Batch Upload"])

# -------------------------
# Helper: cache single prediction result in session_state
# -------------------------
def save_single_prediction_to_session(df_row, result_df_row):
    # store both input row and prediction result
    st.session_state["last_single_input"] = df_row.to_dict()
    st.session_state["last_single_result"] = result_df_row.to_dict()

# -------------------------
# TAB 1: SINGLE APPLICATION
# -------------------------
with tab1:
    st.subheader("Enter Loan Application Details")

    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30, key="s_person_age")
        person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, key="s_person_income")
        person_emp_length = st.number_input("Employment Length (years)", min_value=0, value=5, key="s_person_emp_length")
        loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000, key="s_loan_amnt")

    with col2:
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=7.5, key="s_loan_int_rate")
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=10, key="s_cb_person_cred_hist_length")
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], key="s_home")
        cb_person_default_on_file = st.selectbox("Prior Default on File?", ["N", "Y"], key="s_default_flag")

    loan_intent = st.selectbox(
        "Loan Intent",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        key="s_loan_intent"
    )

    # Button runs prediction and stores result in session_state
    if st.button("🔍 Calculate Risk", key="single_predict"):
        try:
            form_data = {
                "person_age": person_age,
                "person_income": person_income,
                "person_emp_length": person_emp_length,
                "loan_amnt": loan_amnt,
                "loan_int_rate": loan_int_rate,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
                "cb_person_default_on_file": cb_person_default_on_file,
                "person_home_ownership": person_home_ownership,
                "loan_intent": loan_intent,
            }

            df = processor.form_to_dataframe(form_data)  # single-row DataFrame
            result = predictor.predict_batch(df, threshold=decision_threshold)

            # Save to session_state so UI persists even when other widgets change
            save_single_prediction_to_session(df.iloc[0], result.iloc[0])

            st.success("✅ Prediction Complete")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Display cached last single result (if available)
    if "last_single_result" in st.session_state and "last_single_input" in st.session_state:
        last_res = st.session_state["last_single_result"]
        last_input = st.session_state["last_single_input"]

        # Show metrics
        risk_score = last_res.get("risk_score_percent", None)
        default_prob = last_res.get("default_probability", None)
        risk_category = last_res.get("risk_category", None)

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            if risk_score is not None:
                st.metric("Risk Score (Probability of Default)", f"{risk_score:.1f}%")
        with col_res2:
            if risk_category is not None:
                st.write("Risk Category")
                if risk_category == "Approved ✅":
                    bg = "#d4edda"
                    fg = "#155724"
                elif risk_category == "Medium Risk":
                    bg = "#fff3cd"
                    fg = "#856404"
                else:
                    bg = "#f8d7da"
                    fg = "#721c24"

                html = f"""
                <div style="
                    display:inline-block;
                    padding:0.35rem 0.75rem;
                    border-radius:0.5rem;
                    background-color:{bg};
                    color:{fg};
                    font-weight:600;
                ">
                    {risk_category}
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

        st.markdown(f"**Decision Threshold Used:** {decision_threshold:.2f}")

        # generate feedback using the processor (pass a Series)
        row_series = pd.Series(last_input)
        feedback = processor.generate_application_feedback(row_series, default_prob)

        st.markdown("---")
        st.subheader("**What Looks Good in Your Application**")
        for item in feedback["good"]:
            st.markdown(f"- {item}")

        st.subheader("**What needs improvement:**")
        for item in feedback["improve"]:
            st.markdown(f"- {item}")

        st.subheader("Overall Assessment")
        st.write(feedback["overall"])

# -------------------------
# TAB 2: BATCH UPLOAD
# -------------------------
with tab2:
    st.subheader("Upload Batch Applications")

    st.markdown(
        """
        #### Required File Format (CSV/Excel)
        Columns required (case-sensitive):
        - person_age, person_income, person_emp_length
        - loan_amnt, loan_int_rate
        - cb_person_cred_hist_length, cb_person_default_on_file
        - person_home_ownership, loan_intent
        """
    )

    schema_df = processor.get_expected_schema()
    with st.expander("View Required Columns and Examples"):
        st.dataframe(schema_df, width="stretch")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file with loan applications",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        key="batch_uploader"
    )

    # If a file is uploaded, cache the parsed dataframe in session_state (so widget changes don't drop it)
    if uploaded_file is not None:
        # Use cached loader (note: load_file_dataframe uses _processor as first arg)
        try:
            df = load_file_dataframe(processor, uploaded_file)
            # store original df in session state so other widgets won't cause reload
            st.session_state["uploaded_df"] = df
        except Exception as e:
            st.error(f"Error processing file: {e}")
            df = None
    else:
        df = st.session_state.get("uploaded_df", None)

    if df is not None:
        # Add derived column (safe to recompute)
        df = df.copy()
        df["loan_percent_income"] = df["loan_amnt"] / df["person_income"].replace(0, pd.NA)

        st.write(f"📊 Loaded {len(df)} applications")

        # A single button that runs predict_batch and stores results in session_state
        if st.button("🔍 Calculate Risk for All", key="batch_predict"):
            with st.spinner("Processing batch predictions..."):
                try:
                    result_df = predictor.predict_batch(df, threshold=decision_threshold)
                    # store results in session state so they persist across reruns
                    st.session_state["batch_result_df"] = result_df
                    st.success("✅ Batch processing complete")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        # If we have cached results, show them
        result_df = st.session_state.get("batch_result_df", None)
        if result_df is not None:
            # Summary by risk category
            col_s1, col_s2, col_s3 = st.columns(3)
            low = (result_df["risk_category"].astype(str).str.contains("Approved ✅")).sum()
            med = (result_df["risk_category"].astype(str).str.contains("Medium")).sum()
            high = (result_df["risk_category"].astype(str).str.contains("High")).sum()
            total = len(result_df)

            with col_s1:
                st.metric("Approved", low, f"{low / total * 100:.1f}%")
            with col_s2:
                st.metric("Medium Risk", med, f"{med / total * 100:.1f}%")
            with col_s3:
                st.metric("High Risk", high, f"{high / total * 100:.1f}%")

            st.markdown(f"**Decision Threshold Used:** {decision_threshold:.2f}")

            st.markdown("---")
            st.subheader("Results Table")
            st.dataframe(result_df.sort_values("risk_score_percent", ascending=False), use_container_width=True)

            # --------------------------
            # Row-by-row feedback viewer (uses session_state to persist selection)
            # --------------------------
            if "row_index" not in st.session_state:
                st.session_state["row_index"] = 0

            row_index = st.number_input(
                "Enter row number (0-based index) to inspect:",
                min_value=0,
                max_value=len(result_df) - 1,
                value=st.session_state["row_index"],
                step=1,
                key="batch_row_index"
            )
            # persist choice
            st.session_state["row_index"] = int(row_index)

            selected_row = df.iloc[int(st.session_state["row_index"])]
            row_prob = result_df["default_probability"].iloc[int(st.session_state["row_index"])]
            feedback = processor.generate_application_feedback(selected_row, row_prob)

            st.markdown(f"### Application #{int(st.session_state['row_index'])} Feedback")
            st.write(f"- **Risk Score:** {row_prob:.1%}")

            st.markdown("**What Looks Good in This Application**")
            for item in feedback["good"]:
                st.markdown(f"- {item}")

            st.markdown("**What Needs Improvement**")
            for item in feedback["improve"]:
                st.markdown(f"- {item}")

            st.markdown("**Overall Assessment**")
            st.write(feedback["overall"])

            # Download
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name="credit_risk_results.csv",
                mime="text/csv",
            )
