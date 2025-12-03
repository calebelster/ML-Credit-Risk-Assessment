import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add app directory to path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from utils.predictor import CreditRiskPredictor
from utils.data_processor import DataProcessor

st.set_page_config(page_title="Risk Calculator", layout="wide", initial_sidebar_state="expanded")

st.title("💳 Credit Risk Assessment")
st.markdown("**Assess loan default risk for individual or batch applications**")

# Initialize
predictor = CreditRiskPredictor()
processor = DataProcessor()

# Sidebar threshold
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

# Tabs
tab1, tab2 = st.tabs(["Single Application", "Batch Upload"])

# ===================== TAB 1: SINGLE APPLICATION =====================
with tab1:
    st.subheader("Enter Loan Application Details")

    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        person_emp_length = st.number_input("Employment Length (years)", min_value=0, value=5)
        loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)

    with col2:
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=7.5)
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=10)
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        cb_person_default_on_file = st.selectbox("Prior Default on File?", ["N", "Y"])

    loan_intent = st.selectbox(
        "Loan Intent",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    )

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

            df = processor.form_to_dataframe(form_data)
            result = predictor.predict_batch(df, threshold=decision_threshold)

            risk_score = result["risk_score_percent"].values[0]
            default_prob = result["default_probability"].values[0]
            risk_category = result["risk_category"].values[0]
            

            st.success("✅ Prediction Complete")

            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Risk Score (Probability of Default)", f"{risk_score:.1f}%")
            with col_res2:
                st.write("Risk Category")
                if risk_category == "Approved ✅":
                    bg = "#d4edda"   # light green
                    fg = "#155724"
                elif risk_category == "Medium Risk":
                    bg = "#fff3cd"   # light yellow
                    fg = "#856404"
                else:
                    bg = "#f8d7da"   # light red
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

            # Tailored feedback
            feedback = processor.generate_application_feedback(df.iloc[0], default_prob)

            st.markdown("---")
            st.markdown("**What Looks Good in Your Application**")
            for item in feedback["good"]:
                st.markdown(f"- {item}")

            st.markdown("**What needs improvement:**")
            for item in feedback["improve"]:
                st.markdown(f"- {item}")

            st.subheader("Overall Assessment")
            st.write(feedback["overall"])

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ===================== TAB 2: BATCH UPLOAD =====================
with tab2:
    st.subheader("Upload Batch Applications")

    st.markdown(
        """
        #### Required File Format

        Your CSV or Excel file must contain the following columns (case-sensitive):

        - `person_age` (integer): Age in years (≥ 18).
        - `person_income` (float): Annual gross income in USD.
        - `person_emp_length` (integer): Years in current employment.
        - `loan_amnt` (float): Requested loan amount in USD.
        - `loan_int_rate` (float): Nominal interest rate in percent.
        - `cb_person_cred_hist_length` (integer): Years since first credit line.
        - `person_home_ownership` (string): One of `RENT`, `OWN`, `MORTGAGE`, `OTHER`.
        - `loan_intent` (string): One of `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION`.
        - `cb_person_default_on_file` (string): `Y` if there is a prior default, `N` otherwise.

        Additional columns will be ignored (for example, `loan_grade` is not used).
        """
    )

    schema_df = processor.get_expected_schema()
    with st.expander("View Required Columns and Examples"):
        st.dataframe(schema_df, width="stretch")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file with loan applications",
        type=["csv", "xlsx", "xls"],
        help="File must contain all required columns shown above.",
    )

    if uploaded_file:
        try:
            df = processor.csv_to_dataframe(uploaded_file)

            # Optional derived feature for interpretation (not used directly by model)
            df["loan_percent_income"] = df["loan_amnt"] / df["person_income"].replace(0, pd.NA)

            st.write(f"📊 Loaded {len(df)} applications")

            if st.button("🔍 Calculate Risk for All", key="batch_predict"):
                with st.spinner("Processing..."):
                    result_df = predictor.predict_batch(df, threshold=decision_threshold)

                st.success("✅ Batch processing complete")

                # Summary by risk category
                col_s1, col_s2, col_s3 = st.columns(3)
                low = (result_df["risk_category"].astype(str).str.contains("Low")).sum()
                med = (result_df["risk_category"].astype(str).str.contains("Medium")).sum()
                high = (result_df["risk_category"].astype(str).str.contains("High")).sum()
                total = len(result_df)

                with col_s1:
                    st.metric("Low Risk", low, f"{low / total * 100:.1f}%")
                with col_s2:
                    st.metric("Medium Risk", med, f"{med / total * 100:.1f}%")
                with col_s3:
                    st.metric("High Risk", high, f"{high / total * 100:.1f}%")

                st.markdown(f"**Decision Threshold Used:** {decision_threshold:.2f}")

                st.markdown("---")
                st.subheader("Results Table")
                st.dataframe(
                    result_df.sort_values("risk_score_percent", ascending=False),
                    width="stretch",
                )

                # Optional: per-row feedback via selection
                st.markdown("#### View Feedback for a Single Application")
                row_index = st.number_input(
                    "Enter row number (0-based index) to inspect:",
                    min_value=0,
                    max_value=len(result_df) - 1,
                    value=0,
                    step=1,
                )
                selected_row = df.iloc[int(row_index)]
                row_prob = result_df["default_probability"].iloc[int(row_index)]
                feedback = processor.generate_application_feedback(selected_row, row_prob)

        
                st.markdown(f"**Application #{int(row_index)} Feedback**")
                st.write(f"- Risk Score (Probability of Default): {row_prob:.1%}")

                st.markdown("**What Looks Good in Your Application**")
                for item in feedback["good"]:
                    st.markdown(f"- {item}")

                st.markdown("**What needs improvement:**")
                for item in feedback["improve"]:
                    st.markdown(f"- {item}")

                st.markdown("Overall Assessment")
                st.write(feedback["overall"])


                # Download
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="credit_risk_results.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")