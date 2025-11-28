import streamlit as st
import pandas as pd
from pathlib import Path
import sys

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

# Add threshold slider in sidebar
st.sidebar.markdown("### ⚙️ Model Settings")
decision_threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.30,
    max_value=0.70,
    value=0.50,
    step=0.05,
    help="Lower threshold = more conservative (catches more defaults, more false positives). Higher threshold = more lenient (approves more loans, misses some defaults)."
)

st.sidebar.markdown(f"""
#### Current Threshold: {decision_threshold:.2f}
- **Precision** (at 0.50): ~93% - Few false approvals
- **Recall** (at 0.50): ~69% - Catches most defaults
- Lower threshold (0.30-0.40): More conservative
- Higher threshold (0.60-0.70): More lenient
""")

# Tabs for single vs batch
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
        cb_person_default_on_file = st.selectbox("Prior Default?", ["N", "Y"])

    loan_intent = st.selectbox("Loan Intent",
                               ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

    # Calculate loan_percent_income
    if person_income > 0:
        loan_percent_income = loan_amnt / person_income
    else:
        loan_percent_income = 0

    # Predict button
    if st.button("🔍 Calculate Risk", key="single_predict"):
        try:
            form_data = {
                'person_age': person_age,
                'person_income': person_income,
                'person_emp_length': person_emp_length,
                'loan_amnt': loan_amnt,
                'loan_int_rate': loan_int_rate,
                'cb_person_cred_hist_length': cb_person_cred_hist_length,
                'cb_person_default_on_file': cb_person_default_on_file,
                'person_home_ownership': person_home_ownership,
                'loan_intent': loan_intent,
            }

            df = processor.form_to_dataframe(form_data)
            result = predictor.predict_batch(df, threshold=decision_threshold)

            # Display result
            risk_score = result['risk_score_percent'].values[0]
            risk_category = result['risk_category'].values[0]
            default_prob = result['default_probability'].values[0]

            st.success("✅ Prediction Complete")

            col_result1, col_result2, col_result3 = st.columns(3)
            with col_result1:
                st.metric("Risk Score", f"{risk_score:.1f}%")
            with col_result2:
                st.metric("Default Probability", f"{default_prob:.1%}")
            with col_result3:
                color = "🔴" if risk_category == "High Risk" else "🟡" if risk_category == "Medium Risk" else "🟢"
                st.metric("Risk Category", f"{color} {risk_category}")

            st.markdown("---")
            st.markdown(f"**Decision Threshold Used:** {decision_threshold:.2f}")

            # Interpretation
            st.markdown("---")
            if risk_score < 30:
                st.info("✅ **Low Risk** - Strong probability of repayment. Recommend approval.")
            elif risk_score < 60:
                st.warning("⚠️ **Medium Risk** - Moderate repayment probability. Consider additional verification.")
            else:
                st.error("❌ **High Risk** - Elevated default probability. Recommend careful review or rejection.")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ===================== TAB 2: BATCH UPLOAD =====================
with tab2:
    st.subheader("Upload Batch Applications")
    st.markdown("**Note:** CSV/Excel can include all standard fields. `loan_grade` will be ignored if present.")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file with loan applications",
        type=['csv', 'xlsx', 'xls'],
        help="File must contain all required columns"
    )

    if uploaded_file:
        try:
            df = processor.csv_to_dataframe(uploaded_file)

            # Calculate loan_percent_income if not provided
            df['loan_percent_income'] = df['loan_amnt'] / df['person_income']

            st.write(f"📊 Loaded {len(df)} applications")

            if st.button("🔍 Calculate Risk for All", key="batch_predict"):
                with st.spinner("Processing..."):
                    result_df = predictor.predict_batch(df, threshold=decision_threshold)

                st.success("✅ Batch processing complete")

                # Display summary
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    low = (result_df['risk_category'] == 'Low Risk').sum()
                    st.metric("Low Risk", low, f"{low / len(result_df) * 100:.1f}%")
                with col_s2:
                    med = (result_df['risk_category'] == 'Medium Risk').sum()
                    st.metric("Medium Risk", med, f"{med / len(result_df) * 100:.1f}%")
                with col_s3:
                    high = (result_df['risk_category'] == 'High Risk').sum()
                    st.metric("High Risk", high, f"{high / len(result_df) * 100:.1f}%")

                st.markdown(f"**Decision Threshold Used:** {decision_threshold:.2f}")

                st.markdown("---")
                st.subheader("Results Table")
                st.dataframe(result_df.sort_values('risk_score_percent', ascending=False), use_container_width=True)

                # Download
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="credit_risk_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
