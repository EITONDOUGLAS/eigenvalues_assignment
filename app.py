
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# 1. DASHBOARD CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🏦 Loan Analysis and Prediction Dashboard")
st.caption(
    "Interactive loan application analysis and machine learning insights"
)

# --------------------------------------------------
# 2. LOAD DATA AND TRAINED MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("loan_dashboard_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("loan_model_bundle.joblib")

df = load_data()
bundle = load_model()

model = bundle["model"]
model_name = bundle["model_name"]

TARGET = bundle["target"]

if TARGET not in df.columns:
    st.error("Target column not found in dataset.")
    st.stop()

# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def find_column(candidates, columns):
    for candidate in candidates:
        for column in columns:
            if candidate.lower() in column.lower():
                return column
    return None

def get_status_data(data):
    return data[data[TARGET].isin([0, 1])].copy()

# Automatically identify common columns
gender_col = find_column(
    ["gender", "sex"], df.columns
)

marital_col = find_column(
    ["marital_status", "married"], df.columns
)

education_col = find_column(
    ["education"], df.columns
)

property_col = find_column(
    ["property_area", "location"], df.columns
)

income_col = find_column(
    ["applicant_income", "income"], df.columns
)

co_income_col = find_column(
    ["coapplicant_income", "co_applicant_income"], df.columns
)

employment_col = find_column(
    ["employment_status", "employment", "self_employed"], df.columns
)

loan_amount_col = find_column(
    ["loan_amount"], df.columns
)

purpose_col = find_column(
    ["loan_purpose", "purpose", "loan_product", "product"], df.columns
)

credit_col = find_column(
    ["credit_history", "credit_score", "cibil_score"], df.columns
)

# --------------------------------------------------
# 4. SIDEBAR AND INTERACTIVE FILTERS
# --------------------------------------------------

st.sidebar.title("Dashboard Filters")

filtered_df = get_status_data(df)

status_filter = st.sidebar.selectbox(
    "Loan Application Status",
    ["All", "Approved", "Rejected"]
)

if status_filter == "Approved":
    filtered_df = filtered_df[
        filtered_df[TARGET] == 1
    ]

elif status_filter == "Rejected":
    filtered_df = filtered_df[
        filtered_df[TARGET] == 0
    ]

# Select an available categorical filter
categorical_columns = (
    filtered_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
)

filter_column = st.sidebar.selectbox(
    "Filter by Category",
    ["None"] + categorical_columns
)

if filter_column != "None":
    options = sorted(
        filtered_df[filter_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_category = st.sidebar.multiselect(
        f"Select {filter_column}",
        options
    )

    if selected_category:
        filtered_df = filtered_df[
            filtered_df[filter_column]
            .astype(str)
            .isin(selected_category)
        ]

# --------------------------------------------------
# 5. KEY PERFORMANCE INDICATORS
# --------------------------------------------------

st.header("5.2 Key Performance Indicators")

total = len(filtered_df)

approved = (
    filtered_df[TARGET] == 1
).sum()

rejected = (
    filtered_df[TARGET] == 0
).sum()

approval_rate = (
    approved / total * 100 if total > 0 else 0
)

rejection_rate = (
    rejected / total * 100 if total > 0 else 0
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Applications", f"{total:,}")
col2.metric("Approved", f"{approved:,}")
col3.metric("Rejected", f"{rejected:,}")
col4.metric("Approval Rate", f"{approval_rate:.2f}%")
col5.metric("Rejection Rate", f"{rejection_rate:.2f}%")

st.divider()

# --------------------------------------------------
# 6. LOAN APPROVAL AND REJECTION ANALYSIS
# --------------------------------------------------

st.header("5.3 Loan Approval and Rejection Analysis")

if total > 0:

    status_counts = (
        filtered_df[TARGET]
        .map({0: "Rejected", 1: "Approved"})
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Applications")
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            status_counts,
            names="Status",
            values="Applications",
            title="Loan Approval Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            status_counts,
            x="Status",
            y="Applications",
            color="Status",
            title="Approved vs Rejected Applications"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No records match the selected filters.")

# --------------------------------------------------
# 7. APPLICANT DEMOGRAPHIC ANALYSIS
# --------------------------------------------------

st.header("5.4 Applicant Demographic Analysis")

demographic_columns = [
    gender_col,
    marital_col,
    education_col,
    property_col
]

demographic_columns = [
    col for col in demographic_columns
    if col is not None
]

if demographic_columns:

    selected_demo = st.selectbox(
        "Select Demographic Variable",
        demographic_columns
    )

    demo_data = (
        filtered_df.groupby(
            [selected_demo, TARGET],
            dropna=False
        )
        .size()
        .reset_index(name="Applications")
    )

    demo_data["Loan Status"] = demo_data[TARGET].map({
        0: "Rejected",
        1: "Approved"
    })

    fig = px.bar(
        demo_data,
        x=selected_demo,
        y="Applications",
        color="Loan Status",
        barmode="group",
        title=f"Loan Outcomes by {selected_demo}"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No demographic columns were identified.")

# --------------------------------------------------
# 8. INCOME AND EMPLOYMENT ANALYSIS
# --------------------------------------------------

st.header("5.5 Income and Employment Analysis")

income_columns = [
    income_col,
    co_income_col,
    loan_amount_col
]

income_columns = [
    col for col in income_columns
    if col is not None
]

if income_columns:

    selected_income = st.selectbox(
        "Select Financial Variable",
        income_columns
    )

    fig = px.histogram(
        filtered_df,
        x=selected_income,
        color=TARGET,
        nbins=30,
        marginal="box",
        title=f"Distribution of {selected_income}",
        labels={TARGET: "Loan Status"}
    )

    st.plotly_chart(fig, use_container_width=True)

if employment_col is not None:

    employment_data = (
        filtered_df.groupby(
            [employment_col, TARGET],
            dropna=False
        )
        .size()
        .reset_index(name="Applications")
    )

    employment_data["Loan Status"] = (
        employment_data[TARGET].map({
            0: "Rejected",
            1: "Approved"
        })
    )

    fig = px.bar(
        employment_data,
        x=employment_col,
        y="Applications",
        color="Loan Status",
        barmode="group",
        title="Employment and Loan Outcomes"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 9. LOAN PRODUCT AND PURPOSE ANALYSIS
# --------------------------------------------------

st.header("5.6 Loan Product and Purpose Analysis")

if purpose_col is not None:

    purpose_data = (
        filtered_df.groupby(
            [purpose_col, TARGET],
            dropna=False
        )
        .size()
        .reset_index(name="Applications")
    )

    purpose_data["Loan Status"] = (
        purpose_data[TARGET].map({
            0: "Rejected",
            1: "Approved"
        })
    )

    fig = px.bar(
        purpose_data,
        x=purpose_col,
        y="Applications",
        color="Loan Status",
        barmode="group",
        title="Loan Applications by Product or Purpose"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No loan product or purpose column was identified.")

# --------------------------------------------------
# 10. CREDIT RISK ANALYSIS
# --------------------------------------------------

st.header("5.7 Credit Risk Analysis")

if credit_col is not None:

    credit_data = (
        filtered_df.groupby(
            [credit_col, TARGET],
            dropna=False
        )
        .size()
        .reset_index(name="Applications")
    )

    credit_data["Loan Status"] = (
        credit_data[TARGET].map({
            0: "Rejected",
            1: "Approved"
        })
    )

    fig = px.bar(
        credit_data,
        x=credit_col,
        y="Applications",
        color="Loan Status",
        barmode="group",
        title=f"Loan Outcomes by {credit_col}"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No recognized credit-related column was identified.")

# --------------------------------------------------
# 11. APPLICANT SEARCH AND RECORDS
# --------------------------------------------------

st.header("5.8 and 5.9 Interactive Filters and Applicant Records")

search_text = st.text_input(
    "Search Applicant Records",
    placeholder="Enter an applicant name, ID or other available detail..."
)

search_results = filtered_df.copy()

if search_text.strip():

    search_mask = search_results.apply(
        lambda column: column.astype(str).str.contains(
            search_text,
            case=False,
            na=False,
            regex=False
        )
    ).any(axis=1)

    search_results = search_results[search_mask]

st.write(f"Matching Records: {len(search_results)}")

st.dataframe(
    search_results,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 12. EXPORT AND REPORTING
# --------------------------------------------------

st.header("5.11 Export and Reporting")

csv_data = search_results.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Loan Records",
    data=csv_data,
    file_name="filtered_loan_records.csv",
    mime="text/csv"
)

report = f"""
LOAN ANALYSIS DASHBOARD REPORT

Total Applications: {total}
Approved Applications: {approved}
Rejected Applications: {rejected}

Approval Rate: {approval_rate:.2f}%
Rejection Rate: {rejection_rate:.2f}%

Selected Machine Learning Model: {model_name}
"""

st.download_button(
    label="Download Summary Report",
    data=report,
    file_name="loan_dashboard_report.txt",
    mime="text/plain"
)

# --------------------------------------------------
# 13. LOAN APPROVAL PREDICTION INTERFACE
# --------------------------------------------------

st.header("5.10 Loan Approval Prediction Interface")

st.write(
    f"Selected Machine Learning Model: **{model_name}**"
)

feature_columns = bundle["feature_columns"]

numerical_features = bundle["numerical_features"]

categorical_features = bundle["categorical_features"]

with st.form("loan_prediction_form"):

    st.subheader("Enter Applicant Information")

    applicant_input = {
        column: np.nan
        for column in feature_columns
    }

    for column in feature_columns:

        if column in numerical_features:

            default_value = float(
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).median()
            )

            if not np.isfinite(default_value):
                default_value = 0.0

            value = st.number_input(
                column.replace("_", " ").title(),
                value=default_value
            )

            applicant_input[column] = value

        elif column in categorical_features:

            options = (
                df[column]
                .dropna()
                .unique()
                .tolist()
            )

            if options:

                value = st.selectbox(
                    column.replace("_", " ").title(),
                    options
                )

                applicant_input[column] = value

            else:

                applicant_input[column] = st.text_input(
                    column.replace("_", " ").title()
                )

    submitted = st.form_submit_button(
        "Predict Loan Outcome"
    )

    if submitted:

        input_df = pd.DataFrame(
            [applicant_input],
            columns=feature_columns
        )

        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            positive_index = list(model.classes_).index(1)

            approval_probability = (
                probabilities[positive_index] * 100
            )

            st.metric(
                "Estimated Approval Probability",
                f"{approval_probability:.2f}%"
            )

        if prediction == 1:

            st.success(
                "Model Prediction: Approved"
            )

        else:

            st.warning(
                "Model Prediction: Rejected"
            )

        st.caption(
            "This is a model-generated educational prediction, "
            "not an actual lending decision."
        )

# --------------------------------------------------
# 14. DASHBOARD FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Loan Analysis and Machine Learning Dashboard | "
    "Developed using Python, Streamlit and Plotly"
)
