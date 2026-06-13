import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL + COLUMNS
# =========================
model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# =========================
# APP TITLE
# =========================
st.title("AI-Powered Churn Prediction and Retention Intelligence System")

st.write(
    "This app predicts customer churn risk and gives retention recommendations "
    "based on customer behaviour, pricing, contract type, and service usage."
)

# =========================
# USER INPUTS
# =========================
tenure = st.number_input(
    "Tenure / Months with company",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=1000.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=100000.0,
    value=850.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# =========================
# CREATE INPUT DATAFRAME
# =========================
input_data = pd.DataFrame(columns=feature_columns)
input_data.loc[0] = 0

# Numerical values
input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges
input_data["TotalCharges"] = total_charges

# Contract one-hot encoding
if "Contract_One year" in input_data.columns:
    input_data["Contract_One year"] = 1 if contract == "One year" else 0

if "Contract_Two year" in input_data.columns:
    input_data["Contract_Two year"] = 1 if contract == "Two year" else 0

# Internet one-hot encoding
if "InternetService_Fiber optic" in input_data.columns:
    input_data["InternetService_Fiber optic"] = 1 if internet == "Fiber optic" else 0

if "InternetService_No" in input_data.columns:
    input_data["InternetService_No"] = 1 if internet == "No" else 0

# Payment method one-hot encoding
if "PaymentMethod_Electronic check" in input_data.columns:
    input_data["PaymentMethod_Electronic check"] = 1 if payment == "Electronic check" else 0

if "PaymentMethod_Mailed check" in input_data.columns:
    input_data["PaymentMethod_Mailed check"] = 1 if payment == "Mailed check" else 0

if "PaymentMethod_Credit card (automatic)" in input_data.columns:
    input_data["PaymentMethod_Credit card (automatic)"] = 1 if payment == "Credit card (automatic)" else 0

# =========================
# PREDICTION
# =========================
if st.button("Predict Churn Risk"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if probability > 0.7:
        risk = "HIGH"
    elif probability > 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # =========================
    # OUTPUT
    # =========================
    st.subheader("Prediction Result")

    st.write(f"**Churn Probability:** {probability * 100:.2f}%")
    st.write(f"**Risk Level:** {risk}")

    if prediction == 1:
        st.error("Prediction: Customer is likely to churn")
    else:
        st.success("Prediction: Customer is likely to stay")

    # =========================
    # XAI EXPLANATION
    # =========================
    st.subheader("Factors Influencing Prediction")

    influential_factors = []

    if monthly_charges > 80:
        influential_factors.append(
            "High monthly charges may increase churn risk."
        )

    if tenure < 12:
        influential_factors.append(
            "Short customer tenure indicates lower loyalty."
        )

    if contract == "Month-to-month":
        influential_factors.append(
            "Month-to-month contracts make cancellation easier, increasing churn risk."
        )

    if internet == "Fiber optic":
        influential_factors.append(
            "Fiber optic users may have higher service expectations and pricing sensitivity."
        )

    if payment == "Electronic check":
        influential_factors.append(
            "Electronic check payment method is associated with higher churn in the dataset."
        )

    if len(influential_factors) == 0:
        influential_factors.append(
            "This customer profile appears relatively stable based on selected inputs."
        )

    for factor in influential_factors:
        st.write(f"• {factor}")

    # =========================
    # RETENTION RECOMMENDATION
    # =========================
    st.subheader("Retention Recommendation")

    if risk == "HIGH":
        st.write(
            "Offer a retention discount, personalised recommendations, "
            "or a better contract plan. This customer may need quick value "
            "to stay engaged."
        )

    elif risk == "MEDIUM":
        st.write(
            "Monitor customer engagement and provide targeted offers, "
            "content suggestions, or loyalty benefits."
        )

    else:
        st.write(
            "No immediate action needed. Continue maintaining customer "
            "satisfaction and engagement."
        )

    st.subheader("Modern User Behaviour Insight")

    st.write(
        "This project considers churn as more than just a pricing problem. "
        "Modern users are influenced by short-form content, social trends, "
        "peer influence, free trials, discounts, and quick entertainment value. "
        "If a platform does not capture attention quickly or provide personalised "
        "value, customers may lose interest and cancel their subscription."
    )