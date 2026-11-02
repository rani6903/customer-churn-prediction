import streamlit as st
import requests

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")

st.write("Enter customer information to predict churn.")

with st.form("prediction_form"):

    left_col, right_col = st.columns(2)

    # ==========================
    # Left Column
    # ==========================

    with left_col:

        st.subheader("👤 Customer Information")

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            0,
            72,
            12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

    # ==========================
    # Right Column
    # ==========================

    with right_col:

        st.subheader("💳 Services & Billing")

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
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

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    submit = st.form_submit_button(
        "🔮 Predict Churn",
        use_container_width=True
    )

if submit:

    payload = {

        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    response = requests.post(
        "http://127.0.0.1:8000/api/v1/predict",
        json=payload
    )

    if response.status_code == 200:

        result = response.json()

        st.success(result["prediction"])

        # st.metric(
        #     "Probability",
        #     f"{result['probability']:.2f}%"
        # )
        st.metric(
        "Probability",
        f"{result['probability']:.2f}%"
        )

        st.progress(result['probability'] / 100)

        with st.expander("Prediction Details"):
            st.json(result)
    
        if "top_features" in result:

            st.subheader("Top Factors")

            for feature in result["top_features"]:

                st.write(
                    f"**{feature['feature']}** : {feature['impact']:.3f}"
                )

            st.image(
                "artifacts/explanations/waterfall_plot.png",
                caption="Prediction Explanation"
            )
            st.markdown("""
            <style>

            .main{
                padding-top:1rem;
            }

            h1{
                color:#0E76A8;
            }

            </style>
            """, unsafe_allow_html=True)

    else:

        st.error("Prediction failed.")

        
    

    