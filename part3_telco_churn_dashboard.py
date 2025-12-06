import streamlit as st
import requests
import pandas as pd
import json

# --- CONFIGURATION ---
# PASTE YOUR LAMBDA API URL HERE
API_URL = "https://yhj9s420fi.execute-api.us-east-1.amazonaws.com/default/DSCI352_PredictTelcoChurn"

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")

st.title("Telco Customer Churn Predictor")
st.markdown("""
This app connects to a **Serverless AWS Lambda** model. 
Adjust the customer profile below to see how the probability of churn changes in real-time.
""")

# --- TABS ---
tab1, tab2 = st.tabs(["Single Customer Analysis", "Batch Prediction (CSV)"])

# === TAB 1: FULL MANUAL ENTRY ===
with tab1:
    st.header("Customer Profile")
    
    # Organize inputs into 3 neat columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        st.subheader("Services")
        tenure = st.slider("Tenure (Months)", 0, 72, 24)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        # MultipleLines depends on PhoneService usually, but model treats them as categories
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with col2:
        st.subheader("Internet Services")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        st.subheader("Contract & Billing")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1500.0, step=10.0)

    # Payload construction using ALL interactive variables
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    st.write("---")
    if st.button("Predict Churn Risk", type="primary"):
        with st.spinner("Invoking AWS Lambda..."):
            try:
                response = requests.post(API_URL, json=customer_data)
                
                if response.status_code == 200:
                    result = response.json()
                    prob = result['churn_probability']
                    pred = result['prediction']
                    
                    # Visual Gauge
                    st.metric("Churn Probability", f"{prob:.2%}")
                    
                    if prob > 0.5:
                        st.error(f"⚠️ **High Risk** (Prediction: {pred})")
                        st.progress(prob)
                    else:
                        st.success(f"✅ **Low Risk** (Prediction: {pred})")
                        st.progress(prob)
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection failed: {e}")

# === TAB 2: BATCH UPLOAD (Keep this same as before) ===
with tab2:
    st.header("Batch Prediction via CSV")
    uploaded_file = st.file_uploader("Upload File", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        
        if st.button("Run Batch Prediction"):
            results = []
            bar = st.progress(0)
            
            for i, row in df.iterrows():
                # Clean row data
                data = row.where(pd.notnull(row), None).to_dict()
                if "Churn" in data: del data["Churn"]
                
                try:
                    resp = requests.post(API_URL, json=data)
                    if resp.status_code == 200:
                        results.append(resp.json())
                    else:
                        results.append({"prediction": "Error", "churn_probability": 0})
                except:
                    results.append({"prediction": "Error", "churn_probability": 0})
                
                bar.progress((i + 1) / len(df))
            
            # Combine results
            final_df = pd.concat([df, pd.DataFrame(results)], axis=1)
            st.success("Done!")
            st.dataframe(final_df)