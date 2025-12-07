import streamlit as st
import requests
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px

# CONFIGURATION
API_URL = "https://yhj9s420fi.execute-api.us-east-1.amazonaws.com/default/DSCI352_PredictTelcoChurn"

# Set page config must be the first command
st.set_page_config(page_title="Telco Churn Predictor", page_icon="📱", layout="wide")

# HELPER FUNCTIONS
def create_gauge(probability):
    """Creates a gauge chart for churn probability."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def convert_df(df):
    """Converts dataframe to CSV for download."""
    return df.to_csv(index=False).encode('utf-8')

# SIDEBAR
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Amazon_Web_Services_Logo.svg/1024px-Amazon_Web_Services_Logo.svg.png", width=150)
    st.title("Project Controls")
    st.markdown("This dashboard connects to an **AWS Lambda** backend trained on the Telco Churn dataset.")
    st.markdown("---")
    st.write("**Model:** Logistic Regression (SGD)")
    st.write("**Backend:** Python 3.10 (Serverless)")

###########
# MAIN PAGE
###########
st.title("Telco Customer Churn Predictor")

# TABS
tab1, tab2 = st.tabs(["Single Customer Analysis", "Batch Prediction Dashboard"])

# TAB 1: SINGLE CUSTOMER
with tab1:
    st.markdown("### Configure Customer Profile")
    
    # Input Form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (Months)", 0, 72, 24)

        with col2:
            st.subheader("Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with col3:
            st.subheader("Billing")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)

        submit_val = st.form_submit_button("Predict Churn Risk", type="primary")

    if submit_val:
        # Construct Payload
        customer_data = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
        }

        # API Call
        with st.spinner("Analyzing with AWS Lambda..."):
            try:
                response = requests.post(API_URL, json=customer_data)
                if response.status_code == 200:
                    result = response.json()
                    prob = result['churn_probability']
                    pred = result['prediction']
                    
                    # --- RESULTS UI ---
                    st.divider()
                    r_col1, r_col2 = st.columns([1, 2])
                    
                    with r_col1:
                        # Display Gauge Chart
                        st.plotly_chart(create_gauge(prob), use_container_width=True)
                    
                    with r_col2:
                        st.subheader("Prediction Analysis")
                        if prob > 0.5:
                            st.error(f"**High Risk of Churn**")
                            st.write(f"This customer has a **{prob:.1%}** probability of leaving.")
                            st.write("Recommendation: Offer a discount or longer contract.")
                        else:
                            st.success(f"**Loyal Customer**")
                            st.write(f"This customer has a **{prob:.1%}** probability of leaving.")
                            st.write("Recommendation: Upsell new services.")
                            
                        # JSON Debug (Expandable)
                        with st.expander("View Raw JSON Response"):
                            st.json(result)
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")


# TAB 2: BATCH PREDICTION
with tab2:
    st.header("Batch Analytics Dashboard")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload CSV or JSON file", type=["csv", "json"])
    
    # Download Sample Button (Helpful for users)
    sample_csv = "gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\nFemale,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85"
    st.download_button("Download Sample CSV Template", sample_csv, "sample_input.csv", "text/csv")

    if uploaded_file is not None:
        # Load Data
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            raw = json.load(uploaded_file)
            df = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame([raw])
        
        st.write(f"Loaded **{len(df)}** customers.")
        
        if st.button("Run Batch Prediction", type="primary"):
            results = []
            probs = []
            
            # Progress Bar
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            # Loop through rows
            for i, row in df.iterrows():
                data = row.where(pd.notnull(row), None).to_dict()
                if "Churn" in data: del data["Churn"]
                
                try:
                    resp = requests.post(API_URL, json=data)
                    if resp.status_code == 200:
                        res_json = resp.json()
                        results.append(res_json['prediction'])
                        probs.append(res_json['churn_probability'])
                    else:
                        results.append("Error")
                        probs.append(0.0)
                except:
                    results.append("Error")
                    probs.append(0.0)
                
                my_bar.progress((i + 1) / len(df), text=f"Processing row {i+1}/{len(df)}")
            
            my_bar.empty()
            
            # Combine Results
            df["Prediction"] = results
            df["Probability"] = probs
            
            # --- DASHBOARD VISUALS ---
            st.divider()
            st.subheader("Analytics Report")
            
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            churn_count = df[df["Prediction"] == "Yes"].shape[0]
            avg_prob = df["Probability"].mean()
            
            m1.metric("Total Customers", len(df))
            m2.metric("Predicted Churners", churn_count, delta_color="inverse")
            m3.metric("Average Churn Risk", f"{avg_prob:.1%}")
            
            # Charts Row
            c1, c2 = st.columns(2)
            
            with c1:
                # Pie Chart
                fig_pie = px.pie(df, names='Prediction', title='Churn Distribution', 
                                 color='Prediction', color_discrete_map={'Yes':'red', 'No':'green', 'Error':'gray'})
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with c2:
                # Histogram
                fig_hist = px.histogram(df, x="Probability", nbins=20, title="Risk Probability Distribution",
                                        labels={'Probability':'Churn Probability'}, color_discrete_sequence=['darkblue'])
                fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
                st.plotly_chart(fig_hist, use_container_width=True)

            # Detailed Data Table
            st.subheader("Detailed Results")
            
            # Color coding function
            def color_risk(val):
                color = 'red' if val > 0.5 else 'green'
                return f'color: {color}; font-weight: bold'

            st.dataframe(
                df.style.map(color_risk, subset=['Probability']),
                use_container_width=True
            )
            
            # Download Results
            st.download_button(
                "Download Predictions (CSV)", 
                convert_df(df), 
                "telco_predictions.csv", 
                "text/csv", 
                key='download-csv'
            )
