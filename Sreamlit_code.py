import streamlit as st
import pickle
import joblib
import pandas as pd


model = joblib.load(r'C:\Users\ravinder\Desktop\prudent_hac\best_models\logistic_regression.pkl')

# Function to make predictions
def predict_churn(features):
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)[0]
    return prediction

# Streamlit app
def main():
    st.title('Customer Churn Prediction')

    # Input fields for each feature
    senior_citizen = st.selectbox('Senior Citizen', [0, 1])
    partner = st.selectbox('Partner', [0, 1])
    dependents = st.selectbox('Dependents', [0, 1])
    paperless_billing = st.selectbox('Paperless Billing', [0, 1])
    monthly_charges = st.number_input('Monthly Charges')
    tenure = st.number_input('Tenure')
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

    # Predict button
    if st.button('Predict'):
        features = {
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'PaperlessBilling': paperless_billing,
            'MonthlyCharges': monthly_charges,
            'tenure': tenure,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaymentMethod': payment_method
        }
        prediction = predict_churn(features)
        st.write('Predicted Churn:', prediction)

if __name__ == "__main__":
    main()
