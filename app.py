import streamlit as st
import joblib
import pandas as pd

# Function to process the form data and make predictions
def process_form(credit_history_age, monthly_balance, monthly_inhand_salary, annual_income,
                 interest_rate, outstanding_debt, num_of_loan, delay_from_due_date,
                 num_of_delayed_payment, total_emi_per_month, num_credit_inquiries):

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'credit_history_age': [credit_history_age],
        'monthly_balance': [monthly_balance],
        'monthly_inhand_salary': [monthly_inhand_salary],
        'annual_income': [annual_income],
        'interest_rate': [interest_rate],
        'outstanding_debt': [outstanding_debt],
        'num_of_loan': [num_of_loan],
        'delay_from_due_date': [delay_from_due_date],
        'num_of_delayed_payment': [num_of_delayed_payment],
        'total_emi_per_month': [total_emi_per_month],
        'num_credit_inquiries': [num_credit_inquiries]
    })

    # Load the trained model
    model_path = 'content/logistic_regression_model.pkl'
    with open(model_path, 'rb') as file:
        model = joblib.load(file)

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Mapping credit score to labels
    credit_score_label = {0: "Poor", 1: "Average", 2: "Good"}
    predicted_score = credit_score_label[prediction[0]]
    confidence = max(prediction_proba[0])

    # Display result
    st.success(f"Predicted Credit Score: {predicted_score} (Confidence: {confidence:.2%})")

# Function to display the input form
def display_form():
    st.title("Credit Score Prediction")
    st.subheader("Enter Your Details")

    with st.form(key='credit_score_form'):
        credit_history_age = st.number_input("Credit History Age (in weeks)", min_value=0)
        monthly_balance = st.number_input("Monthly Balance ($)", min_value=0.0)
        monthly_inhand_salary = st.number_input("Monthly Inhand Salary ($)", min_value=0.0)
        annual_income = st.number_input("Annual Income ($)", min_value=0.0)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0)
        outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0)
        num_of_loan = st.number_input("Number of Loans", min_value=0)
        delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value=0)
        num_of_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0)
        total_emi_per_month = st.number_input("Total EMI per Month ($)", min_value=0.0)
        num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        process_form(credit_history_age, monthly_balance, monthly_inhand_salary, annual_income,
                     interest_rate, outstanding_debt, num_of_loan, delay_from_due_date,
                     num_of_delayed_payment, total_emi_per_month, num_credit_inquiries)

# Run the Streamlit app
if __name__ == '__main__':
    display_form()