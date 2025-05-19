import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import random
from datetime import datetime
import psycopg2

# Function to connect to the PostgreSQL database
def connect_db():
    try:
        conn = psycopg2.connect(
            host="localhost",        # e.g., "localhost" or IP address of your database server
            database="history", # your database name
            user="postgres",        # your PostgreSQL username
            password="yanno" # your PostgreSQL password
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

def save_to_database(name, age, profession, ssn, credit_history_age, monthly_balance, monthly_inhand_salary,
                     annual_income, interest_rate, outstanding_debt, num_of_loan, delay_from_due_date,
                     num_of_delayed_payment, total_emi_per_month, num_credit_inquiries, predicted_score):
    
    # Connect to the database
    conn = connect_db()
    if conn:
        try:
            # Create a cursor object
            cursor = conn.cursor()

            # Define your SQL insert query
            query = """
            INSERT INTO credit_history (name, age, profession, ssn, credit_history_age, monthly_balance,
                monthly_inhand_salary, annual_income, interest_rate, outstanding_debt, num_of_loan, delay_from_due_date,
                num_of_delayed_payment, total_emi_per_month, num_credit_inquiries, predicted_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Execute the query with the user data
            cursor.execute(query, (name, age, profession, ssn, credit_history_age, monthly_balance, monthly_inhand_salary,
                                   annual_income, interest_rate, outstanding_debt, num_of_loan, delay_from_due_date,
                                   num_of_delayed_payment, total_emi_per_month, num_credit_inquiries, predicted_score))

            # Commit the data
            conn.commit()
            st.success("Data saved successfully!")
        except Exception as e:
            st.error(f"Error saving to database: {e}")
        finally:
            # Close the connection
            cursor.close()
            conn.close()

# Function to fetch the latest 5 entries from the database
def fetch_latest_entries():
    conn = connect_db()
    if conn:
        try:
            cursor = conn.cursor()
            # Query to get the latest 5 entries based on the timestamp or ID (assuming ID is auto-incremented)
            query = """
            SELECT 'REDACTED' AS name, age, profession, 'REDACTED' AS ssn, credit_history_age, monthly_balance, 
                   monthly_inhand_salary, annual_income, interest_rate, outstanding_debt,
                   num_of_loan, delay_from_due_date, num_of_delayed_payment, 
                   total_emi_per_month, num_credit_inquiries, predicted_score
            FROM credit_history
            ORDER BY id DESC
            LIMIT 5;
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            st.error(f"Error fetching data from database: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    return []

# Function to display the latest entries
def display_latest_entries():
    st.subheader("Latest 5 Credit Score Predictions")
    entries = fetch_latest_entries()

    if not entries:
        st.info("No entries found.")
        return

    # Create a DataFrame from the fetched entries
    df = pd.DataFrame(entries, columns=[
        'Name', 'Age', 'Profession', 'SSN', 'Credit History Age', 'Monthly Balance',
        'Monthly Inhand Salary', 'Annual Income', 'Interest Rate', 'Outstanding Debt',
        'Number of Loans', 'Delay from Due Date', 'Number of Delayed Payments',
        'Total EMI per Month', 'Number of Credit Inquiries', 'Predicted Score'
    ])

    # Display the DataFrame in Streamlit
    st.dataframe(df)

# Function to process the form data and make predictions
def process_form(credit_history_age, monthly_balance, monthly_inhand_salary, annual_income,
                 interest_rate, outstanding_debt, num_of_loan, delay_from_due_date,
                 num_of_delayed_payment, total_emi_per_month, num_credit_inquiries,
                 name, age, profession, ssn):

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
    try:
        with open(model_path, 'rb') as file:
            model = joblib.load(file)
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return

    with st.spinner("Predicting your credit score..."):
        try:
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            return

    # Mapping credit score to labels
    credit_score_label = {0: "Poor", 1: "Average", 2: "Good"}
    predicted_score = credit_score_label[prediction[0]]
    confidence = max(prediction_proba[0])

    # Display the profile summary
    st.subheader("Profile Summary")
    st.write(f"**Name**: {name}")
    st.write(f"**Age**: {age}")
    st.write(f"**Profession**: {profession}")
    st.write(f"**SSN**: {ssn}")
    st.write(f"**Month**: {datetime.now().strftime('%B %Y')}")

    # Display the input data
    st.write("### Your Input Data:")
    st.write(f"**Credit History Age**: {credit_history_age} weeks")
    st.write(f"**Monthly Balance**: ${monthly_balance}")
    st.write(f"**Monthly Inhand Salary**: ${monthly_inhand_salary}")
    st.write(f"**Annual Income**: ${annual_income}")
    st.write(f"**Interest Rate**: {interest_rate}%")
    st.write(f"**Outstanding Debt**: ${outstanding_debt}")
    st.write(f"**Number of Loans**: {num_of_loan}")
    st.write(f"**Delay from Due Date**: {delay_from_due_date} days")
    st.write(f"**Number of Delayed Payments**: {num_of_delayed_payment}")
    st.write(f"**Total EMI per Month**: ${total_emi_per_month}")
    st.write(f"**Number of Credit Inquiries**: {num_credit_inquiries}")

    # Display the predicted credit score
    st.success(f"Predicted Credit Score: {predicted_score} (Confidence: {confidence:.2%})")

    save_to_database(name, age, profession, ssn, credit_history_age, monthly_balance, monthly_inhand_salary,
                     annual_income, interest_rate, outstanding_debt, num_of_loan, delay_from_due_date,
                     num_of_delayed_payment, total_emi_per_month, num_credit_inquiries, predicted_score)

    # Define thresholds for each input (example values – you can adjust as needed)
    thresholds = {
        'credit_history_age': [10, 30, 60],             # weeks
        'monthly_balance': [0, 1000, 5000],             # $
        'monthly_inhand_salary': [1000, 3000, 6000],    # $
        'annual_income': [20000, 50000, 100000],        # $
        'interest_rate': [25, 15, 5],                   # % (lower = better)
        'outstanding_debt': [50000, 20000, 5000],       # $
        'num_of_loan': [8, 4, 1],                       # count (lower = better)
        'delay_from_due_date': [30, 15, 0],             # days
        'num_of_delayed_payment': [10, 5, 0],           # count
        'total_emi_per_month': [3000, 1500, 500],       # $
        'num_credit_inquiries': [10, 5, 1]              # count
    }

    # Normalize all features to a 0–100 scale for consistent comparison
    def normalize(value, poor, good, reverse=False):
        # Clip the value between poor and good
        value = min(max(value, min(poor, good)), max(poor, good))
        if reverse:
            # Lower is better (e.g., interest_rate, debt)
            return 100 * (good - value) / (good - poor)
        else:
            return 100 * (value - poor) / (good - poor)

    # Set which features need reverse scaling
    reverse_scale = ['interest_rate', 'outstanding_debt', 'num_of_loan', 
                    'delay_from_due_date', 'num_of_delayed_payment', 
                    'total_emi_per_month', 'num_credit_inquiries']
    features = list(thresholds.keys())
    # Normalized data
    normalized_your_values = []
    normalized_poor = []
    normalized_average = []
    normalized_good = []

    for feature in features:
        p, a, g = thresholds[feature]
        rev = feature in reverse_scale
        normalized_your_values.append(normalize(input_data[feature].iloc[0], p, g, reverse=rev))
        normalized_poor.append(normalize(p, p, g, reverse=rev))
        normalized_average.append(normalize(a, p, g, reverse=rev))
        normalized_good.append(normalize(g, p, g, reverse=rev))

    # Create improved radar chart
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=normalized_poor,
        theta=features,
        fill='toself',
        name='Poor Threshold',
        line=dict(color='red')
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=normalized_average,
        theta=features,
        fill='toself',
        name='Average Threshold',
        line=dict(color='orange')
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=normalized_good,
        theta=features,
        fill='toself',
        name='Good Threshold',
        line=dict(color='green')
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=normalized_your_values,
        theta=features,
        fill='toself',
        name='Your Input',
        line=dict(color='dodgerblue', width=3)
    ))

    fig_radar.update_layout(
        title="Comparison of Your Input with Thresholds",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        legend=dict(orientation="h", y=-0.2),
        height=700,
        width=700,
        margin=dict(t=100, b=100)
    )

    st.plotly_chart(fig_radar)

    # Initialize lists to hold feature labels and confidence scores
    feature_labels = []  # e.g., "credit_history_age - Good"
    feature_scores = []  # similarity/confidence score

    # Loop through each feature and calculate confidence for each category
    for feature in features:
        value = input_data[feature].iloc[0]  # Get the input value for this feature
        p, a, g = thresholds[feature]  # Poor, Average, and Good threshold values
        rev = feature in reverse_scale  # Check if scaling needs to be reversed

        # Normalize the value
        norm_val = normalize(value, p, g, reverse=rev)
        norm_p = normalize(p, p, g, reverse=rev)
        norm_a = normalize(a, p, g, reverse=rev)
        norm_g = normalize(g, p, g, reverse=rev)

        # Calculate distances to the thresholds for Poor, Average, and Good
        dists = {
            'Poor': abs(norm_val - norm_p),
            'Average': abs(norm_val - norm_a),
            'Good': abs(norm_val - norm_g)
        }

        # Get closest classification and calculate confidence score
        closest = min(dists, key=dists.get)  # The class with the smallest distance
        confidence_score = 1 - (dists[closest] / (sum(dists.values()) + 1e-9))  # Normalize confidence score

        # Append the feature label and score to the respective lists
        feature_labels.append(f"{feature} - {closest}")
        feature_scores.append(confidence_score)

    # Create the donut chart with proper color assignment for each classification
    fig_feature_donut = go.Figure(data=[go.Pie(
        labels=feature_labels,
        values=feature_scores,
        hole=0.4,
        textinfo="label+percent",
        hoverinfo="label+value",
        marker=dict(colors=[
            '#ff4b4b' if 'Poor' in lbl else
            '#ffa534' if 'Average' in lbl else
            '#2ecc71' if 'Good' in lbl else '#ffffff'  # Ensure 'Good' is included
            for lbl in feature_labels  # Iterate over feature_labels
        ])
    )])

    # Update layout for clarity
    fig_feature_donut.update_layout(title_text="Per-Feature Classification Confidence")
    st.plotly_chart(fig_feature_donut)

# Function to display the input form
def display_form():
    st.title("Credit Score Prediction")
    st.subheader("Enter Your Details")

    with st.form(key='credit_score_form'):
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=0, max_value=120)
        profession = st.text_input("Profession")
        ssn = str(random.randint(100000000, 999999999))  # Generate a random SSN
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
                     num_of_delayed_payment, total_emi_per_month, num_credit_inquiries,
                     name, age, profession, ssn)

import streamlit as st

# Function to display Terms and Conditions page
def terms_and_conditions():
    st.title("Terms and Conditions")
    st.write("""
    Please read and accept the following terms and conditions before using the app.

    1. **Ethical Use**: The application is designed to predict credit scores using user-provided financial data. The model is built to avoid bias related to age, gender, or other non-financial attributes, ensuring fairness and inclusivity.
    2. **Transparency and Interpretability**: Visualizations such as radar and donut charts are used to make credit score predictions understandable and avoid black-box decision-making.
    3. **Privacy Commitment**: The application collects data only after obtaining explicit user consent. All collected data is stored securely in a PostgreSQL database, adhering to data privacy best practices and legal requirements.
    4. **Data Privacy and Legal Compliance**: The system follows the Philippine Data Privacy Act (RA 10173) by minimizing data collection, maintaining transparency, and ensuring data security.
    5. **Data Accuracy**: Predictions made by this application are for informational purposes only and may not be completely accurate. Users are encouraged to use the results as one of many considerations when making decisions.
    6. **User Responsibility**: Users must ensure the accuracy of the data they provide. Incorrect or incomplete data may result in inaccurate predictions.
    7. **No Liability**: The creators of this application assume no liability for decisions made based on the app’s predictions.

    By accepting, you agree to the terms and conditions stated above.
    """)

# Function to handle the main app content after terms are accepted
def main_app():
    st.title("Welcome to the Credit Score Prediction App!")
    st.write("You have accepted the terms and conditions.")
    
    # Integrate the main credit score prediction app here
    display_form()
    display_latest_entries()

# Run the Streamlit app
if __name__ == '__main__':
    # Terms and Conditions Page
    if 'agreed' not in st.session_state:
        st.session_state.agreed = False

    # Display terms and conditions first
    if not st.session_state.agreed:
        terms_and_conditions()
        if st.checkbox("I accept the terms and conditions"):
            st.session_state.agreed = True
    else:
        main_app()