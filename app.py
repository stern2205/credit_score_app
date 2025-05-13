import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np
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
            return    model_path == 'content/logistic_regression_model.pkl'
    try:
        with open(model_path, 'rb') as file:
            model = joblib.load(file)
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return

    # Mapping credit score to labels
    credit_score_label = {0: "Poor", 1: "Average", 2: "Good"}
    predicted_score = credit_score_label[prediction[0]]
    confidence = max(prediction_proba[0])

    # Display result
    st.success(f"Predicted Credit Score: {predicted_score} (Confidence: {confidence:.2%})")
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

    # 📈 Bar Chart for Input Feature Overview
    st.subheader("Your Input Summary")
    fig_features = go.Figure([go.Bar(
        x=input_data.columns,
        y=input_data.iloc[0],
        marker_color='lightskyblue'
    )])
    fig_features.update_layout(xaxis_title="Feature", yaxis_title="Value")
    st.plotly_chart(fig_features)
    
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