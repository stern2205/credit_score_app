# credit_score_app
# Group 7

## Atienza, John Paul B.
## Calingasan, John Erick Z.
## Lumalang, Carl John D.
## Morales, Johann Daniel P.
## Morillo, Jerime B.

## Project Overview

The Credit Score Prediction App is a web application that predicts an individual's credit score based on various financial and personal information. The application uses a Logistic Regression model to classify the credit score into categories such as 'Poor', 'Average', or 'Good'. The prediction results are displayed through interactive visualizations using Streamlit.

## Features

* User input form to capture personal and financial details

* Predict credit score using a pre-trained Logistic Regression model

* Display profile summary and prediction results

* Interactive visualizations using Plotly (Radar and Donut charts)

* Store and retrieve data from a localhost PostgreSQL database

* Display the latest 5 credit score predictions

## Dataset
[Credit Score - Detailed & Comprehensive EDA by iremnurtokuroglu on Kaggle](https://www.kaggle.com/code/iremnurt)

## Installation and Setup

1. Clone the repository:
```
git clone https://github.com/stern2205/credit_score_app
```
2. OR Download the package.
3. Open the folder in Visual Studio Code.
4. Install dependencies:
```
pip install -r requirements.txt
```
5. Set up the PostgreSQL database:

  Make sure PostgreSQL is installed and running on the localhost machine (your server).

  Create a new database:
  ```
  CREATE DATABASE credit_score_db;
  ```

  Create the table schema
  ```
  CREATE TABLE credit_history (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    age INTEGER,
    profession VARCHAR(100),
    ssn VARCHAR(15),
    credit_history_age INTEGER,
    monthly_balance NUMERIC(12, 2),
    monthly_inhand_salary NUMERIC(12, 2),
    annual_income NUMERIC(12, 2),
    interest_rate NUMERIC(5, 2),
    outstanding_debt NUMERIC(12, 2),
    num_of_loan INTEGER,
    delay_from_due_date INTEGER,
    num_of_delayed_payment INTEGER,
    total_emi_per_month NUMERIC(12, 2),
    num_credit_inquiries INTEGER,
    predicted_score INTEGER
);
  ```

  Update database credentials in the script if needed.

## How to Run the Application

1. Start the Streamlit server:
```
streamlit run app.py
```
Open the application in your browser at:
```
http://localhost:8501
```

## Usage

1. Fill out the credit score prediction form with your details.
2. Click the Submit button to see the predicted credit score.
3. View the interactive radar and donut charts for detailed analysis.
4. Check the latest 5 predictions in the summary section.
