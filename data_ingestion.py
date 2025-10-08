import pandas as pd
import json
import os

print("Starting Step 1: Data Ingestion...")

# Define file paths
# In a real scenario, this would be your original Kaggle dataset file.
# We will create it if it doesn't exist for demonstration.
original_data_path = 'customer_churn_dataset-testing-master.csv'
raw_sql_output_path = 'raw_subscription_data.csv'
raw_json_output_path = 'raw_user_activity_logs.json'

# --- Simulation Setup ---
# For this self-contained project, let's create the source CSV if it's missing.
if not os.path.exists(original_data_path):
    print(f"'{original_data_path}' not found. Creating a dummy file for demonstration.")
    dummy_data = {
        'CustomerID': [1, 2, 3, 4, 5], 'Age': [22, 45, 31, 28, 55],
        'Gender': ['Female', 'Male', 'Male', 'Female', 'Male'],
        'Tenure': [25, 12, 5, 50, 3], 'Usage Frequency': [14, 20, 5, 25, 2],
        'Support Calls': [4, 1, 8, 0, 10], 'Payment Delay': [27, 2, 15, 0, 5],
        'Subscription Type': ['Basic', 'Premium', 'Standard', 'Premium', 'Basic'],
        'Contract Length': ['Monthly', 'Annual', 'Quarterly', 'Annual', 'Monthly'],
        'Total Spend': [598, 1200, 850, 1500, 250],
        'Last Interaction': [9, 30, 15, 2, 25], 'Churn': [1, 0, 1, 0, 1]
    }
    pd.DataFrame(dummy_data).to_csv(original_data_path, index=False)

# --- ETL: Extract Phase ---

# 1. Extract Subscription Data from "PostgreSQL/MySQL"
# In a real pipeline, you would use libraries like psycopg2 or SQLAlchemy to connect to a database.
# Here, we simulate this by reading from the source CSV file.
try:
    print(f"Reading subscription data from '{original_data_path}'...")
    subscription_df = pd.read_csv(original_data_path)

    # These are the columns typically found in a subscription database
    subscription_columns = [
        'CustomerID', 'Age', 'Gender', 'Subscription Type',
        'Contract Length', 'Total Spend', 'Churn'
    ]
    subscription_data = subscription_df[subscription_columns]

    # Save the extracted raw data
    subscription_data.to_csv(raw_sql_output_path, index=False)
    print(f"Successfully extracted and saved subscription data to '{raw_sql_output_path}'.")

except FileNotFoundError:
    print(f"Error: The source file '{original_data_path}' was not found.")
    exit() # Exit if the source data is not available.

# 2. Extract User Activity Data from "JSON Logs"
# In a real pipeline, you might read from Kafka, S3, or log files.
# We simulate this by taking relevant columns from the same source CSV and converting them to JSON logs.
print("Simulating extraction of user activity from JSON logs...")
activity_columns = [
    'CustomerID', 'Tenure', 'Usage Frequency',
    'Support Calls', 'Payment Delay', 'Last Interaction'
]
activity_data = subscription_df[activity_columns]

# Convert the DataFrame to a list of JSON records (logs)
activity_logs = activity_data.to_dict(orient='records')

# Save the logs to a JSON file
with open(raw_json_output_path, 'w') as f:
    json.dump(activity_logs, f, indent=4)

print(f"Successfully simulated and saved user activity logs to '{raw_json_output_path}'.")
print("\nData Ingestion step completed.")
