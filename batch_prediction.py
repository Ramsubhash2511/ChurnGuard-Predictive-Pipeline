import pandas as pd
import joblib

print("Starting Step 3: Batch Prediction...")

# --- Configuration ---
MODEL_PATH = 'best_churn_model.joblib'
LE_GENDER_PATH = 'label_encoder.joblib'
OHE_PATH = 'one_hot_encoder.joblib'
NEW_DATA_PATH = 'new_customers.csv'
PREDICTIONS_OUTPUT_PATH = 'batch_predictions.csv'

# --- Simulation: Create New Unseen Data ---
# In a real scenario, this file would come from a different source.
new_customer_data = {
    'CustomerID': [70001, 70002, 70003, 70004, 70005],
    'Age': [34, 22, 56, 41, 29],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Female'],
    'Subscription Type': ['Premium', 'Basic', 'Standard', 'Premium', 'Basic'],
    'Contract Length': ['Annual', 'Monthly', 'Quarterly', 'Annual', 'Monthly'],
    'Total Spend': [1500, 600, 950, 1800, 750],
    'Tenure': [48, 2, 15, 36, 24],
    'Usage Frequency': [25, 10, 18, 28, 20],
    'Support Calls': [0, 5, 2, 1, 3],
    'Payment Delay': [1, 25, 5, 0, 10],
    'Last Interaction': [5, 12, 8, 3, 20]
}
new_customers_df = pd.DataFrame(new_customer_data)
new_customers_df.to_csv(NEW_DATA_PATH, index=False)
print(f"Generated sample new customer data at '{NEW_DATA_PATH}'.")

# --- Load Model and Encoders ---
print("Loading the saved model and encoders...")
try:
    model = joblib.load(MODEL_PATH)
    le_gender = joblib.load(LE_GENDER_PATH)
    ohe = joblib.load(OHE_PATH)
except FileNotFoundError:
    print("Error: Model or encoder files not found. Please run '2_data_preprocessing_and_training.py' first.")
    exit()
    
# --- Prediction Pipeline for New Data ---
def preprocess_new_data(df, le, ohe):
    """
    Preprocesses new data using the fitted encoders.
    """
    df_processed = df.copy()
    
    # Standardize column names
    df_processed.columns = df_processed.columns.str.replace(' ', '_').str.lower()
    
    # Apply Label Encoder
    df_processed['gender'] = le.transform(df_processed['gender'])
    
    # Apply One-Hot Encoder
    cols_to_encode = ['subscription_type', 'contract_length']
    encoded_array = ohe.transform(df_processed[cols_to_encode])
    encoded_cols = ohe.get_feature_names_out(cols_to_encode)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_processed.index)
    
    # Combine and align columns with the training data
    df_processed = df_processed.drop(columns=cols_to_encode)
    df_processed = pd.concat([df_processed, encoded_df], axis=1)
    
    # Ensure column order matches the model's training columns
    # We load the trained model to get the expected feature names
    training_columns = model.feature_names_in_
    
    # Drop CustomerID as it's not a feature
    df_processed = df_processed.drop(columns=['customerid'], errors='ignore')

    # Reorder columns to match the training set
    return df_processed[training_columns]

# --- Execute Prediction ---
print("Applying preprocessing to new data...")
new_customers_raw = pd.read_csv(NEW_DATA_PATH)
X_new = preprocess_new_data(new_customers_raw, le_gender, ohe)

print("Making predictions...")
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

# Add predictions back to the original new data DataFrame
new_customers_raw['Churn_Prediction'] = predictions
new_customers_raw['Churn_Probability'] = probabilities

# Save the results
new_customers_raw.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)

print(f"\nBatch predictions complete. Results saved to '{PREDICTIONS_OUTPUT_PATH}'.")
print("\n--- Prediction Results ---")
print(new_customers_raw)
print("\nBatch Prediction step completed.")
