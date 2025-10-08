from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder='templates')

# --- Load Model and Encoders at Startup ---
try:
    model = joblib.load('best_churn_model.joblib')
    le_gender = joblib.load('label_encoder.joblib')
    ohe = joblib.load('one_hot_encoder.joblib')
    print("Model and encoders loaded successfully.")
except FileNotFoundError:
    print("Error: Model or encoder files not found. Ensure they are in the same directory.")
    model = None # Set to None to handle errors gracefully

# --- Preprocessing Function ---
def preprocess_api_data(data_dict, le, ohe):
    """
    Preprocesses a single data dictionary from an API request.
    """
    # Create a DataFrame from the input dictionary
    df = pd.DataFrame([data_dict])
    
    # Standardize column names to lowercase and replace spaces
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Apply Label Encoder for Gender
    # We use a try-except block to handle raw string values ('Male'/'Female')
    try:
        df['gender'] = le.transform(df['gender'])
    except ValueError:
        # If the value is already encoded (0/1), just convert it to an integer
        df['gender'] = df['gender'].astype(int)

    # Apply One-Hot Encoder for categorical columns
    cols_to_encode = ['subscription_type', 'contract_length']
    encoded_array = ohe.transform(df[cols_to_encode])
    encoded_cols = ohe.get_feature_names_out(cols_to_encode)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
    
    df = df.drop(columns=cols_to_encode)
    df = pd.concat([df, encoded_df], axis=1)
    
    # Align columns with the model's training data
    training_columns = model.feature_names_in_
    # Create missing columns with a value of 0
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
            
    # Ensure the column order is exactly the same as during training
    return df[training_columns]

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the HTML web form for user input."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles both JSON API requests and form submissions."""
    if not model:
        return jsonify({'error': 'Model is not loaded. Check server logs.'}), 500
        
    try:
        data = request.get_json()
        
        # Convert numeric fields from string to number
        numeric_fields = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'total_spend', 'last_interaction']
        for field in numeric_fields:
            if field in data:
                data[field] = float(data[field])

        # Preprocess the data
        processed_data = preprocess_api_data(data, le_gender, ohe)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0, 1]
        
        # Create response
        response = {
            'churn_prediction': int(prediction),
            'churn_probability': float(probability),
            'interpretation': 'Likely to Churn' if prediction == 1 else 'Unlikely to Churn'
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Ensure the 'templates' directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created 'templates' directory. Please place 'index.html' inside it.")

    print("Starting Flask server...")
    print("Open your browser and go to http://127.0.0.1:5000")
    app.run(debug=True, port=5000)