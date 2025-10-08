import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

# --- Configuration ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
SUBSCRIPTION_DATA_PATH = 'raw_subscription_data.csv'
ACTIVITY_LOGS_PATH = 'raw_user_activity_logs.json'
PREPROCESSED_OUTPUT_PATH = 'preprocessed_churn_data.csv'
MODEL_OUTPUT_PATH = 'best_churn_model.joblib'
LE_GENDER_PATH = 'label_encoder.joblib'
OHE_PATH = 'one_hot_encoder.joblib'

print("Starting Step 2: Data Preprocessing and Model Training...")

# --- ETL: Transform and Load Phase ---

# 1. Load Data
print("Loading raw data...")
try:
    df_subs = pd.read_csv(SUBSCRIPTION_DATA_PATH)
    df_activity = pd.read_json(ACTIVITY_LOGS_PATH)
except FileNotFoundError as e:
    print(f"Error: Raw data file not found. Please run '1_data_ingestion.py' first.")
    print(e)
    exit()

# 2. Join Data (Merge)
print("Joining subscription and activity data...")
df = pd.merge(df_subs, df_activity, on='CustomerID')

# 3. Data Cleaning and Preprocessing
print("Cleaning and preprocessing data...")
# Standardize column names
df.columns = df.columns.str.replace(' ', '_').str.lower()

# --- Feature Engineering & Encoding ---

# a. Encode Gender using LabelEncoder
print("Encoding 'gender' column...")
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
# Save the encoder for later use in prediction
joblib.dump(le, LE_GENDER_PATH)
print(f"Label encoder saved to '{LE_GENDER_PATH}'.")


# b. Encode Subscription Type and Contract Length using OneHotEncoder
print("Encoding 'subscription_type' and 'contract_length' columns...")
cols_to_encode = ['subscription_type', 'contract_length']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform the data
encoded_array = ohe.fit_transform(df[cols_to_encode])
encoded_cols = ohe.get_feature_names_out(cols_to_encode)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

# Drop original columns and concatenate encoded ones
df = df.drop(columns=cols_to_encode)
df = pd.concat([df, encoded_df], axis=1)

# Save the one-hot encoder
joblib.dump(ohe, OHE_PATH)
print(f"One-hot encoder saved to '{OHE_PATH}'.")

# Save the fully preprocessed data (useful for loading into a Data Warehouse)
df.to_csv(PREPROCESSED_OUTPUT_PATH, index=False)
print(f"Fully preprocessed data saved to '{PREPROCESSED_OUTPUT_PATH}'.")


# --- Model Training ---

# 4. Split Data
print("Splitting data into training and testing sets...")
X = df.drop(columns=['churn', 'customerid'])
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 5. Define and Compare Models
print("Training and comparing models using cross-validation...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

for name, model in models.items():
    print(f"  - Training {name}...")
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = np.mean(scores)
    print(f"    {name} Mean ROC AUC: {np.mean(scores):.4f}")

# 6. Select and Train the Best Model
best_model_name = max(cv_results, key=cv_results.get)
print(f"\nBest model based on ROC AUC: {best_model_name}")

best_model = models[best_model_name]
print("Training the best model on the full training data...")
best_model.fit(X_train, y_train)

# --- Model Evaluation ---
print("\nEvaluating the best model on the test set...")
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Test Set ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# Plotting Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Churn', 'Churn'],
            yticklabels=['Not Churn', 'Churn'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix for {best_model_name}")
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'.")

# --- Save the Model ---
print(f"Saving the trained model to '{MODEL_OUTPUT_PATH}'...")
joblib.dump(best_model, MODEL_OUTPUT_PATH)

print("\nPreprocessing and Training step completed.")
