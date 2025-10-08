# ChurnGuard-Predictive-Pipeline

## Customer Churn Analysis for a Subscription Service

This project builds a complete data pipeline to identify, predict, and analyze customer churn for a subscription-based service. The pipeline starts from raw data extraction and ends with a deployed machine learning model for real-time predictions.

## Project Goal

The primary objective is to build a robust system that can:

- **ETL Pipeline:** Extract data from various sources (databases and logs), transform it into a usable format, and load it into a data warehouse.
- **Data Warehousing:** Structure the transformed data within Snowflake for efficient analytics.
- **Predictive Modeling:** Train, evaluate, and save a machine learning model to accurately predict whether a customer will churn.
- **Real-time Analytics:** Use Redis for caching to support real-time dashboards and analytics.
- **Deployment:** Deploy the trained model as an API for on-demand predictions.

## Data Sources

- **Subscription Data:** Stored in a PostgreSQL/MySQL database (simulated with a CSV file).
- **User Activity:** JSON logs capturing user interactions.

## Technology Stack

- **ETL & ML:** Python (`pandas`, `scikit-learn`, `joblib`)
- **Data Warehouse:** Snowflake
- **NoSQL Cache:** Redis
- **API Deployment:** Flask
- **SQL:** Snowflake SQL for querying and analysis

## Environment Setup (for Snowflake)

Before running the pipeline, you need to set up your Snowflake credentials:

1. Create a `.env` file in the root directory of this project.
2. Fill in your Snowflake account details and the path to the preprocessed data file (`preprocessed_churn_data.csv`). You can use `.env.example` as a template.

## Step-by-Step Execution Guide

### Step 1: Data Ingestion
This script reads the raw dataset files and outputs them into standardized formats.

**Run the script:**
```bash
python data_ingestion.py
```
**Output:** `raw_subscription_data.csv`, `raw_user_activity_logs.json`

### Step 2: Data Preprocessing and Model Training
This script cleans the raw data, performs feature engineering, trains the machine learning model, and saves the model and encoders.

**Run the script:**
```bash
python preprocess_and_train.py
```
**Output:** `preprocessed_churn_data.csv`, `best_churn_model.joblib`, `label_encoder.joblib`, `one_hot_encoder.joblib`

### Step 3: Batch Prediction on New Data
This script uses the trained model to make predictions on a new batch of customer data.

**Run the script:**
```bash
python batch_prediction.py
```
**Output:** `batch_predictions.csv`


### Step 4: API Deployment for Real-Time Prediction
This script launches a local web server with a user interface to make real-time predictions.

**Run the script:**
```bash
python flask_app_deploy.py
```
**Action:** Starts a local web server. Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to use the application.

### Step 5: Data Warehousing and SQL Analysis with Snowflake
This step involves setting up the Snowflake environment, loading the data, and running analytical queries.

1. **Sign up for a Snowflake Account:** If you don't have one, sign up for a Snowflake free trial. During setup, you will choose a Snowflake Account URL (e.g., `yourorg-youraccount.snowflakecomputing.com`). This is your `SNOWFLAKE_ACCOUNT` value.
2. **Create the Database and Table:** Open a Snowflake worksheet and run the schema creation commands found in the `sql_queries` file. This will create the necessary database, schema, and table.
3. **Load Data into Snowflake:** Run the Python loader script. This script reads the `preprocessed_churn_data.csv` file and uploads its contents to your Snowflake table.

**Run the script:**
```bash
python snow_flake_loader.py
```

4. **Run Analytical Queries:** Go back to your Snowflake worksheet and use the analytical queries from the `sql_queries` file to calculate key business metrics like overall churn rate, churn by subscription type, and identify high-value customers at risk.
