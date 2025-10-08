import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd
import os
from dotenv import load_dotenv

# --- Load Environment Variables ---
# This loads the credentials from the .env file.
load_dotenv()
print("Loading environment variables for Snowflake connection...")

# --- Snowflake Configuration ---
SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_ROLE = os.getenv('SNOWFLAKE_ROLE')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')
SNOWFLAKE_TABLE = os.getenv('SNOWFLAKE_TABLE')
CSV_FILE_PATH = os.getenv('CSV_FILE_PATH', 'preprocessed_churn_data.csv') # Default path

# --- Data Loading ---
print(f"Reading preprocessed data from '{CSV_FILE_PATH}'...")
try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
    print("Please ensure Step 2 has been run and the file exists, or check the path in your .env file.")
    exit()

# Snowflake stores unquoted identifiers in uppercase.
# It's best practice to convert column names to uppercase to avoid potential issues.
df.columns = [col.upper() for col in df.columns]

# --- Establish Snowflake Connection and Load Data ---
print("Connecting to Snowflake...")
try:
    with snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
        role=SNOWFLAKE_ROLE
    ) as conn:
        print("Successfully connected to Snowflake.")
        
        # Using write_pandas to upload the dataframe
        print(f"Uploading data to table: {SNOWFLAKE_TABLE}...")
        success, nchunks, nrows, _ = write_pandas(
            conn=conn,
            df=df,
            table_name=SNOWFLAKE_TABLE,
            auto_create_table=False, # Table is created by snowflake_schema.sql
            overwrite=True # Overwrite the table if it already has data
        )
        print(f"Upload complete. Success: {success}")
        print(f"Loaded {nrows} rows in {nchunks} chunks.")

except Exception as e:
    print(f"An error occurred: {e}")

print("\nSnowflake data loading step completed.")
