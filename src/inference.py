import sys
import os

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
import os
import argparse
from datetime import datetime, timedelta

# Removed build_preprocess_pipeline as it doesn't exist.
# from src.preprocessing import build_preprocess_pipeline
# Removed CITIES as the model is now city-agnostic for this dataset.
# from src.utils import CITIES
from src.preprocessing import create_lag_features, load_and_preprocess_data
from src.utils import get_aqi_category
# --- Configuration ---
RAW_DATA_PATH = "data/raw/AQM-dataset-updated.csv"
MODEL_SAVE_PATH = "models/aqi_model.joblib" # Aligned with train.py
ENCODER_SAVE_PATH = "models/target_encoder.joblib" # For decoding predictions
NUM_LAGS = 3 # Aligned with train.py and preprocessing.py
PREDICTED_HORIZON_HOURS = 24 # Aligned with train.py

def predict_aqi_category():
    """
    Makes a single prediction for the next 24-hour AQI category using the trained model.
    """
    print(f"--- Making AQI Category Prediction ---")

    # --- 1. Load Model & Encoder ---
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model not found at {MODEL_SAVE_PATH}. Please run train.py first.")
        return
    if not os.path.exists(ENCODER_SAVE_PATH):
        print(f"Target Encoder not found at {ENCODER_SAVE_PATH}. Please run train.py first.")
        return

    model_pipeline = joblib.load(MODEL_SAVE_PATH)
    target_encoder = joblib.load(ENCODER_SAVE_PATH)

    # --- 2. Prepare Data for Inference ---
    # Load the raw data to get the most recent entries for lag feature creation.
    # We will simulate the preprocessing steps on the recent data.
    print(f"Loading raw data from {RAW_DATA_PATH} for inference data preparation...")
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
        return

    # Combine 'Date' and 'Time' into a single datetime column and set as index
    df_raw['DateTime'] = pd.to_datetime(df_raw['Date'] + ' ' + df_raw['Time'], dayfirst=True)
    df_raw = df_raw.set_index('DateTime')
    df_raw = df_raw.drop(columns=['Date', 'Time'])

    # Ensure the dataframe is sorted by index (DateTime)
    df_raw = df_raw.sort_index()

    # Define columns for which to create lag features (same as in preprocessing.py)
    feature_columns_for_lags = ['Temperature', 'Humidity', 'CO_AQI', 'NO2_AQI', 'SO2_AQI']

    # Get enough recent data points to create the required lags
    # We need at least NUM_LAGS + 1 data points to get one complete row with lags
    required_rows = NUM_LAGS + PREDICTED_HORIZON_HOURS # Take enough rows to account for prediction horizon shift
    
    # Let's consider enough data points to form one input sample for prediction.
    # The model predicts the target for PREDICTED_HORIZON_HOURS ahead.
    # So we need data up to 'now' to predict for 'now + PREDICTED_HORIZON_HOURS'.
    # We need the last data point available, plus the `NUM_LAGS` preceding it.
    
    # We need the most recent row, with its lag features.
    # So we take the last NUM_LAGS + 1 rows from the raw data for feature creation.
    
    # Instead of predicting "tomorrow", we will predict for the time that the model
    # was trained to predict. The `train.py` calculates target as `shift(-PREDICTED_HORIZON_HOURS)`.
    # This means the model learns to predict the AQI_Category at `t + PREDICTED_HORIZON_HOURS`
    # given features at time `t`.
    
    # So, for inference, we need to create features for the latest available timestamp (t_latest)
    # to predict for (t_latest + PREDICTED_HORIZON_HOURS).

    # To create lag features for the latest timestamp, we need the latest (NUM_LAGS + 1) rows.
    # Example: if NUM_LAGS=3, for t, we need t-1, t-2, t-3. So for t_latest, we need t_latest, t_latest-1, t_latest-2, t_latest-3.
    # To create lag features for the latest timestamp, we need enough recent rows without NaNs.
    # Take a slightly larger tail to account for potential NaNs in the latest raw data entries.
    # We need at least NUM_LAGS + 1 clean rows to get one final inference_data row.
    df_raw_tail_for_lags = df_raw.tail(NUM_LAGS + 1 + 10).copy() # Take 10 more rows to be safe

    # Drop rows with NaNs in the columns used for lag features before creating lags
    df_raw_tail_for_lags.dropna(subset=feature_columns_for_lags, inplace=True)

    if len(df_raw_tail_for_lags) < NUM_LAGS + 1:
        print("Error: Not enough CLEAN recent data to create features for inference. "
              "Check the raw data for recent NaNs.")
        return

    # Apply lag feature creation on this clean, recent slice
    temp_df = create_lag_features(df_raw_tail_for_lags, feature_columns_for_lags, NUM_LAGS)
    inference_data = temp_df.dropna().tail(1) # Get the most recent row with valid lags

    if inference_data.empty:
        print("Error: Not enough recent data to create features for inference.")
        return

    # Add time-based features from the DateTime index for the inference data
    inference_data['hour'] = inference_data.index.hour
    inference_data['day_of_week'] = inference_data.index.dayofweek
    inference_data['day_of_month'] = inference_data.index.day
    inference_data['month'] = inference_data.index.month

    # Drop original AQI columns that are not part of the model's expected input features
    # (they are used to create lags, but the lags are the features, not the original AQIs)
    # cols_to_drop_from_inference = feature_columns_for_lags + ['Unnamed: 0'] # 'Unnamed: 0' from CSV
    # inference_data = inference_data.drop(columns=[col for col in cols_to_drop_from_inference if col in inference_data.columns])
    
    # Ensure that the 'CO_AQI' column is dropped if it's the target variable, 
    # but only if it's not explicitly used as a feature itself.
    # For now, let's assume the original AQI columns are indeed features, along with their lags.
    # The ColumnTransformer in train.py handles feature selection.
    
    # --- 3. Predict ---
    # The full pipeline handles all preprocessing internally (scaling, one-hot encoding etc.)
    predicted_encoded_category = model_pipeline.predict(inference_data)[0]
    predicted_category = target_encoder.inverse_transform([[predicted_encoded_category]])[0][0]

    prediction_time = inference_data.index[-1] + timedelta(hours=PREDICTED_HORIZON_HOURS)
    print(f"Predicted AQI Category for {prediction_time}: {predicted_category}")


if __name__ == "__main__":
    # Removed argparse as city argument is no longer needed
    # parser = argparse.ArgumentParser(description="Get a quick AQI forecast for tomorrow.")
    # parser.add_argument("city", type=str, choices=CITIES, help="The city to forecast.")
    # args = parser.parse_args()
    predict_aqi_category()
