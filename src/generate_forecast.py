import sys
import os

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
import argparse
from datetime import datetime, timedelta

from src.preprocessing import create_lag_features
from src.utils import get_aqi_category

# --- Configuration ---
RAW_DATA_PATH = "data/raw/AQM-dataset-updated.csv"
MODEL_SAVE_PATH = "models/aqi_model.joblib"
ENCODER_SAVE_PATH = "models/target_encoder.joblib"
RESULTS_DIR = "results"
NUM_LAGS = 3  # Aligned with train.py and preprocessing.py
PREDICTED_HORIZON_HOURS = 24  # Aligned with train.py

def generate_forecast(start_date_str=None):
    """
    Generates a forecast for the AQI category PREDICTED_HORIZON_HOURS ahead
    from the last available data point in the raw dataset.
    The result is saved to a CSV file.
    """
    print(f"--- Generating AQI Category Forecast ---")

    # --- 1. Load Model & Encoder ---
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model not found at {MODEL_SAVE_PATH}. Please run train.py first.")
        return
    if not os.path.exists(ENCODER_SAVE_PATH):
        print(f"Target Encoder not found at {ENCODER_SAVE_PATH}. Please run train.py first.")
        return

    model_pipeline = joblib.load(MODEL_SAVE_PATH)
    target_encoder = joblib.load(ENCODER_SAVE_PATH)

    # --- 2. Prepare Data for Forecast ---
    print(f"Loading raw data from {RAW_DATA_PATH} for forecast data preparation...")
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
        return

    df_raw['DateTime'] = pd.to_datetime(df_raw['Date'] + ' ' + df_raw['Time'], dayfirst=True)
    df_raw = df_raw.set_index('DateTime')
    df_raw = df_raw.drop(columns=['Date', 'Time'])
    df_raw = df_raw.sort_index()

    feature_columns_for_lags = ['Temperature', 'Humidity', 'CO_AQI', 'NO2_AQI', 'SO2_AQI']

    # Get enough recent data points to create the required lags robustly
    df_raw_tail_for_lags = df_raw.tail(NUM_LAGS + 1 + 10).copy() # Take 10 more rows to be safe
    df_raw_tail_for_lags.dropna(subset=feature_columns_for_lags, inplace=True)

    if len(df_raw_tail_for_lags) < NUM_LAGS + 1:
        print("Error: Not enough CLEAN recent data to create features for forecast. "
              "Check the raw data for recent NaNs.")
        return

    temp_df = create_lag_features(df_raw_tail_for_lags, feature_columns_for_lags, NUM_LAGS)
    inference_data = temp_df.dropna().tail(1)

    if inference_data.empty:
        print("Error: Could not create valid features for forecast from recent data.")
        return

    inference_data['hour'] = inference_data.index.hour
    inference_data['day_of_week'] = inference_data.index.dayofweek
    inference_data['day_of_month'] = inference_data.index.day
    inference_data['month'] = inference_data.index.month
    
    # --- 3. Predict AQI Category ---
    predicted_encoded_category = model_pipeline.predict(inference_data)[0]
    predicted_category = target_encoder.inverse_transform([[predicted_encoded_category]])[0][0]

    # --- 4. Format and Save Results ---
    last_known_timestamp = inference_data.index[-1]
    forecast_timestamp = last_known_timestamp + timedelta(hours=PREDICTED_HORIZON_HOURS)

    forecast_result = pd.DataFrame([{
        'Forecast_Timestamp': forecast_timestamp,
        'Predicted_AQI_Category': predicted_category
    }])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = f"aqi_category_forecast_{forecast_timestamp.strftime('%Y%m%d_%H%M')}.csv"
    save_path = os.path.join(RESULTS_DIR, filename)
    
    forecast_result.to_csv(save_path, index=False)
    print(f"Forecast saved successfully to {save_path}")
    print(f"Predicted AQI Category for {forecast_timestamp}: {predicted_category}")
    return forecast_result


if __name__ == "__main__":
    # Removed argparse as city, start_date, and horizon are no longer needed
    # The forecast is for a single future point determined by the model's horizon
    # and the last available data in the dataset.
    generate_forecast()