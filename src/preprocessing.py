import pandas as pd
import numpy as np
import os

def create_lag_features(df, columns, num_lags=1):
    """
    Creates lag features for specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names for which to create lag features.
        num_lags (int): The number of lag features to create.

    Returns:
        pd.DataFrame: The DataFrame with added lag features.
    """
    for col in columns:
        for i in range(1, num_lags + 1):
            df[f'{col}_lag{i}'] = df[col].shift(i)
    return df

def load_and_preprocess_data(raw_data_path, processed_data_path, num_lags=1):
    """
    Loads raw AQM data, preprocesses it, creates lag features, and saves the result.

    Args:
        raw_data_path (str): Path to the raw CSV data file.
        processed_data_path (str): Path to save the processed CSV data file.
        num_lags (int): The number of lag features to create.
    """
    print(f"Loading data from: {raw_data_path}")
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}")
        return

    # Combine 'Date' and 'Time' into a single datetime column
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df = df.set_index('DateTime')
    df = df.drop(columns=['Date', 'Time'])

    # Define columns for which to create lag features
    feature_columns = ['Temperature', 'Humidity', 'CO_AQI', 'NO2_AQI', 'SO2_AQI']

    # Create lag features
    print(f"Creating {num_lags} lag features for: {feature_columns}")
    df_processed = create_lag_features(df.copy(), feature_columns, num_lags)

    # Drop rows with NaN values resulting from lag feature creation
    initial_rows = len(df_processed)
    df_processed.dropna(inplace=True)
    rows_dropped = initial_rows - len(df_processed)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to NaN values from lag feature creation.")

    # Ensure the processed directory exists
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    # Save the processed data
    print(f"Saving processed data to: {processed_data_path}")
    df_processed.to_csv(processed_data_path)
    print("Preprocessing complete.")
    print(f"Processed data shape: {df_processed.shape}")
    print(df_processed.head())
    return df_processed



if __name__ == "__main__":
    RAW_DATA_PATH = 'data/raw/AQM-dataset-updated.csv'
    PROCESSED_DATA_PATH = 'data/processed/preprocessed_aqm_data.csv'
    NUM_LAGS = 3  # Example: create 3 lag features

    load_and_preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH, NUM_LAGS)