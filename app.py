import sys
import os

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np # Still needed for infer_aqi_from_category if we want to display a value based on category

from src.utils import get_aqi_category, CITIES # Import CITIES for dropdown
from src.preprocessing import create_lag_features # Use create_lag_features for in-memory lag creation

# --- Page Configuration ---
st.set_page_config(
    page_title="Pakistan Multi-City AQI Predictor", # Updated title
    page_icon="🌬️",
    layout="centered"
)

# --- App Title ---
st.title("🌬️ Pakistan Multi-City AQI Predictor") # Updated title
st.write("Predicting AQI categories for major cities in Pakistan based on a single dataset.")
st.markdown("---")

# --- File Paths ---
MODEL_SAVE_PATH = "models/aqi_model.joblib" # The full pipeline will be loaded from here
ENCODER_SAVE_PATH = "models/target_encoder.joblib" # Path to saved target encoder
RAW_DATA_PATH = "data/raw/AQM-dataset-updated.csv" # Data path
NUM_LAGS = 3 # Aligned with train.py and preprocessing.py
PREDICTED_HORIZON_HOURS = 24 # Aligned with train.py

# --- Helper Functions ---
@st.cache_resource
def load_components():
    """Loads the trained full pipeline and target encoder, caching them for performance."""
    try:
        full_pipeline = joblib.load(MODEL_SAVE_PATH)
        target_encoder = joblib.load(ENCODER_SAVE_PATH)
        return full_pipeline, target_encoder
    except FileNotFoundError as e:
        st.error(f"Error: Required model component not found: {e}. Please ensure you have run `python src/train.py` successfully.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        st.exception(e)
        return None, None

@st.cache_data
def load_raw_data():
    """Loads the raw data once and caches it."""
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH)
        df_raw['DateTime'] = pd.to_datetime(df_raw['Date'] + ' ' + df_raw['Time'], dayfirst=True)
        df_raw = df_raw.set_index('DateTime')
        df_raw = df_raw.drop(columns=['Date', 'Time'])
        df_raw = df_raw.sort_index()
        return df_raw
    except FileNotFoundError:
        st.error(f"Raw data file not found at {RAW_DATA_PATH}. Please check the path.")
        return pd.DataFrame()

def check_component_existence():
    """Checks if all necessary model component files exist."""
    missing_files = []
    if not os.path.exists(MODEL_SAVE_PATH):
        missing_files.append(MODEL_SAVE_PATH)
    if not os.path.exists(ENCODER_SAVE_PATH):
        missing_files.append(ENCODER_SAVE_PATH)
        
    if missing_files:
        st.error(f"Missing model components: {', '.join(missing_files)}. Please ensure you have run the training script: `python src/train.py`")
        return False
        
    if not os.path.exists(RAW_DATA_PATH):
        st.error(f"Raw data (`{RAW_DATA_PATH.split('/')[-1]}`) not found in `data/raw/` folder.")
        st.info("Please download the dataset and place it in the correct directory.")
        return False
        
    return True

def generate_single_point_forecast(selected_city, full_pipeline, target_encoder, raw_data_df):
    """
    Generates a single point forecast for the AQI category PREDICTED_HORIZON_HOURS ahead
    from the last available data point in the raw dataset.
    """
    st.info(f"Generating forecast for {selected_city} ({PREDICTED_HORIZON_HOURS} hours ahead based on latest data from {RAW_DATA_PATH.split('/')[-1]}).")

    if raw_data_df.empty:
        st.error("Cannot generate forecast as raw data could not be loaded.")
        return pd.DataFrame()

    feature_columns_for_lags = ['Temperature', 'Humidity', 'CO_AQI', 'NO2_AQI', 'SO2_AQI']

    df_raw_tail_for_lags = raw_data_df.tail(NUM_LAGS + 1 + 10).copy() # Take 10 more rows to be safe
    df_raw_tail_for_lags.dropna(subset=feature_columns_for_lags, inplace=True)

    if len(df_raw_tail_for_lags) < NUM_LAGS + 1:
        st.error("Error: Not enough CLEAN recent data to create features for forecast. "
                  "Check the raw data for recent NaNs.")
        return pd.DataFrame()

    temp_df = create_lag_features(df_raw_tail_for_lags, feature_columns_for_lags, NUM_LAGS)
    inference_data = temp_df.dropna().tail(1)

    if inference_data.empty:
        st.error("Error: Could not create valid features for forecast from recent data.")
        return pd.DataFrame()

    inference_data['hour'] = inference_data.index.hour
    inference_data['day_of_week'] = inference_data.index.dayofweek
    inference_data['day_of_month'] = inference_data.index.day
    inference_data['month'] = inference_data.index.month
    
    predicted_encoded_category = full_pipeline.predict(inference_data)[0]
    predicted_category = target_encoder.inverse_transform([[predicted_encoded_category]])[0][0]

    last_known_timestamp = inference_data.index[-1]
    forecast_timestamp = last_known_timestamp + timedelta(hours=PREDICTED_HORIZON_HOURS)

    forecast_result_df = pd.DataFrame([{
        'City': selected_city, # Include selected city in output
        'Forecast Timestamp': forecast_timestamp,
        'Predicted AQI Category': predicted_category
    }])
    
    return forecast_result_df

# --- Main Application ---
if check_component_existence():
    full_pipeline, target_encoder = load_components()
    df_raw_cached = load_raw_data() # Load raw data once and cache it

    # --- Sidebar for User Input ---
    st.sidebar.header("Forecast Options")
    selected_city = st.sidebar.selectbox(
        "Select a City:",
        CITIES,
        index=0 # Default to the first city in CITIES
    )
    st.sidebar.markdown("---")

    # --- City Statistics Section (from the single dataset) ---
    st.sidebar.subheader(f"Statistics for {selected_city} (from {RAW_DATA_PATH.split('/')[-1]})")
    if not df_raw_cached.empty:
        # Calculate global averages from the single dataset
        avg_temp = df_raw_cached['Temperature'].mean()
        avg_humidity = df_raw_cached['Humidity'].mean()
        st.sidebar.metric("Average Temperature (°C)", f"{avg_temp:.2f}")
        st.sidebar.metric("Average Humidity (%)", f"{avg_humidity:.2f}")
        st.sidebar.caption("Note: These statistics are derived from the single provided dataset and may not be truly representative of each specific city.")
    else:
        st.sidebar.info("Data not loaded, cannot display statistics.")
    st.sidebar.markdown("---")


    st.subheader("Generate Forecast")
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner(f"Generating AQI category forecast for {selected_city} ({PREDICTED_HORIZON_HOURS} hours ahead)..."):
            try:
                # Pass the cached raw data to the forecast function
                forecast_df = generate_single_point_forecast(selected_city, full_pipeline, target_encoder, df_raw_cached)
                if not forecast_df.empty:
                    st.session_state['forecast_df'] = forecast_df
                    st.session_state['selected_city_for_display'] = selected_city # Store selected city
            except Exception as e:
                st.error(f"An error occurred during forecast generation: {e}")
                st.exception(e) # Display full exception for debugging

    # --- Display Forecast ---
    if 'forecast_df' in st.session_state and not st.session_state['forecast_df'].empty:
        st.subheader(f"Forecast Result for {st.session_state['selected_city_for_display']}")
        
        display_df = st.session_state['forecast_df'].copy()
        
        st.dataframe(display_df[['Forecast Timestamp', 'Predicted AQI Category']].set_index('Forecast Timestamp')) # Only show relevant columns

        category = display_df['Predicted AQI Category'].iloc[0]
        if category in ["Unhealthy", "Hazardous"]:
            st.error("🚨 RED ALERT: Unhealthy or Hazardous Air Quality Expected!")
            st.warning(f"For {st.session_state['selected_city_for_display']}: Take precautions: Reduce prolonged or heavy exertion. Avoid outdoor activities where air quality is poor.")
        elif category == "Good":
            st.success("✅ Good Air Quality Expected.")
        else:
            st.info(f"ℹ️ {category} Air Quality Expected.")

        st.markdown("---")

        # --- Download Button ---
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_data,
            file_name=f"{st.session_state['selected_city_for_display']}_AQI_forecast_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv",
            mime='text/csv',
        )
    else:
        st.info("Click 'Generate Forecast' to see the prediction for the next 24 hours.")

else:
    st.warning("Application is not fully set up. Please address the issues listed above.")

# --- Footer ---
st.markdown("---")
st.caption("Developed for the SMIT Hackathon.")
st.caption("Note: This forecast uses an XGBoost Classifier directly predicting AQI categories based on historical data from the single provided dataset. The model predicts a categorical AQI value for 24 hours ahead, not an iterative multi-day forecast. Predictions are currently city-agnostic due to data limitations.")