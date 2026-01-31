import sys
import os

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

from src.preprocessing import load_and_preprocess_data # Import the new preprocessing function
from src.utils import CITIES, get_aqi_category # Updated function name

# --- Configuration ---
RAW_DATA_PATH = "data/raw/AQM-dataset-updated.csv"
PROCESSED_DATA_PATH = "data/processed/preprocessed_aqm_data.csv"
MODEL_SAVE_PATH = "models/aqi_model.joblib" # The full pipeline will be saved here
ENCODER_SAVE_PATH = "models/target_encoder.joblib" # Save encoder separately
TARGET_VARIABLE_AQI = "CO_AQI" # Changed target variable to CO_AQI
TARGET_VARIABLE_CATEGORY = "AQI_Category"
PREDICTED_HORIZON_HOURS = 24  # Predict 1 day ahead (AQI Category for next day)
NUM_LAGS = 3 # Number of lag features to create for preprocessing

def train_model():
    """
    Loads data, preprocesses, trains a full pipeline (Preprocessor + XGBClassifier) for AQI_Category,
    and saves the pipeline and target encoder.
    """
    print("--- Model Training Started ---")

    # --- 1. Preprocess Data ---
    print(f"Loading and preprocessing data from {RAW_DATA_PATH}...")
    df = load_and_preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH, NUM_LAGS)

    if df.empty:
        print("ERROR: Preprocessing returned an empty DataFrame. Please check the data file and preprocessing script.")
        # Create dummy files to prevent app from crashing
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        joblib.dump("dummy", MODEL_SAVE_PATH)
        joblib.dump("dummy", ENCODER_SAVE_PATH)
        return

    print(f"Loaded and preprocessed {len(df)} rows.")

    # --- 2. Create Target Variable ---
    # Create the target variable: AQI Category for 24 hours ahead
    # NOTE: The original code likely had a 'City' column which is not present in the preprocessed data.
    # Assuming the target variable is created based on the AQI values directly.
    df[TARGET_VARIABLE_CATEGORY] = df[TARGET_VARIABLE_AQI].apply(get_aqi_category)
    
    category_order = ["Good", "Moderate", "Unhealthy", "Hazardous"] # Defined in utils.py and consistent
    
    target_encoder = OrdinalEncoder(categories=[category_order], handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Shift the category by the prediction horizon.
    # This requires a 'City' column for groupby, which is likely dropped during preprocessing.
    # We need to re-evaluate how the target variable is shifted if 'City' is not available.
    # For now, I will comment out the groupby and apply shift directly.
    # df['target_encoded'] = df.groupby('City')[TARGET_VARIABLE_CATEGORY].shift(-PREDICTED_HORIZON_HOURS)
    df['target_encoded'] = df[TARGET_VARIABLE_CATEGORY].shift(-PREDICTED_HORIZON_HOURS)
    df['target_encoded'] = target_encoder.fit_transform(df[['target_encoded']])
    
    # Remove rows where target is NaN or 'Unknown'
    df = df[df['target_encoded'] != -1]
    df = df.dropna(subset=['target_encoded'])
    
    print(f"Data with target variable. {len(df)} rows ready for training.")

    # --- 3. Feature Selection & Time-based Feature Creation ---
    # Features will be all columns except original AQI, 'AQI_Category', and 'target_encoded'
    # 'Date' and 'City' are no longer columns after preprocessing and setting DateTime as index.
    features_to_drop_from_X = [TARGET_VARIABLE_AQI, TARGET_VARIABLE_CATEGORY, 'target_encoded']
    X = df.drop(columns=[col for col in features_to_drop_from_X if col in df.columns])
    y = df['target_encoded']
    
    # Explicitly add time-based features from the DateTime index
    X['hour'] = X.index.hour
    X['day_of_week'] = X.index.dayofweek
    X['day_of_month'] = X.index.day
    X['month'] = X.index.month

    # Removed 'City_encoded' as it's not generated in preprocessing with current setup.
    # Need to check `src/utils.py` for `CITIES` and if there's an encoding step missing.

    # --- 4. Data Splitting (Time-based) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # --- 5. Build Full Pipeline ---
    
    # Update numeric_features and categorical_features based on the new preprocessing output
    # `City_encoded` and 'Date' are no longer present as columns after preprocessing.
    # Time-based features are now created from the index.
    numeric_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64'] and col not in ['hour', 'day_of_week', 'day_of_month', 'month']]
    categorical_features = ['hour', 'day_of_week', 'day_of_month', 'month']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Impute NaNs potentially created by lags in pre-processing
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor_pipeline = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        objective='multi:softmax',
        num_class=len(category_order),
        random_state=42
    )

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_pipeline),
        ('classifier', model)
    ])

    # --- 6. Training ---
    print("Fitting the full classification pipeline...")
    full_pipeline.fit(X_train, y_train)
    print("Training complete.")

    # --- 7. Evaluation ---
    y_pred = full_pipeline.predict(X_test)
    
    y_test_decoded = target_encoder.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
    y_pred_decoded = target_encoder.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    print(f"\n--- Model Evaluation (Hold-out Set) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test_decoded, y_pred_decoded, labels=category_order, zero_division=0))
    print("------------------------------------")

    # --- 8. Save Model Pipeline and Encoder ---
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(full_pipeline, MODEL_SAVE_PATH)
    joblib.dump(target_encoder, ENCODER_SAVE_PATH)
    print(f"✅ Full model pipeline saved to: {MODEL_SAVE_PATH}")
    print(f"✅ Target Encoder saved to: {ENCODER_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
