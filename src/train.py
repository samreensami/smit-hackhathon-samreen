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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier # Import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from src.preprocessing import load_and_preprocess_data # Import the new preprocessing function
from src.utils import CITIES, get_aqi_category # Updated function name

# --- Configuration ---
RAW_DATA_PATH = "data/raw/AQM-dataset-updated.csv"
PROCESSED_DATA_PATH = "data/processed/preprocessed_aqm_data.csv"
MODEL_SAVE_PATH = "models/aqi_model.joblib" # The best model pipeline will be saved here
ENCODER_SAVE_PATH = "models/target_encoder.joblib" # Save encoder separately
ALL_MODELS_SAVE_PATH = "models/all_models_pipelines.joblib" # Save all model pipelines
MODEL_COMPARISON_SAVE_PATH = "models/model_comparison_results.joblib" # Save model comparison results

TARGET_VARIABLE_AQI = "CO_AQI" # Changed target variable to CO_AQI
TARGET_VARIABLE_CATEGORY = "AQI_Category"
PREDICTED_HORIZON_HOURS = 24  # Predict 1 day ahead (AQI Category for next day)
NUM_LAGS = 3 # Number of lag features to create for preprocessing

def train_model():
    """
    Loads data, preprocesses, trains multiple models, evaluates them,
    and saves the best performing model (by F1-Score) along with the target encoder
    and all trained model pipelines.
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
        joblib.dump({}, ALL_MODELS_SAVE_PATH) # Save empty dict for all models
        joblib.dump(pd.DataFrame(), MODEL_COMPARISON_SAVE_PATH) # Save empty df for comparison
        return

    print(f"Loaded and preprocessed {len(df)} rows.")

    # --- 2. Create Target Variable ---
    df[TARGET_VARIABLE_CATEGORY] = df[TARGET_VARIABLE_AQI].apply(get_aqi_category)
    
    category_order = ["Good", "Moderate", "Unhealthy", "Hazardous"] # Defined in utils.py and consistent
    
    target_encoder = OrdinalEncoder(categories=[category_order], handle_unknown='use_encoded_value', unknown_value=-1)
    
    df['target_encoded'] = df[TARGET_VARIABLE_CATEGORY].shift(-PREDICTED_HORIZON_HOURS)
    df['target_encoded'] = target_encoder.fit_transform(df[['target_encoded']])
    
    df = df[df['target_encoded'] != -1]
    df = df.dropna(subset=['target_encoded'])
    
    print(f"Data with target variable. {len(df)} rows ready for training.")

    # --- 3. Feature Selection & Time-based Feature Creation ---
    features_to_drop_from_X = [TARGET_VARIABLE_AQI, TARGET_VARIABLE_CATEGORY, 'target_encoded']
    X = df.drop(columns=[col for col in features_to_drop_from_X if col in df.columns])
    y = df['target_encoded']
    
    X['hour'] = X.index.hour
    X['day_of_week'] = X.index.dayofweek
    X['day_of_month'] = X.index.day
    X['month'] = X.index.month

    # --- 4. Data Splitting (Time-based) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # --- 5. Build Preprocessing Pipeline (ColumnTransformer) ---
    numeric_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64'] and col not in ['hour', 'day_of_week', 'day_of_month', 'month']]
    categorical_features = ['hour', 'day_of_week', 'day_of_month', 'month']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
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
    
    # --- 6. Define Multiple Models for Comparison ---
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            objective='multi:softmax', num_class=len(category_order), random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced',
            n_jobs=-1 # Use all available cores
        ),
        "DecisionTree": DecisionTreeClassifier( # Replaced SVM with Decision Tree
            random_state=42, class_weight='balanced'
        )
    }

    results = {}
    all_trained_pipelines = {} # To store all trained pipelines
    for name, classifier in models.items():
        print(f"\n--- Training {name} Model ---")
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor_pipeline),
            ('classifier', classifier)
        ])
        
        # --- Training ---
        print(f"Fitting the full classification pipeline for {name}...")
        full_pipeline.fit(X_train, y_train)
        print(f"Training of {name} complete.")
        all_trained_pipelines[name] = full_pipeline # Store the trained pipeline

        # --- Evaluation ---
        y_pred = full_pipeline.predict(X_test)
        
        # Decode predictions for classification report labels
        y_test_decoded = target_encoder.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
        y_pred_decoded = target_encoder.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        print(f"\n--- {name} Model Evaluation (Hold-out Set) ---")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision (macro): {precision:.4f}")
        print(f"  Recall (macro): {recall:.4f}")
        print(f"  F1-Score (macro): {f1:.4f}")
        print(classification_report(y_test_decoded, y_pred_decoded, labels=category_order, zero_division=0))
        print("------------------------------------")
    
    # --- 7. Model Comparison Table ---
    comparison_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [res["accuracy"] for res in results.values()],
        "Precision": [res["precision"] for res in results.values()],
        "Recall": [res["recall"] for res in results.values()],
        "F1-Score": [res["f1_score"] for res in results.values()]
    })
    print("\n--- Model Comparison Summary ---")
    print(comparison_df.to_string(index=False))
    print("------------------------------")

    # --- 8. Select and Save Best Model & All Models ---
    best_model_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model_pipeline = all_trained_pipelines[best_model_name] # Get the best pipeline from the stored ones
    print(f"\n✅ Selected best model by F1-Score: {best_model_name}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(best_model_pipeline, MODEL_SAVE_PATH)
    joblib.dump(target_encoder, ENCODER_SAVE_PATH) # Save encoder as it's used by all models
    joblib.dump(all_trained_pipelines, ALL_MODELS_SAVE_PATH) # Save all trained model pipelines
    joblib.dump(comparison_df, MODEL_COMPARISON_SAVE_PATH) # Save model comparison results

    print(f"✅ Best model pipeline ({best_model_name}) saved to: {MODEL_SAVE_PATH}")
    print(f"✅ Target Encoder saved to: {ENCODER_SAVE_PATH}")
    print(f"✅ All trained model pipelines saved to: {ALL_MODELS_SAVE_PATH}")
    print(f"✅ Model comparison results saved to: {MODEL_COMPARISON_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
