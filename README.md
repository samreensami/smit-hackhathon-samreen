# Air Quality Category Prediction (ML) - Pakistan (SMIT Hackathon)

This repository contains a machine learning project to predict Air Quality Index (AQI) **categories** for a single location, derived from the `AQM-dataset-updated.csv`. While the interactive Streamlit application provides a multi-city selection, it's important to note that the underlying model is trained on and makes predictions based on this single dataset, treating it as a generic location.

## 1. Project Overview

The primary objective is to forecast AQI categories using historical sensor and AQI data. This project uses time-series classification techniques to predict future air quality categories, which is crucial for public health advisories and environmental monitoring.

### Key Features:
- **Time-Series Classification**: Predicts AQI categories for a `PREDICTED_HORIZON_HOURS` (24 hours) ahead.
- **End-to-End Solution**: Includes data preprocessing, model training, and an interactive Streamlit UI.
- **Interactive UI**: A Streamlit application allows users to select a city (for display purposes), generate a forecast, view the predicted AQI category, and get "Red Alerts" for unhealthy conditions.
- **Modular Code**: Well-structured Python code in `src/` for maintainability.

## 2. Repository Structure

The project is organized as follows:

```
smit-hackhathon/
├── data/               # Raw and processed AQI datasets
├── models/             # Saved .joblib models and encoders
├── src/
│   ├── preprocessing.py # Feature engineering & lags logic
│   ├── train.py         # Model training & comparison script
│   └── utils.py         # Helper functions for AQI categories
├── app.py              # Main Streamlit Dashboard
└── requirements.txt    # Project dependencies
```

## 3. Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites
- Python 3.10 or higher
- `pip` for package management

### Step 1: Clone the Repository
```bash
git clone https://github.com/samreensami/smit-hackhathon-samreen
cd smit-hackhathon-samreen
```

### Step 2: Set Up a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
Install all required packages from `requirements.txt`.
```bash
pip install -r requirements.txt
```

### Step 4: Download the Data
1.  **Crucially**, ensure your dataset named `AQM-dataset-updated.csv` is placed inside the `data/raw/` directory. This project expects a CSV file with columns: `"Date"`, `"Time"`, `"Temperature"`, `"Humidity"`, `"CO_AQI"`, `"NO2_AQI"`, `"SO2_AQI"`.

### Step 5: Train the Model
Run the training script. This will perform data preprocessing, train the `ColumnTransformer` and `XGBoost Classifier` within a single pipeline, and save this full pipeline along with the target encoder to the `models/` directory.
```bash
python src/train.py
```
This process might take a few minutes depending on your machine. You should see output indicating the model components are saved.

### Step 6: Run the Inference Script
To perform a single inference using the trained model, run:
```bash
python src/inference.py
```
This will print the predicted AQI category for `PREDICTED_HORIZON_HOURS` ahead based on the latest data.

### Step 7: Generate a Forecast to CSV
To generate a forecast and save it to a CSV file in the `results/` directory, run:
```bash
python src/generate_forecast.py
```
This will output a CSV containing the predicted AQI category for `PREDICTED_HORIZON_HOURS` ahead.

### Step 8: Run the Streamlit Application
Once the model components are trained and saved, you can launch the interactive web app.
```bash
streamlit run app.py
```
Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.

## 4. Methodology

### 4.1. Data Preparation and Feature Engineering
- **Data Loading**: The dataset `AQM-dataset-updated.csv` is loaded.
- **`src/preprocessing.py` (`load_and_preprocess_data` function)**: This function is responsible for initial data processing:
    - **DateTime Index**: Combines `Date` and `Time` columns into a single `DateTime` index.
    - **Lag Features**: Creates lagged values for `Temperature`, `Humidity`, `CO_AQI`, `NO2_AQI`, and `SO2_AQI`. The number of lags (`NUM_LAGS`) is configurable.
    - **Missing Value Handling**: Rows with `NaN` values resulting from lag feature creation are dropped.
- **Target Variable**: The target for the model is the `AQI_Category` for `PREDICTED_HORIZON_HOURS` (24 hours) into the future, derived from the `CO_AQI` values. `AQI_Category` is based on simplified Pakistan EPA thresholds defined in `src/utils.py`:
    - **Good**: 0-50
    - **Moderate**: 51-100
    - **Unhealthy**: 101-250
    - **Hazardous**: 251+
- **Feature Preprocessing (`ColumnTransformer` within pipeline)**: A `ColumnTransformer` processes the features further within the training pipeline:
    - **Numeric Features**: Imputed (for any `NaN`s potentially introduced by lags) and scaled using `StandardScaler`. This includes original sensor/AQI readings and their lagged versions.
    - **Categorical Features**: Time-based features (`hour`, `day_of_week`, `day_of_month`, `month`) extracted from the `DateTime` index are One-Hot Encoded.

### 4.2. Model Training
- **Model Choice**: An `XGBoost Classifier` is used to predict the discrete `AQI_Category`. The categories are internally encoded into numerical values using `OrdinalEncoder` for the classifier.
- **Full Pipeline**: The `ColumnTransformer` and the `XGBoost Classifier` are combined into a single `scikit-learn` `Pipeline`. This `full_pipeline` is what gets trained and saved, ensuring consistency between preprocessing and prediction.
- **Model Saving**: The trained `full_pipeline` (`aqi_model.joblib`) and the `OrdinalEncoder` for the target variable (`target_encoder.joblib`) are both saved to the `models/` directory.
- **Model Evaluation**: The model's performance is evaluated using classification metrics such as `accuracy_score` and a `classification_report` on a hold-out test set.

### 4.3. Forecasting & Application (Streamlit)
- **Forecasting Approach**: The Streamlit application generates a single point AQI category forecast for `PREDICTED_HORIZON_HOURS` ahead based on the latest available data from the `AQM-dataset-updated.csv`.
    - The model predicts a categorical `AQI_Category` directly.
    - The application's UI allows selection of different cities, but it's important to understand that the underlying model is **city-agnostic** because the training data (`AQM-dataset-updated.csv`) does not contain a 'City' column. The displayed forecast is based on the general model and labeled with the selected city for user context.
    - City statistics (average Temperature and Humidity) displayed in the sidebar are calculated from the entire `AQM-dataset-updated.csv` dataset and are presented with a note that they are not truly city-specific due to data limitations.
- **Red Alert System**: The Streamlit app displays a prominent "RED ALERT: Unhealthy or Hazardous Air Quality Expected!" warning if the forecasted AQI category for the selected city falls into these critical ranges.

---
*This project was created for the SMIT Hackathon.*