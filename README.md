# Air Quality Level Prediction (ML) - Pakistan (SMIT Hackathon)

This repository contains a machine learning project to predict Air Quality Index (AQI) **categories** for selected major cities in Pakistan. The project provides an end-to-end solution including data preprocessing, model training, and an interactive Streamlit web application for category forecasting.

## 1. Project Overview

The primary objective is to forecast AQI categories directly from PM2.5 concentrations and other weather data. This project uses time-series classification techniques to predict future air quality categories, which is crucial for public health advisories and environmental monitoring.

### Key Features:
- **Time-Series Classification**: Predicts AQI categories for the next 1-3 days.
- **End-to-End Solution**: Includes data preprocessing, model training, and a Streamlit UI.
- **Interactive UI**: A Streamlit application allows users to select a city, choose the forecast horizon (1-3 days), view a table with 'Date', 'Predicted AQI Value' (inferred), and 'Category', and get "Red Alerts" for unhealthy conditions.
- **Modular Code**: Well-structured Python code in `src/` for maintainability.

## 2. Repository Structure

The project is organized as follows:

```
├── data/
│   └── raw/            # Placeholder for the raw dataset CSV (AQM-dataset-updated.csv).
├── models/             # Stores the trained XGBClassifier, scikit-learn preprocessor, and target encoder.
│   ├── aqi_model.joblib      # The trained full pipeline (ColumnTransformer + XGBClassifier).
│   └── target_encoder.joblib # The OrdinalEncoder used for target categories.
├── src/
│   ├── preprocessing.py # Contains the `preprocess_data` function for data cleaning and feature engineering.
│   ├── train.py         # Main script to train the full pipeline and save it along with the target encoder.
│   └── utils.py         # Helper functions (e.g., AQI category mapping, city lists, city-to-code mapping).
├── app.py              # The Streamlit web application.
├── requirements.txt    # Required Python packages.
└── README.md           # This project report.
```

## 3. Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites
- Python 3.10 or higher
- `pip` for package management

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
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
1.  **Crucially**, ensure your dataset named `AQM-dataset-updated.csv` is placed inside the `data/raw/` directory. This project expects a CSV file with columns similar to: `timestamp`, `city`, `PM2.5`, `temperature`, `humidity`, `wind_speed`.

### Step 5: Train the Model
Run the training script. This will perform data preprocessing, train the `ColumnTransformer` and `XGBoost Classifier` within a single pipeline, and save this full pipeline along with the target encoder to the `models/` directory.
```bash
python src/train.py
```
This process might take a few minutes depending on your machine. You should see output indicating the model components are saved.

### Step 6: Run the Streamlit Application
Once the model components are trained and saved, you can launch the interactive web app.
```bash
streamlit run app.py
```
Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.

## 4. Methodology

### 4.1. Data Preparation and Feature Engineering
- **Data Loading**: The dataset `AQM-dataset-updated.csv` is loaded.
- **`src/preprocessing.py` (`preprocess_data` function)**: This standalone function performs the initial data processing:
    - **Missing Value Handling**: Numeric columns are imputed using the `median` within each city group.
    - **Lag Features**: Creates lagged values for `PM2.5`, `temperature`, and `humidity` (`_lag_1`, `_lag_2`) and a 3-hour rolling average for `PM2.5` (`pm25_rolling_avg_3`), grouped by city.
    - **City Encoding**: The `City` column is converted to a numerical representation (`City_encoded`) using a dictionary mapping defined in `src/utils.py`.
    - **Time-based Features**: After `preprocess_data` is run, additional time-based features (`hour`, `day_of_week`, `day_of_month`, `month`) are extracted from the `Date` column in `src/train.py` (and similarly for forecasting).
- **Target Variable**: The target for the model is the `AQI_Category` for 24 hours (1 day) into the future. `AQI_Category` is derived from `PM2.5` values using simplified Pakistan EPA thresholds defined in `src/utils.py`:
    - **Good**: 0-50
    - **Moderate**: 51-100
    - **Unhealthy**: 101-250
    - **Hazardous**: 251+
- **Feature Preprocessing (`ColumnTransformer` within pipeline)**: A `ColumnTransformer` processes the features further:
    - **Numeric Features**: Imputed (for any NaNs introduced by lags) and scaled using `StandardScaler`.
    - **Categorical Features**: One-hot encoded using `OneHotEncoder`.

### 4.2. Model Training
- **Model Choice**: An `XGBoost Classifier` is used to predict the discrete `AQI_Category`. The categories are internally encoded into numerical values using `OrdinalEncoder` (with a predefined order) for the classifier.
- **Full Pipeline**: The `ColumnTransformer` and the `XGBoost Classifier` are combined into a single `scikit-learn` `Pipeline`. This `full_pipeline` is what gets trained and saved, ensuring consistency between preprocessing and prediction.
- **Model Saving**: The trained `full_pipeline` (`aqi_model.joblib`) and the `OrdinalEncoder` for the target variable (`target_encoder.joblib`) are both saved to the `models/` directory.
- **Model Evaluation**: The model's performance is evaluated using classification metrics such as `accuracy_score` and a `classification_report` on a hold-out test set.

### 4.3. Forecasting (Streamlit Application)
- **Forecasting Approach**: The Streamlit application generates an AQI category forecast for a user-selected number of days (1-3). The process involves:
    1.  Loading historical raw data for the selected city.
    2.  Manually constructing future input data points by approximating future numerical features (like `PM2.5`, `temperature`, `humidity`, `wind_speed`) based on their values from the last available historical record.
    3.  Adding relevant lagged and rolling features for these future points by utilizing the logic similar to `preprocess_data`.
    4.  Adding time-based features (`hour`, `day_of_week`, `day_of_month`, `month`) for the future dates.
    5.  Passing this complete feature set for the future to the loaded `full_pipeline` for prediction.
    6.  Decoding the numerical predictions into human-readable AQI categories using the `target_encoder`.
    - **Predicted AQI Value**: For display purposes, a "Predicted AQI Value" is *inferred* from the predicted AQI category (e.g., using a random value within the category's range). This is **not** a direct numerical prediction from the `XGBClassifier` model itself.
    - **Important Note on Iteration**: This forecasting approach does **not** iteratively feed its own predicted AQI categories back into the model as input for the next day's prediction. The predictions are based on approximations of future numerical inputs from the last known historical data.
- **Red Alert System**: The Streamlit app displays a prominent "RED ALERT: Hazardous Air Quality Expected" warning if any of the forecasted AQI categories are "Unhealthy" or "Hazardous", according to the defined thresholds.

## 5. Usage Examples

### Run the Streamlit App
```bash
streamlit run app.py
```
This will open the web application where you can select a city, choose the forecast horizon, and generate an AQI category forecast.

---
*This project was created for the SMIT Hackathon.*
