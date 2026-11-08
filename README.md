# 🏠 California House Price Prediction System

A Machine Learning web application that predicts California residential property values using an **XGBoost Regression Pipeline** and **Flask**. The application features an interactive UI with instant estimation, input validation, sample presets, and a RESTful API endpoint.

---

## 📖 Project Overview

Accurate property valuation is critical for buyers, sellers, and real estate professionals. This project applies modern Machine Learning techniques to estimate median house values across California neighborhoods based on geographic, demographic, and housing indicators from the California Housing Census dataset.

The system is built with **Python**, **Flask**, **Scikit-Learn**, and **XGBoost**, packaged as an end-to-end reproducible machine learning pipeline.

---

## 🎯 Objectives

- Predict California median house values with high accuracy.
- Complete machine learning pipeline: preprocessing, imputation, standard scaling, and one-hot encoding.
- Build a user-friendly, responsive web application using Flask.
- Provide a programmatic REST API endpoint for integration.
- Include unit testing and automated validation to prevent crashes.

---

## ✨ Features

- 🏠 **Instant Price Prediction**: Real-time evaluation powered by XGBoost.
- ⚡ **One-Click Quick Samples**: Pre-loaded realistic sample data (SF Bay Area, LA Coastal, Central Inland).
- 🛡️ **Robust Input Validation**: Validates numerical bounds, types, and categories with user-friendly error banners.
- 💾 **Form State Persistence**: Inputs remain populated after submission.
- 🔌 **REST API Endpoint**: Programmatic predictions via `/api/predict` (JSON input/output).
- 🏥 **Health Check**: `/health` endpoint for monitoring and cloud deployment.
- 🎨 **Modern Responsive UI**: Clean card-based design with mobile-first CSS.
- 🔄 **Reproducible Training**: Standalone `train_model.py` script to retrain and evaluate the model pipeline at any time.

---

## 🛠 Tech Stack

- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas, NumPy
- **Web Framework**: Flask
- **Frontend**: HTML5, Modern CSS3, Vanilla JavaScript
- **Deployment**: Gunicorn (Linux/Heroku/Render), Waitress / Flask dev server (Windows)

---

## 📂 Project Structure

```
HOUSE PREDICTION MODEL
│
├── static/
│   └── style.css                 # Modern CSS styles and responsive layout
│
├── templates/
│   └── index.html                # Interactive UI template with sample presets
│
├── app.py                        # Flask application & REST API
├── train_model.py                # Standalone model training & evaluation script
├── test_suite.py                 # Comprehensive unit & integration tests
├── test_model.py                 # Quick model verification script
├── housing.csv                   # California housing dataset
├── house_price_model.pkl         # Trained XGBoost pipeline (Preprocessor + Model)
├── housingPricePrediction.ipynb  # Exploratory Data Analysis & Model Comparison notebook
├── Procfile                      # Process file for Heroku/Render deployment
├── requirements.txt              # Project dependencies
├── runtime.txt                   # Target Python runtime specification
└── README.md                     # Documentation
```

---

## 📊 Dataset Features

The project utilizes the **California Housing Dataset** (20,640 records):

| Feature | Type | Description |
|---|---|---|
| `longitude` | Continuous | East-West coordinate location |
| `latitude` | Continuous | North-South coordinate location |
| `housing_median_age` | Continuous | Median age of houses in block (years) |
| `total_rooms` | Continuous | Total count of rooms across block |
| `total_bedrooms` | Continuous | Total count of bedrooms across block |
| `population` | Continuous | Total population living in the block |
| `households` | Continuous | Total number of households (family units) |
| `median_income` | Continuous | Median income for households ($10,000 units) |
| `ocean_proximity` | Categorical | Proximity: `<1H OCEAN`, `INLAND`, `NEAR BAY`, `NEAR OCEAN`, `ISLAND` |
| `median_house_value` | Target | Median house value in USD ($) |

---

## 🤖 Model Performance

The final pipeline uses an **XGBoost Regressor** with a `ColumnTransformer` preprocessing pipeline:
- Numerical Features: `SimpleImputer(strategy="median")` + `StandardScaler()`
- Categorical Features: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`

| Metric | Train Set | Test Set (20% holdout) |
|---|---|---|
| **R² Score** | 0.8880 | **0.8205** |
| **RMSE** | $38,689 | **$48,496** |
| **MAE** | $26,745 | **$32,453** |

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Train or Retrain the Model
```bash
python train_model.py
```

### 3. Run the Automated Tests
```bash
python test_suite.py
```

### 4. Start the Application
```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000/`.

---

## 🔌 REST API Documentation

### Endpoint: `POST /api/predict`
Predict property valuation from JSON payload.

#### Request Headers
`Content-Type: application/json`

#### Request Body
```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41.0,
  "total_rooms": 880.0,
  "total_bedrooms": 129.0,
  "population": 322.0,
  "households": 126.0,
  "median_income": 8.32,
  "ocean_proximity": "NEAR BAY"
}
```

#### Success Response (200 OK)
```json
{
  "success": true,
  "prediction": 430119.5,
  "formatted_prediction": "$430,119.50"
}
```

#### Error Response (400 Bad Request)
```json
{
  "success": false,
  "error": "Missing required field: Longitude"
}
```

---

## 🏥 Health Check

`GET /health`

Response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```
