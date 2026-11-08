"""
House Price Prediction - Model Training Script
Trains an XGBoost regression pipeline on housing.csv and serializes the model.
"""

import os
import shutil
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "housing.csv")
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")
BACKUP_PATH = os.path.join(BASE_DIR, "house_price_model.pkl.bak")
TARGET_COL = "median_house_value"
RANDOM_STATE = 42

def train_and_save_model():
    print(f"Loading dataset from: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Dataset shape: {df.shape}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=RANDOM_STATE,
            )),
        ]
    )

    print("Fitting model pipeline...")
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    train_preds = model_pipeline.predict(X_train)
    test_preds = model_pipeline.predict(X_test)

    train_rmse = root_mean_squared_error(y_train, train_preds)
    train_mae = mean_absolute_error(y_train, train_preds)
    train_r2 = r2_score(y_train, train_preds)

    test_rmse = root_mean_squared_error(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)

    print("\n--- Training Evaluation ---")
    print(f"Train RMSE: ${train_rmse:,.2f} | MAE: ${train_mae:,.2f} | R2: {train_r2:.4f}")
    print(f"Test  RMSE: ${test_rmse:,.2f} | MAE: ${test_mae:,.2f} | R2: {test_r2:.4f}")

    # Backup existing model if backup doesn't exist
    if os.path.exists(MODEL_PATH) and not os.path.exists(BACKUP_PATH):
        shutil.copy(MODEL_PATH, BACKUP_PATH)
        print(f"Original model backed up to {BACKUP_PATH}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_pipeline, f)
    print(f"Model saved successfully to {MODEL_PATH}")

    return model_pipeline

if __name__ == "__main__":
    train_and_save_model()
