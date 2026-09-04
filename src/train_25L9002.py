# Student ID: 25L9002

print("Loading the dataset for Student ID: 25L9002")

# train.py
# MLOps House Price Prediction Project
# Student ID: 25L9002

import os
import pandas as pd
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# Student Information
# --------------------------------------------------

STUDENT_ID = "25L9002"

print("=" * 50)
print("MLOps House Price Prediction")
print(f"Student ID: {STUDENT_ID}")
print("=" * 50)


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

print("\nLoading dataset...")



# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

data = housing.frame

print("Dataset loaded successfully.")
print(f"Dataset shape: {data.shape}")

# Save raw dataset into data/ directory
DATA_PATH = "data/dataset.csv"
data.to_csv(DATA_PATH, index=False)

print(f"Dataset saved to: {DATA_PATH}")


# --------------------------------------------------
# 2. Prepare Features and Target
# --------------------------------------------------

X = data.drop(columns=["MedHouseVal"])
y = data["MedHouseVal"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget: MedHouseVal")


# --------------------------------------------------
# 3. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

EXPERIMENT = "feature branch - 25L9002"
# --------------------------------------------------
# 4. Train Machine Learning Model
# --------------------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------------------------
# 5. Evaluate Model
# --------------------------------------------------

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-" * 30)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")


# --------------------------------------------------
# 6. Save Trained Model
# --------------------------------------------------

os.makedirs("model", exist_ok=True)

MODEL_PATH = f"model/house_price_model_{STUDENT_ID}.joblib"

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print(f"Model path: {MODEL_PATH}")

print("\nTraining pipeline completed successfully.")

