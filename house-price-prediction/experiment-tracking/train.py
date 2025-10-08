#!/usr/bin/env python3

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from mlflow.models import infer_signature

# training a Random Forest Regressor - an ensemble machine learning model that uses multiple decision trees to make predictions.
# The model will be trained to predict house prices based on various property features.
# All data is stored in PostgreSQL (experiments/runs) and MinIO (model artifacts)

# Set up MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("rsinghai-test")

# Enable MLflow's automatic experiment tracking for scikit-learn
# mlflow.sklearn.autolog()
# Manual MLflow tracking (autolog disabled to avoid duplicate runs)

# King County House Sales dataset (kc_house_data.csv), which contains real estate data from King County, Washington (where Seattle is located).
df = pd.read_csv("kc_house_data.csv") 

# choose features: 10 of them
features = ["bedrooms","bathrooms","sqft_living","sqft_above","grade",
            "floors","view",'sqft_lot','waterfront','zipcode']

# getting those features from the dataframe
x = df[features]
y = df["price"] # target variable - price - The sale price of the house (what you're trying to predict)

# Train/Test Split: 70% training data, 30% test data
# Random State: 3 (for reproducibility)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3)

# Start MLflow run in a context, This ensures all logging happens within a single run
with mlflow.start_run():
    # choose model with settings
    # Algorithm: Random Forest with 100 decision trees (n_estimators=100)
    model = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=10)

    # Train the model
    # MLflow triggers logging automatically upon model fitting
    model.fit(x_train, y_train)

    # Performance Metrics: R² score for both training and test datasets
    metrics = {"train_score": model.score(x_train, y_train), "test_score": model.score(x_test, y_test)}
    print(metrics)

    # MLflow Tracking
    # Create proper input example and infer signature
    input_example = x_test.iloc[0:5]  # Use a few rows for better example
    predictions = model.predict(input_example)
    signature = infer_signature(input_example, predictions)

    # Log model, params, and metrics to MLflow
    mlflow.sklearn.log_model(
        model,
        name="rf-regressor",
        signature=signature,
        input_example=input_example
    )
    mlflow.log_params({"n_estimators": 100, "max_depth": 6, "max_features": 10})  # log all params
    mlflow.log_metrics(metrics)  # Performance metrics (train_score and test_score - R² scores)
