#!/usr/bin/env python3

import json
import logging
import os
import pandas as pd
import requests

logger = logging.getLogger('__imageclassifiermodelclient__')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == '__main__':
    # Load dataset used during training
    df = pd.read_csv("../experiment-tracking/kc_house_data.csv")

    # Feature list must match training exactly (order matters)
    features = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_above",
        "grade",
        "floors",
        "view",
        "sqft_lot",
        "waterfront",
        "zipcode",
    ]

    feature_df = df[features]
    row = list(feature_df.loc[7, :])

    # If you ran: docker run -p 5001:5000 -it seldon-app
    # then host port is 5001. Change to 5000 if you mapped directly.
    PREDICT_URL = "http://localhost:9002/predict" # http
    # PREDICT_URL = "http://localhost:5001/predict" # grpc

    payload = {"data": {"ndarray": [row]}}
    resp = requests.post(PREDICT_URL, json=payload, timeout=10)
    print("Status:", resp.status_code)
    print("Response:", resp.text)

    # Basic assertion (optional)
    if resp.status_code != 200:
        raise SystemExit(f"Prediction failed: {resp.status_code} {resp.text}")
