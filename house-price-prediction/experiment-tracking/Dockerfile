FROM python:3.12-slim-trixie

RUN apt-get update && apt-get install -y git \
    && pip install mlflow scikit-learn psycopg2-binary boto3 minio \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /training
COPY ./tests.py .
COPY ./kc_house_data.csv .
COPY ./train.py .
