FROM python:3.12-slim-trixie

# Seldon Core specific
WORKDIR /microservice
COPY ./requirements.txt /microservice
RUN pip3 install --no-cache -r requirements.txt

COPY . /microservice

ENV MODEL_NAME=HousePricePredictionModel
ENV SERVICE_TYPE=MODEL

CMD exec seldon-core-microservice ${MODEL_NAME} --service-type ${SERVICE_TYPE}

# Expose both ports: many Seldon Core microservice builds use 9000 for REST and 5000 for gRPC/metrics depending on config
EXPOSE 9000 5000 6000
