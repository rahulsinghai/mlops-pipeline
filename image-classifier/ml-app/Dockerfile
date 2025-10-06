FROM python:3.12-slim-trixie

RUN apt-get update && DEBIAN_FRONTEND=noninteractive && apt-get install -y \
   curl \
   python3-setuptools && \
apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Seldon Core specific
COPY . /microservice
WORKDIR /microservice
COPY ./requirements.txt /microservice
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm -f get-pip.py
RUN pip3 install --no-cache -r requirements.txt

ENV MODEL_NAME=HousePricePredictionModel
ENV SERVICE_TYPE=MODEL

CMD exec seldon-core-microservice ${MODEL_NAME} --service-type ${SERVICE_TYPE}

# Expose both ports: many Seldon Core microservice builds use 9000 for REST and 5000 for gRPC/metrics depending on config
EXPOSE 9000 5000 6000
