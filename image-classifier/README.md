# A Simple MLOps Pipeline on Your Local Machine

<https://medium.com/data-science/to-serve-man-60246a82d953>

Our service is simply going to read in an image as a tensor and then return it.

```shell
cd image-classifier
pyenv local 3.12.11

# Create a virtual environment
uv venv
source .venv/bin/activate

uv sync

cd ml-app
pip install -r requirements.txt
```

Build the image and push it to your cluster’s repository:

```shell
docker build --no-cache -t image-classifier-model:0.1 .
docker tag image-classifier-model:latest <your repo>/image-classifier-model:0.1
docker push <your repo>/image-classifier-model:0.1
```

**Serve Local**: Now that we have an image built, let’s test it by running our seldon-core-microservice as a stand alone docker container:

```shell
docker run -d --rm --name image-classifier-model -p 5000:5000 image-classifier-model:0.1

docker logs -f image-classifier-model
2020-02-21 01:49:38,651 - seldon_core.microservice:main:190 - INFO:  Starting microservice.py:main
...
2020-02-21 01:49:38,659 - __imageclassifiermodel__:__init__:15 - INFO:  initializing...
2020-02-21 01:49:38,659 - __imageclassifiermodel__:__init__:16 - INFO:  load model here...
2020-02-21 01:49:38,659 - __imageclassifiermodel__:__init__:18 - INFO:  model has been loaded and initialized...
2020-02-21 01:49:38,659 - seldon_core.microservice:main:325 - INFO:  REST microservice running on port 5000
2020-02-21 01:49:38,659 - seldon_core.microservice:main:369 - INFO:  Starting servers
 * Serving Flask app "seldon_core.wrapper" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
2020-02-21 01:49:38,673 - werkzeug:_log:113 - INFO:   * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
```

Notice that the Model class you wrote above got loaded and is being served up as a Flask application on port 5000. The model’s prediction endpoint is at `/predict` or `/api/v0.1/predictions`.

# Deploy It!

Like all things Kubernetes, Seldon Core defines its own Deployment object called a **SeldonDeployment** via a **Custom Resource Definition (CRD)** file.

Let’s define our model’s SeldonDeployment via a **seldon-deploy.yaml** file.
Let’s break it down:
- A **SeldonDeployment** consists of one or more predictors which defines what models are contained within this deployment. Note: You may want to define more than one predictor for **canary** or **multi-arm bandit** type scenarios.
- Every predictor consists of a Pod spec that defines the Docker image of your model code which we built above.
- Since a SeldonDeployment is a type of Deployment, each predictor is backed by one or more **ReplicaSets** which defines how many Pods should be created to back your model (inference graph). This is one of the ways that allows you to scale your SeldonDeployment to meet your compute needs.
- We also set a custom service name since a **SeldonDeployment** which will auto-expose our microservice as Kubernetes Service object.

```shell
kubectl get seldondeployments
kubectl create -f seldon-deploy.yaml
kubectl get all
```

Notice a few things:
- The Pod created that runs your model code has actually two containers in it — your model’s code as well as the “sidecar” **seldon-container-engine** which is injected at deployment time.
- The seldon-container-engine will marshal all requests and responses into the Seldon Core message format as well as monitor your model container’s health and telemetry.
- A ClusterIP service is exposed on port **8000** for us to communicate with our microservice.

Now that our model microservice is up and running, let’s test it using the same client above. We need to either setup a publicly accessible Ingress or use port-forwarding to create a direct connection to our Service object.

In our simple example, let’s just use a simple port-forward:

```shell
kubectl port-forward svc/image-classifier-model-svc 8000:8000

python3 tests.py http://localhost:8000/api/v0.1/predictions test_image.jpg
```

# Clean Up

```shell
kubectl delete -f seldon-deploy.yaml
seldondeployment.machinelearning.seldon.io "image-classifier-model" deleted
```