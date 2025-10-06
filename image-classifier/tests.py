#!/usr/bin/env python3

import base64
import logging
import numpy as np
import requests
import sys

from PIL import Image

logger = logging.getLogger('__imageclassifiermodelclient__')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# a simple client script that will POST an image file to our model’s predict endpoint
# Usage: python3 tests.py http://localhost:5000/api/v0.1/predictions test_image.jpg
# Example: python3 tests.py http://localhost:5001/predict ../data/cat.jpg

# We pass the URL we want to post the image to and the path to the image itself.
# We need to base64 encode the image in preparation for the POST to our Seldon Core microservice running locally as a container.
# We send a POST with a JSON string that contains the binData key and the base64 encoded image as its value.
# If the POST was successful (HTTP STATUS OK 200) we read the data key from the JSON response and extract the tensor which is really our resultant image.
# The tensor has both a shape and values key — the values key is the image itself as an array of pixel intensities.
# We use Pillow to write out the tensor values as a JPEG file called ‘result.jpg’.

if __name__ == '__main__':
    url = sys.argv[1]
    path = sys.argv[2]
    # base64 encode image for HTTP POST
    data = {}
    with open(path, 'rb') as f:
        data['binData'] = base64.b64encode(f.read()).decode('utf-8')
    logger.info("sending image {} to {}".format(path, url))
    response = requests.post(url, json = data, timeout = None)

    logger.info("caught response {}".format(response))
    status_code = response.status_code
    js = response.json()
    if response.status_code == requests.codes['ok']:
        logger.info('converting tensor to image')
        data = js.get('data')
        tensor = data.get('tensor')
        shape = tensor.get('shape')
        values = tensor.get('values')
        logger.info("output image shape = {}".format(shape))
        # Convert Seldon tensor to image
        img_bytes = np.asarray(values)
        img = img_bytes.reshape(shape)
        Image.fromarray(img.astype(np.uint8)).save('result.jpg')
        logger.info('wrote result image to result.jpg')
    elif response.status_code == requests.codes['service_unavailable']:
        logger.error('Model service is not available.')
    elif response.status_code == requests.codes['internal_server_error']:
        logger.error('Internal model error.')
