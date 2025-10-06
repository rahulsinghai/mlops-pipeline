#!/usr/bin/env python3

import pickle
import sklearn
import logging

logger = logging.getLogger('__housepricepredictionmodel__')

class HousePricePredictionModel(object):

    def __init__(self):
        logger.info("Initialising")
        logger.info("Loading model...")
        self._model = None
        with open('model.pkl', "rb") as f:
            self._model = pickle.load(f)

        logger.info("Model has been loaded and initialized...")

    def predict(self, X, features_names):
        """ Seldon Core Prediction API """

        logger.info("predict called...")
        prediction = None
        if self._model:
            logger.info("perform inference here...")
            prediction = self._model.predict(X)

        logger.info("returning prediction...")
        return prediction

    def metrics(self):
        return [
            {"type": "COUNTER", "key": "mycounter", "value": 1}, # a counter which will increase by the given value
            {"type": "GAUGE", "key": "mygauge", "value": 100},   # a gauge which will be set to given value
            {"type": "TIMER", "key": "mytimer", "value": 20.2},  # a timer which will add sum and count metrics - assumed millisecs
        ]