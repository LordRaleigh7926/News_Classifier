import logging
import pandas as pd
import numpy as np
from zenml import step
from zenml.client import Client

from src.model_dev import SupportVectorMachine
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig

import mlflow 

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def model_trainer(x_train: np.ndarray, y_train: np.ndarray, config: ModelNameConfig) -> ClassifierMixin:

    """Trains the model on ingested data

    Args:
        x_train: training features
        y_train: training labels

    """

    if config.model_name == "SVM":
        mlflow.sklearn.autolog()
        model = SupportVectorMachine()
        trained_model = model.train(x_train, y_train)
        return trained_model
    else:
        logging.info("Model Name not listed")
        raise ValueError(f"Model {config.model_name} not listed")


