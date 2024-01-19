import logging
import pandas as pd
import numpy as np
from zenml import step

from sklearn.base import ClassifierMixin
from sklearn.svm import SVC

@step
def model_trainer(x_train: np.ndarray, y_train: np.ndarray) -> ClassifierMixin:
    model = SVC()
    model.fit(x_train, y_train)
    return model