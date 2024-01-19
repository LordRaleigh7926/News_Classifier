import logging
import pandas as pd
from zenml import step

import numpy as np
from sklearn.base import ClassifierMixin

@step
def evaluate(x_test: np.ndarray, y_test: np.ndarray, model: ClassifierMixin) -> float:
    accuracy = model.score(x_test, y_test)
    print("Test Accuracy is - ", accuracy)
    return accuracy