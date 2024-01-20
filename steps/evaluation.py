import logging
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
from zenml import step
from zenml.client import Client

import numpy as np
from sklearn.base import ClassifierMixin
from src.evaluate_model import MSE, R2, RMSE

import mlflow 



experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate(x_test: np.ndarray, y_test: np.ndarray, model: ClassifierMixin) -> Tuple[Annotated[float, "r2"], Annotated[float, 'mse'],Annotated[float, 'rmse']]:

    """Evaluates our model
    """
    try:
        pred = model.predict(x_test)

        MSE_Class = MSE()
        mse = MSE_Class.calculate_score(y_test,pred)
        mlflow.log_metric("mse", mse)

        R2_Class = R2()
        r2 = R2_Class.calculate_score(y_test,pred)
        mlflow.log_metric("r2", r2)


        RMSE_Class = RMSE()
        rmse = RMSE_Class.calculate_score(y_test,pred)
        mlflow.log_metric("rmse", rmse)


        return r2, mse, rmse
    
    except Exception as e:
        logging.error(f"Error in Evaluating Data: {e}")
        raise e




    