import logging 
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):

    """Abstract class for defining the strategy for Evaluation
    """

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):

        """Calculating the score of the model

        Args:
            y_true: The actual values
            y_pred: The predicted values

        Returns: 
            float: The score

        """

        pass


class MSE(Evaluation):

    """
    Evaluation Strategy that uses Mean Squared Error
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):

        """Calculating the score of the model

        Args:
            y_true: The actual values
            y_pred: The predicted values

        Returns: 
            float: The score

        """

        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE is {mse}")
            return mse

        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e

class R2(Evaluation):

    """
    Evaluation Strategy that uses R2
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):

        """Calculating the score of the model

        Args:
            y_true: The actual values
            y_pred: The predicted values

        Returns:
            float: The score

        """

        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 is {r2}")
            return r2

        except Exception as e:
            logging.error(f"Error in calculating R2 Score: {e}")
            raise e
        

class RMSE(Evaluation):

    """
    Evaluation Strategy that uses RMSE
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):

        """Calculating the score of the model

        Args:
            y_true: The actual values
            y_pred: The predicted values

        Returns: 
            float: The score

        """

        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"RMSE is {rmse}")
            return rmse

        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e

