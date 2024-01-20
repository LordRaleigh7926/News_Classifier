import logging 
from abc import ABC, abstractmethod

from sklearn.svm import SVC
import numpy as np

class Model(ABC):

    """Abstract class for all models
    """

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:

        """ Trains the Model

        Args:
            x_train: Training Data
            y_train: Training Labels

        Returns:
            None

        """

        pass

class SupportVectorMachine(Model):

    """ SVC Model
    """

    def train(self, x_train, y_train, **kwargs):

        """ Trains the Model

        Args:
            x_train: Training Data
            y_train: Training Labels

        Returns:
            None

        """

        try:
            svc_model = SVC(**kwargs)
            svc_model.fit(x_train, y_train)
            logging.info("Model Training Complete")
            return svc_model
        
        except Exception as e:
            logging.error(f"Error while training model - {e}")
            raise e
        