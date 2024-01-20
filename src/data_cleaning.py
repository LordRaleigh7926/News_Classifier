import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from pandas.core.api import Series as Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import spacy


def preprocess(text, nlp):
    doc = nlp(text)
    li = []

    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        li.append(token.lemma_)

    return " ".join(li)


class DataStrategy(ABC):

    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DataPreprocessStrategy(DataStrategy):

    """
    Strategy for preprocessing data
    """

    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            law = {"SCIENCE": 0, "BUSINESS": 1, "CRIME": 2, "SPORTS": 3}
            for i, k in enumerate(df.category):
                df.category.values[i] = law.get(k)
            nlp = spacy.load("en_core_web_lg")
            df["processed_text"] = df.text.apply(preprocess, nlp=nlp)
            df["vector"] = df["processed_text"].apply(lambda x: nlp(x).vector)

        except Exception as e:
            logging.error(f"Error in Preprocessing Data: {e}")
            raise e


class DataDivideStrategy(DataStrategy):

    """
    Strategy for dividing data
    """

    def handle_data(self, df: pd.DataFrame) -> np.ndarray:
        
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                df.vector.values,
                df.category.values,
                test_size=0.2,
                random_state=2023,
                stratify=df.category.values,
            )

            x_test = np.stack(x_test)
            x_train = np.stack(x_train)
            y_train = np.stack(y_train)
            y_test = np.stack(y_test)

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            smt = SMOTE(random_state=70)
            x_train, y_train = smt.fit_resample(x_train, y_train)

            return x_train, y_train, x_test, y_test
        
        except Exception as e:
            logging.error(f"Error in Dividing, Scaling, OverSampling Data: {e}")
            raise e
    

class DataCleaning():

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:

        self.df = data
        self.strategy = strategy

    def process_data(self) -> Union[pd.DataFrame, np.ndarray]:

        try:
            return self.strategy.handle_data(self.df)
        
        except Exception as e:
            logging.error(f"Error in handling Data: {e}")
            raise e
        
    


        

