import logging
import pandas as pd
from zenml import step
import numpy as np
from typing_extensions import Annotated
from typing import Tuple
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy

@step
def clean_data(df: pd.DataFrame) -> (
    Tuple[
        Annotated[np.ndarray, "x_train"],
        Annotated[np.ndarray, "y_train"],
        Annotated[np.ndarray, "x_test"],
        Annotated[np.ndarray, "y_test"],
    ]
):
    
    """Cleans data and returns test datasets and train datasets

    Raises:
        e: Raises error if there is any problem in cleanin

    Returns:
        np.ndarray: x_train
        np.ndarray: y_train 
        np.ndarray: x_test 
        np.ndarray: y_test 

    """
    
    try:
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.process_data()

        divide_strategy = DataDivideStrategy()
        division_of_data = DataCleaning(processed_data, divide_strategy)
        x_train, y_train, x_test, y_test = division_of_data.process_data()
        logging.info("Data Cleaned")
        return x_train, y_train, x_test, y_test
    
    except Exception as e:
        logging.error(f"Error in Cleaning Data: {e}")
        raise e
