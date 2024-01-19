import logging
import pandas as pd
from zenml import step
import spacy
import numpy as np
from typing_extensions import Annotated
from typing import Tuple

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def preprocess(text,nlp):

    

    doc = nlp(text)

    li = []

    for token in doc:
        if token.is_stop or token.is_punct:
            continue

        li.append(token.lemma_)

    return " ".join(li)

@step
def clean_data(df: pd.DataFrame) -> (
    Tuple[
        Annotated[np.ndarray, "x_train"],
        Annotated[np.ndarray, "y_train"],
        Annotated[np.ndarray, "x_test"],
        Annotated[np.ndarray, "y_test"],
    ]
):

    law = {"SCIENCE": 0, "BUSINESS": 1, "CRIME": 2, "SPORTS": 3}
    for i, k in enumerate(df.category):
        df.category.values[i] = law.get(k)
    nlp = spacy.load("en_core_web_lg")
    df["processed_text"] = df.text.apply(preprocess, nlp=nlp)
    df["vector"] = df["processed_text"].apply(lambda x: nlp(x).vector)
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
    x_test = scaler.fit_transform(x_test)

    smt = SMOTE(random_state=70)
    x_train, y_train = smt.fit_resample(x_train, y_train)

    return x_train, y_train, x_test, y_test