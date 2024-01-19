from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import model_trainer
from steps.evaluation import evaluate


@pipeline(enable_cache=True)
def train_pipeline(data_path: str):

    df = ingest_data(data_path)
    x_train, y_train, x_test, y_test = clean_data(df)
    model = model_trainer(x_train, y_train)
    evaluate(x_test, y_test, model)
