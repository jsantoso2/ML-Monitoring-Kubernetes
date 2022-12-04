from locust import HttpUser, task
import pandas as pd

## Map feature column names
feature_columns = {
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "chlorides": "chlorides",
    "free sulfur dioxide": "free_sulfur_dioxide",
    "total sulfur dioxide": "total_sulfur_dioxide",
    "density": "density",
    "pH": "ph",
    "sulphates": "sulphates",
    "alcohol": "alcohol_pct_vol",
}

## Read from dataset
dataset = (
    pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        delimiter=";",
    )
    .rename(columns=feature_columns)
    .drop("quality", axis=1)
)


## Define locust class 
class WinePredictionUser(HttpUser):
    @task(3)
    def test(self):
        self.client.get("/test")

    @task(92)
    def prediction(self):
        record = dataset.sample(n=1).copy()
        record = record.values.tolist()
        input_for_pred = {"data": record} 
        self.client.post("/predict", json=input_for_pred)

    @task(5)
    def prediction_bad_value(self):
        input_for_pred = {"data": "bad data"} 
        self.client.post("/predict", json=input_for_pred)