"""
Train a scikit-learn model on UCI Wine Quality Dataset
https://archive.ics.uci.edu/ml/datasets/wine+quality
"""

import logging
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn import preprocessing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def prepare_dataset(test_size=0.05, random_seed=1):
    dataset_1 = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        delimiter=";",
    )
    dataset_2 = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        delimiter=";",
    )

    dataset = pd.concat([dataset_1, dataset_2], ignore_index=True)
    dataset = dataset.rename(columns=lambda x: x.lower().replace(" ", "_"))

    train_df, test_df = train_test_split(dataset, test_size=test_size, random_state=random_seed)
    return {"train": train_df, "test": test_df}


def train():
    logger.info("Preparing dataset...")
    dataset = prepare_dataset()
    train_df = dataset["train"]
    test_df = dataset["test"]

    y_train = train_df["quality"]
    X_train = train_df.drop("quality", axis=1)
    y_test = test_df["quality"]
    X_test = test_df.drop("quality", axis=1)

    logger.info("Training model...")
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = HistGradientBoostingRegressor(max_iter=50, max_depth=5).fit(X_train, y_train)

    ## Regression so output is values
    y_pred = model.predict(X_test)

    mse_value = mean_squared_error(y_test, y_pred)
    logger.info(f"Test MSE: {mse_value}")

    logger.info("Saving artifacts...")
    Path("app/artifacts").mkdir(exist_ok=True)
    dump(model, "app/artifacts/model.joblib")
    dump(scaler, "app/artifacts/scaler.joblib")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()