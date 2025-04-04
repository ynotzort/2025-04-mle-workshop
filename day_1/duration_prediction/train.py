#!/usr/bin/env python
# coding: utf-8
from datetime import date
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

import logging

logger = logging.getLogger(__name__)


def read_dataframe(filename):
    
    logger.info(f"Reading df from {filename}")
    try:
        df = pd.read_parquet(filename)

        df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ["PULocationID", "DOLocationID"]
        df[categorical] = df[categorical].astype(str)
        
        logger.debug(f"DF shape: {df.shape}")

        return df
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        raise e


def train(train_date: date, val_date: date, out_path: str) -> float:
    """This function trains an ml model for price prediction of a taxi trip

    Args:
        train_date (date): the training month
        val_date (date): the validation month
        out_path (str): where to save the model
    """
    try:
        base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
        train_url = base_url.format(year=train_date.year, month=train_date.month)
        val_url = base_url.format(year=val_date.year, month=val_date.month)
        logger.debug(f"training data url: {train_url}")
        logger.debug(f"validation data url: {val_url}")

        df_train = read_dataframe(train_url)
        df_val = read_dataframe(val_url)

        logger.debug(f"the lengths of the dfs: {len(df_train)}, {len(df_val)}", )

        categorical = ["PULocationID", "DOLocationID"]
        numerical = ["trip_distance"]

        train_dicts = df_train[categorical + numerical].to_dict(orient="records")
        val_dicts = df_val[categorical + numerical].to_dict(orient="records")

        target = "duration"
        y_train = df_train[target].values
        y_val = df_val[target].values

        pipeline = make_pipeline(DictVectorizer(), LinearRegression())
        pipeline.fit(train_dicts, y_train)
        y_pred = pipeline.predict(val_dicts)

        mse = mean_squared_error(y_val, y_pred, squared=False)
        logger.info(f"MSE: {mse}")

        # sns.histplot(y_pred, kde=True, stat="density", color='blue', bins=25, label='prediction')
        # sns.histplot(y_val, kde=True, stat="density", color='orange', bins=40, label='actual')
        # plt.legend()

        with open(out_path, "wb") as f_out:
            pickle.dump(pipeline, f_out)
            
        return mse
    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise e
