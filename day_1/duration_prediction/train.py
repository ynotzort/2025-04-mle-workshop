#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


df_train = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-01.parquet')
df_val = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-02.parquet')

print("the lengths of the dfs: ", len(df_train), len(df_val))

categorical = ['PULocationID', 'DOLocationID']
numerical = ['trip_distance']

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
val_dicts = df_val[categorical + numerical].to_dict(orient='records')

target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values

pipeline = make_pipeline(DictVectorizer(), LinearRegression())
pipeline.fit(train_dicts, y_train)
y_pred = pipeline.predict(val_dicts)

print(f"MSE: {mean_squared_error(y_val, y_pred, squared=False)}")

# sns.histplot(y_pred, kde=True, stat="density", color='blue', bins=25, label='prediction')
# sns.histplot(y_val, kde=True, stat="density", color='orange', bins=40, label='actual')
# plt.legend()

with open('lin_reg.bin', 'wb') as f_out:
    pickle.dump(pipeline, f_out)

