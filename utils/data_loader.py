import pandas as pd
import polars as pl
import numpy as np


# load data in datasets directory as pandas dataframe or polars dataframe
# choose dataset between california_housing.csv and iris.csv
def load_data(dataset_name: str) -> pd.DataFrame:
    if dataset_name == "california_housing":
        return pd.read_csv("datasets/california_housing.csv")
    elif dataset_name == "iris":
        return pd.read_csv("datasets/iris.csv")
    else:
        raise ValueError("Invalid dataset name")


def load_data_polars(dataset_name: str) -> pl.DataFrame:
    if dataset_name == "california_housing":
        return pl.read_csv("datasets/california_housing.csv")
    elif dataset_name == "iris":
        return pl.read_csv("datasets/iris.csv")
    else:
        raise ValueError("Invalid dataset name")
