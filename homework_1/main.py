#!/usr/bin/env python

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_data"


def make_duration(my_df: pd.DataFrame) -> pd.DataFrame:
    assert all(
        [i in my_df.columns for i in ["tpep_dropoff_datetime", "tpep_pickup_datetime"]]
    ), "Both tpep_dropoff_datetime and tpep_pickup_datetime columns need to be present"
    my_df["duration"] = (
        my_df["tpep_dropoff_datetime"] - my_df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    return my_df


def filter_duration(
    my_df: pd.DataFrame, lower_bound: float = 1, upper_bound: float = 60
) -> pd.DataFrame:
    assert "duration" in my_df.columns, "The duration column must already be present"
    my_df = my_df[
        (my_df["duration"] >= lower_bound) & (my_df["duration"] <= upper_bound)
    ].reset_index(drop=True)
    return my_df


def main():
    jan = pd.read_parquet(DATA_DIR / "yellow_tripdata_2022-01.parquet")

    print(f"Number of columns: {jan.shape[1]}")

    jan = make_duration(jan)

    print(f"Standard deviation: {jan['duration'].std()}")

    jan_filtered = filter_duration(jan)

    print(f"Fraction retain after drop {len(jan_filtered)/len(jan)}")

    d_vect = DictVectorizer()

    feats = (
        jan_filtered[["PULocationID", "DOLocationID"]].astype(str).to_dict("records")
    )

    feats = d_vect.fit_transform(feats)

    print(f"Dimensionality of matrix: {feats.shape[1]}")

    reg = LinearRegression()

    reg.fit(feats, jan_filtered["duration"])

    jan_rmse = mean_squared_error(
        jan_filtered["duration"], reg.predict(feats), squared=False
    )

    print(f"RMSE on Training Data: {jan_rmse}")

    feb = pd.read_parquet(DATA_DIR / "yellow_tripdata_2022-02.parquet")

    feb = make_duration(feb)

    feb_filtered = filter_duration(feb)

    feb_feats = (
        feb_filtered[["PULocationID", "DOLocationID"]].astype(str).to_dict("records")
    )

    feb_feats = d_vect.transform(feb_feats)

    feb_rmse = mean_squared_error(
        feb_filtered["duration"], reg.predict(feb_feats), squared=False
    )

    print(f"RMSE on Validation Data: {feb_rmse}")


if __name__ == "__main__":
    main()
