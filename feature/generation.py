from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_tianchi_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=" ")


def parse_date_column(series: pd.Series) -> pd.Series:
    text = series.astype("Int64").astype(str).str.zfill(8)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def add_date_features(df: pd.DataFrame) -> None:
    reg_date = parse_date_column(df["regDate"])
    creat_date = parse_date_column(df["creatDate"])

    df["reg_year"] = (df["regDate"] // 10000).astype(float)
    df["reg_month"] = ((df["regDate"] // 100) % 100).astype(float)
    df["reg_day"] = (df["regDate"] % 100).astype(float)
    df["creat_year"] = (df["creatDate"] // 10000).astype(float)
    df["creat_month"] = ((df["creatDate"] // 100) % 100).astype(float)
    df["creat_day"] = (df["creatDate"] % 100).astype(float)
    df["car_age_days"] = (creat_date - reg_date).dt.days
    df["car_age_years"] = df["car_age_days"] / 365.25


def add_frequency_features(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> None:
    all_data = pd.concat([train[cols], test[cols]], axis=0, ignore_index=True)
    for col in cols:
        counts = all_data[col].value_counts(dropna=False)
        train[f"{col}_count"] = train[col].map(counts).astype(float)
        test[f"{col}_count"] = test[col].map(counts).astype(float)


def add_target_stat_features(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        stats = (
            train.groupby(col)["price"]
            .agg(["mean", "median", "std", "min", "max"])
            .rename(columns=lambda name: f"{col}_price_{name}")
        )
        train_stats = train[[col]].merge(stats, left_on=col, right_index=True, how="left")
        test_stats = test[[col]].merge(stats, left_on=col, right_index=True, how="left")
        for feature in stats.columns:
            train[feature] = train_stats[feature].values
            test[feature] = test_stats[feature].values


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = train.copy()
    test = test.copy()
    test_sale_id = test["SaleID"].copy()

    for df in (train, test):
        df["notRepairedDamage"] = df["notRepairedDamage"].replace("-", np.nan).astype(float)
        df["power"] = df["power"].clip(lower=0, upper=600)
        df["regionCode"] = df["regionCode"].astype(float)
        df["region_province"] = (df["regionCode"] // 1000).astype(float)
        df["brand_model"] = df["brand"].astype(str) + "_" + df["model"].astype(str)
        df["power_per_km"] = df["power"] / (df["kilometer"] + 1.0)
        add_date_features(df)

    add_frequency_features(train, test, ["name", "model", "brand", "brand_model", "regionCode"])
    add_target_stat_features(train, test, ["brand", "model", "bodyType", "fuelType", "gearbox"])

    train["is_train"] = 1
    test["is_train"] = 0
    all_data = pd.concat([train.drop(columns=["price"]), test], axis=0, ignore_index=True)

    one_hot_cols = ["brand", "bodyType", "fuelType", "gearbox", "notRepairedDamage", "region_province"]
    all_data = pd.get_dummies(all_data, columns=one_hot_cols, dummy_na=True)

    drop_cols = ["SaleID", "regDate", "creatDate", "brand_model"]
    all_data = all_data.drop(columns=[col for col in drop_cols if col in all_data.columns])
    all_data = all_data.replace([np.inf, -np.inf], np.nan)

    x_train = all_data[all_data["is_train"] == 1].drop(columns=["is_train"])
    x_test = all_data[all_data["is_train"] == 0].drop(columns=["is_train"])

    medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(medians).fillna(0)
    x_test = x_test.fillna(medians).fillna(0)
    y = np.log1p(train["price"])

    return x_train, y, x_test, test_sale_id
