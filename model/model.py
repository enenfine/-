from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

import lightgbm as lgb


def make_model(n_estimators: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.3,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )


def train_and_predict(
    x: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    folds: int = 5,
    n_estimators: int = 1600,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    test_pred = np.zeros(len(x_test), dtype=float)
    feature_importance = pd.DataFrame({"feature": x.columns, "importance": 0.0})
    valid_maes: list[float] = []

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(x), start=1):
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        regressor = make_model(n_estimators)
        regressor.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )

        valid_pred = np.expm1(regressor.predict(x_valid, num_iteration=regressor.best_iteration_))
        valid_true = np.expm1(y_valid)
        fold_mae = mean_absolute_error(valid_true, valid_pred)
        valid_maes.append(fold_mae)
        print(f"fold {fold} MAE: {fold_mae:.4f}")

        test_pred += np.expm1(regressor.predict(x_test, num_iteration=regressor.best_iteration_)) / folds
        feature_importance["importance"] += regressor.feature_importances_ / folds

    mean_mae = float(np.mean(valid_maes))
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    return np.clip(test_pred, 1, None), feature_importance, mean_mae
