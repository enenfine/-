from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from feature.generation import build_features, read_tianchi_csv
from model.model import train_and_predict


DATA_DIR = (CODE_DIR / "../data").resolve()
USER_DATA_DIR = (CODE_DIR / "../user_data").resolve()
PREDICTION_DIR = (CODE_DIR / "../prediction_result").resolve()

TEST_SET = os.environ.get("TEST_SET", "A").upper()
TRAIN_FILE = DATA_DIR / "used_car_train_20200313.csv"
TEST_FILE = DATA_DIR / ("used_car_testB_20200421.csv" if TEST_SET == "B" else "used_car_testA_20200313.csv")
PREDICTION_FILE = PREDICTION_DIR / os.environ.get("PREDICTION_FILE", "predictions.csv")


def main() -> None:
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"using test set: {TEST_SET}")
    print(f"train file: {TRAIN_FILE}")
    print(f"test file: {TEST_FILE}")

    train = read_tianchi_csv(TRAIN_FILE)
    test = read_tianchi_csv(TEST_FILE)
    x_train, y_train, x_test, sale_id = build_features(train, test)

    prediction, feature_importance, mean_mae = train_and_predict(
        x_train,
        y_train,
        x_test,
        folds=5,
        n_estimators=1600,
    )

    feature_importance.to_csv(USER_DATA_DIR / "feature_importance.csv", index=False)
    (USER_DATA_DIR / "metrics.txt").write_text(f"mean_cv_mae={mean_mae:.6f}\n", encoding="utf-8")

    submit = test[["SaleID"]].copy()
    submit["SaleID"] = sale_id.astype(int).values
    submit["price"] = prediction
    submit[["SaleID", "price"]].to_csv(PREDICTION_FILE, index=False)
    print(f"prediction saved to {PREDICTION_FILE}")


if __name__ == "__main__":
    main()
