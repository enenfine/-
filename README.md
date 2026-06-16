# Tianchi Used Car Price Prediction

This project follows the official code-submission directory format for the Tianchi used car price prediction task.

## Directory Structure

```text
project
|-- README.md
|-- data
|   |-- used_car_sample_submit.csv
|   |-- used_car_train_20200313.csv
|   |-- used_car_testA_20200313.csv
|-- user_data
|-- feature
|   |-- generation.py
|-- model
|   |-- model.py
|-- prediction_result
|   |-- predictions.csv
|-- code
|   |-- main.py
|   |-- requirements.txt
```

The competition data files under `data/` are used for local reproduction. According to the competition rule, these original data files do not need to be included when submitting the final zip package.

## Solution

The task is a regression problem. The model predicts used car transaction price from vehicle attributes such as brand, model, registration date, power, mileage, repair status, region code, and anonymous features `v_0` to `v_14`.

The solution combines common public baseline ideas:

1. Feature engineering inspired by Datawhale-style Tianchi baselines.
2. LightGBM regression model with K-fold validation.

Main processing logic:

1. Read train and A-test data from `../data`.
2. Replace the special missing value `notRepairedDamage = "-"` with missing value.
3. Clip abnormal `power` values.
4. Build date features from `regDate` and `creatDate`.
5. Build car age, region, power-per-kilometer, frequency, and target-statistic features.
6. One-hot encode low-cardinality categorical features.
7. Train LightGBM on `log1p(price)`.
8. Restore predictions with `expm1`.
9. Save the final file to `../prediction_result/predictions.csv`.

## Run

Install dependencies listed in `code/requirements.txt`.

Program entry:

```bash
cd code
python main.py
```

The program reads:

```text
../data/used_car_train_20200313.csv
../data/used_car_testA_20200313.csv
```

and writes:

```text
../prediction_result/predictions.csv
```

Intermediate files are written to:

```text
../user_data/
```

## Output Format

`prediction_result/predictions.csv` follows the competition format:

```csv
SaleID,price
150000,38123.45
150001,9284.21
```
