"""Compare the pricing candidates in MLflow and register the winner as `challenger`.

Five steps:
  1. point the client at the tracking server and pick the experiment
  2. evaluate every candidate -- four models, the gradient boosting over a grid of 36
     combinations -- one MLflow run each, all on the same split and the same folds
  3. pick the winner on cross-validated MAE, measured on the training split only
  4. refit the winner on every car, log it in its own run and register it
  5. point the `challenger` alias at the version just created

This script never touches `champion`, the alias the API serves. Moving it is a separate and
deliberate act, done by promote.py after a review of the runs.

The winner is chosen on cross-validation, not on the test set: picking the best of 40 runs on
the test MAE would make that MAE optimistic, and promote.py needs it honest to compare a
challenger with the champion.

Artifacts go straight to R2: the tracking server runs with --no-serve-artifacts, so this
client uploads them itself (hence boto3 in requirements.txt).
"""

import itertools
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

load_dotenv()

EXPERIMENT_NAME = "getaround-pricing"
ALIAS = "challenger"
RANDOM_STATE = 42

DATA = Path(__file__).resolve().parent.parent / "data" / "get_around_pricing_project.csv"

NUMERIC = ["mileage", "engine_power"]
CATEGORICAL = ["model_key", "fuel", "paint_color", "car_type"]
BOOLEAN = ["private_parking_available", "has_gps", "has_air_conditioning", "automatic_car",
           "has_getaround_connect", "has_speed_regulator", "winter_tires"]
FEATURES = NUMERIC + CATEGORICAL + BOOLEAN
TARGET = "rental_price_per_day"

# The gradient boosting grid of the notebook, section 6.2: 2 x 2 x 3 x 3 = 36 combinations.
GRID = {
    "learning_rate": [0.05, 0.1],
    "max_iter": [200, 400],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 40],
}


def candidates():
    """Every model to evaluate, as (run name, estimator, parameters to log).

    The median price is not a model anyone would serve: it is the bar every other run has to
    clear, logged as a run so the comparison in MLflow shows it.
    """
    yield "median price (baseline)", DummyRegressor(strategy="median"), {"strategy": "median"}
    yield "linear regression", LinearRegression(), {}
    yield ("random forest",
           RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
           {"n_estimators": 300})
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        name = "gradient boosting " + " ".join(f"{k}={v}" for k, v in params.items())
        yield name, HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params), params


def build_pipeline(estimator):
    """Scaler + one-hot encoder + regressor, as one object.

    The whole pipeline is what gets logged, so the API owns no feature engineering and only
    ever calls .predict(...). handle_unknown matters for a reason that is not statistical:
    the endpoint is public and will be sent a model_key that was never in the training set.
    An unknown level has to be encoded, not raise.
    """
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                              sparse_output=False), CATEGORICAL),
        ("bool", "passthrough", BOOLEAN),
    ])
    return Pipeline([("pre", preprocessor), ("model", estimator)])


def main():
    start = time.time()

    registered_model_name = os.environ["MLFLOW_REGISTERED_MODEL_NAME"]

    # --- STEP 1 --- the tracking server and the experiment
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Needed for the last step only: aliases act on registered models, which live outside the
    # notion of an active run, so the fluent API has no equivalent.
    client = MlflowClient()

    pricing = pd.read_csv(DATA).drop(columns=["Unnamed: 0"])

    # Three physically impossible rows -- a mileage of -64 km, one of 1 000 376 km, and a
    # zero-power engine. 0.06% of the file, too small to change a score; they go because
    # serving a prediction fitted on a negative mileage is indefensible.
    impossible = ((pricing["mileage"] < 0) | (pricing["mileage"] > 500_000)
                  | (pricing["engine_power"] == 0))
    clean = pricing[~impossible].reset_index(drop=True)

    X, y = clean[FEATURES], clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    folds = KFold(5, shuffle=True, random_state=RANDOM_STATE)

    # --- STEP 2 --- one run per candidate, same split, same folds
    results = []
    for name, estimator, params in candidates():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({**params, "estimator": type(estimator).__name__,
                               "features": len(FEATURES),
                               "rows_dropped": int(impossible.sum()),
                               "random_state": RANDOM_STATE})

            cv = cross_validate(build_pipeline(estimator), X_train, y_train, cv=folds,
                                scoring={"mae": "neg_mean_absolute_error", "r2": "r2"})
            predictions = build_pipeline(estimator).fit(X_train, y_train).predict(X_test)
            metrics = {
                "cv_mae": float(-cv["test_mae"].mean()),
                "cv_r2": float(cv["test_r2"].mean()),
                "cv_r2_std": float(cv["test_r2"].std()),
                "test_mae": float(mean_absolute_error(y_test, predictions)),
                "test_r2": float(r2_score(y_test, predictions)),
                "test_median_abs_error": float(np.median(np.abs(y_test - predictions))),
            }
            mlflow.log_metrics(metrics)
            results.append({"name": name, "run_id": run.info.run_id,
                            "estimator": estimator, **metrics})
            print(f"[INFO] {name:<75} cv_mae {metrics['cv_mae']:6.2f}"
                  f"  test_mae {metrics['test_mae']:6.2f}")

    # --- STEP 3 --- the winner, on cross-validated MAE
    winner = min(results, key=lambda r: r["cv_mae"])
    print(f"\n[INFO] Winner: {winner['name']} -- cv_mae {winner['cv_mae']:.2f}, "
          f"test_mae {winner['test_mae']:.2f}")

    # --- STEP 4 --- refit on every car, log the model in the winner's own run
    # The split existed to choose the model. Once chosen, there is no reason to serve a
    # version trained on 80% of the data. The run keeps the metrics measured on the split.
    final = build_pipeline(winner["estimator"]).fit(X, y)
    with mlflow.start_run(run_id=winner["run_id"]):
        mlflow.log_metric("training_rows", len(X))

        # The signature is inferred rather than written by hand: the input is a fixed frame of
        # 13 named columns, and freezing it is what makes the column ORDER a contract the API
        # cannot get wrong.
        input_example = X.head(2)
        signature = infer_signature(input_example, final.predict(input_example))

        model_info = mlflow.sklearn.log_model(
            sk_model=final,
            name="model",
            signature=signature,
            input_example=input_example,
            # What is needed to LOAD the model, not to train it. scikit-learn is pinned to
            # the version that fitted it: unpickling under a different minor version works
            # and only warns, which is precisely the risk.
            pip_requirements=[
                "scikit-learn==1.9.0",
                "pandas==2.3.3",
                "numpy==2.5.2",
            ],
            registered_model_name=registered_model_name,
        )

    # --- STEP 5 --- the new version becomes the challenger, nothing more
    version = model_info.registered_model_version
    client.set_registered_model_alias(
        name=registered_model_name, alias=ALIAS, version=version)
    print(f"[INFO] Registered as version {version} of '{registered_model_name}'")
    print(f"[INFO] Alias '{ALIAS}' now points to version {version}")
    print("[INFO] Review the runs, then run training/promote.py to make it the champion")
    print(f"--- Total time: {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
