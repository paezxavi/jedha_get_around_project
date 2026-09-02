"""Fit the Getaround pricing model and register it in the shared MLflow registry.

Five steps, mirroring the reference training script:
  1. point the client at the tracking server and pick the experiment
  2. fit on the training split and measure on the held-out one
  3. refit on every car -- the split existed to choose the model, not to ship one
  4. log a servable sklearn model and register it
  5. move the alias to the version just created

The API resolves `models:/<name>@<alias>`, never a version number, so promoting a better model
is an alias move and needs no redeployment.

Artifacts go straight to R2: the tracking server runs with --no-serve-artifacts, so this
client uploads them itself (hence boto3 in requirements.txt).

The model and its hyper-parameters are not decided here. They come from section 6 of
getaround_analysis.ipynb, which compares five candidates and tunes the winner over a grid of
36 combinations; this script is the reproducible path from that decision to a served artefact.
"""

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

load_dotenv()

EXPERIMENT_NAME = "getaround-pricing"
RANDOM_STATE = 42

DATA = Path(__file__).resolve().parent.parent / "data" / "get_around_pricing_project.csv"

NUMERIC = ["mileage", "engine_power"]
CATEGORICAL = ["model_key", "fuel", "paint_color", "car_type"]
BOOLEAN = ["private_parking_available", "has_gps", "has_air_conditioning", "automatic_car",
           "has_getaround_connect", "has_speed_regulator", "winter_tires"]
FEATURES = NUMERIC + CATEGORICAL + BOOLEAN
TARGET = "rental_price_per_day"

# Chosen in section 6.2 of the notebook: gradient boosting won the five-model comparison on
# both the cross-validated and the held-out R², and this is the best of 36 grid combinations.
BEST_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 400,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 10,
}


def build_pipeline():
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
    return Pipeline([
        ("pre", preprocessor),
        ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE, **BEST_PARAMS)),
    ])


def main():
    start = time.time()

    registered_model_name = os.environ["MLFLOW_REGISTERED_MODEL_NAME"]
    alias = os.environ["MLFLOW_MODEL_ALIAS"]

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

    with mlflow.start_run(run_name="gradient boosting, tuned — registered") as run:
        mlflow.log_params({**BEST_PARAMS, "estimator": "HistGradientBoostingRegressor",
                           "features": len(FEATURES), "rows_dropped": int(impossible.sum()),
                           "random_state": RANDOM_STATE})

        # --- measured on data the model has not seen ---
        evaluation = build_pipeline().fit(X_train, y_train)
        predictions = evaluation.predict(X_test)
        cv = cross_val_score(build_pipeline(), X_train, y_train,
                             cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="r2")
        metrics = {
            "cv_r2": float(cv.mean()),
            "cv_r2_std": float(cv.std()),
            "test_r2": float(r2_score(y_test, predictions)),
            "test_mae": float(mean_absolute_error(y_test, predictions)),
            "test_median_abs_error": float(np.median(np.abs(y_test - predictions))),
        }
        mlflow.log_metrics(metrics)
        for name, value in metrics.items():
            print(f"[INFO] {name:>22}: {value:.4f}")

        # --- the model that ships is refitted on every car ---
        # The split existed to choose the model. Once chosen, there is no reason to serve a
        # version trained on 80% of the data.
        final = build_pipeline().fit(X, y)
        mlflow.log_metric("training_rows", len(X))

        # The signature is inferred rather than written by hand: unlike an image tensor, the
        # input here is a fixed frame of 13 named columns, and freezing it is the point --
        # it is what makes the column ORDER a contract the API cannot get wrong.
        input_example = X.head(2)
        signature = infer_signature(input_example, final.predict(input_example))

        model_info = mlflow.sklearn.log_model(
            sk_model=final,
            name="model",
            signature=signature,
            input_example=input_example,
            # What is needed to LOAD the model, not to train it. scikit-learn is pinned to
            # the version that fitted it: unpickling under a different minor version works
            # and only warns, which is precisely the risk -- a silent change of behaviour in
            # a served model is the failure nobody notices.
            pip_requirements=[
                "scikit-learn==1.9.0",
                "pandas==2.3.3",
                "numpy==2.5.2",
            ],
            registered_model_name=registered_model_name,
        )

        version = model_info.registered_model_version
        print(f"[INFO] Model logged as version {version} of '{registered_model_name}'")

        client.set_registered_model_alias(
            name=registered_model_name, alias=alias, version=version)
        print(f"[INFO] Alias '{alias}' now points to version {version}")
        print(f"[INFO] Run: {run.info.run_id}")
        print(f"--- Total time: {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
