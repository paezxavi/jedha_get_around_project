"""FastAPI pricing service for Getaround car owners.

The model is pulled from the MLflow registry at startup and kept in memory for the lifetime
of the process. This service owns no model code and no feature engineering: the whole
scikit-learn pipeline -- scaler, one-hot encoder, regressor -- travels inside the logged
model, so all we ever call is `.predict(...)`.

That is the point of the split. A served model that reproduces its own preprocessing is a
served model that quietly stops matching its notebook the first time an encoder changes. And
because the model is resolved by ALIAS rather than by version, promoting a better one is an
alias move in the registry, not a redeployment of this image.
"""

import os
from contextlib import asynccontextmanager
from string import Template

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# Local convenience only. On the Space there is no .env: Hugging Face injects the secrets
# straight into the environment, so nothing below may depend on this file existing.
load_dotenv()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
REGISTERED_MODEL_NAME = os.environ["MLFLOW_REGISTERED_MODEL_NAME"]
MODEL_ALIAS = os.environ["MLFLOW_MODEL_ALIAS"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
MODEL = None

# The column order and the column types are the endpoint's contract, and both come from the
# model's own logged signature rather than from a literal here -- see the lifespan below.
FEATURES = None
DTYPES = None
# The champion's held-out MAE, read from its run: /health hands it to the dashboard.
TEST_MAE = None

# Two real rows of the training file, used as the example in /docs and in the README so a
# visitor can copy one and get an answer without inventing thirteen values.
EXAMPLE = [
    [140411, 100, "Citroën", "diesel", "black", "convertible",
     True, True, False, False, True, True, True],
    [13929, 317, "Citroën", "petrol", "grey", "convertible",
     True, True, False, False, False, True, True],
]

# Rendered as the page body of /docs, which is where a visitor lands. It opens on a markdown
# h1: Swagger UI puts `info.title` in an h2, and the brief asks the documentation page for an
# h1 title. The limitation belongs here rather than in a footnote -- three quarters of what
# this model knows comes from two columns an owner cannot change, so it answers "what is my
# car worth" and not "what should I do about it", and a pricing endpoint read as the second
# is a misleading one. The $-placeholders are the champion's own metrics, read from its MLflow
# run at startup, so the page describes the model actually served and holds no figure of its own.
DESCRIPTION = Template("""
# Getaround Pricing API

Suggests a **daily rental price in euros** for a car, learned from the $rows cars in
Getaround's pricing dataset.

On a held-out fifth of that data the model is wrong by **€$mae on average**, and by **€$median
for half the cars**. Charging every car the median price is wrong by €$baseline.

**It predicts what owners *do* charge for a car like this one.** Most of what it knows comes
from `engine_power` and `mileage`; the equipment options weigh little. Read it as a market
rate for a given car, not as a list of things to change.

---

## `POST /predict`

**Input** — a JSON object with a single key, `input`, holding a list of cars. Each car is a
list of **13 values, in this exact order**:

| # | field | type | example |
|---|---|---|---|
| 0 | `mileage` | integer, km | `140411` |
| 1 | `engine_power` | integer, hp | `100` |
| 2 | `model_key` | string | `"Citroën"` |
| 3 | `fuel` | string | `"diesel"` |
| 4 | `paint_color` | string | `"black"` |
| 5 | `car_type` | string | `"convertible"` |
| 6 | `private_parking_available` | boolean | `true` |
| 7 | `has_gps` | boolean | `true` |
| 8 | `has_air_conditioning` | boolean | `false` |
| 9 | `automatic_car` | boolean | `false` |
| 10 | `has_getaround_connect` | boolean | `true` |
| 11 | `has_speed_regulator` | boolean | `true` |
| 12 | `winter_tires` | boolean | `true` |

**Output** — a JSON object with a single key, `prediction`, holding one price per car sent,
in the same order.

**An unknown category is accepted, not rejected.** A `model_key`, `fuel`, `paint_color` or
`car_type` the model never saw is encoded as "infrequent" rather than raising, so the
endpoint answers for any car — including one whose brand is not in the training set.

### curl

```bash
curl -i -H "Content-Type: application/json" -X POST \\
  -d '{"input": [[140411, 100, "Citroën", "diesel", "black", "convertible",
                  true, true, false, false, true, true, true]]}' \\
  https://lambla-getaround-pricing-api.hf.space/predict
```

### python

```python
import requests

response = requests.post(
    "https://lambla-getaround-pricing-api.hf.space/predict",
    json={"input": [[140411, 100, "Citroën", "diesel", "black", "convertible",
                     True, True, False, False, True, True, True]]},
)
print(response.json())      # {"prediction": [107.91]}
```

**Errors** — a car that does not hold 13 values, or a value that cannot be read as its
column's type (a word where `mileage` expects a number), is answered with a **422** and a
message naming the problem, never with a 500.

---

## `GET /health`

Liveness, plus which model URI was loaded, whether it is in memory, the 13 columns in
order and the model's held-out MAE in euros.

## `GET /`

Redirects here, to `/docs`.
""")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything before the yield runs once, before the server accepts any request. Blocking
    # here is the intended behaviour: nothing else is running yet, and an API that starts
    # without a model would only answer opaque 500s. No try/except on purpose -- a failure
    # must kill the process with its traceback rather than serve a broken endpoint.
    global MODEL, FEATURES, DTYPES, TEST_MAE

    print(f"Loading {MODEL_URI} ...")
    MODEL = mlflow.pyfunc.load_model(MODEL_URI)

    # The column order comes from the signature the training script inferred and logged, so
    # this file holds no list of column names that could drift away from the model's.
    schema = MODEL.metadata.get_input_schema()
    FEATURES = schema.input_names()
    DTYPES = dict(zip(FEATURES, schema.pandas_types()))
    print(f"Model loaded: {len(FEATURES)} features.")

    # /docs is built on its first request, after this point, so it shows these figures. The
    # baseline is the median-price run of the same training session as the champion.
    client = MlflowClient()
    run = client.get_run(MODEL.metadata.run_id)
    TEST_MAE = run.data.metrics["test_mae"]
    baseline = client.search_runs(
        [run.info.experiment_id],
        filter_string="attributes.run_name = 'median price (baseline)'",
        order_by=["attributes.start_time DESC"], max_results=1)
    app.description = DESCRIPTION.substitute(
        rows=f"{run.data.metrics['training_rows']:,.0f}".replace(",", " "),
        mae=f"{run.data.metrics['test_mae']:.2f}",
        median=f"{run.data.metrics['test_median_abs_error']:.2f}",
        baseline=f"{baseline[0].data.metrics['test_mae']:.2f}" if baseline else "n/a",
    )

    yield

    # Nothing to release on shutdown: no connection, no file handle, no temporary directory.


app = FastAPI(
    title="Getaround Pricing API",
    version="1.0.0",
    lifespan=lifespan,
    # ReDoc renders the same schema a second time at a second URL. The brief asks for one
    # documentation page, at /docs, so the duplicate is turned off.
    redoc_url=None,
)


class PredictionInput(BaseModel):
    # list[list] rather than list[list[float]]: a car carries four strings and seven booleans
    # alongside its two numbers, and a typed inner list would reject "Citroën" outright.
    input: list[list] = Field(..., examples=[EXAMPLE])


class PredictionOutput(BaseModel):
    prediction: list[float] = Field(..., examples=[[107.91, 264.38]])


# Plumbing, not an endpoint: HF serves the Space at "/", and without this a visitor lands on
# a 404. include_in_schema keeps it out of /docs, which stays a description of real endpoints.
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["service"])
def health():
    """Liveness probe. On a remote Space this is the only window into whether the startup
    load actually succeeded."""
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "model_loaded": MODEL is not None,
        "features": FEATURES,
        "test_mae": TEST_MAE,
    }


@app.post("/predict", response_model=PredictionOutput, tags=["pricing"])
def predict(payload: PredictionInput):
    """Suggest a daily rental price, in euros, for each car sent."""
    # The DataFrame is what makes the column ORDER a contract rather than a convention: the
    # pipeline's ColumnTransformer selects by name, so a caller who sends the thirteen values
    # in a different order would get a wrong price rather than an error. The names come from
    # the logged signature, never from a literal in this file.
    # The brief assumes well-formed input and leaves error handling as a bonus. Without these
    # two checks a malformed car reaches pandas and comes back as an opaque 500.
    for i, car in enumerate(payload.input):
        if len(car) != len(FEATURES):
            raise HTTPException(422, f"car {i} has {len(car)} values, expected "
                                     f"{len(FEATURES)} in this order: {', '.join(FEATURES)}")
    cars = pd.DataFrame(payload.input, columns=FEATURES)

    # The signature records mileage and engine_power as integers and the seven options as
    # booleans. A JSON list arrives as Python objects, so the frame comes out as `object`
    # dtype and MLflow's schema enforcement rejects it before the pipeline ever sees it.
    # Casting to the logged schema is the caller's side of the contract, done here once.
    try:
        cars = cars.astype(DTYPES)
    except (ValueError, TypeError) as error:
        raise HTTPException(422, f"a value does not match its column's type: {error}")

    return {"prediction": [round(float(price), 2) for price in MODEL.predict(cars)]}


if __name__ == "__main__":
    # Port 7860 is the Hugging Face Spaces standard. Read from the environment so the same
    # file runs locally on any port.
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
