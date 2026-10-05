---
title: getaround-pricing-api
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Getaround — Pricing API

Suggests a **daily rental price** for a car, from its mileage, engine power, model, fuel, body
type, colour and seven equipment options.

The model is not stored here. It is pulled from an MLflow Model Registry at startup through the
`champion` alias, so promoting a new version needs no rebuild of this image. What the registry
hands over is the **whole** scikit-learn pipeline — scaler and one-hot encoder included — so this
service owns no feature engineering and only ever calls `.predict(...)`. That is the point: a
served model that reproduces its own preprocessing is one that quietly stops matching its notebook
the first time an encoder changes.

## Endpoints

| | |
|---|---|
| `POST /predict` | a list of cars in, one price per car out |
| `GET /health` | liveness, plus which model URI was loaded and whether it is in memory |

`/` redirects to **`/docs`**, where the full input contract is documented and you can send a car
and watch the model answer.

### Request

```bash
curl -i -H "Content-Type: application/json" -X POST \
  -d '{"input": [[140411, 100, "Citroën", "diesel", "black", "convertible", true, true, false, false, true, true, true]]}' \
  https://<space-url>/predict
```

Each car is a list of **13 values in a fixed order** — `mileage`, `engine_power`, `model_key`,
`fuel`, `paint_color`, `car_type`, then the seven booleans `private_parking_available`, `has_gps`,
`has_air_conditioning`, `automatic_car`, `has_getaround_connect`, `has_speed_regulator`,
`winter_tires`. The order is documented at `/docs` and read from the model's own logged
**signature**, never from a literal in this service — so it is written once, by the training
script, and cannot drift.

### Response

```json
{"prediction": [110.21]}
```

One price in euros per day, per car sent, in the same order.

**An unknown category is accepted, not rejected.** A brand, fuel, colour or body type the model
never saw is encoded as "infrequent" rather than raising, so the endpoint answers for any car.

## Accuracy, and how to read it

On a held-out fifth of the data the model is wrong by **€10.61 on average** and by **€6.77 for half
the cars**. Charging every car the median price is wrong by €23.49. `/docs` shows these figures
read live from the served model's MLflow run, and `/health` returns its `test_mae`.

## Errors

A car that does not hold 13 values, or a value that cannot be read as its column's type, is
answered with a **422** and a message naming the problem — never with a 500.

## Known limitation

**77.5% of what the model knows comes from `engine_power` and `mileage`** — two columns an owner
cannot change. The seven equipment options and the colour together account for 11.1%. Fitted on
the car's own characteristics alone it reaches CV R² 0.708; adding everything the owner controls
takes it to 0.759.

So this endpoint answers *"what is my car worth"*, not *"what should I change"*. It is the right
answer for an owner setting a price, and the wrong one to present as an optimiser.

## Configuration

Five Space **secrets**, all of them credentials for this project's MLflow stack:
`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_TOKEN`, `MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`. Two Space **variables** name the model rather than granting access to it:
`MLFLOW_REGISTERED_MODEL_NAME` and `MLFLOW_MODEL_ALIAS`. The app reads all of them at startup and
fails immediately, with a traceback, if one is missing — an API that starts without a model would
only answer opaque 500s.
