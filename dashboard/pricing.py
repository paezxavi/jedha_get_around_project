"""Pricing page: an owner describes a car, the /predict API prices it.

The page holds no model. It sends the car to the pricing API exactly as curl would, and shows
what comes back -- that call is the point of the page. Two things it never writes itself:
  - the column order, read from the API's /health, which reads it from the model's signature;
  - the model's error, read from the same /health, which reads it from the champion's run.

The dropdowns are filled from the pricing file shipped in this image, so every choice offered
is a value the model was trained on.
"""

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("PRICING_API_URL", "https://lambla-getaround-pricing-api.hf.space")
DATA = Path(__file__).parent / "data" / "get_around_pricing_project.csv"

# A sleeping Space can take a minute or two to wake up on its first request.
TIMEOUT = 180

# The seven equipment columns, with the label an owner reads on the form.
EQUIPMENT = {
    "private_parking_available": "Private parking",
    "has_gps": "GPS",
    "has_air_conditioning": "Air conditioning",
    "automatic_car": "Automatic gearbox",
    "has_getaround_connect": "Getaround Connect",
    "has_speed_regulator": "Cruise control",
    "winter_tires": "Winter tyres",
}


@st.cache_data
def load_cars():
    return pd.read_csv(DATA)


@st.cache_data(ttl=3600, show_spinner=False)
def api_contract():
    """The column order and the model's MAE, as the API reports them on /health."""
    health = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
    health.raise_for_status()
    return health.json()


cars = load_cars()

st.title("What should my car rent for?")
st.markdown(
    "Describe the car and the pricing model suggests a **daily rental price**. The answer "
    f"comes from the Getaround pricing API, called live at `{API_URL}/predict`."
)

with st.form("car"):
    left, middle, right = st.columns(3)
    car = {}
    # Defaults are the typical car of the file: the median for numbers, the most common value
    # for everything else.
    car["mileage"] = left.number_input(
        "Mileage (km)", min_value=0, value=int(cars["mileage"].median()), step=1000)
    car["engine_power"] = left.number_input(
        "Engine power (hp)", min_value=1, value=int(cars["engine_power"].median()), step=5)
    for column, label, col in [("model_key", "Brand", middle), ("fuel", "Fuel", middle),
                               ("paint_color", "Colour", right), ("car_type", "Body type", right)]:
        options = sorted(cars[column].unique())
        car[column] = col.selectbox(label, options,
                                    index=options.index(cars[column].mode()[0]))
    st.markdown("**Equipment**")
    boxes = st.columns(4)
    for i, (column, label) in enumerate(EQUIPMENT.items()):
        car[column] = boxes[i % 4].checkbox(label, value=bool(cars[column].mode()[0]))
    submitted = st.form_submit_button("Suggest a price", type="primary")

if submitted:
    try:
        with st.spinner("Asking the pricing API — a sleeping Space can take a minute to wake up"):
            contract = api_contract()
            # The order of the 13 values is the API's contract; it is read, never assumed.
            payload = {"input": [[car[column] for column in contract["features"]]]}
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT)
    except requests.RequestException as error:
        st.error(f"The pricing API did not answer ({type(error).__name__}). It may still be "
                 "waking up: try again in a minute.")
        st.stop()

    if response.status_code != 200:
        st.error(f"The pricing API refused the request (HTTP {response.status_code}): "
                 f"{response.json().get('detail', response.text)}")
        st.stop()

    price = response.json()["prediction"][0]
    st.metric("Suggested daily price", f"€{price:.2f}")
    if contract.get("test_mae") is not None:
        st.caption(
            f"The market rate for a car like this one, as owners price it. On cars it never "
            f"saw, the model is wrong by €{contract['test_mae']:.2f} on average. Engine power "
            f"and mileage weigh most: the equipment options move the price little."
        )

    with st.expander("Request sent to the API, and its answer"):
        st.code(f"POST {API_URL}/predict\n\n{json.dumps(payload, ensure_ascii=False)}",
                language="text")
        st.code(json.dumps(response.json()), language="json")
