from __future__ import annotations

from pathlib import Path
import math

import joblib
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "car_price_model.pkl"

st.set_page_config(
    page_title="Car price predictor",
    page_icon=":material/directions_car:",
    layout="centered",
)

@st.cache_resource
def load_model_bundle(model_path: str, modified_ns: int) -> dict:
    del modified_ns
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error("The trained model is missing.")
    st.info("Run python train_model.py from the project folder to create it.")
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    required = {
        "model",
        "feature_columns",
        "numeric_features",
        "categorical_features",
        "category_options",
        "numeric_defaults",
        "mae",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"The model bundle is missing: {', '.join(missing)}")
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.info("Run python train_model.py to create a compatible model.")
    st.stop()

model = bundle["model"]
feature_columns = bundle["feature_columns"]
category_options = bundle["category_options"]
category_defaults = bundle.get("category_defaults", {})
numeric_defaults = bundle["numeric_defaults"]
brand_options = category_options.get("brand", [])
models_by_brand = bundle.get("models_by_brand", {})
default_model_by_brand = bundle.get("default_model_by_brand", {})


def format_inr(value: float) -> str:
    rounded = int(round(float(value)))
    sign = "−" if rounded < 0 else ""
    digits = str(abs(rounded))
    if len(digits) <= 3:
        return f"{sign}₹{digits}"
    last_three = digits[-3:]
    prefix = digits[:-3]
    groups = []
    while prefix:
        groups.insert(0, prefix[-2:])
        prefix = prefix[:-2]
    return f"{sign}₹{','.join(groups + [last_three])}"


st.title("Used car price predictor")
st.caption("Enter a few car details to get an estimated price.")

if not brand_options:
    st.error("No car brands were saved in the trained model.")
    st.stop()

default_brand = bundle.get("default_brand")
brand_index = brand_options.index(default_brand) if default_brand in brand_options else 0

with st.container(border=True):
    brand = st.selectbox(
        "Car brand",
        brand_options,
        index=brand_index,
        key="car_brand",
    )

    model_options = models_by_brand.get(brand, [])
    if not model_options:
        st.error("No models are available for this brand.")
        st.stop()

    default_model = default_model_by_brand.get(brand)
    model_index = model_options.index(default_model) if default_model in model_options else 0
    car_model = st.selectbox(
        "Car model",
        model_options,
        index=model_index,
        key="car_model",
    )

    with st.form("car_price_form"):
        input_columns = st.columns(2)
        with input_columns[0]:
            year_min = int(bundle.get("model_year_min", 1980))
            year_max = int(bundle.get("model_year_max", 2020))
            year_default = max(
                year_min,
                min(year_max, int(numeric_defaults.get("model_year", year_max))),
            )
            model_year = st.number_input(
                "Year of manufacture",
                min_value=year_min,
                max_value=year_max,
                value=year_default,
                step=1,
            )

        with input_columns[1]:
            km_max = int(bundle.get("km_input_max", 1_000_000))
            km_default = max(
                0,
                min(km_max, int(numeric_defaults.get("km_driven", 0))),
            )
            km_driven = st.number_input(
                "Kilometres driven",
                min_value=0,
                max_value=km_max,
                value=km_default,
                step=1_000,
            )

        category_inputs = {}
        feature_labels = {
            "fuel": "Fuel type",
            "seller_type": "Seller type",
            "transmission": "Transmission",
            "owner": "Previous owners",
        }
        category_features = [
            feature
            for feature in bundle["categorical_features"]
            if feature not in {"brand", "model"}
        ]
        category_columns = st.columns(2)
        for index, feature in enumerate(category_features):
            options = category_options.get(feature, [])
            if not options:
                continue
            default_value = category_defaults.get(feature)
            default_index = options.index(default_value) if default_value in options else 0
            with category_columns[index % 2]:
                category_inputs[feature] = st.selectbox(
                    feature_labels.get(feature, feature.replace("_", " ").title()),
                    options,
                    index=default_index,
                )

        submitted = st.form_submit_button(
            "Predict car price",
            type="primary",
            width="stretch",
        )

if submitted:
    vehicle = {
        "brand": brand,
        "model": car_model,
        "model_year": int(model_year),
        "km_driven": int(km_driven),
        **category_inputs,
    }
    row = {feature: vehicle[feature] for feature in feature_columns}
    input_frame = pd.DataFrame([row], columns=feature_columns)

    try:
        model_output = float(model.predict(input_frame)[0])
        prediction = (
            math.expm1(model_output)
            if bundle.get("target_transform") == "log"
            else model_output
        )
        if not math.isfinite(prediction) or prediction <= 0:
            raise ValueError("The model returned an invalid price.")

        lower = max(0.0, prediction + float(bundle.get("residual_q10", 0.0)))
        upper = max(lower, prediction + float(bundle.get("residual_q90", 0.0)))

        st.subheader("Estimated price")
        st.success(format_inr(prediction))
        st.caption(
            f"Indicative range: {format_inr(lower)} – {format_inr(upper)}"
        )
        st.caption(
            f"Typical test error (MAE): {format_inr(bundle['mae'])}. "
            "The range is a guide, not a guaranteed price."
        )
    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")

