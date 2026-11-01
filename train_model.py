from __future__ import annotations

from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "CAR DETAILS FROM CAR DEKHO.csv"
MODEL_PATH = PROJECT_DIR / "car_price_model.pkl"
RANDOM_STATE = 42

NUMERIC_FEATURES = ["model_year", "km_driven"]
CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MULTIWORD_BRANDS = ("Land Rover",)


def normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def split_brand_model(name: str) -> tuple[str, str]:
    clean_name = str(name).strip()
    for brand in MULTIWORD_BRANDS:
        if clean_name.casefold().startswith((brand + " ").casefold()):
            return brand, clean_name[len(brand):].strip()
    parts = clean_name.split(maxsplit=1)
    return (parts[0], parts[1].strip()) if len(parts) == 2 else (parts[0], parts[0])


def parse_number(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(",", "", regex=False)
    found = text.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(found, errors="coerce")


def prepare_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Keep the Car Dekho CSV beside train_model.py."
        )

    df = pd.read_csv(csv_path)
    source_rows = len(df)
    df.columns = [normalize_column(column) for column in df.columns]

    required = {
        "name",
        "year",
        "selling_price",
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    df["model_year"] = pd.to_numeric(df["year"], errors="coerce")
    df["selling_price"] = parse_number(df["selling_price"])
    df["km_driven"] = parse_number(df["km_driven"])
    df["name"] = df["name"].astype("string").str.strip()

    parsed_names = df["name"].dropna().map(split_brand_model)
    name_parts = pd.DataFrame(
        parsed_names.tolist(),
        index=parsed_names.index,
        columns=["brand", "model"],
    )
    df["brand"] = pd.NA
    df["model"] = pd.NA
    df.loc[name_parts.index, "brand"] = name_parts["brand"]
    df.loc[name_parts.index, "model"] = name_parts["model"]

    categorical_columns = [
        "brand",
        "model",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
    ]
    for column in categorical_columns:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].replace("", pd.NA)

    df = df.dropna(subset=["model_year", "selling_price", "brand", "model"]).copy()
    df = df[
        df["selling_price"].gt(0)
        & df["model_year"].between(1980, pd.Timestamp.now().year)
        & (df["km_driven"].isna() | df["km_driven"].ge(0))
    ].copy()

    if df.empty:
        raise ValueError("No valid car records remain after cleaning.")

    X = df[FEATURE_COLUMNS].copy()
    y = df["selling_price"].astype(float).copy()

    summary = {
        "source_rows": source_rows,
        "duplicates_removed": duplicates_removed,
        "rows_used": len(df),
        "rows_removed_during_cleaning": source_rows - duplicates_removed - len(df),
        "model_year_min": int(X["model_year"].min()),
        "model_year_max": int(X["model_year"].max()),
        "target_min": float(y.min()),
        "target_median": float(y.median()),
        "target_max": float(y.max()),
    }
    return X, y, summary


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=2,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def score_regression(actual: pd.Series, predicted: np.ndarray) -> dict:
    return {
        "MAE (INR)": float(mean_absolute_error(actual, predicted)),
        "RMSE (INR)": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def format_inr(value: float) -> str:
    number = str(int(round(value)))
    if len(number) <= 3:
        return f"₹{number}"
    last_three = number[-3:]
    prefix = number[:-3]
    groups = []
    while prefix:
        groups.insert(0, prefix[-2:])
        prefix = prefix[:-2]
    return f"₹{','.join(groups + [last_three])}"


def main() -> None:
    X, y, summary = prepare_dataset(CSV_PATH)
    row_ids = np.arange(len(X))

    # The source has no listing date, so use a random holdout after deduplication.
    development_ids, test_ids = train_test_split(
        row_ids, test_size=0.20, random_state=RANDOM_STATE
    )
    train_ids, validation_ids = train_test_split(
        development_ids, test_size=0.25, random_state=RANDOM_STATE
    )

    estimators = {
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=250,
            max_features=0.9,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_features=0.9,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=2,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
    }

    validation_results = []
    for name, estimator in estimators.items():
        for target_transform in ("raw", "log"):
            candidate_name = f"{name} ({target_transform} target)"
            training_target = (
                np.log1p(y.iloc[train_ids])
                if target_transform == "log"
                else y.iloc[train_ids]
            )
            pipeline = Pipeline(
                [
                    ("preprocessor", make_preprocessor()),
                    ("model", clone(estimator)),
                ]
            )
            pipeline.fit(X.iloc[train_ids], training_target)
            model_output = pipeline.predict(X.iloc[validation_ids])
            predictions = (
                np.expm1(model_output)
                if target_transform == "log"
                else model_output
            )
            metrics = score_regression(y.iloc[validation_ids], predictions)
            validation_results.append(
                {
                    "Model": candidate_name,
                    "Validation MAE (INR)": metrics["MAE (INR)"],
                    "Validation RMSE (INR)": metrics["RMSE (INR)"],
                }
            )
            print(
                f"{candidate_name}: validation MAE {format_inr(metrics['MAE (INR)'])}; "
                f"RMSE {format_inr(metrics['RMSE (INR)'])}"
            )

    validation_results.sort(key=lambda row: row["Validation MAE (INR)"])
    best_candidate = validation_results[0]["Model"]
    best_name, transform_label = best_candidate.rsplit(" (", 1)
    best_transform = transform_label.removesuffix(" target)")

    fit_ids = np.concatenate([train_ids, validation_ids])
    final_target = (
        np.log1p(y.iloc[fit_ids])
        if best_transform == "log"
        else y.iloc[fit_ids]
    )
    final_pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("model", clone(estimators[best_name])),
        ]
    )
    final_pipeline.fit(X.iloc[fit_ids], final_target)

    model_output = final_pipeline.predict(X.iloc[test_ids])
    test_predictions = (
        np.expm1(model_output) if best_transform == "log" else model_output
    )
    test_metrics = score_regression(y.iloc[test_ids], test_predictions)
    test_metrics["R²"] = float(r2_score(y.iloc[test_ids], test_predictions))

    baseline_prediction = float(y.iloc[fit_ids].median())
    baseline_predictions = np.full(len(test_ids), baseline_prediction, dtype=float)
    baseline_metrics = score_regression(y.iloc[test_ids], baseline_predictions)

    residuals = y.iloc[test_ids].to_numpy() - test_predictions
    residual_q10, residual_q90 = np.quantile(residuals, [0.10, 0.90])
    absolute_error_p90 = float(np.quantile(np.abs(residuals), 0.90))

    category_options = {
        feature: sorted(X[feature].dropna().astype(str).unique().tolist())
        for feature in CATEGORICAL_FEATURES
    }
    category_defaults = {}
    for feature in CATEGORICAL_FEATURES:
        modes = X[feature].dropna().astype(str).mode()
        category_defaults[feature] = (
            str(modes.iloc[0]) if not modes.empty else category_options[feature][0]
        )

    numeric_defaults = {
        feature: float(X[feature].median()) for feature in NUMERIC_FEATURES
    }
    models_by_brand = {
        str(brand): sorted(group["model"].dropna().astype(str).unique().tolist())
        for brand, group in X.groupby("brand", dropna=True)
    }
    default_model_by_brand = {
        str(brand): str(group["model"].dropna().astype(str).mode().iloc[0])
        for brand, group in X.groupby("brand", dropna=True)
        if not group["model"].dropna().empty
    }

    bundle = {
        "model": final_pipeline,
        "model_name": best_candidate,
        "target_transform": best_transform,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_options": category_options,
        "category_defaults": category_defaults,
        "numeric_defaults": numeric_defaults,
        "default_brand": category_defaults["brand"],
        "models_by_brand": models_by_brand,
        "default_model_by_brand": default_model_by_brand,
        "price_unit": "INR",
        "market_scope": "India-wide",
        "price_type": "Car Dekho dataset price",
        "dataset_note": (
            "This estimate is learned from the supplied Car Dekho dataset. "
            "The data has model years from "
            f"{summary['model_year_min']} to {summary['model_year_max']} and no listing date, "
            "city, or vehicle-condition field. Treat the result as a dataset-based estimate, "
            "not a guaranteed current market quote."
        ),
        "source_rows": int(summary["source_rows"]),
        "duplicates_removed": int(summary["duplicates_removed"]),
        "rows_removed_during_cleaning": int(summary["rows_removed_during_cleaning"]),
        "rows_used": int(summary["rows_used"]),
        "train_rows": int(len(fit_ids)),
        "test_rows": int(len(test_ids)),
        "model_year_min": summary["model_year_min"],
        "model_year_max": summary["model_year_max"],
        "km_input_max": max(
            100_000, int(np.ceil(X["km_driven"].max() / 10_000) * 10_000)
        ),
        "target_min": summary["target_min"],
        "target_median": summary["target_median"],
        "target_max": summary["target_max"],
        "mae": test_metrics["MAE (INR)"],
        "rmse": test_metrics["RMSE (INR)"],
        "r2": test_metrics["R²"],
        "baseline_mae": baseline_metrics["MAE (INR)"],
        "baseline_rmse": baseline_metrics["RMSE (INR)"],
        "residual_q10": float(residual_q10),
        "residual_q90": float(residual_q90),
        "absolute_error_p90": absolute_error_p90,
        "model_comparison": validation_results,
        "evaluation_method": (
            "Random 60/20/20 train/validation/test split after exact duplicate removal"
        ),
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
    }

    joblib.dump(bundle, MODEL_PATH, compress=3)

    print("\nValidation model comparison:")
    print(pd.DataFrame(validation_results).to_string(index=False))
    print(f"\nSelected model: {best_candidate}")
    print(f"Rows used: {summary['rows_used']:,}")
    print(f"Exact duplicates removed: {summary['duplicates_removed']:,}")
    print(f"Test MAE: {format_inr(test_metrics['MAE (INR)'])}")
    print(f"Median-baseline MAE: {format_inr(baseline_metrics['MAE (INR)'])}")
    print(f"Test RMSE: {format_inr(test_metrics['RMSE (INR)'])}")
    print(f"Test R²: {test_metrics['R²']:.3f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

