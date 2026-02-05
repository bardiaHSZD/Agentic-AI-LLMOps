"""Used Car Price Prediction with a simple feature store.

This script mirrors the regression workflow in app_regression.py while
demonstrating a lightweight feature store that registers engineered features
and retrieves them for reuse.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "xgboost is required. Install it with 'pip install xgboost'."
    ) from exc


class FeatureStore:
    """Lightweight in-memory feature store for DataFrame feature engineering."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}

    def register(self, name: str, transformer: Callable[[pd.DataFrame], pd.Series]) -> None:
        if name in self._registry:
            raise ValueError(f"Feature '{name}' is already registered.")
        self._registry[name] = transformer

    def get(self, name: str) -> Callable[[pd.DataFrame], pd.Series]:
        if name not in self._registry:
            raise KeyError(f"Feature '{name}' not found in store.")
        return self._registry[name]

    def list_features(self) -> list[str]:
        return sorted(self._registry.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train regression models for used car price prediction."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("./data/cardekho_imputated.csv"),
        help="Path to the dataset CSV (default: ./data/cardekho_imputated.csv).",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    return pd.read_csv(path, index_col=[0])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop("car_name", axis=1, inplace=True)
    df.drop("brand", axis=1, inplace=True)
    return df


def build_feature_store() -> FeatureStore:
    store = FeatureStore()

    def price_per_km(df: pd.DataFrame) -> pd.Series:
        return df["selling_price"] / df["km_driven"].replace(0, np.nan)

    store.register("PricePerKm", price_per_km)
    return store


def engineer_features(df: pd.DataFrame, store: FeatureStore) -> pd.DataFrame:
    df = df.copy()
    price_per_km_feature = store.get("PricePerKm")
    df["PricePerKm"] = price_per_km_feature(df)
    df["PricePerKm"].fillna(df["PricePerKm"].median(), inplace=True)
    return df


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    num_features = features.select_dtypes(exclude="object").columns
    onehot_columns = ["seller_type", "fuel_type", "transmission_type"]

    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder(drop="first")

    return ColumnTransformer(
        [
            ("OneHotEncoder", oh_transformer, onehot_columns),
            ("StandardScaler", numeric_transformer, num_features),
        ],
        remainder="passthrough",
    )


def evaluate_model(true, predicted) -> tuple[float, float, float]:
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


def report_performance(
    name: str,
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
) -> None:
    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(
        y_train, y_train_pred
    )
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(
        y_test, y_test_pred
    )

    print(name)
    print("Model performance for Training set")
    print(f"- Root Mean Squared Error: {model_train_rmse:.4f}")
    print(f"- Mean Absolute Error: {model_train_mae:.4f}")
    print(f"- R2 Score: {model_train_r2:.4f}")
    print("----------------------------------")
    print("Model performance for Test set")
    print(f"- Root Mean Squared Error: {model_test_rmse:.4f}")
    print(f"- Mean Absolute Error: {model_test_mae:.4f}")
    print(f"- R2 Score: {model_test_r2:.4f}")
    print("=" * 35)
    print()


def run_baseline_models(X_train, X_test, y_train, y_test) -> None:
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Adaboost Regressor": AdaBoostRegressor(),
        "Graident BoostRegressor": GradientBoostingRegressor(),
        "Xgboost Regressor": XGBRegressor(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        report_performance(name, y_train, y_train_pred, y_test, y_test_pred)


def run_hyperparameter_tuning(X_train, y_train) -> dict[str, dict]:
    rf_params = {
        "max_depth": [5, 8, 15, None, 10],
        "max_features": [5, 7, "auto", 8],
        "min_samples_split": [2, 8, 15, 20],
        "n_estimators": [100, 200, 500, 1000],
    }

    xgboost_params = {
        "learning_rate": [0.1, 0.01],
        "max_depth": [5, 8, 12, 20, 30],
        "n_estimators": [100, 200, 300],
        "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4],
    }

    randomcv_models = [
        ("RF", RandomForestRegressor(), rf_params),
        ("XGboost", XGBRegressor(), xgboost_params),
    ]

    model_param: dict[str, dict] = {}
    for name, model, params in randomcv_models:
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=100,
            cv=3,
            verbose=2,
            n_jobs=-1,
        )
        random_search.fit(X_train, y_train)
        model_param[name] = random_search.best_params_

    for model_name, params in model_param.items():
        print(f"---------------- Best Params for {model_name} -------------------")
        print(params)

    return model_param


def run_tuned_models(X_train, X_test, y_train, y_test) -> None:
    models = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            min_samples_split=2,
            max_features=5,
            max_depth=None,
            n_jobs=-1,
        ),
        "Xgboost Regressor": XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            colsample_bytree=0.5,
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        report_performance(name, y_train, y_train_pred, y_test, y_test_pred)


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()

    df = load_data(args.data_path)
    df = clean_data(df)

    feature_store = build_feature_store()
    print("==== Feature Store Registered Features ====")
    print(feature_store.list_features())
    print()

    df = engineer_features(df, feature_store)

    X = df.drop(["selling_price"], axis=1)
    y = df["selling_price"]

    label_encoder = LabelEncoder()
    X = X.copy()
    X["model"] = label_encoder.fit_transform(X["model"])

    preprocessor = build_preprocessor(X)
    X = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("==== Baseline Models ====")
    run_baseline_models(X_train, X_test, y_train, y_test)

    print("==== Hyperparameter Tuning ====")
    run_hyperparameter_tuning(X_train, y_train)

    print("==== Tuned Models ====")
    run_tuned_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()