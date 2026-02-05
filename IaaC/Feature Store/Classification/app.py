"""Holiday Package Prediction with a simple feature store.

This script mirrors the classification workflow in app_classification.py while
demonstrating a lightweight feature store that registers engineered features
and retrieves them for reuse.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - user environment dependent
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
        description="Train classification models for holiday package prediction."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("Travel.csv"),
        help="Path to the dataset CSV (default: Travel.csv).",
    )
    parser.add_argument(
        "--roc-path",
        type=Path,
        default=Path("auc.png"),
        help="Output path for ROC curve plot (default: auc.png).",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Gender"] = df["Gender"].replace("Fe Male", "Female")
    df["MaritalStatus"] = df["MaritalStatus"].replace("Single", "Unmarried")

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["TypeofContact"].fillna(df["TypeofContact"].mode()[0], inplace=True)
    df["DurationOfPitch"].fillna(df["DurationOfPitch"].median(), inplace=True)
    df["NumberOfFollowups"].fillna(df["NumberOfFollowups"].mode()[0], inplace=True)
    df["PreferredPropertyStar"].fillna(
        df["PreferredPropertyStar"].mode()[0], inplace=True
    )
    df["NumberOfTrips"].fillna(df["NumberOfTrips"].median(), inplace=True)
    df["NumberOfChildrenVisiting"].fillna(
        df["NumberOfChildrenVisiting"].mode()[0], inplace=True
    )
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

    if "CustomerID" in df.columns:
        df.drop("CustomerID", inplace=True, axis=1)

    return df


def build_feature_store() -> FeatureStore:
    store = FeatureStore()

    def total_visiting(df: pd.DataFrame) -> pd.Series:
        return df["NumberOfPersonVisiting"] + df["NumberOfChildrenVisiting"]

    store.register("TotalVisiting", total_visiting)
    return store


def engineer_features(df: pd.DataFrame, store: FeatureStore) -> pd.DataFrame:
    df = df.copy()

    total_visiting_feature = store.get("TotalVisiting")
    df["TotalVisiting"] = total_visiting_feature(df)

    df.drop(
        columns=["NumberOfPersonVisiting", "NumberOfChildrenVisiting"],
        inplace=True,
    )
    return df


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    cat_features = features.select_dtypes(include="object").columns
    num_features = features.select_dtypes(exclude="object").columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    oh_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first")),
        ]
    )

    return ColumnTransformer(
        [
            ("OneHotEncoder", oh_transformer, cat_features),
            ("StandardScaler", numeric_transformer, num_features),
        ]
    )


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> None:
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_train_accuracy = accuracy_score(y_train, y_train_pred)
    model_train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    model_train_precision = precision_score(y_train, y_train_pred)
    model_train_recall = recall_score(y_train, y_train_pred)
    model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)

    model_test_accuracy = accuracy_score(y_test, y_test_pred)
    model_test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    model_test_precision = precision_score(y_test, y_test_pred)
    model_test_recall = recall_score(y_test, y_test_pred)
    model_test_rocauc_score = roc_auc_score(y_test, y_test_pred)

    print(name)
    print("Model performance for Training set")
    print(f"- Accuracy: {model_train_accuracy:.4f}")
    print(f"- F1 score: {model_train_f1:.4f}")
    print(f"- Precision: {model_train_precision:.4f}")
    print(f"- Recall: {model_train_recall:.4f}")
    print(f"- Roc Auc Score: {model_train_rocauc_score:.4f}")
    print("----------------------------------")
    print("Model performance for Test set")
    print(f"- Accuracy: {model_test_accuracy:.4f}")
    print(f"- F1 score: {model_test_f1:.4f}")
    print(f"- Precision: {model_test_precision:.4f}")
    print(f"- Recall: {model_test_recall:.4f}")
    print(f"- Roc Auc Score: {model_test_rocauc_score:.4f}")
    print("=" * 35)
    print()


def plot_roc_curve(model, X_train, y_train, X_test, y_test, output_path: Path) -> None:
    model.fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    plt.figure()
    plt.plot(fpr, tpr, label=f"Xgboost ROC (area = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1-Specificity(False Positive Rate)")
    plt.ylabel("Sensitivity(True Positive Rate)")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.show()


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

    X = df.drop(["ProdTaken"], axis=1)
    y = df["ProdTaken"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    print("==== Baseline Models ====")
    models = {
        "Logisitic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boost": GradientBoostingClassifier(),
        "Adaboost": AdaBoostClassifier(),
        "Xgboost": XGBClassifier(),
    }

    for name, model in models.items():
        evaluate_model(name, model, X_train, y_train, X_test, y_test)

    print("==== Hyperparameter Tuning ====")
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
        ("RF", RandomForestClassifier(), rf_params),
        ("Xgboost", XGBClassifier(), xgboost_params),
    ]

    model_param = {}
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

    for model_name in model_param:
        print(f"---------------- Best Params for {model_name} -------------------")
        print(model_param[model_name])

    print("==== Tuned Models ====")
    tuned_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=1000, min_samples_split=2, max_features=7, max_depth=None
        ),
        "Xgboost": XGBClassifier(
            n_estimators=200, max_depth=12, learning_rate=0.1, colsample_bytree=1
        ),
    }

    for name, model in tuned_models.items():
        evaluate_model(name, model, X_train, y_train, X_test, y_test)

    print("==== ROC Curve ====")
    plot_roc_curve(tuned_models["Xgboost"], X_train, y_train, X_test, y_test, args.roc_path)


if __name__ == "__main__":
    main()