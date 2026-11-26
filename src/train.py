"""Breast cancer classification with scikit-learn.

Covers: data exploration, preprocessing, Logistic Regression and Random Forest,
cross-validation, hyperparameter tuning, and evaluation (accuracy, precision,
recall, F1, confusion matrix, ROC curve).

Run:  python src/train.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)


def load_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # In this dataset target 0 = malignant, 1 = benign
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    return df, data.feature_names


def explore(df):
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Class balance:\n", df["diagnosis"].value_counts())

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="diagnosis", ax=ax)
    ax.set_title("Class distribution")
    fig.tight_layout()
    fig.savefig(OUT / "class_distribution.png", dpi=150)
    plt.close(fig)

    top = df.drop(columns=["diagnosis"]).corr()["target"].abs().sort_values(ascending=False)[1:11].index
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[list(top)].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation of the 10 features most related to the target")
    fig.tight_layout()
    fig.savefig(OUT / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def build_models():
    logreg = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]
    )
    forest = Pipeline(
        [("scale", StandardScaler()), ("clf", RandomForestClassifier(random_state=SEED))]
    )
    grids = {
        "Logistic Regression": (logreg, {"clf__C": [0.01, 0.1, 1, 10, 100]}),
        "Random Forest": (
            forest,
            {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10]},
        ),
    }
    return grids


def main():
    df, feature_names = load_data()
    explore(df)

    X = df[list(feature_names)]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    results = {}
    fitted = {}
    for name, (pipe, grid) in build_models().items():
        baseline_cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")
        search = GridSearchCV(pipe, grid, cv=5, scoring="f1")
        search.fit(X_train, y_train)
        best = search.best_estimator_
        pred = best.predict(X_test)
        results[name] = {
            "baseline_cv_f1_mean": round(float(baseline_cv.mean()), 4),
            "best_params": search.best_params_,
            "tuned_cv_f1": round(float(search.best_score_), 4),
            "test_accuracy": round(accuracy_score(y_test, pred), 4),
            "test_precision": round(precision_score(y_test, pred), 4),
            "test_recall": round(recall_score(y_test, pred), 4),
            "test_f1": round(f1_score(y_test, pred), 4),
        }
        fitted[name] = best
        print(f"\n{name}: {json.dumps(results[name], indent=2)}")

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["malignant", "benign"], ax=ax, colorbar=False
        )
        ax.set_title(f"Confusion matrix: {name}")
        fig.tight_layout()
        fig.savefig(OUT / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curves on the test set")
    fig.tight_layout()
    fig.savefig(OUT / "roc_curves.png", dpi=150)
    plt.close(fig)

    forest = fitted["Random Forest"].named_steps["clf"]
    imp = pd.Series(forest.feature_importances_, index=feature_names).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax)
    ax.set_title("Top 10 feature importances (Random Forest)")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)

    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved plots and metrics.json to", OUT)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
