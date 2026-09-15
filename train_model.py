"""Train and evaluate an SVM breast-cancer classification demo."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
FEATURES = [
    "mean radius", "mean texture", "mean smoothness", "mean compactness", "mean concavity"
]


def score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, model.predict(x_test)), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
    }


def main():
    data = load_breast_cancer(as_frame=True)
    x = data.frame[FEATURES]
    # scikit-learn encodes malignant as 0 and benign as 1.
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=5000))
        ]),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    comparison = {name: score(model, x_train, x_test, y_train, y_test)
                  for name, model in candidates.items()}

    search = GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True, random_state=42))]),
        {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.1], "model__kernel": ["rbf", "linear"]},
        scoring="roc_auc", cv=5, n_jobs=-1,
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_
    prediction = best.predict(x_test)
    probability = best.predict_proba(x_test)[:, 1]
    metrics = {
        "features": FEATURES,
        "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn copy)",
        "comparison": comparison,
        "best_svm_parameters": search.best_params_,
        "svm_test_accuracy": round(accuracy_score(y_test, prediction), 4),
        "svm_test_roc_auc": round(roc_auc_score(y_test, probability), 4),
        "confusion_matrix_rows_actual_0_malignant_1_benign": confusion_matrix(y_test, prediction).tolist(),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": best, "features": FEATURES}, MODEL_DIR / "breast_cancer_svm.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("\nSaved model/breast_cancer_svm.joblib and model/metrics.json")


if __name__ == "__main__":
    main()
