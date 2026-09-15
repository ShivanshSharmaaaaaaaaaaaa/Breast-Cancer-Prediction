"""Minimal Flask interface for the trained demo pipeline."""
from __future__ import annotations

from pathlib import Path
import joblib
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "breast_cancer_svm.joblib"
app = Flask(__name__)


def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run: python train_model.py")
    return joblib.load(MODEL_PATH)


def predict(values):
    artifact = load_artifact()
    if len(values) != len(artifact["features"]):
        raise ValueError(f"Expected {len(artifact['features'])} numeric values.")
    model = artifact["model"]
    benign_probability = float(model.predict_proba([values])[0][1])
    label = "Benign" if benign_probability >= 0.5 else "Malignant"
    confidence = benign_probability if label == "Benign" else 1 - benign_probability
    return {"prediction": label, "confidence": round(confidence * 100, 1),
            "benign_probability": round(benign_probability * 100, 1)}


@app.route("/", methods=["GET", "POST"])
def index():
    result = error = None
    defaults = [14.1, 20.3, 0.10, 0.11, 0.09]
    if request.method == "POST":
        try:
            defaults = [float(request.form[name]) for name in load_artifact()["features"]]
            result = predict(defaults)
        except (ValueError, FileNotFoundError) as exc:
            error = str(exc)
    return render_template("index.html", values=defaults, result=result, error=error,
                           features=load_artifact()["features"] if MODEL_PATH.exists() else [])


@app.post("/api/predict")
def api_predict():
    try:
        return jsonify(predict(request.get_json(force=True)["features"]))
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True)
